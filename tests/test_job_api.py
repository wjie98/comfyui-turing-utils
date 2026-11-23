"""Native queue integration, multipart media and task-owned cleanup; no weights."""

import asyncio
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import wave

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

import execution
import nodes
from comfy import model_management
from comfy_execution.progress import get_progress_state
from comfy_extras.nodes_audio import LoadAudio, SaveAudio
from comfy_extras.nodes_video import LoadVideo, SaveVideo
from server import PromptServer
from comfyui_turing_utils.job_api import routes
from comfyui_turing_utils.job_api.client import ApiError, Client, ServerRestarted
from comfyui_turing_utils.job_api.service import Jobs
from comfyui_turing_utils.job_api import project
from comfyui_turing_utils.nodes.minimax_sampler import MiniMaxH3KSampler
from comfyui_turing_utils.runtime.workflow_prompt import install_workflow_prompt_compiler


class Text:
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}

    def execute(self, text):
        return (text + "!",)


class Result(Text):
    OUTPUT_NODE = True

    def execute(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


class BrokenText(Text):
    def execute(self, text):
        raise ValueError("test execution failed")


class FileText(Text):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING",)}}

    def execute(self, path):
        return (Path(path).read_text(),)


class ForeignResult(Result):
    def execute(self, text):
        return {
            "ui": {
                "images": [{"filename": "keep.txt", "subfolder": "", "type": "output"}]
            },
            "result": (text,),
        }


class NestedFile(FileText):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"options": ("FILE_OPTIONS",)}}

    def execute(self, options):
        return (Path(options["file"]).read_text() + options["other"],)


class WaitingText(Text):
    async def execute(self, text):
        while True:
            model_management.throw_exception_if_processing_interrupted()
            await asyncio.sleep(0.01)


class ProgressText(Text):
    @classmethod
    def INPUT_TYPES(cls):
        return {**super().INPUT_TYPES(), "hidden": {"unique_id": "UNIQUE_ID"}}

    async def execute(self, text, unique_id):
        for value in range(6):
            get_progress_state().update_progress(unique_id, value, 5)
            await self.steps.put(value)
            await self.release.get()
        return (text + "!",)


class Server:
    client_id = None
    last_node_id = None
    number = 0

    def __init__(self, root):
        self.app = web.Application()
        self.routes = web.RouteTableDef()
        self.prompt_queue = execution.PromptQueue(self)
        self.on_prompt_handlers = []
        self.node_replace_manager = SimpleNamespace(
            apply_replacements=lambda prompt: None
        )
        self.user_manager = SimpleNamespace(
            get_request_user_filepath=lambda request, *args, **kwargs: root
            / request.headers.get("Comfy-User", "default")
        )

    def send_sync(self, *args):
        pass

    def queue_updated(self):
        pass

    add_on_prompt_handler = PromptServer.add_on_prompt_handler
    trigger_on_prompt = PromptServer.trigger_on_prompt


def text_graph(class_type="ApiText", text="source"):
    return {
        "compute": {"class_type": class_type, "inputs": {"text": text}},
        "out": {"class_type": "ApiResult", "inputs": {"text": ["compute", 0]}},
    }


class JobApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        patch = mock.patch(
            "folder_paths.get_output_directory", return_value=str(self.output)
        )
        patch.start()
        self.addCleanup(patch.stop)
        extra = {
            "ApiText": Text,
            "ApiResult": Result,
            "ApiBroken": BrokenText,
            "ApiWaiting": WaitingText,
            "ApiProgress": ProgressText,
            "ApiFile": FileText,
            "ApiForeign": ForeignResult,
            "ApiNestedFile": NestedFile,
        }
        for cls in (LoadAudio, SaveAudio, LoadVideo, SaveVideo):
            schema = cls.GET_SCHEMA()
            extra[schema.node_id] = cls
        patch = mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, extra)
        patch.start()
        self.addCleanup(patch.stop)
        self.server = Server(self.root)
        install_workflow_prompt_compiler(self.server)
        ProgressText.steps, ProgressText.release = asyncio.Queue(), asyncio.Queue()
        routes.install_routes(self.server)
        self.jobs = self.server._turing_job_api
        self.server.app.add_routes(self.server.routes)
        self.client = TestClient(TestServer(self.server.app))
        await self.client.start_server()
        self.executor = execution.PromptExecutor(
            self.server, cache_args={"ram": 0, "ram_inactive": 0, "lru": 100}
        )
        self.worker = asyncio.create_task(self.work())

    async def asyncTearDown(self):
        await self.client.close()
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)

    async def work(self):
        while True:
            queued = self.server.prompt_queue.get(0)
            if queued is None:
                await asyncio.sleep(0.01)
                continue
            item, identity = queued
            await self.executor.execute_async(item[2], item[1], item[3], item[4])
            self.server.prompt_queue.task_done(
                identity,
                self.executor.history_result,
                status=execution.PromptQueue.ExecutionStatus(
                    "success" if self.executor.success else "error",
                    self.executor.success,
                    self.executor.status_messages,
                ),
            )

    async def submit(self, data=None, uploads=None, graph=None):
        request = {
            "api_version": 2,
            "workflow": "graph",
            "files": {},
            "outputs": {"result": {"node_id": "out"}},
            **(data or {}),
        }
        body = FormData()
        body.add_field("request", json.dumps(request), content_type="application/json")
        body.add_field(
            "graph",
            json.dumps(text_graph() if graph is None else graph),
            filename="workflow.json",
        )
        for name, (filename, payload) in (uploads or {}).items():
            body.add_field(name, payload, filename=filename)
        response = await self.client.post("/turing/jobs", data=body)
        return response.status, await response.json()

    async def finished(self, identity):
        async with asyncio.timeout(10):
            while True:
                response = await self.client.get("/turing/jobs/" + identity)
                state = await response.json()
                if state["status"] in {"succeeded", "failed"}:
                    await self.jobs.jobs[identity].runner
                    return state
                await asyncio.sleep(0.01)

    async def test_native_execution_download_delete_and_owner_isolation(self):
        status, submitted = await self.submit(graph=text_graph(text="hello"))
        self.assertEqual(status, 202, submitted)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "succeeded", state)
        job = self.jobs.jobs[state["id"]]
        self.assertFalse((job.root / "project").exists())
        self.assertEqual(list(job.root.iterdir()), [job.root / "outputs"])
        response = await self.client.get(state["outputs"][0]["url"])
        self.assertEqual(await response.text(), "hello!")
        self.assertEqual(
            (
                await self.client.get(
                    "/turing/jobs/" + job.id, headers={"Comfy-User": "other"}
                )
            ).status,
            404,
        )
        self.assertEqual(
            (
                await self.client.delete(
                    "/turing/jobs/" + job.id, headers={"Comfy-User": "other"}
                )
            ).status,
            204,
        )
        self.assertTrue(job.root.exists())
        self.assertEqual(
            (await self.client.delete("/turing/jobs/" + job.id)).status, 204
        )
        self.assertFalse(job.root.exists())
        self.assertEqual(self.server.prompt_queue.history, {})
        self.assertFalse((self.root / "default").exists())

    async def test_native_node_ids_can_be_prompt_or_nodes(self):
        graph = {
            "prompt": {"class_type": "ApiText", "inputs": {"text": "literal"}},
            "nodes": {"class_type": "ApiResult", "inputs": {"text": ["prompt", 0]}},
        }
        status, submitted = await self.submit(
            {"outputs": {"result": {"node_id": "nodes"}}}, graph=graph
        )
        self.assertEqual(status, 202, submitted)
        self.assertEqual((await self.finished(submitted["id"]))["status"], "succeeded")

    async def test_image_video_audio_and_text_in_one_request(self):
        image = BytesIO()
        Image.new("RGB", (32, 32), "red").save(image, format="PNG")
        audio = BytesIO()
        with wave.open(audio, "wb") as stream:
            stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 800)
        video = self.root / "video.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(
                    np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
                )
            ):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        graph = text_graph()
        graph.update(
            {
                "li": {"class_type": "LoadImage", "inputs": {"image": "file.png"}},
                "si": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["li", 0], "filename_prefix": "outside"},
                },
                "la": {"class_type": "LoadAudio", "inputs": {"audio": "file.wav"}},
                "sa": {
                    "class_type": "SaveAudio",
                    "inputs": {
                        "audio": ["la", 0],
                        "filename_prefix": "outside",
                        "format": "flac",
                    },
                },
                "lv": {"class_type": "LoadVideo", "inputs": {"file": "file.mp4"}},
                "sv": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["lv", 0],
                        "filename_prefix": "outside",
                        "format": "mp4",
                        "codec": {"codec": "h264"},
                    },
                },
            }
        )
        files = {
            name: {"part": name, "targets": [{"node_id": node, "input": field}]}
            for name, node, field in (
                ("picture", "li", "image"),
                ("sound", "la", "audio"),
                ("clip", "lv", "file"),
            )
        }
        status, submitted = await self.submit(
            {
                "files": files,
                "outputs": {
                    name: {"node_id": node}
                    for name, node in (
                        ("image", "si"),
                        ("audio", "sa"),
                        ("video", "sv"),
                        ("text", "out"),
                    )
                },
            },
            {
                "picture": ("x.png", image.getvalue()),
                "sound": ("x.wav", audio.getvalue()),
                "clip": ("x.mp4", video.read_bytes()),
            },
            graph,
        )
        self.assertEqual(status, 202, submitted)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "succeeded", state)
        self.assertEqual(
            {item["kind"] for item in state["outputs"]},
            {"image", "audio", "video", "text"},
        )
        self.assertEqual(state["progress"]["executions_started"], 1)
        self.assertEqual(state["progress"]["outputs"], {"completed": 4, "total": 4})
        self.assertFalse((self.output / "outside").exists())
        for output in state["outputs"]:
            response = await self.client.get(output["url"])
            self.assertEqual(len(await response.read()), output["bytes"])

    async def test_invalid_bindings_formats_and_native_validation_leave_no_files(self):
        cases = [
            ({"template_id": "old.json"}, None),
            ({"api_version": 1}, None),
            ({"outputs": {"bad": {"node_id": "compute"}}}, None),
            (
                {
                    "files": {
                        "missing": {
                            "part": "missing",
                            "targets": [{"node_id": "compute", "input": "text"}],
                        }
                    }
                },
                None,
            ),
            ({}, {"nodes": []}),
            ({}, text_graph("MissingNode")),
            ({}, {"out": {"class_type": "ApiResult", "inputs": {}}}),
            (
                {"outputs": {"good": {"node_id": "out"}, "bad": {"node_id": "bad"}}},
                {**text_graph(), "bad": {"class_type": "ApiResult", "inputs": {}}},
            ),
        ]
        for data, graph in cases:
            with self.subTest(data=data, graph=graph):
                status, result = await self.submit(data, graph=graph)
                self.assertEqual(status, 400, result)
                if graph == {"out": {"class_type": "ApiResult", "inputs": {}}}:
                    details = result["error_details"]
                    self.assertEqual(details["code"], "workflow_invalid")
                    self.assertIsInstance(details["node_errors"]["out"], dict)
                    self.assertEqual(
                        details["node_errors"]["out"]["errors"][0]["type"],
                        "required_input_missing",
                    )
                if graph == text_graph("MissingNode"):
                    self.assertEqual(
                        result["error_details"]["issues"][0]["node_id"], "compute"
                    )
                self.assertFalse(self.jobs.jobs)
                self.assertEqual(
                    list((self.jobs.root / self.jobs.session_id).iterdir()), []
                )
        self.assertEqual((await self.client.get("/turing/templates")).status, 404)

    async def test_execution_failure_cleans_files_and_expires(self):
        _, submitted = await self.submit(graph=text_graph("ApiBroken"))
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "failed")
        self.assertIn("test execution failed", state["error"])
        self.assertEqual(state["error_details"]["node_id"], "compute")
        self.assertEqual(state["error_details"]["class_type"], "ApiBroken")
        self.assertFalse(self.jobs.jobs[state["id"]].root.exists())
        self.assertEqual(self.server.prompt_queue.history, {})
        self.jobs.jobs[state["id"]].expires = time.time() - 1
        self.jobs.sweep_interval = 0.01
        self.jobs.sweeper.cancel()
        await asyncio.gather(self.jobs.sweeper, return_exceptions=True)
        self.jobs.sweeper = asyncio.create_task(self.jobs._sweep())
        await asyncio.sleep(0.05)
        self.assertNotIn(state["id"], self.jobs.jobs)

    async def test_migration_error_is_reported_and_leaves_no_native_job(self):
        graph = {
            **text_graph(),
            "legacy": {
                "class_type": "TuringUtilsMiniMaxH3KSampler",
                "inputs": {"sigmas": "preset"},
            },
        }
        # An unrelated native hook can rewrite the request envelope after compilation.
        self.server.add_on_prompt_handler(lambda data: dict(data))
        with mock.patch.dict(
            nodes.NODE_CLASS_MAPPINGS,
            {"TuringUtilsMiniMaxH3KSampler": MiniMaxH3KSampler},
        ):
            status, result = await self.submit(graph=graph)
        self.assertEqual(status, 400, result)
        issue = result["error_details"]["issues"][0]
        self.assertEqual(issue["node_id"], "legacy")
        self.assertEqual(issue["code"], "workflow_migration_required")
        self.assertIn("final-size original LATENT", issue["message"])
        self.assertFalse(self.jobs.jobs)
        self.assertEqual(self.server.prompt_queue.get_current_queue_volatile(), ([], []))
        self.assertEqual(list((self.jobs.root / self.jobs.session_id).iterdir()), [])
        status, submitted = await self.submit()
        self.assertEqual(status, 202, submitted)
        self.assertEqual((await self.finished(submitted["id"]))["status"], "succeeded")

    async def test_request_id_deduplicates_and_rejects_deleted_or_different_requests(
        self,
    ):
        data = {"request_id": str(uuid.uuid4())}
        _, first = await self.submit(data)
        _, second = await self.submit(data)
        self.assertEqual(first["id"], second["id"])
        await self.finished(first["id"])
        self.assertEqual(
            (await self.submit({**data, "outputs": {"different": {"node_id": "out"}}}))[
                0
            ],
            400,
        )
        await self.client.delete("/turing/jobs/" + first["id"])
        self.assertEqual((await self.submit(data))[0], 400)
        self.assertEqual((await self.submit({"session_id": "old"}))[0], 400)

    async def test_delete_running_native_job_stops_before_removing_files(self):
        _, submitted = await self.submit(graph=text_graph("ApiWaiting"))
        job = self.jobs.jobs[submitted["id"]]
        async with asyncio.timeout(5):
            while not self.server.prompt_queue.currently_running:
                await asyncio.sleep(0.01)
        self.assertEqual(
            (await self.client.delete("/turing/jobs/" + job.id)).status, 202
        )
        async with asyncio.timeout(5):
            while job.id in self.jobs.jobs:
                await asyncio.sleep(0.01)
        self.assertFalse(job.root.exists())
        self.assertFalse(self.server.prompt_queue.currently_running)

    async def test_download_is_pinned_while_delete_is_pending(self):
        _, submitted = await self.submit()
        state = await self.finished(submitted["id"])
        job = self.jobs.jobs[state["id"]]
        entered, release = asyncio.Event(), asyncio.Event()
        original = web.StreamResponse.write

        async def blocked(response, data):
            entered.set()
            await release.wait()
            return await original(response, data)

        with mock.patch.object(web.StreamResponse, "write", blocked):
            response = await self.client.get(state["outputs"][0]["url"])
            await entered.wait()
            self.assertEqual(
                (await self.client.delete("/turing/jobs/" + job.id)).status, 202
            )
            self.assertTrue(job.root.exists())
            release.set()
            self.assertEqual(await response.text(), "source!")
        async with asyncio.timeout(5):
            while job.id in self.jobs.jobs:
                await asyncio.sleep(0.01)

    async def test_restart_removes_previous_session_and_keeps_unrelated_files(self):
        old_id = self.jobs.session_id
        ordinary = self.output / "ordinary-project"
        ordinary.mkdir()
        (ordinary / "keep.txt").write_text("keep")
        await self.client.close()
        orphan = self.jobs.root / old_id / uuid.uuid4().hex
        orphan.mkdir()
        (orphan / "old-video.mp4").write_bytes(b"unfinished")
        replacement = Jobs(self.server, self.output)
        await replacement.start()
        try:
            self.assertTrue(replacement.ready, replacement.error)
            self.assertFalse(orphan.parent.exists())
            self.assertTrue((ordinary / "keep.txt").exists())
        finally:
            await replacement.close()

    async def test_another_instance_cannot_delete_active_session(self):
        sentinel = self.jobs.root / self.jobs.session_id / "sentinel"
        sentinel.write_text("active")
        replacement = Jobs(self.server, self.output)
        await replacement.start()
        self.assertFalse(replacement.ready)
        self.assertTrue(sentinel.exists())
        await replacement.close()

    async def test_client_wait_download_verify_then_delete(self):
        async with Client(str(self.client.make_url("")), poll_interval=0.01) as client:
            result = await client.run(
                workflow=text_graph(),
                outputs={"result": {"node_id": "out"}},
                timeout=10,
            )
            paths = await client.download_all(result["id"], self.root / "downloads")
            self.assertEqual(paths[0].read_text(), "source!")
            self.assertNotIn(result["id"], self.jobs.jobs)

    async def test_local_capability_detach_timeout_resume_snapshots_and_cleans(
        self,
    ):
        root = project.initialize(self.root, str(self.client.make_url("")))
        directory = root / "capabilities" / "text"
        directory.mkdir()
        manifest = {
            "name": "text",
            "workflow": "workflow.api.json",
            "usage": "USAGE.md",
            "parameters": {
                "prompt": {
                    "type": "string",
                    "default": "hello",
                    "targets": [{"node_id": "compute", "input": "text"}],
                }
            },
            "outputs": {"text": {"node_id": "out"}},
        }
        project.write(directory / "capability.json", manifest)
        project.write(directory / "workflow.api.json", text_graph("ApiProgress"))
        (directory / "USAGE.md").write_text("Verified text generation.\n")
        async with Client(str(self.client.make_url("")), poll_interval=0.01) as client:
            record = await project.run(
                client, root, "text", {"prompt": "edited"}, {}, detach=True
            )
            self.assertEqual(await asyncio.wait_for(ProgressText.steps.get(), 5), 0)
            with self.assertRaises(TimeoutError) as caught:
                await project.resume(client, root, record["id"], timeout=0.02)
            self.assertEqual(caught.exception.details["job_id"], record["id"])
            waiting = project.read(root / "runs" / record["id"] / "run.json")
            self.assertEqual(waiting["status"], "running")
            self.assertEqual(waiting["activity"], "waiting_timeout")
            for _ in range(6):
                ProgressText.release.put_nowait(None)
            record = await project.resume(client, root, record["id"], timeout=10)
        self.assertEqual(record["state"]["progress"]["executions_started"], 1)
        self.assertEqual((root / record["outputs"][0]).read_text(), "edited!")
        saved = root / "runs" / record["id"]
        self.assertEqual(
            (saved / "USAGE.md").read_text(), "Verified text generation.\n"
        )
        self.assertEqual(
            project.read(saved / "workflow.api.json")["compute"]["inputs"]["text"],
            "edited",
        )
        self.assertTrue(project.read(saved / "run.json")["server_deleted"])
        self.assertNotIn(record["job_id"], self.jobs.jobs)
        self.assertEqual(
            project.verification(root, "text")["status"], "execution_verified"
        )
        self.assertEqual(
            (await project.status(None, root, record["id"]))["status"], "succeeded"
        )

    async def test_aliased_outputs_have_unique_download_names(self):
        async with Client(str(self.client.make_url("")), poll_interval=0.01) as client:
            state = await client.run(
                workflow=text_graph(),
                outputs={"a/b": {"node_id": "out"}, "a?b": {"node_id": "out"}},
                timeout=10,
            )
            paths = await client.download_all(state["id"], self.root / "downloads")
            self.assertEqual(len(set(paths)), 2)

    async def test_preview_image_batch_is_managed_and_reports_each_image(self):
        graph = {
            "empty": {
                "class_type": "EmptyImage",
                "inputs": {"width": 16, "height": 16, "batch_size": 2, "color": 0},
            },
            "out": {"class_type": "PreviewImage", "inputs": {"images": ["empty", 0]}},
        }
        status, submitted = await self.submit(graph=graph)
        self.assertEqual(status, 202, submitted)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "succeeded", state)
        self.assertEqual([entry["index"] for entry in state["outputs"]], [0, 1])
        self.assertEqual({entry["kind"] for entry in state["outputs"]}, {"image"})
        self.assertFalse((self.jobs.jobs[state["id"]].root / "work").exists())

    async def test_absolute_file_binding_and_duplicate_targets(self):
        graph = text_graph()
        graph["compute"] = {
            "class_type": "ApiFile",
            "inputs": {"path": "placeholder.txt"},
        }
        binding = {
            "part": "source",
            "targets": [
                {"node_id": "compute", "input": "path", "path_mode": "absolute"}
            ],
        }
        status, submitted = await self.submit(
            {"files": {"source": binding}},
            {"source": ("source.txt", "你好".encode())},
            graph,
        )
        self.assertEqual(status, 202, submitted)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "succeeded", state)
        self.assertEqual(
            await (await self.client.get(state["outputs"][0]["url"])).text(), "你好"
        )
        binding["targets"] *= 2
        status, result = await self.submit(
            {"files": {"source": binding}}, {"source": ("source.txt", b"text")}, graph
        )
        self.assertEqual(status, 400, result)

    async def test_nested_file_target_preserves_sibling_options(self):
        graph = text_graph()
        graph["compute"] = {
            "class_type": "ApiNestedFile",
            "inputs": {"options": {"file": "placeholder", "other": " retained"}},
        }
        files = {
            "source": {
                "part": "source",
                "targets": [
                    {
                        "node_id": "compute",
                        "input": "options",
                        "path": ["file"],
                        "path_mode": "absolute",
                    }
                ],
            }
        }
        status, submitted = await self.submit(
            {"files": files}, {"source": ("x.txt", b"nested")}, graph
        )
        self.assertEqual(status, 202, submitted)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "succeeded", state)
        self.assertEqual(
            await (await self.client.get(state["outputs"][0]["url"])).text(),
            "nested retained",
        )

    async def test_unmanaged_output_is_rejected_without_reading_or_removing_it(self):
        external = self.output / "keep.txt"
        external.write_text("user-owned")
        graph = text_graph()
        graph["out"]["class_type"] = "ApiForeign"
        _, submitted = await self.submit(graph=graph)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["status"], "failed", state)
        self.assertIn("outside", state["error"])
        self.assertEqual(external.read_text(), "user-owned")
        self.assertFalse(self.jobs.jobs[state["id"]].root.exists())

    async def test_native_prompt_id_conflict_does_not_interrupt_other_job(self):
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)
        identity = str(uuid.uuid4())
        self.server.prompt_queue.put((999, identity, {}, {}, [], {}))
        status, result = await self.submit({"request_id": identity})
        self.assertEqual(status, 400, result)
        self.assertFalse(self.jobs.jobs)
        self.assertEqual(
            [item[1] for item in self.server.prompt_queue.queue], [identity]
        )
        self.server.prompt_queue.delete_queue_item(lambda item: item[1] == identity)

    async def test_upload_cancellation_removes_partial_directory(self):
        identity = str(uuid.uuid4())
        release = asyncio.Event()
        request = {
            "api_version": 2,
            "workflow": "graph",
            "request_id": identity,
            "outputs": {"result": {"node_id": "out"}},
            "files": {
                "text": {
                    "part": "file",
                    "targets": [{"node_id": "compute", "input": "text"}],
                }
            },
        }
        document = json.dumps(request)

        async def body():
            yield (
                '--upload\r\nContent-Disposition: form-data; name="request"\r\n\r\n'
                + document
                + '\r\n--upload\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
            ).encode()
            yield b"unfinished" * 8192
            await release.wait()
            yield b"\r\n--upload--\r\n"

        pending = asyncio.create_task(
            self.client.post(
                "/turing/jobs",
                data=body(),
                headers={"Content-Type": "multipart/form-data; boundary=upload"},
            )
        )
        async with asyncio.timeout(5):
            while identity not in self.jobs.jobs:
                await asyncio.sleep(0.01)
        job = self.jobs.jobs[identity]
        self.assertEqual(
            (await self.client.delete("/turing/jobs/" + identity)).status, 202
        )
        async with asyncio.timeout(5):
            while identity in self.jobs.jobs:
                await asyncio.sleep(0.01)
        pending.cancel()
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        self.assertFalse(job.root.exists())

    async def test_native_progress_is_live_without_polling_and_isolated_from_queued_job(
        self,
    ):
        _, submitted = await self.submit(graph=text_graph("ApiProgress"))
        _, second = await self.submit()
        self.assertEqual(await ProgressText.steps.get(), 0)
        for value in range(1, 5):
            ProgressText.release.put_nowait(None)
            self.assertEqual(await ProgressText.steps.get(), value)
            await asyncio.sleep(0)
            state = self.jobs.state(self.jobs.jobs[submitted["id"]])
            self.assertEqual(
                next(
                    node
                    for node in state["progress"]["active_nodes"]
                    if node["class_type"] == "ApiProgress"
                )["value"],
                value,
            )
            other = self.jobs.state(self.jobs.jobs[second["id"]])
            self.assertEqual(other["status"], "queued")
            self.assertEqual(other["queue"], {"position": 1, "ahead": 1})
            self.assertFalse(other["progress"]["active_nodes"])
        for _ in range(2):
            ProgressText.release.put_nowait(None)
        state = await self.finished(submitted["id"])
        self.assertEqual(state["progress"]["executions_completed"], 1)
        await self.finished(second["id"])

    async def test_history_is_reused_only_for_same_owner_and_workload(self):
        for _ in range(3):
            _, submitted = await self.submit()
            state = await self.finished(submitted["id"])
        self.assertEqual(state["timing"]["history_samples"], 3)
        _, submitted = await self.submit(graph=text_graph(text="different"))
        self.assertEqual(
            (await self.finished(submitted["id"]))["timing"]["history_samples"], 1
        )

    async def test_client_preserves_server_results_when_checksum_fails(self):
        async with Client(str(self.client.make_url("")), poll_interval=0.01) as client:
            result = await client.run(
                workflow=text_graph(),
                outputs={"result": {"node_id": "out"}},
                timeout=10,
            )
            job = self.jobs.jobs[result["id"]]
            job.outputs[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "verification"):
                await client.download_all(job.id, self.root / "downloads")
            self.assertIn(job.id, self.jobs.jobs)
            self.assertEqual(list((self.root / "downloads").iterdir()), [])

    async def test_queued_delete_keeps_unrelated_native_task(self):
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)
        _, submitted = await self.submit()
        job = self.jobs.jobs[submitted["id"]]
        other_id = str(uuid.uuid4())
        self.server.prompt_queue.put((999, other_id, {}, {}, [], {}))
        self.assertEqual(
            (await self.client.delete("/turing/jobs/" + job.id)).status, 202
        )
        await job.runner
        self.assertFalse(job.root.exists())
        self.assertEqual(
            [item[1] for item in self.server.prompt_queue.queue], [other_id]
        )
        self.server.prompt_queue.delete_queue_item(lambda item: item[1] == other_id)

    async def test_cleanup_failure_is_reported_and_can_be_retried(self):
        _, submitted = await self.submit()
        state = await self.finished(submitted["id"])
        job = self.jobs.jobs[state["id"]]
        original = shutil.rmtree

        def fail(path, *args, **kwargs):
            if Path(path) == job.root:
                raise PermissionError("test file is busy")
            return original(path, *args, **kwargs)

        with mock.patch("shutil.rmtree", side_effect=fail):
            response = await self.client.delete("/turing/jobs/" + job.id)
            self.assertEqual(response.status, 202)
            self.assertIn("busy", (await response.json())["cleanup_error"])
        self.assertEqual(
            (await self.client.delete("/turing/jobs/" + job.id)).status, 204
        )

    async def test_startup_rejects_symlink_without_removing_external_files(self):
        await self.client.close()
        external = self.root / "external"
        external.mkdir()
        (external / "keep.txt").write_text("keep")
        (self.jobs.root / uuid.uuid4().hex).symlink_to(
            external, target_is_directory=True
        )
        replacement = Jobs(self.server, self.output)
        await replacement.start()
        self.assertFalse(replacement.ready)
        self.assertTrue((external / "keep.txt").exists())

    async def test_previous_session_404_is_reported_by_client(self):
        async with Client(str(self.client.make_url(""))) as client:
            client.session_id = "old-session"
            with self.assertRaises(ServerRestarted):
                await client.status(str(uuid.uuid4()))


class ClientRestartTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = "old"
        self.restarts = 0
        self.allowed = True
        self.change_session = True
        app = web.Application()

        async def status(request):
            return web.json_response(
                {
                    "session_id": self.session,
                    "ready": True,
                    "restart_supported": True,
                    "restart_url": "/manager/reboot",
                }
            )

        async def restart(request):
            self.restarts += 1
            if not self.allowed:
                raise web.HTTPForbidden(text="Manager policy")
            if self.change_session:
                self.session = "new"
            return web.Response(text="Restarting")

        app.router.add_get("/turing/server", status)
        app.router.add_post("/manager/reboot", restart)
        self.http = TestClient(TestServer(app))
        await self.http.start_server()
        self.client = await Client(
            str(self.http.make_url("")), poll_interval=0.01
        ).__aenter__()

    async def asyncTearDown(self):
        await self.client.__aexit__()
        await self.http.close()

    async def test_restart_waits_for_a_changed_session(self):
        result = await self.client.restart(timeout=1)
        self.assertEqual(result["session_id"], "new")
        self.assertEqual(self.client.session_id, "new")
        self.assertEqual(self.restarts, 1)

    async def test_success_response_without_a_new_session_is_not_a_restart(self):
        self.change_session = False
        with self.assertRaises(TimeoutError):
            await self.client.restart(timeout=0.05)
        self.assertEqual(self.restarts, 1)

    async def test_manager_permission_failure_is_not_bypassed_or_retried(self):
        self.allowed = False
        with self.assertRaises(ApiError) as caught:
            await self.client.restart(timeout=1)
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(self.restarts, 1)


if __name__ == "__main__":
    unittest.main()
