"""Progress isolation and estimates use deterministic clocks, without weights."""

from collections import deque
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfyui_turing_utils.job_api.native_queue import NativeQueue
from comfyui_turing_utils.job_api.progress import Node, Observer, Progress


class JobProgressTest(unittest.TestCase):
    def start(self, progress, identity="first", now=10):
        progress.event(
            "execution_start",
            {"prompt_id": identity, "timestamp": 100000},
            now,
            {"same": ("Sampler", "Generate")},
        )

    def update(
        self,
        progress,
        value,
        maximum=5,
        now=10,
        identity="first",
        node="same",
        state="running",
    ):
        progress.event(
            "progress_state",
            {
                "prompt_id": identity,
                "nodes": {
                    node: {
                        "state": state,
                        "value": value,
                        "max": maximum,
                        "real_node_id": "same",
                    }
                },
            },
            now,
        )

    def test_first_slow_step_is_excluded_and_estimate_expires(self):
        node = Node("sampler", "Sampler", None, 0, 0)
        node.update(0, 5, 0)
        node.update(1, 5, 100)
        node.update(2, 5, 110)
        self.assertIsNone(node.snapshot(110)["eta_seconds"])
        node.update(3, 5, 120)
        self.assertEqual(node.snapshot(120)["eta_seconds"], 20)
        self.assertEqual(node.snapshot(125)["eta_seconds"], 15)
        self.assertIsNone(node.snapshot(145)["eta_seconds"])
        node.update(0, 3, 150)
        self.assertIsNone(node.snapshot(150)["eta_seconds"])

    def test_duplicate_snapshots_do_not_change_rate_or_report_fake_steps(self):
        progress = Progress()
        self.start(progress)
        self.update(progress, 0, 1, now=10)
        self.assertIsNone(progress.nodes["same"].snapshot(10)["value"])
        for value, now in ((1, 100), (2, 110), (2, 115), (3, 120)):
            self.update(progress, value, now=now)
        self.assertEqual(list(progress.nodes["same"].rates), [10, 10])
        self.update(progress, 5, now=140, state="finished")
        self.update(progress, 5, now=180, state="finished")
        self.assertEqual(progress.nodes["same"].snapshot(200)["elapsed_seconds"], 130)

    def test_multi_prompt_and_dynamic_nodes_ignore_stale_updates(self):
        progress = Progress()
        self.start(progress)
        self.update(progress, 2, now=12)
        self.update(progress, 1, now=14, node="dynamic")
        self.assertEqual(progress.nodes["dynamic"].class_type, "Sampler")
        progress.event("execution_success", {"prompt_id": "first"}, 15)
        self.assertEqual(progress.completed_executions, 1)
        self.start(progress, "second", 20)
        self.update(progress, 5, now=25, identity="first")
        self.assertFalse(progress.nodes)
        self.assertEqual(progress.executions, 2)
        self.update(progress, 1, now=22, identity="second")
        self.assertEqual(progress.nodes["same"].value, 1)
        progress.event("execution_error", {"prompt_id": "second"}, 25)
        self.assertEqual(progress.nodes["same"].state, "failed")
        self.assertEqual(progress.completed_executions, 1)

    def test_elapsed_queue_and_history_eta_are_separate_and_freeze(self):
        with mock.patch(
            "comfyui_turing_utils.job_api.progress.time.monotonic", return_value=0
        ):
            progress = Progress()
        progress.transition("queued", 2)
        self.start(progress, now=12)
        progress.signature = ("owner", "workload")
        history = {progress.signature: deque([100, 110])}
        with mock.patch(
            "comfyui_turing_utils.job_api.progress.time.monotonic", return_value=32
        ):
            snapshot = progress.snapshot(history)
        self.assertEqual(snapshot["timing"]["queue_seconds"], 10)
        self.assertEqual(snapshot["timing"]["execution_seconds"], 20)
        self.assertEqual(snapshot["timing"]["eta_seconds"], 85)
        with mock.patch(
            "comfyui_turing_utils.job_api.progress.time.monotonic", return_value=132
        ):
            self.assertEqual(
                progress.snapshot(history)["timing"]["estimate_status"], "overdue"
            )
        with mock.patch(
            "comfyui_turing_utils.job_api.progress.time.monotonic", return_value=42
        ):
            progress.finish("succeeded")
        self.assertEqual(progress.snapshot(history)["timing"]["elapsed_seconds"], 42)
        self.assertEqual(progress.snapshot(history)["timing"]["eta_seconds"], 0)
        self.assertEqual(progress.snapshot({})["timing"]["elapsed_seconds"], 42)

    def test_queued_or_inconsistent_history_has_no_total_eta(self):
        progress = Progress()
        progress.signature = ("owner", "workload")
        history = {progress.signature: deque([10, 100])}
        with mock.patch(
            "comfyui_turing_utils.job_api.progress.time.monotonic",
            return_value=progress.created + 1,
        ):
            progress.transition("queued")
            self.assertEqual(
                progress.snapshot(history)["timing"]["estimate_status"], "queued"
            )
            self.start(progress, now=progress.created + 1)
            self.assertIsNone(progress.snapshot(history)["timing"]["eta_seconds"])
            self.assertIsNone(progress.snapshot({})["timing"]["eta_seconds"])


class JobProgressObserverTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_thread_events_require_native_ownership_and_still_reach_server(
        self,
    ):
        forwarded = []
        item = (
            0,
            "job",
            {"node": {"class_type": "Sampler", "inputs": {"secret": "private"}}},
            {"turing_api_job": "job"},
        )
        server = SimpleNamespace(
            send_sync=lambda *args: forwarded.append(args),
            prompt_queue=SimpleNamespace(
                get_current_queue_volatile=lambda: ([item], [])
            ),
        )
        original = server.send_sync
        job = SimpleNamespace(status="queued", progress=Progress())
        jobs = {"job": job}
        observer = Observer(jobs)
        unsubscribe = NativeQueue(server).subscribe(observer.event, accepts=jobs.__contains__)
        try:
            data = {"prompt_id": "job", "timestamp": 100000}
            await asyncio.to_thread(
                server.send_sync, "execution_start", data, "ordinary-client"
            )
            await asyncio.sleep(0)
            self.assertEqual(job.status, "queued")
            item[3]["turing_api_job"] = "another"
            await asyncio.to_thread(server.send_sync, "execution_start", data, "job")
            await asyncio.sleep(0)
            self.assertEqual(job.status, "queued")
            item[3]["turing_api_job"] = "job"
            await asyncio.to_thread(server.send_sync, "execution_start", data, "job")
            await asyncio.sleep(0)
            self.assertEqual(job.status, "running")
            self.assertEqual(job.progress.graph, {"node": ("Sampler", None)})
            self.assertEqual(len(forwarded), 3)
            job.status = "succeeded"
            job.progress.finish("succeeded")
            await asyncio.to_thread(server.send_sync, "execution_start", data, "job")
            await asyncio.sleep(0)
            self.assertEqual(job.status, "succeeded")
        finally:
            unsubscribe()
        self.assertIs(server.send_sync, original)

    async def test_history_is_owner_scoped_bounded_and_excludes_failed_tasks(self):
        observer = Observer({})
        for index in range(140):
            progress = Progress()
            progress.signature = ("owner", str(index))
            progress.executions = 1
            progress.finish("succeeded")
            observer.remember(SimpleNamespace(status="succeeded", progress=progress))
        self.assertEqual(len(observer.history), 128)
        self.assertNotIn(("owner", "0"), observer.history)
        self.assertNotIn(("other", "139"), observer.history)
        progress.signature = ("owner", "failed")
        observer.remember(SimpleNamespace(status="failed", progress=progress))
        self.assertNotIn(progress.signature, observer.history)
