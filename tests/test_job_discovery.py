"""Discovery stays small, uses native schemas and survives unavailable jobs."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

import nodes
from comfy_api.latest import io
from comfyui_turing_utils.job_api import discovery, routes
from comfyui_turing_utils.job_api.client import ApiError, Client
from comfyui_turing_utils.runtime.capabilities import CapabilityResult


class Legacy:
    CATEGORY = "Testing/Discovery"
    DESCRIPTION = "An H3 reference utility."
    SEARCH_ALIASES = ("identity transfer",)

    @classmethod
    def INPUT_TYPES(cls):
        raise AssertionError("Catalog must not enumerate legacy loader inputs")


class Internal(Legacy):
    DEV_ONLY = True


class Deprecated(Legacy):
    DEPRECATED = True


class Dynamic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Discovery Dynamic β",
            display_name="Dynamic discovery",
            category="Testing/Discovery",
            search_aliases=["branch search"],
            inputs=[
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option("single", []),
                        io.DynamicCombo.Option(
                            "refine",
                            [
                                io.Float.Input(
                                    "strength",
                                    default=0.25,
                                    tooltip="Refinement strength",
                                ),
                            ],
                        ),
                    ],
                ),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=4,
                    ),
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(None)


class BrokenMeta(type):
    def __getattribute__(cls, name):
        if name == "DEV_ONLY":
            raise RuntimeError("private source path must not leak")
        return super().__getattribute__(name)


class Broken(metaclass=BrokenMeta):
    pass


class DiscoveryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.jobs = SimpleNamespace(
            ready=True,
            error=None,
            session_id="test-session",
            result_ttl=37.5,
            start=mock.AsyncMock(),
            close=mock.AsyncMock(),
        )
        self.server = SimpleNamespace(
            routes=web.RouteTableDef(),
            app=web.Application(),
            _turing_material_tasks=mock.Mock(),
        )
        for patch in (
            mock.patch.object(routes, "Jobs", return_value=self.jobs),
            mock.patch.object(routes.args, "max_upload_size", 0.125),
            mock.patch.object(
                discovery, "runtime_summary", return_value={"device": {"type": "cpu"}}
            ),
            mock.patch.dict(
                nodes.NODE_CLASS_MAPPINGS,
                {
                    "Alpha": Legacy,
                    "Beta": Legacy,
                    "_Private": Legacy,
                    "Internal": Internal,
                    "Old": Deprecated,
                    "Discovery Dynamic β": Dynamic,
                },
                clear=True,
            ),
            mock.patch.dict(
                nodes.NODE_DISPLAY_NAME_MAPPINGS, {"Beta": "Visible alias"}, clear=True
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        routes.install_routes(self.server)

        async def native_info(request):
            name = request.match_info["class_type"]
            return web.json_response(
                {name: Dynamic.GET_NODE_INFO_V1()}
                if name == "Discovery Dynamic β"
                else {}
            )

        async def native_models(request):
            folder = request.match_info.get("folder")
            return web.json_response(
                ["diffusion_models"]
                if folder is None
                else [folder + "/model.safetensors"]
            )

        self.server.app.add_routes(self.server.routes)
        self.server.app.router.add_get("/object_info/{class_type}", native_info)
        self.server.app.router.add_get("/models", native_models)
        self.server.app.router.add_get("/models/{folder}", native_models)
        self.http = TestClient(TestServer(self.server.app))
        await self.http.start_server()
        self.addAsyncCleanup(self.http.close)
        self.client = await Client(str(self.http.make_url(""))).__aenter__()
        self.addAsyncCleanup(self.client.__aexit__)

    async def test_server_is_the_single_entrypoint_for_state_links_and_usage(self):
        state = await self.client.server()
        self.assertTrue(state["ready"])
        self.assertEqual(state["session_id"], "test-session")
        self.assertEqual(state["api_version"], 2)
        self.assertEqual(state["links"]["nodes"], "/turing/nodes")
        self.assertEqual(state["links"]["node_info"], "/object_info/{class_type}")
        self.assertEqual(state["usage"]["limits"]["upload_bytes"], 131072)
        self.assertEqual(state["usage"]["limits"]["request_json_bytes"], 131072)
        self.assertEqual(state["usage"]["lifecycle"]["result_ttl_seconds"], 37.5)
        self.assertTrue(state["usage"]["raw_prompt_supported"])
        self.assertEqual((await self.http.get("/turing/help")).status, 404)

    async def test_discovery_survives_unavailable_jobs_and_runtime_detection(self):
        self.jobs.ready, self.jobs.error = False, "Cleanup is blocked"
        with mock.patch.object(
            discovery, "runtime_summary", side_effect=RuntimeError("private path")
        ):
            response = await self.http.get("/turing/server")
            self.assertEqual(response.status, 503)
            state = await response.json()
        self.assertEqual(state["error"], "Cleanup is blocked")
        self.assertIn("usage", state)
        self.assertNotIn("private path", json.dumps(state["runtime"]))
        self.assertTrue((await self.client.nodes())["items"])
        self.assertEqual((await self.http.get("/turing/templates")).status, 404)

    async def test_search_filters_aliases_categories_and_stable_pagination(self):
        with mock.patch(
            "folder_paths.get_filename_list",
            side_effect=AssertionError("Unexpected model scan"),
        ):
            first = await self.client.nodes(
                "identity transfer", category="testing", limit=1
            )
            self.assertEqual([item["class_type"] for item in first["items"]], ["Alpha"])
            second = await self.client.nodes(
                "identity transfer",
                category="Testing/Discovery",
                limit=1,
                cursor=first["next_cursor"],
            )
            self.assertEqual([item["class_type"] for item in second["items"]], ["Beta"])
            self.assertIsNone(second["next_cursor"])
            self.assertEqual(second["total"], 2)
            self.assertEqual(second["items"][0]["display_name"], "Visible alias")
            self.assertFalse(
                (await self.client.nodes(category="Testing/Discover"))["items"]
            )
            all_nodes = await self.client.nodes(
                include_internal=True, include_deprecated=True
            )
            self.assertEqual(len(all_nodes["items"]), 6)
            internal = next(
                item for item in all_nodes["items"] if item["class_type"] == "Internal"
            )
            self.assertTrue(internal["internal"])

    async def test_native_details_keep_dynamic_branches_autogrow_and_tooltips(self):
        result = await self.client.nodes("branch search")
        self.assertEqual(result["total"], 1)
        item = result["items"][0]
        self.assertIn("%CE%B2", item["detail_url"])
        self.assertEqual(item["display_name"], "Dynamic discovery")
        info = await self.client.node_info(item["class_type"])
        inputs = info["input"]["required"]
        options = inputs["mode"][1]["options"]
        strength = options[1]["inputs"]["required"]["strength"][1]
        self.assertEqual(strength["default"], 0.25)
        self.assertEqual(strength["tooltip"], "Refinement strength")
        self.assertEqual(inputs["images"][1]["template"]["min"], 0)
        self.assertEqual(inputs["images"][1]["template"]["max"], 4)
        with self.assertRaises(ApiError) as caught:
            await self.client.node_info("missing")
        self.assertEqual(caught.exception.status, 404)

    async def test_broken_third_party_metadata_does_not_hide_other_nodes(self):
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {"Broken": Broken}):
            result = await self.client.nodes()
        self.assertEqual(len(result["items"]), 3)
        self.assertEqual(result["errors"][0]["class_type"], "Broken")
        self.assertNotIn("private source path", json.dumps(result))

    async def test_invalid_search_parameters_are_reported(self):
        for query in (
            {"limit": 0},
            {"limit": 101},
            {"limit": "bad"},
            {"include_internal": "yes"},
            {"include_deprecated": "1"},
            {"cursor": "missing"},
            {"unknown": "field"},
        ):
            with self.subTest(query=query):
                response = await self.http.get("/turing/nodes", params=query)
                self.assertEqual(response.status, 400)
                self.assertIn("error", await response.json())

    async def test_client_uses_native_model_routes(self):
        self.assertEqual(await self.client.models(), ["diffusion_models"])
        self.assertEqual(
            await self.client.models("custom space"), ["custom space/model.safetensors"]
        )


class RuntimeSummaryTest(unittest.TestCase):
    def test_binary_support_is_cached_without_model_execution(self):
        discovery.runtime_summary.cache_clear()
        self.addCleanup(discovery.runtime_summary.cache_clear)
        runtime = SimpleNamespace(
            device=SimpleNamespace(
                device=SimpleNamespace(type="cpu"),
                name="",
                architecture=None,
                total_memory=0,
            ),
            kernel=SimpleNamespace(installed=False, version=(0, 0, 0)),
            supports=lambda feature: CapabilityResult(False, "No compatible binary"),
        )
        with mock.patch.object(
            discovery, "runtime_capabilities", return_value=runtime
        ) as probe:
            with mock.patch.object(
                discovery.model_management, "get_torch_device", return_value="cpu"
            ):
                first = discovery.runtime_summary()
                second = discovery.runtime_summary()
        probe.assert_called_once_with("cpu")
        self.assertEqual(first, second)
        self.assertFalse(first["support"]["sol"]["supported"])
        self.assertEqual(first["support"]["sol"]["reason"], "No compatible binary")
