from __future__ import annotations

import ast
from functools import cache
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMFY_ROOT = ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


class PackageArchitectureTest(unittest.TestCase):
    def test_plugin_has_no_workspace_or_frontend_runtime(self):
        source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("WEB_DIRECTORY", source)
        self.assertFalse((ROOT / "web").exists())
        self.assertFalse((ROOT / "comfyui_turing_utils" / "workspace").exists())

    def test_kernel_package_is_accessed_only_through_facade(self):
        package = ROOT / "comfyui_turing_utils"
        offenders = []
        for path in package.rglob("*.py"):
            if path.name == "kernel_api.py":
                continue
            tree = syntax(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(
                    name.startswith("comfyui_turing_utils_kernel") for name in modules
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_implementation_package_uses_relative_self_imports(self):
        """ComfyUI may load the plugin root under a generated module name."""

        package = ROOT / "comfyui_turing_utils"
        offenders = []
        for path in package.rglob("*.py"):
            tree = syntax(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                if any(
                    name == "comfyui_turing_utils"
                    or name.startswith("comfyui_turing_utils.")
                    for name in modules
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


    def test_convrot_quantization_does_not_orchestrate_model_loading(self):
        path = ROOT / "comfyui_turing_utils" / "quantization" / "convrot.py"
        tree = syntax(path)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any("adapters" in name for name in imported))
        self.assertFalse(any("attention" in name for name in imported))
        self.assertFalse(any(name == "folder_paths" for name in imported))
        self.assertTrue(
            (ROOT / "comfyui_turing_utils" / "loading" / "convrot.py").is_file()
        )

    def test_h3_services_are_separate_from_node_schemas(self):
        node_imports = tuple(imports(ROOT / "comfyui_turing_utils/nodes/minimax_references.py"))
        assert "comfyui_turing_utils.adapters.minimax.reference_execution" in node_imports
        assert not any(name.startswith("comfy_api") for name in imports(
            ROOT / "comfyui_turing_utils/adapters/minimax/references.py"
        ))

    def test_builtin_model_adapters_are_registered_once(self):
        from comfyui_turing_utils import bootstrap_builtin_integrations
        from comfyui_turing_utils.adapters.registry import registered_model_adapters

        bootstrap_builtin_integrations()
        bootstrap_builtin_integrations()
        self.assertEqual(registered_model_adapters(), ("minimax_h3", "wan", "qwen_image21"))

    def test_server_integrations_install_once_after_standalone_bootstrap(self):
        # Other tests import and patch ComfyUI in this process.
        script = textwrap.dedent(
            """
            import os
            import sys
            from types import SimpleNamespace

            from aiohttp import web
            from comfy import memory_management, model_management
            from comfy_execution.graph import ExecutionList
            from server import PromptServer

            sys.path.insert(0, sys.argv[1])
            from comfyui_turing_utils import bootstrap_builtin_integrations
            from comfyui_turing_utils.bootstrap import initialize_plugin

            original_staging = ExecutionList.stage_node_execution
            bootstrap_builtin_integrations()
            assert ExecutionList.stage_node_execution is original_staging
            initialize_plugin()
            assert ExecutionList.stage_node_execution is not original_staging
            installed_staging = ExecutionList.stage_node_execution

            handlers = []
            server = SimpleNamespace(
                add_on_prompt_handler=handlers.append,
                app=web.Application(),
                routes=web.RouteTableDef(),
            )
            PromptServer.instance = server
            from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS

            assert "TuringUtilsStageBarrier" in NODE_CLASS_MAPPINGS
            assert not handlers and not list(server.routes)

            pin_setting = "COMFYUI_TURING_UTILS_PINNED_MEMORY_GIB"
            os.environ[pin_setting] = "6"
            memory_management.aimdo_enabled = True
            model_management.WINDOWS = True
            model_management.DISABLE_PINNED_MEMORY = True
            model_management.MAX_PINNED_MEMORY = -1
            initialize_plugin()
            assert model_management.MAX_PINNED_MEMORY == -1

            jobs = server._turing_job_api
            installed_handlers = tuple(handlers)
            assert installed_handlers
            route_count = len(server.routes)
            model_management.DISABLE_PINNED_MEMORY = False
            initialize_plugin()
            assert model_management.MAX_PINNED_MEMORY == 6 * 1024**3
            assert ExecutionList.stage_node_execution is installed_staging
            assert server._turing_job_api is jobs
            assert tuple(handlers) == installed_handlers
            assert len(server.app.cleanup_ctx) == 1
            assert len(server.routes) == route_count
            assert any(route.path == "/turing/jobs" for route in server.routes)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(ROOT)],
            cwd=COMFY_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_registered_node_ids_match_the_maintained_surface(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS

        self.assertEqual(
            tuple(NODE_CLASS_MAPPINGS),
            (
                "TuringUtilsVideoFramesPadding",
                "TuringUtilsAttentionStrategy",
                "TuringUtilsConvRotDiffusionModelLoader",
                "TuringUtilsConvRotCLIPLoader",
                "TuringUtilsQwenImage21Optimize",
                "TuringUtilsQwenImage21Conditioning",
                "TuringUtilsSetVideoLatentNoiseMask",
                "TuringUtilsVideoLatentCompositeMasked",
                "TuringUtilsBerniniContextWindowsCore",
                "TuringUtilsBerniniInpaintCondition",
                "TuringUtilsKrea2IdentityEditConditioning",
                "TuringUtilsIsInputPresent",
                "TuringUtilsLazyIfElse",
                "TuringUtilsStageBarrier",
                "TuringUtilsStagePath",
                "TuringUtilsSigmasToString",
                "TuringUtilsH3AddNoise",
                "TuringUtilsH3LatentInfo",
                "TuringUtilsH3KeyframeReference",
                "TuringUtilsH3ImageReference",
                "TuringUtilsH3VideoReference",
                "TuringUtilsH3AudioReference",
                "TuringUtilsH3SemanticReference",
                "TuringUtilsH3BuildConditioning",
                "_TuringUtilsH3UpscaleLoader",
                "TuringUtilsMiniMaxH3LatentUpscale",
                "_TuringUtilsH3UpscaleApply",
                "TuringUtilsMiniMaxH3KSampler",
                "TuringUtilsMiniMaxH3ImageKSampler",
                "_TuringUtilsH3KSamplerApply",
                "TuringUtilsResizeImageIfPresent",
                "TuringUtilsVideoMotionContactSheet",
                "TuringUtilsMultimodalPromptChat",
                "TuringUtilsMiniMaxH3VideoVAEDecode",
                "TuringUtilsMiniMaxH3VideoVAEEncode",
                "TuringUtilsLoadIndexedVideoSegment",
                "TuringUtilsSaveIndexedVideoSegment",
                "TuringUtilsMergeIndexedVideoSegments",
                "TuringUtilsVideoPrefixContextNoise",
                "TuringUtilsVideoContinuationConcat",
                "TuringUtilsTrimVideoContinuationPrefix",
                "TuringUtilsH3SetAudioPrefixNoiseMask",
                "TuringUtilsVideoMaskGuidedCrop",
                "TuringUtilsVideoMaskGuidedStitch",
                "TuringUtilsVideoPadForOutpaint",
                "TuringUtilsMaskToVisualPrompts",
                "_TuringUtilsSeCLoader",
                "TuringUtilsSeCTrackVisualConcept",
                "_TuringUtilsSeCApply",
            ),
        )


if __name__ == "__main__":
    unittest.main()


PACKAGE = Path(__file__).resolve().parents[1] / "comfyui_turing_utils"


@cache
def syntax(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imports(path):
    relative = path.relative_to(PACKAGE).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts.pop()
    package = ".".join(["comfyui_turing_utils", *parts])
    for node in ast.walk(syntax(path)):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            yield module
            yield from (module + "." + alias.name for alias in node.names)


def test_shared_domains_do_not_import_node_schemas_or_canvas():
    forbidden = (
        "comfyui_turing_utils.nodes",
        "comfyui_turing_utils.workspace",
        "comfyui_turing_utils.registration",
    )
    for domain in ("media", "prompt", "attention", "quantization"):
        for path in (PACKAGE / domain).rglob("*.py"):
            for module in imports(path):
                assert not module.startswith(forbidden), (path, module)


def test_generic_attention_and_quantization_do_not_import_model_adapters():
    for domain in ("attention", "quantization"):
        for path in (PACKAGE / domain).rglob("*.py"):
            for module in imports(path):
                assert not module.startswith("comfyui_turing_utils.adapters"), (
                    path,
                    module,
                )


def test_shared_execution_does_not_depend_on_its_consumers():
    boundaries = {
        "attention/execution.py": (
            "comfyui_turing_utils.attention.dense",
            "comfyui_turing_utils.attention.sol",
            "comfyui_turing_utils.attention.sla",
            "comfyui_turing_utils.attention.patches",
        ),
        "adapters/minimax/execution.py": (
            "comfyui_turing_utils.adapters.minimax.acceleration",
            "comfyui_turing_utils.adapters.minimax.attention_ops",
            "comfyui_turing_utils.adapters.minimax.mlp",
        ),
    }
    for file, forbidden in boundaries.items():
        for module in imports(PACKAGE / file):
            assert not module.startswith(forbidden), (file, module)
