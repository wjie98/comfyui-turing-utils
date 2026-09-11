from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PackageArchitectureTest(unittest.TestCase):
    def test_frontend_extension_is_registered(self):
        source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('WEB_DIRECTORY = "./web"', source)
        self.assertTrue((ROOT / "web" / "keyframe_outputs.js").is_file())
        self.assertTrue((ROOT / "web" / "stage_barrier_outputs.js").is_file())

    def test_kernel_package_is_accessed_only_through_facade(self):
        package = ROOT / "comfyui_turing_utils"
        offenders = []
        for path in package.rglob("*.py"):
            if path.name == "kernel_api.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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

    def test_attention_depends_on_layout_contract_not_minimax_adapter(self):
        attention_root = ROOT / "comfyui_turing_utils" / "attention"
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in attention_root.glob("*.py")
        )
        self.assertNotIn("adapters.minimax", source)
        self.assertIn("ensure_attention_layout_provider", source)

    def test_convrot_quantization_does_not_orchestrate_model_loading(self):
        path = ROOT / "comfyui_turing_utils" / "quantization" / "convrot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any("adapters" in name for name in imported))
        self.assertFalse(any("attention" in name for name in imported))
        self.assertFalse(any(name == "folder_paths" for name in imported))
        self.assertTrue((ROOT / "comfyui_turing_utils" / "loading" / "convrot.py").is_file())

    def test_h3_services_are_separate_from_node_schemas(self):
        node_source = (
            ROOT / "comfyui_turing_utils" / "nodes" / "minimax_references.py"
        ).read_text(encoding="utf-8")
        service_source = (
            ROOT
            / "comfyui_turing_utils"
            / "adapters"
            / "minimax"
            / "references.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("torchaudio", node_source)
        self.assertNotIn("@dataclass", node_source)
        self.assertNotIn("comfy_api", service_source)
        self.assertTrue(
            (
                ROOT
                / "comfyui_turing_utils"
                / "adapters"
                / "minimax"
                / "video_vae_encode.py"
            ).is_file()
        )

    def test_builtin_model_adapters_are_registered_once(self):
        from comfyui_turing_utils import bootstrap_builtin_integrations
        from comfyui_turing_utils.adapters.registry import registered_model_adapters

        bootstrap_builtin_integrations()
        bootstrap_builtin_integrations()
        self.assertEqual(registered_model_adapters(), ("minimax_h3", "wan"))

    def test_package_root_uses_explicit_bootstrap(self):
        source = (ROOT / "comfyui_turing_utils" / "__init__.py").read_text(
            encoding="utf-8"
        )
        entrypoint = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("apply_minimax_adapter", source)
        self.assertIn("bootstrap_builtin_integrations()", entrypoint)

    def test_minimax_policy_and_memory_services_have_explicit_owners(self):
        minimax = ROOT / "comfyui_turing_utils" / "adapters" / "minimax"
        acceleration = (minimax / "acceleration.py").read_text(encoding="utf-8")
        planning = (minimax / "memory_planning.py").read_text(encoding="utf-8")
        state = (minimax / "memory_state.py").read_text(encoding="utf-8")
        policy = (minimax / "activation_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("class _MiniMaxMemoryShape", acceleration)
        self.assertIn("class _MiniMaxMemoryShape", planning)
        self.assertIn("class ActivationRuntimePlan", state)
        self.assertNotIn("torch.cuda.mem_get_info", policy)

    def test_quantization_and_kernel_attention_are_layered(self):
        quantization = ROOT / "comfyui_turing_utils" / "quantization"
        dispatch = (quantization / "dispatch.py").read_text(encoding="utf-8")
        self.assertIn("from .capabilities import", dispatch)
        self.assertIn("from .workspace import", dispatch)
        sage = ROOT / "kernel" / "comfyui_turing_utils_kernel" / "turing_sage"
        core = (sage / "core.py").read_text(encoding="utf-8")
        self.assertNotIn("@dataclass", core)
        self.assertTrue((sage / "records.py").is_file())
        self.assertTrue((sage / "sparse_policy.py").is_file())

    def test_root_contains_only_the_attention_compatibility_facade(self):
        retired_modules = (
            "attention_nodes.py",
            "bernini_nodes.py",
            "convrot_nodes.py",
            "minimax_adapter.py",
            "minimax_layout.py",
            "minimax_nodes.py",
            "nodes.py",
            "precision.py",
            "reference_nodes.py",
            "turing_fusions.py",
            "turing_ops.py",
            "wan_adapter.py",
            "wan_nodes.py",
        )
        self.assertFalse(any((ROOT / name).exists() for name in retired_modules))
        self.assertTrue((ROOT / "attention.py").is_file())

    def test_registered_node_ids_match_the_maintained_surface(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS

        self.assertEqual(
            tuple(NODE_CLASS_MAPPINGS),
            (
                "TuringUtilsConvRotDiffusionModelLoader",
                "TuringUtilsConvRotCLIPLoader",
                "TuringUtilsWanVideoFramesPadding",
                "TuringUtilsMiniMaxH3VideoFramesPadding",
                "TuringUtilsBerniniContextWindowsCore",
                "TuringUtilsBerniniInpaintCondition",
                "TuringUtilsKrea2IdentityEditConditioning",
                "TuringUtilsIsInputPresent",
                "TuringUtilsLazyIfElse",
                "TuringUtilsStageBarrier",
                "TuringUtilsStagePath",
                "TuringUtilsH3ConcatAVLatent",
                "TuringUtilsH3SeparateAVLatent",
                "TuringUtilsH3AddNoise",
                "TuringUtilsH3LatentInfo",
                "TuringUtilsH3KeyframeReference",
                "TuringUtilsH3ImageReference",
                "TuringUtilsH3VideoReference",
                "TuringUtilsH3AudioReference",
                "TuringUtilsH3SemanticReference",
                "TuringUtilsH3BuildConditioning",
                "TuringUtilsMiniMaxH3LatentUpscaleModelLoader",
                "TuringUtilsMiniMaxH3LatentUpscale",
                "TuringUtilsMiniMaxH3BlockCachePatch",
                "TuringUtilsSolAttentionStrategy",
                "TuringUtilsSlaAttentionStrategy",
                "TuringUtilsH3ImageSolAttention",
                "TuringUtilsH3StaticVirtualKV",
                "TuringUtilsResizeImageIfPresent",
                "TuringUtilsVideoMotionContactSheet",
                "TuringUtilsMultimodalChatOptions",
                "TuringUtilsMultimodalPromptChat",
                "TuringUtilsMiniMaxH3VideoVAEDecode",
                "TuringUtilsMiniMaxH3VideoVAEEncode",
            ),
        )


if __name__ == "__main__":
    unittest.main()
