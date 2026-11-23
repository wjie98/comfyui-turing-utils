from __future__ import annotations

import ast
from functools import cache
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

    def test_package_root_uses_explicit_bootstrap(self):
        source = (ROOT / "comfyui_turing_utils" / "__init__.py").read_text(
            encoding="utf-8"
        )
        entrypoint = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("apply_minimax_adapter", source)
        self.assertIn("bootstrap_builtin_integrations()", entrypoint)


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
