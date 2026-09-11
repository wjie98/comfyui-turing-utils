from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

import attention as attention_backends  # noqa: E402
from comfyui_turing_utils.adapters.minimax import image_sol  # noqa: E402
from comfyui_turing_utils.nodes import attention as attention_nodes  # noqa: E402


class FakePatcher:
    def __init__(self):
        self.load_device = torch.device("cuda", 0)
        self.model_options = {"transformer_options": {"existing": True}}

    def clone(self):
        clone = FakePatcher()
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return clone


class SparseAttentionNodeTest(unittest.TestCase):
    def test_h3_virtual_kv_node_is_explicit_about_modes(self):
        node = attention_nodes.H3StaticVirtualKV
        self.assertEqual(node.TITLE, "Configure H3 Static Virtual KV")
        inputs = node.INPUT_TYPES()["required"]
        self.assertEqual(tuple(inputs), ("model", "mode"))
        self.assertEqual(
            inputs["mode"][0], ["conservative", "fast", "residual"]
        )
        model = object()
        patched = object()
        with mock.patch(
            "comfyui_turing_utils.nodes.attention.apply_h3_virtual_kv",
            return_value=patched,
        ) as apply_patch:
            self.assertEqual(node().patch(model, mode="fast"), (patched,))
        apply_patch.assert_called_once_with(model, mode="fast")

    def test_h3_image_sol_schema_is_h3_specific(self):
        node = attention_nodes.H3ImageSolAttentionPatch
        self.assertEqual(node.TITLE, "Configure H3 Image Sol Attention")
        inputs = node.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            (
                "model",
                "temporal_layout",
                "sparse_reference_image",
                "sparse_reference_video",
                "sparse_reference_audio",
                "dense_prefix_steps",
                "dense_suffix_steps",
                "dense_prefix_layers",
                "dense_suffix_layers",
            ),
        )
        self.assertEqual(
            inputs["temporal_layout"][0],
            ["dense_anchor_grid", "dense_window"],
        )
        self.assertFalse(inputs["sparse_reference_image"][1]["default"])
        self.assertTrue(inputs["sparse_reference_video"][1]["default"])
        self.assertFalse(inputs["sparse_reference_audio"][1]["default"])
        self.assertEqual(inputs["dense_prefix_steps"][1]["default"], 1)
        self.assertEqual(inputs["dense_suffix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_prefix_layers"][1]["default"], 2)
        self.assertEqual(inputs["dense_suffix_layers"][1]["default"], 0)
        self.assertEqual(
            tuple(node.INPUT_TYPES()["optional"]), ("debug_route_density",)
        )

    def test_h3_image_sol_node_forwards_controls(self):
        model = object()
        patched = object()
        with mock.patch(
            "comfyui_turing_utils.nodes.attention.apply_h3_image_sol_attention",
            return_value=patched,
        ) as apply_patch:
            output = attention_nodes.H3ImageSolAttentionPatch().patch(
                model,
                temporal_layout="dense_window",
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
                debug_route_density=True,
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            temporal_layout="dense_window",
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            debug_route_density=True,
        )

    def test_h3_image_sol_uses_fixed_sol_policy(self):
        model = FakePatcher()
        dense_override = object()
        runtime = SimpleNamespace(
            dense_backend="sdpa",
            dense_override=dense_override,
        )
        installed_model = FakePatcher()
        installed = SimpleNamespace(
            model=installed_model,
            layout=SimpleNamespace(
                model_kind="minimax_h3",
                installed=True,
                reason=None,
            ),
        )
        override = object()
        with mock.patch.object(
            image_sol, "is_minimax_h3_model", return_value=True
        ), mock.patch.object(
            image_sol, "attention_base_runtime", return_value=runtime
        ), mock.patch.object(
            image_sol, "make_sparse_attention_override", return_value=override
        ) as make_override, mock.patch.object(
            image_sol, "install_attention_strategy", return_value=installed
        ) as install:
            result = image_sol.apply_h3_image_sol_attention(
                model,
                temporal_layout="dense_window",
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
                debug_route_density=True,
            )

        self.assertIs(result, installed_model)
        make_override.assert_called_once_with(
            torch.device("cuda", 0),
            routing_threshold=1_000_000.0,
            prefix_policy="auto",
            manual_prefix_tokens=0,
            skipped_residual="1x64",
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            debug_route_density=True,
            use_w8a8=None,
            dense_backend="sdpa",
            dense_override=dense_override,
        )
        install.assert_called_once_with(
            model,
            override,
            strategy="H3 image Sol",
            backend="h3_image_sol",
            implementation="h3_image_sol:dense_window",
            runtime_config=runtime,
        )
        self.assertEqual(
            installed_model.model_options["transformer_options"][
                "turing_utils_h3_image_sol_temporal_layout"
            ],
            "dense_window",
        )

    def test_h3_image_sol_rejects_other_models_before_backend_setup(self):
        with mock.patch.object(
            image_sol, "is_minimax_h3_model", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "requires MiniMax H3"):
                image_sol.apply_h3_image_sol_attention(FakePatcher())

    def test_schema_exposes_tunable_sparse_parameters(self):
        self.assertEqual(
            attention_nodes.SolSparseAttentionPatch.TITLE,
            "Configure Sol Sparse Attention",
        )
        inputs = attention_nodes.SolSparseAttentionPatch.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            (
                "model",
                "routing_threshold",
                "prefix_policy",
                "manual_prefix_tokens",
                "skipped_residual",
                "sparse_reference_image",
                "sparse_reference_video",
                "sparse_reference_audio",
                "dense_prefix_steps",
                "dense_suffix_steps",
                "dense_prefix_layers",
                "dense_suffix_layers",
            ),
        )
        self.assertEqual(inputs["routing_threshold"][1]["default"], 1.0)
        self.assertEqual(inputs["prefix_policy"][0][0], "auto")
        self.assertEqual(inputs["manual_prefix_tokens"][1]["default"], 0)
        self.assertEqual(inputs["skipped_residual"][0][0], "1x64")
        self.assertFalse(inputs["sparse_reference_image"][1]["default"])
        self.assertTrue(inputs["sparse_reference_video"][1]["default"])
        self.assertFalse(inputs["sparse_reference_audio"][1]["default"])
        self.assertEqual(inputs["dense_prefix_steps"][0], "INT")
        self.assertEqual(inputs["dense_prefix_steps"][1]["default"], 1)
        self.assertEqual(inputs["dense_suffix_steps"][0], "INT")
        self.assertEqual(inputs["dense_suffix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_prefix_layers"][1]["default"], 2)
        self.assertEqual(inputs["dense_suffix_layers"][1]["default"], 0)
        optional = attention_nodes.SolSparseAttentionPatch.INPUT_TYPES()["optional"]
        self.assertEqual(tuple(optional), ("debug_route_density",))
        self.assertFalse(optional["debug_route_density"][1]["default"])

    def test_patch_clones_model_and_installs_generic_override(self):
        model = FakePatcher()
        override = object()
        with mock.patch(
            "attention.make_sparse_attention_override", return_value=override
        ) as make_override:
            patched = attention_backends.apply_sparse_attention_patch(
                model,
                min_sequence_tokens=8192,
                routing_threshold=0.85,
                prefix_policy="manual",
                manual_prefix_tokens=256,
                skipped_residual="1x64",
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
            )

        self.assertIsNot(patched, model)
        self.assertNotIn(
            "optimized_attention_override", model.model_options["transformer_options"]
        )
        options = patched.model_options["transformer_options"]
        runtime = options[attention_backends.ATTENTION_RUNTIME_CONFIG_KEY]
        self.assertEqual(runtime.strategy, "sol")
        self.assertEqual(runtime.dense_backend, "sdpa")
        self.assertIs(runtime.strategy_override, override)
        self.assertTrue(options["existing"])
        make_override.assert_called_once_with(
            torch.device("cuda", 0),
            min_sequence_tokens=8192,
            routing_threshold=0.85,
            prefix_policy="manual",
            manual_prefix_tokens=256,
            skipped_residual="1x64",
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            debug_route_density=False,
            use_w8a8=None,
            dense_backend="sdpa",
            dense_override=mock.ANY,
        )

    def test_sla_schema_matches_semantic_sparse_controls(self):
        node = attention_nodes.SlaSparseAttentionPatch
        self.assertEqual(node.TITLE, "Configure SLA Sparse Attention")
        inputs = node.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(inputs),
            (
                "model",
                "sparsity_ratio",
                "prefix_policy",
                "manual_prefix_tokens",
                "sparse_reference_image",
                "sparse_reference_video",
                "sparse_reference_audio",
                "dense_prefix_steps",
                "dense_suffix_steps",
                "dense_prefix_layers",
                "dense_suffix_layers",
            ),
        )
        self.assertEqual(inputs["sparsity_ratio"][1]["default"], 0.85)
        self.assertEqual(inputs["dense_prefix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_suffix_steps"][1]["default"], 0)
        self.assertEqual(inputs["dense_prefix_layers"][1]["default"], 0)
        self.assertEqual(inputs["dense_suffix_layers"][1]["default"], 0)
        self.assertEqual(inputs["prefix_policy"][0][0], "auto")
        self.assertFalse(inputs["sparse_reference_image"][1]["default"])
        self.assertTrue(inputs["sparse_reference_video"][1]["default"])
        self.assertFalse(inputs["sparse_reference_audio"][1]["default"])
        optional = node.INPUT_TYPES()["optional"]
        self.assertEqual(tuple(optional), ("debug_route_density",))

    def test_legacy_node_classes_are_removed(self):
        self.assertFalse(hasattr(attention_nodes, "LegacySolSparseAttentionPatch"))
        self.assertFalse(hasattr(attention_nodes, "LegacySlaSparseAttentionPatch"))

    def test_sla_node_returns_the_patched_model(self):
        model = object()
        patched = object()
        with mock.patch(
            "comfyui_turing_utils.nodes.attention.apply_sla_attention_patch",
            return_value=patched,
        ) as apply_patch:
            output = attention_nodes.SlaSparseAttentionPatch().patch(
                model,
                sparsity_ratio=0.8,
                prefix_policy="manual",
                manual_prefix_tokens=128,
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            sparsity_ratio=0.8,
            prefix_policy="manual",
            manual_prefix_tokens=128,
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            debug_route_density=False,
        )

    def test_node_returns_the_patched_model(self):
        model = object()
        patched = object()
        with mock.patch(
            "comfyui_turing_utils.nodes.attention.apply_sparse_attention_patch",
            return_value=patched,
        ) as apply_patch:
            output = attention_nodes.SolSparseAttentionPatch().patch(
                model,
                routing_threshold=0.85,
                prefix_policy="manual",
                manual_prefix_tokens=256,
                skipped_residual="1x64",
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=2,
                dense_suffix_steps=1,
                dense_prefix_layers=3,
                dense_suffix_layers=4,
            )
        self.assertEqual(output, (patched,))
        apply_patch.assert_called_once_with(
            model,
            routing_threshold=0.85,
            prefix_policy="manual",
            manual_prefix_tokens=256,
            skipped_residual="1x64",
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=True,
            dense_prefix_steps=2,
            dense_suffix_steps=1,
            dense_prefix_layers=3,
            dense_suffix_layers=4,
            debug_route_density=False,
        )


if __name__ == "__main__":
    unittest.main()
