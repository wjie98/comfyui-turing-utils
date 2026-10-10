from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.attention import patches as attention_patches, runtime as attention_runtime  # noqa: E402
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
    def test_schema_exposes_tunable_sparse_parameters(self):
        inputs = attention_nodes.sol_inputs()["required"]
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
        optional = attention_nodes.sol_inputs()["optional"]
        self.assertEqual(tuple(optional), ("debug_route_density",))
        self.assertFalse(optional["debug_route_density"][1]["default"])

    def test_patch_clones_model_and_installs_generic_override(self):
        model = FakePatcher()
        override = object()
        with mock.patch(
            "comfyui_turing_utils.attention.patches.make_sparse_attention_override", return_value=override
        ) as make_override:
            patched = attention_patches.apply_sparse_attention_patch(
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
        runtime = options[attention_runtime.ATTENTION_RUNTIME_CONFIG_KEY]
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
            dense_backend="sdpa",
            dense_override=mock.ANY,
        )

    def test_sla_schema_matches_semantic_sparse_controls(self):
        inputs = attention_nodes.sla_inputs()["required"]
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
        optional = attention_nodes.sla_inputs()["optional"]
        self.assertEqual(tuple(optional), ("debug_route_density",))

    def test_legacy_node_classes_are_removed(self):
        self.assertFalse(hasattr(attention_nodes, "LegacySolSparseAttentionPatch"))
        self.assertFalse(hasattr(attention_nodes, "LegacySlaSparseAttentionPatch"))

    def test_sla_node_returns_the_patched_model(self):
        model = object()
        patched = object()
        apply_patch = mock.Mock(return_value=patched)
        with mock.patch.dict(attention_nodes._ATTENTION_STRATEGIES, {"sla": apply_patch}):
            output = attention_nodes.AttentionStrategy.execute(
                model, dict(strategy="sla",
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
            )).result
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
        apply_patch = mock.Mock(return_value=patched)
        with mock.patch.dict(attention_nodes._ATTENTION_STRATEGIES, {"sol": apply_patch}):
            output = attention_nodes.AttentionStrategy.execute(
                model, dict(strategy="sol",
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
            )).result
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
