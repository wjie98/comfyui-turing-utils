from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

from comfyui_turing_utils_kernel import turing_sage  # noqa: E402
from comfyui_turing_utils_kernel.turing_sage import core, quant  # noqa: E402
from comfyui_turing_utils_kernel.turing_sage.custom_ops import (  # noqa: E402
    qk_rms_rope_int8,
    qk_rms_rope_int8_mapped,
)


class TuringSageQuantContractTest(unittest.TestCase):
    def test_key_tile_auto_policy_uses_resources_not_product_names(self):
        core._KEY_TILE_CACHE.clear()
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 6)
        ):
            self.assertEqual(
                core._automatic_key_tile_tokens(
                    torch.device("cuda:0"),
                    key_length=60186,
                    head_dim=128,
                    use_w8a8=True,
                ),
                64,
            )
            self.assertEqual(
                core._automatic_key_tile_tokens(
                    torch.device("cuda:0"),
                    key_length=60186,
                    head_dim=64,
                    use_w8a8=True,
                ),
                128,
            )
        core._KEY_TILE_CACHE.clear()
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(7, 5)
        ):
            self.assertEqual(
                core._automatic_key_tile_tokens(
                    torch.device("cuda:0"),
                    key_length=60186,
                    head_dim=128,
                    use_w8a8=True,
                ),
                128,
            )
            self.assertEqual(
                core._automatic_key_tile_tokens(
                    torch.device("cuda:0"),
                    key_length=512,
                    head_dim=128,
                    use_w8a8=True,
                ),
                64,
            )

    def test_fused_qk_preprocessor_has_fake_tensor_contract(self):
        query = torch.empty((2, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        key = torch.empty((2, 2, 129, 128), dtype=torch.bfloat16, device="meta")
        query_norm = torch.empty((128,), dtype=torch.bfloat16, device="meta")
        key_norm = torch.empty((128,), dtype=torch.bfloat16, device="meta")
        freqs = torch.empty((1, 129, 1, 64, 2, 2), dtype=torch.bfloat16, device="meta")

        query_int8, query_scale, key_int8, key_scale = qk_rms_rope_int8(
            query,
            key,
            query_norm,
            key_norm,
            freqs,
            freqs,
            1e-6,
            128,
            "HND",
            "head",
            True,
            True,
            True,
        )

        self.assertEqual(query_int8.shape, query.shape)
        self.assertEqual(query_int8.dtype, torch.int8)
        self.assertEqual(query_scale.shape, (2, 4, 12))
        self.assertEqual(query_scale.dtype, torch.float32)
        self.assertEqual(key_int8.shape, key.shape)
        self.assertEqual(key_int8.dtype, torch.int8)
        self.assertEqual(key_scale.shape, (2, 2, 3))
        self.assertEqual(key_scale.dtype, torch.float32)

    def test_fused_qk_preprocessor_is_a_fullgraph_leaf(self):
        query = torch.empty((1, 4, 129, 64), dtype=torch.float16, device="meta")
        key = torch.empty((1, 2, 129, 64), dtype=torch.float16, device="meta")
        query_norm = torch.empty((64,), dtype=torch.float16, device="meta")
        key_norm = torch.empty((64,), dtype=torch.float16, device="meta")
        freqs = torch.empty((0,), dtype=torch.float16, device="meta")

        compiled = torch.compile(
            lambda q, k, qw, kw, f: qk_rms_rope_int8(
                q,
                k,
                qw,
                kw,
                f,
                f,
                1e-6,
                0,
                "HND",
                "head",
                False,
                False,
                False,
            ),
            backend="eager",
            fullgraph=True,
        )
        query_int8, query_scale, key_int8, key_scale = compiled(
            query, key, query_norm, key_norm, freqs
        )
        self.assertEqual(query_int8.shape, query.shape)
        self.assertEqual(query_scale.shape, (1, 4, 12))
        self.assertEqual(key_int8.shape, key.shape)
        self.assertEqual(key_scale.shape, (1, 2, 3))

    def test_mapped_qk_fake_contract_uses_logical_key_length(self):
        query = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        key = torch.empty((1, 2, 129, 128), dtype=torch.bfloat16, device="meta")
        query_norm = torch.empty((128,), dtype=torch.bfloat16, device="meta")
        key_norm = torch.empty((128,), dtype=torch.bfloat16, device="meta")
        query_freqs = torch.empty(
            (1, 129, 1, 64, 2, 2), dtype=torch.bfloat16, device="meta"
        )
        key_freqs = torch.empty(
            (1, 321, 1, 64, 2, 2), dtype=torch.bfloat16, device="meta"
        )
        source_indices = torch.empty((321,), dtype=torch.int32, device="meta")

        q_int8, q_scale, k_int8, k_scale = qk_rms_rope_int8_mapped(
            query,
            key,
            query_norm,
            key_norm,
            query_freqs,
            key_freqs,
            source_indices,
            1e-6,
            128,
            "HND",
            "head",
            True,
            True,
            True,
        )

        self.assertEqual(q_int8.shape, query.shape)
        self.assertEqual(q_scale.shape, (1, 4, 12))
        self.assertEqual(k_int8.shape, (1, 2, 321, 128))
        self.assertEqual(k_scale.shape, (1, 2, 6))

    def test_compiled_attention_facades_have_fake_tensor_contracts(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        v = torch.empty_like(k)

        stable = turing_sage.sageattn_compiled(q, k, v)
        w8a8 = turing_sage.w8a8attn_compiled(q, k, v)
        sla = turing_sage.sla_sparse_sageattn_compiled(
            q,
            k,
            v,
            dense_query_ranges=((0, 64),),
            exact_kv_ranges=((128, 151),),
        )

        for output in (stable, w8a8, sla):
            self.assertEqual(output.shape, q.shape)
            self.assertEqual(output.dtype, q.dtype)
            self.assertEqual(output.device.type, "meta")

    def test_compiled_attention_facade_is_a_fullgraph_leaf(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        v = torch.empty_like(k)

        compiled = torch.compile(
            lambda query, key, value: turing_sage.sageattn_compiled(query, key, value),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(q, k, v)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.device.type, "meta")

    def test_compiled_sla_facade_is_a_fullgraph_leaf(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        v = torch.empty_like(k)

        compiled = torch.compile(
            lambda query, key, value: turing_sage.sla_sparse_sageattn_compiled(
                query,
                key,
                value,
                dense_query_ranges=((0, 64),),
                exact_kv_ranges=((128, 151),),
            ),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(q, k, v)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.device.type, "meta")

    def test_split_quantizers_preserve_production_scale_contract(self):
        q = torch.empty((2, 4, 129, 128), dtype=torch.bfloat16)
        k = torch.empty((2, 2, 151, 128), dtype=torch.bfloat16)
        with (
            mock.patch.object(quant._fused, "quant_per_warp_int8_cuda") as quant_q,
            mock.patch.object(quant._fused, "quant_per_block_int8_cuda") as quant_k,
        ):
            q_int8, q_scale = quant.quantize_query_per_warp(q)
            k_int8, k_scale = quant.quantize_key_per_block(k)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 12))
        self.assertEqual(k_scale.shape, (2, 2, 3))
        quant_q.assert_called_once_with(q, q_int8, q_scale, 64, 16, 1)
        quant_k.assert_called_once_with(k, k_int8, k_scale, 64, 1)

    def test_production_per_warp_scale_shapes(self):
        q = torch.empty((2, 4, 65, 64), dtype=torch.float16)
        k = torch.empty((2, 2, 73, 64), dtype=torch.float16)
        with (
            mock.patch.object(quant._fused, "quant_per_warp_int8_cuda") as quant_q,
            mock.patch.object(quant._fused, "quant_per_block_int8_cuda") as quant_k,
        ):
            q_int8, q_scale, k_int8, k_scale = quant.per_warp_int8(q, k)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 8))
        self.assertEqual(k_scale.shape, (2, 2, 2))
        quant_q.assert_called_once()
        quant_k.assert_called_once()

    def test_hadamard_quantizer_fuses_qk_and_preserves_scale_contract(self):
        q = torch.empty((2, 4, 129, 128), dtype=torch.bfloat16)
        k = torch.empty((2, 2, 151, 128), dtype=torch.bfloat16)
        with mock.patch.object(
            quant._fused, "quant_qk_per_warp_int8_rotated_anchored_cuda"
        ) as fused:
            q_int8, q_scale, k_int8, k_scale = quant.per_warp_int8_hadamard(q, k)

        self.assertEqual(q_int8.shape, q.shape)
        self.assertEqual(k_int8.shape, k.shape)
        self.assertEqual(q_scale.shape, (2, 4, 12))
        self.assertEqual(k_scale.shape, (2, 2, 3))
        args = fused.call_args.args
        for actual, expected in zip(args[:6], (q, k, q_int8, k_int8, q_scale, k_scale)):
            self.assertIs(actual, expected)
        self.assertEqual(args[6].shape, (2, 2))
        self.assertEqual(args[6].dtype, torch.int32)
        self.assertEqual(args[7:], (64, 16, 64, 1))

    def test_hadamard_quantizer_can_disable_k_stabilization(self):
        q = torch.empty((1, 2, 65, 64), dtype=torch.float16)
        k = torch.empty((1, 1, 73, 64), dtype=torch.float16)
        with mock.patch.object(
            quant._fused, "quant_qk_per_warp_int8_rotated_cuda"
        ) as fused:
            quant.per_warp_int8_hadamard(q, k, stabilize_k=False)
        fused.assert_called_once()

    def test_production_quant_module_has_no_experimental_api(self):
        self.assertFalse(hasattr(quant, "per_thread_int4"))
        self.assertFalse(hasattr(quant, "per_thread_int4_fused"))
        self.assertFalse(hasattr(quant, "sage2_score_correction"))
        self.assertFalse(hasattr(quant, "per_block_int8"))

    def test_public_sage_api_exposes_only_stable_and_current_sparse_entrypoints(self):
        self.assertEqual(
            turing_sage.__all__,
            [
                "available",
                "fused_qk_preprocessing_available",
                "overlap_accumulate_available",
                "overlap_accumulate_compiled",
                "overlap_blend_available",
                "overlap_blend_compiled",
                "precompute_rms_rope_k_anchor",
                "prequantize_rms_rope_qk",
                "prequantize_sageattn",
                "prequantize_sla_sageattn",
                "prequantize_sla_sageattn_from_qk",
                "prequantize_sol_sageattn",
                "preflight",
                "preflight_sparse",
                "preflight_sla",
                "preflight_w8a8",
                "run_attention_correctness_gate",
                "sageattn",
                "sageattn_compiled",
                "sageattn_from_prequantized",
                "sageattn_varlen",
                "sageattn_varlen_compiled",
                "sla_available",
                "sla_sparse_sageattn",
                "sla_sparse_sageattn_compiled",
                "sla_sparse_sageattn_from_prequantized",
                "sol_sparse_sageattn",
                "sol_sparse_sageattn_compiled",
                "sol_sparse_sageattn_from_prequantized",
                "split_prequantization_available",
                "sparse_available",
                "w8a8attn",
                "w8a8attn_compiled",
                "w8a8attn_varlen",
                "w8a8attn_varlen_compiled",
                "w8a8_available",
                "w8a8_varlen_available",
            ],
        )
        self.assertFalse(hasattr(turing_sage, "sageattn_sage1"))
        self.assertFalse(hasattr(turing_sage, "sageattn_sage2"))
        self.assertFalse(hasattr(turing_sage, "sage_"))


if __name__ == "__main__":
    unittest.main()
