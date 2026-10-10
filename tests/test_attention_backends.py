from __future__ import annotations

import sys
import gc
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.attention import (
    patches as attention_patches,
    protocol as attention_protocol,
    stable as attention_stable,
)  # noqa: E402
from comfyui_turing_utils.attention import (
    dense as attention_dense,
    execution as attention_execution,
)  # noqa: E402
from comfy.ldm.modules import attention as comfy_attention  # noqa: E402


class FakeModel:
    def __init__(self):
        self.model_options = {}


class AttentionBackendsTest(unittest.TestCase):
    def test_external_prepared_executor_consumes_projected_qkv_once(self):
        tensors = [torch.zeros((1, 2, 64, 128), dtype=torch.bfloat16) for _ in range(3)]
        q, k, v = (
            comfy_attention.AttentionTensorContainer(tensor) for tensor in tensors
        )
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RotaryEmbeddingSpec(None, 0, "none"),
        )
        output = torch.zeros((1, 64, 256), dtype=torch.bfloat16)
        dense = mock.Mock(return_value=output)
        executor = attention_dense._make_external_prepared_executor(dense, "sage")
        request = attention_protocol.PreparedAttention.from_hnd(
            q, k, v, heads=2, qk_transform=spec, transformer_options={}
        )

        with mock.patch(
            "comfyui_turing_utils.attention.dense._prepared_qk_transform",
            return_value=(tensors[0], tensors[1]),
        ) as transform:
            outcome = executor(request)

        self.assertIs(outcome.output, output)
        transform.assert_called_once_with(tensors[0], tensors[1], spec)
        dense.assert_called_once()
        self.assertIs(dense.call_args.args[1], tensors[0])
        self.assertIs(dense.call_args.args[2], tensors[1])
        self.assertIs(dense.call_args.args[3], tensors[2])
        self.assertTrue(dense.call_args.kwargs["skip_reshape"])
        self.assertIsNone(q.tensor)
        self.assertIsNone(k.tensor)
        self.assertIsNone(v.tensor)

    def test_external_prepared_h3_transform_matches_comfy_fused_semantics(self):
        import comfy.quant_ops
        from comfy.ldm.minimax.model import rope_rotation_table

        query = torch.randn((1, 2, 8, 128), dtype=torch.bfloat16)
        key = torch.randn_like(query)
        query_weight = torch.randn(128, dtype=torch.bfloat16)
        key_weight = torch.randn(128, dtype=torch.bfloat16)
        freqs = rope_rotation_table(
            torch.randn((8, 64), dtype=torch.float32), torch.bfloat16
        )
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(query_weight, 1e-6, "head"),
            attention_protocol.RMSNormSpec(key_weight, 1e-6, "head"),
            attention_protocol.RotaryEmbeddingSpec(freqs, 64, "split_half"),
        )

        actual_query, actual_key = attention_execution._prepared_qk_transform(
            query.clone(), key.clone(), spec
        )
        expected_query, expected_key = comfy.quant_ops.ck.rms_rope_split_half(
            query.transpose(1, 2),
            key.transpose(1, 2),
            freqs,
            query_weight,
            key_weight,
            epsilon=1e-6,
            rot_dim=64,
        )

        torch.testing.assert_close(actual_query, expected_query.transpose(1, 2))
        torch.testing.assert_close(actual_key, expected_key.transpose(1, 2))

    def test_external_prepared_wan_row_transform_matches_comfy_semantics(self):
        import comfy.rmsnorm
        from comfy.ldm.flux.math import apply_rope1, rope

        batch, heads, tokens, head_dim = 1, 2, 8, 64
        query = torch.randn((batch, heads, tokens, head_dim))
        key = torch.randn_like(query)
        query_weight = torch.randn(heads * head_dim)
        key_weight = torch.randn(heads * head_dim)
        freqs = rope(torch.randn((batch, tokens)), head_dim, 10_000).unsqueeze(2)
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(query_weight, 1e-6, "row"),
            attention_protocol.RMSNormSpec(key_weight, 1e-6, "row"),
            attention_protocol.RotaryEmbeddingSpec(freqs, head_dim, "interleaved"),
        )

        actual_query, actual_key = attention_execution._prepared_qk_transform(
            query.clone(), key.clone(), spec
        )

        def expected(value, weight):
            value = value.transpose(1, 2)
            value = comfy.rmsnorm.rms_norm(
                value.reshape(batch, tokens, heads * head_dim),
                weight,
                1e-6,
            ).reshape(batch, tokens, heads, head_dim)
            return apply_rope1(value, freqs).transpose(1, 2)

        torch.testing.assert_close(actual_query, expected(query, query_weight))
        torch.testing.assert_close(actual_key, expected(key, key_weight))

    def test_fused_qk_preprocessing_requires_022_kernel_abi(self):
        sage_module = SimpleNamespace(fused_qk_preprocessing_available=lambda: True)
        for version, expected in (("0.21.0", False), ("0.22.0", True)):
            with (
                self.subTest(version=version),
                mock.patch.dict(
                    sys.modules,
                    {
                        "comfyui_turing_utils_kernel": SimpleNamespace(
                            __version__=version
                        ),
                        "comfyui_turing_utils_kernel.turing_sage": sage_module,
                    },
                ),
            ):
                self.assertEqual(
                    attention_stable.fused_qk_preprocessing_available(), expected
                )

    def test_adapter_qk_preprocessor_consumes_inputs_before_attention(self):
        tensors = [torch.zeros((1, 2, 64, 128), dtype=torch.bfloat16) for _ in range(3)]
        references = [weakref.ref(tensor) for tensor in tensors]
        q, k, v = (
            comfy_attention.AttentionTensorContainer(tensor) for tensor in tensors
        )
        del tensors
        call = SimpleNamespace(
            heads=2,
            kv_heads=2,
            head_dim=128,
            query_tokens=64,
            key_tokens=64,
            tensor_layout="HND",
            skip_output_reshape=False,
        )
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RotaryEmbeddingSpec(None, 0, "none"),
        )

        qk_calls = []

        def quantize(query, key, received_spec, **kwargs):
            qk_calls.append((received_spec, kwargs))
            return "qk"

        def finish(qk, value, inspected, **kwargs):
            self.assertEqual(qk, "qk")
            self.assertIs(inspected, call)
            return "packed"

        def execute(packed, *, kernel):
            gc.collect()
            self.assertEqual((packed, kernel), ("packed", "sage"))
            self.assertTrue(all(reference() is None for reference in references))
            return "output"

        def inspect(*args, **kwargs):
            return call, None

        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.inspect_turing_attention_call",
                new=inspect,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.prequantize_turing_qk",
                new=quantize,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.prequantize_turing_attention_from_qk",
                new=finish,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_attention_from_prequantized",
                new=execute,
            ),
        ):
            processor = attention_dense._make_dense_prepared_executor("sage")
            request = attention_protocol.PreparedAttention.from_hnd(
                q, k, v, heads=2, qk_transform=spec, transformer_options={}
            )
            outcome = processor(request)

        self.assertEqual(outcome.output, "output")
        self.assertEqual(len(qk_calls), 1)
        self.assertIsNone(q.tensor)
        self.assertIsNone(k.tensor)
        self.assertIsNone(v.tensor)

    def test_sol_adapter_preprocessor_reuses_fused_qk_and_releases_inputs(self):
        tensors = [
            torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16) for _ in range(3)
        ]
        references = [weakref.ref(tensor) for tensor in tensors]
        q, k, v = (
            comfy_attention.AttentionTensorContainer(tensor) for tensor in tensors
        )
        del tensors
        attention_call = SimpleNamespace(
            heads=2,
            kv_heads=2,
            head_dim=128,
            input_dtype=torch.bfloat16,
            query_tokens=4096,
            key_tokens=4096,
            tensor_layout="HND",
            skip_output_reshape=False,
        )
        sol_call = SimpleNamespace(
            attention=attention_call,
            dense_query_ranges=(),
            exact_kv_ranges=(),
            residual_subblocks=1,
        )
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RotaryEmbeddingSpec(None, 0, "none"),
        )

        def finish(qk, value, inspected, **kwargs):
            self.assertEqual(qk, "qk")
            self.assertIs(inspected, sol_call)
            return "packed-sol"

        def inspect(*args, **kwargs):
            return sol_call, None

        def quantize(*args, **kwargs):
            return "qk"

        def execute(packed, *, return_stats):
            gc.collect()
            self.assertEqual((packed, return_stats), ("packed-sol", False))
            self.assertTrue(all(reference() is None for reference in references))
            return "sol-output"

        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled"),
            mock.patch("comfyui_turing_utils.attention.sol.preflight_bundled_sparse"),
            mock.patch(
                "comfyui_turing_utils.attention.sol.fused_qk_preprocessing_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.inspect_sol_attention_call",
                new=inspect,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.prequantize_turing_qk", new=quantize
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.prequantize_turing_sol_attention_from_qk",
                new=finish,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.turing_sol_attention_from_prequantized",
                new=execute,
            ),
        ):
            override = attention_patches.make_sparse_attention_override(
                torch.device("cuda", 0),
                dense_prefix_layers=0,
                dense_backend="sage",
            )
            request = attention_protocol.PreparedAttention.from_hnd(
                q, k, v, heads=2, qk_transform=spec, transformer_options={}
            )
            outcome = override.prepared_attention_executor(request)

        self.assertEqual(outcome.output, "sol-output")
        self.assertIsNone(q.tensor)
        self.assertIsNone(k.tensor)
        self.assertIsNone(v.tensor)

    def test_split_prequantization_requires_020_kernel_abi(self):
        sage_module = SimpleNamespace(split_prequantization_available=lambda: True)
        for version, expected in (("0.19.0", False), ("0.20.0", True)):
            with (
                self.subTest(version=version),
                mock.patch.dict(
                    sys.modules,
                    {
                        "comfyui_turing_utils_kernel": SimpleNamespace(
                            __version__=version
                        ),
                        "comfyui_turing_utils_kernel.turing_sage": sage_module,
                    },
                ),
            ):
                self.assertEqual(
                    attention_stable.split_prequantization_available(), expected
                )

    def test_sla_adapter_preprocessor_reuses_fused_qk_and_releases_inputs(self):
        tensors = [
            torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16) for _ in range(3)
        ]
        references = [weakref.ref(tensor) for tensor in tensors]
        q, k, v = (
            comfy_attention.AttentionTensorContainer(tensor) for tensor in tensors
        )
        del tensors
        attention_call = SimpleNamespace(
            heads=2,
            kv_heads=2,
            head_dim=128,
            input_dtype=torch.bfloat16,
            query_tokens=4096,
            key_tokens=4096,
            tensor_layout="HND",
            skip_output_reshape=False,
        )
        sla_call = SimpleNamespace(
            attention=attention_call,
            dense_query_ranges=(),
            exact_kv_ranges=(),
        )
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RotaryEmbeddingSpec(None, 0, "none"),
        )

        def finish(qk, value, inspected, **kwargs):
            self.assertEqual(qk, "qk")
            self.assertIs(inspected, sla_call)
            return "packed-sla"

        def inspect(*args, **kwargs):
            return sla_call, None

        def quantize(*args, **kwargs):
            return "qk"

        def execute(packed, *, return_stats):
            gc.collect()
            self.assertEqual((packed, return_stats), ("packed-sla", False))
            self.assertTrue(all(reference() is None for reference in references))
            return "sla-output"

        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.bundled_sla_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled"),
            mock.patch("comfyui_turing_utils.attention.sla.preflight_bundled_sla"),
            mock.patch(
                "comfyui_turing_utils.attention.sla.fused_qk_preprocessing_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.inspect_sla_attention_call",
                new=inspect,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.prequantize_turing_qk", new=quantize
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.prequantize_turing_sla_attention_from_qk",
                new=finish,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.turing_sla_attention_from_prequantized",
                new=execute,
            ),
        ):
            override = attention_patches.make_sla_attention_override(
                torch.device("cuda", 0),
                dense_prefix_steps=0,
                dense_prefix_layers=0,
                dense_backend="sage",
            )
            request = attention_protocol.PreparedAttention.from_hnd(
                q, k, v, heads=2, qk_transform=spec, transformer_options={}
            )
            outcome = override.prepared_attention_executor(request)

        self.assertEqual(outcome.output, "sla-output")
        self.assertIsNone(q.tensor)
        self.assertIsNone(k.tensor)
        self.assertIsNone(v.tensor)

    def test_container_path_releases_inputs_before_output_allocation(self):
        references = []

        def containers():
            tensors = [
                torch.zeros((1, 2, 64, 128), dtype=torch.bfloat16) for _ in range(3)
            ]
            references.extend(weakref.ref(tensor) for tensor in tensors)
            return tuple(
                comfy_attention.AttentionTensorContainer(tensor) for tensor in tensors
            )

        def prequantize(*args, **kwargs):
            return "packed"

        def consume(quantized, *, kernel):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            self.assertEqual(kernel, "w8a8")
            return torch.zeros((1, 2, 64, 128), dtype=torch.bfloat16)

        q, k, v = containers()
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.prequantize_turing_attention",
                new=prequantize,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_attention_from_prequantized",
                side_effect=consume,
            ),
        ):
            function = attention_dense._make_dense_container_function("w8a8")
            output = function(
                q,
                k,
                v,
                2,
                skip_reshape=True,
                skip_output_reshape=True,
            )

        self.assertEqual(output.shape, (1, 2, 64, 128))
        self.assertIsNone(q.tensor)
        self.assertIsNone(k.tensor)
        self.assertIsNone(v.tensor)

    def test_container_preflight_falls_back_without_partial_consumption(self):
        tensors = [torch.zeros((1, 2, 64, 128), dtype=torch.bfloat16) for _ in range(3)]
        q, k, v = (
            comfy_attention.AttentionTensorContainer(tensor) for tensor in tensors
        )
        fallback = mock.Mock(return_value="fallback")
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense._default_attention_fallback",
                return_value=fallback,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.inspect_turing_attention_call",
                return_value=(None, "unsupported"),
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.prequantize_turing_attention"
            ) as prequant,
        ):
            function = attention_dense._make_dense_container_function("sage")
            output = function(q, k, v, 2, skip_reshape=True)

        self.assertEqual(output, "fallback")
        prequant.assert_not_called()
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.args[:3], tuple(tensors))
        self.assertIsNone(q.tensor)
        self.assertIsNone(k.tensor)
        self.assertIsNone(v.tensor)

    def test_sparse_backend_requires_the_fused_routing_abi(self):
        sage_module = SimpleNamespace(sparse_available=lambda: True)
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.9.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.11.1"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.12.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.15.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.16.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.22.3"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_sparse_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.23.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertTrue(attention_stable.bundled_sparse_available())

    def test_w8a8_backend_requires_023_kernel_abi(self):
        sage_module = SimpleNamespace(w8a8_available=lambda: True)
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.22.3"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertFalse(attention_stable.bundled_w8a8_available())
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.23.0"),
                "comfyui_turing_utils_kernel.turing_sage": sage_module,
            },
        ):
            self.assertTrue(attention_stable.bundled_w8a8_available())

    def test_backend_choices_are_stable(self):
        self.assertEqual(
            attention_stable.attention_backend_choices(),
            ("w8a8", "sage", "sdpa"),
        )

    def test_aliases_normalize_to_node_options(self):
        self.assertEqual(attention_stable.normalize_attention_backend(None), "w8a8")
        self.assertEqual(
            attention_stable.normalize_attention_backend("torch-sdpa"), "sdpa"
        )
        self.assertEqual(
            attention_stable.normalize_attention_backend("sage attention"), "sage"
        )
        self.assertEqual(
            attention_stable.normalize_attention_backend("sage_attn"), "sage"
        )
        self.assertEqual(attention_stable.normalize_attention_backend("sage_"), "sage")
        self.assertEqual(
            attention_stable.normalize_attention_backend("turing-sage"), "sage"
        )
        with self.assertRaises(ValueError):
            attention_stable.normalize_attention_backend("flash-attn")
        with self.assertRaises(ValueError):
            attention_stable.normalize_attention_backend("auto")
        with self.assertRaises(ValueError):
            attention_stable.normalize_attention_backend("default")
        with self.assertRaises(ValueError):
            attention_stable.normalize_attention_backend("sage1")
        with self.assertRaises(ValueError):
            attention_stable.normalize_attention_backend("sol-sparse")

    def test_backend_registration_rejects_alias_collisions_without_partial_registration(
        self,
    ):
        backend = attention_stable.AttentionBackend(
            option="test_collision",
            attention_function="unused",
            label="sage",
        )
        with self.assertRaisesRegex(ValueError, "alias collision"):
            attention_stable.register_attention_backend(backend)
        self.assertNotIn("test_collision", attention_stable.attention_backend_choices())

    def test_external_sage_sends_fp32_qkv_to_pytorch_attention(self):
        sage = mock.Mock()
        pytorch = mock.Mock(return_value="pytorch")
        original = mock.Mock(return_value="original")
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: (
                sage if name == "sage" else pytorch if name == "pytorch" else default
            ),
        ):
            override = attention_patches.make_attention_override("sage")
            q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
            self.assertEqual(
                override(original, q, q, q, 2, skip_reshape=True), "pytorch"
            )
        pytorch.assert_called_once()
        original.assert_not_called()
        sage.assert_not_called()

    def test_external_sage_sends_mixed_qkv_to_original_attention(self):
        sage = mock.Mock()
        original = mock.Mock(return_value="original")
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sage if name == "sage" else default,
        ):
            override = attention_patches.make_attention_override("sage")

        bf16 = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        fp32 = bf16.float()
        self.assertEqual(
            override(original, bf16, fp32, bf16, 2, skip_reshape=True), "original"
        )
        original.assert_called_once()
        sage.assert_not_called()

    def test_sage_does_not_recast_supported_qkv_dtype(self):
        captured = {}

        def sage(q, k, v, *args, **kwargs):
            captured["q"] = q
            return q

        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sage if name == "sage" else default,
        ):
            override = attention_patches.make_attention_override("sage")

        q = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(captured["q"], q)
        self.assertIs(out, q)

    def test_external_w8a8_recoverable_rejection_uses_original_backend(self):
        kitchen = mock.Mock(
            side_effect=RuntimeError("Q/K strides must preserve 4-element alignment")
        )
        kitchen.container_function = mock.Mock()
        original = mock.Mock(return_value="fallback")
        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: (
                kitchen if name == "comfy_kitchen_int8" else default
            ),
        ):
            override = attention_patches.make_attention_override("w8a8")

        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        self.assertEqual(
            override(original, q, q, q, 2, skip_reshape=True),
            "fallback",
        )
        original.assert_called_once()

    def test_external_w8a8_container_rejection_preserves_fallback_inputs(self):
        kitchen = mock.Mock(side_effect=RuntimeError("kernel is not supported"))
        kitchen.container_function = mock.Mock()
        fallback = mock.Mock(return_value="fallback")
        with (
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: (
                    kitchen if name == "comfy_kitchen_int8" else default
                ),
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense._default_attention_fallback",
                return_value=fallback,
            ),
        ):
            override = attention_patches.make_attention_override("w8a8")

        containers = tuple(
            comfy_attention.AttentionTensorContainer(
                torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
            )
            for _ in range(3)
        )
        self.assertEqual(
            override.container_function(*containers, 2, skip_reshape=True),
            "fallback",
        )
        fallback.assert_called_once()
        self.assertTrue(all(container.tensor is None for container in containers))

    def test_sage_fp32_fallback_runs_through_comfy_attention_wrapper(self):
        model = FakeModel()
        with (
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: (
                    comfy_attention.attention_sage
                    if name == "sage"
                    else comfy_attention.attention_pytorch
                    if name == "pytorch"
                    else default
                ),
            ),
            mock.patch("comfy.ldm.modules.attention.sageattn") as sage,
        ):
            attention_patches.apply_attention_backend(model, "sage")
            transformer_options = model.model_options["transformer_options"]
            q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
            output = comfy_attention.optimized_attention(
                q,
                q,
                q,
                heads=2,
                skip_reshape=True,
                transformer_options=transformer_options,
            )
        self.assertEqual(output.shape, (1, 4, 16))
        sage.assert_not_called()

    def test_sdpa_keeps_fp32_qkv_unchanged(self):
        captured = {}

        def sdpa(q, k, v, *args, **kwargs):
            captured["q"] = q
            return q

        with mock.patch(
            "comfy.ldm.modules.attention.get_attention_function",
            side_effect=lambda name, default: sdpa if name == "pytorch" else default,
        ):
            override = attention_patches.make_attention_override("sdpa")

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(captured["q"], q)
        self.assertIs(out, q)

    def test_turing_sdpa_converts_bf16_qkv_to_fp16_and_restores_output(self):
        captured = {}

        def sdpa(q, k, v, *args, **kwargs):
            captured["dtypes"] = (q.dtype, k.dtype, v.dtype)
            captured["mask_dtype"] = kwargs["mask"].dtype
            return q

        with (
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: sdpa
                if name == "pytorch"
                else default,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
        ):
            override = attention_patches.make_attention_override(
                "sdpa", device=torch.device("cuda", 0)
            )
            q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
            mask = torch.zeros(4, 4, dtype=torch.float32)
            out = override(
                lambda *args, **kwargs: None,
                q,
                q,
                q,
                2,
                mask=mask,
                skip_reshape=True,
            )

        self.assertEqual(captured["dtypes"], (torch.float16,) * 3)
        self.assertEqual(captured["mask_dtype"], torch.float16)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_turing_sdpa_container_consumes_inputs_before_fp16_call(self):
        captured = {}

        def sdpa(q, k, v, *args, **kwargs):
            captured["dtypes"] = (q.dtype, k.dtype, v.dtype)
            return q

        with (
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: sdpa
                if name == "pytorch"
                else default,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
        ):
            override = attention_patches.make_attention_override(
                "sdpa", device=torch.device("cuda", 0)
            )
            containers = tuple(
                comfy_attention.AttentionTensorContainer(
                    torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
                )
                for _ in range(3)
            )
            out = override.container_function(*containers, 2, skip_reshape=True)

        self.assertEqual(captured["dtypes"], (torch.float16,) * 3)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(all(container.tensor is None for container in containers))

    def test_sdpa_backend_overrides_optimized_attention(self):
        model = FakeModel()
        attention_patches.apply_attention_backend(model, "sdpa")
        transformer_options = model.model_options["transformer_options"]
        self.assertEqual(transformer_options["turing_utils_attention_backend"], "sdpa")
        self.assertIn("optimized_attention_override", transformer_options)

        q = torch.randn(1, 8, 16)
        k = torch.randn(1, 8, 16)
        v = torch.randn(1, 8, 16)
        out = comfy_attention.optimized_attention(
            q, k, v, heads=2, transformer_options=transformer_options
        )
        self.assertEqual(tuple(out.shape), (1, 8, 16))

    def test_turing_explicit_sage_uses_stable_bundled_baseline(self):
        model = FakeModel()
        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled"
            ) as preflight,
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_sage_attention",
                return_value=q,
            ) as kernel,
        ):
            attention_patches.apply_attention_backend(
                model, "sage", device=torch.device("cuda", 0)
            )
            override = model.model_options["transformer_options"][
                "optimized_attention_override"
            ]
            out = override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        self.assertIs(out, q)
        kernel.assert_called_once()
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(
            model.model_options["transformer_options"][
                "turing_utils_attention_backend"
            ],
            "sage",
        )
        self.assertEqual(
            model.model_options["transformer_options"][
                "turing_utils_attention_implementation"
            ],
            "bundled_turing_sage",
        )

    def test_non_turing_sage_exposes_external_prepared_executor(self):
        target = mock.Mock(return_value="sage")
        backend = attention_stable.AttentionBackend(
            option="sage",
            attention_function="sage",
            label="sage",
        )
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense._select_attention_backend",
                return_value=(backend, target),
            ),
        ):
            override = attention_patches.make_attention_override(
                "sage", device=torch.device("cuda", 0)
            )

        self.assertEqual(
            override.turing_utils_attention_implementation,
            "comfy:sage",
        )
        self.assertTrue(
            callable(getattr(override, "prepared_attention_executor", None))
        )
        self.assertEqual(
            override.prepared_attention_executor.turing_utils_attention_backend,
            "sage",
        )

    def test_turing_explicit_sage_selects_bundled_backend(self):
        model = FakeModel()
        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled"
            ) as preflight,
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_sage_attention",
                return_value=q,
            ) as kernel,
        ):
            attention_patches.apply_attention_backend(
                model, "sage", device=torch.device("cuda", 0)
            )
            override = model.model_options["transformer_options"][
                "optimized_attention_override"
            ]
            override(lambda *args, **kwargs: None, q, q, q, 2, skip_reshape=True)

        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertNotIn("variant", kernel.call_args.kwargs)

    def test_turing_explicit_w8a8_selects_production_bundled_backend(self):
        model = FakeModel()
        q = torch.randn(1, 2, 4, 128, dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"
            ) as preflight,
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_w8a8_attention",
                return_value=q,
            ) as kernel,
        ):
            attention_patches.apply_attention_backend(
                model, "w8a8", device=torch.device("cuda", 0)
            )
            transformer_options = model.model_options["transformer_options"]
            override = transformer_options["optimized_attention_override"]
            output = override(
                lambda *args, **kwargs: None,
                q,
                q,
                q,
                2,
                skip_reshape=True,
            )

        self.assertIs(output, q)
        kernel.assert_called_once()
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(transformer_options["turing_utils_attention_backend"], "w8a8")
        self.assertEqual(
            transformer_options["turing_utils_attention_implementation"],
            "bundled_turing_w8a8",
        )

    def test_explicit_w8a8_uses_kitchen_on_unsupported_device(self):
        kitchen = mock.Mock(return_value="kitchen")
        kitchen.container_function = mock.Mock(return_value="container")
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_attention_device",
                return_value=False,
            ),
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: (
                    kitchen if name == "comfy_kitchen_int8" else default
                ),
            ),
        ):
            override = attention_patches.make_attention_override(
                "w8a8", device=torch.device("cuda", 0)
            )
        self.assertEqual(
            override.turing_utils_attention_implementation, "comfy:comfy_kitchen_int8"
        )
        containers = tuple(
            comfy_attention.AttentionTensorContainer(
                torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
            )
            for _ in range(3)
        )
        self.assertEqual(
            override.container_function(*containers, 2, skip_reshape=True),
            "kitchen",
        )
        self.assertTrue(all(container.tensor is None for container in containers))

    def test_explicit_w8a8_uses_same_bundled_path_on_ampere(self):
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"
            ) as preflight,
        ):
            override = attention_patches.make_attention_override(
                "w8a8", device=torch.device("cuda", 0)
            )

        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(
            override.turing_utils_attention_implementation,
            "bundled_turing_w8a8",
        )
        self.assertTrue(
            callable(getattr(override, "prepared_attention_executor", None))
        )

    def test_legacy_sage_alias_uses_external_sage_on_non_turing_device(self):
        sage = mock.Mock(return_value="sage")
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch(
                "comfy.ldm.modules.attention.get_attention_function",
                side_effect=lambda name, default: sage if name == "sage" else default,
            ),
        ):
            override = attention_patches.make_attention_override(
                "sage", device=torch.device("cuda", 0)
            )
        self.assertEqual(override.turing_utils_attention_backend, "sage")
        self.assertEqual(override.turing_utils_attention_implementation, "comfy:sage")

    def test_sparse_override_preflights_independent_kernel(self):
        q = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled"
            ) as stable_preflight,
            mock.patch(
                "comfyui_turing_utils.attention.sol.preflight_bundled_sparse"
            ) as preflight,
            mock.patch(
                "comfyui_turing_utils.attention.sol.turing_sol_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sparse_attention_override(
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
                debug_route_density=True,
                dense_backend="sage",
            )
            output = override(
                mock.Mock(),
                q,
                q,
                q,
                2,
                skip_reshape=True,
            )

        self.assertIs(output, q)
        sparse.assert_called_once()
        stable_preflight.assert_called_once_with(torch.device("cuda", 0))
        preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(sparse.call_args.kwargs["min_sequence_tokens"], 8192)
        self.assertEqual(sparse.call_args.kwargs["routing_threshold"], 0.85)
        self.assertEqual(sparse.call_args.kwargs["prefix_policy"], "manual")
        self.assertEqual(sparse.call_args.kwargs["manual_prefix_tokens"], 256)
        self.assertEqual(sparse.call_args.kwargs["skipped_residual"], "1x64")
        self.assertTrue(sparse.call_args.kwargs["sparse_reference_image"])
        self.assertFalse(sparse.call_args.kwargs["sparse_reference_video"])
        self.assertTrue(sparse.call_args.kwargs["sparse_reference_audio"])
        self.assertTrue(sparse.call_args.kwargs["debug_route_density"])
        self.assertFalse(sparse.call_args.kwargs["use_w8a8"])
        self.assertIsInstance(sparse.call_args.kwargs["debug_route_keys"], set)
        self.assertIsInstance(sparse.call_args.kwargs["debug_route_state"], dict)
        self.assertEqual(
            override.turing_utils_attention_implementation,
            "bundled_sol_sparse",
        )

    def test_sparse_override_uses_stable_sage_for_first_and_last_layers(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled"),
            mock.patch("comfyui_turing_utils.attention.sol.preflight_bundled_sparse"),
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_sage_attention",
                return_value=q,
            ) as stable,
            mock.patch(
                "comfyui_turing_utils.attention.sol.turing_sol_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sparse_attention_override(
                torch.device("cuda", 0), debug_route_density=True, dense_backend="sage"
            )
            for layer_index in (0, 1, 49):
                override(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    2,
                    skip_reshape=True,
                    transformer_options={
                        "turing_utils_attention_layout": {
                            "layer_index": layer_index,
                            "layer_count": 50,
                        }
                    },
                )

        self.assertEqual(stable.call_count, 2)
        self.assertEqual(sparse.call_count, 1)
        self.assertEqual(
            sparse.call_args.kwargs["debug_context"]["last_sparse_layer"], 49
        )

    def test_sol_and_sla_dense_protection_use_external_sage_prepared_path(self):
        spec = attention_protocol.QKTransformSpec(
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RMSNormSpec(
                torch.ones(128, dtype=torch.bfloat16), 1e-6, "head"
            ),
            attention_protocol.RotaryEmbeddingSpec(None, 0, "none"),
        )
        expected = torch.zeros((1, 4096, 256), dtype=torch.bfloat16)
        for maker, available, preflight in (
            (
                attention_patches.make_sparse_attention_override,
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                "comfyui_turing_utils.attention.sol.preflight_bundled_sparse",
            ),
            (
                attention_patches.make_sla_attention_override,
                "comfyui_turing_utils.attention.sla.bundled_sla_available",
                "comfyui_turing_utils.attention.sla.preflight_bundled_sla",
            ),
        ):
            with self.subTest(strategy=maker.__name__):
                qkv = [
                    comfy_attention.AttentionTensorContainer(
                        torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
                    )
                    for _ in range(3)
                ]
                dense = mock.Mock()
                dense.turing_utils_attention_implementation = "comfy:sage"
                dense.prepared_attention_executor = mock.Mock(
                    return_value=attention_protocol.AttentionExecutionOutcome(expected)
                )
                with (
                    mock.patch(
                        f"{maker.__module__}.is_supported_attention_device",
                        return_value=True,
                    ),
                    mock.patch(available, return_value=True),
                    mock.patch(preflight),
                    mock.patch(
                        f"{maker.__module__}.fused_qk_preprocessing_available",
                        return_value=True,
                    ),
                ):
                    override = maker(
                        torch.device("cuda", 0),
                        dense_backend="sage",
                        dense_override=dense,
                        dense_prefix_steps=0,
                        dense_prefix_layers=1,
                    )
                request = attention_protocol.PreparedAttention.from_hnd(
                    *qkv,
                    heads=2,
                    qk_transform=spec,
                    transformer_options={
                        "turing_utils_attention_layout": {
                            "layer_index": 0,
                            "layer_count": 50,
                        }
                    },
                )
                outcome = override.prepared_attention_executor(request)

                self.assertIs(outcome.output, expected)
                dense.prepared_attention_executor.assert_called_once_with(request)
                self.assertEqual(override.turing_utils_dense_backend, "sage")
                self.assertEqual(override.turing_utils_sparse_numeric_backend, "fp16")

    def test_sparse_w8a8_preflights_and_uses_w8a8_for_protected_layers(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled"),
            mock.patch("comfyui_turing_utils.attention.sol.preflight_bundled_sparse"),
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"
            ) as w8a8_preflight,
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_w8a8_attention",
                return_value=q,
            ) as dense,
            mock.patch(
                "comfyui_turing_utils.attention.sol.turing_sol_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sparse_attention_override(
                torch.device("cuda", 0), dense_backend="w8a8"
            )
            override(
                mock.Mock(),
                q,
                q,
                q,
                2,
                skip_reshape=True,
                transformer_options={
                    "turing_utils_attention_layout": {
                        "layer_index": 0,
                        "layer_count": 50,
                    }
                },
            )
            override(
                mock.Mock(),
                q,
                q,
                q,
                2,
                skip_reshape=True,
                transformer_options={
                    "turing_utils_attention_layout": {
                        "layer_index": 2,
                        "layer_count": 50,
                    }
                },
            )

        w8a8_preflight.assert_called_once_with(torch.device("cuda", 0))
        dense.assert_called_once()
        sparse.assert_called_once()
        self.assertTrue(sparse.call_args.kwargs["use_w8a8"])

    def test_sparse_ampere_uses_bundled_sol_and_shared_dense_backend(self):
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.preflight_bundled_sparse"
            ) as sparse_preflight,
            mock.patch(
                "comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"
            ) as w8a8_preflight,
        ):
            override = attention_patches.make_sparse_attention_override(
                torch.device("cuda", 0), dense_backend="w8a8"
            )

        sparse_preflight.assert_called_once_with(torch.device("cuda", 0))
        w8a8_preflight.assert_called_once_with(torch.device("cuda", 0))
        self.assertEqual(
            override.turing_utils_attention_implementation,
            "bundled_sol_sparse",
        )
        self.assertEqual(
            override.turing_utils_dense_implementation,
            "bundled_turing_w8a8",
        )

    def test_overlapping_dense_layer_ranges_bypass_sol_for_every_layer(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled"),
            mock.patch("comfyui_turing_utils.attention.sol.preflight_bundled_sparse"),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"),
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_w8a8_attention",
                return_value=q,
            ) as dense,
            mock.patch(
                "comfyui_turing_utils.attention.sol.turing_sol_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sparse_attention_override(
                torch.device("cuda", 0),
                dense_prefix_layers=26,
                dense_suffix_layers=25,
            )
            for layer_index in range(50):
                override(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    2,
                    skip_reshape=True,
                    transformer_options={
                        "turing_utils_attention_layout": {
                            "layer_index": layer_index,
                            "layer_count": 50,
                        }
                    },
                )

        self.assertEqual(dense.call_count, 50)
        sparse.assert_not_called()

    def test_sparse_rejects_unsupported_device(self):
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.is_supported_attention_device",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "CUDA Tensor Core GPU"),
        ):
            attention_patches.make_sparse_attention_override(torch.device("cuda", 0))

    def test_sla_override_forwards_fixed_topk_and_semantic_controls(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.bundled_sla_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled"),
            mock.patch(
                "comfyui_turing_utils.attention.sla.preflight_bundled_sla"
            ) as preflight,
            mock.patch(
                "comfyui_turing_utils.attention.sla.turing_sla_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sla_attention_override(
                torch.device("cuda", 0),
                min_sequence_tokens=2048,
                sparsity_ratio=0.85,
                prefix_policy="manual",
                manual_prefix_tokens=128,
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                dense_prefix_steps=0,
                dense_prefix_layers=0,
                dense_backend="sage",
                debug_route_density=True,
            )
            output = override(mock.Mock(), q, q, q, 2, skip_reshape=True)

        self.assertIs(output, q)
        preflight.assert_called_once_with(torch.device("cuda", 0))
        sparse.assert_called_once()
        kwargs = sparse.call_args.kwargs
        self.assertEqual(kwargs["min_sequence_tokens"], 2048)
        self.assertEqual(kwargs["sparsity_ratio"], 0.85)
        self.assertEqual(kwargs["prefix_policy"], "manual")
        self.assertEqual(kwargs["manual_prefix_tokens"], 128)
        self.assertTrue(kwargs["sparse_reference_image"])
        self.assertFalse(kwargs["sparse_reference_video"])
        self.assertTrue(kwargs["sparse_reference_audio"])
        self.assertFalse(kwargs["use_w8a8"])
        self.assertTrue(kwargs["debug_route_density"])
        self.assertEqual(
            override.turing_utils_attention_implementation,
            "bundled_sla_sparse",
        )

    def test_sla_overlapping_dense_layers_bypass_sparse_kernel(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.bundled_sla_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.sla.preflight_bundled_sla"),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"),
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_w8a8_attention",
                return_value=q,
            ) as dense,
            mock.patch(
                "comfyui_turing_utils.attention.sla.turing_sla_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sla_attention_override(
                torch.device("cuda", 0),
                dense_prefix_layers=26,
                dense_suffix_layers=25,
            )
            for layer_index in range(50):
                override(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    2,
                    skip_reshape=True,
                    transformer_options={
                        "turing_utils_attention_layout": {
                            "layer_index": layer_index,
                            "layer_count": 50,
                        }
                    },
                )

        self.assertEqual(dense.call_count, 50)
        sparse.assert_not_called()

    def test_sla_rejects_invalid_sparsity_before_preflight(self):
        with self.assertRaisesRegex(ValueError, "sparsity_ratio"):
            attention_patches.make_sla_attention_override(
                torch.device("cuda", 0), sparsity_ratio=1.0
            )

    def test_zero_sla_sparsity_dispatches_directly_to_dense_backend(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.dense.is_supported_turing_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sla.bundled_sla_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.dense.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.sla.preflight_bundled_sla"),
            mock.patch("comfyui_turing_utils.attention.dense.preflight_bundled_w8a8"),
            mock.patch(
                "comfyui_turing_utils.attention.dense.turing_w8a8_attention",
                return_value=q,
            ) as dense,
            mock.patch(
                "comfyui_turing_utils.attention.sla.turing_sla_sparse_attention"
            ) as sparse,
        ):
            override = attention_patches.make_sla_attention_override(
                torch.device("cuda", 0), sparsity_ratio=0.0
            )
            output = override(mock.Mock(), q, q, q, 2, skip_reshape=True)

        self.assertIs(output, q)
        dense.assert_called_once()
        sparse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
