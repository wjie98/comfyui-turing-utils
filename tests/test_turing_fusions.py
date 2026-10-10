from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.quantization import dispatch as turing_ops
from comfyui_turing_utils.quantization import fusions as turing_fusions


class SegmentValidationTest(unittest.TestCase):
    def test_accepts_nonuniform_contiguous_segments(self):
        self.assertEqual(
            turing_fusions._normalized_segments(
                [(0, 1, 2), (1, 7, 0), (7, 19, 1)], 19, 3
            ),
            (0, 1, 2, 1, 7, 0, 7, 19, 1),
        )

    def test_rejects_gaps_and_invalid_modulation_rows(self):
        with self.assertRaisesRegex(ValueError, "contiguously"):
            turing_fusions._normalized_segments([(0, 2, 0), (3, 4, 0)], 4, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            turing_fusions._normalized_segments([(0, 4, 1)], 4, 1)

    def test_large_w8_workspace_selects_fixed_workspace_path(self):
        limit = turing_ops.TURING_INT8_GLOBAL_WORKSPACE_LIMIT
        self.assertGreater(turing_ops.turing_int8_workspace_bytes(4, 8), 0)
        self.assertEqual(
            turing_ops.turing_int8_workspace_bytes(limit // (4 * 8), 8),
            0,
        )
        self.assertEqual(
            turing_ops.turing_int8_workspace_bytes(limit // 4, 1),
            limit,
        )

    def test_codebook_workspace_is_bounded_by_output_chunks(self):
        chunk_rows = turing_ops.TURING_CODEBOOK_W4A8_CHUNK_ROWS
        self.assertEqual(
            turing_ops.turing_codebook_w4a8_workspace_bytes(28672, 5376),
            28672 * chunk_rows,
        )
        self.assertEqual(
            turing_ops.turing_codebook_w4a8_workspace_bytes(5376, 28672),
            5376 * chunk_rows,
        )
        self.assertEqual(
            turing_ops.turing_codebook_w4a8_workspace_bytes(0, 28672),
            0,
        )


class FusionDispatchTest(unittest.TestCase):
    @staticmethod
    def _w8a8_weight(out_features=4, in_features=8):
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        qdata = torch.zeros((out_features, in_features), dtype=torch.int8)
        params = TensorWiseINT8Layout.Params(
            scale=torch.ones(out_features, dtype=torch.float32),
            orig_dtype=torch.bfloat16,
            orig_shape=(out_features, in_features),
            convrot=True,
            convrot_groupsize=256,
        )
        return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)

    @staticmethod
    def _w4_weight(linear_dtype: str, out_features=4, in_features=256):
        from comfy.quant_ops import QuantizedTensor, TensorCoreConvRotW4A4Layout

        qdata = torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
        params = TensorCoreConvRotW4A4Layout.Params(
            scale=torch.ones((out_features, in_features // 64), dtype=torch.float32),
            orig_dtype=torch.bfloat16,
            orig_shape=(out_features, in_features),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype=linear_dtype,
        )
        return QuantizedTensor(qdata, "TensorCoreConvRotW4A4Layout", params)

    @staticmethod
    def _codebook_w4_weight(out_features=8, in_features=256):
        from comfy.quant_ops import AsymW4A8Int8Layout, QuantizedTensor

        qdata = torch.zeros((out_features, in_features // 2), dtype=torch.int8)
        params = AsymW4A8Int8Layout.Params(
            scale=torch.ones(
                (out_features, in_features // 16), dtype=torch.float8_e4m3fn
            ),
            s_channel=torch.ones(out_features, dtype=torch.float32),
            codebook=torch.linspace(-1, 1, 16, dtype=torch.float32),
            correction=None,
            group_size=16,
            convrot_groupsize=256,
            orig_dtype=torch.bfloat16,
            orig_shape=(out_features, in_features),
        )
        return QuantizedTensor(qdata, "AsymW4A8Int8Layout", params)

    def test_direct_turing_input_act_preserves_cast_lifecycle(self):
        weight = self._w8a8_weight()
        linear = SimpleNamespace(weight=weight)
        x = torch.zeros((2, 16), dtype=torch.bfloat16)
        output = torch.empty((2, 4), dtype=torch.bfloat16)
        offload = (None, None, None)

        with (
            mock.patch(
                "comfyui_turing_utils.hardware.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(weight, None, offload),
            ) as cast,
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.int8_linear",
                return_value=output,
            ) as int8_linear,
        ):
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertIs(result, output)
        cast.assert_called_once_with(
            linear,
            x,
            offloadable=True,
            compute_dtype=torch.bfloat16,
            want_requant=True,
        )
        uncast.assert_called_once_with(linear, weight, None, offload)
        self.assertEqual(int8_linear.call_args.kwargs["input_act"], "swiglu")
        self.assertTrue(int8_linear.call_args.kwargs["convrot"])

    def test_direct_turing_input_act_supports_w4a4_and_w4a8(self):
        for linear_dtype in ("int4", "int8"):
            with self.subTest(linear_dtype=linear_dtype):
                weight = self._w4_weight(linear_dtype)
                linear = SimpleNamespace(weight=weight)
                x = torch.zeros((2, 512), dtype=torch.bfloat16)
                output = torch.empty((2, 4), dtype=torch.bfloat16)
                offload = (None, None, None)
                with (
                    mock.patch(
                        "comfyui_turing_utils.hardware.is_supported_attention_device",
                        return_value=True,
                    ),
                    mock.patch(
                        "comfyui_turing_utils.quantization.dispatch.is_supported_tensor_core_device",
                        return_value=True,
                    ),
                    mock.patch(
                        "comfy.ops.cast_bias_weight",
                        return_value=(weight, None, offload),
                    ),
                    mock.patch("comfy.ops.uncast_bias_weight") as uncast,
                    mock.patch(
                        "comfyui_turing_utils.quantization.dispatch.convrot_w4a4_linear",
                        return_value=output,
                    ) as kernel,
                ):
                    result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

                self.assertIs(result, output)
                self.assertEqual(kernel.call_args.kwargs["linear_dtype"], linear_dtype)
                self.assertEqual(kernel.call_args.kwargs["input_act"], "swiglu")
                uncast.assert_called_once_with(linear, weight, None, offload)

    def test_direct_turing_input_act_forwards_fused_gelu(self):
        weight = self._w8a8_weight(out_features=4, in_features=256)
        linear = SimpleNamespace(weight=weight)
        x = torch.zeros((2, 256), dtype=torch.bfloat16)
        output = torch.empty((2, 4), dtype=torch.bfloat16)
        offload = (None, None, None)
        with (
            mock.patch(
                "comfyui_turing_utils.hardware.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch(
                "comfy.ops.cast_bias_weight", return_value=(weight, None, offload)
            ),
            mock.patch("comfy.ops.uncast_bias_weight"),
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.int8_linear",
                return_value=output,
            ) as kernel,
        ):
            result = turing_fusions.turing_linear_input_act(linear, x, "gelu_tanh")

        self.assertIs(result, output)
        self.assertEqual(kernel.call_args.kwargs["input_act"], "gelu_tanh")

    def test_direct_turing_input_act_supports_grouped_codebook_w4a8(self):
        weight = self._codebook_w4_weight()
        linear = SimpleNamespace(weight=weight)
        x = torch.zeros((2, 512), dtype=torch.bfloat16)
        output = torch.empty((2, 8), dtype=torch.bfloat16)
        offload = (None, None, None)
        with (
            mock.patch(
                "comfyui_turing_utils.hardware.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(weight, None, offload),
            ),
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.codebook_w4a8_linear",
                return_value=output,
            ) as kernel,
        ):
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertIs(result, output)
        self.assertEqual(kernel.call_args.kwargs["group_size"], 16)
        self.assertEqual(kernel.call_args.kwargs["input_act"], "swiglu")
        uncast.assert_called_once_with(linear, weight, None, offload)

    def test_codebook_w4a8_swiglu_uses_native_turing_path(self):
        from comfy_kitchen.backends import cuda as kitchen_cuda

        x = torch.zeros((2, 512), dtype=torch.bfloat16)
        qdata = torch.zeros((8, 128), dtype=torch.int8)
        s_rel = torch.ones((8, 16), dtype=torch.float8_e4m3fn)
        s_channel = torch.ones(8, dtype=torch.float32)
        codebook = torch.linspace(-1, 1, 16, dtype=torch.float32)
        qactivation = torch.zeros((2, 256), dtype=torch.int8)
        activation_scale = torch.ones(2, dtype=torch.float32)
        expected = torch.zeros((2, 8), dtype=torch.bfloat16)
        native = mock.Mock(return_value=expected)

        with (
            mock.patch.object(
                turing_ops, "is_supported_tensor_core_device", return_value=True
            ),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int8_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.object(turing_ops, "_kernel_op", return_value=native),
            mock.patch.object(kitchen_cuda, "w4a8_int8_linear") as fallback,
        ):
            result = turing_ops.codebook_w4a8_linear(
                x,
                qdata,
                s_rel,
                s_channel,
                codebook=codebook,
                input_act="swiglu",
            )

        torch.testing.assert_close(result, expected)
        self.assertEqual(result.data_ptr(), expected.data_ptr())
        fallback.assert_not_called()
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.args[1], 256)
        self.assertEqual(quantize.call_args.kwargs["input_act"], "swiglu")
        native.assert_called_once_with(
            qactivation,
            qdata,
            activation_scale,
            s_rel,
            s_channel,
            codebook,
            None,
            16,
        )

    def test_large_w8_gemm_attempts_fixed_workspace_even_when_latency_heuristic_declines(
        self,
    ):
        from comfy_kitchen.backends import cuda as kitchen_cuda

        qactivation = torch.zeros((2, 16), dtype=torch.int8)
        weight = torch.zeros((8, 16), dtype=torch.int8)
        activation_scale = torch.ones(2, dtype=torch.float32)
        weight_scale = torch.ones(8, dtype=torch.float32)
        expected = torch.zeros((2, 8), dtype=torch.bfloat16)
        with (
            mock.patch.object(turing_ops, "TURING_INT8_GLOBAL_WORKSPACE_LIMIT", 16),
            mock.patch.object(
                kitchen_cuda, "_prefer_turing_fused_int8", return_value=False
            ),
            mock.patch.object(
                kitchen_cuda,
                "_int8_linear_turing_quantized",
                return_value=expected,
            ) as fused,
        ):
            output = turing_ops._turing_int8_gemm(
                qactivation,
                weight,
                activation_scale,
                weight_scale,
                None,
                torch.bfloat16,
            )

        self.assertIs(output, expected)
        fused.assert_called_once()

    def test_direct_turing_input_act_keeps_dense_fallback(self):
        linear = SimpleNamespace(weight=torch.zeros(4, 8, dtype=torch.bfloat16))
        x = torch.zeros((2, 16), dtype=torch.bfloat16)
        output = torch.empty((2, 4), dtype=torch.bfloat16)

        with mock.patch("comfy.ops.linear_input_act", return_value=output) as fallback:
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertIs(result, output)
        fallback.assert_called_once_with(linear, x, "swiglu")

    def test_direct_turing_input_act_handles_dequantized_cast(self):
        quantized = self._w8a8_weight()
        dense = torch.zeros((4, 8), dtype=torch.bfloat16)
        linear = SimpleNamespace(weight=quantized)
        x = torch.zeros((2, 16), dtype=torch.bfloat16)
        offload = (None, None, None)

        with (
            mock.patch(
                "comfyui_turing_utils.hardware.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(dense, None, offload),
            ),
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch(
                "comfyui_turing_utils.quantization.dispatch.int8_linear"
            ) as int8_linear,
        ):
            result = turing_fusions.turing_linear_input_act(linear, x, "swiglu")

        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(result.dtype, torch.bfloat16)
        int8_linear.assert_not_called()
        uncast.assert_called_once_with(linear, dense, None, offload)

    def test_segmented_norm_preserves_cast_weight_lifecycle(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        weight = torch.ones(8, dtype=torch.bfloat16)
        scale = torch.zeros(2, 8, dtype=torch.bfloat16)
        shift = torch.zeros(2, 8, dtype=torch.bfloat16)
        table = torch.tensor([[0, 2, 0], [2, 4, 1]], dtype=torch.int32)
        output = torch.empty_like(x)
        kernel = mock.Mock(return_value=output)
        norm = SimpleNamespace(eps=1.0e-5)

        with (
            mock.patch(
                "comfy.ops.cast_bias_weight",
                return_value=(weight, None, (None, None, None)),
            ) as cast,
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            mock.patch.object(turing_fusions, "_segment_table", return_value=table),
            mock.patch.dict(
                sys.modules,
                {
                    "comfyui_turing_utils_kernel": SimpleNamespace(
                        turing_segmented_rms_adaln=kernel
                    )
                },
            ),
        ):
            result = turing_fusions.segmented_rms_adaln(
                norm, x, shift, scale, [(0, 2, 0), (2, 4, 1)]
            )

        self.assertIs(result, output)
        cast.assert_called_once_with(norm, x, offloadable=True)
        uncast.assert_called_once_with(norm, weight, None, (None, None, None))
        kernel.assert_called_once()
        self.assertIs(kernel.call_args.args[0], x)
        self.assertIs(kernel.call_args.args[1], weight)
        self.assertIs(kernel.call_args.args[4], table)


if __name__ == "__main__":
    unittest.main()
