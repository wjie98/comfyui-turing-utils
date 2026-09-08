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

from comfyui_turing_utils import precision as bf16_policy  # noqa: E402
from comfyui_turing_utils import hardware  # noqa: E402
from comfyui_turing_utils.quantization import dispatch as turing_ops  # noqa: E402
from comfy_kitchen.backends import cuda as kitchen_cuda  # noqa: E402


SUMMARY = SimpleNamespace(w4a4=1, w4a8=1, w8a8=1)
NO_CONVROT = SimpleNamespace(w4a4=0, w4a8=0, w8a8=0)
BF16_CONFIG = SimpleNamespace(supported_inference_dtypes=[torch.bfloat16, torch.float32])
FP16_BF16_CONFIG = SimpleNamespace(
    supported_inference_dtypes=[torch.float16, torch.bfloat16, torch.float32]
)


class BF16PolicyTest(unittest.TestCase):
    @staticmethod
    def _w8_weight(dtype=torch.float32, *, convrot=True):
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        qdata = torch.zeros((4, 8), dtype=torch.int8)
        params = TensorWiseINT8Layout.Params(
            scale=torch.ones(1, dtype=torch.float32),
            orig_dtype=dtype,
            orig_shape=(4, 8),
            convrot=convrot,
            convrot_groupsize=256,
        )
        return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)

    @staticmethod
    def _w4_weight(dtype=torch.float32, *, linear_dtype="int4"):
        from comfy.quant_ops import QuantizedTensor, TensorCoreConvRotW4A4Layout

        qdata = torch.zeros((4, 4), dtype=torch.uint8)
        params = TensorCoreConvRotW4A4Layout.Params(
            scale=torch.ones((4, 1), dtype=torch.float32),
            orig_dtype=dtype,
            orig_shape=(4, 8),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype=linear_dtype,
        )
        return QuantizedTensor(qdata, "TensorCoreConvRotW4A4Layout", params)

    @staticmethod
    def _module_with_weight(weight):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
        return module

    def test_non_turing_cuda_keeps_comfyui_policy(self):
        with (
            mock.patch("comfyui_turing_utils.precision._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            dtype = bf16_policy.select_compute_dtype(BF16_CONFIG, torch.device("cuda", 0))
        self.assertIsNone(dtype)

    def test_model_without_declared_bf16_keeps_comfyui_policy(self):
        config = SimpleNamespace(supported_inference_dtypes=[torch.float16, torch.float32])
        with mock.patch("comfyui_turing_utils.precision._explicit_dtype_override", return_value=False):
            dtype = bf16_policy.select_compute_dtype(config, torch.device("cuda", 0))
        self.assertIsNone(dtype)

    def test_explicit_comfyui_dtype_override_wins(self):
        with mock.patch("comfyui_turing_utils.precision._explicit_dtype_override", return_value=True):
            dtype = bf16_policy.select_compute_dtype(BF16_CONFIG, torch.device("cuda", 0))
        self.assertIsNone(dtype)

    def test_supported_turing_selects_bf16_independently_of_runtime_preflight(self):
        with (
            mock.patch("comfyui_turing_utils.precision._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True),
        ):
            dtype = bf16_policy.select_compute_dtype(BF16_CONFIG, torch.device("cuda", 1))
        self.assertIs(dtype, torch.bfloat16)

    def test_turing_model_with_fp16_support_keeps_comfyui_policy(self):
        with (
            mock.patch("comfyui_turing_utils.precision._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True),
        ):
            dtype = bf16_policy.select_compute_dtype(
                FP16_BF16_CONFIG, torch.device("cuda", 0)
            )
        self.assertIsNone(dtype)

    def test_turing_preflight_failure_does_not_silently_fallback_to_fp32(self):
        with (
            mock.patch("comfyui_turing_utils.precision.is_supported_tensor_core_device", return_value=True),
            mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True),
            mock.patch("comfyui_turing_utils.precision.bundled_w8a8_available", return_value=True),
            mock.patch("comfyui_turing_utils.precision.preflight_bundled_w8a8", side_effect=RuntimeError("attention self-test")),
            self.assertRaisesRegex(RuntimeError, "attention self-test"),
        ):
            bf16_policy.prepare_turing_runtime(
                NO_CONVROT, torch.device("cuda", 0), "w8a8"
            )

    def test_legacy_sage_alias_preflights_the_canonical_bundled_backend(self):
        with (
            mock.patch("comfyui_turing_utils.precision.is_supported_tensor_core_device", return_value=True),
            mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True),
            mock.patch("comfyui_turing_utils.precision._check_kernel_contract"),
            mock.patch("comfyui_turing_utils.precision.bundled_available", return_value=True),
            mock.patch("comfyui_turing_utils.precision.preflight_bundled") as preflight,
        ):
            bf16_policy.prepare_turing_runtime(
                NO_CONVROT, torch.device("cuda", 0), "sage_"
            )
        preflight.assert_called_once_with(torch.device("cuda", 0))

    def test_turing_runtime_rejects_stale_independent_kernel(self):
        with (
            mock.patch.dict(
                sys.modules, {"comfyui_turing_utils_kernel": SimpleNamespace(__version__="0.4.9")}
            ),
            self.assertRaisesRegex(RuntimeError, "comfyui-turing-utils-kernel>=0.8.0"),
        ):
            bf16_policy._check_kernel_contract()

    def test_w8a8_alone_registers_turing_backend_before_preflight(self):
        summary = SimpleNamespace(w4a4=0, w4a8=0, w8a8=1)
        with (
            mock.patch(
                "comfy_kitchen.list_backends",
                return_value={
                    "cuda": {
                        "available": True,
                        "disabled": False,
                        "capabilities": ("int8_linear",),
                    }
                },
            ),
            mock.patch("comfyui_turing_utils.precision.is_supported_tensor_core_device", return_value=True),
            mock.patch("comfyui_turing_utils.precision._check_kitchen_contract"),
            mock.patch("comfyui_turing_utils.precision.register_backend", return_value=True) as register,
            mock.patch("comfyui_turing_utils.precision.backend_available", return_value=True),
            mock.patch("comfyui_turing_utils.precision.preflight_kitchen") as preflight,
        ):
            bf16_policy.prepare_turing_runtime(summary, torch.device("cuda", 0), "sdpa")
        register.assert_called_once_with()
        preflight.assert_called_once_with(torch.device("cuda", 0), False, True)

    def test_scoped_runtime_accepts_installed_kitchen_cuda_when_globally_disabled(self):
        summary = SimpleNamespace(w4a4=0, w4a8=0, w8a8=1)
        device = torch.device("cuda", 0)
        with (
            mock.patch(
                "comfy_kitchen.list_backends",
                return_value={
                    "cuda": {
                        "available": True,
                        "disabled": True,
                        "unavailable_reason": "requires CUDA 13",
                        "capabilities": ("int8_linear",),
                    }
                },
            ),
            mock.patch(
                "comfyui_turing_utils.precision.is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.precision._check_kernel_contract"),
            mock.patch("comfyui_turing_utils.precision._check_kitchen_contract"),
            mock.patch(
                "comfyui_turing_utils.precision.register_backend", return_value=True
            ) as register,
            mock.patch(
                "comfyui_turing_utils.precision.backend_available", return_value=True
            ),
            mock.patch(
                "comfyui_turing_utils.precision.preflight_kitchen"
            ) as preflight,
        ):
            bf16_policy.prepare_turing_runtime(summary, device, "sdpa")

        register.assert_called_once_with()
        preflight.assert_called_once_with(device, False, True)

    def test_codebook_w4a8_requires_new_kernel_and_uses_its_preflight(self):
        summary = SimpleNamespace(
            w4a4=0, w4a8=0, codebook_w4a8=1, w8a8=0
        )
        with (
            mock.patch(
                "comfy_kitchen.list_backends",
                return_value={
                    "cuda": {
                        "available": True,
                        "disabled": False,
                        "capabilities": ("convrot_w4a4_linear", "int8_linear", "w4a8_int8_linear"),
                    }
                },
            ),
            mock.patch(
                "comfyui_turing_utils.precision.is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.precision._check_kernel_contract") as contract,
            mock.patch("comfyui_turing_utils.precision._check_kitchen_contract"),
            mock.patch("comfyui_turing_utils.precision.register_backend", return_value=True),
            mock.patch("comfyui_turing_utils.precision.backend_available", return_value=True),
            mock.patch("comfyui_turing_utils.precision.preflight_kitchen"),
            mock.patch("comfyui_turing_utils.precision.preflight_codebook_w4a8") as preflight,
        ):
            bf16_policy.prepare_turing_runtime(
                summary, torch.device("cuda", 0), "sdpa"
            )

        self.assertEqual(
            contract.call_args_list,
            [
                mock.call(),
                mock.call(bf16_policy.MIN_CODEBOOK_W4A8_KERNEL_VERSION),
            ],
        )
        preflight.assert_called_once_with(torch.device("cuda", 0))

    def test_gtx16_keeps_comfyui_fallback(self):
        with (
            mock.patch("comfyui_turing_utils.precision._explicit_dtype_override", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 Ti"),
            mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=False),
        ):
            dtype = bf16_policy.select_compute_dtype(BF16_CONFIG, torch.device("cuda", 0))
        self.assertIsNone(dtype)

    def test_turing_convrot_logical_dtype_is_normalized_without_copying_qdata(self):
        weight = self._w8_weight(torch.float32, convrot=True)
        module = self._module_with_weight(weight)
        root = torch.nn.Module()
        root.fc = module
        old_qdata_ptr = module.weight._qdata.data_ptr()
        old_scale_ptr = module.weight._params.scale.data_ptr()

        with mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True):
            count = bf16_policy.normalize_turing_convrot_weight_dtypes(
                root, torch.device("cuda", 0), torch.bfloat16
            )

        self.assertEqual(count, 1)
        self.assertIs(module.weight.dtype, torch.bfloat16)
        self.assertIs(module.weight._params.orig_dtype, torch.bfloat16)
        self.assertEqual(module.weight._qdata.data_ptr(), old_qdata_ptr)
        self.assertEqual(module.weight._params.scale.data_ptr(), old_scale_ptr)
        self.assertIs(module.weight_comfy_model_dtype, torch.bfloat16)

    def test_turing_dtype_normalization_does_not_touch_dense_or_nonconvrot_weights(self):
        dense = torch.nn.Linear(8, 4, bias=False, dtype=torch.float32)
        plain_int8 = self._module_with_weight(self._w8_weight(torch.float32, convrot=False))
        root = torch.nn.Module()
        root.dense = dense
        root.plain_int8 = plain_int8
        dense_weight = dense.weight
        plain_weight = plain_int8.weight

        with mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True):
            count = bf16_policy.normalize_turing_convrot_weight_dtypes(
                root, torch.device("cuda", 0), torch.bfloat16
            )

        self.assertEqual(count, 0)
        self.assertIs(dense.weight, dense_weight)
        self.assertIs(dense.weight.dtype, torch.float32)
        self.assertIs(plain_int8.weight, plain_weight)
        self.assertIs(plain_int8.weight.dtype, torch.float32)
        self.assertFalse(hasattr(plain_int8, "weight_comfy_model_dtype"))

    def test_turing_w4a4_and_w4a8_logical_dtypes_are_normalized(self):
        root = torch.nn.Module()
        root.w4a4 = self._module_with_weight(self._w4_weight(linear_dtype="int4"))
        root.w4a8 = self._module_with_weight(self._w4_weight(linear_dtype="int8"))

        with mock.patch("comfyui_turing_utils.precision.is_supported_turing_device", return_value=True):
            count = bf16_policy.normalize_turing_convrot_weight_dtypes(
                root, torch.device("cuda", 0), torch.bfloat16
            )

        self.assertEqual(count, 2)
        self.assertIs(root.w4a4.weight.dtype, torch.bfloat16)
        self.assertEqual(root.w4a4.weight._params.linear_dtype, "int4")
        self.assertIs(root.w4a8.weight.dtype, torch.bfloat16)
        self.assertEqual(root.w4a8.weight._params.linear_dtype, "int8")

    def test_convrot_dtype_normalization_is_disabled_off_turing_or_without_bf16(self):
        for supported, dtype in ((False, torch.bfloat16), (True, None), (True, torch.float32)):
            with self.subTest(supported=supported, dtype=dtype):
                module = self._module_with_weight(self._w8_weight(torch.float32, convrot=True))
                with mock.patch(
                    "comfyui_turing_utils.precision.is_supported_turing_device", return_value=supported
                ):
                    count = bf16_policy.normalize_turing_convrot_weight_dtypes(
                        module, torch.device("cuda", 0), dtype
                    )
                self.assertEqual(count, 0)
                self.assertIs(module.weight.dtype, torch.float32)

    def test_device_check_uses_requested_tensor_device(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", side_effect=lambda index: (7, 5) if index == 1 else (8, 6)),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA T4"),
        ):
            self.assertTrue(turing_ops.is_supported_turing_device(torch.device("cuda", 1)))
            self.assertFalse(turing_ops.is_supported_turing_device(torch.device("cuda", 0)))

    def test_integer_attention_supports_ampere_and_newer(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda index: ((7, 5), (8, 0), (8, 9), (9, 0))[index],
            ),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA A100"),
        ):
            self.assertTrue(hardware.is_supported_attention_device(torch.device("cuda", 0)))
            self.assertTrue(hardware.is_supported_attention_device(torch.device("cuda", 1)))
            self.assertTrue(hardware.is_supported_attention_device(torch.device("cuda", 2)))
            self.assertTrue(hardware.is_supported_attention_device(torch.device("cuda", 3)))

        self.assertFalse(hardware.is_supported_attention_device(torch.device("cpu")))

    def test_ampere_uses_the_same_convrot_runtime_preflight(self):
        summary = SimpleNamespace(w4a4=0, w4a8=0, w8a8=1)
        device = torch.device("cuda", 0)
        with (
            mock.patch(
                "comfy_kitchen.list_backends",
                return_value={
                    "cuda": {
                        "available": True,
                        "disabled": False,
                        "capabilities": ("int8_linear",),
                    }
                },
            ),
            mock.patch(
                "comfyui_turing_utils.precision.is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.precision.is_supported_turing_device",
                return_value=False,
            ),
            mock.patch("comfyui_turing_utils.precision._check_kernel_contract"),
            mock.patch("comfyui_turing_utils.precision._check_kitchen_contract"),
            mock.patch("comfyui_turing_utils.precision.register_backend", return_value=True),
            mock.patch("comfyui_turing_utils.precision.backend_available", return_value=True),
            mock.patch("comfyui_turing_utils.precision.preflight_kitchen") as preflight_linear,
            mock.patch(
                "comfyui_turing_utils.precision.bundled_w8a8_available",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.precision.preflight_bundled_w8a8"
            ) as preflight_attention,
        ):
            bf16_policy.prepare_turing_runtime(summary, device, "w8a8")

        preflight_linear.assert_called_once_with(device, False, True)
        preflight_attention.assert_called_once_with(device)

    def test_gtx_16_series_is_not_treated_as_supported_turing(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 Ti"),
        ):
            self.assertFalse(turing_ops.is_supported_turing_device(torch.device("cuda", 0)))

    def test_w4a8_preflight_reports_missing_independent_kernel(self):
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=True),
            mock.patch.object(turing_ops, "_kernel_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "does not provide W4A8"),
        ):
            turing_ops.preflight_w4a8(torch.device("cuda", 0))

    def test_codebook_w4a8_preflight_reports_missing_independent_kernel(self):
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=True),
            mock.patch.object(turing_ops, "_kernel_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "codebook W4A8"),
        ):
            turing_ops.preflight_codebook_w4a8(torch.device("cuda", 0))

    def test_int8_activation_uses_staged_bf16_rotation_above_48k_shared_limit(self):
        x = torch.empty((3, 5376), dtype=torch.bfloat16)
        with (
            mock.patch.object(kitchen_cuda, "quantize_int8_rowwise_convrot64") as fused,
            mock.patch.object(kitchen_cuda, "quantize_int8_convrot_staged", return_value=("q", "s")) as staged,
            mock.patch.dict(sys.modules, {"comfyui_turing_utils_kernel": SimpleNamespace()}),
        ):
            result = turing_ops._quantize_turing_int8_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        staged.assert_called_once_with(x, 256)
        fused.assert_not_called()

    def test_int8_activation_keeps_small_rotation_under_48k(self):
        x = torch.empty((3, 256), dtype=torch.bfloat16)
        with (
            mock.patch.object(
                kitchen_cuda,
                "quantize_int8_rowwise_convrot64",
                return_value=("q", "s"),
            ) as fused,
            mock.patch.object(kitchen_cuda, "quantize_int8_convrot_staged") as staged,
        ):
            result = turing_ops._quantize_turing_int8_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        fused.assert_called_once_with(x, 256)
        staged.assert_not_called()

    def test_bf16_rowbuffer_convrot_replaces_staged_h3_shapes_with_optin_shared(self):
        rowbuffer = mock.Mock(return_value=("q", "s"))
        for hidden_size in (5376, 7168, 14336):
            x = torch.empty((3, hidden_size), dtype=torch.bfloat16)
            with (
                self.subTest(hidden_size=hidden_size),
                mock.patch.object(
                    turing_ops, "is_supported_tensor_core_device", return_value=True
                ),
                mock.patch.object(kitchen_cuda, "quantize_int8_rowwise_convrot64") as fused,
                mock.patch.object(kitchen_cuda, "quantize_int8_convrot_staged") as staged,
                mock.patch.dict(
                    sys.modules,
                    {"comfyui_turing_utils_kernel": SimpleNamespace(
                        turing_bf16_int8_convrot_quantize=rowbuffer
                    )},
                ),
            ):
                result = turing_ops._quantize_turing_int8_activation(x, 256)
            self.assertEqual(result, ("q", "s"))
            rowbuffer.assert_called_with(x, 256, swiglu=False)
            fused.assert_not_called()
            staged.assert_not_called()
            rowbuffer.reset_mock()

    def test_bf16_rowbuffer_convrot_absorbs_h3_swiglu(self):
        x = torch.empty((3, 28672), dtype=torch.bfloat16)
        rowbuffer = mock.Mock(return_value=("q", "s"))
        staged_swiglu = mock.Mock()
        with (
            mock.patch.object(
                turing_ops, "is_supported_tensor_core_device", return_value=True
            ),
            mock.patch.dict(
                sys.modules,
                {"comfyui_turing_utils_kernel": SimpleNamespace(
                    turing_bf16_int8_convrot_quantize=rowbuffer,
                    turing_swiglu_int8_convrot_quantize=staged_swiglu,
                )},
            ),
        ):
            result = turing_ops._quantize_turing_int8_activation(
                x, 256, input_act="swiglu"
            )
        self.assertEqual(result, ("q", "s"))
        rowbuffer.assert_called_once_with(x, 256, swiglu=True)
        staged_swiglu.assert_not_called()

    def test_int8_swiglu_rejects_odd_input_width_before_cuda_dispatch(self):
        x = torch.empty((3, 513), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "width must be even"):
            turing_ops._quantize_turing_int8_activation(
                x, 256, input_act="swiglu"
            )

    def test_w8a8_uses_shared_staged_quantizer(self):
        x = torch.ones((2, 10752), dtype=torch.bfloat16)
        weight = torch.zeros((8, 5376), dtype=torch.int8)
        weight_scale = torch.ones((), dtype=torch.float32)
        qactivation = torch.zeros((2, 5376), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        output = torch.zeros((2, 8), dtype=torch.bfloat16)
        fused_swiglu = mock.Mock(return_value=(qactivation, activation_scale))
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=True),
            mock.patch.object(kitchen_cuda, "_prefer_turing_fused_int8", return_value=False),
            mock.patch.object(turing_ops, "_turing_cublas_int8_bf16", return_value=None),
            mock.patch.object(
                kitchen_cuda,
                "_int4_linear_via_int8_values",
                return_value=output,
            ) as linear,
            mock.patch.dict(
                sys.modules,
                {"comfyui_turing_utils_kernel": SimpleNamespace(
                    turing_swiglu_int8_convrot_quantize=fused_swiglu
                )},
            ),
        ):
            result = turing_ops.int8_linear(
                x,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
                input_act="swiglu",
            )
        self.assertTrue(torch.equal(result, output))
        fused_swiglu.assert_called_once()
        torch.testing.assert_close(fused_swiglu.call_args.args[0], x)
        self.assertEqual(fused_swiglu.call_args.args[1], 256)
        linear.assert_called_once()
        self.assertEqual(linear.call_args.args[3].shape, (8,))

    def test_w8a8_fused_turing_gemm_keeps_scalar_weight_scale(self):
        qactivation = torch.zeros((4, 256), dtype=torch.int8)
        weight = torch.zeros((1024, 256), dtype=torch.int8)
        activation_scale = torch.ones((4, 1), dtype=torch.float32)
        weight_scale = torch.ones((), dtype=torch.float32)
        expected = torch.zeros((4, 1024), dtype=torch.bfloat16)
        with (
            mock.patch.object(kitchen_cuda, "_prefer_turing_fused_int8", return_value=True),
            mock.patch.object(
                kitchen_cuda,
                "_int8_linear_turing_quantized",
                return_value=expected,
            ) as fused,
            mock.patch.object(kitchen_cuda, "_int4_linear_via_int8_values") as fallback,
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
        self.assertEqual(fused.call_args.args[3].numel(), 1)
        fallback.assert_not_called()

    def test_w8a8_contraction_uses_bundled_bf16_epilogue_path(self):
        qactivation = torch.zeros((4, 256), dtype=torch.int8)
        weight = torch.zeros((64, 256), dtype=torch.int8)
        activation_scale = torch.ones((4, 1), dtype=torch.float32)
        weight_scale = torch.ones((), dtype=torch.float32)
        expected = torch.zeros((4, 64), dtype=torch.bfloat16)
        with (
            mock.patch.object(kitchen_cuda, "_prefer_turing_fused_int8", return_value=False),
            mock.patch.object(
                turing_ops,
                "_turing_cublas_int8_bf16",
                return_value=expected,
            ) as fast_epilogue,
            mock.patch.object(kitchen_cuda, "_int4_linear_via_int8_values") as fallback,
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
        self.assertEqual(fast_epilogue.call_args.args[3].numel(), 1)
        fallback.assert_not_called()

    def test_w4a8_linear_uses_shared_staged_quantizer(self):
        x = torch.empty((2, 10752), dtype=torch.bfloat16)
        qweight = torch.empty((8, 2688), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        qactivation = torch.empty((2, 5376), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        output = torch.zeros((2, 8), dtype=torch.bfloat16)
        linear = mock.Mock(return_value=output)
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=True),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int8_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.dict(sys.modules, {"comfyui_turing_utils_kernel": SimpleNamespace(turing_w4a8_linear=linear)}),
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int8",
                input_act="swiglu",
            )
        self.assertTrue(torch.equal(result, output))
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.args[1], 256)
        self.assertEqual(quantize.call_args.kwargs["input_act"], "swiglu")
        linear.assert_called_once()
        self.assertIs(linear.call_args.args[0], qactivation)
        self.assertIs(linear.call_args.args[1], qweight)
        self.assertIs(linear.call_args.args[2], activation_scale)

    def test_w4a4_uses_grouped_rotation_when_fused_shared_memory_is_full(self):
        x = torch.empty((3, 28672), dtype=torch.bfloat16)
        rotated = torch.empty_like(x)
        with (
            mock.patch.object(kitchen_cuda, "quantize_int4_rowwise_convrot64") as fused,
            mock.patch.object(kitchen_cuda, "rotate_int8_convrot_weight", return_value=rotated) as rotate,
            mock.patch.object(kitchen_cuda, "quantize_int4_rowwise", return_value=("q", "s")) as quantize,
        ):
            result = turing_ops._quantize_turing_int4_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        fused.assert_not_called()
        rotate.assert_called_once_with(x, 256)
        quantize.assert_called_once_with(rotated)

    def test_w4a4_uses_optin_rowbuffer_above_kitchen_default_limit(self):
        x = torch.empty((3, 16384), dtype=torch.bfloat16)
        rowbuffer = mock.Mock(return_value=("q", "s"))
        with (
            mock.patch.object(
                turing_ops, "is_supported_tensor_core_device", return_value=True
            ),
            mock.patch.object(kitchen_cuda, "quantize_int4_rowwise_convrot64") as fused,
            mock.patch.object(kitchen_cuda, "rotate_int8_convrot_weight") as rotate,
            mock.patch.dict(
                sys.modules,
                {"comfyui_turing_utils_kernel": SimpleNamespace(
                    turing_bf16_int4_convrot_quantize=rowbuffer
                )},
            ),
        ):
            result = turing_ops._quantize_turing_int4_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        rowbuffer.assert_called_once_with(x, 256, swiglu=False)
        fused.assert_not_called()
        rotate.assert_not_called()

    def test_w4a4_prefers_local_bf16_rotation_when_it_fits(self):
        x = torch.empty((3, 14336), dtype=torch.bfloat16)
        rowbuffer = mock.Mock(return_value=("q", "s"))
        with (
            mock.patch.object(
                turing_ops, "is_supported_tensor_core_device", return_value=True
            ),
            mock.patch.object(
                kitchen_cuda,
                "quantize_int4_rowwise_convrot64",
                return_value=("q", "s"),
            ) as fused,
            mock.patch.object(kitchen_cuda, "rotate_int8_convrot_weight") as rotate,
            mock.patch.dict(
                sys.modules,
                {"comfyui_turing_utils_kernel": SimpleNamespace(
                    turing_bf16_int4_convrot_quantize=rowbuffer
                )},
            ),
        ):
            result = turing_ops._quantize_turing_int4_activation(x, 256)
        self.assertEqual(result, ("q", "s"))
        rowbuffer.assert_called_once_with(x, 256, swiglu=False)
        fused.assert_not_called()
        rotate.assert_not_called()

    def test_w4a4_linear_uses_int4_staged_helper(self):
        x = torch.empty((2, 512), dtype=torch.bfloat16)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        qactivation = torch.empty((2, 128), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        output = torch.zeros((2, 8), dtype=torch.bfloat16)
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=True),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int4_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.object(kitchen_cuda, "int4_linear", return_value=output) as linear,
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int4",
                input_act="swiglu",
            )
        self.assertTrue(torch.equal(result, output))
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.args[1], 256)
        self.assertEqual(quantize.call_args.kwargs["input_act"], "swiglu")
        linear.assert_called_once()

    def test_w4_paths_fuse_gelu_into_quantization(self):
        x = torch.linspace(-1, 1, 512, dtype=torch.bfloat16).reshape(2, 256)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        qactivation = torch.empty((2, 128), dtype=torch.int8)
        activation_scale = torch.ones((2, 1), dtype=torch.float32)
        output = torch.zeros((2, 8), dtype=torch.bfloat16)
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=True),
            mock.patch.object(
                turing_ops,
                "_quantize_turing_int4_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.object(kitchen_cuda, "int4_linear", return_value=output),
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int4",
                input_act="gelu_tanh",
            )

        self.assertTrue(torch.equal(result, output))
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.kwargs["input_act"], "gelu_tanh")

    def test_w4a4_swiglu_uses_bf16_rowbuffer_when_it_fits(self):
        x = torch.empty((3, 28672), dtype=torch.bfloat16)
        rowbuffer = mock.Mock(return_value=("q", "s"))
        staged = mock.Mock()
        with (
            mock.patch.object(
                turing_ops, "is_supported_tensor_core_device", return_value=True
            ),
            mock.patch.dict(
                sys.modules,
                {"comfyui_turing_utils_kernel": SimpleNamespace(
                    turing_bf16_int4_convrot_quantize=rowbuffer,
                    turing_swiglu_int4_convrot_quantize=staged,
                )},
            ),
        ):
            result = turing_ops._quantize_turing_int4_activation(
                x, 256, input_act="swiglu"
            )

        self.assertEqual(result, ("q", "s"))
        rowbuffer.assert_called_once_with(x, 256, swiglu=True)
        staged.assert_not_called()

    def test_unsupported_w8a8_delegates_to_kitchen(self):
        x = torch.empty((2, 256), dtype=torch.bfloat16)
        weight = torch.empty((8, 256), dtype=torch.int8)
        weight_scale = torch.ones((), dtype=torch.float32)
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=False),
            mock.patch.object(kitchen_cuda, "int8_linear", return_value="official") as official,
        ):
            result = turing_ops.int8_linear(
                x,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
            )
        self.assertEqual(result, "official")
        official.assert_called_once()

    def test_unsupported_w4a4_delegates_to_kitchen(self):
        x = torch.empty((2, 256), dtype=torch.bfloat16)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        with (
            mock.patch.object(turing_ops, "is_supported_tensor_core_device", return_value=False),
            mock.patch.object(
                kitchen_cuda,
                "convrot_w4a4_linear",
                return_value="official",
            ) as official,
        ):
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=64,
                linear_dtype="int4",
            )
        self.assertEqual(result, "official")
        official.assert_called_once()

    def test_nonstandard_w4a8_group_size_delegates_to_kitchen(self):
        x = torch.empty((1, 256), dtype=torch.bfloat16)
        qweight = torch.empty((8, 128), dtype=torch.int8)
        wscales = torch.ones(8, dtype=torch.float32)
        with mock.patch.object(
            kitchen_cuda,
            "convrot_w4a4_linear",
            return_value="official",
        ) as official:
            result = turing_ops.convrot_w4a4_linear(
                x,
                qweight,
                wscales,
                convrot_groupsize=256,
                quant_group_size=128,
                linear_dtype="int8",
            )
        self.assertEqual(result, "official")
        official.assert_called_once_with(
            x,
            qweight,
            wscales,
            bias=None,
            convrot_groupsize=256,
            quant_group_size=128,
            linear_dtype="int8",
        )


if __name__ == "__main__":
    unittest.main()
