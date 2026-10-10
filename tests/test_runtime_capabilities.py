from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils import hardware  # noqa: E402
from comfyui_turing_utils.runtime import capabilities  # noqa: E402


class DeviceCapabilitiesTest(unittest.TestCase):
    def test_ampere_reports_real_optin_shared_memory(self):
        properties = SimpleNamespace(
            name="NVIDIA GeForce RTX 3070",
            total_memory=16 * 1024**3,
            multi_processor_count=46,
            shared_memory_per_block=48 * 1024,
            shared_memory_per_block_optin=99 * 1024,
            shared_memory_per_multiprocessor=100 * 1024,
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ),
            mock.patch.object(
                torch.cuda, "get_device_name", return_value=properties.name
            ),
        ):
            result = hardware.device_capabilities(torch.device("cuda", 0))

        self.assertEqual(result.architecture, "sm86")
        self.assertTrue(result.tensor_core)
        self.assertTrue(result.native_bf16)
        self.assertEqual(result.multiprocessor_count, 46)
        self.assertEqual(result.optin_shared_memory_per_block, 99 * 1024)

    def test_low_end_turing_name_remains_excluded(self):
        properties = SimpleNamespace(
            name="NVIDIA GeForce GTX 1650", total_memory=4 * 1024**3
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)),
            mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ),
            mock.patch.object(
                torch.cuda, "get_device_name", return_value=properties.name
            ),
        ):
            result = hardware.device_capabilities(torch.device("cuda", 0))
            supported = hardware.is_supported_turing_device(torch.device("cuda", 0))

        self.assertFalse(result.tensor_core)
        self.assertFalse(supported)


class KernelCapabilitiesTest(unittest.TestCase):
    def test_operator_features_are_versioned_independently(self):
        package = SimpleNamespace(
            __version__="0.30.0",
            turing_segmented_rms_adaln=lambda *args: None,
        )
        extension = SimpleNamespace(
            turing_segmented_rms_adaln=lambda *args: None,
            turing_segmented_mod_gate=lambda *args: None,
            turing_segmented_mod_gate_rms_adaln=lambda *args: None,
            turing_swiglu_int8_convrot_quantize_scaled=lambda *args: None,
            turing_swiglu_convrot_shard_inplace=lambda *args: None,
            turing_int8_convrot_quantize_from_partials=lambda *args: None,
        )
        sage = SimpleNamespace(
            available=lambda: True,
            sparse_available=lambda: True,
            sla_available=lambda: True,
            w8a8_available=lambda: True,
            split_prequantization_available=lambda: True,
            fused_qk_preprocessing_available=lambda: True,
            overlap_accumulate_available=lambda: False,
            precompute_rms_rope_k_anchor=lambda *args: None,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": package,
                "comfyui_turing_utils_kernel._C": extension,
                "comfyui_turing_utils_kernel.turing_sage": sage,
            },
        ):
            result = capabilities.kernel_capabilities()

        self.assertTrue(result.supports("core_fusions").supported)
        self.assertTrue(result.supports("ffn_channel_sharding").supported)
        self.assertTrue(result.supports("ffn_half_width").supported)
        self.assertTrue(result.supports("sla").supported)
        self.assertTrue(result.supports("reusable_k_anchor").supported)
        self.assertFalse(result.supports("overlap_accumulate").supported)

    def test_python_wrapper_does_not_mask_a_stale_extension(self):
        package = SimpleNamespace(
            __version__="0.30.0",
            turing_segmented_rms_adaln=lambda *args: None,
            turing_swiglu_int8_convrot_quantize_scaled=lambda *args: None,
        )
        extension = SimpleNamespace(
            turing_segmented_rms_adaln=lambda *args: None,
        )
        sage = SimpleNamespace(available=lambda: False)
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": package,
                "comfyui_turing_utils_kernel._C": extension,
                "comfyui_turing_utils_kernel.turing_sage": sage,
            },
        ):
            result = capabilities.kernel_capabilities()

        self.assertFalse(result.supports("core_fusions").supported)
        self.assertFalse(result.supports("ffn_channel_sharding").supported)
        self.assertFalse(result.supports("ffn_half_width").supported)

    def test_asymmetric_qk_rope_requires_native_schema_two(self):
        package = SimpleNamespace(__version__="0.38.0")
        sage = SimpleNamespace(
            fused_qk_preprocessing_available=lambda: True,
        )
        for schema, expected in ((0, False), (2, True)):
            with (
                self.subTest(schema=schema),
                mock.patch.dict(
                    sys.modules,
                    {
                        "comfyui_turing_utils_kernel": package,
                        "comfyui_turing_utils_kernel._sage_fused_sm75": SimpleNamespace(
                            qk_preprocess_protocol_schema=schema
                        ),
                        "comfyui_turing_utils_kernel.turing_sage": sage,
                    },
                ),
            ):
                result = capabilities.kernel_capabilities()
                self.assertEqual(
                    result.supports("asymmetric_qk_rope").supported,
                    expected,
                )

    def test_mapped_kv_requires_schema_three_and_both_native_symbols(self):
        package = SimpleNamespace(__version__="0.41.0")
        sage = SimpleNamespace(fused_qk_preprocessing_available=lambda: True)
        fused = SimpleNamespace(
            qk_preprocess_protocol_schema=3,
            quant_qk_rms_rope_int8_mapped_cuda=lambda *args: None,
            gather_value_int8_mapped_cuda=lambda *args: None,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": package,
                "comfyui_turing_utils_kernel._sage_fused_sm75": fused,
                "comfyui_turing_utils_kernel.turing_sage": sage,
            },
        ):
            result = capabilities.kernel_capabilities()

        self.assertTrue(result.supports("mapped_kv").supported)

    def test_mapped_sparse_kv_requires_native_summary_symbol(self):
        package = SimpleNamespace(__version__="0.41.0")
        sage = SimpleNamespace(fused_qk_preprocessing_available=lambda: True)
        fused = SimpleNamespace(
            qk_preprocess_protocol_schema=3,
            quant_qk_rms_rope_int8_mapped_cuda=lambda *args: None,
            gather_value_int8_mapped_cuda=lambda *args: None,
        )
        qattn = SimpleNamespace(
            sol_w8a8_precompute_mapped_summaries=lambda *args: None,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": package,
                "comfyui_turing_utils_kernel._sage_fused_sm75": fused,
                "comfyui_turing_utils_kernel._sage_qattn_sm75": qattn,
                "comfyui_turing_utils_kernel.turing_sage": sage,
            },
        ):
            result = capabilities.kernel_capabilities()

        self.assertTrue(result.supports("mapped_sparse_kv").supported)

    def test_mapped_sparse_fp16_kv_requires_qk_map_and_native_attention(self):
        package = SimpleNamespace(__version__="0.41.0")
        sage = SimpleNamespace(fused_qk_preprocessing_available=lambda: True)
        fused = SimpleNamespace(
            qk_preprocess_protocol_schema=3,
            quant_qk_rms_rope_int8_mapped_cuda=lambda *args: None,
        )
        qattn = SimpleNamespace(
            sol_sparse_online_int8_f16_mapped_attn=lambda *args: None,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "comfyui_turing_utils_kernel": package,
                "comfyui_turing_utils_kernel._sage_fused_sm75": fused,
                "comfyui_turing_utils_kernel._sage_qattn_sm75": qattn,
                "comfyui_turing_utils_kernel.turing_sage": sage,
            },
        ):
            result = capabilities.kernel_capabilities()

        self.assertTrue(result.supports("mapped_sparse_fp16_kv").supported)
        self.assertFalse(result.supports("mapped_sparse_kv").supported)

    def test_missing_kernel_package_has_actionable_reason(self):
        with mock.patch(
            "comfyui_turing_utils.runtime.capabilities.load_kernel_package",
            side_effect=ImportError("not installed"),
        ):
            result = capabilities.kernel_capabilities()

        self.assertFalse(result.installed)
        self.assertIn("not installed", result.supports("sol").reason)


if __name__ == "__main__":
    unittest.main()
