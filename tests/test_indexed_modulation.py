from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT.parents[1]))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax import acceleration
from comfyui_turing_utils import kernel_api
from comfyui_turing_utils.kernel_api import (
    load_kernel_package,
    segmented_modulation_schema,
)
from comfyui_turing_utils.quantization.fusions import indexed_modulation_rows


class IndexedModulationMappingTest(unittest.TestCase):
    def test_schema_comes_from_native_binary_not_package_version(self):
        for extension, expected in (
            (SimpleNamespace(), 0),
            (SimpleNamespace(segmented_modulation_schema=2), 2),
        ):
            with (
                self.subTest(expected=expected),
                mock.patch.object(
                    kernel_api, "load_kernel_extension", return_value=extension
                ),
            ):
                self.assertEqual(segmented_modulation_schema(), expected)
        with mock.patch.object(
            kernel_api, "load_kernel_extension", side_effect=ImportError
        ):
            self.assertEqual(segmented_modulation_schema(), 0)

    def test_mixed_indices_broadcast_and_noncontiguous_rows(self):
        indices = torch.tensor([0, 9, 2, 9, 1, 9])[::2]
        segments = [(0, 2, 1), (2, 5, indices), (5, 7, torch.tensor(2))]
        packed = indexed_modulation_rows(segments, 7, 3, torch.device("cpu"))
        self.assertEqual(packed.tolist(), [1, 1, 0, 2, 1, 2, 2])
        self.assertEqual(packed.dtype, torch.int64)
        indices.fill_(0)
        self.assertEqual(packed.tolist(), [1, 1, 0, 2, 1, 2, 2])
        updated = indexed_modulation_rows(segments, 7, 3, torch.device("cpu"))
        self.assertEqual(updated.tolist(), [1, 1, 0, 0, 0, 2, 2])

    def test_invalid_shapes_and_types_are_not_silently_cast(self):
        for row in (
            torch.ones(3),
            torch.ones(2, 2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                indexed_modulation_rows([(0, 3, row)], 3, 3, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "contiguously"):
            indexed_modulation_rows(
                [(1, 3, torch.ones(2, dtype=torch.long))], 3, 3, torch.device("cpu")
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class IndexedModulationCUDATest(unittest.TestCase):
    def setUp(self):
        self.assertGreaterEqual(
            segmented_modulation_schema(), 2, "rebuild the development kernel"
        )
        self.kernel = load_kernel_package()
        torch.manual_seed(531)

    def test_scalar_table_preserves_legacy_fused_math(self):
        table = torch.tensor([(0, 3, 1), (3, 19, 0)], device="cuda", dtype=torch.int32)
        ids = torch.tensor([1] * 3 + [0] * 16, device="cuda")
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.randn(19, 384, device="cuda", dtype=dtype)
                other = torch.randn_like(x)
                weight = torch.randn(384, device="cuda", dtype=dtype)
                scale, shift, gate = (
                    torch.randn(2, 1152, device="cuda", dtype=dtype) * 0.2
                ).chunk(3, -1)
                sequential_x = x.clone()
                self.kernel.turing_segmented_mod_gate(sequential_x, gate, other, table)
                sequential_norm = self.kernel.turing_segmented_rms_adaln(
                    sequential_x, weight, scale, shift, table
                )
                fused_x = x.clone()
                fused_norm = self.kernel.turing_segmented_mod_gate_rms_adaln(
                    fused_x, gate, other, weight, scale, shift, table
                )
                torch.testing.assert_close(fused_x, sequential_x, rtol=0, atol=0)
                torch.testing.assert_close(fused_norm, sequential_norm, rtol=0, atol=0)
                normalized = sequential_x.float() * torch.rsqrt(
                    sequential_x.float().square().mean(-1, keepdim=True) + 1e-5
                )
                reference = (
                    normalized * weight.float() * (1 + scale.float()[ids])
                    + shift.float()[ids]
                )
                tolerance = {
                    torch.float32: (2e-6, 2e-6),
                    torch.float16: (0.001, 0.001),
                    torch.bfloat16: (0.008, 0.008),
                }[dtype]
                torch.testing.assert_close(
                    fused_norm.float(), reference, rtol=tolerance[0], atol=tolerance[1]
                )

    def test_three_ops_match_native_modulation(self):
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for hidden in (127, 128, 5376):
                with self.subTest(dtype=dtype, hidden=hidden):
                    x = torch.randn(29, hidden, device="cuda", dtype=dtype)
                    other = torch.randn_like(x)
                    weight = torch.randn(hidden, device="cuda", dtype=dtype)
                    # AdaLN's chunked outputs have a row stride larger than hidden.
                    scale, shift, gate = (
                        torch.randn(9, hidden * 3, device="cuda", dtype=dtype) * 0.2
                    ).chunk(3, -1)
                    ids = torch.tensor([0, 3, 6, -1, 3, 0] * 4, device="cuda")
                    segments = [(0, 5, 1), (5, 29, ids)]
                    rows = indexed_modulation_rows(segments, 29, 9, x.device)
                    expected_norm = _mod_scale_shift(
                        torch.nn.functional.rms_norm(x, (hidden,), weight, 1e-5),
                        shift,
                        scale,
                        segments,
                    )
                    actual_norm = self.kernel.turing_segmented_rms_adaln(
                        x, weight, scale, shift, rows, 1e-5
                    )
                    tolerance = {
                        torch.float32: (2e-6, 2e-6),
                        torch.float16: (0.002, 0.002),
                        torch.bfloat16: (0.016, 0.016),
                    }[dtype]
                    torch.testing.assert_close(
                        actual_norm, expected_norm, rtol=tolerance[0], atol=tolerance[1]
                    )
                    error = actual_norm.float() - expected_norm.float()
                    relative_rms = (
                        (error.square().mean() / expected_norm.float().square().mean())
                        .sqrt()
                        .item()
                    )
                    self.assertLess(
                        relative_rms,
                        {
                            torch.float32: 1e-6,
                            torch.float16: 4e-5,
                            torch.bfloat16: 3e-4,
                        }[dtype],
                    )
                    expected_x = _mod_gate(x.clone(), gate, other, segments)
                    actual_x = x.clone()
                    self.kernel.turing_segmented_mod_gate(actual_x, gate, other, rows)
                    torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
                    fused_x = x.clone()
                    fused_norm = self.kernel.turing_segmented_mod_gate_rms_adaln(
                        fused_x,
                        gate,
                        other,
                        weight,
                        scale,
                        shift,
                        rows,
                        1e-5,
                    )
                    torch.testing.assert_close(fused_x, expected_x, rtol=0, atol=0)
                    expected_fused_norm = _mod_scale_shift(
                        torch.nn.functional.rms_norm(
                            expected_x, (hidden,), weight, 1e-5
                        ),
                        shift,
                        scale,
                        segments,
                    )
                    torch.testing.assert_close(
                        fused_norm,
                        expected_fused_norm,
                        rtol=tolerance[0],
                        atol=tolerance[1],
                    )

    def test_rejects_mismatched_native_index_shape(self):
        x = torch.ones(4, 128, device="cuda")
        params = torch.ones(3, 128, device="cuda")
        weight = torch.ones(128, device="cuda")
        for indices in (
            torch.zeros(3, device="cuda", dtype=torch.long),
            torch.zeros(4, 2, device="cuda", dtype=torch.int32),
        ):
            with self.subTest(shape=indices.shape):
                for op in (
                    lambda: self.kernel.turing_segmented_rms_adaln(
                        x, weight, params, params, indices
                    ),
                    lambda: self.kernel.turing_segmented_mod_gate(
                        x, params, x.clone(), indices
                    ),
                    lambda: self.kernel.turing_segmented_mod_gate_rms_adaln(
                        x, params, x.clone(), weight, params, params, indices
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "per-token indices"):
                        op()

    def test_masked_block_uses_all_fusions_and_one_mapping(self):
        import comfy.model_management
        import comfy.ops
        from comfy.ldm.minimax.model import DiTBlock, _mod_gate

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                block = DiTBlock(
                    hidden=128,
                    heads=1,
                    head_dim=128,
                    ffn=64,
                    t_dim=32,
                    eps=1e-5,
                    qk_eps=1e-6,
                    dtype=dtype,
                    device="cuda",
                    operations=SimpleNamespace(
                        Linear=torch.nn.Linear,
                        RMSNorm=comfy.ops.disable_weight_init.RMSNorm,
                    ),
                )
                with torch.no_grad():
                    for module in block.modules():
                        if isinstance(module, torch.nn.RMSNorm):
                            module.weight.fill_(1.0)
                x = torch.randn(27, 128, device="cuda", dtype=dtype)
                t_emb = torch.randn(3, 32, device="cuda", dtype=dtype)
                patterns = {
                    "video_binary": (2, torch.tensor([0, 3] * 8, device="cuda")),
                    "audio_soft": (torch.tensor([2, 5, 8, 2] * 2, device="cuda"), 0),
                    "both_soft": (
                        torch.tensor([2, 5, 8, 2] * 2, device="cuda"),
                        torch.tensor([0, 3, 6, 3] * 4, device="cuda"),
                    ),
                    "constant": (2, torch.zeros(16, device="cuda", dtype=torch.long)),
                }
                audit = mock.Mock()
                patched = acceleration._make_block_forward(block, 0, _mod_gate, audit)
                for name, (audio, video) in patterns.items():
                    with (
                        self.subTest(mask=name),
                        torch.no_grad(),
                        mock.patch.object(comfy.model_management, "in_training", True),
                    ):
                        segments = [(0, 3, 1), (3, 11, audio), (11, 27, video)]
                        options = {}
                        attention = mock.Mock(wraps=block.attn.forward)
                        expected = block(
                            x.clone(),
                            t_emb,
                            segments,
                            None,
                            transformer_options=options,
                            attention=attention,
                        )
                        attention.reset_mock()
                        with (
                            mock.patch.object(
                                acceleration,
                                "indexed_modulation_rows",
                                wraps=indexed_modulation_rows,
                            ) as pack,
                            mock.patch.object(
                                acceleration,
                                "segmented_rms_adaln",
                                wraps=acceleration.segmented_rms_adaln,
                            ) as norm,
                            mock.patch.object(
                                acceleration,
                                "segmented_mod_gate_rms_adaln",
                                wraps=acceleration.segmented_mod_gate_rms_adaln,
                            ) as gate_norm,
                            mock.patch.object(
                                acceleration,
                                "segmented_mod_gate",
                                wraps=acceleration.segmented_mod_gate,
                            ) as gate,
                        ):
                            actual = patched(
                                x.clone(),
                                t_emb,
                                segments,
                                None,
                                transformer_options=options,
                                attention=attention,
                            )
                        pack.assert_called_once()
                        norm.assert_called_once()
                        gate_norm.assert_called_once()
                        gate.assert_called_once()
                        self.assertIs(
                            norm.call_args.args[-1], gate_norm.call_args.args[-1]
                        )
                        self.assertIs(norm.call_args.args[-1], gate.call_args.args[-1])
                        attention.assert_called_once()
                        self.assertTrue(audit.record.call_args.args[1])
                        tolerance = {
                            torch.float32: (3e-6, 3e-6),
                            torch.float16: (0.003, 0.003),
                            torch.bfloat16: (0.02, 0.02),
                        }[dtype]
                        torch.testing.assert_close(
                            actual, expected, rtol=tolerance[0], atol=tolerance[1]
                        )
