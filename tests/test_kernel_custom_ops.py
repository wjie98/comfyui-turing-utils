import sys
from pathlib import Path
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))

import comfyui_turing_utils_kernel as kernel  # noqa: E402


class KernelCustomOpContractTest(unittest.TestCase):
    def test_overlap_accumulate_fake_contract_is_a_mutating_fullgraph_leaf(self):
        values = torch.empty((2, 3, 11, 64), dtype=torch.float16, device="meta")
        local_indices = torch.empty((19, 3), dtype=torch.int32, device="meta")
        weights = torch.empty((19, 3), dtype=torch.float32, device="meta")
        output_indices = torch.empty((19,), dtype=torch.int32, device="meta")
        output = torch.empty((2, 31, 64), dtype=torch.float32, device="meta")
        compiled = torch.compile(
            lambda value, local, weight, indices, destination: (
                kernel.turing_sage.overlap_accumulate_compiled(
                    value,
                    local,
                    weight,
                    indices,
                    destination,
                )
            ),
            backend="eager",
            fullgraph=True,
        )
        actual = compiled(
            values,
            local_indices,
            weights,
            output_indices,
            output,
        )
        self.assertEqual(actual.shape, output.shape)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "meta")

    def test_overlap_blend_fake_contract_is_a_fullgraph_leaf(self):
        values = torch.empty((2, 5, 17, 64), dtype=torch.float16, device="meta")
        local_indices = torch.empty((31, 5), dtype=torch.int32, device="meta")
        weights = torch.empty((31, 5), dtype=torch.float32, device="meta")
        compiled = torch.compile(
            lambda value, local, weight: kernel.turing_sage.overlap_blend_compiled(
                value, local, weight
            ),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(values, local_indices, weights)
        self.assertEqual(output.shape, (2, 31, 64))
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.device.type, "meta")

    @unittest.skipUnless(
        torch.cuda.is_available() and kernel.turing_sage.overlap_blend_available(),
        "the rebuilt CUDA overlap epilogue is required",
    )
    def test_overlap_blend_matches_ordered_fp32_reference(self):
        torch.manual_seed(47)
        batch, windows, tokens, channels, global_tokens = 2, 5, 17, 96, 31
        local_indices = torch.full(
            (global_tokens, windows), -1, dtype=torch.int32, device="cuda"
        )
        weights = torch.zeros(
            (global_tokens, windows), dtype=torch.float32, device="cuda"
        )
        for global_index in range(global_tokens):
            owners = [global_index % windows, (global_index + 2) % windows]
            local_indices[global_index, owners[0]] = global_index % tokens
            local_indices[global_index, owners[1]] = (global_index * 3) % tokens
            weights[global_index, owners[0]] = 0.35
            weights[global_index, owners[1]] = 0.65

        for dtype, tolerance in (
            (torch.float16, 2**-10),
            (torch.bfloat16, 2**-7),
        ):
            base = torch.randn(
                batch,
                windows,
                tokens + 3,
                channels,
                dtype=dtype,
                device="cuda",
            )
            values = base[:, :, :tokens]
            expected = torch.zeros(
                batch, global_tokens, channels, dtype=torch.float32, device="cuda"
            )
            for global_index in range(global_tokens):
                for window in range(windows):
                    local = int(local_indices[global_index, window])
                    if local >= 0:
                        expected[:, global_index].add_(
                            values[:, window, local].float()
                            * weights[global_index, window]
                        )
            expected = expected.to(dtype)
            first = kernel.turing_sage.overlap_blend_compiled(
                values, local_indices, weights
            )
            second = kernel.turing_sage.overlap_blend_compiled(
                values, local_indices, weights
            )
            torch.testing.assert_close(first, expected, rtol=0.0, atol=tolerance)
            self.assertTrue(torch.equal(first, second))

    @unittest.skipUnless(
        torch.cuda.is_available() and kernel.turing_sage.overlap_accumulate_available(),
        "the rebuilt CUDA streaming overlap accumulator is required",
    )
    def test_overlap_accumulate_matches_sequential_fp32_reference(self):
        torch.manual_seed(53)
        batch, tokens, channels, output_tokens = 2, 13, 80, 37
        for dtype in (torch.float16, torch.bfloat16):
            destination = torch.randn(
                batch,
                output_tokens,
                channels,
                dtype=torch.float32,
                device="cuda",
            )
            expected = destination.clone()
            payloads = []
            for windows, affected_tokens in ((3, 17), (2, 11)):
                values = torch.randn(
                    batch,
                    windows,
                    tokens,
                    channels,
                    dtype=dtype,
                    device="cuda",
                )
                output_indices = (
                    torch.randperm(
                        output_tokens,
                        device="cuda",
                        dtype=torch.int64,
                    )[:affected_tokens]
                    .sort()
                    .values.to(torch.int32)
                )
                local_indices = torch.full(
                    (affected_tokens, windows),
                    -1,
                    dtype=torch.int32,
                    device="cuda",
                )
                weights = torch.zeros(
                    (affected_tokens, windows),
                    dtype=torch.float32,
                    device="cuda",
                )
                for row in range(affected_tokens):
                    for window in range(windows):
                        if (row + window) % 3:
                            local_indices[row, window] = (row * 5 + window) % tokens
                            weights[row, window] = 0.2 + 0.1 * window
                for row in range(affected_tokens):
                    output_index = int(output_indices[row])
                    for window in range(windows):
                        local = int(local_indices[row, window])
                        if local >= 0:
                            expected[:, output_index].add_(
                                values[:, window, local].float() * weights[row, window]
                            )
                payloads.append((values, local_indices, weights, output_indices))

            results = []
            for _repeat in range(2):
                actual = destination.clone()
                for values, local_indices, weights, output_indices in payloads:
                    returned = kernel.turing_sage.overlap_accumulate_compiled(
                        values,
                        local_indices,
                        weights,
                        output_indices,
                        actual,
                    )
                    self.assertIs(returned, actual)
                results.append(actual)
            torch.testing.assert_close(results[0], expected, rtol=0.0, atol=0.0)
            self.assertTrue(torch.equal(results[0], results[1]))

    def test_convrot_fake_contracts(self):
        x = torch.empty((3, 512), dtype=torch.bfloat16, device="meta")
        expectations = {
            "turing_swiglu_int8_convrot_quantize": (3, 256),
            "turing_swiglu_int4_convrot_quantize": (3, 128),
            "turing_gelu_int8_convrot_quantize": (3, 512),
            "turing_gelu_int4_convrot_quantize": (3, 256),
            "turing_bf16_gelu_int8_convrot_quantize": (3, 512),
            "turing_bf16_gelu_int4_convrot_quantize": (3, 256),
        }
        for name, shape in expectations.items():
            quantized, scale = getattr(kernel, name)(x)
            self.assertEqual(quantized.shape, shape)
            self.assertEqual(quantized.dtype, torch.int8)
            self.assertEqual(scale.shape, (3, 1))
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(scale.device.type, "meta")

        for bits, shape in ((8, (3, 256)), (4, (3, 128))):
            quantized, scale = getattr(
                kernel, f"turing_bf16_int{bits}_convrot_quantize"
            )(x, swiglu=True)
            self.assertEqual(quantized.shape, shape)
            self.assertEqual(scale.shape, (3, 1))

        scaled = kernel.turing_swiglu_int8_convrot_quantize_scaled(
            x,
            torch.empty((3,), dtype=torch.float32, device="meta"),
        )
        self.assertEqual(scaled.shape, (3, 256))
        self.assertEqual(scaled.dtype, torch.int8)
        self.assertEqual(scaled.device.type, "meta")
        scaled_destination = torch.empty((3, 256), dtype=torch.int8, device="meta")
        self.assertIsNone(
            kernel.turing_swiglu_int8_convrot_quantize_scaled_out(
                x,
                torch.empty((3,), dtype=torch.float32, device="meta"),
                scaled_destination,
            )
        )
        rotated_gate = torch.empty((3, 512), dtype=torch.bfloat16, device="meta")
        partials = torch.empty((3, 2), dtype=torch.float32, device="meta")
        self.assertIsNone(
            kernel.turing_swiglu_convrot_shard_inplace(
                rotated_gate,
                torch.empty((3, 256), dtype=torch.bfloat16, device="meta"),
                partials,
                256,
            )
        )
        sharded, sharded_scale = kernel.turing_int8_convrot_quantize_from_partials(
            rotated_gate, partials
        )
        self.assertEqual(sharded.shape, rotated_gate.shape)
        self.assertEqual(sharded.dtype, torch.int8)
        self.assertEqual(sharded_scale.shape, (3, 1))

    def test_linear_epilogue_and_norm_fake_contracts(self):
        activation = torch.empty((7, 128), dtype=torch.int8, device="meta")
        weight = torch.empty((64, 64), dtype=torch.int8, device="meta")
        row_scale = torch.empty((7,), dtype=torch.float32, device="meta")
        column_scale = torch.empty((64,), dtype=torch.float32, device="meta")
        linear = kernel.turing_w4a8_linear(activation, weight, row_scale, column_scale)
        self.assertEqual(linear.shape, (7, 64))
        self.assertEqual(linear.dtype, torch.bfloat16)

        codebook_linear = kernel.turing_codebook_w4a8_linear(
            activation,
            weight,
            row_scale,
            torch.empty((64, 8), dtype=torch.uint8, device="meta"),
            column_scale,
            torch.empty((16,), dtype=torch.float32, device="meta"),
        )
        self.assertEqual(codebook_linear.shape, (7, 64))
        self.assertEqual(codebook_linear.dtype, torch.bfloat16)

        long_codebook_linear = kernel.turing_codebook_w4a8_linear(
            torch.empty((9000, 128), dtype=torch.int8, device="meta"),
            weight,
            torch.empty((9000,), dtype=torch.float32, device="meta"),
            torch.empty((64, 8), dtype=torch.uint8, device="meta"),
            column_scale,
            torch.empty((16,), dtype=torch.float32, device="meta"),
            chunk_rows=-1,
        )
        self.assertEqual(long_codebook_linear.shape, (9000, 64))

        int8_linear = kernel.turing_int8_linear(
            activation,
            torch.empty((64, 128), dtype=torch.int8, device="meta"),
            row_scale,
            column_scale,
        )
        self.assertEqual(int8_linear.shape, (7, 64))
        self.assertEqual(int8_linear.dtype, torch.bfloat16)

        direct_linear = torch.empty((7, 64), dtype=torch.bfloat16, device="meta")
        self.assertIsNone(
            kernel.turing_int8_linear_out(
                activation,
                torch.empty((64, 128), dtype=torch.int8, device="meta"),
                row_scale,
                column_scale,
                direct_linear,
            )
        )

        accumulator = torch.empty((7, 80), dtype=torch.int32, device="meta")
        epilogue = kernel.turing_dequantize_int8_bf16(
            accumulator, row_scale, column_scale, 64
        )
        self.assertEqual(epilogue.shape, (7, 64))
        self.assertEqual(epilogue.dtype, torch.bfloat16)

        x2 = torch.empty((7, 128), dtype=torch.bfloat16, device="meta")
        segments = torch.empty((2, 3), dtype=torch.int32, device="meta")
        normalized = kernel.turing_segmented_rms_adaln(
            x2,
            torch.empty((128,), dtype=x2.dtype, device="meta"),
            torch.empty((2, 128), dtype=x2.dtype, device="meta"),
            torch.empty((2, 128), dtype=x2.dtype, device="meta"),
            segments,
        )
        self.assertEqual(normalized.shape, x2.shape)
        self.assertEqual(normalized.dtype, x2.dtype)
        residual = torch.empty_like(x2)
        gate = torch.empty((2, 128), dtype=x2.dtype, device="meta")
        self.assertIsNone(
            kernel.turing_segmented_mod_gate(x2, gate, residual, segments)
        )
        fused_normalized = kernel.turing_segmented_mod_gate_rms_adaln(
            x2,
            gate,
            residual,
            torch.empty((128,), dtype=x2.dtype, device="meta"),
            torch.empty((2, 128), dtype=x2.dtype, device="meta"),
            torch.empty((2, 128), dtype=x2.dtype, device="meta"),
            segments,
        )
        self.assertEqual(fused_normalized.shape, x2.shape)
        self.assertEqual(fused_normalized.dtype, x2.dtype)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5),
        "CUDA Tensor Cores required",
    )
    def test_codebook_float_scales_inline_and_staged(self):
        # Independent integer oracle, non-full N tile, biased output, and
        # multiple chunks. Both scale dtypes must retain identical semantics.
        torch.manual_seed(803)
        m, n, k = 8193, 80, 96
        qx = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
        packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.int8)
        book = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        sx = torch.full((m,), 0.001, device="cuda")
        sw = torch.full((n,), 0.002, device="cuda")
        bias = torch.randn(n, device="cuda") * 0.01
        for dtype in (torch.float32, torch.float8_e4m3fn):
            scales = (torch.rand(n, k // 16, device="cuda") * 30).to(dtype)
            w8 = (
                (book[codes.long()] * scales.float().repeat_interleave(16, 1))
                .round()
                .clamp(-127, 127)
            )
            reference = (qx.float() @ w8.t()) * sx[:, None] * sw[None, :] + bias
            for chunk in (-1, 32):
                actual = kernel.turing_codebook_w4a8_linear(
                    qx,
                    packed,
                    sx,
                    scales,
                    sw,
                    book,
                    bias,
                    chunk_rows=chunk,
                )
                torch.testing.assert_close(
                    actual.float(), reference, rtol=0.004, atol=2e-6
                )

    def test_sol_and_varlen_are_fullgraph_leaves(self):
        q = torch.empty((1, 4, 129, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 151, 128), dtype=torch.bfloat16, device="meta")
        compiled = torch.compile(
            lambda query, key, value: kernel.turing_sage.sol_sparse_sageattn_compiled(
                query,
                key,
                value,
                dense_query_ranges=((0, 64),),
                exact_kv_ranges=((0, 64),),
            ),
            backend="eager",
            fullgraph=True,
        )
        self.assertEqual(compiled(q, k, k).shape, q.shape)

        qv = torch.empty((129, 4, 128), dtype=q.dtype, device="meta")
        kv = torch.empty((151, 2, 128), dtype=q.dtype, device="meta")
        cu = torch.empty((2,), dtype=torch.int32, device="meta")
        output = kernel.turing_sage.sageattn_varlen_compiled(
            qv, kv, kv, cu, cu, 129, 151
        )
        self.assertEqual(output.shape, qv.shape)
        w8a8_output = kernel.turing_sage.w8a8attn_varlen_compiled(
            qv, kv, kv, cu, cu, 129, 151, is_causal=True
        )
        self.assertEqual(w8a8_output.shape, qv.shape)


if __name__ == "__main__":
    unittest.main()
