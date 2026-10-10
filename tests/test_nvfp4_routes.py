import unittest
from unittest import mock
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from comfy_kitchen.backends import cuda as ck
from comfyui_turing_utils_kernel.ops import (
    turing_nvfp4_convrot_quantize,
    turing_nvfp4_convrot_quantize_out,
    turing_int8_linear_out,
)
from comfyui_turing_utils_kernel.nvfp4 import (
    linear,
    quantize_activation,
    _linear_pipeline,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NVFP4RoutesTest(unittest.TestCase):
    def test_automatic_pipeline_and_capture_serial_fallback(self):
        if torch.cuda.get_device_capability() != (8, 6):
            self.skipTest("automatic pipeline is validated for SM86")
        from comfyui_turing_utils_kernel import nvfp4

        k, n, m = 8192, 4097, 8192
        w = torch.randn(4112, k, device="cuda", dtype=torch.bfloat16) * 0.02
        s = w.float().abs().max() / (448 * 6)
        p, b = ck.quantize_nvfp4(w, s)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(n, device="cuda", dtype=x.dtype)
        with mock.patch.object(
            nvfp4, "_linear_pipeline", wraps=nvfp4._linear_pipeline
        ) as pipeline:
            expected = linear(x, p, b, s, bias, output_columns=n)
            pipeline.assert_called_once()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = linear(x, p, b, s, bias, output_columns=n)
            pipeline.assert_called_once()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_weight_conversion_out_buffers(self):
        for k in (17, 5376, 16385):
            w = torch.randn(
                144, (k + 15) // 16 * 16, device="cuda", dtype=torch.bfloat16
            )
            s = w.float().abs().max() / (448 * 6)
            p, b = ck.quantize_nvfp4(w, s)
            q, scales = turing_nvfp4_convrot_quantize(p, b, s, 3, 19, k)
            out, os = torch.empty_like(q), torch.empty_like(scales)
            turing_nvfp4_convrot_quantize_out(p, b, s, out, os, 3, k)
            torch.testing.assert_close(out, q, rtol=0, atol=0)
            torch.testing.assert_close(os, scales, rtol=0, atol=0)
            if k == 17:
                torch.library.opcheck(
                    turing_nvfp4_convrot_quantize_out,
                    (p, b.view(torch.uint8), s, out, os, 3, k),
                )
                with self.assertRaisesRegex(RuntimeError, "output buffer"):
                    turing_nvfp4_convrot_quantize_out(p, b, s, out[:, :128], os, 3, k)

    def test_pipeline_tail_lifetime_and_stream(self):
        torch.manual_seed(489)
        for k in (17, 257, 8193):
            w = torch.randn(
                144, (k + 15) // 16 * 16, device="cuda", dtype=torch.bfloat16
            )
            s = w.float().abs().max() / (448 * 6)
            p, b = ck.quantize_nvfp4(w, s)
            x = torch.randn(31, k, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(137, device="cuda", dtype=torch.bfloat16)
            expected = linear(x, p, b, s, bias, output_columns=137)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                qx, sx = quantize_activation(x)
                padded_bias = torch.nn.functional.pad(bias.float(), (0, 7))
                for _ in range(4):
                    out = torch.full((31, 192), -37.0, device="cuda", dtype=x.dtype)
                    _linear_pipeline(qx, sx, p, b, s, out, padded_bias, 32, k, 144)
                    torch.empty(
                        (32, qx.shape[1]), device="cuda", dtype=torch.int8
                    ).fill_(23)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(out[:, :137], expected, rtol=0, atol=0)
            self.assertTrue(bool((out[:, 144:] == -37).all()))

    def test_fp16_zero_and_tiny_activation(self):
        for k in (17, 256, 5376):
            for swiglu in (False, True):
                for amplitude in (0.0, 1e-4, 0.001):
                    x = torch.full(
                        (3, k * (2 if swiglu else 1)),
                        amplitude,
                        device="cuda",
                        dtype=torch.float16,
                    )
                    reference = x
                    if swiglu:
                        gate, up = x.chunk(2, dim=-1)
                        reference = torch.nn.functional.silu(gate) * up
                    q, s = quantize_activation(x, swiglu=swiglu)
                    self.assertTrue(bool(s.isfinite().all()))
                    self.assertFalse(bool((q == -128).any()))
                    restored = ck.rotate_int8_convrot_weight(
                        (q.float() * s.reshape(-1, 1)).contiguous(), 256
                    )[:, :k]
                    torch.testing.assert_close(
                        restored, reference.float(), rtol=0.02, atol=2e-7
                    )
                    if amplitude == 0:
                        self.assertEqual(int(q.count_nonzero()), 0)

    def test_long_wide_k_gemm_schedule_with_tail(self):
        torch.manual_seed(327)
        x = torch.randint(-20, 21, (8192, 8192), device="cuda", dtype=torch.int8)
        w = torch.randint(-20, 21, (264, 8192), device="cuda", dtype=torch.int8)
        sx = torch.full((8192,), 0.03125, device="cuda")
        sw = torch.full((264,), 0.015625, device="cuda")
        bias = torch.randn(264, device="cuda")
        output = torch.full((8192, 4096), -19.0, device="cuda", dtype=torch.bfloat16)
        turing_int8_linear_out(x, w, sx, sw, output[:, :264], bias)
        expected = ((x.float() @ w.float().T) * sx[:, None] * sw + bias).bfloat16()
        torch.testing.assert_close(output[:, :264], expected, rtol=0, atol=0)
        self.assertTrue(bool((output[:, 264:] == -19).all()))

    def test_swiglu_fusion_precision_and_fallbacks(self):
        torch.manual_seed(171)
        for k in (17, 256, 5376, 14336):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = torch.randn(35, k * 2, device="cuda", dtype=dtype)
                gate, up = x.float().chunk(2, dim=-1)
                ref = torch.nn.functional.silu(gate) * up
                q, s = quantize_activation(x, swiglu=True)
                restored = ck.rotate_int8_convrot_weight(
                    (q.float() * s.reshape(-1, 1)).contiguous(), 256
                )[:, :k]
                self.assertLess(float((restored - ref).norm() / ref.norm()), 0.025)
                if k == 256 and dtype == torch.bfloat16:
                    with mock.patch.object(
                        ck, "_convrot_fused_shared_memory_fits", return_value=False
                    ):
                        fallback_q, fallback_s = quantize_activation(x, swiglu=True)
                    torch.testing.assert_close(fallback_s, s, rtol=1e-5, atol=1e-7)
                    self.assertLess(float((fallback_q != q).float().mean()), 0.001)
                # Channel pre-scales do not commute with the rotation. This
                # combination must preserve SwiGLU -> scale -> rotate order.
                pre_scale = torch.linspace(0.5, 2, k, device="cuda", dtype=dtype)
                actual = quantize_activation(x, swiglu=True, pre_scale=pre_scale)
                g, u = x.chunk(2, dim=-1)
                expected = quantize_activation(
                    torch.nn.functional.silu(g) * u * pre_scale
                )
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_chunk_lifetime_and_graph(self):
        torch.manual_seed(861)
        w = torch.randn(2064, 528, device="cuda", dtype=torch.bfloat16)
        scale = w.float().abs().amax() / (448 * 6)
        p, b = ck.quantize_nvfp4(w, scale)
        x = torch.randn(65, 513, device="cuda", dtype=torch.bfloat16)
        reference = linear(x, p, b, scale, output_columns=2051)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                result = linear(x, p, b, scale, output_columns=2051)
                torch.empty((2048, 768), dtype=torch.int8, device="cuda").fill_(77)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = linear(x, p, b, scale, output_columns=2051)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)

    def test_wide_output_chunk_dispatch(self):
        torch.manual_seed(319)
        x = torch.randint(-20, 21, (137, 256), device="cuda", dtype=torch.int8)
        w = torch.randint(-20, 21, (264, 256), device="cuda", dtype=torch.int8)
        sx = torch.full((137,), 0.03125, device="cuda")
        sw = torch.full((264,), 0.015625, device="cuda")
        bias = torch.randn(264, device="cuda")
        reference = x.float() @ w.float().T
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            expected = (reference * sx[:, None] * sw + bias).to(dtype)
            for stride in (512, 21504):
                output = torch.full((137, stride), -19.0, device="cuda", dtype=dtype)
                view = output[:, 8:272]
                turing_int8_linear_out(x, w, sx, sw, view, bias)
                torch.testing.assert_close(view, expected, rtol=0, atol=0)
                self.assertTrue(bool((output[:, :8] == -19).all()))
                self.assertTrue(bool((output[:, 272:] == -19).all()))

    def test_logical_tails_dtypes_and_stream(self):
        torch.manual_seed(83)
        for k in (1, 15, 17, 255, 257, 5377, 16385, 32769):
            w = torch.randn(
                144, (k + 15) // 16 * 16, device="cuda", dtype=torch.bfloat16
            )
            scale = (w.float().abs().max() / (448 * 6)).reshape(1)
            p, b = ck.quantize_nvfp4(w, scale)
            # Nonzero serialized tail must be ignored, not included in rotation.
            stored = ck.dequantize_nvfp4(p, scale, b, torch.float32)[:, :k]
            padded = torch.nn.functional.pad(stored, (0, (-k) % 256)).contiguous()
            eq, es = ck.quantize_int8_convrot_weight(padded, 256)
            q, qs = turing_nvfp4_convrot_quantize(p, b, scale, 0, 144, k)
            torch.testing.assert_close(qs, es.flatten(), rtol=3e-6, atol=1e-8)
            delta = (q.short() - eq.short()).abs()
            self.assertLessEqual(int(delta.max()), 1)
            # Very short zero-padded rows repeat transform values, including
            # S8 half-steps. FP32 sum ordering can repeat a one-code difference.
            # Bound the code magnitude above and the GEMM error below as well.
            self.assertLess(
                float((delta != 0).float().mean()), 0.01 if k < 256 else 0.001
            )
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = torch.randn(19, k, device="cuda", dtype=dtype)
                bias = torch.randn(137, device="cuda")
                y = linear(x, p, b, scale, bias, output_columns=137)
                ref = x.float() @ stored[:137].T + bias
                self.assertEqual(y.dtype, dtype)
                self.assertEqual(y.shape, (19, 137))
                self.assertLess(float((y.float() - ref).norm() / ref.norm()), 0.03)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                other = linear(x, p, b, scale, bias, output_columns=137)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(other, y, rtol=0, atol=0)

    def test_fused_weight_quantization(self):
        torch.manual_seed(537)
        for k in (256, 512, 768, 2304, 5376, 6400, 14336, 14592, 16384):
            w = torch.randn(144, k, device="cuda", dtype=torch.bfloat16)
            w[:, ::128] *= 30
            scale = (w.float().abs().max() / (448 * 6)).reshape(1)
            packed, blocks = ck.quantize_nvfp4(w, scale)
            stored = ck.dequantize_nvfp4(packed, scale, blocks, torch.float32)
            eq, es = ck.quantize_int8_convrot_weight(stored, 256)
            for start, count in ((0, 144), (1, 17), (127, 17)):
                q, s = turing_nvfp4_convrot_quantize(
                    packed, blocks, scale, start, count
                )
                torch.testing.assert_close(
                    s.reshape(-1),
                    es.reshape(-1)[start : start + count],
                    rtol=2e-6,
                    atol=1e-8,
                )
                delta = (q.short() - eq[start : start + count].short()).abs()
                self.assertLessEqual(int(delta.max()), 1)
                self.assertLess(float((delta != 0).float().mean()), 0.001)
            zq, zs = turing_nvfp4_convrot_quantize(
                torch.zeros_like(packed), blocks, scale
            )
            self.assertEqual(int(zq.count_nonzero()), 0)
            self.assertTrue(bool(zs.isfinite().all()))

    def test_fused_invalid_storage_and_fake(self):
        p = torch.zeros(16, 128, device="cuda", dtype=torch.uint8)
        b = torch.zeros(128, 16, device="cuda", dtype=torch.uint8)
        s = torch.ones(1, device="cuda")
        torch.library.opcheck(turing_nvfp4_convrot_quantize, (p, b, s))
        for start, count in ((-1, 1), (15, 2), (0, 0)):
            with self.assertRaisesRegex(RuntimeError, "row range"):
                turing_nvfp4_convrot_quantize(p, b, s, start, count)
        with self.assertRaisesRegex(RuntimeError, "scale storage"):
            turing_nvfp4_convrot_quantize(p, b[:, :4].contiguous(), s)
        with FakeTensorMode() as mode:
            q, scales = turing_nvfp4_convrot_quantize(
                mode.from_tensor(p), mode.from_tensor(b), mode.from_tensor(s), 1, 7
            )
            self.assertEqual(q.shape, (7, 256))
            self.assertEqual(scales.shape, (7,))
