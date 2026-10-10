"""FP16 VAE contracts: no BF16 intermediate and no transposed weight copy."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT.parents[1]))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.ops
import comfyui_turing_utils_kernel as kernel
from comfy_kitchen.backends.eager import quantization as eager
from comfyui_turing_utils.quantization import backend, dispatch
from comfyui_turing_utils.quantization.operator_scope import use_turing_operator_backend


def fp32_quantization_reference(
    x, group_size=256, input_act=None, input_act_weight=None, input_act_eps=0.0
):
    """Independent dense FP64 rotation oracle, rounded only at the FP32 quantizer."""
    value = x.double()
    if input_act == "swiglu":
        gate, up = value.chunk(2, dim=-1)
        value = torch.nn.functional.silu(gate) * up
    elif input_act == "gelu_tanh":
        value = torch.nn.functional.gelu(value, approximate="tanh")
    elif input_act == "rms_norm":
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + input_act_eps
        )
        value = value * input_act_weight.double()
    h = eager._build_hadamard(group_size, device=x.device, dtype=torch.float64)
    rotated = (value.reshape(-1, group_size) @ h).reshape(x.size(0), -1).float()
    scale = (rotated.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
    q = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale, rotated


class VAEInt8FP16Test(unittest.TestCase):
    def test_fake_contracts_preserve_fp16_output(self):
        x = torch.empty((33, 256), device="meta", dtype=torch.float16)
        q, scale = kernel.turing_fp16_int8_convrot_quantize(x)
        w = torch.empty((2048, 256), device="meta", dtype=torch.int8)
        ws = torch.empty((2048,), device="meta")
        result = kernel.turing_fp16_int8_linear(q, w, scale, ws)
        self.assertEqual(q.shape, (33, 256))
        self.assertEqual(scale.shape, (33, 1))
        self.assertEqual(result.shape, (33, 2048))
        self.assertEqual(result.dtype, torch.float16)
        q, scale = kernel.turing_fp16_int8_convrot_quantize(
            torch.empty((33, 512), device="meta", dtype=torch.float16),
            input_act="swiglu",
        )
        self.assertEqual(q.shape, (33, 256))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_fused_quantizer_matches_high_precision_oracle(self):
        torch.manual_seed(932)
        for k in (256, 2048, 8192, 65536):
            for activation in (None, "swiglu", "gelu_tanh", "rms_norm"):
                with self.subTest(k=k, activation=activation):
                    x = torch.randn(
                        (9, k * (2 if activation == "swiglu" else 1)),
                        device="cuda",
                        dtype=torch.float16,
                    )
                    x[0].zero_()
                    x[1].mul_(1e-6)
                    norm_weight = torch.randn(k, device=x.device, dtype=x.dtype)
                    expected_q, expected_scale, rotated = fp32_quantization_reference(
                        x, 256, activation, norm_weight, 1e-6
                    )
                    q, scale = dispatch._quantize_turing_int8_activation(
                        x, 256, activation, norm_weight, 1e-6
                    )
                    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=0)
                    # FP32 FHT/activation reductions may cross a half-code tie;
                    # they must never introduce a larger than one-code error.
                    differences = (q.int() - expected_q.int()).abs()
                    self.assertLessEqual(differences.max().item(), 1)
                    self.assertLessEqual((differences != 0).float().mean().item(), 1e-4)
                    error = (q.float() * scale - rotated).abs()
                    self.assertTrue(torch.all(error <= scale * 0.501).item())
                    self.assertEqual(q.dtype, torch.int8)
                    self.assertEqual(scale.dtype, torch.float32)
                    self.assertEqual(torch.count_nonzero(q[0]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_fused_quantization_extremes(self):
        x = torch.zeros((8, 256), dtype=torch.float16, device="cuda")
        x[1, 0] = torch.finfo(torch.float16).max
        x[2, 0] = -torch.finfo(torch.float16).max
        x[3, 0] = float("inf")
        x[4, 0] = float("-inf")
        x[5, 0] = float("nan")
        x[6, 0] = 2**-24
        x[7, :2] = torch.tensor([2**-14, -(2**-14)], device="cuda", dtype=torch.float16)
        q, scale = kernel.turing_fp16_int8_convrot_quantize(x)
        finite = torch.tensor([0, 1, 2, 6, 7], device="cuda")
        expected_q, expected_scale, _ = fp32_quantization_reference(x[finite])
        torch.testing.assert_close(scale[finite], expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(q[finite], expected_q, rtol=0, atol=0)
        self.assertTrue(torch.isnan(scale[3:6]).all().item())
        self.assertEqual(torch.count_nonzero(q[3:6]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_tiny_rows_keep_int8_accuracy(self):
        torch.manual_seed(938)
        for amplitude in (1.0, 1e-4, 1e-6, 1e-7, 0.0):
            with self.subTest(amplitude=amplitude):
                x = (torch.randn(33, 2048, device="cuda") * amplitude).half()
                _, _, rotated = fp32_quantization_reference(x)
                q, scale = kernel.turing_fp16_int8_convrot_quantize(x)
                self.assertTrue(torch.isfinite(scale).all().item())
                self.assertTrue((scale > 0).all().item())
                if amplitude:
                    relative_rmse = (
                        q.float() * scale - rotated
                    ).norm() / rotated.norm()
                    self.assertLess(relative_rmse.item(), 0.012)
                else:
                    self.assertEqual(torch.count_nonzero(q).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_fused_quantizer_validation_and_noncontiguous_input(self):
        x = torch.randn(256, 9, device="cuda", dtype=torch.float16).T
        q, scale = kernel.turing_fp16_int8_convrot_quantize(x)
        qc, sc = kernel.turing_fp16_int8_convrot_quantize(x.contiguous())
        torch.testing.assert_close(q, qc, rtol=0, atol=0)
        torch.testing.assert_close(scale, sc, rtol=0, atol=0)
        for kwargs, message in (
            ({"group_size": 64}, "group_size=256"),
            ({"input_act": "rms_norm"}, "requires input_act_weight"),
            ({"input_act": "rms_norm", "input_act_weight": x[0, :128]}, "length K"),
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                kernel.turing_fp16_int8_convrot_quantize(x, **kwargs)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            kernel.turing_fp16_int8_convrot_quantize(x, input_act="unknown")
        norm_weight = torch.randn(256, device="cuda")
        q, scale = kernel.turing_fp16_int8_convrot_quantize(
            x, input_act="rms_norm", input_act_weight=norm_weight, input_act_eps=1e-6
        )
        qr, sr, _ = fp32_quantization_reference(
            x, input_act="rms_norm", input_act_weight=norm_weight, input_act_eps=1e-6
        )
        torch.testing.assert_close(q, qr, rtol=0, atol=0)
        torch.testing.assert_close(scale, sr, rtol=1e-6, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_gemm_matches_scale_and_bias_rounding(self):
        torch.manual_seed(933)
        # N=16384 exercises the Ampere schedule, the other shapes the SM75 mainloop.
        for m, n, k in ((1, 8, 256), (33, 2048, 256), (33, 16384, 256)):
            q = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
            weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
            scale = torch.rand((m, 1), device="cuda") * 0.01
            ws = torch.rand(n, device="cuda") * 0.01
            acc = q.float() @ weight.float().T
            for bias_dtype in (None, torch.float16, torch.float32):
                with self.subTest(m=m, n=n, bias=bias_dtype):
                    bias = (
                        None
                        if bias_dtype is None
                        else torch.randn(n, device="cuda", dtype=bias_dtype)
                    )
                    expected = (acc * (scale * ws)).half()
                    if bias is not None:
                        expected = expected + bias.half()
                    actual = kernel.turing_fp16_int8_linear(q, weight, scale, ws, bias)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_fused_quantizer_graph_capture_and_compile(self):
        x = torch.randn(9, 256, device="cuda", dtype=torch.float16)
        operation = kernel.turing_fp16_int8_convrot_quantize
        expected = operation(x)
        compiled = torch.compile(operation, backend="eager", fullgraph=True)
        for actual, reference in zip(compiled(x), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            operation(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = operation(x)
        graph.replay()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        x.zero_()
        graph.replay()
        self.assertEqual(torch.count_nonzero(captured[0]).item(), 0)
        self.assertTrue((captured[1] > 0).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_36_block_decoder_fp32_quantization_and_scope_restoration(self):
        from comfy.ldm.minimax.vae import ViT3DDecoder
        from comfyui_turing_utils.adapters.minimax import video_vae

        torch.manual_seed(935)
        ops = comfy.ops.mixed_precision_ops(
            {"mixed_ops": True}, compute_dtype=torch.float16
        )
        # Native H3 depth, reduced width to keep the regression inexpensive.
        with torch.device("cuda"):
            decoder = ViT3DDecoder(
                patch_size=2,
                patch_size_t=1,
                in_channels=4,
                out_channels=3,
                num_layers=36,
                heads=4,
                dim_head=64,
                operations=ops,
            ).half()
        config = torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 256,
                    }
                ).encode()
            ),
            dtype=torch.uint8,
        )
        with torch.inference_mode():
            for name, parameter in decoder.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1)
                elif "scale" in name:
                    parameter.fill_(0.1)
                elif name.endswith("bias"):
                    parameter.zero_()
                else:
                    parameter.normal_(0, 0.01)
            for module in decoder.modules():
                if isinstance(module, ops.Linear) and module.in_features % 256 == 0:
                    state = {
                        "weight": torch.randint(
                            -64,
                            65,
                            (module.out_features, module.in_features),
                            device="cuda",
                            dtype=torch.int8,
                        ),
                        "weight_scale": torch.full(
                            (module.out_features, 1), 0.00025, device="cuda"
                        ),
                        "comfy_quant": config,
                    }
                    if module.bias is not None:
                        state["bias"] = torch.zeros(
                            module.out_features, device="cuda", dtype=torch.float16
                        )
                    module.load_state_dict(state, strict=True)
            value = torch.randn(1, 4, 2, 4, 4, device="cuda", dtype=torch.float16)
            with video_vae._decoder_overrides(decoder, "sdpa", value.device):
                reference = decoder(value)
                with video_vae._vae_operator_scope():
                    with mock.patch.object(
                        dispatch,
                        "_quantize_turing_int8_activation",
                        wraps=dispatch._quantize_turing_int8_activation,
                    ) as quantizer:
                        candidate = decoder(value)
                    activations = [
                        call.args[2]
                        if len(call.args) > 2
                        else call.kwargs.get("input_act")
                        for call in quantizer.call_args_list
                    ]
                    self.assertIn("rms_norm", activations)
                    self.assertIn("swiglu", activations)

                    def oracle(*args, **kwargs):
                        q, scale, _ = fp32_quantization_reference(*args, **kwargs)
                        return q, scale

                    with mock.patch.object(
                        dispatch, "_quantize_turing_int8_activation", side_effect=oracle
                    ):
                        high_precision = decoder(value)
                restored = decoder(value)
        # The numerical contract is FP32 quantization, not native FP16 rounding.
        # Quantization ties may diverge through 36 residual blocks; keep the
        # overall discrepancy below the INT8 quantization noise floor.
        relative_error = (
            candidate.float() - high_precision.float()
        ).norm() / high_precision.float().norm()
        self.assertLess(relative_error.item(), 0.012)
        self.assertEqual(candidate.dtype, torch.float16)
        self.assertTrue(torch.isfinite(candidate).all().item())
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scoped_dispatch_preserves_weight_storage_and_fp32_fallback(self):
        import comfy_kitchen

        backend.register_backend()
        torch.manual_seed(934)
        x = torch.randn((33, 256), device="cuda", dtype=torch.float16)
        w = torch.randint(-64, 65, (64, 256), device="cuda", dtype=torch.int8)
        ws = torch.full((64,), 0.001, device="cuda")
        operation = dispatch._kernel_op
        seen = []

        def lookup(name):
            op = operation(name)
            if name != "turing_fp16_int8_linear":
                return op

            def gemm(q, weight, *args):
                seen.append((weight.data_ptr(), weight.stride()))
                return op(q, weight, *args)

            return gemm

        with (
            mock.patch.object(dispatch, "_kernel_op", side_effect=lookup),
            use_turing_operator_backend(),
        ):
            actual = comfy_kitchen.int8_linear(
                x, w, ws, out_dtype=x.dtype, convrot=True
            )
        self.assertEqual(seen, [(w.data_ptr(), w.stride())])
        q, scale, _ = fp32_quantization_reference(x)
        expected = ((q.float() @ w.float().T) * (scale * ws)).half()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with (
            mock.patch.object(
                dispatch,
                "_kernel_op",
                side_effect=AssertionError("FP32 must not use FP16/BF16 kernel"),
            ),
            use_turing_operator_backend(),
        ):
            actual = comfy_kitchen.int8_linear(
                x.float(), w, ws, out_dtype=torch.float32, convrot=True
            )
        self.assertEqual(actual.dtype, torch.float32)
        # Unsupported dtypes delegate to Kitchen's CUDA backend, not eager.
        # CUDA and eager have different FP32 scale multiplication order. This
        # is a fallback-preservation contract, so use the actual native oracle.
        from comfy_kitchen.backends import cuda as kitchen_cuda

        expected = kitchen_cuda.int8_linear(
            x.float(), w, ws, out_dtype=torch.float32, convrot=True
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
