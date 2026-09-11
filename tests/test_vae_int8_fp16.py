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
from comfy_kitchen.backends._activations import apply_input_act
from comfyui_turing_utils.quantization import dispatch
from comfyui_turing_utils.quantization.operator_scope import use_turing_operator_backend


class VAEInt8FP16Test(unittest.TestCase):
    def test_fake_contracts_preserve_fp16_output(self):
        x = torch.empty((33, 256), device="meta", dtype=torch.float16)
        q, scale = kernel.turing_fp16_int8_quantize(x)
        w = torch.empty((2048, 256), device="meta", dtype=torch.int8)
        ws = torch.empty((2048,), device="meta")
        result = kernel.turing_fp16_int8_linear(q, w, scale, ws)
        self.assertEqual(q.shape, (33, 256))
        self.assertEqual(scale.shape, (33, 1))
        self.assertEqual(result.shape, (33, 2048))
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_quantizer_matches_eager_boundaries(self):
        torch.manual_seed(932)
        for k in (256, 2048, 8192):
            for swiglu in (False, True):
                with self.subTest(k=k, swiglu=swiglu):
                    x = torch.randn((33, k * (2 if swiglu else 1)), device="cuda", dtype=torch.float16)
                    x[0].zero_()
                    x[1].mul_(1e-6)
                    activated = apply_input_act(x, "swiglu" if swiglu else None)
                    h = eager._build_hadamard(256, device=x.device, dtype=x.dtype)
                    expected_q, expected_scale = eager.quantize_and_rotate_rowwise(activated, h, 256)
                    q, scale = dispatch._quantize_turing_int8_activation(x, 256, "swiglu" if swiglu else None)
                    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
                    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_row_quantization_preserves_eager_extremes(self):
        x = torch.zeros((8, 259), dtype=torch.float16, device="cuda")
        x[1, 0] = torch.finfo(torch.float16).max
        x[2, 0] = -torch.finfo(torch.float16).max
        x[3, 0] = float("inf")
        x[4, 0] = float("-inf")
        x[5, 0] = float("nan")
        x[6, 0] = 2**-24
        x[7, :2] = torch.tensor([2**-14, -(2**-14)], device="cuda", dtype=torch.float16)
        q, scale = kernel.turing_fp16_int8_quantize(x)
        expected_q, expected_scale = eager.quantize_int8_rowwise(x)
        torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)

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
                    bias = None if bias_dtype is None else torch.randn(n, device="cuda", dtype=bias_dtype)
                    expected = (acc * (scale * ws)).half()
                    if bias is not None:
                        expected = expected + bias.half()
                    actual = kernel.turing_fp16_int8_linear(q, weight, scale, ws, bias)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_36_block_decoder_scope_preserves_native_rotation(self):
        from comfy.ldm.minimax.vae import ViT3DDecoder
        from comfyui_turing_utils.adapters.minimax import video_vae

        torch.manual_seed(935)
        ops = comfy.ops.mixed_precision_ops({"mixed_ops": True}, compute_dtype=torch.float16)
        # Native H3 depth, reduced width to keep the regression inexpensive.
        with torch.device("cuda"):
            decoder = ViT3DDecoder(
                patch_size=2, patch_size_t=1, in_channels=4, out_channels=3,
                num_layers=36, heads=4, dim_head=64, operations=ops,
            ).half()
        config = torch.tensor(list(json.dumps({
            "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256,
        }).encode()), dtype=torch.uint8)
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
                        "weight": torch.randint(-64, 65, (module.out_features, module.in_features), device="cuda", dtype=torch.int8),
                        "weight_scale": torch.full((module.out_features, 1), 0.00025, device="cuda"),
                        "comfy_quant": config,
                    }
                    if module.bias is not None:
                        state["bias"] = torch.zeros(module.out_features, device="cuda", dtype=torch.float16)
                    module.load_state_dict(state, strict=True)
            value = torch.randn(1, 4, 2, 4, 4, device="cuda", dtype=torch.float16)
            options = video_vae._attention_options("sdpa", value.device)
            reference = video_vae._decoder_forward(decoder, value, options)
            with video_vae._vae_operator_scope():
                candidate = video_vae._decoder_forward(decoder, value, options)
            restored = video_vae._decoder_forward(decoder, value, options)
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scoped_dispatch_preserves_weight_storage_and_fp32_fallback(self):
        import comfy_kitchen

        dispatch.register_backend()
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

        with mock.patch.object(dispatch, "_kernel_op", side_effect=lookup), use_turing_operator_backend():
            actual = comfy_kitchen.int8_linear(x, w, ws, out_dtype=x.dtype, convrot=True)
        self.assertEqual(seen, [(w.data_ptr(), w.stride())])
        torch.testing.assert_close(actual, eager.int8_linear(x, w, ws, out_dtype=x.dtype, convrot=True), rtol=0, atol=0)
        with mock.patch.object(dispatch, "_kernel_op", side_effect=AssertionError("FP32 must not use FP16/BF16 kernel")), use_turing_operator_backend():
            actual = comfy_kitchen.int8_linear(x.float(), w, ws, out_dtype=torch.float32, convrot=True)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, eager.int8_linear(x.float(), w, ws, out_dtype=torch.float32, convrot=True), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
