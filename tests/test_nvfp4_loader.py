import json
import sys
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

import comfy.ops
import comfy.lora
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.backends import cuda as ck
from comfyui_turing_utils.quantization.nvfp4 import nvfp4_operations
from comfyui_turing_utils.nodes.loaders import ConvRotDiffusionModelLoader
from comfyui_turing_utils.loading import convrot as loader


class NVFP4LoaderTest(unittest.TestCase):
    def test_node_does_not_expose_activation_scaling(self):
        self.assertNotIn(
            "activation_scaling", ConvRotDiffusionModelLoader.INPUT_TYPES()["required"]
        )

    def test_node_uses_official_resolver(self):
        sentinel = object()
        with (
            mock.patch(
                "folder_paths.get_full_path_or_raise", return_value="/model.safetensors"
            ) as resolve,
            mock.patch.object(loader, "_convrot_skip_reason", return_value=None),
            mock.patch.object(
                loader, "load_convrot_model", return_value=sentinel
            ) as load,
        ):
            self.assertEqual(
                ConvRotDiffusionModelLoader().load_diffusion_model("test.safetensors"),
                (sentinel,),
            )
        resolve.assert_called_once_with("diffusion_models", "test.safetensors")
        load.assert_called_once_with(
            "/model.safetensors", False, attention_backend="w8a8"
        )

    def test_rejects_non_nvfp4_before_constructing_model(self):
        with (
            mock.patch.object(
                loader.comfy.model_management,
                "get_torch_device",
                return_value=torch.device("cuda"),
            ),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            mock.patch.object(loader, "load_nvfp4_backend"),
            mock.patch.object(
                loader.comfy.utils,
                "load_torch_file",
                return_value=({"weight": torch.ones(8, 16)}, {}),
            ),
            mock.patch.object(
                loader.comfy.sd, "load_diffusion_model_state_dict"
            ) as load,
        ):
            with self.assertRaisesRegex(ValueError, "supported ConvRot"):
                loader.load_convrot_model("/not-nvfp4.safetensors")
            load.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA NVFP4 staging requires a GPU")
class NVFP4LinearTest(unittest.TestCase):
    def test_deferred_dtype_and_cpu_fallback(self):
        source = self.make_linear()
        ops = nvfp4_operations(None, torch.device("cpu"), full_precision_mm=True)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            layer = ops.Linear(64, 128, device="cpu", dtype=dtype)
            layer.load_state_dict(source.state_dict())
            self.assertEqual(layer.weight.dtype, dtype)
            x = torch.randn(3, 64, dtype=dtype)
            torch.testing.assert_close(
                layer(x),
                torch.nn.functional.linear(x, layer.weight.dequantize(), layer.bias),
                rtol=0,
                atol=0,
            )

    def test_real_mixed_clip_checkpoint_load_and_encode(self):
        from safetensors.torch import save_file
        from comfy.clip_model import CLIPTextModel
        from comfyui_turing_utils import bootstrap_builtin_integrations
        from comfyui_turing_utils.quantization.convrot import _summarize_convrot_modules
        from comfyui_turing_utils.kernel_api import load_nvfp4_backend

        bootstrap_builtin_integrations()
        config = dict(
            num_hidden_layers=12,
            hidden_size=256,
            num_attention_heads=4,
            intermediate_size=512,
            hidden_act="quick_gelu",
            max_position_embeddings=77,
            eos_token_id=2,
        )
        model = CLIPTextModel(config, torch.float32, "cpu", comfy.ops.manual_cast)
        state = {
            key: torch.randn_like(value) * 0.01
            for key, value in model.state_dict().items()
        }
        name = "text_model.encoder.layers.0.mlp.fc1"
        w = state[name + ".weight"].cuda().bfloat16()
        scale = w.float().abs().max() / (448 * 6)
        packed, blocks = ck.quantize_nvfp4(w, scale, pad_16x=True)
        state[name + ".weight"] = packed.cpu()
        state[name + ".weight_scale"] = blocks.cpu()
        state[name + ".weight_scale_2"] = scale.cpu()
        configs = {name: {"format": "nvfp4"}}
        name = "text_model.encoder.layers.1.mlp.fc1"
        w = state[name + ".weight"]
        scale = w.abs().amax(dim=1) / 127
        state[name + ".weight"] = (
            (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        state[name + ".weight_scale"] = scale[:, None]
        configs[name] = {"format": "int8_tensorwise", "per_row": True}
        with tempfile.TemporaryDirectory(prefix="mixed-clip-") as directory:
            path = Path(directory) / "clip.safetensors"
            save_file(
                state,
                path,
                metadata={"_quantization_metadata": json.dumps({"layers": configs})},
            )
            for device in (torch.device("cpu"), torch.device("cuda")):
                clip = loader.load_convrot_clip(
                    path,
                    model_options={
                        "load_device": device,
                        "offload_device": torch.device("cpu"),
                        "model_config": config.copy(),
                    },
                    disable_dynamic=True,
                )
                self.assertEqual(
                    _summarize_convrot_modules(clip.cond_stage_model).nvfp4, 1
                )
                backend = load_nvfp4_backend()
                with mock.patch.object(
                    backend, "linear", wraps=backend.linear
                ) as fast_linear:
                    result = clip.encode_from_tokens(
                        {"l": [[(1, 1.0), (2, 1.0)] + [(2, 1.0)] * 75]}
                    )
                self.assertEqual(fast_linear.call_count > 0, device.type == "cuda")
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(result.shape, (1, 77, 256))
                comfy.model_management.unload_all_models()

    def test_installed_runtime_probe(self):
        from comfyui_turing_utils_kernel.nvfp4 import validate_runtime

        validate_runtime(torch.device("cuda"))

    def make_linear(self, full=False, k=64, n=128):
        torch.manual_seed(74)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        scale = w.float().abs().max() / (448 * 6)
        packed, blocks = ck.quantize_nvfp4(w, scale, pad_16x=True)
        ops = nvfp4_operations(torch.bfloat16, torch.device("cuda"))
        layer = ops.Linear(k, n, device="cpu", dtype=torch.bfloat16)
        config = {"format": "nvfp4", "full_precision_matrix_mult": full}
        state = {
            "weight": packed.cpu(),
            "weight_scale": blocks.cpu(),
            "weight_scale_2": scale.cpu(),
            "bias": torch.zeros(n, dtype=torch.bfloat16),
            "comfy_quant": torch.tensor(
                list(json.dumps(config).encode()), dtype=torch.uint8
            ),
        }
        layer.load_state_dict(state)
        return layer

    def test_forward_offload_dtype_and_dense_hooks(self):
        layer = self.make_linear()
        self.assertIsInstance(layer.weight, QuantizedTensor)
        pointer = layer.weight._qdata.data_ptr()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(2, 9, 64, device="cuda", dtype=dtype)
            y = layer(x)
            expected = torch.nn.functional.linear(
                x.float(), layer.weight.cuda().dequantize().float()
            )
            self.assertEqual(y.shape, (2, 9, 128))
            self.assertEqual(y.dtype, dtype)
            self.assertLess(
                float(((y.float() - expected).norm() / expected.norm()).detach()), 0.04
            )
        self.assertEqual(layer.weight._qdata.data_ptr(), pointer)
        self.assertEqual(layer.weight.device.type, "cpu")
        layer.weight_function = [lambda w: w + 0.125]
        x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
        y = layer(x)
        expected = torch.nn.functional.linear(
            x, layer.weight.cuda().dequantize() + 0.125
        )
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        layer.weight_function = []

    def test_requantized_patch_stays_compact_and_changes_result(self):
        layer = self.make_linear().cuda()
        x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
        before = layer(x)
        dense = layer.convert_weight(layer.weight)
        layer.set_weight(dense + 0.25, seed=0)
        self.assertIsInstance(layer.weight, QuantizedTensor)
        self.assertEqual(layer.weight._qdata.dtype, torch.uint8)
        self.assertEqual(layer.weight._qdata.shape, (128, 32))
        after = layer(x)
        self.assertGreater(float((before - after).abs().max()), 0.1)
        self.assertFalse(any("prepared" in name for name in vars(layer)))

    def test_lora_requantization_matches_official_layout(self):
        for k, n in ((64, 128), (257, 129)):
            layer = self.make_linear(k=k, n=n).cuda()
            official = comfy.ops.mixed_precision_ops(
                compute_dtype=torch.bfloat16
            ).Linear(k, n, device="cuda", dtype=torch.bfloat16)
            official.load_state_dict(layer.state_dict())
            source = layer.weight
            patches = []
            for rank in (4, 8, 16):
                up = torch.randn(n, rank, device="cuda") * 0.03
                down = torch.randn(rank, k, device="cuda") * 0.03
                patches.append(
                    (
                        0.7,
                        LoRAAdapter(set(), (up, down, rank, None, None, None)),
                        1.0,
                        None,
                        None,
                    )
                )
            for dtype in (torch.float32, torch.bfloat16):
                # Accumulate the complete patch list before the single NVFP4
                # requantization, as both resident and low-VRAM Comfy paths do.
                dense = source.to(dtype=dtype).dequantize()
                merged = comfy.lora.calculate_weight(
                    patches, dense.clone(), "weight", intermediate_dtype=dtype
                )
                # Kitchen CUDA's deterministic quantizer accepts FP16/BF16,
                # not FP32; positive-seed Comfy stochastic quantization does.
                for seed in (1931,) if dtype == torch.float32 else (0, 1931):
                    expected = source.requantize_from_float(
                        merged.clone(), scale="recalculate", stochastic_rounding=seed
                    ).to(source.dtype)
                    official.set_weight(merged.clone(), seed=seed)
                    layer.set_weight(merged.clone(), seed=seed)
                    for actual in (layer.weight, official.weight):
                        torch.testing.assert_close(
                            actual._qdata, expected._qdata, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual._params.block_scale.view(torch.uint8),
                            expected._params.block_scale.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            actual._params.scale, expected._params.scale, rtol=0, atol=0
                        )
                    before = layer.weight._qdata.clone()
                    layer(torch.randn(3, k, device="cuda", dtype=torch.bfloat16))
                    torch.testing.assert_close(
                        layer.weight._qdata, before, rtol=0, atol=0
                    )

    def test_fp32_activation_is_not_rounded_through_bf16(self):
        layer = self.make_linear()
        x = torch.randn(3, 64, device="cuda", dtype=torch.float32)
        from comfyui_turing_utils_kernel import nvfp4 as backend

        with mock.patch.object(
            backend, "quantize_activation", wraps=backend.quantize_activation
        ) as quant:
            y = layer(x)
        self.assertEqual(quant.call_args.args[0].dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_swiglu_and_dense_hook_fallback(self):
        for k in (64, 256):
            layer = self.make_linear(k=k)
            x = torch.randn(2, 9, k * 2, device="cuda", dtype=torch.bfloat16)
            gate, up = x.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            actual, expected = layer.forward_swiglu(x), layer(activated)
            self.assertLess(
                float(
                    (actual.float() - expected.float()).norm() / expected.float().norm()
                ),
                0.02,
            )
            layer.weight_function = [lambda w: w + 0.125]
            torch.testing.assert_close(
                layer.forward_swiglu(x), layer(activated), rtol=0, atol=0
            )

    def test_noncontiguous_input(self):
        layer = self.make_linear()
        x = torch.randn(2, 64, 9, device="cuda", dtype=torch.float16).transpose(1, 2)
        torch.testing.assert_close(layer(x), layer(x.contiguous()), rtol=0, atol=0)

    def test_padded_checkpoint_uses_kernel(self):
        from comfyui_turing_utils_kernel import nvfp4 as backend

        for k, n in ((17, 13), (257, 129), (5377, 137)):
            layer = self.make_linear(k=k, n=n)
            x = torch.randn(7, k, device="cuda", dtype=torch.float32)
            with mock.patch.object(backend, "linear", wraps=backend.linear) as call:
                y = layer(x)
            call.assert_called_once()
            ref = torch.nn.functional.linear(
                x, layer.weight.cuda().dequantize().float()
            )
            self.assertEqual(y.shape, (7, n))
            self.assertLess(float((y - ref).norm() / ref.norm()), 0.04)

    def test_full_precision_override_and_cpu(self):
        layer = self.make_linear(full=True)
        for device in ("cpu", "cuda"):
            x = torch.randn(3, 64, dtype=torch.bfloat16, device=device)
            expected = torch.nn.functional.linear(
                x, layer.weight.to(device).dequantize()
            )
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_unquantized_layer_is_unchanged(self):
        ops = nvfp4_operations(torch.bfloat16, torch.device("cuda"))
        layer = ops.Linear(64, 32, device="cpu", dtype=torch.bfloat16)
        w = torch.randn(32, 64, dtype=torch.bfloat16)
        b = torch.zeros(32, dtype=torch.bfloat16)
        layer.load_state_dict({"weight": w, "bias": b})
        x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            layer(x), torch.nn.functional.linear(x, w.cuda(), b.cuda()), rtol=0, atol=0
        )
