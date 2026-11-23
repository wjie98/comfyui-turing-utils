"""Reference-cache correctness and model-local Qwen 2.1 installation."""

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

import comfy.ops
from comfy.ldm.qwen_image21 import model as qwen
from comfyui_turing_utils.adapters import qwen_image21 as adapter


def small_model():
    torch.manual_seed(21)
    model = qwen.QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, attention_head_dim=64,
        num_attention_heads=2, context_in_dim=32, axes_dims_rope=(16, 24, 24),
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.03)
    model.current_patcher = SimpleNamespace(get_free_memory=lambda device: 1 << 40)
    model.reset_prefix_cache(True)
    return model


class QwenImage21Test(unittest.TestCase):
    def setUp(self):
        self.pin = mock.patch("comfy.model_management.pin_memory", return_value=False)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def test_cached_target_path_matches_native_and_skips_reference_projection(self):
        model = small_model()
        original = model._forward
        optimized = adapter._make_model_forward(model, original)
        context = torch.randn(1, 7, 32)
        refs = [torch.randn(1, 4, 3, 4), torch.randn(1, 4, 2, 3)]
        inputs = [torch.randn(1, 4, 4, 5) for _ in range(3)]
        options = {"qwen_image21_cache": {"device": "cpu"}}
        with torch.inference_mode():
            expected = [original(x, torch.tensor([t]), context, refs, [2, 5], dict(options))
                        for x, t in zip(inputs, (1.0, 0.7, 0.3))]
            model.reset_prefix_cache(True)
            with mock.patch.object(model, "build_sequence", wraps=model.build_sequence) as build:
                actual = [optimized(x, torch.tensor([t]), context, refs, [2, 5], dict(options))
                          for x, t in zip(inputs, (1.0, 0.7, 0.3))]
            self.assertEqual(build.call_count, 1)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            model.reset_prefix_cache(False)
            self.assertIsNone(model.prefix_cache)

    def test_changed_prompt_reference_order_and_target_geometry_refill_cache(self):
        model = small_model()
        original = model._forward
        optimized = adapter._make_model_forward(model, original)
        context = torch.randn(1, 7, 32)
        refs = [torch.randn(1, 4, 2, 3), torch.randn(1, 4, 2, 3)]
        options = {"qwen_image21_cache": {"device": "cpu"}}
        cases = [(context, refs, 4), (context + 0.1, refs, 4),
                 (context, refs[::-1], 4), (context, refs, 5),
                 (context + 0.1, refs, 4), (context, refs[::-1], 4)]
        expected = []
        inputs = [torch.randn(1, 4, height, 5) for _, _, height in cases]
        with torch.inference_mode():
            for x, (text, images, _) in zip(inputs, cases):
                model.reset_prefix_cache(True)
                expected.append(original(x, torch.tensor([0.7]), text, images, [2, 5], dict(options)))
            model.reset_prefix_cache(True)
            for x, (text, images, _), target in zip(inputs, cases, expected):
                actual = optimized(x, torch.tensor([0.7]), text, images, [2, 5], dict(options))
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
            model.reset_prefix_cache(False)

    def test_hooks_disabled_optimization_and_cache_off_use_original(self):
        model = small_model()
        original = mock.Mock(return_value="native")
        optimized = adapter._make_model_forward(model, original)
        x = torch.zeros(1, 4, 4, 5)
        context = torch.zeros(1, 7, 32)
        cases = [
            {adapter.OPTIONS_KEY: {"enabled": False}},
            {"qwen_image21_cache": {"device": "off"}},
            {"patches": {"attn1_patch": [object()]}},
            {"patches_replace": {"dit": {("single_block", 0): object()}}},
        ]
        for options in cases:
            self.assertEqual(optimized(x, torch.tensor([1.]), context, transformer_options=options), "native")
        self.assertEqual(original.call_count, len(cases))

    def test_installer_does_not_touch_other_models_and_is_idempotent(self):
        model = small_model()

        class Patcher:
            def __init__(self, diffusion):
                self.model = SimpleNamespace(diffusion_model=diffusion)
                self.object_patches = {}

            def get_model_object(self, name):
                if name in self.object_patches:
                    return self.object_patches[name]
                value = self.model
                for part in name.split("."):
                    value = getattr(value, part)
                return value

            def add_object_patch(self, name, value):
                self.object_patches[name] = value

        unrelated = Patcher(torch.nn.Linear(4, 4))
        self.assertEqual(adapter.apply_qwen_image21_adapter(unrelated, torch.device("cpu")), 0)
        self.assertEqual(unrelated.object_patches, {})
        patcher = Patcher(model)
        self.assertEqual(adapter.apply_qwen_image21_adapter(patcher, torch.device("cpu")), 2)
        self.assertEqual(adapter.apply_qwen_image21_adapter(patcher, torch.device("cpu")), 0)
        legacy = "prefix_cache" not in inspect.signature(model._forward).parameters
        expected = {f"diffusion_model.transformer_blocks.{i}.attn.forward" for i in range(2)}
        if legacy:
            expected.add("diffusion_model._forward")
        self.assertEqual(set(patcher.object_patches), expected)

    def test_encoder_scope_is_qwen21_only_and_is_installed_once(self):
        from comfy.text_encoders.qwen_image21 import QwenImage21TEModel

        class Encoder(QwenImage21TEModel):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def encode_token_weights(self, value):
                return value + 1

        encoder = Encoder()
        patches, attachments = {}, {}
        patcher = SimpleNamespace(
            load_device=torch.device("cuda"),
            get_attachment=lambda name: attachments.get(name),
            set_attachments=lambda name, value: attachments.update({name: value}),
            get_model_object=lambda name: patches.get(name, getattr(encoder, name)),
            add_object_patch=lambda name, value: patches.update({name: value}))
        clip = SimpleNamespace(cond_stage_model=encoder, patcher=patcher)
        unrelated = SimpleNamespace(cond_stage_model=torch.nn.Linear(4, 4))
        self.assertFalse(adapter.apply_qwen_image21_clip_adapter(unrelated))
        self.assertTrue(adapter.apply_qwen_image21_clip_adapter(clip))
        self.assertFalse(adapter.apply_qwen_image21_clip_adapter(clip))
        with mock.patch("comfy.ops.use_quantized_matmul") as scope:
            self.assertEqual(patches["encode_token_weights"](5), 6)
            scope.assert_called_once()
            scope.return_value.__enter__.assert_called_once()
            scope.return_value.__exit__.assert_called_once_with(None, None, None)

    def test_shared_qkv_preserves_bypass_lora_forward(self):
        linears = [torch.nn.Linear(256, 256, bias=False) for _ in range(3)]
        linears[0].forward = lambda value: value + 3
        attention = SimpleNamespace(to_q=linears[0], to_k=linears[1], to_v=linears[2])
        original = mock.Mock(return_value="bypass preserved")
        # weak_method needs a weak-referenceable owner.
        class Owner:
            pass
        owner = Owner()
        owner.__dict__.update(attention.__dict__)
        forward = adapter._make_attention_forward(owner, original)
        x = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
        with mock.patch.object(adapter, "convrot_w8_plain_tensors", return_value=(None, None)), \
                mock.patch.object(adapter, "_cast_linear", side_effect=AssertionError("bypassed LoRA")):
            self.assertEqual(forward(x, None, None, 0), "bypass preserved")


class QwenConditioningTest(unittest.TestCase):
    def test_positive_only_matches_native_reference_preprocessing(self):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        from comfyui_turing_utils.nodes.qwen_image21 import QwenImage21Conditioning

        class Clip:
            def __init__(self):
                self.calls = []

            def tokenize(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return prompt

            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.ones(1, 3, 4), {"image_slots": [1, 2]}]]

        class Vae:
            def __init__(self):
                self.pixels = []

            def encode(self, pixels):
                self.pixels.append(pixels.clone())
                return pixels.mean(dim=-1)[:, None, ::16, ::16].repeat(1, 64, 1, 1)

        torch.manual_seed(21)
        images = {"image_2": torch.rand(1, 32, 64, 3),
                  "image_1": torch.rand(2, 64, 96, 4), "image_3": None}
        for with_vae, resolution in ((True, 64), (False, 0)):
            with self.subTest(with_vae=with_vae):
                native_clip, clip = Clip(), Clip()
                native_vae, vae = (Vae(), Vae()) if with_vae else (None, None)
                native, _, latent = TextEncodeQwenImage21.execute(
                    native_clip, "replace clothes", "", native_vae, resolution, images).result
                actual, actual_latent = QwenImage21Conditioning.execute(
                    clip, "replace clothes", vae, resolution, images).result
                self.assertEqual(len(native_clip.calls), 2)
                self.assertEqual(len(clip.calls), 1)
                torch.testing.assert_close(actual[0][0], native[0][0], rtol=0, atol=0)
                torch.testing.assert_close(actual_latent["samples"], latent["samples"], rtol=0, atol=0)
                self.assertEqual(actual[0][1]["image_slots"], native[0][1]["image_slots"])
                native_args, actual_args = native_clip.calls[0][1], clip.calls[0][1]
                self.assertEqual(actual_args["keep_vision"], not with_vae)
                self.assertTrue(actual_args["prevent_empty_text"])
                for a, b in zip(actual_args["images"], native_args["images"]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(a.shape[-1], 3)
                if with_vae:
                    self.assertEqual(vae.pixels[0].shape[-1], 4)
                    for a, b in zip(vae.pixels, native_vae.pixels):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    for a, b in zip(actual[0][1]["reference_latents"], native[0][1]["reference_latents"]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
