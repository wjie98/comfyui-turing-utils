"""Qwen Image 2.1 model-local optimization switch."""

import math

import torch
from comfy_api.latest import io
import comfy.model_management
import comfy.utils
import node_helpers

from ..adapters.qwen_image21 import configure


class QwenImage21Optimize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True,
                "tooltip": "Share W8A8 QKV input quantization. Reference cache settings follow the native Qwen Image 2.1 Cache node."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Turing Utils/Models"
    DESCRIPTION = "Share W8A8 QKV input quantization while preserving ComfyUI's reference cache settings."

    def apply(self, model, enabled=True):
        return (configure(model, enabled),)


class QwenImage21Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsQwenImage21Conditioning",
            display_name="Qwen Image 2.1 多图条件",
            category="Turing Utils/Models",
            description="CFG 1 / BasicGuider conditioning. Encodes only the positive prompt. Reference resize, alpha handling and latent alignment match the native Qwen 2.1 node. Use the native node for CFG > 1.",
            inputs=[io.Clip.Input("clip"), io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Vae.Input("vae", optional=True),
                    io.Int.Input("resolution", default=1024, min=0, max=4096, step=32),
                    io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 17)], min=0))],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent")])

    @classmethod
    def execute(cls, clip, prompt, vae=None, resolution=1024, images=None):
        ref_latents, vision = [], []
        width = height = resolution or 1024
        for name in sorted(images or {}, key=lambda key: int(key.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            samples = image[:1].movedim(-1, 1)
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                w = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                h = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                w, h = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
            w, h = max(32, w), max(32, h)
            pixels = image[:1] if (w, h) == (samples.shape[3], samples.shape[2]) else comfy.utils.common_upscale(
                samples, w, h, "lanczos", "disabled").movedim(1, -1)
            if not vision:
                width, height = w, h
            rgb = pixels[:, :, :, :3]
            if pixels.shape[-1] > 3:
                rgb = rgb * pixels[:, :, :, 3:] + (1.0 - pixels[:, :, :, 3:])
            vision.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(pixels))
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(
            prompt, images=vision, keep_vision=not ref_latents, prevent_empty_text=True))
        if ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros((1, 64, height // 16, width // 16), device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, {"samples": latent})
