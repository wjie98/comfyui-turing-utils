"""Dedicated MiniMax H3 video VAE nodes."""

from __future__ import annotations

import comfy.model_management

from ..adapters.minimax.video_vae import (
    decode_video,
    require_h3_video_vae,
)
from ..adapters.minimax.video_vae_encode import encode_video


class MiniMaxH3VideoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "attention": (
                    ["sdpa", "sage", "w8a8", "native"],
                    {
                        "default": "w8a8",
                        "tooltip": "Decoder attention only. Native preserves the upstream attention forward and backend selection; eligible linear operators still use Turing Utils. On Turing, BF16 SDPA inputs compute in FP16 to avoid the math fallback. W8A8 does not change VAE weight quantization.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "Official ComfyUI H3 VAE decoding with scoped fused operators and "
        "selectable attention. ComfyUI owns tiling, batching, transfers, memory "
        "management and OOM recovery. Accepts the official VAELoader. Lightweight "
        "tqdm shows progress in completed logical tiles, including batched tile forwards."
    )

    def decode(
        self,
        samples,
        vae,
        attention,
    ):
        require_h3_video_vae(vae)
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        with comfy.model_management.cuda_device_context(vae.device):
            images = decode_video(
                vae,
                latent,
                attention,
            )
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return (images,)


class MiniMaxH3VideoVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "Official ComfyUI H3 VAE encoding with scoped fused operators. ComfyUI "
        "owns tiling, batching, transfers, memory management and OOM recovery. "
        "Accepts the official VAELoader and preserves its compute/output dtype. "
        "Lightweight tqdm shows progress in completed logical tiles."
    )

    def encode(
        self,
        pixels,
        vae,
    ):
        require_h3_video_vae(vae)
        with comfy.model_management.cuda_device_context(vae.device):
            latent = encode_video(
                vae,
                pixels,
            )
        return ({"samples": latent},)
