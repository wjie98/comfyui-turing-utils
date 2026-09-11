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
                    ["sdpa", "sage", "w8a8"],
                    {
                        "default": "sdpa",
                        "tooltip": "Decoder attention only. On Turing, BF16 SDPA inputs are consumed through containers and computed as FP16 to avoid the slow math fallback. W8A8 refers to QK attention, not VAE weight quantization.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Turing Utils/MiniMax H3"
    DESCRIPTION = (
        "Native H3 independent-window decoding and linear stitching with fused "
        "operators, selectable attention, and completed-tile progress. Accepts "
        "the official VAELoader; operators activate only during this node. Weight "
        "prefetch and output storage follow ComfyUI's VAE lifecycle."
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
        "MiniMax H3 video encoder with asynchronous pixel buffering, automatic "
        "tile batching, ComfyUI-managed block-level dynamic-weight prefetch, "
        "and output storage matching ComfyUI's VAE intermediate dtype. Accepts "
        "the official VAELoader and preserves its compute dtype."
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
