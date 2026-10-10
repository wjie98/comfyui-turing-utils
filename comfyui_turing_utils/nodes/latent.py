"""Video latent masks and compositing with explicit temporal mapping."""

from __future__ import annotations

from comfy_api.latest import io
from ..media.temporal import VIDEO_MASK_SPECS
from ..media.latent_masks import (
    _video_samples,
    composite_video_latents,
    _map_video_mask,
)


class SetVideoLatentNoiseMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsSetVideoLatentNoiseMask",
            display_name="Set Video Latent Noise Mask",
            category="Turing Utils/Mask",
            description=(
                "Set a video-only noise mask. Matching mask/latent time counts map directly; "
                "otherwise merge image-frame masks by the selected VAE's temporal groups. "
                "Uses maximum/union in time and space. Mismatched counts are errors, "
                "never temporally interpolated, padded, or truncated."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip="Standalone video latent [B,C,T,H,W]. Separate audio/video latents first.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="[frames,H,W], or [B,frames,H,W]. Missing mask preserves the input latent unchanged. 0 preserves, 1 redraws.",
                ),
                io.Combo.Input(
                    "type",
                    options=list(VIDEO_MASK_SPECS),
                    default="minimax",
                    tooltip=(
                        "Used only when frame counts differ. wan (2.1/2.2), hunyuan_video and "
                        "hunyuan_video_15: first frame then groups of 4; ltxv (LTX-Video/LTX-2 video): "
                        "groups of 8; mochi: groups of 6. minimax (H3 video): [1,4,4,4,4]*n+[1,4]."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, samples, mask=None, type="minimax"):
        video = _video_samples(samples)
        output = samples.copy()
        if mask is None:
            return io.NodeOutput(output)
        output["noise_mask"] = _map_video_mask(video, mask, type)
        return io.NodeOutput(output)


class VideoLatentCompositeMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoLatentCompositeMasked",
            display_name="Video Latent Composite Masked",
            category="Turing Utils/Mask",
            description=(
                "Composite matching standalone video latents using hard coverage. Union the replacement "
                "latent's inherited noise_mask with the optional mask; every nonzero value means replace, "
                "not opacity. The same binary region is saved as noise_mask. With neither mask, replace all. "
                "The original latent's mask is ignored. No resizing or feathering of latent samples."
            ),
            inputs=[
                io.Latent.Input(
                    "original_latent",
                    tooltip="Clean original high-resolution video [B,C,T,H,W]. Mask=0 preserves these samples. Its metadata is retained, but its noise_mask is replaced.",
                ),
                io.Latent.Input(
                    "replacement_latent",
                    tooltip="Clean replacement video, e.g. upscaled first-pass denoised_output. Shape must match original_latent. Its noise_mask is inherited; mask=1 uses these samples.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Additional [frames,H,W] or [B,frames,H,W] mask. Its nonzero coverage is unioned with the inherited mask; even small positive values mean full replacement.",
                ),
                io.Combo.Input(
                    "type",
                    options=list(VIDEO_MASK_SPECS),
                    default="minimax",
                    tooltip="Same temporal mapping as Set Video Latent Noise Mask. Only used to map additional image-frame masks, never to retime inherited latent masks.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, original_latent, replacement_latent, type="minimax", mask=None):
        return io.NodeOutput(
            composite_video_latents(original_latent, replacement_latent, type, mask)
        )
