"""MiniMax H3 latent noise and spatial upscale nodes."""

from __future__ import annotations

import folder_paths
from comfy_api.latest import io
from ..adapters.minimax.latent_noise import add_h3_noise_for_resampling
from ..adapters.minimax.latent_upscaler import (
    load_h3_latent_upscaler,
    upscale_h3_latent,
)


H3LatentUpscaleModel = io.Custom("TURING_UTILS_H3_LATENT_UPSCALE_MODEL")


class H3AddNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3AddNoise",
            display_name="H3 Add Noise",
            category="Turing Utils/MiniMax H3",
            description=(
                "Re-noise clean H3 latents for continuation with DisableNoise. "
                "Processes every supplied stream, including both video and audio in an AV latent. "
                "Uses the continuation video schedule for standalone audio too; split AV upstream "
                "if only one stream should change."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The H3 MODEL used by the continuation sampler, including its sampling patches."),
                io.Noise.Input("noise", tooltip="Connect RandomNoise to control the new noise seed."),
                io.Sigmas.Input("sigmas", tooltip="Remaining sampling schedule. Its FIRST sigma is the target level (0 <= sigma < 1)."),
                io.Latent.Input("latent_image", tooltip="Clean x0: denoised_output, VAE-encoded latent, or upscaled clean latent. Do not pass an already-noisy sampler output."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, model, noise, sigmas, latent_image) -> io.NodeOutput:
        return io.NodeOutput(add_h3_noise_for_resampling(model, noise, sigmas, latent_image))


class _H3UpscaleLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TuringUtilsH3UpscaleLoader",
            display_name="H3 Upscale Loader (Internal)",
            is_dev_only=True,
            category="",
            description=(
                "Load an attention-free 3D MiniMax H3 latent upscaler from "
                "models/latent_upscale_models with ComfyUI-managed VRAM offloading."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("latent_upscale_models"),
                ),
                io.Combo.Input(
                    "precision",
                    options=["auto", "fp16", "bf16", "fp32"],
                    default="auto",
                    tooltip=(
                        "Auto selects the efficient native compute type for the current device. "
                        "FP16 is normally preferred on Turing."
                    ),
                ),
            ],
            outputs=[H3LatentUpscaleModel.Output(display_name="upscale_model")],
        )

    @classmethod
    def execute(cls, model_name: str, precision: str) -> io.NodeOutput:
        return io.NodeOutput(load_h3_latent_upscaler(model_name, precision))


class _H3UpscaleApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TuringUtilsH3UpscaleApply",
            is_dev_only=True,
            display_name="MiniMax H3 Latent Upscale (Internal Apply)",
            category="",
            description=(
                "Learned spatial pixel-count upscale for MiniMax H3 AV latents. The video stream and "
                "optional FL2AV keyframe latents are enlarged together; audio and Ref2AV "
                "references remain unchanged. Video noise_mask uses conservative spatial maximum "
                "coverage; time and audio masks are preserved."
            ),
            inputs=[
                H3LatentUpscaleModel.Input("upscale_model"),
                io.Latent.Input("latent"),
                io.Conditioning.Input("conditioning", optional=True, tooltip="Optional FL2AV conditioning whose first/last keyframe latents should follow the same spatial upscale."),
                io.Float.Input(
                    "scale",
                    default=2.0,
                    min=1.0,
                    max=16.0,
                    step=0.1,
                    tooltip=(
                        "Total spatial pixel/token multiplier. For example, 2.0 scales H/W "
                        "by sqrt(2), while 4.0 scales H/W by 2. Time and audio are preserved. "
                        "Output H/W are rounded up to H3's 2x2 latent patch grid."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Conditioning.Output(display_name="conditioning"),
                io.Int.Output("width", tooltip="Aligned pixel width produced by the upscaled H3 video latent."),
                io.Int.Output("height", tooltip="Aligned pixel height produced by the upscaled H3 video latent."),
            ],
        )

    @classmethod
    def execute(cls, upscale_model, latent, conditioning=None, scale: float = 2.0) -> io.NodeOutput:
        output_latent, output_conditioning, width, height = upscale_h3_latent(
            upscale_model,
            latent,
            conditioning,
            scale,
        )
        return io.NodeOutput(output_latent, output_conditioning, width, height)


class MiniMaxH3LatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = _H3UpscaleApply.define_schema()
        schema.node_id = "TuringUtilsMiniMaxH3LatentUpscale"
        schema.display_name = "MiniMax H3 Latent Upscale"
        schema.is_dev_only = False
        schema.category = "Turing Utils/MiniMax H3"
        schema.inputs = _H3UpscaleLoader.define_schema().inputs + schema.inputs[1:]
        next(item for item in schema.inputs if item.id == "precision").advanced = True
        return schema

    @classmethod
    def execute(cls, model_name, latent, precision="auto", conditioning=None, scale=2.0):
        return _H3UpscaleApply.execute(
            load_h3_latent_upscaler(model_name, precision), latent, conditioning, scale)
