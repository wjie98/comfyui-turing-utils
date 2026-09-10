"""ComfyUI video-padding and AV-latent nodes for MiniMax H3."""

from __future__ import annotations

import torch

import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy_api.latest import io
from ..adapters.minimax.block_cache import install_minimax_block_cache
from ..adapters.minimax.latent_noise import add_h3_noise_for_resampling
from ..adapters.minimax.latent_upscaler import (
    load_h3_latent_upscaler,
    upscale_h3_latent,
)
from .wan import WanVideoFramesPadding, repeat_last_frame


H3LatentUpscaleModel = io.Custom("TURING_UTILS_H3_LATENT_UPSCALE_MODEL")


def _is_h3_frame_count(frame_count: int) -> bool:
    return frame_count >= 5 and frame_count % 17 == 5


def _ceil_h3_frame_count(frame_count: int) -> int:
    frame_count = max(int(frame_count), 5)
    remainder = (frame_count - 5) % 17
    return frame_count if remainder == 0 else frame_count + 17 - remainder


class MiniMaxH3BlockCachePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "profile": (
                    ["auto", "standard", "4-step LoRA", "8-step LoRA"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Auto selects a conservative preset only for an exact "
                            "4- or 8-step trajectory; all other schedules use standard."
                        ),
                    },
                ),
                "cache_device": (
                    ["auto", "gpu", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Auto honors ComfyUI's free-VRAM reserve; skipped blocks "
                            "also stay out of its Dynamic VRAM prefetch queue. CPU uses "
                            "ComfyUI-managed pinned memory when available."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Turing Utils/optimization"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Profile-driven MiniMax H3 transformer-block caching. Auto selects the "
        "dedicated 4-step or 8-step policy only for exact matching trajectories; "
        "cache storage participates in ComfyUI's VRAM and pinned-memory lifecycle."
    )

    def patch(self, model, profile, cache_device):
        return (
            install_minimax_block_cache(
                model,
                profile,
                cache_device,
            ),
        )


class MiniMaxH3VideoFramesPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_frame_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Use 0 to round up to MiniMax H3's 17*n+5 frame grid.",
                    },
                ),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "length", "input_length")
    FUNCTION = "pad"
    CATEGORY = "Turing Utils/video"
    TITLE = "MiniMax H3 Video Frames Padding"

    def pad(self, image, target_frame_count: int, mask=None):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE shaped [frames, height, width, channels], got {tuple(image.shape)}")
        frame_count = int(image.shape[0])
        if frame_count < 1:
            raise ValueError("MiniMax H3 Video Frames Padding requires at least one input frame")
        height = int(image.shape[1])
        width = int(image.shape[2])
        mask = WanVideoFramesPadding._validate_mask(mask, frame_count, height, width)

        if target_frame_count == 0:
            output_length = _ceil_h3_frame_count(frame_count)
        else:
            output_length = int(target_frame_count)
            if not _is_h3_frame_count(output_length):
                raise ValueError(f"target_frame_count must be 0 or 17*n+5; got {target_frame_count}")
            if output_length < frame_count:
                raise ValueError(
                    f"target_frame_count={output_length} is shorter than the input video "
                    f"({frame_count} frames). Trim upstream if needed."
                )

        pad_count = output_length - frame_count
        image = repeat_last_frame(image, pad_count)
        if mask is not None:
            mask = repeat_last_frame(mask, pad_count)
        return (image, mask, width, height, output_length, frame_count)


class H3ConcatAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3ConcatAVLatent",
            display_name="H3 Concat AV Latent",
            category="Turing Utils/latent",
            description="Merge MiniMax H3 video and audio latents into one native AV latent. An existing AV video input keeps its video stream and replaces its audio stream.",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[io.Latent.Output(display_name="av_latent")],
        )

    @staticmethod
    def _samples(latent, name: str):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError(f"{name} must be a LATENT dictionary")
        return latent["samples"]

    @staticmethod
    def _validate_video(video: torch.Tensor):
        if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 24:
            shape = tuple(video.shape) if hasattr(video, "shape") else type(video).__name__
            raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {shape}")
        if int(video.shape[0]) < 1 or int(video.shape[2]) < 1:
            raise ValueError(f"H3 video latent batch and time dimensions must be non-empty, got {tuple(video.shape)}")
        return video

    @staticmethod
    def _validate_audio(audio: torch.Tensor):
        if not torch.is_tensor(audio) or audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2):
            shape = tuple(audio.shape) if hasattr(audio, "shape") else type(audio).__name__
            raise ValueError(f"Expected H3 audio latent [B,32,2,T], got {shape}")
        if int(audio.shape[0]) < 1 or int(audio.shape[-1]) < 1:
            raise ValueError(f"H3 audio latent batch and time dimensions must be non-empty, got {tuple(audio.shape)}")
        return audio

    @classmethod
    def _av_streams(cls, samples):
        if not getattr(samples, "is_nested", False):
            raise ValueError("Expected a MiniMax H3 nested video/audio latent")
        streams = list(samples.unbind())
        if len(streams) != 2:
            raise ValueError(f"Expected exactly two H3 latent streams, got {len(streams)}")
        video = cls._validate_video(streams[0])
        audio = cls._validate_audio(streams[1])
        if int(video.shape[0]) != int(audio.shape[0]):
            raise ValueError("H3 video and audio latent batch sizes must match")
        return video, audio

    @staticmethod
    def _mask_streams(mask):
        if not getattr(mask, "is_nested", False):
            raise ValueError("An H3 AV latent noise_mask must contain nested video/audio masks")
        streams = list(mask.unbind())
        if len(streams) != 2:
            raise ValueError(f"Expected exactly two H3 noise-mask streams, got {len(streams)}")
        return streams

    @classmethod
    def fit_audio(cls, reference, audio, noise_mask):
        """Trim or zero-pad replacement audio to an existing H3 audio stream."""
        cls._validate_audio(reference)
        cls._validate_audio(audio)
        if tuple(reference.shape) == tuple(audio.shape):
            return audio, noise_mask
        if tuple(reference.shape[:-1]) != tuple(audio.shape[:-1]):
            raise ValueError(
                f"H3 audio latent {tuple(audio.shape)} cannot be fitted to {tuple(reference.shape)}"
            )

        length = int(reference.shape[-1])
        if noise_mask is not None:
            noise_mask = comfy.utils.reshape_mask(noise_mask, audio.shape)
        if int(audio.shape[-1]) > length:
            audio = audio.narrow(-1, 0, length)
            if noise_mask is not None:
                noise_mask = noise_mask.narrow(-1, 0, length)
        else:
            pad = torch.zeros_like(audio.narrow(-1, 0, 1)).repeat(1, 1, 1, length - int(audio.shape[-1]))
            audio = torch.cat((audio, pad), dim=-1)
            if noise_mask is not None:
                noise_mask = torch.cat((noise_mask, torch.ones_like(pad)), dim=-1)
        return audio, noise_mask

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        video_container = cls._samples(video_latent, "video_latent")
        audio_samples = cls._validate_audio(cls._samples(audio_latent, "audio_latent"))
        video_noise_mask = video_latent.get("noise_mask")
        audio_noise_mask = audio_latent.get("noise_mask")

        if getattr(video_container, "is_nested", False):
            video_samples, reference_audio = cls._av_streams(video_container)
            if video_noise_mask is not None:
                video_noise_mask = cls._mask_streams(video_noise_mask)[0]
            audio_samples, audio_noise_mask = cls.fit_audio(reference_audio, audio_samples, audio_noise_mask)
        else:
            video_samples = cls._validate_video(video_container)
            if getattr(video_noise_mask, "is_nested", False):
                raise ValueError("A standalone H3 video latent must use a standalone video noise_mask")

        if getattr(audio_noise_mask, "is_nested", False):
            raise ValueError("A standalone H3 audio latent must use a standalone audio noise_mask")
        if int(video_samples.shape[0]) != int(audio_samples.shape[0]):
            raise ValueError("H3 video and audio latent batch sizes must match")

        output = video_latent.copy()
        output.update(audio_latent)
        output["samples"] = comfy.nested_tensor.NestedTensor((video_samples, audio_samples))
        if video_noise_mask is not None or audio_noise_mask is not None:
            if video_noise_mask is None:
                video_noise_mask = torch.ones_like(video_samples)
            if audio_noise_mask is None:
                audio_noise_mask = torch.ones_like(audio_samples)
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))
        else:
            output.pop("noise_mask", None)
        return io.NodeOutput(output)


class H3SeparateAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3SeparateAVLatent",
            display_name="H3 Separate AV Latent",
            category="Turing Utils/latent",
            description="Split a native MiniMax H3 AV latent into standalone video and audio latents.",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        samples = H3ConcatAVLatent._samples(av_latent, "av_latent")
        video_samples, audio_samples = H3ConcatAVLatent._av_streams(samples)
        video_latent = av_latent.copy()
        video_latent["samples"] = video_samples
        audio_latent = av_latent.copy()
        audio_latent["samples"] = audio_samples

        noise_mask = av_latent.get("noise_mask")
        if noise_mask is not None:
            video_mask, audio_mask = H3ConcatAVLatent._mask_streams(noise_mask)
            video_latent["noise_mask"] = video_mask
            audio_latent["noise_mask"] = audio_mask
        return io.NodeOutput(video_latent, audio_latent)


class H3AddNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3AddNoise",
            display_name="H3 Add Noise",
            category="Turing Utils/latent",
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


class MiniMaxH3LatentUpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMiniMaxH3LatentUpscaleModelLoader",
            display_name="Load MiniMax H3 Latent Upscaler",
            category="Turing Utils/loaders",
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


class MiniMaxH3LatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMiniMaxH3LatentUpscale",
            display_name="MiniMax H3 Latent Upscale",
            category="Turing Utils/latent",
            description=(
                "Learned spatial pixel-count upscale for MiniMax H3 AV latents. The video stream and "
                "optional FL2AV keyframe latents are enlarged together; audio and Ref2AV "
                "references remain unchanged."
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
