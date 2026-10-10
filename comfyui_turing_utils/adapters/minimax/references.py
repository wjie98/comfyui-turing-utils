"""Pure MiniMax H3 multimodal reference preparation services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

import torch
import torchaudio

from ...profiling import WORKFLOW_TIMELINE

import comfy.utils


H3_MODEL_FPS = 24.0
H3_QWEN_VIDEO_FPS = 2.0
H3_PIXEL_ALIGNMENT = 32
H3_SPATIAL_DOWNSCALE = 16
H3_MAX_KEYFRAME_REFERENCES = 32


@dataclass(frozen=True)
class H3ReferenceManifest:
    first_frame: bool = False
    last_frame: bool = False
    image_count: int = 0
    video_audio: tuple[bool, ...] = ()
    audio_count: int = 0


@dataclass(frozen=True)
class H3KeyframeReferenceData:
    image: torch.Tensor
    latent: torch.Tensor


@dataclass(frozen=True)
class H3ImageReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3VideoReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3AudioReferenceData:
    items: tuple[dict, ...]


@dataclass(frozen=True)
class H3SemanticReferenceData:
    conditioning: object
    manifest: H3ReferenceManifest


def _samples(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("latent must be a LATENT dictionary")
    return latent["samples"]


def _video_latent(latent) -> torch.Tensor:
    samples = _samples(latent)
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
        if not streams:
            raise ValueError("H3 AV latent contains no video stream")
        samples = streams[0]
    if not torch.is_tensor(samples) or samples.ndim != 5 or int(samples.shape[1]) != 24:
        shape = (
            tuple(samples.shape)
            if hasattr(samples, "shape")
            else type(samples).__name__
        )
        raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {shape}")
    return samples


def _frame_count_from_latent_t(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t == 1:
        return 1
    if latent_t >= 2 and (latent_t - 2) % 5 == 0:
        return ((latent_t - 2) // 5) * 17 + 5
    raise ValueError(
        f"H3 video latent T={latent_t} is outside the canonical 5*k+2 temporal grid"
    )


def h3_latent_info(latent) -> tuple[int, int, int, float]:
    video = _video_latent(latent)
    return (
        int(video.shape[-1]) * H3_SPATIAL_DOWNSCALE,
        int(video.shape[-2]) * H3_SPATIAL_DOWNSCALE,
        _frame_count_from_latent_t(int(video.shape[2])),
        H3_MODEL_FPS,
    )


def _validate_pixels(image: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(image)
        or image.ndim != 4
        or int(image.shape[0]) < 1
        or int(image.shape[-1]) < 3
    ):
        shape = tuple(image.shape) if hasattr(image, "shape") else type(image).__name__
        raise ValueError(
            f"{name} must be IMAGE frames [N,H,W,C] with at least three channels; got {shape}"
        )
    return image[..., :3]


def _crop_to_alignment(image: torch.Tensor, name: str) -> torch.Tensor:
    image = _validate_pixels(image, name)
    height, width = int(image.shape[1]), int(image.shape[2])
    aligned_width = width - width % H3_PIXEL_ALIGNMENT
    aligned_height = height - height % H3_PIXEL_ALIGNMENT
    if aligned_width < H3_PIXEL_ALIGNMENT or aligned_height < H3_PIXEL_ALIGNMENT:
        raise ValueError(
            f"{name} must be at least {H3_PIXEL_ALIGNMENT}x{H3_PIXEL_ALIGNMENT}; "
            f"got {width}x{height}"
        )
    left = (width - aligned_width) // 2
    top = (height - aligned_height) // 2
    return image[:, top : top + aligned_height, left : left + aligned_width]


def _align_keyframe_pixels(image: torch.Tensor, latent, name: str) -> torch.Tensor:
    image = _validate_pixels(image, name)
    if latent is None:
        return _crop_to_alignment(image, name)
    width, height, _, _ = h3_latent_info(latent)
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, "lanczos", "center"
    ).movedim(1, -1)


def _align_reference_pixels(
    image: torch.Tensor,
    latent,
    name: str,
    megapixels: float,
) -> torch.Tensor:
    image = _validate_pixels(image, name)
    height, width = int(image.shape[1]), int(image.shape[2])
    if latent is None:
        megapixels = float(megapixels)
        if not math.isfinite(megapixels) or megapixels <= 0.0:
            raise ValueError("megapixels must be finite and positive")
        target_area = megapixels * 1_000_000.0
    else:
        target_width, target_height, _, _ = h3_latent_info(latent)
        target_area = target_width * target_height
    scale = min(1.0, math.sqrt(target_area / (width * height)))
    aligned_width = max(
        H3_PIXEL_ALIGNMENT,
        round(width * scale / H3_PIXEL_ALIGNMENT) * H3_PIXEL_ALIGNMENT,
    )
    aligned_height = max(
        H3_PIXEL_ALIGNMENT,
        round(height * scale / H3_PIXEL_ALIGNMENT) * H3_PIXEL_ALIGNMENT,
    )
    return comfy.utils.common_upscale(
        image.movedim(-1, 1),
        aligned_width,
        aligned_height,
        "lanczos",
        "disabled",
    ).movedim(1, -1)


def _validate_visual_latent(value, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 5
        or int(value.shape[0]) != 1
        or int(value.shape[1]) != 24
    ):
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} VAE must produce [1,24,T,H,W], got {shape}")
    return value


def _encode_visual(vae, pixels: torch.Tensor, name: str) -> torch.Tensor:
    device = getattr(vae, "load_device", pixels.device)
    value = WORKFLOW_TIMELINE.call(
        f"visual_reference_encode:{name}", device, vae.encode, pixels
    )
    return _validate_visual_latent(value, name)


def _encode_audio(audio_vae, audio, name: str) -> torch.Tensor:
    if (
        not isinstance(audio, Mapping)
        or "waveform" not in audio
        or "sample_rate" not in audio
    ):
        raise ValueError(
            f"{name} must be an AUDIO mapping with waveform and sample_rate"
        )
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = (
            tuple(waveform.shape)
            if hasattr(waveform, "shape")
            else type(waveform).__name__
        )
        raise ValueError(f"{name} waveform must be [B,C,L], got {shape}")
    source_rate = int(audio["sample_rate"])
    target_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    encode_input = waveform[:1].movedim(1, -1)
    device = getattr(audio_vae, "load_device", encode_input.device)
    value = WORKFLOW_TIMELINE.call(
        f"audio_reference_encode:{name}",
        device,
        audio_vae.encode,
        encode_input,
    )
    if (
        not torch.is_tensor(value)
        or value.ndim != 4
        or tuple(value.shape[1:3]) != (32, 2)
    ):
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        raise ValueError(f"{name} Audio VAE must produce [1,32,2,T], got {shape}")
    return value


def _natural_key(value: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", str(value))
    return (
        str(value)[: match.start()] if match else str(value),
        int(match.group(1)) if match else -1,
    )


def _dynamic_entries(values) -> tuple[tuple[str, object], ...]:
    if not isinstance(values, dict):
        return ()
    return tuple(
        (str(key), value)
        for key, value in sorted(
            values.items(), key=lambda item: _natural_key(str(item[0]))
        )
        if value is not None
    )


def _dynamic_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else -1


def _trim_h3_reference_video(frames: torch.Tensor) -> torch.Tensor:
    frame_count = int(frames.shape[0])
    if frame_count < 5:
        raise ValueError(
            "H3 reference videos require at least five input frames at 24 FPS"
        )
    valid_count = frame_count - (frame_count - 5) % 17
    return frames[:valid_count]


def _manifest(
    first_frame: H3KeyframeReferenceData | None,
    last_frame: H3KeyframeReferenceData | None,
    image_reference: H3ImageReferenceData | None,
    video_reference: H3VideoReferenceData | None,
    audio_reference: H3AudioReferenceData | None,
) -> H3ReferenceManifest:
    return H3ReferenceManifest(
        first_frame=first_frame is not None,
        last_frame=last_frame is not None,
        image_count=len(image_reference.items) if image_reference else 0,
        video_audio=tuple(
            item["audio_latent"] is not None for item in video_reference.items
        )
        if video_reference
        else (),
        audio_count=len(audio_reference.items) if audio_reference else 0,
    )


def _reference_presentation(
    image_reference: H3ImageReferenceData | None,
    video_reference: H3VideoReferenceData | None,
    audio_reference: H3AudioReferenceData | None,
) -> list[dict]:
    presentation = []
    if image_reference:
        presentation.extend(
            {"type": "image", "data": item["image"]} for item in image_reference.items
        )
    if video_reference:
        for item in video_reference.items:
            if item["audio_latent"] is not None:
                presentation.append({"type": "audio"})
            presentation.append(
                {
                    "type": "video",
                    "data": item["qwen_frames"],
                    "timestamps": item["timestamps"],
                }
            )
    if audio_reference:
        presentation.extend({"type": "audio"} for _ in audio_reference.items)
    return presentation


def _keyframe_reference(value, name: str):
    if value is None:
        return None
    if not isinstance(value, H3KeyframeReferenceData):
        raise ValueError(f"{name} must come from H3 Keyframe Reference")
    return value


def _tokenize_semantic(clip, prompt: str, first_frame, last_frame, presentation):
    images = [item.image for item in (first_frame, last_frame) if item is not None]
    if not images:
        return (
            clip.tokenize(prompt, minimax_ref_items=presentation)
            if presentation
            else clip.tokenize(prompt)
        )
    if not presentation:
        return clip.tokenize(prompt, images=images)

    # Upstream exposes the official FL2VA and Ref2VA presentations as exclusive
    # arguments. Their raw H3 token streams are composable before Qwen encoding.
    keyframe_tokens = clip.tokenize("", images=images)
    reference_tokens = clip.tokenize(prompt, minimax_ref_items=presentation)
    key = "qwen3vl_32b"
    try:
        keyframe_rows = keyframe_tokens[key]
        reference_rows = reference_tokens[key]
        if len(keyframe_rows) != 1 or len(reference_rows) != 1:
            raise ValueError
        return {key: [keyframe_rows[0] + reference_rows[0]]}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "clip is not a compatible MiniMax H3 Qwen3-VL encoder"
        ) from error


def _reference_blocks(
    image_reference: H3ImageReferenceData | None,
    video_reference: H3VideoReferenceData | None,
    audio_reference: H3AudioReferenceData | None,
) -> list[dict]:
    refs = []
    if image_reference:
        refs.extend(
            {
                "kind": "image",
                "latent_h": int(item["latent"].shape[-2]),
                "latent_w": int(item["latent"].shape[-1]),
                "latent": item["latent"],
            }
            for item in image_reference.items
        )
    if video_reference:
        refs.extend(
            {
                "kind": "video_audio" if item["audio_latent"] is not None else "video",
                "latent_t": int(item["latent"].shape[2]),
                "latent_h": int(item["latent"].shape[-2]),
                "latent_w": int(item["latent"].shape[-1]),
                "ref_audio_t": int(item["audio_latent"].shape[-1])
                if item["audio_latent"] is not None
                else 0,
                "latent": item["latent"],
                "audio_latent": item["audio_latent"],
            }
            for item in video_reference.items
        )
    if audio_reference:
        refs.extend(
            {
                "kind": "audio",
                "ref_audio_t": int(item["audio_latent"].shape[-1]),
                "audio_latent": item["audio_latent"],
            }
            for item in audio_reference.items
        )
    return refs


__all__ = [
    "H3_MODEL_FPS",
    "H3_QWEN_VIDEO_FPS",
    "H3_PIXEL_ALIGNMENT",
    "H3_SPATIAL_DOWNSCALE",
    "H3_MAX_KEYFRAME_REFERENCES",
    "H3ReferenceManifest",
    "H3KeyframeReferenceData",
    "H3ImageReferenceData",
    "H3VideoReferenceData",
    "H3AudioReferenceData",
    "H3SemanticReferenceData",
    "h3_latent_info",
]
