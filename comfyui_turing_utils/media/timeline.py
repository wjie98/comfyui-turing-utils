"""Continuation audio timelines and explicit sequence-mask broadcasting."""

from __future__ import annotations

import math
from fractions import Fraction

import torch
import torch.nn.functional as F
import torchaudio
from collections.abc import Mapping


_DEFAULT_AUDIO_SAMPLE_RATE = 32000
_AAC_FRAME_SAMPLES = 1024


def _frame_rate(value: float) -> Fraction:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("frame_rate must be finite and positive")
    return Fraction(str(value)).limit_denominator(1_000_000)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _validate_images(images, name: str) -> torch.Tensor:
    if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[-1]) < 3:
        shape = (
            tuple(images.shape) if hasattr(images, "shape") else type(images).__name__
        )
        raise ValueError(
            f"{name} must be IMAGE frames [frames,height,width,channels], got {shape}"
        )
    if any(int(size) < 1 for size in images.shape[:3]):
        raise ValueError(f"{name} must contain at least one non-empty frame")
    return images[..., :3]


def _validate_audio(audio, name: str):
    if (
        not isinstance(audio, Mapping)
        or "waveform" not in audio
        or "sample_rate" not in audio
    ):
        raise ValueError(f"{name} must be an AUDIO dictionary")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = (
            tuple(waveform.shape)
            if hasattr(waveform, "shape")
            else type(waveform).__name__
        )
        raise ValueError(
            f"{name} waveform must be [batch,channels,samples], got {shape}"
        )
    if int(waveform.shape[0]) != 1 or int(waveform.shape[1]) < 1:
        raise ValueError(f"{name} must contain one batch and at least one channel")
    if sample_rate <= 0:
        raise ValueError(f"{name} sample_rate must be positive")
    if not waveform.dtype.is_floating_point:
        waveform = waveform.float()
    return waveform, sample_rate


def _fit_waveform(waveform: torch.Tensor, length: int) -> torch.Tensor:
    current = int(waveform.shape[-1])
    if current >= length:
        return waveform[..., :length]
    padding = torch.zeros(
        (*waveform.shape[:-1], length - current),
        device=waveform.device,
        dtype=waveform.dtype,
    )
    return torch.cat((waveform, padding), dim=-1)


def _resample_waveform(
    waveform: torch.Tensor, source_rate: int, target_rate: int
) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _match_audio_channels(
    waveform: torch.Tensor, channels: int, name: str
) -> torch.Tensor:
    current = int(waveform.shape[1])
    if current == channels:
        return waveform
    if current == 1:
        return waveform.expand(-1, channels, -1)
    raise ValueError(
        f"{name} has {current} channels and cannot be matched to {channels}"
    )


def _audio_timeline(
    prefix_audio,
    body_audio,
    prefix_frames: int,
    body_frames: int,
    frame_rate: Fraction,
    mode: str,
):
    prefix = (
        _validate_audio(prefix_audio, "prefix_audio")
        if prefix_audio is not None
        else None
    )
    body = _validate_audio(body_audio, "body_audio") if body_audio is not None else None
    sample_rate = (
        prefix[1]
        if prefix is not None
        else body[1]
        if body is not None
        else _DEFAULT_AUDIO_SAMPLE_RATE
    )
    prefix_length = _ceil_fraction(
        Fraction(prefix_frames * sample_rate, 1) / frame_rate
    )
    body_length = _ceil_fraction(Fraction(body_frames * sample_rate, 1) / frame_rate)

    present = [item[0] for item in (prefix, body) if item is not None]
    if present:
        reference = present[0]
        channels = max(int(waveform.shape[1]) for waveform in present)
        batch = int(reference.shape[0])
        device = reference.device
        dtype = reference.dtype
    else:
        channels = 2
        batch = 1
        device = torch.device("cpu")
        dtype = torch.float32

    def prepare(item, length: int, name: str):
        if item is None:
            return torch.zeros((batch, channels, length), device=device, dtype=dtype)
        waveform, source_rate = item
        waveform = _resample_waveform(waveform, source_rate, sample_rate)
        waveform = _match_audio_channels(waveform, channels, name).to(
            device=device, dtype=dtype
        )
        return _fit_waveform(waveform, length)

    body_waveform = prepare(body, body_length, "body_audio")
    if mode == "concat":
        prefix_waveform = prepare(prefix, prefix_length, "prefix_audio")
        waveform = torch.cat((prefix_waveform, body_waveform), dim=-1)
    elif mode == "replace":
        waveform = body_waveform.clone()
        if prefix is not None and prefix_length > 0:
            prefix_waveform = prepare(prefix, prefix_length, "prefix_audio")
            waveform[..., :prefix_length] = prefix_waveform
    else:
        raise ValueError(f"Unknown continuation mode: {mode!r}")
    return {"waveform": waveform, "sample_rate": sample_rate}, prefix_length


def _prepare_mask(
    mask, frames: int, height: int, width: int, default: float, device
) -> torch.Tensor:
    if mask is None:
        return torch.full(
            (frames, height, width), default, device=device, dtype=torch.float32
        )
    if not torch.is_tensor(mask) or mask.ndim not in (2, 3):
        shape = tuple(mask.shape) if hasattr(mask, "shape") else type(mask).__name__
        raise ValueError(
            f"MASK must be [height,width] or [frames,height,width], got {shape}"
        )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if int(mask.shape[0]) == 1 and frames != 1:
        mask = mask.expand(frames, -1, -1)
    elif int(mask.shape[0]) != frames:
        raise ValueError(
            f"MASK contains {int(mask.shape[0])} frames, expected 1 or {frames}"
        )
    if not mask.dtype.is_floating_point and mask.dtype != torch.bool:
        mask = mask.float()
    if not torch.isfinite(mask).all() or mask.amin() < 0 or mask.amax() > 1:
        raise ValueError("MASK values must be finite and within [0,1]")
    mask = mask.to(device=device, dtype=torch.float32)
    if tuple(mask.shape[-2:]) != (height, width):
        mask = F.interpolate(
            mask.unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
    return mask.clone()


def _trim_info_values(trim_info):
    if not isinstance(trim_info, dict) or int(trim_info.get("version", 0)) != 1:
        raise ValueError("trim_info must come from Video Continuation Concat")
    prefix_frames = int(trim_info["prefix_frames"])
    trim_frames = int(trim_info.get("trim_frames", prefix_frames))
    rate = Fraction(
        int(trim_info["frame_rate_numerator"]),
        int(trim_info["frame_rate_denominator"]),
    )
    if prefix_frames < 0 or not 0 <= trim_frames <= prefix_frames or rate <= 0:
        raise ValueError("trim_info contains an invalid prefix boundary")
    return prefix_frames, trim_frames, rate


def concat_video_continuation(
    prefix_images=None,
    prefix_mask=None,
    prefix_audio=None,
    body_images=None,
    body_mask=None,
    body_audio=None,
    frame_rate=24.0,
    mode="concat",
):
    if body_images is None:
        raise ValueError("body_images is required")
    if mode not in ("concat", "replace"):
        raise ValueError(f"Unknown continuation mode: {mode!r}")
    body_images = _validate_images(body_images, "body_images")
    rate = _frame_rate(frame_rate)
    body_frames, height, width, _ = body_images.shape
    prefix_frames = 0
    if prefix_images is not None:
        prefix_images = _validate_images(prefix_images, "prefix_images")
        if tuple(prefix_images.shape[1:3]) != (height, width):
            raise ValueError(
                f"prefix_images and body_images must have the same height/width; got "
                f"{tuple(prefix_images.shape[1:3])} and {(height, width)}"
            )
        prefix_frames = int(prefix_images.shape[0])
        prefix_images = prefix_images.to(
            device=body_images.device, dtype=body_images.dtype
        )
    elif prefix_mask is not None or prefix_audio is not None:
        raise ValueError("prefix_mask and prefix_audio require prefix_images")

    if mode == "replace" and prefix_frames > int(body_frames):
        raise ValueError(
            f"replace mode cannot fit {prefix_frames} prefix frames into a {int(body_frames)}-frame body"
        )

    prefix_mask = _prepare_mask(
        prefix_mask, prefix_frames, height, width, 0.0, body_images.device
    )
    body_mask = _prepare_mask(
        body_mask, int(body_frames), height, width, 1.0, body_images.device
    )
    if mode == "concat":
        images = (
            body_images
            if prefix_images is None
            else torch.cat((prefix_images, body_images), dim=0)
        )
        mask = torch.cat((prefix_mask, body_mask), dim=0)
        trim_frames = prefix_frames
    else:
        images = (
            body_images
            if prefix_images is None
            else torch.cat((prefix_images, body_images[prefix_frames:]), dim=0)
        )
        mask = body_mask.clone()
        if prefix_frames > 0:
            mask[:prefix_frames] = prefix_mask
        trim_frames = prefix_frames
    audio, prefix_audio_samples = _audio_timeline(
        prefix_audio,
        body_audio,
        prefix_frames,
        int(body_frames),
        rate,
        mode,
    )
    trim_info = {
        "version": 1,
        "composition_mode": mode,
        "prefix_frames": prefix_frames,
        "trim_frames": trim_frames,
        "body_frames": int(body_frames),
        "frame_rate_numerator": rate.numerator,
        "frame_rate_denominator": rate.denominator,
        "prefix_audio_samples": prefix_audio_samples,
        "total_audio_samples": int(audio["waveform"].shape[-1]),
        "audio_sample_rate": int(audio["sample_rate"]),
        "prefix_audio_present": prefix_audio is not None,
    }
    return images, mask, audio, trim_info
