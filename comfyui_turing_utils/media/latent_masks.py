"""Explicit video-latent mask mapping and validation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .temporal import VIDEO_MASK_SPECS, _image_frame_groups


def _video_samples(samples):
    video = samples["samples"]
    if getattr(video, "is_nested", False):
        raise ValueError(
            "Video mask nodes accept standalone video only. "
            "Use a Separate AV Latent node before this node, then concatenate afterward."
        )
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError(
            "Expected a video latent shaped [B,C,T,H,W], not an image or audio latent."
        )
    if any(size < 1 for size in video.shape):
        raise ValueError("Video latent dimensions must be non-empty.")
    return video


def _validate_mask_values(mask):
    if mask.is_complex() or any(size < 1 for size in mask.shape):
        raise ValueError("MASK must contain non-empty, real-valued frames.")
    if not torch.isfinite(mask).all() or mask.amin() < 0 or mask.amax() > 1:
        raise ValueError(
            "MASK values must be finite and within [0,1]; 0 preserves and 1 redraws."
        )


def _map_video_mask(video, mask, model_type, *, hard=False):
    if model_type not in VIDEO_MASK_SPECS:
        raise ValueError(f"Unknown video mask type: {model_type!r}.")
    if (
        not isinstance(mask, torch.Tensor)
        or getattr(mask, "is_nested", False)
        or mask.ndim not in (3, 4)
    ):
        raise ValueError("Expected MASK shaped [frames,H,W] or [B,frames,H,W].")
    _validate_mask_values(mask)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    batch, _, latent_frames, height, width = video.shape
    mask_batch, mask_frames, mask_height, mask_width = mask.shape
    if mask_batch not in (1, batch):
        raise ValueError(
            f"MASK batch must be 1 or match video batch {batch}; got {mask_batch}."
        )

    groups = None
    if mask_frames != latent_frames:
        groups = _image_frame_groups(model_type, latent_frames)
        expected_frames = sum(groups)
        if mask_frames != expected_frames:
            raise ValueError(
                f"Received {mask_frames} mask frames for {model_type} latent T={latent_frames}. "
                f"Expected {latent_frames} latent-frame masks or {expected_frames} image-frame masks. "
                "No temporal interpolation, padding, or truncation is performed."
            )

    # Hard coverage is decided before float32 conversion, including tiny positive inputs.
    mask = (mask > 0).float() if hard else mask.float()
    # Spatial and temporal maxima commute; shrink first to avoid large temporal intermediates.
    if (mask_height, mask_width) != (height, width):
        mask = F.adaptive_max_pool2d(
            mask.reshape(-1, 1, mask_height, mask_width), (height, width)
        ).reshape(mask_batch, mask_frames, height, width)
    if groups is not None:
        mask = torch.stack(
            [part.amax(dim=1) for part in mask.split(groups, dim=1)], dim=1
        )
    return mask.unsqueeze(1).expand(batch, 1, latent_frames, height, width).clone()


def composite_video_latents(
    original_latent, replacement_latent, type="minimax", mask=None
):
    original = _video_samples(original_latent)
    replacement = _video_samples(replacement_latent)
    if original.shape != replacement.shape:
        raise ValueError(
            "original_latent and replacement_latent must have identical [B,C,T,H,W] shapes; "
            f"got {tuple(original.shape)} and {tuple(replacement.shape)}. Align them before compositing."
        )
    if type not in VIDEO_MASK_SPECS:
        raise ValueError(f"Unknown video mask type: {type!r}.")

    batch, channels, frames, height, width = original.shape
    coverage = None
    inherited = replacement_latent.get("noise_mask")
    if inherited is not None:
        if (
            not isinstance(inherited, torch.Tensor)
            or getattr(inherited, "is_nested", False)
            or inherited.ndim != 5
        ):
            raise ValueError(
                "replacement_latent noise_mask must be standalone [B,1 or C,T,H,W]. Use Set Video Latent Noise Mask first."
            )
        _validate_mask_values(inherited)
        if (
            inherited.shape[0] not in (1, batch)
            or inherited.shape[1] not in (1, channels)
            or inherited.shape[2:] != original.shape[2:]
        ):
            raise ValueError(
                "Inherited noise_mask must match replacement_latent T/H/W, with batch 1 or B and channels 1 or C."
            )
        coverage = (inherited.amax(dim=1, keepdim=True) > 0).to(original.device)
    if mask is not None:
        additional = _map_video_mask(original, mask, type, hard=True).to(
            device=original.device, dtype=torch.bool
        )
        coverage = additional if coverage is None else coverage | additional
    if coverage is None:
        coverage = torch.ones(
            (batch, 1, frames, height, width),
            device=original.device,
            dtype=torch.bool,
        )

    output = original_latent.copy()
    output["samples"] = torch.where(coverage, replacement.to(original), original)
    output["noise_mask"] = coverage.expand(batch, 1, frames, height, width).to(
        torch.float32
    )
    return output
