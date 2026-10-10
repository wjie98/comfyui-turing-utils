"""Video frame grids and image-to-latent temporal groups."""

from __future__ import annotations

import torch


# Image frames per latent after the first frame; H3 uses a repeating block instead.
VIDEO_MASK_SPECS = {
    "wan": 4,
    "minimax": None,
    "ltxv": 8,
    "hunyuan_video": 4,
    "hunyuan_video_15": 4,
    "mochi": 6,
}


def _image_frame_groups(model_type: str, latent_frames: int) -> tuple[int, ...]:
    temporal_stride = VIDEO_MASK_SPECS[model_type]
    if temporal_stride is not None:
        return (1,) + (temporal_stride,) * (latent_frames - 1)
    if latent_frames == 1:
        return (1,)
    if latent_frames < 2 or (latent_frames - 2) % 5:
        raise ValueError(
            f"MiniMax H3 image-frame mapping requires 1 or 5*n+2 latent time positions; "
            f"got {latent_frames}. Supply exactly {latent_frames} masks for direct mapping."
        )
    return (1, 4, 4, 4, 4) * ((latent_frames - 2) // 5) + (1, 4)


def repeat_last_frame(tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Append copies of the final frame along the first dimension."""
    count = int(count)
    if count <= 0:
        return tensor
    if tensor.shape[0] < 1:
        raise ValueError("Cannot repeat the final item of an empty tensor")
    tail = tensor[-1:].repeat(count, *([1] * (tensor.ndim - 1)))
    return torch.cat((tensor, tail), dim=0)


def validate_frame_mask(mask, frame_count: int, height: int, width: int):
    if mask is None:
        return None
    if mask.ndim == 2 and frame_count == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim < 3:
        raise ValueError(
            f"Expected MASK tensor shaped [frames, height, width], got {tuple(mask.shape)}."
        )
    if int(mask.shape[0]) != frame_count:
        raise ValueError(
            f"MASK frame count must match IMAGE frame count before padding; "
            f"got mask={int(mask.shape[0])}, image={frame_count}."
        )
    if int(mask.shape[-2]) != height or int(mask.shape[-1]) != width:
        raise ValueError(
            f"MASK spatial size must match IMAGE before padding; got mask="
            f"{int(mask.shape[-1])}x{int(mask.shape[-2])}, image={width}x{height}."
        )
    return mask


def padded_frame_count(count, model_type, target=0):
    stride = VIDEO_MASK_SPECS[model_type]
    minimum, step = (
        (5, 17) if stride is None else (7 if model_type == "mochi" else 1, stride)
    )
    if target:
        if target < count or target < minimum or (target - minimum) % step:
            raise ValueError(
                f"target_frame_count must be >= {max(count, minimum)} and on the {minimum}+{step}*n {model_type} grid"
            )
        return target
    return minimum + (max(0, count - minimum) + step - 1) // step * step
