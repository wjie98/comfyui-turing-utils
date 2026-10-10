"""Protected-tail chroma noise for video continuation context."""

from __future__ import annotations

import random

import torch


_CHROMA_BLOCK_SIZE = 16
_CHROMA_POC_GRID = (36, 64)
# Empirical MIT-licensed recipe documented by MacroSony and packaged for
# ComfyUI by beijinren/ComfyUI-H3-Context-Noise.
_CHROMA_PALETTE = (
    (185, 115, 215),
    (115, 195, 140),
    (150, 148, 162),
    (205, 150, 192),
    (138, 182, 148),
    (160, 120, 175),
)


def _noise_alpha_schedule(
    frame_count: int, strength: float, end_strength: float, transition_frames: int
) -> list[float]:
    frame_count = int(frame_count)
    strength = float(strength)
    end_strength = float(end_strength)
    transition_frames = int(transition_frames)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 0.0 <= end_strength <= strength <= 1.0:
        raise ValueError("end_strength must be between 0 and strength")
    if transition_frames < 0:
        raise ValueError("transition_frames must not be negative")
    transition_frames = min(transition_frames, frame_count)
    if transition_frames == 0:
        return [strength] * frame_count
    result = []
    for position in range(frame_count):
        from_end = frame_count - 1 - position
        if from_end >= transition_frames:
            result.append(strength)
        else:
            result.append(
                strength
                + (end_strength - strength)
                * (transition_frames - from_end)
                / transition_frames
            )
    return result


def _nearest_indices(output_size: int, input_size: int, device) -> torch.Tensor:
    scale = input_size / output_size
    values = [
        min(input_size - 1, int((position + 0.5) * scale))
        for position in range(output_size)
    ]
    return torch.tensor(values, dtype=torch.long, device=device)


def _coarse_noise_frame(
    pattern: str, grid_width: int, grid_height: int, palette_rng, torch_generator
) -> torch.Tensor:
    if pattern == "poc_chroma_blocks":
        rows = [
            [palette_rng.choice(_CHROMA_PALETTE) for _ in range(grid_width)]
            for _ in range(grid_height)
        ]
        return torch.tensor(rows, dtype=torch.float32).div_(255.0)
    if pattern == "gaussian_rgb":
        return (
            torch.randn(
                (grid_height, grid_width, 3),
                generator=torch_generator,
                dtype=torch.float32,
            )
            .mul_(0.25)
            .add_(0.5)
            .clamp_(0.0, 1.0)
        )
    if pattern == "uniform_rgb":
        return torch.rand(
            (grid_height, grid_width, 3), generator=torch_generator, dtype=torch.float32
        )
    raise ValueError(f"Unknown pattern: {pattern!r}")


def _add_prefix_chroma_blocks(
    images: torch.Tensor,
    strength: float,
    seed: int,
    *,
    end_strength: float = 0.10,
    transition_frames: int = 4,
    tail_protection_frames: int = 5,
    pattern: str = "poc_chroma_blocks",
    grid_mode: str = "poc_36x64",
    block_size: int = _CHROMA_BLOCK_SIZE,
) -> torch.Tensor:
    frame_count, height, width, _ = images.shape
    tail_protection_frames = int(tail_protection_frames)
    if tail_protection_frames < 0:
        raise ValueError("tail_protection_frames must not be negative")
    affected_frames = frame_count - min(tail_protection_frames, frame_count)
    if affected_frames == 0:
        return images.clone()
    alphas = _noise_alpha_schedule(
        affected_frames,
        strength,
        end_strength,
        transition_frames,
    )
    if grid_mode == "poc_36x64":
        grid_width, grid_height = _CHROMA_POC_GRID
    elif grid_mode == "block_size":
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        grid_width = max(1, round(width / block_size))
        grid_height = max(1, round(height / block_size))
    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode!r}")

    palette_rng = random.Random(int(seed))
    torch_generator = torch.Generator(device="cpu").manual_seed(
        int(seed) & 0xFFFFFFFFFFFFFFFF
    )
    row_indices = _nearest_indices(height, grid_height, images.device)
    column_indices = _nearest_indices(width, grid_width, images.device)
    output = images.clone()
    for position, alpha in enumerate(alphas):
        noise = _coarse_noise_frame(
            pattern, grid_width, grid_height, palette_rng, torch_generator
        ).to(device=images.device, dtype=images.dtype)
        noise = noise.index_select(0, row_indices).index_select(1, column_indices)
        output[position] = output[position] * (1.0 - alpha) + noise * alpha
    return output
