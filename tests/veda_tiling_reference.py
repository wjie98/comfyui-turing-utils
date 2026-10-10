"""Tensor gather/scatter oracle for the CUDA preparation kernels."""

import torch

from comfyui_turing_utils.adapters.minimax.veda.tiling import TileLayout


def gather_tiles(
    x: torch.Tensor, layout: TileLayout, heads: torch.Tensor
) -> torch.Tensor:
    """Permutes rows of x into tile order, zeroing padding slots.

    Args:
        x: [S, H, D] packed tensor (any strides).
        layout: Tile layout of this head group.
        heads: [H'] int64 head indices on x's device.

    Returns:
        [N, H', D] contiguous, tile order (the kernel's seq-major layout).
    """
    out = x[layout.gather_index[:, None], heads[None, :]]
    if layout.pad_slots.numel():
        out.index_fill_(0, layout.pad_slots, 0)
    return out


def scatter_tiles_(
    out: torch.Tensor, tiled: torch.Tensor, layout: TileLayout, heads: torch.Tensor
) -> None:
    """Writes tile-ordered rows back into out[S + 1, H, D] in place.

    Padding slots land in the spare row out[S], so the inverse permutation
    is a single index write without masking.
    """
    out[layout.scatter_index[:, None], heads[None, :]] = tiled
