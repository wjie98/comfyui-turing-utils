"""Chunkable Veda equal-cost routing; no persistent GPU route cache.

Selection follows Veda-on-ComfyUI / Miowtion (MIT; see LICENSE): separate
reference/target budgets, forced diagonal, fractional quotas by global row.
"""

from __future__ import annotations

import math

import torch

from ....kernel_api import load_kernel_extension

from .tiling import TILE_SIZE, TileLayout


def compact_reduces_groups(
    heads: int, budget: int, full_cost: int, compact_cost: int
) -> bool:
    """More heads per group is useful only if it removes an entire group."""
    desired = min(heads, 14)
    full = min(desired, max(0, budget) // max(full_cost, 1))
    compact = min(desired, max(0, budget) // max(compact_cost, 1))
    return compact > 0 and (
        full == 0 or math.ceil(heads / compact) < math.ceil(heads / full)
    )


def choose_score_rows(
    video_tiles: int, group_size: int, budget: int, base_cost: int
) -> int:
    """Amortize route launches without reducing the already selected head group.

    base_cost includes the conservative 128-row score scratch. Keep a bounded
    256-row maximum; larger blocks can change Top-K dispatch costs substantially.
    """
    base, candidate = min(128, video_tiles), min(256, video_tiles)
    extra = (candidate - base) * video_tiles * 64
    return (
        candidate
        if group_size > 0 and (base_cost + extra) * group_size <= budget
        else base
    )


def choose_projected_chunk(
    *, rows, heads, dim, hidden, element_size, available, workspace
):
    """Prefer full projection unless it exceeds the non-evicting budget.

    workspace is the conservative whole-preparation cost for this head group.
    Return the largest safe projected chunk, or 0 to retain ordinary projection.
    No benchmark captures, model evictions, or persistent GPU reservations.
    """
    budget = max(0, available - 64 * 1024**2)
    full = workspace + 3 * rows * heads * dim * element_size
    if full <= budget:
        return 0
    weight_pack = 3 * heads * dim * (hidden + 8)
    per_row = hidden * (element_size + 3) + heads * dim * 12 + 16
    for tiles in (64, 32, 16, 8, 4, 1):
        if workspace + weight_pack + min(rows, tiles * 128) * per_row <= budget:
            return tiles
    return 0


def select_tiles(
    scores: torch.Tensor,
    layout: TileLayout,
    keep_ratio: float,
    reference_keep_ratio: float,
    *,
    row_start: int = 0,
):
    """Return compact (indices, keep) for [heads, query_chunk, video_tiles]."""
    for ratio in (keep_ratio, reference_keep_ratio):
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("Veda keep ratios must be in (0, 1]")
    heads, queries, columns = scores.shape
    if (
        columns != layout.n_video_tiles
        or row_start < 0
        or row_start + queries > columns
    ):
        raise ValueError("Veda scores do not match the video tile layout")
    blocks = [
        (0, layout.n_ref_tiles, layout.ref_tokens, reference_keep_ratio),
        (layout.n_ref_tiles, columns, layout.target_tokens, keep_ratio),
    ]
    native = load_kernel_extension("_sage_qattn_sm75")
    indices, keeps = [], []
    for start, stop, tokens, ratio in blocks:
        count = stop - start
        if count == 0:
            continue
        if ratio == 1:
            indices.append(
                torch.arange(start, stop, device=scores.device).expand(
                    heads, queries, count
                )
            )
            keeps.append(layout.kv_ok[start:stop].expand(heads, queries, count))
            continue
        budget = ratio * math.ceil(tokens / TILE_SIZE) ** 2 / count
        low = min(count, max(1, math.floor(budget)))
        high = min(count, low + 1)
        fraction = round(min(max(budget - low, 0.0), 1.0), 12)
        values = native.veda_prepare_scores(
            scores.float().contiguous(), layout.valid_count, start, stop, row_start
        )
        best, selected = values.topk(high, dim=-1, sorted=True)
        index, keep = native.veda_finish_selection(
            best, selected, layout.valid_count, start, row_start, low, fraction
        )
        indices.append(index)
        keeps.append(keep)
    if len(indices) == 1:
        return indices[0], keeps[0]
    return torch.cat(indices, -1), torch.cat(keeps, -1)


def estimate_workspace_bytes(
    *,
    slots: int,
    video_tiles: int,
    heads: int,
    head_dim: int,
    projection_bytes_per_head: int,
    element_size: int,
    score_rows: int,
    fused_prepare: bool = False,
    chunk_tiles: int = 0,
    head_major_prepare: bool = False,
    use_w8a8: bool = False,
) -> int:
    """Conservative live workspace in addition to the caller's Q/K/V tensors.

    Includes staged projections, gathered Q/K/V/O, INT8 Q/K, pool temporaries,
    features, FP32 score/top-k scratch, and a packed 128x64 route bitset.
    It deliberately excludes the CPU bundle, never made GPU-resident as a unit.
    """
    # FP32 inputs require three temporary BF16 quantizer operands. For half
    # inputs, HND physical storage aliases V rather than making a fourth copy.
    tile_bytes = (
        5 * element_size
        if element_size == 4
        else (3 if head_major_prepare else 4) * element_size + 2
    )
    gathered = slots * head_dim * tile_bytes
    if fused_prepare and chunk_tiles:
        chunk = min(slots, chunk_tiles * TILE_SIZE)
        # Full compact Q/K/V and output, plus one floating tile chunk and its
        # temporary quantization result. FP32 also needs BF16 conversion temps.
        gathered = slots * head_dim * (2 * element_size + 2)
        gathered += chunk * head_dim * tile_bytes
    pool = 0 if fused_prepare else video_tiles * TILE_SIZE * head_dim * element_size * 3
    if use_w8a8:
        # V conversion overlaps its floating input; channel scales span all tiles.
        gathered += slots * head_dim + head_dim * 4
    features = video_tiles * head_dim * 4 * 10
    scoring = min(score_rows, video_tiles) * video_tiles * 64
    route = math.ceil(slots / TILE_SIZE) * math.ceil(math.ceil(slots / 64) / 32) * 4
    return heads * (
        projection_bytes_per_head + gathered + pool + features + scoring + route
    )
