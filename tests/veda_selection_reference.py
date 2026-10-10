"""Independent tensor routing oracle; never dispatched by production code."""

import math

import torch

from comfyui_turing_utils.adapters.minimax.veda.tiling import TILE_SIZE, TileLayout


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
    rows = torch.arange(row_start, row_start + queries, device=scores.device)
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
        values = scores[:, :, start:stop].float().clone()
        values.masked_fill_(~layout.kv_ok[None, None, start:stop], -torch.inf)
        own = (rows >= start) & (rows < stop)
        col = (
            (rows - start).clamp(0, count - 1)[None, :, None].expand(heads, queries, 1)
        )
        values.scatter_(
            -1, col, torch.where(own[None, :, None], torch.inf, values.gather(-1, col))
        )
        best, selected = values.topk(high, dim=-1, sorted=True)
        # FP64 preserves the upstream Bresenham pattern even for tiny fractions.
        ramp = (
            torch.arange(
                row_start,
                row_start + queries + 1,
                device=scores.device,
                dtype=torch.float64,
            )
            * fraction
        )
        extra = ramp[1:].floor() > ramp[:-1].floor()
        keep = (
            torch.arange(high, device=scores.device)[None, None, :]
            < (low + extra.long())[None, :, None]
        )
        indices.append(selected + start)
        keeps.append(keep & (best > -torch.inf) & layout.kv_ok[selected + start])
    if len(indices) == 1:
        return indices[0], keeps[0]
    return torch.cat(indices, -1), torch.cat(keeps, -1)
