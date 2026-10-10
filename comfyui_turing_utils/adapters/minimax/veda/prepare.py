"""Bounded tile preparation: retain INT8 Q/K, V and predictor features.

Tile chunks are multiples of 128, preserving Sage's 16/64-token quantization
groups exactly. Predictor pooling happens BEFORE quantization. This is not the
SOL compact path. Rotated packing intentionally omits K-anchor stabilization.
"""

from dataclasses import replace

import torch

from ....kernel_api import load_kernel_extension
from .quantization import prequantize, finish_value


def prepare_compact(
    q,
    k,
    v,
    layout,
    heads,
    *,
    chunk_tiles=64,
    projector=None,
    host_layout=None,
    head_list=None,
    use_w8a8=False,
):
    if chunk_tiles < 1:
        raise ValueError("Veda tile chunk must be positive")
    native = load_kernel_extension("_sage_qattn_sm75")
    group, slots, dim = heads.numel(), layout.num_slots, q.shape[-1]
    dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    qfeatures = torch.empty(
        (group, layout.n_video_tiles, dim * 3), device=q.device, dtype=torch.float32
    )
    kfeatures = torch.empty_like(qfeatures)
    packed = None
    if projector is not None and (host_layout is None or head_list is None):
        raise ValueError(
            "Projected Veda preparation requires host layout and head indices"
        )
    projection_plan = None
    if projector is not None:
        local_heads = torch.arange(group, device=q.device, dtype=torch.long)
        plans = getattr(host_layout, "_projection_chunks", None)
        if plans is None:
            plans = host_layout._projection_chunks = {}
        projection_plan = plans.get(chunk_tiles)
        if projection_plan is None:
            projection_plan = []
            for start in range(0, layout.n_tiles, chunk_tiles):
                perm = host_layout.perm[
                    start * 128 : min(start + chunk_tiles, layout.n_tiles) * 128
                ]
                valid = perm >= 0
                projection_plan.append(
                    (
                        perm[valid].pin_memory(),
                        (valid.long().cumsum(0) - 1).clamp_min(0).pin_memory(),
                    )
                )
            # One chunk geometry per CPU layout; no unbounded tuning cache.
            plans.clear()
            plans[chunk_tiles] = projection_plan
    for tile in range(0, layout.n_tiles, chunk_tiles):
        end = min(tile + chunk_tiles, layout.n_tiles)
        start_row, end_row = tile * 128, end * 128
        nvideo = max(0, min(end, layout.n_video_tiles) - tile)
        if projector is None:
            tq, tk, tv, fq, fk = native.veda_gather_pool(
                q,
                k,
                v,
                layout.gather_index[start_row:end_row],
                heads,
                layout.valid_count[tile:end],
                nvideo,
            )
        else:
            # Project real source rows only, in tile order. RoPE must use these
            # original sequence indices, never the padded/permuted positions.
            source_plan, gather_plan = projection_plan[tile // chunk_tiles]
            source_indices = source_plan.to(q.device, non_blocking=True)
            local_gather = gather_plan.to(q.device, non_blocking=True)
            projected = projector(source_indices, head_list)
            expected = (source_indices.numel(), group, dim)
            if (
                any(
                    tuple(t.shape) != expected
                    or t.device != q.device
                    or t.dtype != q.dtype
                    for t in projected
                )
                or len(projected) != 3
            ):
                raise ValueError(
                    "Veda projector must return matching post-RoPE NHD Q/K/V"
                )
            tq, tk, tv, fq, fk = native.veda_gather_pool(
                *projected,
                local_gather,
                local_heads,
                layout.valid_count[tile:end],
                nvideo,
            )
            del projected, source_indices, local_gather
        if nvideo:
            qfeatures[:, tile : tile + nvideo].copy_(fq)
            kfeatures[:, tile : tile + nvideo].copy_(fk)
        part = prequantize(
            *[x.transpose(0, 1).unsqueeze(0).to(dtype) for x in (tq, tk, tv)],
            use_w8a8=use_w8a8,
        )
        if packed is None:
            packed = replace(
                part,
                query_int8=torch.empty(
                    (1, group, slots, dim), device=q.device, dtype=torch.int8
                ),
                key_int8=torch.empty(
                    (1, group, slots, dim), device=q.device, dtype=torch.int8
                ),
                query_scale=torch.empty(
                    (1, group, slots // 16), device=q.device, dtype=torch.float32
                ),
                key_scale=torch.empty(
                    (1, group, slots // 64), device=q.device, dtype=torch.float32
                ),
                value=torch.empty((1, group, slots, dim), device=q.device, dtype=dtype),
            )
        packed.query_int8[:, :, start_row:end_row].copy_(part.query_int8)
        packed.key_int8[:, :, start_row:end_row].copy_(part.key_int8)
        packed.query_scale[:, :, start_row // 16 : end_row // 16].copy_(
            part.query_scale
        )
        packed.key_scale[:, :, start_row // 64 : end_row // 64].copy_(part.key_scale)
        packed.value[:, :, start_row:end_row].copy_(part.value)
        del tq, tk, tv, fq, fk, part
    return finish_value(packed), qfeatures, kfeatures
