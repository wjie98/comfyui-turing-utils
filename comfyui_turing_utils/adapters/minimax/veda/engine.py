"""H3 Veda inference, with forward-local tiling and bounded head/score staging."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ....kernel_api import load_kernel_extension
from ....log import get_logger
from ..memory_state import runtime_memory
from . import h3_layout, tiling
from .predictor import PredictorBundle, Projection, ProjectionTransfer, project_features, score_tiles
from .selection import choose_score_rows, compact_reduces_groups, estimate_workspace_bytes, select_tiles
from .quantization import prequantize, finish_value


LOG = get_logger("minimax.veda")


@dataclass(frozen=True)
class VedaConfig:
    bundle: PredictorBundle
    keep_ratio: float = 0.1
    reference_keep_ratio: float = 1.0
    plan_policy: str = "nearest"
    debug: bool = False


def workspace_per_head(config: VedaConfig, packed_layout, layer: int, element_size: int,
                       capability: tuple[int, int], cache=None, *, chunk_tiles=32) -> int:
    """CPU size planning before QKV projection chooses its head shard."""
    if config.keep_ratio == config.reference_keep_ratio == 1:
        return 0
    spec = layout_spec(packed_layout, cache)
    bundle = config.bundle
    plan = bundle.plans.select(spec.target.grid).plan
    references = spec.references if config.reference_keep_ratio < 1 else ()
    ref_tiles = sum(tiling.least_padding_shape(s.grid).num_tiles(s.grid) for s in references)
    ref_rows = sum(s.grid[0] * s.grid[1] * s.grid[2] for s in references)
    target_rows = spec.target.grid[0] * spec.target.grid[1] * spec.target.grid[2]
    globals_count = (spec.seq_len - ref_rows - target_rows + 127) // 128
    video_tiles = max(plan.shapes[i].num_tiles(spec.target.grid)
                      for i in plan.head_shape[layer]) + ref_tiles
    stage = bundle.staged_bytes_per_head
    if bundle.precision == "bf16" and capability < (8, 0):
        stage *= 3
    return estimate_workspace_bytes(slots=(video_tiles + globals_count) * 128,
                                    video_tiles=video_tiles, heads=1, head_dim=bundle.head_dim,
                                    projection_bytes_per_head=stage, element_size=element_size,
                                    score_rows=128, fused_prepare=True,
                                    head_major_prepare=True, chunk_tiles=chunk_tiles,
                                    use_w8a8=bundle.precision == "w8a8")


def layout_spec(packed_layout, cache):
    if cache is None:
        return h3_layout.describe(packed_layout)
    if cache.get("packed_layout") is not packed_layout:
        cache.clear()
        cache["packed_layout"] = packed_layout
        cache["spec"] = h3_layout.describe(packed_layout)
    return cache["spec"]


def pack_routes(indices: torch.Tensor, keep: torch.Tensor, layout: tiling.TileLayout,
                *, out: torch.Tensor | None = None, row_start: int = 0):
    """Pack one query-row chunk, without retaining full quadratic Top-K lists."""
    native = load_kernel_extension("_sage_qattn_sm75")
    if out is None:
        out = torch.empty((1, indices.shape[0], indices.shape[1], (layout.n_tiles * 2 + 31) // 32),
                          device=indices.device, dtype=torch.int32)
    native.veda_pack_routes(indices.contiguous(), keep.contiguous(),
                            layout.valid_count, layout.n_video_tiles, out, row_start)
    return out


def run_sparse_tiles(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     layout: tiling.TileLayout, routes: torch.Tensor, *, use_w8a8=False) -> torch.Tensor:
    """NHD tiled Q/K/V -> NHD; one softmax over selected video + global K/V."""
    native = load_kernel_extension("_sage_qattn_sm75")
    if not hasattr(native, "veda_sparse_online_attn"):
        raise RuntimeError("Veda requires rebuilding the Turing Utils CUDA kernel")
    attention_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    packed = finish_value(prequantize(
        q.transpose(0, 1).unsqueeze(0).to(attention_dtype),
        k.transpose(0, 1).unsqueeze(0).to(attention_dtype),
        v.transpose(0, 1).unsqueeze(0).to(attention_dtype),
        use_w8a8=use_w8a8,
    ))
    return run_sparse_packed(packed, layout, routes, v.dtype)


def run_sparse_packed(packed, layout, routes, output_dtype):
    native = load_kernel_extension("_sage_qattn_sm75")
    device = packed.query_int8.device
    sparse_queries = torch.zeros(layout.n_tiles * 2, dtype=torch.uint8, device=device)
    sparse_queries[:layout.n_video_tiles * 2] = 1
    output = torch.empty(packed.query_int8.shape, device=device, dtype=packed.value.dtype)
    vi = getattr(packed, "value_int8", None)
    vs = getattr(packed, "value_scale", None)
    empty = torch.empty(0, device=device)
    with torch.cuda.device(device):
        native.veda_sparse_online_attn(
            packed.query_int8, packed.key_int8, packed.value, output,
            packed.query_scale, packed.key_scale, routes, sparse_queries,
            layout.valid_count, packed.sm_scale,
            vi if vi is not None else empty, vs if vs is not None else empty, int(vi is not None),
        )
    return output[0].transpose(0, 1).to(output_dtype)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
           config: VedaConfig, packed_layout, layer: int, head_start: int = 0, cache=None,
           prepare_chunk_tiles: int = -1, score_chunk_rows: int = 0,
           projector=None):
    """HND tensors, already RMS-normalized and RoPE-transformed."""
    bundle = config.bundle
    if score_chunk_rows < 0:
        raise ValueError("score_chunk_rows must be nonnegative (0 selects automatically)")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("H3 Veda currently requires matching batch-one Q/K/V")
    if q.shape[-1] != bundle.head_dim or not 0 <= layer < bundle.num_layers:
        raise ValueError("Veda predictor dimensions do not match the H3 model")
    if head_start < 0 or head_start + q.shape[1] > bundle.num_heads:
        raise ValueError("Veda head shard is outside the predictor head range")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) < (7, 5):
        raise RuntimeError("H3 Veda requires an SM75-or-newer NVIDIA GPU")
    spec = layout_spec(packed_layout, cache)
    if spec.seq_len != q.shape[2]:
        raise ValueError("H3 Veda packed layout and Q/K/V sequence lengths disagree")
    choice = bundle.plans.select(spec.target.grid)
    if config.plan_policy == "strict" and not choice.exact:
        raise ValueError(f"Veda has no exact tile plan: {choice.how}")
    plan = choice.plan
    report_key = ("reported_plan", spec.target.grid)
    if cache is None or report_key not in cache:
        (LOG.debug if choice.exact else LOG.warning)("Veda plan: %s", choice.how)
        if spec.skipped:
            LOG.warning("Veda keeps unmapped references dense: %s", "; ".join(spec.skipped))
        if cache is not None:
            cache[report_key] = True
    # A bounded GPU layout may survive this call within a single forward.
    # Only the CPU bundle survives between sampling runs.
    output = torch.empty((spec.seq_len + 1, q.shape[1], bundle.head_dim),
                         dtype=v.dtype, device=v.device)
    q, k, v = (x[0].transpose(0, 1) for x in (q, k, v))
    shape_heads = {}
    heads_key = ("shape_heads", plan.name, layer, head_start, q.shape[1])
    if cache is not None and heads_key in cache:
        shape_heads = cache[heads_key]
    else:
        for local_head in range(q.shape[1]):
            shape_id = plan.head_shape[layer][head_start + local_head]
            shape_heads.setdefault(shape_id, []).append(local_head)
        if cache is not None:
            cache[heads_key] = shape_heads
    layer_transfer = None
    for shape_id, local_heads in shape_heads.items():
        shape = plan.shapes[shape_id]
        # Keeping references dense means BOTH directions remain dense, not
        # merely allowing every video query to read the reference columns.
        references = spec.references if config.reference_keep_ratio < 1 else ()
        spans = [tiling.TiledSpan(s.start, s.grid, tiling.least_padding_shape(s.grid))
                 for s in references]
        spans.append(tiling.TiledSpan(spec.target.start, spec.target.grid, shape))
        layout_key = (shape, config.reference_keep_ratio < 1)
        host_layout = None if cache is None else cache.get(layout_key)
        if host_layout is None:
            host_layout = tiling.build_tile_layout(spans, spec.seq_len)
            if cache is not None:
                cache[layout_key] = host_layout
        # Byte-bounded forward-local LRU. A single-entry cache thrashes when
        # predictor heads use several tile shapes in each layer. Allocations
        # remain visible to the non-evicting budget below; no weights cached.
        gpu_key = (layout_key, q.device)
        layouts = {} if cache is None else cache.setdefault("gpu_layouts", {})
        cached = layouts.pop(gpu_key, None)
        if cached is not None:
            layout = cached[0]
            layouts[gpu_key] = cached
        else:
            layout_bytes = sum(value.numel() * value.element_size()
                               for value in vars(host_layout).values() if isinstance(value, torch.Tensor))
            while layouts and sum(entry[1] for entry in layouts.values()) + layout_bytes > 4 * 1024**2:
                del layouts[next(iter(layouts))]
            layout = tiling.TileLayout(**{
                field.name: (value.to(q.device) if isinstance(value, torch.Tensor) else value)
                for field in fields(host_layout) for value in (getattr(host_layout, field.name),)
            })
            if cache is not None and layout_bytes <= 4 * 1024**2:
                layouts[gpu_key] = (layout, layout_bytes)
        score_rows = min(score_chunk_rows or 128, layout.n_video_tiles)
        stage_bytes = bundle.staged_bytes_per_head
        if bundle.precision == "bf16" and torch.cuda.get_device_capability(q.device) < (8, 0):
            stage_bytes *= 3  # BF16 storage + FP32 emulation operands.
        native = load_kernel_extension("_sage_qattn_sm75")
        sizes = dict(
            slots=layout.num_slots, video_tiles=layout.n_video_tiles, heads=1,
            head_dim=bundle.head_dim, projection_bytes_per_head=stage_bytes,
            element_size=q.element_size(), score_rows=score_rows,
            fused_prepare=True, head_major_prepare=True, use_w8a8=bundle.precision == "w8a8",
        )
        available, _, _ = runtime_memory(q.device)
        budget = max(0, available - 64 * 1024**2 - getattr(projector, "workspace_bytes", 0))
        full_cost = estimate_workspace_bytes(**sizes)
        compact_cost = estimate_workspace_bytes(**sizes, chunk_tiles=32)
        chunks = prepare_chunk_tiles
        if chunks < 0:
            # Pay extra chunk launches only when they permit a larger head
            # group under the same non-evicting budget.
            fewer_groups = compact_reduces_groups(len(local_heads), budget, full_cost, compact_cost)
            chunks = 32 if fewer_groups else 0
        per_head = estimate_workspace_bytes(**sizes, chunk_tiles=chunks)
        if projector is not None:
            chunks = max(1, chunks or 32)
            # The regular whole-path estimate is retained conservatively:
            # projection input gathering/GEMM scratch belongs to the callback.
            per_head = max(per_head, estimate_workspace_bytes(**sizes, chunk_tiles=chunks))
        group_size = min(len(local_heads), 14, budget // max(per_head, 1))
        if group_size < 1:
            raise RuntimeError(f"Veda needs {per_head / 1024**2:.0f} MiB for one head; "
                               f"non-evicting budget is {budget / 1024**2:.0f} MiB. "
                               "Reduce the H3 QKV head group or resolution.")
        if score_chunk_rows == 0:
            score_rows = choose_score_rows(layout.n_video_tiles, group_size, budget, per_head)
            sizes["score_rows"] = score_rows
            per_head = max(per_head, estimate_workspace_bytes(**sizes, chunk_tiles=chunks))
        if config.debug:
            LOG.info("layer=%d plan=%s exact=%s shape=%s group=%d workspace=%.1f MiB tile_chunk=%d score_rows=%d",
                     layer, plan.name, choice.exact, shape, group_size,
                     group_size * per_head / 1024**2, chunks, score_rows)
        for start in range(0, len(local_heads), group_size):
            group = local_heads[start:start + group_size]
            head_indices = torch.tensor(group, device=q.device, dtype=torch.long)
            global_heads = [head_start + h for h in group]
            upload_stream = None if cache is None else cache.get("upload_stream")
            if upload_stream is None or upload_stream.device != q.device:
                upload_stream = torch.cuda.Stream(device=q.device)
                if cache is not None:
                    cache["upload_stream"] = upload_stream
            # Multiple shapes otherwise upload separate tiny head sets and
            # incur a host/copy-stream transition for every shape. Stage only
            # this call's heads once when <=8 MiB and budget slack covers it.
            # Later budget reads see this live allocation; never cache it.
            call_stage_bytes = bundle.staged_bytes_per_head * q.shape[1]
            if (layer_transfer is None and len(shape_heads) > 1 and start == 0 and
                    shape_id == next(iter(shape_heads)) and
                    call_stage_bytes <= 8 * 1024**2 and
                    call_stage_bytes <= budget - per_head * group_size):
                layer_transfer = ProjectionTransfer(
                    (bundle.proj_q[layer], bundle.proj_k[layer]),
                    list(range(head_start, head_start + q.shape[1])), q.device, upload_stream)
            transfer = layer_transfer or ProjectionTransfer(
                (bundle.proj_q[layer], bundle.proj_k[layer]), global_heads, q.device, upload_stream)
            native = load_kernel_extension("_sage_qattn_sm75")
            packed = None
            if chunks:
                from .prepare import prepare_compact
                try:
                    packed, qfeatures, kfeatures = prepare_compact(
                        q, k, v, layout, head_indices, chunk_tiles=chunks,
                        projector=projector, host_layout=host_layout, head_list=group,
                        use_w8a8=bundle.precision == "w8a8")
                finally:
                    if callable(getattr(projector, "release", None)):
                        projector.release()
            else:
                tq, tk, tv, qfeatures, kfeatures = native.veda_gather_pool(
                    q, k, v, layout.gather_index, head_indices,
                    layout.valid_count, layout.n_video_tiles)
            pq, pk = transfer.consume(q.device)
            if layer_transfer is not None:
                if group == list(range(group[0], group[-1] + 1)):
                    pq, pk = (Projection(p.weight[group[0]:group[-1]+1],
                                         None if p.scale is None else p.scale[group[0]:group[-1]+1])
                              for p in (pq, pk))
                else:
                    pq, pk = (Projection(p.weight.index_select(0, head_indices),
                                         None if p.scale is None else p.scale.index_select(0, head_indices))
                              for p in (pq, pk))
            qhat = project_features(qfeatures, pq, bundle.precision)
            del pq
            khat = project_features(kfeatures, pk, bundle.precision)
            del qfeatures, kfeatures
            del pk, transfer
            # Scoring always uses FP32. Convert once per head group, not the
            # entire K projection again for every query chunk. No change to
            # projection rounding or Top-K selection semantics.
            qhat, khat = qhat.float(), khat.float()
            routes = torch.zeros((1, len(group), layout.n_tiles, (layout.n_tiles * 2 + 31) // 32),
                                 dtype=torch.int32, device=q.device)
            for row in range(0, layout.n_video_tiles, score_rows):
                scores = score_tiles(qhat[:, row:row + score_rows], khat)
                index, keep = select_tiles(scores, layout, config.keep_ratio,
                                           config.reference_keep_ratio, row_start=row)
                pack_routes(index, keep, layout, out=routes, row_start=row)
                del scores, index, keep
            del qhat, khat
            if packed is None:
                attention_dtype = torch.bfloat16 if tq.dtype == torch.float32 else tq.dtype
                packed = finish_value(prequantize(*[
                    x.transpose(0, 1).unsqueeze(0).to(attention_dtype) for x in (tq, tk, tv)],
                    use_w8a8=bundle.precision == "w8a8"))
                del tq, tk, tv
            result = run_sparse_packed(packed, layout, routes, v.dtype)
            native.veda_scatter_tiles(output, result, layout.scatter_index, head_indices, layout.valid_count)
            del packed, result, routes
    return output[:-1].transpose(0, 1).unsqueeze(0)
