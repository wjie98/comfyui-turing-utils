"""Runtime activation-memory policy for MiniMax H3 inference.

The policy deliberately uses ComfyUI's live free-memory accounting and its
``--reserve-vram`` value.  It therefore adapts to Dynamic VRAM instead of
assuming that model weights are permanently resident.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from ...log import get_logger
from ...hardware import device_capabilities
from .memory_state import (
    ActivationRuntimePlan,
    current_model_is_dynamic as _current_model_is_dynamic,
    dynamic_vram_reclaimable as _dynamic_vram_reclaimable,
    dynamic_vbars as _dynamic_vbars,
    dynamic_weight_prefetch_reserve as _dynamic_weight_prefetch_reserve,
    log_memory_diagnostics as _log_memory_diagnostics,
    memory_diagnostics as _memory_diagnostics,
    planning_available as _planning_available,
    runtime_memory as _runtime_memory,
    should_log as _should_log,
)
from .memory_state import ensure_dynamic_vram_headroom as _ensure_headroom
from .policy_config import activation_mode as _mode
from .policy_config import override_chunk_rows as _override_chunk_rows
from .policy_config import override_ffn_channels as _override_ffn_channels
from .policy_config import override_head_group as _override_head_group


LOG = get_logger("minimax.policy")
_MIB = 1024**2
_ATTENTION_QUERY_TILE_ROWS = 64
_ATTENTION_TARGET_WAVES = 4
_ATTENTION_CTAS_PER_SM = 2
_MAX_BALANCED_SHARDS = 4
_MIN_SATURATED_ROW_TILES = 4
_MIN_SATURATED_GEMM_WIDTH = 1024
@dataclasses.dataclass(frozen=True, slots=True)
class ActivationDecision:
    operation: str
    mode: str
    rows: int
    chunk_rows: int
    available_bytes: int
    full_peak_bytes: int
    streamed_peak_bytes: int
    reserve_bytes: int

    @property
    def streamed(self) -> bool:
        return 0 < self.chunk_rows < self.rows

    @property
    def tier(self) -> int:
        return 1 if self.streamed else 0


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionDecision:
    mode: str
    rows: int
    heads: int
    head_group: int
    saturation_group: int
    cache_quantized_input: bool
    available_bytes: int
    estimated_peak_bytes: int
    reserve_bytes: int

    @property
    def sharded(self) -> bool:
        return 0 < self.head_group < self.heads

    @property
    def tier(self) -> int:
        return 2 if self.sharded else 0


@dataclasses.dataclass(frozen=True, slots=True)
class FFNChannelDecision:
    mode: str
    rows: int
    expanded_size: int
    chunk_rows: int
    chunk_channels: int
    available_bytes: int
    estimated_peak_bytes: int
    reserve_bytes: int

    @property
    def sharded(self) -> bool:
        return 0 < self.chunk_channels < self.expanded_size

    @property
    def tier(self) -> int:
        if self.sharded:
            return 3
        return 1 if 0 < self.chunk_rows < self.rows else 0


def ensure_dynamic_vram_headroom(
    base_model,
    device: torch.device,
    *,
    rows: int,
    operation: str,
    estimated_peak_bytes: int,
    runtime_plan: ActivationRuntimePlan | None = None,
) -> int:
    """Compatibility facade over the shared DynamicVRAM state service."""
    return _ensure_headroom(
        base_model,
        device,
        rows=rows,
        operation=operation,
        estimated_peak_bytes=estimated_peak_bytes,
        runtime_plan=runtime_plan,
        _runtime_memory_fn=_runtime_memory,
        _dynamic_vbars_fn=_dynamic_vbars,
        _diagnostics_fn=_log_memory_diagnostics,
    )


def _align_rows(value: int, alignment: int, minimum: int) -> int:
    value = value // alignment * alignment
    return max(value, minimum) if value >= minimum else 0


def _prefer_saturated_row_streaming(
    *,
    operation: str,
    rows: int,
    cap: int,
    base_model,
    device: torch.device,
) -> bool:
    """Keep a saturated QKV tile when monolithic output cannot add MFU.

    Four 16K tiles are enough to amortize the row-streamed dispatches, while
    each tile already contains substantially more W8 GEMM work than a current
    GPU can execute concurrently.  For a dynamically paged DiT, making the
    projection output larger beyond that point only consumes residency that
    can hold the current/prefetched weights.  MLP is deliberately excluded:
    its full fused fc1/fc2 path still benefits from avoiding repeated calls.

    This is a workload- and residency-based rule, not a GPU-architecture
    branch.  Explicit throughput mode remains the way to force a monolithic
    projection for profiling.
    """
    return bool(
        operation == "qkv"
        and cap > 0
        and math.ceil(int(rows) / int(cap)) >= _MIN_SATURATED_ROW_TILES
        and _current_model_is_dynamic(base_model, device)
    )


def balanced_saturation_size(
    total: int,
    *,
    alignment: int,
    minimum: int,
    max_shards: int = _MAX_BALANCED_SHARDS,
) -> int:
    """Return the smallest aligned, balanced shard near the MFU plateau."""
    total = int(total)
    alignment = max(int(alignment), 1)
    target = max(
        int(minimum),
        math.ceil(total / max(int(max_shards), 1)),
    )
    legal = [
        size
        for size in range(alignment, total + 1, alignment)
        if total % size == 0 and size >= target
    ]
    if legal:
        return min(legal)
    return min(
        total,
        math.ceil(target / alignment) * alignment,
    )


def _attention_saturation_group(
    device: torch.device,
    *,
    rows: int,
    heads: int,
    head_dim: int,
    legal_groups: list[int],
) -> int:
    """Find the smallest legal group that already supplies ample GPU work.

    Attention exposes one CTA per 64-query tile and head. Four resident-grid
    waves are sufficient to make launch width cease being the dominant MFU
    limiter. QKV projection also keeps at least a 1024-channel output tile,
    while at most four balanced groups bound repeated launch/anchor overhead.
    """
    capabilities = device_capabilities(device)
    sm_count = max(int(capabilities.multiprocessor_count), 1)
    query_blocks = max(
        math.ceil(int(rows) / _ATTENTION_QUERY_TILE_ROWS),
        1,
    )
    grid_heads = math.ceil(
        sm_count * _ATTENTION_CTAS_PER_SM * _ATTENTION_TARGET_WAVES
        / query_blocks
    )
    gemm_heads = math.ceil(_MIN_SATURATED_GEMM_WIDTH / int(head_dim))
    pass_heads = math.ceil(int(heads) / _MAX_BALANCED_SHARDS)
    minimum = max(grid_heads, gemm_heads, pass_heads, 1)
    balanced = [
        group
        for group in legal_groups
        if group >= minimum and heads % group == 0
    ]
    if balanced:
        return min(balanced)
    candidates = [group for group in legal_groups if group >= minimum]
    return min(candidates) if candidates else max(legal_groups)


def estimate_attention_lifecycle_peak(
    *,
    rows: int,
    heads: int,
    head_dim: int,
    hidden_size: int,
    element_size: int,
    head_group: int,
    compact_qk: bool,
    cache_quantized_input: bool,
    quantized_value: bool,
    logical_key_rows: int | None = None,
    residual_subblocks: int = 0,
) -> int:
    """Estimate the peak across projection, V8 preparation and attention."""
    rows = int(rows)
    heads = int(heads)
    head_dim = int(head_dim)
    hidden_size = int(hidden_size)
    element_size = int(element_size)
    group = int(head_group)
    key_rows = rows if logical_key_rows is None else int(logical_key_rows)
    residual_subblocks = int(residual_subblocks)
    if residual_subblocks not in (0, 1, 2):
        raise ValueError("residual_subblocks must be 0, 1, or 2")
    if key_rows < rows:
        raise ValueError("logical K rows cannot be shorter than physical Q rows")
    features = rows * group * head_dim
    key_features = key_rows * group * head_dim
    output = rows * heads * head_dim * element_size
    destination = 0 if group == heads else output
    input_cache = rows * (hidden_size + 4) if cache_quantized_input else 0
    if not compact_qk:
        if key_rows != rows and (quantized_value or residual_subblocks):
            qkv_projection = 3 * features * element_size
            qk_scales = (
                ((rows + 63) // 64) * group * 4 * 4
                + ((key_rows + 63) // 64) * group * 4
            )
            qk_compact = features + key_features + qk_scales
            value_int8 = features + key_features if quantized_value else 0
            summaries = 0
            if residual_subblocks:
                key_blocks = (key_rows + 63) // 64
                padded_key_blocks = ((key_blocks + 15) // 16) * 16
                residual_tokens = 64 // residual_subblocks
                residual_summaries = (key_rows + residual_tokens - 1) // residual_tokens
                padded_residual_summaries = (
                    (residual_summaries + 15) // 16
                ) * 16
                summaries = (
                    group * padded_key_blocks * head_dim * 2
                    + 2 * group * padded_residual_summaries * head_dim * 2
                    + 2 * group * head_dim * 4
                )
            preparation_peak = qkv_projection + qk_compact + value_int8 + summaries
            retained_value = (
                key_features if quantized_value else features * element_size
            )
            execution_peak = (
                qk_compact
                + retained_value
                + summaries
                + features * element_size
            )
            return destination + max(preparation_peak, execution_peak) + input_cache
        return destination + features * 8 + input_cache

    query_blocks = (rows + 63) // 64
    key_blocks = (key_rows + 63) // 64
    qk_scales = (query_blocks * 4 + key_blocks) * group * 4
    compact = features + key_features + features * element_size + qk_scales
    value_int8 = key_features if quantized_value else 0
    padded_blocks = ((key_blocks + 15) // 16) * 16
    if residual_subblocks:
        residual_tokens = 64 // residual_subblocks
        residual_summary_count = (key_rows + residual_tokens - 1) // residual_tokens
        padded_residual_summaries = (
            (residual_summary_count + 15) // 16
        ) * 16
        summaries = (
            group * padded_blocks * head_dim * 2
            + 2 * group * padded_residual_summaries * head_dim * 2
            + 2 * group * head_dim * 4
        )
    else:
        summaries = (
            3 * group * padded_blocks * head_dim * 2
            + 2 * group * head_dim * 4
        ) if quantized_value else 0
    result = features * element_size
    execution_peak = compact + value_int8 + summaries + result
    tile_rows = min(rows, 16_384)
    projected_tile = (
        tile_rows * group * head_dim * element_size * 2
        + tile_rows * (hidden_size + 4)
    )
    return (
        destination
        + max(compact + projected_tile, execution_peak)
        + input_cache
    )


def decide_activation_chunks(
    x: torch.Tensor,
    *,
    operation: str,
    hidden_size: int,
    expanded_size: int,
    heads: int | None = None,
    runtime_plan: ActivationRuntimePlan | None = None,
    base_model=None,
) -> ActivationDecision:
    """Select the full or streamed H3 activation path.

    ``expanded_size`` is the SwiGLU post-split width for ``mlp`` and the QKV
    inner width (heads * head_dim) for ``qkv``.
    """
    rows = int(x.shape[0])
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0:
        return ActivationDecision(operation, mode, rows, 0, 0, 0, 0, 0)

    available, reserve, usable = _runtime_memory(x.device, base_model)
    planned_available = _planning_available(
        runtime_plan,
        x.device,
        rows,
        available,
        mode,
        operation,
    )
    element = int(x.element_size())
    hidden_size = int(hidden_size)
    expanded_size = int(expanded_size)

    # Leave room for one streamed block's weights, allocator fragmentation,
    # CUDA graph/kernel scratch, and the desktop compositor.  The compositor's
    # long-lived allocation is separately represented by --reserve-vram.
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = _dynamic_weight_prefetch_reserve(base_model, x.device)
    working = max(planned_available - safety - weight_scratch, 0)

    if operation == "mlp":
        # Persistent returned hidden output plus fc1 BF16, fused fc2 A8 input,
        # and the current hidden output tile. Long H3 contractions use the
        # fixed-workspace fused fc2 rather than a full INT32 accumulator.
        persistent = rows * hidden_size * element
        per_row = 2 * expanded_size * element + expanded_size + hidden_size * element
        # At H3's 5,376x14,336 contractions a 16K-row tile already exposes
        # far more Tensor Core work than the GPU can run concurrently. Larger
        # tiles increase the transient by about 1.26 GiB per additional 16K
        # rows without improving steady-state MFU.
        cap, alignment, minimum = 16384, 256, 2048
    elif operation == "qkv":
        # Streamed projection retains Q/K INT8 and V BF16.  One tile holds the
        # packed BF16 projection plus its quantized input row.  Q/K scales are
        # allocated for ceil(rows/64) blocks and remain live with Q/K/V.
        # ``expanded_size`` is heads*head_dim. There are five FP32 scale lanes
        # per head block; infer heads from H3's 128-wide heads when possible.
        inferred_heads = (
            max(int(heads), 1)
            if heads is not None
            else max(expanded_size // 128, 1)
        )
        scale_bytes = ((rows + 63) // 64) * inferred_heads * 5 * 4
        persistent = (
            rows * (2 * expanded_size + expanded_size * element)
            + scale_bytes
        )
        per_row = (
            3 * expanded_size * element
            + hidden_size
            + 4
        )
        cap, alignment, minimum = 16384, 64, 1024
    else:
        raise ValueError(f"unknown H3 activation operation: {operation}")

    full_peak = persistent + rows * per_row
    override = _override_chunk_rows(operation)
    saturation_limited = bool(
        mode == "auto"
        and override is None
        and _prefer_saturated_row_streaming(
            operation=operation,
            rows=rows,
            cap=cap,
            base_model=base_model,
            device=x.device,
        )
    )
    if mode == "throughput" and override is None:
        chunk_rows = 0
        selection = "forced_throughput"
    elif override is not None:
        chunk_rows = min(rows, _align_rows(override, alignment, minimum))
        selection = "override"
    elif (
        mode == "auto"
        and not saturation_limited
        and full_peak <= int(working * 0.86)
    ):
        chunk_rows = 0
        selection = "full_fit"
    else:
        tile_budget = max(working - persistent, 0)
        chunk_rows = min(
            rows,
            cap,
            _align_rows(tile_budget // max(per_row, 1), alignment, minimum),
        )
        if chunk_rows <= 0:
            # A small tile is still materially safer than falling back to the
            # full activation.  An eventual CUDA OOM will then report the real
            # irreducible floor instead of requesting a multi-GiB temporary.
            chunk_rows = min(rows, minimum)
        selection = "saturated_residency" if saturation_limited else "memory"

    streamed_peak = persistent + (chunk_rows or rows) * per_row
    decision = ActivationDecision(
        operation,
        mode,
        rows,
        chunk_rows,
        available,
        full_peak,
        streamed_peak,
        reserve,
    )
    key = (
        x.device.index,
        operation,
        mode,
        rows,
        chunk_rows,
        selection,
        reserve // (256 * _MIB),
    )
    if _should_log(runtime_plan, key):
        LOG.info(
            "MiniMax H3 activation policy: op=%s mode=%s tier=%d rows=%d path=%s "
            "chunk_rows=%d selection=%s available=%.2f GiB reserve=%.2f GiB "
            "planned_available=%.2f GiB weight_prefetch=%.2f GiB "
            "estimated_peak(full/selected)=%.2f/%.2f GiB %s",
            operation,
            mode,
            decision.tier,
            rows,
            "streamed" if decision.streamed else "throughput",
            chunk_rows,
            selection,
            available / 1024**3,
            reserve / 1024**3,
            planned_available / 1024**3,
            weight_scratch / 1024**3,
            full_peak / 1024**3,
            streamed_peak / 1024**3,
            _log_memory_diagnostics(x.device, base_model),
        )
    return decision


def decide_attention_heads(
    x: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    compact_qk: bool,
    quantized_input: bool,
    quantized_value: bool = False,
    runtime_plan: ActivationRuntimePlan | None = None,
    base_model=None,
    logical_key_rows: int | None = None,
    residual_subblocks: int = 0,
) -> AttentionDecision:
    """Choose a whole-head group without changing global sequence attention."""
    rows = int(x.shape[0])
    heads = int(heads)
    head_dim = int(head_dim)
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0 or heads <= 0:
        return AttentionDecision(
            mode,
            rows,
            heads,
            heads,
            heads,
            False,
            0,
            0,
            0,
        )

    available, reserve, usable = _runtime_memory(x.device, base_model)
    planned_available = _planning_available(
        runtime_plan,
        x.device,
        rows,
        available,
        mode,
        "attention",
    )
    element = int(x.element_size())
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = _dynamic_weight_prefetch_reserve(base_model, x.device)
    working = max(planned_available - safety - weight_scratch, 0)

    def peak(group: int, cache_input: bool) -> int:
        return estimate_attention_lifecycle_peak(
            rows=rows,
            heads=heads,
            head_dim=head_dim,
            hidden_size=int(x.shape[-1]),
            element_size=element,
            head_group=group,
            compact_qk=compact_qk,
            cache_quantized_input=cache_input and group != heads,
            quantized_value=quantized_value,
            logical_key_rows=logical_key_rows,
            residual_subblocks=residual_subblocks,
        )

    # A cut is legal whenever both sides cover complete ConvRot-256 blocks.
    # This is the TP-style gcd boundary: D128 permits every two heads, for
    # example, rather than only divisors such as 56/28/14/8/4/2.
    block_heads = 256 // math.gcd(256, head_dim)
    groups = [
        group
        for group in range(heads, 0, -1)
        if group % block_heads == 0
        and (heads - group) % block_heads == 0
    ]
    if not groups:
        groups = [heads]
    override = _override_head_group()
    if override is not None:
        groups = [group for group in groups if group <= override] or [groups[-1]]

    saturation_group = (
        groups[0]
        if override is not None
        else _attention_saturation_group(
            x.device,
            rows=rows,
            heads=heads,
            head_dim=head_dim,
            legal_groups=groups,
        )
    )

    # Head sharding changes only the number of exact passes, whereas evicting
    # hot model pages forces immediate PCIe reloads. Keep a small allocator
    # margin and accept another head pass instead of manufacturing headroom by
    # weight eviction.
    fit_limit = int(working * 0.96)
    full_fits = peak(heads, False) <= fit_limit
    allow_input_cache = bool(
        quantized_input and not _current_model_is_dynamic(base_model, x.device)
    )
    if (mode == "throughput" or full_fits) and override is None:
        head_group = heads
        cache_input = False
    else:
        head_group = groups[-1]
        cache_input = False
        fitting = []
        for group in groups:
            cached_fits = (
                allow_input_cache and peak(group, True) <= fit_limit
            )
            plain_fits = peak(group, False) <= fit_limit
            if cached_fits or plain_fits:
                fitting.append((group, cached_fits))
        saturated = [
            item for item in fitting if item[0] == saturation_group
        ]
        selected = (
            saturated[0]
            if saturated
            else (fitting[0] if fitting else None)
        )
        if selected is not None:
            head_group = selected[0]
            cache_input = bool(selected[1])
    estimated = peak(head_group, cache_input)
    decision = AttentionDecision(
        mode,
        rows,
        heads,
        head_group,
        saturation_group,
        cache_input,
        available,
        estimated,
        reserve,
    )
    key = (
        x.device.index,
        "attention_heads",
        mode,
        rows,
        heads,
        head_group,
        saturation_group,
        cache_input,
        reserve // (256 * _MIB),
    )
    if _should_log(runtime_plan, key):
        LOG.info(
            "MiniMax H3 attention policy: mode=%s tier=%d rows=%d heads=%d "
            "head_group=%d saturation_group=%d input_cache=%s available=%.2f GiB "
            "reserve=%.2f GiB planned_available=%.2f GiB weight_prefetch=%.2f GiB "
            "estimated_peak=%.2f GiB %s",
            mode,
            decision.tier,
            rows,
            heads,
            head_group,
            saturation_group,
            cache_input,
            available / 1024**3,
            reserve / 1024**3,
            planned_available / 1024**3,
            weight_scratch / 1024**3,
            estimated / 1024**3,
            _log_memory_diagnostics(x.device, base_model),
        )
    return decision


def decide_ffn_channels(
    x: torch.Tensor,
    *,
    expanded_size: int,
    chunk_rows: int,
    half_width: bool = False,
    runtime_plan: ActivationRuntimePlan | None = None,
    base_model=None,
) -> FFNChannelDecision:
    """Select an exact ConvRot-aligned FFN intermediate shard."""
    rows = int(x.shape[0])
    expanded_size = int(expanded_size)
    mode = _mode()
    if x.device.type != "cuda" or rows <= 0 or expanded_size <= 0:
        return FFNChannelDecision(
            mode, rows, expanded_size, chunk_rows, 0, 0, 0, 0
        )
    available, reserve, usable = _runtime_memory(x.device, base_model)
    planned_available = _planning_available(
        runtime_plan,
        x.device,
        rows,
        available,
        mode,
        "mlp",
    )
    safety = max(768 * _MIB, int(usable * 0.075))
    weight_scratch = _dynamic_weight_prefetch_reserve(base_model, x.device)
    working = max(planned_available - safety - weight_scratch, 0)
    tile_rows = min(rows, chunk_rows or rows)
    hidden = int(x.shape[-1])
    persistent = rows * hidden * int(x.element_size())
    # Full fc1 BF16, fused INT8 activation, and the BF16 output tile.
    unsharded = persistent + tile_rows * (expanded_size * 5 + hidden * 2)
    override = _override_ffn_channels()
    if override is None and (mode == "throughput" or unsharded <= working):
        chunk_channels = 0
        estimated = unsharded
    else:
        # An explicit channel override without row streaming must not create a
        # sequence-sized INT32 accumulator.  This affects only the opt-in or
        # tight-memory channel path; the normal full path stays unchanged.
        if chunk_rows <= 0:
            tile_rows = min(rows, 4096)
            chunk_rows = tile_rows
        # Retain the complete compressed A8 row instead of a larger INT32
        # output accumulator. ABI 0.33 additionally retains one F-wide BF16
        # gate/rotated buffer and consumes only a C-wide BF16 up shard. Older
        # ABIs retain a 2*C gate/up shard and a small A8 scratch during their
        # exact two-pass reconstruction.
        if half_width:
            fixed_per_row = (
                expanded_size * 3
                + expanded_size // 64
                + hidden * 3
                + 8
            )
            channel_bytes = 2
        else:
            fixed_per_row = expanded_size + hidden * 3
            channel_bytes = 5
        budget = max(
            working - persistent - tile_rows * fixed_per_row, 0
        )
        automatic = max(
            (budget // max(tile_rows * channel_bytes, 1)) // 256 * 256,
            256,
        )
        saturation_channels = balanced_saturation_size(
            expanded_size,
            alignment=256,
            minimum=_MIN_SATURATED_GEMM_WIDTH,
        )
        requested = (
            min(automatic, saturation_channels)
            if override is None
            else override
        )
        chunk_channels = min(
            expanded_size,
            max((requested // 256) * 256, 256),
        )
        estimated = (
            persistent
            + tile_rows * (
                fixed_per_row + chunk_channels * channel_bytes
            )
        )
    decision = FFNChannelDecision(
        mode,
        rows,
        expanded_size,
        chunk_rows,
        chunk_channels,
        available,
        estimated,
        reserve,
    )
    key = (
        x.device.index,
        "ffn_channels",
        mode,
        rows,
        chunk_rows,
        chunk_channels,
        reserve // (256 * _MIB),
    )
    if _should_log(runtime_plan, key):
        LOG.info(
            "MiniMax H3 FFN policy: mode=%s tier=%d rows=%d row_chunk=%d "
            "channel_chunk=%d available=%.2f GiB reserve=%.2f GiB "
            "planned_available=%.2f GiB weight_prefetch=%.2f GiB "
            "half_width=%s estimated_peak=%.2f GiB %s",
            mode,
            decision.tier,
            rows,
            chunk_rows,
            chunk_channels,
            available / 1024**3,
            reserve / 1024**3,
            planned_available / 1024**3,
            weight_scratch / 1024**3,
            half_width,
            estimated / 1024**3,
            _log_memory_diagnostics(x.device, base_model),
        )
    return decision


__all__ = [
    "ActivationRuntimePlan",
    "ActivationDecision",
    "AttentionDecision",
    "FFNChannelDecision",
    "balanced_saturation_size",
    "decide_activation_chunks",
    "decide_attention_heads",
    "decide_ffn_channels",
    "ensure_dynamic_vram_headroom",
    "estimate_attention_lifecycle_peak",
]
