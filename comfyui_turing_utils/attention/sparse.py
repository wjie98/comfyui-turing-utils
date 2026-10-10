"""Production Sol/SLA sparse-attention policy and exact modality protection."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from .layout import (
    ATTENTION_LAYOUT_KEY,
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    attention_semantic_layout,
    has_complete_attention_layout,
)
from .stable import (
    LOG,
    SPARSE_AUTO_MIN_SEQUENCE,
    SPARSE_PREFIX_POLICY,
    SPARSE_REFERENCE_AUDIO,
    SPARSE_REFERENCE_IMAGE,
    SPARSE_REFERENCE_VIDEO,
    SPARSE_ROUTING_THRESHOLD,
    SPARSE_SKIPPED_RESIDUAL,
    SPARSE_USE_W8A8,
    AttentionCall,
    PrequantizedAttentionCall,
    _LOGGED_SPARSE_DENSE_REASONS,
    _LOGGED_SPARSE_KERNELS,
    _sol_sparse_sageattn,
    finish_turing_attention_output,
    inspect_turing_attention_call,
    normalize_turing_attention_tensors,
)
from ..kernel_api import load_turing_sage


@dataclasses.dataclass(frozen=True, slots=True)
class SolAttentionCall:
    attention: AttentionCall
    effective_min_sequence: int
    dense_query_ranges: tuple[tuple[int, int], ...]
    exact_kv_ranges: tuple[tuple[int, int], ...]
    residual_subblocks: int


@dataclasses.dataclass(frozen=True, slots=True)
class SlaAttentionCall:
    attention: AttentionCall
    effective_min_sequence: int
    dense_query_ranges: tuple[tuple[int, int], ...]
    exact_kv_ranges: tuple[tuple[int, int], ...]


def _sparse_dense_baseline(
    reason: str,
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    **kwargs,
) -> torch.Tensor:
    if reason not in _LOGGED_SPARSE_DENSE_REASONS:
        LOG.debug("Sparse attention uses the selected dense backend for %s", reason)
        _LOGGED_SPARSE_DENSE_REASONS.add(reason)
    return fallback(q, k, v, heads, **kwargs)


def _sparse_prefix_tokens(
    policy: str,
    manual_tokens: int,
    transformer_options,
    sequence_limit: int,
) -> int:
    if policy == "none":
        return 0
    if policy == "manual":
        return min(manual_tokens, sequence_limit)
    layout = attention_semantic_layout(transformer_options)
    if layout is None:
        raw = (
            transformer_options.get(ATTENTION_LAYOUT_KEY)
            if isinstance(transformer_options, dict)
            else None
        )
        prefix = raw.get("dense_prefix_tokens", 0) if isinstance(raw, dict) else 0
        return (
            min(max(prefix, 0), sequence_limit)
            if isinstance(prefix, int) and not isinstance(prefix, bool)
            else 0
        )
    first_sparse = next(
        (
            segment.start
            for segment in layout.query_segments
            if segment.sparse_query_allowed
        ),
        sequence_limit,
    )
    return min(max(first_sparse, 0), sequence_limit)


def _coalesce_token_ranges(ranges, sequence_limit: int) -> tuple[tuple[int, int], ...]:
    normalized = sorted(
        (max(0, int(start)), min(sequence_limit, int(stop)))
        for start, stop in ranges
        if int(stop) > 0 and int(start) < sequence_limit and int(stop) > int(start)
    )
    merged: list[list[int]] = []
    for start, stop in normalized:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((start, stop) for start, stop in merged)


def _sparse_protected_ranges(
    policy: str,
    manual_tokens: int,
    transformer_options,
    sequence_limit: int,
    *,
    sparse_reference_image: bool,
    sparse_reference_video: bool,
    sparse_reference_audio: bool,
    axis: str = "query",
) -> tuple[tuple[int, int], ...]:
    """Return token spans whose Query stays dense and whose KV stays exact."""
    if policy == "none":
        return ()
    if policy == "manual":
        stop = min(max(int(manual_tokens), 0), sequence_limit)
        return ((0, stop),) if stop else ()
    layout = attention_semantic_layout(transformer_options)
    if layout is None:
        prefix = _sparse_prefix_tokens(
            policy, manual_tokens, transformer_options, sequence_limit
        )
        return ((0, prefix),) if prefix else ()
    if axis not in {"query", "key"}:
        raise ValueError("sparse protected-range axis must be query or key")
    segments = layout.query_segments if axis == "query" else layout.key_segments

    def is_protected(segment) -> bool:
        reference_override = None
        if segment.role in {"reference_image", "reference_video_anchor"}:
            reference_override = sparse_reference_image
        elif segment.role in {"reference_video", "context_video", "pose_video"}:
            reference_override = sparse_reference_video
        elif segment.role == "reference_audio":
            reference_override = sparse_reference_audio
        allowed = (
            bool(reference_override)
            if reference_override is not None
            else (
                segment.sparse_query_allowed
                if axis == "query"
                else segment.sparse_key_allowed
            )
        )
        exact_kv = (
            not bool(reference_override)
            if reference_override is not None
            else segment.exact_kv
        )
        return not allowed if axis == "query" else exact_kv or not allowed

    return _coalesce_token_ranges(
        (
            (segment.start, segment.stop)
            for segment in segments
            if is_protected(segment)
        ),
        sequence_limit,
    )


def _required_sparse_layout_missing(
    transformer_options,
    query_length: int,
    key_length: int,
) -> bool:
    if not isinstance(transformer_options, dict):
        return False
    requirement = transformer_options.get(ATTENTION_LAYOUT_REQUIREMENT_KEY)
    if not isinstance(requirement, str) or not requirement:
        return False
    return not has_complete_attention_layout(
        transformer_options,
        query_length,
        key_sequence_length=key_length,
        provider=requirement,
    )


def _sparse_dense_schedule(
    transformer_options,
    prefix_steps: int,
    suffix_steps: int,
    state: dict[str, object] | None = None,
    *,
    track_step: bool = False,
) -> bool:
    """Select dense steps from the sampler's actual sigma schedule.

    Prefix and suffix counts are local to every sampler invocation. Runtime
    state lives in ``transformer_options`` by default, so a shared model can
    serve multiple samplers without mutable state leaking between ModelPatcher
    branches.
    """
    if (max(prefix_steps, suffix_steps) <= 0 and not track_step) or not isinstance(
        transformer_options, dict
    ):
        return False
    if state is None:
        state = transformer_options.setdefault(
            "_turing_utils_sparse_schedule_state", {}
        )
        if not isinstance(state, dict):
            state = {}
            transformer_options["_turing_utils_sparse_schedule_state"] = state
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigmas = transformer_options.get("sigmas")
    if not torch.is_tensor(sample_sigmas) or not torch.is_tensor(current_sigmas):
        return False
    if sample_sigmas.numel() < 2 or current_sigmas.numel() == 0:
        return False
    # Keep strong references to the tensors used for the cached decision.  ComfyUI
    # reuses the same transformer_options tensors for every block in one model
    # evaluation, then installs a new current-sigma tensor for the next sampler
    # evaluation.  Identity therefore avoids a device synchronization per block
    # and, unlike Tensor._version, is valid for tensors created in inference mode.
    if (
        state.get("sample_sigmas") is sample_sigmas
        and state.get("current_sigmas") is current_sigmas
        and state.get("policy") == (prefix_steps, suffix_steps)
    ):
        return bool(state["dense"])
    current = current_sigmas.flatten()[0].to(sample_sigmas)
    step = int(torch.argmin((sample_sigmas.flatten() - current).abs()).item())
    sampling_steps = sample_sigmas.numel() - 1
    effective_prefix_steps = min(prefix_steps, sampling_steps)
    effective_suffix_steps = min(suffix_steps, sampling_steps)
    dense = step < effective_prefix_steps or (
        effective_suffix_steps > 0 and step >= sampling_steps - effective_suffix_steps
    )
    state.clear()
    state.update(
        sample_sigmas=sample_sigmas,
        current_sigmas=current_sigmas,
        dense=dense,
        step=step,
        sampling_steps=sampling_steps,
        prefix_steps=effective_prefix_steps,
        suffix_steps=effective_suffix_steps,
        policy=(prefix_steps, suffix_steps),
    )
    return dense


def _sparse_dense_layer(
    transformer_options,
    dense_prefix_layers: int,
    dense_suffix_layers: int = 0,
) -> bool:
    if (dense_prefix_layers <= 0 and dense_suffix_layers <= 0) or not isinstance(
        transformer_options, dict
    ):
        return False
    layout = attention_semantic_layout(transformer_options)
    if layout is None:
        raw = (
            transformer_options.get(ATTENTION_LAYOUT_KEY)
            if isinstance(transformer_options, dict)
            else None
        )
        if not isinstance(raw, dict):
            return False
        layer_index = raw.get("layer_index", -1)
        layer_count = raw.get("layer_count", 0)
    else:
        layer_index = layout.layer_index
        layer_count = layout.layer_count
    if (
        isinstance(layer_count, int)
        and not isinstance(layer_count, bool)
        and layer_count > 0
        and dense_prefix_layers + dense_suffix_layers >= layer_count
    ):
        # An overlapping prefix/suffix intentionally turns the patch into the
        # selected dense backend without entering any Sol preprocessing path.
        return 0 <= layer_index < layer_count
    if 0 <= layer_index < dense_prefix_layers:
        return True
    return (
        dense_suffix_layers > 0
        and isinstance(layer_count, int)
        and not isinstance(layer_count, bool)
        and layer_count > 0
        and 0 <= layer_index < layer_count
        and layer_index >= max(layer_count - dense_suffix_layers, 0)
    )


def _inspect_sparse_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask,
    skip_reshape: bool,
    skip_output_reshape: bool,
    min_sequence_tokens: int,
    prefix_policy: str,
    manual_prefix_tokens: int,
    sparse_reference_image: bool,
    sparse_reference_video: bool,
    sparse_reference_audio: bool,
    transformer_options,
    kwargs: dict,
    reject_fully_protected_query: bool,
) -> tuple[
    tuple[AttentionCall, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    | None,
    str | None,
]:
    call, reason = inspect_turing_attention_call(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        enable_gqa=bool(kwargs.get("enable_gqa", False)),
        low_precision_attention=kwargs.get("low_precision_attention", True),
        is_causal=bool(kwargs.get("is_causal", False)),
        kernel="sol",
        require_long_sequence=True,
    )
    if reason is not None:
        return None, reason
    effective_min_sequence = min_sequence_tokens or SPARSE_AUTO_MIN_SEQUENCE
    if (
        call.query_tokens < effective_min_sequence
        or call.key_tokens < effective_min_sequence
    ):
        return None, f"sequences shorter than {effective_min_sequence} tokens"
    if _required_sparse_layout_missing(
        transformer_options,
        call.query_tokens,
        call.key_tokens,
    ):
        return None, "required semantic attention layout metadata is unavailable"
    dense_query_ranges = _sparse_protected_ranges(
        prefix_policy,
        manual_prefix_tokens,
        transformer_options,
        call.query_tokens,
        sparse_reference_image=bool(sparse_reference_image),
        sparse_reference_video=bool(sparse_reference_video),
        sparse_reference_audio=bool(sparse_reference_audio),
        axis="query",
    )
    if (
        reject_fully_protected_query
        and sum(stop - start for start, stop in dense_query_ranges) >= call.query_tokens
    ):
        return None, "all Query tokens are protected"
    exact_kv_ranges = _sparse_protected_ranges(
        prefix_policy,
        manual_prefix_tokens,
        transformer_options,
        call.key_tokens,
        sparse_reference_image=bool(sparse_reference_image),
        sparse_reference_video=bool(sparse_reference_video),
        sparse_reference_audio=bool(sparse_reference_audio),
        axis="key",
    )
    return (call, effective_min_sequence, dense_query_ranges, exact_kv_ranges), None


def inspect_sol_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask,
    skip_reshape: bool,
    skip_output_reshape: bool,
    min_sequence_tokens: int,
    prefix_policy: str,
    manual_prefix_tokens: int,
    skipped_residual: str,
    sparse_reference_image: bool,
    sparse_reference_video: bool,
    sparse_reference_audio: bool,
    transformer_options,
    kwargs: dict,
) -> tuple[SolAttentionCall | None, str | None]:
    common, reason = _inspect_sparse_attention_call(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        min_sequence_tokens=min_sequence_tokens,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        transformer_options=transformer_options,
        kwargs=kwargs,
        reject_fully_protected_query=False,
    )
    if reason is not None:
        return None, reason
    residual_subblocks = {"1x64": 1, "2x32": 2}.get(
        str(skipped_residual).strip().lower()
    )
    if residual_subblocks is None:
        raise ValueError("skipped_residual must be 1x64 or 2x32")
    call, effective_min_sequence, dense_query_ranges, exact_kv_ranges = common
    return SolAttentionCall(
        attention=call,
        effective_min_sequence=effective_min_sequence,
        dense_query_ranges=dense_query_ranges,
        exact_kv_ranges=exact_kv_ranges,
        residual_subblocks=residual_subblocks,
    ), None


def inspect_sla_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask,
    skip_reshape: bool,
    skip_output_reshape: bool,
    min_sequence_tokens: int,
    prefix_policy: str,
    manual_prefix_tokens: int,
    sparse_reference_image: bool,
    sparse_reference_video: bool,
    sparse_reference_audio: bool,
    transformer_options,
    kwargs: dict,
) -> tuple[SlaAttentionCall | None, str | None]:
    common, reason = _inspect_sparse_attention_call(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        min_sequence_tokens=min_sequence_tokens,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        transformer_options=transformer_options,
        kwargs=kwargs,
        reject_fully_protected_query=True,
    )
    if reason is not None:
        return None, reason
    call, effective_min_sequence, dense_query_ranges, exact_kv_ranges = common
    return SlaAttentionCall(
        attention=call,
        effective_min_sequence=effective_min_sequence,
        dense_query_ranges=dense_query_ranges,
        exact_kv_ranges=exact_kv_ranges,
    ), None


def prequantize_turing_sol_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    call: SolAttentionCall,
    *,
    routing_threshold: float,
    scale: float | None,
    use_w8a8: bool,
) -> PrequantizedAttentionCall:
    q, k, v = normalize_turing_attention_tensors(q, k, v, call.attention)
    if call.attention.tensor_layout == "NHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    state = load_turing_sage().prequantize_sol_sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        sm_scale=scale,
        dense_query_ranges=call.dense_query_ranges,
        exact_kv_ranges=call.exact_kv_ranges,
        threshold_sigma=routing_threshold,
        residual_subblocks=call.residual_subblocks,
        use_w8a8=bool(use_w8a8),
        key_tile_tokens=0,
        rotate_qk=True,
        stabilize_k=True,
    )
    return PrequantizedAttentionCall(state, call.attention)


def prequantize_turing_sol_attention_from_qk(
    qk,
    value: torch.Tensor,
    call: SolAttentionCall,
    *,
    routing_threshold: float,
    scale: float | None,
    use_w8a8: bool,
    value_source_indices: torch.Tensor | None = None,
) -> PrequantizedAttentionCall:
    state = load_turing_sage().prequantize_sol_sageattn_from_qk(
        qk,
        value,
        sm_scale=scale,
        dense_query_ranges=call.dense_query_ranges,
        exact_kv_ranges=call.exact_kv_ranges,
        threshold_sigma=routing_threshold,
        residual_subblocks=call.residual_subblocks,
        use_w8a8=bool(use_w8a8),
        key_tile_tokens=0,
        value_source_indices=value_source_indices,
    )
    return PrequantizedAttentionCall(state, call.attention)


def prequantize_turing_sla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    call: SlaAttentionCall,
    *,
    sparsity_ratio: float,
    scale: float | None,
    use_w8a8: bool,
) -> PrequantizedAttentionCall:
    q, k, v = normalize_turing_attention_tensors(q, k, v, call.attention)
    if call.attention.tensor_layout == "NHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    state = load_turing_sage().prequantize_sla_sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        sm_scale=scale,
        dense_query_ranges=call.dense_query_ranges,
        exact_kv_ranges=call.exact_kv_ranges,
        sparsity_ratio=sparsity_ratio,
        use_w8a8=bool(use_w8a8),
        key_tile_tokens=0,
        rotate_qk=True,
        stabilize_k=True,
    )
    return PrequantizedAttentionCall(state, call.attention)


def prequantize_turing_sla_attention_from_qk(
    qk,
    value: torch.Tensor,
    call: SlaAttentionCall,
    *,
    sparsity_ratio: float,
    scale: float | None,
    use_w8a8: bool,
) -> PrequantizedAttentionCall:
    state = load_turing_sage().prequantize_sla_sageattn_from_qk(
        qk,
        value,
        sm_scale=scale,
        dense_query_ranges=call.dense_query_ranges,
        exact_kv_ranges=call.exact_kv_ranges,
        sparsity_ratio=sparsity_ratio,
        use_w8a8=bool(use_w8a8),
        key_tile_tokens=0,
    )
    return PrequantizedAttentionCall(state, call.attention)


def turing_sol_attention_from_prequantized(
    quantized: PrequantizedAttentionCall,
    *,
    return_stats: bool,
):
    result = load_turing_sage().sol_sparse_sageattn_from_prequantized(
        quantized.kernel_state,
        return_stats=return_stats,
    )
    if return_stats:
        output, selected, possible_blocks = result
    else:
        output = result
    if quantized.call.tensor_layout == "NHD":
        output = output.transpose(1, 2)
    output = finish_turing_attention_output(output, quantized.call)
    return (output, selected, possible_blocks) if return_stats else output


def turing_sla_attention_from_prequantized(
    quantized: PrequantizedAttentionCall,
    *,
    return_stats: bool,
):
    result = load_turing_sage().sla_sparse_sageattn_from_prequantized(
        quantized.kernel_state,
        return_stats=return_stats,
    )
    if return_stats:
        output, selected, possible_blocks = result
    else:
        output = result
    if quantized.call.tensor_layout == "NHD":
        output = output.transpose(1, 2)
    output = finish_turing_attention_output(output, quantized.call)
    return (output, selected, possible_blocks) if return_stats else output


def turing_sol_sparse_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    debug_route_density: bool = False,
    debug_route_keys: set[tuple] | None = None,
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] | None = None,
    debug_context: dict | None = None,
    use_w8a8: bool = SPARSE_USE_W8A8,
    **kwargs,
) -> torch.Tensor:
    original_q, original_k, original_v = q, k, v
    common = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }

    def dense(reason: str):
        return _sparse_dense_baseline(
            reason,
            fallback,
            original_q,
            original_k,
            original_v,
            heads,
            **common,
        )

    transformer_options = kwargs.get("transformer_options")
    sol_call, reason = inspect_sol_attention_call(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        min_sequence_tokens=min_sequence_tokens,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        skipped_residual=skipped_residual,
        sparse_reference_image=bool(sparse_reference_image),
        sparse_reference_video=bool(sparse_reference_video),
        sparse_reference_audio=bool(sparse_reference_audio),
        transformer_options=transformer_options,
        kwargs=kwargs,
    )
    if reason is not None:
        return dense(reason)
    call = sol_call.attention
    input_dtype = call.input_dtype
    effective_min_sequence = sol_call.effective_min_sequence
    dense_query_ranges = sol_call.dense_query_ranges
    exact_kv_ranges = sol_call.exact_kv_ranges
    residual_subblocks = sol_call.residual_subblocks
    skipped_residual = "1x64" if residual_subblocks == 1 else "2x32"
    q_shape = (call.batch, call.heads, call.query_tokens, call.head_dim)
    k_shape = (call.batch, call.kv_heads, call.key_tokens, call.head_dim)
    kernel_key = (
        q.device.index,
        input_dtype,
        q_shape,
        k_shape,
        effective_min_sequence,
        dense_query_ranges,
        exact_kv_ranges,
        routing_threshold,
        residual_subblocks,
        bool(sparse_reference_image),
        bool(sparse_reference_video),
        bool(sparse_reference_audio),
        bool(use_w8a8),
    )
    if kernel_key not in _LOGGED_SPARSE_KERNELS:
        LOG.debug(
            "Bundled Sol sparse attention active: dtype=%s Q=%s K=%s "
            "min_sequence=%d prefix_policy=%s dense_query_ranges=%s exact_kv_ranges=%s "
            "selected_qk=int8 score_domain=int8_consistent threshold=%.2f "
            "skipped_residual=%s local_radius=1 "
            "sparse_reference=(image=%s,video=%s,audio=%s) pv=%s",
            input_dtype,
            q_shape,
            k_shape,
            effective_min_sequence,
            prefix_policy,
            dense_query_ranges,
            exact_kv_ranges,
            routing_threshold,
            skipped_residual,
            bool(sparse_reference_image),
            bool(sparse_reference_video),
            bool(sparse_reference_audio),
            "w8a8" if use_w8a8 else "fp16",
        )
        _LOGGED_SPARSE_KERNELS.add(kernel_key)

    route_keys = debug_route_keys if debug_route_keys is not None else set()
    context = debug_context or {}
    step = context.get("step")
    layer_index = context.get("layer_index")
    layer_count = context.get("layer_count")
    last_sparse_layer = context.get(
        "last_sparse_layer",
        layer_count - 1 if isinstance(layer_count, int) else None,
    )
    aggregate_route_stats = (
        debug_route_density
        and debug_route_state is not None
        and isinstance(step, int)
        and not isinstance(step, bool)
        and isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        and isinstance(layer_count, int)
        and not isinstance(layer_count, bool)
        and layer_count > 0
        and 0 <= layer_index < layer_count
        and isinstance(last_sparse_layer, int)
        and not isinstance(last_sparse_layer, bool)
        and 0 <= last_sparse_layer < layer_count
    )
    collect_route_stats = debug_route_density and (
        aggregate_route_stats or kernel_key not in route_keys
    )
    q, k, v = normalize_turing_attention_tensors(q, k, v, call)
    if call.tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    sparse_result = _sol_sparse_sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        sm_scale=kwargs.get("scale"),
        dense_query_ranges=dense_query_ranges,
        exact_kv_ranges=exact_kv_ranges,
        threshold_sigma=routing_threshold,
        residual_subblocks=residual_subblocks,
        return_stats=collect_route_stats,
        use_w8a8=bool(use_w8a8),
        key_tile_tokens=0,
        rotate_qk=True,
        stabilize_k=True,
    )
    if collect_route_stats:
        output, selected_device, possible_blocks = sparse_result
        try:
            protected_query_tokens = sum(
                stop - start for start, stop in dense_query_ranges
            )
            sparse_query_tokens = call.query_tokens - protected_query_tokens
            if aggregate_route_stats:
                aggregate_key = (step, context.get("sampling_steps"), kernel_key)
                entries = debug_route_state.setdefault(aggregate_key, [])
                entries.append((selected_device, possible_blocks, layer_index))
                if layer_index == last_sparse_layer:
                    selected = torch.cat([entry[0] for entry in entries]).float()
                    possible = torch.tensor(
                        [entry[1] for entry in entries],
                        device=selected.device,
                        dtype=torch.float32,
                    )
                    density = selected / possible.clamp_min(1.0)
                    summary = (
                        torch.stack(
                            (
                                selected.sum(),
                                possible.sum(),
                                density.min(),
                                density.mean(),
                                density.max(),
                            )
                        )
                        .cpu()
                        .tolist()
                    )
                    first_layer = min(entry[2] for entry in entries)
                    last_layer = max(entry[2] for entry in entries)
                    LOG.info(
                        "[Turing sparse debug] step=%s/%s layers=%d-%d calls=%d "
                        "selected=%d/%d density[min/mean/max]=%.4f/%.4f/%.4f "
                        "Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d threshold=%.2f "
                        "protected_q=%d local=1 residual=%s",
                        step,
                        context.get("sampling_steps"),
                        first_layer,
                        last_layer,
                        len(entries),
                        int(summary[0]),
                        int(summary[1]),
                        summary[2],
                        summary[3],
                        summary[4],
                        call.query_tokens,
                        sparse_query_tokens,
                        call.key_tokens,
                        call.heads,
                        call.kv_heads,
                        routing_threshold,
                        protected_query_tokens,
                        skipped_residual,
                    )
                    del debug_route_state[aggregate_key]
            else:
                selected_blocks = int(selected_device.item())
                LOG.info(
                    "[Turing sparse debug] Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d selected=%d/%d "
                    "density=%.4f threshold=%.2f protected_q=%d local=1 "
                    "residual=%s step=%s/%s layer=%s/%s",
                    call.query_tokens,
                    sparse_query_tokens,
                    call.key_tokens,
                    call.heads,
                    call.kv_heads,
                    selected_blocks,
                    possible_blocks,
                    selected_blocks / possible_blocks if possible_blocks else 0.0,
                    routing_threshold,
                    protected_query_tokens,
                    skipped_residual,
                    context.get("step"),
                    context.get("sampling_steps"),
                    layer_index,
                    layer_count,
                )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            LOG.warning_once(
                "[Turing sparse debug] route density unavailable: %s", error
            )
        if not aggregate_route_stats:
            route_keys.add(kernel_key)
    else:
        output = sparse_result
    result = (
        output
        if call.skip_output_reshape
        else output.transpose(1, 2).reshape(call.batch, -1, call.heads * call.head_dim)
    )
    return result.to(input_dtype) if input_dtype == torch.float32 else result


def turing_sla_sparse_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    min_sequence_tokens: int = 0,
    sparsity_ratio: float = 0.85,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    debug_route_density: bool = False,
    use_w8a8: bool = SPARSE_USE_W8A8,
    **kwargs,
) -> torch.Tensor:
    common = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }
    sla_call, reason = inspect_sla_attention_call(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        min_sequence_tokens=min_sequence_tokens,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        sparse_reference_image=bool(sparse_reference_image),
        sparse_reference_video=bool(sparse_reference_video),
        sparse_reference_audio=bool(sparse_reference_audio),
        transformer_options=kwargs.get("transformer_options"),
        kwargs=kwargs,
    )
    if reason is not None:
        return _sparse_dense_baseline(
            f"SLA: {reason}", fallback, q, k, v, heads, **common
        )
    call = sla_call.attention
    kernel_key = (
        "sla",
        q.device.index,
        call.input_dtype,
        call.query_tokens,
        call.key_tokens,
        call.heads,
        call.kv_heads,
        sla_call.dense_query_ranges,
        sla_call.exact_kv_ranges,
        float(sparsity_ratio),
        bool(use_w8a8),
    )
    if kernel_key not in _LOGGED_SPARSE_KERNELS:
        LOG.debug(
            "Bundled SLA sparse attention active: dtype=%s Q=%d K=%d "
            "topology=128x64 sparsity_ratio=%.2f smooth_k=True "
            "dense_query_ranges=%s exact_kv_ranges=%s pv=%s",
            call.input_dtype,
            call.query_tokens,
            call.key_tokens,
            sparsity_ratio,
            sla_call.dense_query_ranges,
            sla_call.exact_kv_ranges,
            "w8a8" if use_w8a8 else "fp16",
        )
        _LOGGED_SPARSE_KERNELS.add(kernel_key)
    quantized = prequantize_turing_sla_attention(
        q,
        k,
        v,
        sla_call,
        sparsity_ratio=sparsity_ratio,
        scale=kwargs.get("scale"),
        use_w8a8=use_w8a8,
    )
    result = turing_sla_attention_from_prequantized(
        quantized,
        return_stats=debug_route_density,
    )
    if not debug_route_density:
        return result
    output, selected, possible = result
    selected_blocks = int(selected.item())
    LOG.warning_once(
        "[Turing SLA debug] Q=%d K=%d selected=%d/%d density=%.4f "
        "target_sparsity=%.2f protected_q=%d",
        call.query_tokens,
        call.key_tokens,
        selected_blocks,
        possible,
        selected_blocks / possible if possible else 0.0,
        sparsity_ratio,
        sum(stop - start for start, stop in sla_call.dense_query_ranges),
    )
    return output
