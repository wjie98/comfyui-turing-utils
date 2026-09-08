"""MiniMax H3 memory planning and capability-based CUDA fusions."""

from __future__ import annotations

import dataclasses
import inspect
import math
import weakref
from collections import Counter
from collections.abc import Sequence

import torch

from ...log import get_logger
from ..methods import OriginalMethod, weak_method
from ...attention.integration import AttentionSiteStatus, execute_projected_attention
from ...attention.layout import ATTENTION_LAYOUT_REQUIREMENT_KEY, attention_semantic_layout
from ...attention.protocol import (
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
    prepared_attention_executor,
)
from ...attention.stable import (
    fused_qk_preprocessing_available,
    precompute_turing_k_anchor,
    prequantize_turing_qk,
    reusable_k_anchor_available,
)
from ...hardware import is_supported_attention_device
from ...kernel_api import kernel_extension_has_symbol, load_kernel_package
from ...profiling import CUDA_PHASE_PROFILER
from .activation_policy import (
    decide_activation_chunks,
    decide_attention_heads,
    decide_ffn_channels,
    ensure_dynamic_vram_headroom,
    estimate_attention_lifecycle_peak,
)
from ...runtime.capabilities import kernel_capabilities
from .layout import (
    ATTENTION_LAYOUT_KEY,
    RUNTIME_CONTEXT_ATTR,
    RUNTIME_OUTER_WRAPPER_KEY,
    ensure_minimax_attention_layout_provider,
    mark_forward_as_minimax_layout_provider,
    minimax_temporal_topology,
    publish_minimax_attention_layout,
)
from ...quantization.fusions import (
    convrot_linear_input_act_from_weight,
    convrot_w8_output_slice,
    convrot_w8_plain_tensors,
    convrot_weight_kind,
    fused_convrot_linear_input_act,
    is_turing_convrot_linear,
    segmented_mod_gate,
    segmented_mod_gate_rms_adaln,
    segmented_rms_adaln,
)
from ...quantization.dispatch import (
    int8_linear_from_quantized,
    quantize_convrot_int8_activation,
    quantize_convrot_swiglu_activation,
    quantize_convrot_swiglu_with_scale,
    quantize_convrot_from_partials,
    rotate_convrot_swiglu_shard_inplace,
    convrot_swiglu_channel_sharding_available,
    convrot_swiglu_half_width_available,
)
from .memory_planning import (
    _MEMORY_ADAPTER_ATTR,
    _MEMORY_CONTEXT_ATTR,
    _MEMORY_SHAPE_KEY,
    _MiniMaxActivationProfile,
    _MiniMaxMemoryCond,
    _MiniMaxMemoryShape,
    _activation_profile,
    _install_memory_planning,
    _linear_workspace_requirements,
    _make_extra_conds,
    _make_extra_conds_shapes,
    _make_memory_required,
    _make_outer_sample_wrapper,
    _minimax_memory_shape,
)


LOG = get_logger("minimax.fusion")


def _profile_cuda(phase: str, function, /, *args, **kwargs):
    if CUDA_PHASE_PROFILER.enabled:
        return CUDA_PHASE_PROFILER.call(phase, function, *args, **kwargs)
    return function(*args, **kwargs)


def _virtual_kv_logical_rows(
    transformer_options: dict,
    physical_rows: int,
) -> int | None:
    if transformer_options.get("turing_utils_attention_strategy") != "h3_virtual_kv":
        return None
    layout = attention_semantic_layout(transformer_options)
    if layout is None or layout.provider != "minimax_h3":
        return None
    if layout.validate(physical_rows, physical_rows) is not None:
        return None
    targets = [segment for segment in layout.key_segments if segment.role == "target_video"]
    if len(targets) != 1:
        return None
    target = targets[0]
    topologies = [
        topology
        for topology in layout.topologies
        if topology.topology_id == target.topology_id
    ]
    if len(topologies) != 1:
        return None
    tokens_per_frame = int(topologies[0].tokens_per_frame)
    if target.stop - target.start != 2 * tokens_per_frame:
        return None
    return physical_rows + 5 * tokens_per_frame


def _direct_int8_output_available() -> bool:
    return kernel_extension_has_symbol("turing_int8_linear_out")


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_FORWARD_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
)
_ATTENTION_FORWARD_PARAMETERS = (
    "x",
    "rope_freqs",
    "transformer_options",
)
_PREPARED_ATTENTION_FORWARD_ATTR = "_turing_utils_minimax_prepared_attention"
_OUTER_SAMPLE_WRAPPER_KEY = RUNTIME_OUTER_WRAPPER_KEY
_ATTENTION_LAYOUT_KEY = ATTENTION_LAYOUT_KEY
_STREAMED_QKV_EXECUTOR_ATTR = "turing_utils_streamed_qkv_executor"
_ATTENTION_FALLBACK_WARNINGS_KEY = object()


def _runtime_activation_plan(base_model):
    if base_model is None:
        return None
    try:
        context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    except ReferenceError:
        return None
    return context.get("activation_plan") if isinstance(context, dict) else None


def _warn_attention_fallback(
    transformer_options,
    *,
    path: str,
    rows: int,
    reason: str,
) -> None:
    signature = (str(path), int(rows), str(reason))
    if isinstance(transformer_options, dict):
        warnings = transformer_options.setdefault(
            _ATTENTION_FALLBACK_WARNINGS_KEY, set()
        )
        if not isinstance(warnings, set):
            warnings = set()
            transformer_options[_ATTENTION_FALLBACK_WARNINGS_KEY] = warnings
        if signature in warnings:
            return
        warnings.add(signature)
    LOG.warning(
        "MiniMax prepared attention fallback: path=%s rows=%d reason=%s",
        *signature,
    )


def _weak_model_reference(base_model):
    if base_model is None:
        return None
    try:
        return weakref.proxy(base_model)
    except TypeError:
        # Test doubles and a few third-party wrappers are not weak-referenceable.
        return base_model


class _RuntimeDispatchAudit:
    """Report the first full MiniMax block pass without CUDA timing events."""

    def __init__(self, expected_blocks: int, expected_mlps: int):
        self.expected = {"block": expected_blocks, "mlp": expected_mlps}
        self.counts = {"block": Counter(), "mlp": Counter()}
        self.dtypes = {"block": Counter(), "mlp": Counter()}
        self.shapes = {"block": Counter(), "mlp": Counter()}
        self.reasons = {"block": Counter(), "mlp": Counter()}
        self.logged_phases: set[str] = set()

    def record(
        self,
        phase: str,
        fused: bool,
        x: torch.Tensor,
        reason: str | None = None,
    ) -> None:
        if phase in self.logged_phases or self.expected[phase] == 0:
            return
        self.counts[phase]["fused" if fused else "fallback"] += 1
        self.dtypes[phase][str(x.dtype)] += 1
        self.shapes[phase][str(tuple(x.shape))] += 1
        if reason is not None:
            self.reasons[phase][reason] += 1

        calls = self.counts[phase]["fused"] + self.counts[phase]["fallback"]
        if calls < self.expected[phase]:
            return

        log = LOG.warning if self.counts[phase]["fallback"] else LOG.info
        log(
            "MiniMax fused runtime dispatch: phase=%s fused=%d fallback=%d "
            "dtypes=[%s] shapes=[%s] reasons=[%s]",
            phase,
            self.counts[phase]["fused"],
            self.counts[phase]["fallback"],
            _format_counts(self.dtypes[phase].elements()),
            _format_counts(self.shapes[phase].elements()),
            _format_counts(self.reasons[phase].elements()) or "none",
        )
        self.logged_phases.add(phase)


def _format_counts(values) -> str:
    counts = Counter(values)
    return ",".join(
        f"{value}:{count}"
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _audit_fc2(blocks: Sequence[torch.nn.Module]) -> int:
    linears = []
    for block in blocks:
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "fc2"):
            linears.append(mlp.fc2)
    if not linears:
        return 0
    kinds = [
        convrot_weight_kind(getattr(linear, "weight", None)) or "other"
        for linear in linears
    ]
    eligible = sum(kind != "other" for kind in kinds)
    LOG.info(
        "MiniMax fused fc2 dispatch: blocks=%d eligible=%d formats=[%s]",
        len(linears),
        eligible,
        _format_counts(kinds),
    )
    return eligible


def _compatible_block_forward(block_type: type[torch.nn.Module]) -> bool:
    parameters = tuple(inspect.signature(block_type.forward).parameters)
    return parameters == ("self", *_BLOCK_FORWARD_PARAMETERS)


def _compatible_attention_forward(attention_type: type[torch.nn.Module]) -> bool:
    parameters = tuple(inspect.signature(attention_type.forward).parameters)
    return parameters == ("self", *_ATTENTION_FORWARD_PARAMETERS)


def _qk_transform(attention, x, rope_freqs) -> QKTransformSpec:
    import comfy.model_management

    query_norm = comfy.model_management.cast_to(
        attention.q_norm.weight, device=x.device, dtype=x.dtype
    )
    key_norm = comfy.model_management.cast_to(
        attention.k_norm.weight, device=x.device, dtype=x.dtype
    )
    rot_dim = int(rope_freqs.shape[-3] * 2) if rope_freqs is not None else 0
    return QKTransformSpec(
        query_norm=RMSNormSpec(
            query_norm, float(attention.q_norm.eps), "head"
        ),
        key_norm=RMSNormSpec(
            key_norm, float(attention.k_norm.eps), "head"
        ),
        rotary=RotaryEmbeddingSpec(
            rope_freqs,
            rot_dim,
            "split_half" if rope_freqs is not None else "none",
        ),
    )


def _linear_with_cast_weight(linear, x, weight, bias):
    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        import comfy.model_management

        x = x * comfy.model_management.cast_to_device(
            pre_quant_scale, x.device, x.dtype
        )
    function = getattr(linear, "_forward", None)
    return (
        function(x, weight, bias)
        if callable(function)
        else torch.nn.functional.linear(x, weight, bias)
    )


def _slice_qk_transform(
    transform: QKTransformSpec,
    start: int,
    stop: int,
    full_rows: int,
) -> QKTransformSpec:
    freqs = transform.freqs
    if torch.is_tensor(freqs) and freqs.ndim >= 2 and freqs.shape[1] == full_rows:
        rotary = dataclasses.replace(transform.rotary, freqs=freqs[:, start:stop])
        return dataclasses.replace(transform, rotary=rotary)
    return transform


def _sample_qk_transform(
    transform: QKTransformSpec,
    indices: torch.Tensor,
    full_rows: int,
) -> QKTransformSpec:
    freqs = transform.freqs
    if torch.is_tensor(freqs) and freqs.ndim >= 2 and freqs.shape[1] == full_rows:
        rotary = dataclasses.replace(
            transform.rotary,
            freqs=freqs.index_select(1, indices),
        )
        return dataclasses.replace(transform, rotary=rotary)
    return transform


def _global_k_anchor(
    attention,
    x: torch.Tensor,
    weight,
    bias,
    transform: QKTransformSpec,
    inner: int,
    heads: int,
    head_dim: int,
):
    sequence = int(x.shape[0])
    sample_rows = [index * (sequence - 1) // 8 for index in range(9)]
    sample_indices = torch.tensor(
        sample_rows, dtype=torch.long, device=x.device
    )
    sampled = x.index_select(0, sample_indices)
    projected = _linear_with_cast_weight(
        attention.qkv_proj, sampled, weight, bias
    )
    key = projected[:, inner : 2 * inner]
    key = key.view(9, heads, head_dim).transpose(0, 1).unsqueeze(0)
    sample_transform = _sample_qk_transform(
        transform, sample_indices, sequence
    )
    anchor_indices, anchor_values = precompute_turing_k_anchor(
        key, sample_transform
    )
    lookup = sample_indices.to(dtype=torch.int32)
    selected = anchor_indices.clamp_min(0).to(dtype=torch.long)
    anchor_indices = torch.where(
        anchor_indices >= 0,
        lookup[selected],
        anchor_indices,
    )
    return anchor_indices, anchor_values


def _qkv_input_tile(linear, x: torch.Tensor) -> torch.Tensor:
    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is None:
        return x
    import comfy.model_management

    scale = comfy.model_management.cast_to_device(
        pre_quant_scale, x.device, x.dtype
    )
    return x * scale


def _quantize_qkv_rows(linear, x: torch.Tensor):
    return quantize_convrot_int8_activation(
        _qkv_input_tile(linear, x), 256
    )


def _cache_quantized_qkv_input(linear, x: torch.Tensor, chunk_rows: int):
    qactivation = torch.empty_like(x, dtype=torch.int8)
    activation_scale = torch.empty(
        (x.shape[0],), dtype=torch.float32, device=x.device
    )
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        tile, scale = _quantize_qkv_rows(linear, x[start:stop])
        qactivation[start:stop].copy_(tile)
        activation_scale[start:stop].copy_(scale.reshape(-1))
        del tile, scale
    return qactivation, activation_scale


def _w8_qkv_component(
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    component: int,
    head_start: int,
    head_stop: int,
    heads: int,
    head_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    inner = heads * head_dim
    start = component * inner + head_start * head_dim
    stop = component * inner + head_stop * head_dim
    return convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        start,
        stop,
        output_dtype,
    )


def _head_group_k_anchor(
    attention,
    x: torch.Tensor,
    transform: QKTransformSpec,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_start: int,
    head_stop: int,
    quantized_input,
):
    sequence = int(x.shape[0])
    sample_rows = [index * (sequence - 1) // 8 for index in range(9)]
    sample_indices = torch.tensor(
        sample_rows, dtype=torch.long, device=x.device
    )
    if quantized_input is None:
        qactivation, activation_scale = _quantize_qkv_rows(
            attention.qkv_proj, x.index_select(0, sample_indices)
        )
    else:
        cached_activation, cached_scale = quantized_input
        qactivation = cached_activation.index_select(0, sample_indices)
        activation_scale = cached_scale.index_select(0, sample_indices)
    key = _w8_qkv_component(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        component=1,
        head_start=head_start,
        head_stop=head_stop,
        heads=int(attention.heads),
        head_dim=int(attention.head_dim),
        output_dtype=x.dtype,
    )
    group = head_stop - head_start
    key = key.view(9, group, attention.head_dim).transpose(0, 1).unsqueeze(0)
    sample_transform = _sample_qk_transform(
        transform, sample_indices, sequence
    )
    anchor_indices, anchor_values = precompute_turing_k_anchor(
        key, sample_transform
    )
    lookup = sample_indices.to(dtype=torch.int32)
    selected = anchor_indices.clamp_min(0).to(dtype=torch.long)
    anchor_indices = torch.where(
        anchor_indices >= 0,
        lookup[selected],
        anchor_indices,
    )
    return anchor_indices, anchor_values


def _stream_qkv_head_group(
    attention,
    x: torch.Tensor,
    transform: QKTransformSpec,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_start: int,
    head_stop: int,
    chunk_rows: int,
    quantized_input,
):
    """Project one full-sequence head group into compact Q/K/V storage."""
    sequence = int(x.shape[0])
    group = head_stop - head_start
    head_dim = int(attention.head_dim)
    q_int8 = torch.empty(
        (1, group, sequence, head_dim), dtype=torch.int8, device=x.device
    )
    k_int8 = torch.empty_like(q_int8)
    q_scale = torch.empty(
        (1, group, ((sequence + 63) // 64) * 4),
        dtype=torch.float32,
        device=x.device,
    )
    k_scale = torch.empty(
        (1, group, (sequence + 63) // 64),
        dtype=torch.float32,
        device=x.device,
    )
    value = torch.empty(
        (1, group, sequence, head_dim), dtype=x.dtype, device=x.device
    )
    k_anchor = _head_group_k_anchor(
        attention,
        x,
        transform,
        qweight,
        weight_scale,
        bias,
        head_start,
        head_stop,
        quantized_input,
    )
    qk_type = None
    route_original_basis = False
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        if quantized_input is None:
            qactivation, activation_scale = _quantize_qkv_rows(
                attention.qkv_proj, x[start:stop]
            )
        else:
            cached_activation, cached_scale = quantized_input
            qactivation = cached_activation[start:stop]
            activation_scale = cached_scale[start:stop]
        query = _w8_qkv_component(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            bias,
            component=0,
            head_start=head_start,
            head_stop=head_stop,
            heads=int(attention.heads),
            head_dim=head_dim,
            output_dtype=x.dtype,
        )
        key = _w8_qkv_component(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            bias,
            component=1,
            head_start=head_start,
            head_stop=head_stop,
            heads=int(attention.heads),
            head_dim=head_dim,
            output_dtype=x.dtype,
        )
        rows = stop - start
        query = query.view(rows, group, head_dim).transpose(0, 1).unsqueeze(0)
        key = key.view(rows, group, head_dim).transpose(0, 1).unsqueeze(0)
        tile_transform = _slice_qk_transform(
            transform, start, stop, sequence
        )
        q_scale_start = (start // 64) * 4
        k_scale_start = start // 64
        tile_q_scale = q_scale[
            :,
            :,
            q_scale_start : q_scale_start + ((rows + 63) // 64) * 4,
        ]
        tile_k_scale = k_scale[
            :,
            :,
            k_scale_start : k_scale_start + (rows + 63) // 64,
        ]
        tile_qk = prequantize_turing_qk(
            query,
            key,
            tile_transform,
            kernel="sol",
            k_anchor=k_anchor,
            qk_output=(
                q_int8[:, :, start:stop],
                tile_q_scale,
                k_int8[:, :, start:stop],
                tile_k_scale,
            ),
        )
        value_tile = _w8_qkv_component(
            qactivation,
            activation_scale,
            qweight,
            weight_scale,
            bias,
            component=2,
            head_start=head_start,
            head_stop=head_stop,
            heads=int(attention.heads),
            head_dim=head_dim,
            output_dtype=x.dtype,
        )
        value[:, :, start:stop].copy_(
            value_tile.view(rows, group, head_dim).transpose(0, 1).unsqueeze(0)
        )
        qk_type = type(tile_qk)
        route_original_basis = bool(tile_qk.route_original_basis)
        del query, key, value_tile, tile_qk
        if quantized_input is None:
            del qactivation, activation_scale

    if qk_type is None:
        raise RuntimeError("head-sharded H3 QKV projection produced no tiles")
    qk = qk_type(
        query_int8=q_int8,
        query_scale=q_scale,
        key_int8=k_int8,
        key_scale=k_scale,
        tensor_layout="HND",
        input_dtype=x.dtype,
        original_head_dim=head_dim,
        route_original_basis=route_original_basis,
    )
    return qk, value


def _project_qkv_head_group(
    attention,
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_start: int,
    head_stop: int,
    chunk_rows: int,
    quantized_input,
):
    sequence = int(x.shape[0])
    group = head_stop - head_start
    head_dim = int(attention.head_dim)
    outputs = [
        torch.empty(
            (sequence, group, head_dim), dtype=x.dtype, device=x.device
        )
        for _ in range(3)
    ]
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        if quantized_input is None:
            qactivation, activation_scale = _quantize_qkv_rows(
                attention.qkv_proj, x[start:stop]
            )
        else:
            cached_activation, cached_scale = quantized_input
            qactivation = cached_activation[start:stop]
            activation_scale = cached_scale[start:stop]
        rows = stop - start
        for component, destination in enumerate(outputs):
            tile = _w8_qkv_component(
                qactivation,
                activation_scale,
                qweight,
                weight_scale,
                bias,
                component=component,
                head_start=head_start,
                head_stop=head_stop,
                heads=int(attention.heads),
                head_dim=head_dim,
                output_dtype=x.dtype,
            )
            destination[start:stop].copy_(tile.view(rows, group, head_dim))
            del tile
        if quantized_input is None:
            del qactivation, activation_scale
    return tuple(outputs)


def _apply_minimax_qk_transform(attention, query, key, rope_freqs):
    import comfy.model_management
    import comfy.quant_ops

    sequence, group, head_dim = query.shape
    if rope_freqs is not None:
        query = query.view(1, sequence, group, head_dim)
        key = key.view(1, sequence, group, head_dim)
        query_weight = comfy.model_management.cast_to(
            attention.q_norm.weight, device=query.device
        )
        key_weight = comfy.model_management.cast_to(
            attention.k_norm.weight, device=key.device
        )
        rot_dim = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            query, key = comfy.quant_ops.ck.rms_rope_split_half(
                query,
                key,
                rope_freqs,
                query_weight,
                key_weight,
                epsilon=attention.q_norm.eps,
                rot_dim=rot_dim,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                query,
                key,
                rope_freqs,
                query_weight,
                key_weight,
                epsilon=attention.q_norm.eps,
                rot_dim=rot_dim,
            )
        return query[0], key[0]
    return attention.q_norm(query), attention.k_norm(key)


def _head_sharded_attention(
    attention,
    x: torch.Tensor,
    rope_freqs,
    transform: QKTransformSpec,
    transformer_options: dict,
    attention_container,
    executor,
    head_group: int,
    cache_quantized_input: bool,
):
    import comfy.ops
    from comfy.ldm.modules.attention import optimized_attention

    comfy.ops.run_every_op()
    weight, bias, offload_stream = _profile_cuda(
        "minimax.qkv_weight_wait",
        comfy.ops.cast_bias_weight,
        attention.qkv_proj,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        plain = convrot_w8_plain_tensors(weight)
        if plain is None:
            _warn_attention_fallback(
                transformer_options,
                path="head_sharded",
                rows=int(x.shape[0]),
                reason="cast QKV weight is not plain W8A8",
            )
            return None
        qweight, weight_scale = plain
        quantized_input = (
            _cache_quantized_qkv_input(attention.qkv_proj, x, 16_384)
            if cache_quantized_input
            else None
        )
        sequence = int(x.shape[0])
        heads = int(attention.heads)
        head_dim = int(attention.head_dim)
        output = torch.empty(
            (sequence, heads * head_dim), dtype=x.dtype, device=x.device
        )
        streamed_executor = (
            getattr(executor, _STREAMED_QKV_EXECUTOR_ATTR, None)
            if executor is not None
            else None
        )
        compact = callable(streamed_executor) and reusable_k_anchor_available()
        for head_start in range(0, heads, head_group):
            head_stop = min(head_start + head_group, heads)
            group = head_stop - head_start
            if compact:
                qk, value = _stream_qkv_head_group(
                    attention,
                    x,
                    transform,
                    qweight,
                    weight_scale,
                    bias,
                    head_start,
                    head_stop,
                    16_384,
                    quantized_input,
                )
                outcome = streamed_executor(
                    qk,
                    value,
                    heads=group,
                    qk_transform=transform,
                    transformer_options=transformer_options,
                )
                del qk, value
                if not outcome.supported:
                    _warn_attention_fallback(
                        transformer_options,
                        path="head_sharded",
                        rows=sequence,
                        reason=outcome.reason,
                    )
                    return None
                group_output = outcome.output.squeeze(0)
            else:
                query, key, value = _project_qkv_head_group(
                    attention,
                    x,
                    qweight,
                    weight_scale,
                    bias,
                    head_start,
                    head_stop,
                    16_384,
                    quantized_input,
                )
                if (
                    transformer_options.get("turing_utils_attention_strategy")
                    == "h3_virtual_kv"
                ):
                    outcome = execute_projected_attention(
                        query.transpose(0, 1).unsqueeze(0),
                        key.transpose(0, 1).unsqueeze(0),
                        value.transpose(0, 1).unsqueeze(0),
                        heads=group,
                        qk_transform=transform,
                        transformer_options=transformer_options,
                        container_factory=attention_container,
                    )
                    if not outcome.supported:
                        _warn_attention_fallback(
                            transformer_options,
                            path="head_sharded_virtual_kv",
                            rows=sequence,
                            reason=outcome.reason,
                        )
                        return None
                    group_output = outcome.output.squeeze(0)
                else:
                    query, key = _apply_minimax_qk_transform(
                        attention, query, key, rope_freqs
                    )
                    group_output = optimized_attention(
                        attention_container(query.transpose(0, 1).unsqueeze(0)),
                        attention_container(key.transpose(0, 1).unsqueeze(0)),
                        attention_container(value.transpose(0, 1).unsqueeze(0)),
                        group,
                        mask=None,
                        skip_reshape=True,
                        transformer_options=transformer_options,
                    ).squeeze(0)
                del query, key, value
            _profile_cuda(
                "minimax.head_output_store",
                output[:, head_start * head_dim : head_stop * head_dim].copy_,
                group_output,
            )
            del group_output
        return output
    finally:
        comfy.ops.uncast_bias_weight(
            attention.qkv_proj, weight, bias, offload_stream
        )


def _stream_qkv_projection(
    attention,
    x: torch.Tensor,
    transform: QKTransformSpec,
    chunk_rows: int,
):
    """Project H3 QKV by rows while retaining only Q/K INT8 and V BF16."""
    import comfy.ops

    sequence = int(x.shape[0])
    heads = int(attention.heads)
    head_dim = int(attention.head_dim)
    inner = heads * head_dim
    q_int8 = torch.empty(
        (1, heads, sequence, head_dim), dtype=torch.int8, device=x.device
    )
    k_int8 = torch.empty_like(q_int8)
    q_scale = torch.empty(
        (1, heads, ((sequence + 63) // 64) * 4),
        dtype=torch.float32,
        device=x.device,
    )
    k_scale = torch.empty(
        (1, heads, (sequence + 63) // 64),
        dtype=torch.float32,
        device=x.device,
    )
    value = torch.empty(
        (1, heads, sequence, head_dim), dtype=x.dtype, device=x.device
    )

    comfy.ops.run_every_op()
    original_w8 = convrot_weight_kind(attention.qkv_proj.weight) == "w8a8"
    weight, bias, offload_stream = _profile_cuda(
        "minimax.qkv_weight_wait",
        comfy.ops.cast_bias_weight,
        attention.qkv_proj,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=original_w8,
    )
    qk_type = None
    route_original_basis = False
    try:
        plain = convrot_w8_plain_tensors(weight) if original_w8 else None
        if plain is not None:
            qweight, weight_scale = plain
            k_anchor = _head_group_k_anchor(
                attention,
                x,
                transform,
                qweight,
                weight_scale,
                bias,
                0,
                heads,
                None,
            )
        else:
            k_anchor = _global_k_anchor(
                attention,
                x,
                weight,
                bias,
                transform,
                inner,
                heads,
                head_dim,
            )
        for start in range(0, sequence, chunk_rows):
            stop = min(start + chunk_rows, sequence)
            if plain is None:
                projected = _profile_cuda(
                    "minimax.qkv_projection_tile",
                    _linear_with_cast_weight,
                    attention.qkv_proj,
                    x[start:stop],
                    weight,
                    bias,
                )
            else:
                qactivation, activation_scale = _profile_cuda(
                    "minimax.qkv_input_quantize",
                    _quantize_qkv_rows,
                    attention.qkv_proj,
                    x[start:stop],
                )
                projected = _profile_cuda(
                    "minimax.qkv_projection_tile",
                    convrot_w8_output_slice,
                    qactivation,
                    activation_scale,
                    qweight,
                    weight_scale,
                    bias,
                    0,
                    3 * inner,
                    x.dtype,
                )
                del qactivation, activation_scale
            query, key, value_tile = projected.split(inner, dim=-1)
            tile_rows = stop - start
            query = query.view(tile_rows, heads, head_dim).transpose(0, 1).unsqueeze(0)
            key = key.view(tile_rows, heads, head_dim).transpose(0, 1).unsqueeze(0)
            value_tile = (
                value_tile.view(tile_rows, heads, head_dim)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            tile_transform = _slice_qk_transform(
                transform, start, stop, sequence
            )
            q_scale_start = (start // 64) * 4
            k_scale_start = start // 64
            tile_q_scale = q_scale[
                :,
                :,
                q_scale_start : q_scale_start + ((tile_rows + 63) // 64) * 4,
            ]
            tile_k_scale = k_scale[
                :,
                :,
                k_scale_start : k_scale_start + (tile_rows + 63) // 64,
            ]
            tile_qk = _profile_cuda(
                "attention.qk_norm_rope_quant",
                prequantize_turing_qk,
                query,
                key,
                tile_transform,
                kernel="sol",
                k_anchor=k_anchor,
                qk_output=(
                    q_int8[:, :, start:stop],
                    tile_q_scale,
                    k_int8[:, :, start:stop],
                    tile_k_scale,
                ),
            )
            _profile_cuda(
                "attention.value_prepare",
                value[:, :, start:stop].copy_,
                value_tile,
            )
            qk_type = type(tile_qk)
            route_original_basis = bool(tile_qk.route_original_basis)
            del projected, query, key, value_tile, tile_qk
    finally:
        comfy.ops.uncast_bias_weight(
            attention.qkv_proj, weight, bias, offload_stream
        )

    if qk_type is None:
        raise RuntimeError("streamed H3 QKV projection produced no tiles")
    qk = qk_type(
        query_int8=q_int8,
        query_scale=q_scale,
        key_int8=k_int8,
        key_scale=k_scale,
        tensor_layout="HND",
        input_dtype=x.dtype,
        original_head_dim=head_dim,
        route_original_basis=route_original_basis,
    )
    return qk, value


def _make_attention_forward(
    attention,
    attention_container,
    original=None,
    base_model=None,
):
    original = OriginalMethod.capture(
        attention.forward if original is None else original,
        attention,
    )
    base_model = _weak_model_reference(base_model)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        executor = prepared_attention_executor(transformer_options)
        if (
            x.ndim != 2
            or x.shape[0] < 64
            or x.dtype not in (torch.float16, torch.bfloat16)
            or self.head_dim not in (64, 128)
            or (
                torch.is_grad_enabled()
                and (
                    x.requires_grad
                    or any(parameter.requires_grad for parameter in self.parameters())
                )
            )
        ):
            return original(
                self,
                x,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )

        transform = _qk_transform(self, x, rope_freqs)
        virtual_kv = (
            transformer_options.get("turing_utils_attention_strategy")
            == "h3_virtual_kv"
        )
        logical_key_rows = (
            _virtual_kv_logical_rows(transformer_options, int(x.shape[0]))
            if virtual_kv
            else None
        )
        virtual_kv_mode = getattr(
            executor, "turing_utils_h3_virtual_kv_mode", None
        )
        virtual_kv_capability = getattr(
            executor,
            "turing_utils_h3_virtual_kv_mapped_capability",
            "mapped_sparse_kv" if virtual_kv_mode == "residual" else "mapped_kv",
        )
        virtual_kv_numeric_backend = getattr(
            executor,
            "turing_utils_h3_virtual_kv_numeric_backend",
            "w8a8",
        )
        mapped_virtual_kv = bool(
            virtual_kv
            and logical_key_rows is not None
            and virtual_kv_mode in {"fast", "residual"}
            and getattr(
                executor,
                "turing_utils_h3_virtual_kv_mapped_available",
                False,
            )
            and kernel_capabilities().supports(virtual_kv_capability).supported
        )
        virtual_residual_subblocks = (
            2 if mapped_virtual_kv and virtual_kv_mode == "residual" else 0
        )
        streamed_executor = (
            getattr(executor, _STREAMED_QKV_EXECUTOR_ATTR, None)
            if executor is not None
            else None
        )
        # Row streaming relies on the v0.30 direct-output ABI even when
        # K-anchor stabilization is disabled. Older kernels keep the complete
        # projection path instead of failing after committing partial output.
        stream_abi_available = reusable_k_anchor_available()
        qkv_is_w8 = convrot_weight_kind(self.qkv_proj.weight) == "w8a8"
        runtime_plan = _runtime_activation_plan(base_model)
        head_decision = None
        # Exact mapped virtual K/V can use whole-head sharding because every
        # shard retains physical K/V until the fused logical-RoPE preprocessor.
        # The compact streamed path still prequantizes physical K too early.
        if qkv_is_w8 and (not virtual_kv or mapped_virtual_kv):
            head_decision = decide_attention_heads(
                x,
                heads=int(self.heads),
                head_dim=int(self.head_dim),
                compact_qk=bool(
                    callable(streamed_executor)
                    and stream_abi_available
                    and not virtual_kv
                ),
                quantized_input=True,
                quantized_value=bool(
                    callable(streamed_executor)
                    or (mapped_virtual_kv and virtual_kv_numeric_backend == "w8a8")
                ),
                runtime_plan=runtime_plan,
                base_model=base_model,
                logical_key_rows=logical_key_rows,
                residual_subblocks=virtual_residual_subblocks,
            )
            if head_decision.sharded:
                profile_shape = (1, self.heads, x.shape[0], self.head_dim)
                CUDA_PHASE_PROFILER.begin_operation(
                    "attention",
                    profile_shape,
                    adapter="minimax",
                    path="head_sharded",
                    head_group=head_decision.head_group,
                )
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="attention_heads",
                        estimated_peak_bytes=head_decision.estimated_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                output = _head_sharded_attention(
                    self,
                    x,
                    rope_freqs,
                    transform,
                    transformer_options,
                    attention_container,
                    executor,
                    head_decision.head_group,
                    head_decision.cache_quantized_input,
                )
                if output is not None:
                    output = _profile_cuda(
                        "minimax.out_projection", self.out_proj, output
                    )
                    CUDA_PHASE_PROFILER.complete_operation(
                        "attention", profile_shape
                    )
                    return output
                CUDA_PHASE_PROFILER.cancel_operation()

            if base_model is not None:
                # Automatic tiers already fit immediately usable memory, so
                # this is normally a no-op. It remains useful for an explicit
                # throughput override, and may evict only inactive-model
                # VBARs; the current diffusion weights are never targeted.
                ensure_dynamic_vram_headroom(
                    base_model,
                    x.device,
                    rows=int(x.shape[0]),
                    operation="attention_execute",
                    estimated_peak_bytes=head_decision.estimated_peak_bytes,
                    runtime_plan=runtime_plan,
                )

        if (
            virtual_kv
            and logical_key_rows is not None
            and head_decision is None
            and base_model is not None
        ):
            virtual_peak = estimate_attention_lifecycle_peak(
                rows=int(x.shape[0]),
                heads=int(self.heads),
                head_dim=int(self.head_dim),
                hidden_size=int(x.shape[-1]),
                element_size=int(x.element_size()),
                head_group=int(self.heads),
                compact_qk=False,
                cache_quantized_input=False,
                quantized_value=bool(
                    mapped_virtual_kv and virtual_kv_numeric_backend == "w8a8"
                ),
                logical_key_rows=logical_key_rows,
                residual_subblocks=virtual_residual_subblocks,
            )
            if not mapped_virtual_kv:
                # Exact fallback materializes logical BF16 K and V before the
                # selected dense backend preprocesses them.
                virtual_peak += (
                    2
                    * logical_key_rows
                    * int(self.heads)
                    * int(self.head_dim)
                    * int(x.element_size())
                )
            ensure_dynamic_vram_headroom(
                base_model,
                x.device,
                rows=int(x.shape[0]),
                operation="virtual_kv",
                estimated_peak_bytes=virtual_peak,
                runtime_plan=runtime_plan,
            )

        if executor is None:
            _warn_attention_fallback(
                transformer_options,
                path="prepared",
                rows=int(x.shape[0]),
                reason="prepared executor is unavailable",
            )
            return original(
                self,
                x,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
        decision = None
        if callable(streamed_executor) and stream_abi_available:
            decision = decide_activation_chunks(
                x,
                operation="qkv",
                hidden_size=int(x.shape[-1]),
                expanded_size=int(self.heads * self.head_dim),
                heads=int(self.heads),
                runtime_plan=runtime_plan,
                base_model=base_model,
            )
        profile_shape = (1, self.heads, x.shape[0], self.head_dim)
        CUDA_PHASE_PROFILER.begin_operation(
            "attention",
            profile_shape,
            adapter="minimax",
            path="row_streamed" if decision is not None and decision.streamed else "full",
            qkv_rows=(decision.chunk_rows if decision is not None else 0),
        )
        if decision is not None:
            if decision.streamed:
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="qkv",
                        estimated_peak_bytes=decision.streamed_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                qk, value = _stream_qkv_projection(
                    self,
                    x,
                    transform,
                    decision.chunk_rows,
                )
                outcome = streamed_executor(
                    qk,
                    value,
                    heads=self.heads,
                    qk_transform=transform,
                    transformer_options=transformer_options,
                )
                del qk, value
                if not outcome.supported:
                    CUDA_PHASE_PROFILER.cancel_operation()
                    raise RuntimeError(
                        "streamed H3 QKV executor rejected a committed projection: "
                        f"{outcome.reason}"
                    )
                output = outcome.output.squeeze(0)
                output = _profile_cuda(
                    "minimax.out_projection", self.out_proj, output
                )
                CUDA_PHASE_PROFILER.complete_operation(
                    "attention", profile_shape
                )
                return output

        qkv = _profile_cuda("minimax.qkv_projection", self.qkv_proj, x)
        sequence = x.shape[0]
        inner = self.heads * self.head_dim
        query, key, value = qkv.split(inner, dim=-1)
        query = query.view(sequence, self.heads, self.head_dim)
        key = key.view(sequence, self.heads, self.head_dim)
        value = value.view(sequence, self.heads, self.head_dim)
        del qkv
        outcome = execute_projected_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            heads=self.heads,
            qk_transform=transform,
            transformer_options=transformer_options,
            container_factory=attention_container,
        )
        if not outcome.supported:
            del query, key, value
            CUDA_PHASE_PROFILER.cancel_operation()
            _warn_attention_fallback(
                transformer_options,
                path="projected",
                rows=sequence,
                reason=outcome.reason,
            )
            return original(
                self,
                x,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
        output = outcome.output.squeeze(0)
        output = _profile_cuda("minimax.out_projection", self.out_proj, output)
        CUDA_PHASE_PROFILER.complete_operation("attention", profile_shape)
        return output

    setattr(forward, _PREPARED_ATTENTION_FORWARD_ATTR, True)
    return weak_method(forward, attention)


def _has_prepared_attention_forward(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _PREPARED_ATTENTION_FORWARD_ATTR, False))


def install_minimax_attention_sites(model, device: torch.device) -> AttentionSiteStatus:
    """Install only the model-side H3 handoff to generic attention backends."""
    try:
        from comfy.ldm.minimax.model import Attention, DiTBlock
        from comfy.ldm.modules.attention import AttentionTensorContainer
    except ImportError:
        return AttentionSiteStatus(None, 0, "minimax_unavailable")
    root = getattr(model, "model", model)
    if not callable(getattr(root, "named_modules", None)):
        return AttentionSiteStatus(None, 0, "not_minimax_h3")
    candidates = [
        (name, block)
        for name, block in root.named_modules()
        if name and isinstance(block, DiTBlock)
    ]
    if not candidates:
        return AttentionSiteStatus(None, 0, "not_minimax_h3")
    if not is_supported_attention_device(device):
        return AttentionSiteStatus("minimax_h3", 0, "not_supported_tensor_core")
    if not callable(getattr(model, "add_object_patch", None)):
        return AttentionSiteStatus("minimax_h3", 0, "model_patcher_api_unavailable")
    if not fused_qk_preprocessing_available():
        return AttentionSiteStatus("minimax_h3", 0, "fused_qk_unavailable")
    if not _compatible_attention_forward(Attention):
        return AttentionSiteStatus("minimax_h3", 0, "attention_contract_changed")

    object_patches = getattr(model, "object_patches", {})
    installed = 0
    for name, block in candidates:
        if type(block.attn) is not Attention:
            continue
        key = f"{name}.attn.forward"
        current = object_patches.get(key, block.attn.forward)
        if _has_prepared_attention_forward(current):
            continue
        model.add_object_patch(
            key,
            _make_attention_forward(
                block.attn,
                AttentionTensorContainer,
                current,
                root,
            ),
        )
        installed += 1
    return AttentionSiteStatus("minimax_h3", installed, None)


def _block_fusion_blocker(
    x: torch.Tensor,
    t_emb: torch.Tensor,
    device_index: int,
) -> str | None:
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES or x.ndim != 2:
        return f"input={x.device.type}/{x.dtype}/ndim{x.ndim}"
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if index != device_index:
        return f"device_index={index},expected={device_index}"
    if torch.is_grad_enabled() and (x.requires_grad or t_emb.requires_grad):
        return "grad_enabled"
    return None


def _stream_mlp(mlp: torch.nn.Module, x: torch.Tensor, chunk_rows: int):
    """Evaluate an H3 SwiGLU MLP in row tiles with one weight cast per layer."""
    import comfy.ops

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = _profile_cuda(
        "minimax.mlp.fc1_weight_wait",
        comfy.ops.cast_bias_weight,
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=False,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = _profile_cuda(
            "minimax.mlp.fc2_weight_wait",
            comfy.ops.cast_bias_weight,
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        output_features = int(getattr(mlp.fc2, "out_features", x.shape[-1]))
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for start in range(0, x.shape[0], chunk_rows):
            stop = min(start + chunk_rows, x.shape[0])
            expanded = _profile_cuda(
                "minimax.mlp.fc1_tile",
                _linear_with_cast_weight,
                mlp.fc1,
                x[start:stop],
                fc1_weight,
                fc1_bias,
            )
            if _direct_int8_output_available():
                tile = _profile_cuda(
                    "minimax.mlp.swiglu_fc2_tile",
                    convrot_linear_input_act_from_weight,
                    fc2_weight,
                    fc2_bias,
                    expanded,
                    "swiglu",
                    output[start:stop],
                )
            else:
                tile = _profile_cuda(
                    "minimax.mlp.swiglu_fc2_tile",
                    convrot_linear_input_act_from_weight,
                    fc2_weight,
                    fc2_bias,
                    expanded,
                    "swiglu",
                )
                _profile_cuda(
                    "minimax.mlp.output_store", output[start:stop].copy_, tile
                )
            del expanded, tile
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(
                mlp.fc2, fc2_weight, fc2_bias, fc2_stream
            )
        comfy.ops.uncast_bias_weight(
            mlp.fc1, fc1_weight, fc1_bias, fc1_stream
        )


def _ffn_expanded_shard(
    mlp,
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    expanded_size: int,
    start: int,
    stop: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Project one aligned gate/up interval without a full fc1 output."""
    width = stop - start
    expanded = torch.empty(
        (qactivation.shape[0], 2 * width),
        dtype=output_dtype,
        device=qactivation.device,
    )
    direct = _direct_int8_output_available()
    gate = convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        start,
        stop,
        output_dtype,
        output=expanded[:, :width] if direct else None,
    )
    if not direct:
        expanded[:, :width].copy_(gate)
    del gate
    up = convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        expanded_size + start,
        expanded_size + stop,
        output_dtype,
        output=expanded[:, width:] if direct else None,
    )
    if not direct:
        expanded[:, width:].copy_(up)
    del up
    return expanded


def _stream_mlp_channels(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    *,
    chunk_rows: int,
    chunk_channels: int,
):
    """Exact two-pass H3 FFN evaluation over ConvRot-aligned channels.

    The first pass finds the same whole-row activation scale as the unsharded
    fused SwiGLU quantizer.  The second pass recomputes each aligned interval,
    uses that common scale, and writes directly into the final compact A8 row.
    The unchanged fused fc2 consumes that row once, so no shard boundary enters
    the contraction or its BF16 epilogue.
    """
    import comfy.ops

    if chunk_channels <= 0 or chunk_channels % 256:
        return None
    if getattr(mlp.fc2, "pre_quant_scale", None) is not None:
        # This is not used by MiniMax H3 today.  Falling back avoids changing
        # the ordering of a future SmoothQuant-style pre-scale.
        return None

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = _profile_cuda(
        "minimax.mlp.fc1_weight_wait",
        comfy.ops.cast_bias_weight,
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = _profile_cuda(
            "minimax.mlp.fc2_weight_wait",
            comfy.ops.cast_bias_weight,
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        fc1_plain = convrot_w8_plain_tensors(fc1_weight)
        fc2_plain = convrot_w8_plain_tensors(fc2_weight)
        if fc1_plain is None or fc2_plain is None or fc2_bias is not None:
            return None
        fc1_qweight, fc1_weight_scale = fc1_plain
        fc2_qweight, fc2_weight_scale = fc2_plain
        expanded_size = int(getattr(mlp.fc2, "in_features", 0))
        if (
            expanded_size <= 0
            or expanded_size % 256
            or fc1_qweight.shape[0] != 2 * expanded_size
            or fc2_qweight.shape[1] != expanded_size
        ):
            return None

        output_features = int(fc2_qweight.shape[0])
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for row_start in range(0, x.shape[0], chunk_rows):
            row_stop = min(row_start + chunk_rows, x.shape[0])
            qactivation, input_scale = _profile_cuda(
                "minimax.mlp.input_quantize",
                _quantize_qkv_rows,
                mlp.fc1,
                x[row_start:row_stop],
            )

            whole_row_scale = None
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                expanded = _profile_cuda(
                    "minimax.mlp.channel_scale_projection",
                    _ffn_expanded_shard,
                    mlp,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size,
                    start,
                    stop,
                    x.dtype,
                )
                local_quantized, local_scale = (
                    quantize_convrot_swiglu_activation(expanded, 256)
                )
                del expanded, local_quantized
                if whole_row_scale is None:
                    whole_row_scale = local_scale
                else:
                    torch.maximum(
                        whole_row_scale, local_scale, out=whole_row_scale
                    )
                    del local_scale

            activated = torch.empty(
                (row_stop - row_start, expanded_size),
                dtype=torch.int8,
                device=x.device,
            )
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                expanded = _profile_cuda(
                    "minimax.mlp.channel_output_projection",
                    _ffn_expanded_shard,
                    mlp,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size,
                    start,
                    stop,
                    x.dtype,
                )
                direct = _direct_int8_output_available()
                quantized = quantize_convrot_swiglu_with_scale(
                    expanded,
                    whole_row_scale,
                    256,
                    output=activated[:, start:stop] if direct else None,
                )
                del expanded
                if not direct:
                    activated[:, start:stop].copy_(quantized)
                del quantized

            if _direct_int8_output_available():
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                    output=output[row_start:row_stop],
                )
            else:
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                )
                _profile_cuda(
                    "minimax.mlp.output_store",
                    output[row_start:row_stop].copy_,
                    tile,
                )
            del (
                activated,
                input_scale,
                qactivation,
                tile,
                whole_row_scale,
            )
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(
                mlp.fc2, fc2_weight, fc2_bias, fc2_stream
            )
        comfy.ops.uncast_bias_weight(
            mlp.fc1, fc1_weight, fc1_bias, fc1_stream
        )


def _stream_mlp_half_width(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    *,
    chunk_rows: int,
    chunk_channels: int,
):
    """Single-pass exact FC1 with one half-width in-place ConvRot buffer.

    Gate channels are projected once into the final rotated workspace. Up
    channels are projected in aligned intervals, consumed immediately by the
    identical FP32 SwiGLU/FHT sequence, and released. The final reduction uses
    all channel partials, so row scale and INT8 values match the full-width
    fused quantizer without a second FC1 pass.
    """
    import comfy.ops

    if chunk_channels <= 0 or chunk_channels % 256:
        return None
    if getattr(mlp.fc2, "pre_quant_scale", None) is not None:
        return None

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = _profile_cuda(
        "minimax.mlp.fc1_weight_wait",
        comfy.ops.cast_bias_weight,
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = _profile_cuda(
            "minimax.mlp.fc2_weight_wait",
            comfy.ops.cast_bias_weight,
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        fc1_plain = convrot_w8_plain_tensors(fc1_weight)
        fc2_plain = convrot_w8_plain_tensors(fc2_weight)
        if fc1_plain is None or fc2_plain is None or fc2_bias is not None:
            return None
        fc1_qweight, fc1_weight_scale = fc1_plain
        fc2_qweight, fc2_weight_scale = fc2_plain
        expanded_size = int(getattr(mlp.fc2, "in_features", 0))
        if (
            expanded_size <= 0
            or expanded_size % 256
            or fc1_qweight.shape[0] != 2 * expanded_size
            or fc2_qweight.shape[1] != expanded_size
        ):
            return None

        output_features = int(fc2_qweight.shape[0])
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for row_start in range(0, x.shape[0], chunk_rows):
            row_stop = min(row_start + chunk_rows, x.shape[0])
            qactivation, input_scale = _profile_cuda(
                "minimax.mlp.input_quantize",
                _quantize_qkv_rows,
                mlp.fc1,
                x[row_start:row_stop],
            )
            rotated_gate = _profile_cuda(
                "minimax.mlp.fc1_gate",
                convrot_w8_output_slice,
                qactivation,
                input_scale,
                fc1_qweight,
                fc1_weight_scale,
                fc1_bias,
                0,
                expanded_size,
                x.dtype,
            )
            partial_absmax = torch.empty(
                (row_stop - row_start, expanded_size // 256),
                dtype=torch.float32,
                device=x.device,
            )
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                up = _profile_cuda(
                    "minimax.mlp.fc1_up_tile",
                    convrot_w8_output_slice,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size + start,
                    expanded_size + stop,
                    x.dtype,
                )
                _profile_cuda(
                    "minimax.mlp.swiglu_rotate_tile",
                    rotate_convrot_swiglu_shard_inplace,
                    rotated_gate,
                    up,
                    partial_absmax,
                    start,
                )
                del up

            activated, whole_row_scale = _profile_cuda(
                "minimax.mlp.activation_quantize",
                quantize_convrot_from_partials,
                rotated_gate,
                partial_absmax,
            )
            if _direct_int8_output_available():
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                    output=output[row_start:row_stop],
                )
            else:
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                )
                _profile_cuda(
                    "minimax.mlp.output_store",
                    output[row_start:row_stop].copy_,
                    tile,
                )
            del (
                activated,
                input_scale,
                partial_absmax,
                qactivation,
                rotated_gate,
                tile,
                whole_row_scale,
            )
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(
                mlp.fc2, fc2_weight, fc2_bias, fc2_stream
            )
        comfy.ops.uncast_bias_weight(
            mlp.fc1, fc1_weight, fc1_bias, fc1_stream
        )


def _make_mlp_forward(
    mlp: torch.nn.Module,
    audit: _RuntimeDispatchAudit,
    base_model=None,
):
    original = OriginalMethod.capture(mlp.forward, mlp)
    base_model = _weak_model_reference(base_model)

    def forward(self, x: torch.Tensor):
        blocker = None
        if x.dtype != torch.bfloat16:
            blocker = f"dtype={x.dtype}"
        elif not is_turing_convrot_linear(self.fc2):
            blocker = "fc2_not_turing_convrot"
        elif not is_supported_attention_device(x.device):
            blocker = f"device={x.device}"
        audit.record("mlp", blocker is None, x, blocker)
        if blocker is not None:
            return original(self, x)
        expanded_size = int(getattr(self.fc2, "in_features", 0))
        runtime_plan = _runtime_activation_plan(base_model)
        decision = decide_activation_chunks(
            x,
            operation="mlp",
            hidden_size=int(x.shape[-1]),
            expanded_size=expanded_size,
            runtime_plan=runtime_plan,
            base_model=base_model,
        )
        channel_decision = None
        if convrot_swiglu_channel_sharding_available():
            half_width = convrot_swiglu_half_width_available()
            channel_decision = decide_ffn_channels(
                x,
                expanded_size=expanded_size,
                chunk_rows=decision.chunk_rows,
                half_width=half_width,
                runtime_plan=runtime_plan,
                base_model=base_model,
            )
            if channel_decision.sharded:
                profile_shape = (int(x.shape[0]), int(x.shape[-1]))
                CUDA_PHASE_PROFILER.begin_operation(
                    "mlp",
                    profile_shape,
                    adapter="minimax",
                    path=(
                        "half_width_channel_sharded"
                        if half_width
                        else "channel_sharded_two_pass"
                    ),
                    row_chunk=channel_decision.chunk_rows,
                    channel_chunk=channel_decision.chunk_channels,
                )
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="ffn_channels",
                        estimated_peak_bytes=channel_decision.estimated_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                stream_function = (
                    _stream_mlp_half_width
                    if half_width
                    else _stream_mlp_channels
                )
                output = stream_function(
                    self, x,
                    chunk_rows=channel_decision.chunk_rows,
                    chunk_channels=channel_decision.chunk_channels,
                )
                if output is not None:
                    CUDA_PHASE_PROFILER.complete_operation(
                        "mlp", profile_shape
                    )
                    return output
                CUDA_PHASE_PROFILER.cancel_operation()
        profile_shape = (int(x.shape[0]), int(x.shape[-1]))
        CUDA_PHASE_PROFILER.begin_operation(
            "mlp",
            profile_shape,
            adapter="minimax",
            path="row_streamed" if decision.streamed else "full",
            row_chunk=decision.chunk_rows,
            channel_chunk=0,
        )
        if base_model is not None:
            ensure_dynamic_vram_headroom(
                base_model,
                x.device,
                rows=int(x.shape[0]),
                operation="mlp",
                estimated_peak_bytes=decision.streamed_peak_bytes,
                runtime_plan=runtime_plan,
            )
        if decision.streamed:
            output = _stream_mlp(self, x, decision.chunk_rows)
        else:
            expanded = _profile_cuda("minimax.mlp.fc1", self.fc1, x)
            output = _profile_cuda(
                "minimax.mlp.swiglu_fc2",
                fused_convrot_linear_input_act,
                self.fc2,
                expanded,
                "swiglu",
            )
            del expanded
        CUDA_PHASE_PROFILER.complete_operation("mlp", profile_shape)
        return output

    return weak_method(forward, mlp)


_minimax_temporal_topology = minimax_temporal_topology


def _make_block_forward(
    block: torch.nn.Module,
    device_index: int,
    mod_gate,
    audit: _RuntimeDispatchAudit,
    layer_index: int = 0,
    layer_count: int = 0,
    base_model=None,
    diffusion_model=None,
):
    original = OriginalMethod.capture(block.forward, block)
    if base_model is not None:
        base_model = weakref.proxy(base_model)
    if diffusion_model is not None:
        diffusion_model = weakref.proxy(diffusion_model)

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        publish_minimax_attention_layout(
            transformer_options,
            mod_segments,
            layer_index=layer_index,
            layer_count=layer_count,
            base_model=base_model,
            diffusion_model=diffusion_model,
        )
        blocker = _block_fusion_blocker(x, t_emb, device_index)
        audit.record("block", blocker is None, x, blocker)
        if blocker is not None:
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = segmented_rms_adaln(self.norm1, x, shift_msa, scale_msa, mod_segments)
        x, h = segmented_mod_gate_rms_adaln(
            self.norm2,
            x,
            gate_msa,
            self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
            shift_mlp,
            scale_mlp,
            mod_segments,
        )
        return segmented_mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    return mark_forward_as_minimax_layout_provider(weak_method(forward, block))

def apply_minimax_adapter(model, device: torch.device) -> int:
    """Install MiniMax-only forward substitutions through the ModelPatcher."""
    if not is_supported_attention_device(device):
        return 0
    if not hasattr(model, "add_object_patch"):
        raise RuntimeError("MiniMax CUDA integration requires a ComfyUI ModelPatcher")

    try:
        from comfy.ldm.minimax.model import DiTBlock, _mod_gate
    except ImportError:
        return 0
    if not _compatible_block_forward(DiTBlock):
        LOG.warning("MiniMax CUDA fusions are disabled because the DiTBlock forward contract changed")
        return 0

    root = getattr(model, "model", model)
    candidates = [
        (name, block)
        for name, block in root.named_modules()
        if name and isinstance(block, DiTBlock)
    ]
    if not candidates:
        return 0
    try:
        setattr(root, "_turing_utils_minimax_layer_count", len(candidates))
    except (AttributeError, TypeError):
        pass

    layout_status = ensure_minimax_attention_layout_provider(model)
    if layout_status.installed:
        transformer_options = model.model_options.setdefault(
            "transformer_options", {}
        )
        transformer_options[ATTENTION_LAYOUT_REQUIREMENT_KEY] = (
            layout_status.model_kind
        )
    if layout_status.required and not layout_status.installed:
        LOG.warning(
            "MiniMax H3 attention layout provider could not be installed: %s",
            layout_status.reason,
        )

    diffusion_model = getattr(root, "diffusion_model", None)
    if diffusion_model is not None:
        _install_memory_planning(model, root, diffusion_model)

    eligible_fc2 = _audit_fc2([block for _, block in candidates])
    index = device.index if device.index is not None else torch.cuda.current_device()
    block_fusions = 0
    mlp_fusions = 0
    attention_fusions = install_minimax_attention_sites(model, device).installed
    try:
        kernel_package = load_kernel_package()
        segmented_block_ops = (
            getattr(kernel_package, "turing_segmented_rms_adaln"),
            getattr(kernel_package, "turing_segmented_mod_gate"),
            getattr(kernel_package, "turing_segmented_mod_gate_rms_adaln"),
        )
    except (ImportError, OSError, AttributeError):
        segmented_block_ops = ()

    block_ops_available = bool(segmented_block_ops) and all(
        callable(op) for op in segmented_block_ops
    )
    expected_blocks = len(candidates) if block_ops_available else 0
    audit = _RuntimeDispatchAudit(expected_blocks, eligible_fc2)

    for layer_index, (name, block) in enumerate(candidates):
        if hasattr(block.mlp, "fc2") and is_turing_convrot_linear(block.mlp.fc2):
            model.add_object_patch(
                f"{name}.mlp.forward",
                _make_mlp_forward(block.mlp, audit, root),
            )
            mlp_fusions += 1
        if block_ops_available:
            model.add_object_patch(
                f"{name}.forward",
                _make_block_forward(
                    block,
                    index,
                    _mod_gate,
                    audit,
                    layer_index,
                    len(candidates),
                    root,
                    diffusion_model,
                ),
            )
            block_fusions += 1

    if block_fusions:
        LOG.info("Enabled MiniMax segmented RMSNorm+AdaLN on %d CUDA blocks", block_fusions)
    if mlp_fusions:
        LOG.info("Enabled MiniMax fused/streamed ConvRot SwiGLU on %d MLP layers", mlp_fusions)
    if attention_fusions:
        LOG.info(
            "Enabled MiniMax fused Q/K RMSNorm+RoPE+INT8 preprocessing on %d attention layers",
            attention_fusions,
        )
    if eligible_fc2 and mlp_fusions != eligible_fc2:
        raise RuntimeError("MiniMax fused fc2 adapter did not patch every eligible layer")
    return max(block_fusions, mlp_fusions, attention_fusions)
