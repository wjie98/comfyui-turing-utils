"""H3 QKV projection, head streaming and prepared-attention sites."""

from __future__ import annotations

import dataclasses
import inspect
import torch
from ..methods import OriginalMethod, weak_method
from ...attention.integration import AttentionSiteStatus, execute_projected_attention
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
from ...profiling import CUDA_PHASE_PROFILER
from .activation_policy import decide_activation_chunks, decide_attention_heads
from .memory_state import ensure_dynamic_vram_headroom
from .veda.engine import workspace_per_head as veda_workspace_per_head
from ...quantization.fusions import (
    convrot_w8_output_slice,
    convrot_w8_plain_tensors,
    convrot_weight_kind,
)
from .execution import (
    LOG,
    quantize_linear_rows,
    _profile_cuda,
    _runtime_activation_plan,
    _weak_model_reference,
    _linear_with_cast_weight,
)


_ATTENTION_FORWARD_PARAMETERS = (
    "x",
    "rope_freqs",
    "transformer_options",
)


_PREPARED_ATTENTION_FORWARD_ATTR = "_turing_utils_minimax_prepared_attention"


_STREAMED_QKV_EXECUTOR_ATTR = "turing_utils_streamed_qkv_executor"


_ATTENTION_FALLBACK_WARNINGS_KEY = object()


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
        query_norm=RMSNormSpec(query_norm, float(attention.q_norm.eps), "head"),
        key_norm=RMSNormSpec(key_norm, float(attention.k_norm.eps), "head"),
        rotary=RotaryEmbeddingSpec(
            rope_freqs,
            rot_dim,
            "split_half" if rope_freqs is not None else "none",
        ),
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
    sample_indices = torch.tensor(sample_rows, dtype=torch.long, device=x.device)
    sampled = x.index_select(0, sample_indices)
    projected = _linear_with_cast_weight(attention.qkv_proj, sampled, weight, bias)
    key = projected[:, inner : 2 * inner]
    key = key.view(9, heads, head_dim).transpose(0, 1).unsqueeze(0)
    sample_transform = _sample_qk_transform(transform, sample_indices, sequence)
    anchor_indices, anchor_values = precompute_turing_k_anchor(key, sample_transform)
    lookup = sample_indices.to(dtype=torch.int32)
    selected = anchor_indices.clamp_min(0).to(dtype=torch.long)
    anchor_indices = torch.where(
        anchor_indices >= 0,
        lookup[selected],
        anchor_indices,
    )
    return anchor_indices, anchor_values


def _cache_quantized_qkv_input(linear, x: torch.Tensor, chunk_rows: int):
    qactivation = torch.empty_like(x, dtype=torch.int8)
    activation_scale = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        tile, scale = quantize_linear_rows(linear, x[start:stop])
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
    sample_indices = torch.tensor(sample_rows, dtype=torch.long, device=x.device)
    if quantized_input is None:
        qactivation, activation_scale = quantize_linear_rows(
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
    sample_transform = _sample_qk_transform(transform, sample_indices, sequence)
    anchor_indices, anchor_values = precompute_turing_k_anchor(key, sample_transform)
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
    value = torch.empty((1, group, sequence, head_dim), dtype=x.dtype, device=x.device)
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
            qactivation, activation_scale = quantize_linear_rows(
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
        tile_transform = _slice_qk_transform(transform, start, stop, sequence)
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
        torch.empty((sequence, group, head_dim), dtype=x.dtype, device=x.device)
        for _ in range(3)
    ]
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        if quantized_input is None:
            qactivation, activation_scale = quantize_linear_rows(
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


def _veda_projected_head_group(
    attention,
    x,
    transform,
    qweight,
    weight_scale,
    bias,
    head_start,
    head_stop,
    quantized_input,
    options,
):
    """Project tile-ordered real rows; never materialize full floating Q/K/V."""
    from .veda.engine import attend
    from ...attention.execution import _attention_layer_metadata
    import comfy.model_prefetch

    sequence, dim = x.shape[0], int(attention.head_dim)
    group = head_stop - head_start
    selected_group = None
    selected_weights = None

    def release():
        nonlocal selected_group, selected_weights
        selected_group = selected_weights = None

    def project(indices, local_heads):
        nonlocal selected_group, selected_weights
        if quantized_input is None:
            qa, scale = quantize_linear_rows(
                attention.qkv_proj, x.index_select(0, indices)
            )
        else:
            qa, scale = (t.index_select(0, indices) for t in quantized_input)
        contiguous = local_heads == list(
            range(local_heads[0], local_heads[0] + len(local_heads))
        )
        if not contiguous and selected_group != tuple(local_heads):
            # One bounded INT8 row pack per tile-shape group, reused across
            # row chunks. Per-head tiny GEMMs are much slower than this copy.
            inner = int(attention.heads) * dim
            slices = [
                (
                    c * inner + (head_start + h) * dim,
                    c * inner + (head_start + h + 1) * dim,
                )
                for c in range(3)
                for h in local_heads
            ]
            selected_weights = (
                torch.cat([qweight[start:stop] for start, stop in slices], dim=0),
                weight_scale
                if weight_scale.numel() == 1
                else torch.cat(
                    [weight_scale.reshape(-1)[start:stop] for start, stop in slices]
                ),
                None
                if bias is None
                else torch.cat([bias[start:stop] for start, stop in slices]),
            )
            selected_group = tuple(local_heads)
        weights = (qweight, weight_scale, bias) if contiguous else selected_weights
        start = head_start + local_heads[0] if contiguous else 0
        total_heads = int(attention.heads) if contiguous else len(local_heads)
        projected = []
        for component in range(3):
            projected.append(
                _w8_qkv_component(
                    qa,
                    scale,
                    *weights,
                    component=component,
                    head_start=start,
                    head_stop=start + len(local_heads),
                    heads=total_heads,
                    head_dim=dim,
                    output_dtype=x.dtype,
                ).view(indices.numel(), len(local_heads), dim)
            )
        selected_transform = _sample_qk_transform(transform, indices, sequence)
        query, key = _apply_minimax_qk_transform(
            attention, projected[0], projected[1], selected_transform.freqs
        )
        return query, key, projected[2]

    # Fixed callback scratch: input gather/quantization and GEMM intermediates.
    # Additional tile QKV/features/quantization are budgeted inside attend().
    chunk_tiles = int(options.get("turing_utils_veda_projection_chunk_tiles", 16))
    if chunk_tiles < 1:
        raise ValueError("Veda projection chunk must contain at least one tile")
    chunk_rows = min(sequence, chunk_tiles * 128)
    project.workspace_bytes = chunk_rows * (
        x.shape[1] * (x.element_size() + 3) + group * dim * 12 + 16
    )
    project.workspace_bytes += 3 * group * dim * (qweight.shape[1] + 8)
    project.release = release
    layer, _ = _attention_layer_metadata(options)
    meta = x.new_empty((1, 1, 1, dim)).expand(1, group, sequence, dim)
    with comfy.model_prefetch.pause_malloc_graph():
        result = attend(
            meta,
            meta,
            meta,
            config=options["turing_utils_veda"],
            packed_layout=options["minimax_h3_layout"],
            layer=layer,
            head_start=head_start,
            cache=options.get("turing_utils_veda_forward_cache"),
            projector=project,
            prepare_chunk_tiles=chunk_tiles,
        )
    return result.transpose(1, 2).flatten(2).squeeze(0)


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
        veda = transformer_options.get("turing_utils_attention_strategy") == "veda"
        compact = (
            callable(streamed_executor) and reusable_k_anchor_available() and not veda
        )
        for head_start in range(0, heads, head_group):
            head_stop = min(head_start + head_group, heads)
            group = head_stop - head_start
            group_options = transformer_options
            if veda:
                group_options = dict(
                    transformer_options, turing_utils_attention_head_start=head_start
                )
            veda_schedule = transformer_options.get("turing_utils_veda_schedule")
            if (
                veda
                and veda_schedule is not None
                and not veda_schedule.is_dense(transformer_options)
                and transformer_options.get("turing_utils_veda_auto_schedule", False)
            ):
                from .memory_state import runtime_memory
                from .veda.selection import choose_projected_chunk
                from ...attention.execution import _attention_layer_metadata

                layer, _ = _attention_layer_metadata(transformer_options)
                layout = transformer_options.get("minimax_h3_layout")
                if layer is not None and layout is not None:
                    workspace = (
                        veda_workspace_per_head(
                            transformer_options["turing_utils_veda"],
                            layout,
                            layer,
                            x.element_size(),
                            torch.cuda.get_device_capability(x.device),
                            transformer_options.get("turing_utils_veda_forward_cache"),
                            chunk_tiles=0,
                        )
                        * group
                    )
                    available, _, _ = runtime_memory(x.device)
                    tiles = choose_projected_chunk(
                        rows=sequence,
                        heads=group,
                        dim=head_dim,
                        hidden=x.shape[1],
                        element_size=x.element_size(),
                        available=available,
                        workspace=workspace,
                    )
                    if tiles:
                        group_options = dict(
                            group_options,
                            turing_utils_veda_projected_qkv=True,
                            turing_utils_veda_projection_chunk_tiles=tiles,
                        )
            if (
                veda
                and veda_schedule is not None
                and not veda_schedule.is_dense(transformer_options)
                and group_options.get("turing_utils_veda_projected_qkv", False)
            ):
                group_output = _veda_projected_head_group(
                    attention,
                    x,
                    transform,
                    qweight,
                    weight_scale,
                    bias,
                    head_start,
                    head_stop,
                    quantized_input,
                    group_options,
                )
            elif compact:
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
                    transformer_options=group_options,
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
        comfy.ops.uncast_bias_weight(attention.qkv_proj, weight, bias, offload_stream)


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
    value = torch.empty((1, heads, sequence, head_dim), dtype=x.dtype, device=x.device)

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
                    quantize_linear_rows,
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
                value_tile.view(tile_rows, heads, head_dim).transpose(0, 1).unsqueeze(0)
            )
            tile_transform = _slice_qk_transform(transform, start, stop, sequence)
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
        comfy.ops.uncast_bias_weight(attention.qkv_proj, weight, bias, offload_stream)

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
        streamed_executor = (
            getattr(executor, _STREAMED_QKV_EXECUTOR_ATTR, None)
            if executor is not None
            else None
        )
        # Row streaming relies on the v0.30 direct-output ABI even when
        # K-anchor stabilization is disabled. Older kernels keep the complete
        # projection path instead of failing after committing partial output.
        stream_abi_available = reusable_k_anchor_available()
        veda_extra_workspace = 0
        if transformer_options.get("turing_utils_attention_strategy") == "veda":
            # Veda TripPool needs floating post-RoPE Q/K, before the attention
            # rotation. Compact QKV would discard those values too early.
            streamed_executor = None
            from ...attention.execution import _attention_layer_metadata

            layer, _ = _attention_layer_metadata(transformer_options)
            packed_layout = transformer_options.get("minimax_h3_layout")
            if (
                layer is not None
                and packed_layout is not None
                and packed_layout.seq_len == x.shape[0]
            ):
                veda_extra_workspace = veda_workspace_per_head(
                    transformer_options["turing_utils_veda"],
                    packed_layout,
                    layer,
                    x.element_size(),
                    torch.cuda.get_device_capability(x.device),
                    transformer_options.get("turing_utils_veda_forward_cache"),
                )
        qkv_is_w8 = convrot_weight_kind(self.qkv_proj.weight) == "w8a8"
        runtime_plan = _runtime_activation_plan(base_model)
        head_decision = None
        if qkv_is_w8:
            head_decision = decide_attention_heads(
                x,
                heads=int(self.heads),
                head_dim=int(self.head_dim),
                compact_qk=bool(callable(streamed_executor) and stream_abi_available),
                quantized_input=True,
                quantized_value=callable(streamed_executor),
                runtime_plan=runtime_plan,
                base_model=base_model,
                extra_workspace_per_head=veda_extra_workspace,
            )
            if head_decision.sharded or transformer_options.get(
                "turing_utils_veda_projected_qkv", False
            ):
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
                    CUDA_PHASE_PROFILER.complete_operation("attention", profile_shape)
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
            path="row_streamed"
            if decision is not None and decision.streamed
            else "full",
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
                output = _profile_cuda("minimax.out_projection", self.out_proj, output)
                CUDA_PHASE_PROFILER.complete_operation("attention", profile_shape)
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
