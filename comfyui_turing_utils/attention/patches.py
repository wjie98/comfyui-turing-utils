"""Attention overrides and loader-independent ModelPatcher installation."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..profiling import CUDA_PHASE_PROFILER
from .layout import attention_semantic_layout
from .integration import ensure_prepared_attention_sites
from .orchestration import install_attention_strategy
from .protocol import (
    AttentionBackendCapabilities,
    AttentionExecutionOutcome,
    MAPPED_KV_EXECUTOR_ATTR,
    MAPPED_RESIDUAL_CAPABILITY_ATTR,
    MAPPED_RESIDUAL_EXECUTOR_ATTR,
    PreparedAttention,
)
from .runtime import (
    AttentionRuntimeConfig,
    attention_runtime_config,
    install_attention_runtime,
)
from .sparse_runtime import DenseAttentionFallback, SparseSchedule
from .sparse import (
    SolAttentionCall,
    inspect_sol_attention_call,
    inspect_sla_attention_call,
    prequantize_turing_sla_attention,
    prequantize_turing_sla_attention_from_qk,
    prequantize_turing_sol_attention,
    prequantize_turing_sol_attention_from_qk,
    turing_sol_attention_from_prequantized,
    turing_sol_sparse_attention,
    turing_sla_attention_from_prequantized,
    turing_sla_sparse_attention,
)
from .stable import (
    LOG,
    SPARSE_DENSE_PREFIX_LAYERS,
    SPARSE_DENSE_PREFIX_STEPS,
    SPARSE_DENSE_SUFFIX_LAYERS,
    SPARSE_DENSE_SUFFIX_STEPS,
    SPARSE_PREFIX_POLICY,
    SPARSE_REFERENCE_AUDIO,
    SPARSE_REFERENCE_IMAGE,
    SPARSE_REFERENCE_VIDEO,
    SPARSE_ROUTING_THRESHOLD,
    SPARSE_SKIPPED_RESIDUAL,
    SLA_DENSE_PREFIX_LAYERS,
    SLA_DENSE_PREFIX_STEPS,
    SLA_DENSE_SUFFIX_LAYERS,
    SLA_DENSE_SUFFIX_STEPS,
    SLA_SPARSITY_RATIO,
    AttentionBackend,
    _BACKENDS,
    _comfy_attention_function,
    _select_attention_backend,
    bundled_available,
    bundled_sparse_available,
    bundled_sla_available,
    bundled_w8a8_available,
    fused_qk_preprocessing_available,
    is_supported_attention_device,
    is_supported_turing_device,
    inspect_turing_attention_call,
    normalize_attention_backend,
    prequantize_turing_attention,
    prequantize_turing_attention_from_qk,
    prequantize_turing_qk,
    preflight_bundled,
    preflight_bundled_sparse,
    preflight_bundled_sla,
    preflight_bundled_w8a8,
    split_prequantization_available,
    turing_attention_from_prequantized,
    turing_sage_attention,
    turing_w8a8_attention,
)


_LOGGED_EXTERNAL_BACKEND_REJECTIONS: set[tuple[str, str]] = set()


def _bootstrap_attention_integrations() -> None:
    # Applying a public patch is an explicit integration action. Keep ordinary
    # package imports light while preserving direct-API behavior for callers
    # that do not enter through ComfyUI's plugin root.
    from ..bootstrap import bootstrap_builtin_integrations

    bootstrap_builtin_integrations()


def _profiled(phase: str, function: Callable, /, *args, **kwargs):
    if CUDA_PHASE_PROFILER.enabled:
        return CUDA_PHASE_PROFILER.call(phase, function, *args, **kwargs)
    return function(*args, **kwargs)


def _profile_route_stats(selected: torch.Tensor, possible: int) -> None:
    """Attach sparse-route density without synchronizing the sampler.

    ``selected`` stays on the device until the profiler report is emitted at
    the existing sampler-boundary fence.  Normal execution therefore pays no
    scalar readback or extra synchronization cost.
    """
    if not CUDA_PHASE_PROFILER.enabled:
        return
    selected_total = selected if selected.numel() == 1 else selected.sum()
    CUDA_PHASE_PROFILER.sample_tensor("route_selected_blocks", selected_total)
    CUDA_PHASE_PROFILER.sample("route_possible_blocks", int(possible))


def _attention_layer_metadata(transformer_options) -> tuple[int | None, int | None]:
    layout = attention_semantic_layout(transformer_options)
    if layout is not None:
        return layout.layer_index, layout.layer_count
    raw = (
        transformer_options.get("turing_utils_attention_layout")
        if isinstance(transformer_options, dict)
        else None
    )
    if not isinstance(raw, dict):
        return None, None
    layer_index = raw.get("layer_index")
    layer_count = raw.get("layer_count")
    return (
        layer_index
        if isinstance(layer_index, int) and not isinstance(layer_index, bool)
        else None,
        layer_count
        if isinstance(layer_count, int) and not isinstance(layer_count, bool)
        else None,
    )


def _prepared_call_mismatch(request: PreparedAttention, call) -> str | None:
    expected = (
        request.heads,
        request.kv_heads,
        request.head_dim,
        request.query_tokens,
        request.key_tokens,
        request.tensor_layout,
        request.skip_output_reshape,
    )
    actual = (
        call.heads,
        call.kv_heads,
        call.head_dim,
        call.query_tokens,
        call.key_tokens,
        call.tensor_layout,
        call.skip_output_reshape,
    )
    return (
        None
        if expected == actual
        else "prepared-attention metadata does not match Q/K/V"
    )


def _inspect_mapped_attention_call(
    capabilities: AttentionBackendCapabilities,
    request: PreparedAttention,
    key_source_indices: torch.Tensor,
):
    """Validate a physical Q/K/V request against its logical K/V sequence."""
    reason = capabilities.unsupported_reason(request)
    if reason is not None:
        return None, reason
    if (
        not torch.is_tensor(key_source_indices)
        or key_source_indices.ndim != 1
        or key_source_indices.dtype != torch.int32
    ):
        return None, "mapped K/V requires a one-dimensional int32 source map"
    query_view, key_view, value_view = request.peek_qkv()
    if key_source_indices.device != key_view.device:
        return None, "mapped K/V source map is on a different device"
    logical_tokens = int(key_source_indices.numel())
    if logical_tokens < 64:
        return None, "mapped K/V requires at least 64 logical tokens"
    logical_key = key_view[:, :, :1, :].expand(
        key_view.shape[0], key_view.shape[1], logical_tokens, key_view.shape[3]
    )
    logical_value = value_view[:, :, :1, :].expand_as(logical_key)
    call, reason = inspect_turing_attention_call(
        query_view,
        logical_key,
        logical_value,
        request.heads,
        mask=request.mask,
        skip_reshape=True,
        skip_output_reshape=request.skip_output_reshape,
        enable_gqa=request.heads != request.kv_heads,
        low_precision_attention=request.low_precision_attention,
        is_causal=request.is_causal,
        kernel="sol",
        require_long_sequence=True,
    )
    if reason is None and (
        call.heads != request.heads
        or call.kv_heads != request.kv_heads
        or call.head_dim != request.head_dim
        or call.query_tokens != request.query_tokens
        or call.key_tokens != logical_tokens
    ):
        reason = "mapped prepared-attention metadata does not match Q/K/V"
    del query_view, key_view, value_view, logical_key, logical_value
    return call, reason


def _make_mapped_residual_executor(
    capabilities: AttentionBackendCapabilities,
    *,
    use_w8a8: bool,
    kernel_capability: str,
) -> Callable:
    """Build the H3 mapped-Sol bridge for one inherited numeric backend."""

    def mapped_residual_executor(
        request: PreparedAttention,
        key_source_indices: torch.Tensor,
        *,
        exact_kv_ranges: tuple[tuple[int, int], ...],
        residual_subblocks: int = 2,
        routing_threshold: float = 1_000_000.0,
    ) -> AttentionExecutionOutcome:
        call, reason = _inspect_mapped_attention_call(
            capabilities, request, key_source_indices
        )
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        sol_call = SolAttentionCall(
            attention=call,
            effective_min_sequence=64,
            dense_query_ranges=(),
            exact_kv_ranges=tuple(exact_kv_ranges),
            residual_subblocks=int(residual_subblocks),
        )
        query, key, value = request.consume_qkv()
        qk = _profiled(
            "attention.qk_norm_rope_quant",
            prequantize_turing_qk,
            query,
            key,
            request.qk_transform,
            kernel="sol",
            key_source_indices=key_source_indices,
        )
        del query, key
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sol_attention_from_qk,
            qk,
            value,
            sol_call,
            routing_threshold=float(routing_threshold),
            scale=request.scale,
            use_w8a8=bool(use_w8a8),
            value_source_indices=key_source_indices,
        )
        del qk, value
        collect_stats = CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sol_attention_from_prequantized,
            quantized,
            return_stats=collect_stats,
        )
        if collect_stats:
            output, selected, possible_blocks = result
            _profile_route_stats(selected, possible_blocks)
        else:
            output = result
        return AttentionExecutionOutcome(output)

    setattr(
        mapped_residual_executor,
        MAPPED_RESIDUAL_CAPABILITY_ATTR,
        str(kernel_capability),
    )
    return mapped_residual_executor


def _make_dense_prepared_executor(kernel: str) -> Callable:
    capabilities = AttentionBackendCapabilities(
        supports_causal=kernel in {"sage", "w8a8"},
    )

    def executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = capabilities.unsupported_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        query_view, key_view, value_view = request.peek_qkv()
        call, reason = inspect_turing_attention_call(
            query_view,
            key_view,
            value_view,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            enable_gqa=request.heads != request.kv_heads,
            low_precision_attention=request.low_precision_attention,
            is_causal=request.is_causal,
            kernel=kernel,
            require_long_sequence=True,
        )
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        reason = _prepared_call_mismatch(request, call)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        del query_view, key_view, value_view
        query, key, value = request.consume_qkv()
        qk = _profiled(
            "attention.qk_norm_rope_quant",
            prequantize_turing_qk,
            query,
            key,
            request.qk_transform,
            kernel=kernel,
        )
        del query, key
        quantized = _profiled(
            "attention.value_prepare",
            prequantize_turing_attention_from_qk,
            qk,
            value,
            call,
            kernel=kernel,
            scale=request.scale,
            is_causal=request.is_causal,
        )
        del qk, value
        return AttentionExecutionOutcome(
            _profiled(
                "attention.execute",
                turing_attention_from_prequantized,
                quantized,
                kernel=kernel,
            )
        )

    executor.capabilities = capabilities
    if kernel == "w8a8":
        def mapped_kv_executor(
            request: PreparedAttention,
            key_source_indices: torch.Tensor,
        ) -> AttentionExecutionOutcome:
            call, reason = _inspect_mapped_attention_call(
                capabilities, request, key_source_indices
            )
            if reason is not None:
                return AttentionExecutionOutcome.unsupported(reason)
            query, key, value = request.consume_qkv()
            qk = _profiled(
                "attention.qk_norm_rope_quant",
                prequantize_turing_qk,
                query,
                key,
                request.qk_transform,
                kernel="w8a8",
                key_source_indices=key_source_indices,
            )
            del query, key
            quantized = _profiled(
                "attention.value_prepare",
                prequantize_turing_attention_from_qk,
                qk,
                value,
                call,
                kernel="w8a8",
                scale=request.scale,
                is_causal=request.is_causal,
                value_source_indices=key_source_indices,
            )
            del qk, value
            return AttentionExecutionOutcome(
                _profiled(
                    "attention.execute",
                    turing_attention_from_prequantized,
                    quantized,
                    kernel="w8a8",
                )
            )

        setattr(executor, MAPPED_KV_EXECUTOR_ATTR, mapped_kv_executor)

        def streamed_qkv_executor(
            qk,
            value: torch.Tensor,
            *,
            heads: int,
            qk_transform,
            transformer_options,
        ) -> AttentionExecutionOutcome:
            del qk_transform
            if not torch.is_tensor(value) or value.ndim != 4 or value.shape[2] < 64:
                return AttentionExecutionOutcome.unsupported(
                    "streamed QKV requires HND W8A8 with at least 64 tokens"
                )
            prototype = value[:, :, :1, :].expand(
                value.shape[0], value.shape[1], value.shape[2], value.shape[3]
            )
            call, reason = inspect_turing_attention_call(
                prototype,
                prototype,
                prototype,
                heads,
                mask=None,
                skip_reshape=True,
                skip_output_reshape=False,
                enable_gqa=False,
                low_precision_attention=True,
                is_causal=False,
                kernel="w8a8",
                require_long_sequence=True,
            )
            if reason is not None:
                return AttentionExecutionOutcome.unsupported(reason)
            quantized = _profiled(
                "attention.value_prepare",
                prequantize_turing_attention_from_qk,
                qk,
                value,
                call,
                kernel="w8a8",
                scale=None,
            )
            return AttentionExecutionOutcome(
                _profiled(
                    "attention.execute",
                    turing_attention_from_prequantized,
                    quantized,
                    kernel="w8a8",
                )
            )

        executor.turing_utils_streamed_qkv_executor = streamed_qkv_executor
    mapped_residual_executor = _make_mapped_residual_executor(
        capabilities,
        use_w8a8=kernel == "w8a8",
        kernel_capability=(
            "mapped_sparse_kv"
            if kernel == "w8a8"
            else "mapped_sparse_fp16_kv"
        ),
    )
    setattr(executor, MAPPED_RESIDUAL_EXECUTOR_ATTR, mapped_residual_executor)
    return executor


def _prepared_external_call_reason(request: PreparedAttention) -> str | None:
    query, key, value = request.peek_qkv()
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return "prepared Q/K/V dtypes differ"
    if query.device != key.device or query.device != value.device:
        return "prepared Q/K/V devices differ"
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"prepared Q/K/V dtype {query.dtype} is unsupported"
    frequency_specs = (
        ("query", request.qk_transform.freqs, request.query_tokens, request.heads),
        ("key", request.qk_transform.key_freqs, request.key_tokens, request.kv_heads),
    )
    for name, freqs, tokens, heads in frequency_specs:
        if not torch.is_tensor(freqs):
            continue
        if freqs.device != query.device:
            return f"prepared {name} RoPE frequencies are on a different device"
        if freqs.ndim < 3:
            return f"prepared {name} RoPE frequencies have no token/head axes"
        if int(freqs.shape[1]) not in {1, tokens}:
            return f"prepared {name} RoPE token count does not match {name}"
        frequency_heads = int(freqs.shape[2])
        if frequency_heads not in {1, heads}:
            return f"prepared {name} RoPE head count does not match {name}"
    return None


def _prepared_rms_norm_hnd(value: torch.Tensor, spec) -> torch.Tensor:
    weight = spec.weight.to(device=value.device, dtype=value.dtype)
    if spec.scope == "head":
        return torch.nn.functional.rms_norm(
            value,
            (value.shape[-1],),
            weight=weight,
            eps=spec.epsilon,
        )
    batch, heads, tokens, head_dim = value.shape
    nhd = value.transpose(1, 2)
    flattened = nhd.reshape(batch, tokens, heads * head_dim)
    normalized = torch.nn.functional.rms_norm(
        flattened,
        (heads * head_dim,),
        weight=weight,
        eps=spec.epsilon,
    )
    return normalized.view(batch, tokens, heads, head_dim).transpose(1, 2)


def _prepared_rope_one(
    value: torch.Tensor,
    freqs: torch.Tensor,
    *,
    rot_dim: int,
    pairing: str,
) -> torch.Tensor:
    nhd = value.transpose(1, 2)
    prefix = nhd[..., :rot_dim]
    if pairing == "interleaved":
        source = prefix.to(freqs.dtype).reshape(*prefix.shape[:-1], -1, 1, 2)
    elif pairing == "split_half":
        source = (
            prefix.reshape(*prefix.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .to(freqs.dtype)
        )
    else:
        raise ValueError(f"unsupported prepared RoPE pairing: {pairing}")
    if source.shape[2] != 1 and freqs.shape[2] != 1:
        if source.shape[2] != freqs.shape[2]:
            freqs = freqs[:, :, : source.shape[2]]
    rotated = freqs[..., 0] * source[..., 0]
    rotated.addcmul_(freqs[..., 1], source[..., 1])
    if pairing == "split_half":
        rotated = rotated.movedim(-1, -2)
    rotated = rotated.reshape(*prefix.shape).to(value.dtype)
    if rot_dim != nhd.shape[-1]:
        rotated = torch.cat((rotated, nhd[..., rot_dim:]), dim=-1)
    return rotated.transpose(1, 2)


def _prepared_qk_transform(
    query: torch.Tensor,
    key: torch.Tensor,
    spec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a prepared request's exact floating Q/K transform once.

    Bundled Sage/W8A8/Sol consume this contract through their quantizing fused
    preprocessor.  External Sage and SDPA need the same RMSNorm/RoPE semantics
    in floating point before receiving the already-projected tensors.
    """
    pairing = spec.rotary.pairing
    query_freqs = spec.freqs
    key_freqs = spec.key_freqs
    if pairing != "none" and spec.query_norm.scope == "head":
        try:
            import comfy.quant_ops

            q_nhd = query.transpose(1, 2)
            k_nhd = key.transpose(1, 2)
            q_weight = spec.query_norm_weight.to(device=query.device, dtype=query.dtype)
            k_weight = spec.key_norm_weight.to(device=key.device, dtype=key.dtype)
            suffix = "_split_half" if pairing == "split_half" else ""
            fused = getattr(comfy.quant_ops.ck, f"rms_rope{suffix}", None)
            fused_one = getattr(comfy.quant_ops.ck, f"rms_rope{suffix}1", None)
            kwargs = {"epsilon": spec.epsilon}
            if pairing == "split_half":
                kwargs["rot_dim"] = spec.rot_dim
            if (
                q_nhd.shape == k_nhd.shape
                and key_freqs is query_freqs
                and callable(fused)
            ):
                transformed = fused(
                    q_nhd,
                    k_nhd,
                    query_freqs,
                    q_weight,
                    k_weight,
                    **kwargs,
                )
                return transformed[0].transpose(1, 2), transformed[1].transpose(1, 2)
            if callable(fused_one):
                return (
                    fused_one(q_nhd, query_freqs, q_weight, **kwargs).transpose(1, 2),
                    fused_one(k_nhd, key_freqs, k_weight, **kwargs).transpose(1, 2),
                )
        except (AttributeError, ImportError, RuntimeError, TypeError):
            # The mathematical fallback below keeps official/older Comfy builds
            # functional when their fused floating transform rejects the call.
            pass

    query = _prepared_rms_norm_hnd(query, spec.query_norm)
    key = _prepared_rms_norm_hnd(key, spec.key_norm)
    if pairing == "none":
        return query, key
    return (
        _prepared_rope_one(
            query,
            query_freqs,
            rot_dim=spec.rot_dim,
            pairing=pairing,
        ),
        _prepared_rope_one(
            key,
            key_freqs,
            rot_dim=spec.rot_dim,
            pairing=pairing,
        ),
    )


def _make_external_prepared_executor(
    dense_override: Callable,
    backend: str,
) -> Callable:
    """Adapt projected Q/K/V directly to a ComfyUI-owned dense backend."""
    capabilities = AttentionBackendCapabilities(supports_mask=True)

    def executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = capabilities.unsupported_reason(request)
        if reason is None:
            reason = _prepared_external_call_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        query, key, value = request.consume_qkv()
        query, key = _profiled(
            "attention.qk_norm_rope",
            _prepared_qk_transform,
            query,
            key,
            request.qk_transform,
        )
        output = _profiled(
            "attention.execute",
            dense_override,
            _default_attention_fallback(),
            query,
            key,
            value,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            enable_gqa=request.heads != request.kv_heads,
            low_precision_attention=request.low_precision_attention,
            is_causal=request.is_causal,
            scale=request.scale,
        )
        del query, key, value
        return AttentionExecutionOutcome(output)

    executor.capabilities = capabilities
    executor.turing_utils_attention_backend = backend
    mapped_residual_executor = _make_mapped_residual_executor(
        capabilities,
        use_w8a8=False,
        kernel_capability="mapped_sparse_fp16_kv",
    )
    setattr(executor, MAPPED_RESIDUAL_EXECUTOR_ATTR, mapped_residual_executor)
    return executor


def _default_attention_fallback() -> Callable:
    from comfy.ldm.modules import attention as comfy_attention

    return comfy_attention.optimized_attention


def _container_fallback(fallback: Callable, q, k, v, heads: int, *args, **kwargs):
    return _dtype_compatible_fallback(
        fallback,
        q.take(),
        k.take(),
        v.take(),
        heads,
        *args,
        **kwargs,
    )


def _make_dense_container_function(kernel: str) -> Callable:
    fallback = _default_attention_fallback()

    def container_function(
        q,
        k,
        v,
        heads: int,
        mask=None,
        attn_precision=None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs,
    ):
        call, reason = inspect_turing_attention_call(
            q.peek(),
            k.peek(),
            v.peek(),
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            enable_gqa=bool(kwargs.get("enable_gqa", False)),
            low_precision_attention=kwargs.get("low_precision_attention", True),
            is_causal=bool(kwargs.get("is_causal", False)),
            kernel=kernel,
            require_long_sequence=True,
        )
        if reason is not None:
            return _container_fallback(
                fallback,
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        query = q.take()
        key = k.take()
        value = v.take()
        quantized = prequantize_turing_attention(
            query,
            key,
            value,
            call,
            kernel=kernel,
            scale=kwargs.get("scale"),
            is_causal=bool(kwargs.get("is_causal", False)),
        )
        del query, key, value
        return turing_attention_from_prequantized(quantized, kernel=kernel)

    return container_function


def _uses_bundled_turing_attention(option: str, device: torch.device | None) -> bool:
    option = normalize_attention_backend(option)
    return bool(
        device is not None
        and (
            (option == "w8a8" and is_supported_attention_device(device))
            or (option == "sage" and is_supported_turing_device(device))
        )
    )


def _dtype_compatible_fallback(original: Callable, *args, **kwargs):
    qkv = args[:3]
    if (
        len(qkv) == 3
        and all(isinstance(value, torch.Tensor) for value in qkv)
        and all(value.dtype == torch.float32 for value in qkv)
    ):
        pytorch_attention = _comfy_attention_function("pytorch")
        if pytorch_attention is None:
            raise RuntimeError(
                "ComfyUI PyTorch attention is unavailable for the FP32 fallback"
            )
        return pytorch_attention(*args, **kwargs)
    return original(*args, **kwargs)


def _recoverable_external_backend_rejection(error: Exception) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "alignment",
            "not implemented",
            "not supported",
            "unsupported",
            "no kernel image",
            "requires cuda",
        )
    )


def _external_backend_call(
    backend: AttentionBackend,
    target: Callable,
    fallback: Callable,
    *args,
    **kwargs,
):
    try:
        return target(*args, **kwargs)
    except (RuntimeError, NotImplementedError) as error:
        if backend.option != "w8a8" or not _recoverable_external_backend_rejection(
            error
        ):
            raise
        key = (backend.attention_function, str(error).splitlines()[0])
        if key not in _LOGGED_EXTERNAL_BACKEND_REJECTIONS:
            LOG.warning(
                "External %s attention rejected this call (%s); using the "
                "pre-existing ComfyUI attention backend",
                backend.attention_function,
                key[1],
            )
            _LOGGED_EXTERNAL_BACKEND_REJECTIONS.add(key)
        return fallback(*args, **kwargs)


def _make_external_container_function(
    backend: AttentionBackend,
    target: Callable,
) -> Callable:
    fallback = _default_attention_fallback()

    def container_function(q, k, v, heads: int, *args, **kwargs):
        q.peek(), k.peek(), v.peek()
        query, key, value = q.take(), k.take(), v.take()
        compatible_fallback = lambda *fallback_args, **fallback_kwargs: (
            _dtype_compatible_fallback(
                fallback,
                *fallback_args,
                **fallback_kwargs,
            )
        )
        return _external_backend_call(
            backend,
            target,
            compatible_fallback,
            query,
            key,
            value,
            heads,
            *args,
            **kwargs,
        )

    return container_function


def _uses_turing_bf16_sdpa(q, k, v) -> bool:
    return bool(
        all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v))
        and q.dtype is torch.bfloat16
        and k.dtype is torch.bfloat16
        and v.dtype is torch.bfloat16
        and q.device == k.device == v.device
        and is_supported_turing_device(q.device)
    )


def _convert_sdpa_mask_to_fp16(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Match floating additive masks to FP16 Q/K/V; boolean masks stay exact."""
    if args and torch.is_tensor(args[0]) and args[0].is_floating_point():
        positional = list(args)
        positional[0] = positional[0].to(torch.float16)
        return tuple(positional), kwargs
    mask = kwargs.get("mask")
    if torch.is_tensor(mask) and mask.is_floating_point():
        kwargs = dict(kwargs)
        kwargs["mask"] = mask.to(torch.float16)
    return args, kwargs


def _turing_sdpa_fp16(
    target: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *args,
    **kwargs,
):
    """Avoid Turing's BF16 SDPA math fallback while preserving its API dtype."""
    use_fp16 = _uses_turing_bf16_sdpa(q, k, v)
    if not use_fp16:
        return target(q, k, v, heads, *args, **kwargs)

    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    args, kwargs = _convert_sdpa_mask_to_fp16(args, kwargs)
    return target(q, k, v, heads, *args, **kwargs).to(torch.bfloat16)


def _make_sdpa_container_function(target: Callable) -> Callable:
    """Consume Q/K/V before SDPA conversion so BF16 and FP16 overlap is bounded."""

    def container_function(q, k, v, heads: int, *args, **kwargs):
        # Validate every owner before the first take to avoid a partial transfer.
        query_view, key_view, value_view = q.peek(), k.peek(), v.peek()
        use_fp16 = _uses_turing_bf16_sdpa(query_view, key_view, value_view)
        del query_view, key_view, value_view

        query = q.take()
        if use_fp16:
            query = query.to(torch.float16)
        key = k.take()
        if use_fp16:
            key = key.to(torch.float16)
        value = v.take()
        if use_fp16:
            value = value.to(torch.float16)

        if not use_fp16:
            return target(query, key, value, heads, *args, **kwargs)
        args, kwargs = _convert_sdpa_mask_to_fp16(args, kwargs)
        return target(query, key, value, heads, *args, **kwargs).to(torch.bfloat16)

    return container_function


def make_attention_override(
    option: str, device: torch.device | None = None
) -> Callable:
    option = normalize_attention_backend(option)
    bundled_turing = _uses_bundled_turing_attention(option, device)
    if bundled_turing:
        if option == "w8a8" and not bundled_w8a8_available():
            raise RuntimeError(
                "The bundled Turing W8A8 extension is unavailable. "
                "Rebuild comfyui-turing-utils-kernel 0.23.0 or newer with sm75 enabled."
            )
        if not bundled_available():
            raise RuntimeError(
                "The bundled Turing Sage extensions are unavailable. "
                "Rebuild comfyui-turing-utils-kernel with COMFYUI_TURING_UTILS_ARCH_LIST including 7.5."
            )
        if option == "w8a8":
            preflight_bundled_w8a8(device)
        else:
            preflight_bundled(device)
        backend = _BACKENDS[option]
        target = turing_w8a8_attention if option == "w8a8" else turing_sage_attention
        implementation = (
            "bundled_turing_w8a8" if option == "w8a8" else "bundled_turing_sage"
        )
    else:
        backend, target = _select_attention_backend(option)
        implementation = f"comfy:{backend.attention_function}"

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: _dtype_compatible_fallback(
            original, *fallback_args, **fallback_kwargs
        )
        if bundled_turing:
            return target(fallback, *args, **kwargs)
        if (
            backend.option == "sage"
            and len(args) >= 3
            and all(isinstance(value, torch.Tensor) for value in args[:3])
            and any(value.dtype == torch.float32 for value in args[:3])
        ):
            return fallback(*args, **kwargs)
        if backend.option == "sdpa" and len(args) >= 4:
            return _turing_sdpa_fp16(target, *args, **kwargs)
        return _external_backend_call(backend, target, fallback, *args, **kwargs)

    attention_override.turing_utils_attention_backend = backend.option
    attention_override.turing_utils_attention_implementation = implementation
    if bundled_turing and split_prequantization_available():
        attention_override.container_function = _make_dense_container_function(
            "w8a8" if option == "w8a8" else "sage"
        )
    if bundled_turing and fused_qk_preprocessing_available():
        attention_override.prepared_attention_executor = _make_dense_prepared_executor(
            "w8a8" if option == "w8a8" else "sage"
        )
    if not bundled_turing and backend.option == "sdpa":
        attention_override.container_function = _make_sdpa_container_function(target)
    elif not bundled_turing and callable(getattr(target, "container_function", None)):
        attention_override.container_function = _make_external_container_function(
            backend,
            target,
        )
    if not bundled_turing:
        attention_override.prepared_attention_executor = (
            _make_external_prepared_executor(attention_override, backend.option)
        )
    return attention_override


def _sparse_options(min_sequence_tokens, prefix_policy, manual_prefix_tokens):
    min_sequence_tokens = int(min_sequence_tokens)
    prefix_policy = str(prefix_policy).strip().lower()
    manual_prefix_tokens = int(manual_prefix_tokens)
    if min_sequence_tokens < 0:
        raise ValueError("min_sequence_tokens must be non-negative")
    if prefix_policy not in {"auto", "none", "manual"}:
        raise ValueError("prefix_policy must be auto, none, or manual")
    if manual_prefix_tokens < 0:
        raise ValueError("manual_prefix_tokens must be non-negative")
    return min_sequence_tokens, prefix_policy, manual_prefix_tokens


def make_sparse_attention_override(
    device: torch.device,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
    dense_backend: str = "w8a8",
    dense_override: Callable | None = None,
) -> Callable:
    min_sequence_tokens, prefix_policy, manual_prefix_tokens = _sparse_options(
        min_sequence_tokens, prefix_policy, manual_prefix_tokens,
    )
    routing_threshold = float(routing_threshold)
    skipped_residual = str(skipped_residual).strip().lower()
    sparse_reference_image = bool(sparse_reference_image)
    sparse_reference_video = bool(sparse_reference_video)
    sparse_reference_audio = bool(sparse_reference_audio)
    dense_prefix_steps = int(dense_prefix_steps)
    dense_suffix_steps = int(dense_suffix_steps)
    dense_prefix_layers = int(dense_prefix_layers)
    dense_suffix_layers = int(dense_suffix_layers)
    debug_route_density = bool(debug_route_density)
    dense_backend = normalize_attention_backend(dense_backend)
    use_w8a8 = dense_backend == "w8a8"
    if not math.isfinite(routing_threshold):
        raise ValueError("routing_threshold must be finite")
    if skipped_residual not in {"1x64", "2x32"}:
        raise ValueError("skipped_residual must be 1x64 or 2x32")
    if dense_prefix_steps < 0:
        raise ValueError("dense_prefix_steps must be non-negative")
    if dense_suffix_steps < 0:
        raise ValueError("dense_suffix_steps must be non-negative")
    if dense_prefix_layers < 0:
        raise ValueError("dense_prefix_layers must be non-negative")
    if dense_suffix_layers < 0:
        raise ValueError("dense_suffix_layers must be non-negative")
    if not is_supported_attention_device(device):
        raise RuntimeError(
            "Sol sparse attention requires a CUDA Tensor Core GPU (sm75 or newer)"
        )
    if not bundled_sparse_available():
        raise RuntimeError(
            "The bundled Sol sparse extension is unavailable. Rebuild "
            "comfyui-turing-utils-kernel 0.28.0 or newer with "
            "COMFYUI_TURING_UTILS_ARCH_LIST including the target GPU architecture."
        )
    preflight_bundled_sparse(device)
    if use_w8a8:
        if not bundled_w8a8_available():
            raise RuntimeError(
                "Sol W8A8 requires comfyui-turing-utils-kernel 0.28.0 or newer"
            )
    schedule = SparseSchedule(
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
    )
    schedule_state_for = schedule.state_for
    debug_route_keys: set[tuple] = set()
    debug_route_state: dict[tuple, list[tuple[torch.Tensor, int, int]]] = {}
    debug_dense_reasons: set[str] = set()
    if dense_override is None:
        dense_override = make_attention_override(dense_backend, device=device)
    dense_prepared_executor = getattr(
        dense_override, "prepared_attention_executor", None
    )
    dense_container = getattr(dense_override, "container_function", None)
    if use_w8a8 and not callable(dense_prepared_executor):
        # Preserve the native force-dense W8A8 path on every sm75+ target.
        dense_prepared_executor = _make_dense_prepared_executor("w8a8")
        dense_container = _make_dense_container_function("w8a8")
    dense_fallback = DenseAttentionFallback(
        dense_override,
        default_fallback=_default_attention_fallback,
        prepared_executor=dense_prepared_executor,
        container=dense_container,
    )
    dense_streamed_qkv_executor = dense_fallback.streamed_qkv_executor
    run_dense_prepared = dense_fallback.run_prepared
    run_dense_container = dense_fallback.run_container
    sparse_capabilities = AttentionBackendCapabilities(
        supports_semantic_sparse=True,
    )

    def route_debug_context(transformer_options, kernel_key: tuple):
        schedule_state = schedule_state_for(transformer_options)
        layer_index, layer_count = _attention_layer_metadata(transformer_options)
        step = schedule_state.get("step")
        sampling_steps = schedule_state.get("sampling_steps")
        last_sparse_layer = (
            layer_count - dense_suffix_layers - 1
            if isinstance(layer_count, int) and not isinstance(layer_count, bool)
            else None
        )
        aggregate = (
            debug_route_density
            and isinstance(step, int)
            and not isinstance(step, bool)
            and isinstance(layer_index, int)
            and not isinstance(layer_index, bool)
            and isinstance(layer_count, int)
            and not isinstance(layer_count, bool)
            and layer_count > 0
            and 0 <= layer_index < layer_count
            and isinstance(last_sparse_layer, int)
            and 0 <= last_sparse_layer < layer_count
        )
        return {
            "collect": debug_route_density
            and (aggregate or kernel_key not in debug_route_keys),
            "aggregate": aggregate,
            "step": step,
            "sampling_steps": sampling_steps,
            "layer_index": layer_index,
            "layer_count": layer_count,
            "last_sparse_layer": last_sparse_layer,
        }

    def record_route_stats(
        selected_device: torch.Tensor,
        possible_blocks: int,
        sol_call,
        kernel_key: tuple,
        context: dict,
    ) -> None:
        protected_query_tokens = sum(
            stop - start for start, stop in sol_call.dense_query_ranges
        )
        sparse_query_tokens = sol_call.attention.query_tokens - protected_query_tokens
        if context["aggregate"]:
            aggregate_key = (
                context["step"],
                context["sampling_steps"],
                kernel_key,
            )
            entries = debug_route_state.setdefault(aggregate_key, [])
            entries.append((selected_device, possible_blocks, context["layer_index"]))
            if context["layer_index"] != context["last_sparse_layer"]:
                return
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
            LOG.info(
                "[Turing sparse debug] step=%s/%s layers=%d-%d calls=%d "
                "selected=%d/%d density[min/mean/max]=%.4f/%.4f/%.4f "
                "Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d threshold=%.2f "
                "protected_q=%d local=1 residual=%s",
                context["step"],
                context["sampling_steps"],
                min(entry[2] for entry in entries),
                max(entry[2] for entry in entries),
                len(entries),
                int(summary[0]),
                int(summary[1]),
                summary[2],
                summary[3],
                summary[4],
                sol_call.attention.query_tokens,
                sparse_query_tokens,
                sol_call.attention.key_tokens,
                sol_call.attention.heads,
                sol_call.attention.kv_heads,
                routing_threshold,
                protected_query_tokens,
                skipped_residual,
            )
            del debug_route_state[aggregate_key]
            return

        selected_blocks = int(selected_device.item())
        LOG.info(
            "[Turing sparse debug] Q=%d Qsparse=%d K=%d Hq=%d Hkv=%d "
            "selected=%d/%d density=%.4f threshold=%.2f protected_q=%d "
            "local=1 residual=%s step=%s/%s layer=%s/%s",
            sol_call.attention.query_tokens,
            sparse_query_tokens,
            sol_call.attention.key_tokens,
            sol_call.attention.heads,
            sol_call.attention.kv_heads,
            selected_blocks,
            possible_blocks,
            selected_blocks / possible_blocks if possible_blocks else 0.0,
            routing_threshold,
            protected_query_tokens,
            skipped_residual,
            context["step"],
            context["sampling_steps"],
            context["layer_index"],
            context["layer_count"],
        )
        debug_route_keys.add(kernel_key)

    def prepared_executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = sparse_capabilities.unsupported_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        transformer_options = request.transformer_options
        if schedule.is_dense(transformer_options):
            return run_dense_prepared(request)

        query_view, key_view, value_view = request.peek_qkv()
        sol_call, reason = inspect_sol_attention_call(
            query_view,
            key_view,
            value_view,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            min_sequence_tokens=min_sequence_tokens,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            skipped_residual=skipped_residual,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            transformer_options=transformer_options,
            kwargs={
                "enable_gqa": request.heads != request.kv_heads,
                "low_precision_attention": request.low_precision_attention,
                "is_causal": request.is_causal,
            },
        )
        if reason is not None:
            return run_dense_prepared(request)
        reason = _prepared_call_mismatch(request, sol_call.attention)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)

        del query_view, key_view, value_view
        query, key, value = request.consume_qkv()
        qk = _profiled(
            "attention.qk_norm_rope_quant",
            prequantize_turing_qk,
            query,
            key,
            request.qk_transform,
            kernel="sol",
        )
        del query, key
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sol_attention_from_qk,
            qk,
            value,
            sol_call,
            routing_threshold=routing_threshold,
            scale=request.scale,
            use_w8a8=use_w8a8,
        )
        del qk, value
        debug_key = (
            sol_call.attention.input_dtype,
            sol_call.attention.query_tokens,
            sol_call.attention.key_tokens,
            sol_call.dense_query_ranges,
            sol_call.exact_kv_ranges,
            routing_threshold,
            sol_call.residual_subblocks,
            use_w8a8,
        )
        debug_context = route_debug_context(transformer_options, debug_key)
        collect_debug_stats = debug_context["collect"]
        collect_stats = collect_debug_stats or CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sol_attention_from_prequantized,
            quantized,
            return_stats=collect_stats,
        )
        if not collect_stats:
            return AttentionExecutionOutcome(result)
        output, selected, possible = result
        _profile_route_stats(selected, possible)
        if collect_debug_stats:
            record_route_stats(selected, possible, sol_call, debug_key, debug_context)
        return AttentionExecutionOutcome(output)

    def streamed_qkv_executor(
        qk,
        value: torch.Tensor,
        *,
        heads: int,
        qk_transform,
        transformer_options,
    ) -> AttentionExecutionOutcome:
        """Finish attention from row-streamed Q/K INT8 and retained V BF16."""
        if (
            not use_w8a8
            or not torch.is_tensor(value)
            or value.ndim != 4
            or value.shape[2] < 64
        ):
            return AttentionExecutionOutcome.unsupported(
                "streamed QKV requires HND W8A8 with at least 64 tokens"
            )
        prototype = value[:, :, :1, :].expand(
            value.shape[0], value.shape[1], value.shape[2], value.shape[3]
        )

        def dense_result() -> AttentionExecutionOutcome:
            if not callable(dense_streamed_qkv_executor):
                return AttentionExecutionOutcome.unsupported(
                    "the selected dense backend cannot consume streamed QKV"
                )
            return dense_streamed_qkv_executor(
                qk,
                value,
                heads=heads,
                qk_transform=qk_transform,
                transformer_options=transformer_options,
            )

        if schedule.is_dense(transformer_options):
            return dense_result()

        sol_call, reason = inspect_sol_attention_call(
            prototype,
            prototype,
            prototype,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=False,
            min_sequence_tokens=min_sequence_tokens,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            skipped_residual=skipped_residual,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            transformer_options=transformer_options,
            kwargs={
                "enable_gqa": False,
                "low_precision_attention": True,
                "is_causal": False,
            },
        )
        if reason is not None:
            return dense_result()
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sol_attention_from_qk,
            qk,
            value,
            sol_call,
            routing_threshold=routing_threshold,
            scale=None,
            use_w8a8=True,
        )
        debug_key = (
            sol_call.attention.input_dtype,
            sol_call.attention.query_tokens,
            sol_call.attention.key_tokens,
            sol_call.dense_query_ranges,
            sol_call.exact_kv_ranges,
            routing_threshold,
            sol_call.residual_subblocks,
            True,
        )
        debug_context = route_debug_context(transformer_options, debug_key)
        collect_debug_stats = debug_context["collect"]
        collect_stats = collect_debug_stats or CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sol_attention_from_prequantized,
            quantized,
            return_stats=collect_stats,
        )
        if not collect_stats:
            return AttentionExecutionOutcome(result)
        output, selected, possible = result
        _profile_route_stats(selected, possible)
        if collect_debug_stats:
            record_route_stats(selected, possible, sol_call, debug_key, debug_context)
        return AttentionExecutionOutcome(output)

    prepared_executor.capabilities = sparse_capabilities
    if use_w8a8:
        prepared_executor.turing_utils_streamed_qkv_executor = streamed_qkv_executor

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: dense_override(
            original, *fallback_args, **fallback_kwargs
        )
        transformer_options = kwargs.get("transformer_options")
        schedule_state = schedule_state_for(transformer_options)
        dense_schedule = schedule.dense_step(transformer_options)
        dense_layer = schedule.dense_layer(transformer_options)
        if debug_route_density and dense_schedule:
            debug_key = f"schedule:{schedule_state.get('step')}"
            if debug_key not in debug_dense_reasons:
                LOG.info(
                    "[Sol sparse debug] dense backend selected by schedule: "
                    "step=%s/%s prefix_steps=%s suffix_steps=%s",
                    schedule_state.get("step"),
                    schedule_state.get("sampling_steps"),
                    schedule_state.get("prefix_steps"),
                    schedule_state.get("suffix_steps"),
                )
                debug_dense_reasons.add(debug_key)
        if debug_route_density and dense_layer:
            layer_index, layer_count = _attention_layer_metadata(transformer_options)
            debug_key = f"layer:{layer_index}"
            if debug_key not in debug_dense_reasons:
                LOG.info(
                    "[Sol sparse debug] dense backend selected for protected layer %s/%s",
                    layer_index,
                    layer_count,
                )
                debug_dense_reasons.add(debug_key)
        if dense_schedule or dense_layer:
            return fallback(*args, **kwargs)
        debug_context = None
        if debug_route_density:
            layer_index, layer_count = _attention_layer_metadata(transformer_options)
            debug_context = {
                "step": schedule_state.get("step"),
                "sampling_steps": schedule_state.get("sampling_steps"),
                "layer_index": layer_index,
                "layer_count": layer_count,
                "last_sparse_layer": (
                    layer_count - dense_suffix_layers - 1
                    if isinstance(layer_count, int)
                    and not isinstance(layer_count, bool)
                    else None
                ),
            }
        return turing_sol_sparse_attention(
            fallback,
            *args,
            min_sequence_tokens=min_sequence_tokens,
            routing_threshold=routing_threshold,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            skipped_residual=skipped_residual,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            debug_route_density=debug_route_density,
            debug_route_keys=debug_route_keys if debug_route_density else None,
            debug_route_state=debug_route_state if debug_route_density else None,
            debug_context=debug_context,
            use_w8a8=use_w8a8,
            **kwargs,
        )

    if split_prequantization_available():

        def container_function(
            q,
            k,
            v,
            heads: int,
            mask=None,
            attn_precision=None,
            skip_reshape: bool = False,
            skip_output_reshape: bool = False,
            **kwargs,
        ):
            transformer_options = kwargs.get("transformer_options")
            if schedule.is_dense(transformer_options):
                return run_dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            sol_call, reason = inspect_sol_attention_call(
                q.peek(),
                k.peek(),
                v.peek(),
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                min_sequence_tokens=min_sequence_tokens,
                prefix_policy=prefix_policy,
                manual_prefix_tokens=manual_prefix_tokens,
                skipped_residual=skipped_residual,
                sparse_reference_image=sparse_reference_image,
                sparse_reference_video=sparse_reference_video,
                sparse_reference_audio=sparse_reference_audio,
                transformer_options=transformer_options,
                kwargs=kwargs,
            )
            if reason is not None:
                return run_dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            query = q.take()
            key = k.take()
            value = v.take()
            quantized = prequantize_turing_sol_attention(
                query,
                key,
                value,
                sol_call,
                routing_threshold=routing_threshold,
                scale=kwargs.get("scale"),
                use_w8a8=use_w8a8,
            )
            del query, key, value
            debug_key = (
                sol_call.attention.input_dtype,
                sol_call.attention.query_tokens,
                sol_call.attention.key_tokens,
                sol_call.dense_query_ranges,
                sol_call.exact_kv_ranges,
                routing_threshold,
                sol_call.residual_subblocks,
                use_w8a8,
            )
            debug_context = route_debug_context(transformer_options, debug_key)
            collect_debug_stats = debug_context["collect"]
            collect_stats = collect_debug_stats or CUDA_PHASE_PROFILER.enabled
            result = turing_sol_attention_from_prequantized(
                quantized,
                return_stats=collect_stats,
            )
            if collect_stats:
                output, selected, possible = result
                _profile_route_stats(selected, possible)
                if collect_debug_stats:
                    record_route_stats(
                        selected, possible, sol_call, debug_key, debug_context
                    )
                return output
            return result

        attention_override.container_function = container_function

    attention_override.turing_utils_attention_backend = "sol_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_sol_sparse"
    attention_override.turing_utils_dense_implementation = getattr(
        dense_override,
        "turing_utils_attention_implementation",
        f"inherited:{dense_backend}",
    )
    attention_override.turing_utils_dense_backend = dense_backend
    attention_override.turing_utils_sparse_numeric_backend = (
        "w8a8" if use_w8a8 else "fp16"
    )
    if fused_qk_preprocessing_available():
        attention_override.prepared_attention_executor = prepared_executor
    return attention_override


def make_sla_attention_override(
    device: torch.device,
    min_sequence_tokens: int = 0,
    sparsity_ratio: float = SLA_SPARSITY_RATIO,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SLA_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SLA_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SLA_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SLA_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
    dense_backend: str = "w8a8",
    dense_override: Callable | None = None,
) -> Callable:
    min_sequence_tokens, prefix_policy, manual_prefix_tokens = _sparse_options(
        min_sequence_tokens, prefix_policy, manual_prefix_tokens,
    )
    sparsity_ratio = float(sparsity_ratio)
    sparse_reference_image = bool(sparse_reference_image)
    sparse_reference_video = bool(sparse_reference_video)
    sparse_reference_audio = bool(sparse_reference_audio)
    dense_prefix_steps = int(dense_prefix_steps)
    dense_suffix_steps = int(dense_suffix_steps)
    dense_prefix_layers = int(dense_prefix_layers)
    dense_suffix_layers = int(dense_suffix_layers)
    debug_route_density = bool(debug_route_density)
    dense_backend = normalize_attention_backend(dense_backend)
    use_w8a8 = dense_backend == "w8a8"
    if not math.isfinite(sparsity_ratio) or not 0.0 <= sparsity_ratio < 1.0:
        raise ValueError("sparsity_ratio must be finite and in [0, 1)")
    if (
        min(
            dense_prefix_steps,
            dense_suffix_steps,
            dense_prefix_layers,
            dense_suffix_layers,
        )
        < 0
    ):
        raise ValueError("dense step/layer counts must be non-negative")
    if not is_supported_attention_device(device):
        raise RuntimeError(
            "SLA sparse attention requires a CUDA Tensor Core GPU (sm75 or newer)"
        )
    if not bundled_sla_available():
        raise RuntimeError(
            "The bundled SLA extension is unavailable. Rebuild "
            "comfyui-turing-utils-kernel 0.29.1 or newer for this GPU."
        )
    preflight_bundled_sla(device)
    if use_w8a8 and not bundled_w8a8_available():
        raise RuntimeError("SLA W8A8 requires the bundled W8A8 attention ABI")

    schedule = SparseSchedule(
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
    )
    schedule_state_for = schedule.state_for
    debug_route_keys: set[tuple] = set()
    if dense_override is None:
        dense_override = make_attention_override(dense_backend, device=device)
    dense_fallback = DenseAttentionFallback(
        dense_override,
        default_fallback=_default_attention_fallback,
    )
    dense_streamed_qkv_executor = dense_fallback.streamed_qkv_executor
    run_dense_prepared = dense_fallback.run_prepared
    run_dense_container = dense_fallback.run_container
    sparse_capabilities = AttentionBackendCapabilities(
        supports_semantic_sparse=True,
    )

    def inspect(
        request_q,
        request_k,
        request_v,
        heads,
        *,
        mask,
        skip_reshape,
        skip_output_reshape,
        transformer_options,
        kwargs,
    ):
        return inspect_sla_attention_call(
            request_q,
            request_k,
            request_v,
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
        )

    def collect_stats(result, sla_call, transformer_options):
        schedule_state = schedule_state_for(transformer_options)
        output, selected, possible = result
        layer_index, _ = _attention_layer_metadata(transformer_options)
        debug_key = (
            schedule_state.get("step"),
            layer_index,
            sla_call.attention.query_tokens,
            sla_call.attention.key_tokens,
            sla_call.dense_query_ranges,
            sla_call.exact_kv_ranges,
        )
        if debug_key not in debug_route_keys:
            selected_blocks = int(selected.item())
            LOG.info(
                "[Turing SLA debug] step=%s layer=%s Q=%d K=%d "
                "selected=%d/%d density=%.4f target_sparsity=%.2f "
                "protected_q=%d",
                schedule_state.get("step"),
                layer_index,
                sla_call.attention.query_tokens,
                sla_call.attention.key_tokens,
                selected_blocks,
                possible,
                selected_blocks / possible if possible else 0.0,
                sparsity_ratio,
                sum(stop - start for start, stop in sla_call.dense_query_ranges),
            )
            debug_route_keys.add(debug_key)
        return output

    def is_dense(transformer_options) -> bool:
        return schedule.is_dense(
            transformer_options,
            force_dense=sparsity_ratio == 0.0,
        )

    def prepared_executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        reason = sparse_capabilities.unsupported_reason(request)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        transformer_options = request.transformer_options
        if is_dense(transformer_options):
            return run_dense_prepared(request)
        query_view, key_view, value_view = request.peek_qkv()
        sla_call, reason = inspect(
            query_view,
            key_view,
            value_view,
            request.heads,
            mask=request.mask,
            skip_reshape=True,
            skip_output_reshape=request.skip_output_reshape,
            transformer_options=transformer_options,
            kwargs={
                "enable_gqa": request.heads != request.kv_heads,
                "low_precision_attention": request.low_precision_attention,
                "is_causal": request.is_causal,
            },
        )
        if reason is not None:
            return run_dense_prepared(request)
        reason = _prepared_call_mismatch(request, sla_call.attention)
        if reason is not None:
            return AttentionExecutionOutcome.unsupported(reason)
        del query_view, key_view, value_view
        query, key, value = request.consume_qkv()
        qk = _profiled(
            "attention.qk_norm_rope_quant",
            prequantize_turing_qk,
            query,
            key,
            request.qk_transform,
            kernel="sla",
        )
        del query, key
        quantized = _profiled(
            "attention.value_route_prepare",
            prequantize_turing_sla_attention_from_qk,
            qk,
            value,
            sla_call,
            sparsity_ratio=sparsity_ratio,
            scale=request.scale,
            use_w8a8=use_w8a8,
        )
        del qk, value
        profile_route_density = CUDA_PHASE_PROFILER.enabled
        result = _profiled(
            "attention.execute",
            turing_sla_attention_from_prequantized,
            quantized,
            return_stats=debug_route_density or profile_route_density,
        )
        if debug_route_density or profile_route_density:
            output, selected, possible = result
            _profile_route_stats(selected, possible)
            if debug_route_density:
                output = collect_stats(
                    (output, selected, possible), sla_call, transformer_options
                )
        else:
            output = result
        return AttentionExecutionOutcome(output)

    prepared_executor.capabilities = sparse_capabilities
    if use_w8a8:

        def streamed_qkv_executor(
            qk,
            value: torch.Tensor,
            *,
            heads: int,
            qk_transform,
            transformer_options,
        ) -> AttentionExecutionOutcome:
            """Finish SLA from the shared row-streamed H3 QKV representation."""
            if not torch.is_tensor(value) or value.ndim != 4 or value.shape[2] < 64:
                return AttentionExecutionOutcome.unsupported(
                    "streamed QKV requires HND W8A8 with at least 64 tokens"
                )
            if is_dense(transformer_options):
                if not callable(dense_streamed_qkv_executor):
                    return AttentionExecutionOutcome.unsupported(
                        "the selected dense backend cannot consume streamed QKV"
                    )
                return dense_streamed_qkv_executor(
                    qk,
                    value,
                    heads=heads,
                    qk_transform=qk_transform,
                    transformer_options=transformer_options,
                )

            prototype = value[:, :, :1, :].expand(
                value.shape[0], value.shape[1], value.shape[2], value.shape[3]
            )
            sla_call, reason = inspect(
                prototype,
                prototype,
                prototype,
                heads,
                mask=None,
                skip_reshape=True,
                skip_output_reshape=False,
                transformer_options=transformer_options,
                kwargs={
                    "enable_gqa": False,
                    "low_precision_attention": True,
                    "is_causal": False,
                },
            )
            if reason is not None:
                return AttentionExecutionOutcome.unsupported(reason)
            quantized = _profiled(
                "attention.value_route_prepare",
                prequantize_turing_sla_attention_from_qk,
                qk,
                value,
                sla_call,
                sparsity_ratio=sparsity_ratio,
                scale=None,
                use_w8a8=True,
            )
            profile_route_density = CUDA_PHASE_PROFILER.enabled
            result = _profiled(
                "attention.execute",
                turing_sla_attention_from_prequantized,
                quantized,
                return_stats=debug_route_density or profile_route_density,
            )
            if debug_route_density or profile_route_density:
                output, selected, possible = result
                _profile_route_stats(selected, possible)
                if debug_route_density:
                    output = collect_stats(
                        (output, selected, possible),
                        sla_call,
                        transformer_options,
                    )
            else:
                output = result
            return AttentionExecutionOutcome(output)

        prepared_executor.turing_utils_streamed_qkv_executor = streamed_qkv_executor

    def attention_override(original: Callable, *args, **kwargs):
        fallback = lambda *fallback_args, **fallback_kwargs: dense_override(
            original, *fallback_args, **fallback_kwargs
        )
        if is_dense(kwargs.get("transformer_options")):
            return fallback(*args, **kwargs)
        return turing_sla_sparse_attention(
            fallback,
            *args,
            min_sequence_tokens=min_sequence_tokens,
            sparsity_ratio=sparsity_ratio,
            prefix_policy=prefix_policy,
            manual_prefix_tokens=manual_prefix_tokens,
            sparse_reference_image=sparse_reference_image,
            sparse_reference_video=sparse_reference_video,
            sparse_reference_audio=sparse_reference_audio,
            debug_route_density=debug_route_density,
            use_w8a8=use_w8a8,
            **kwargs,
        )

    if split_prequantization_available():

        def container_function(
            q,
            k,
            v,
            heads: int,
            mask=None,
            attn_precision=None,
            skip_reshape: bool = False,
            skip_output_reshape: bool = False,
            **kwargs,
        ):
            transformer_options = kwargs.get("transformer_options")
            if is_dense(transformer_options):
                return run_dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            sla_call, reason = inspect(
                q.peek(),
                k.peek(),
                v.peek(),
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                transformer_options=transformer_options,
                kwargs=kwargs,
            )
            if reason is not None:
                return run_dense_container(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            query = q.take()
            key = k.take()
            value = v.take()
            quantized = prequantize_turing_sla_attention(
                query,
                key,
                value,
                sla_call,
                sparsity_ratio=sparsity_ratio,
                scale=kwargs.get("scale"),
                use_w8a8=use_w8a8,
            )
            del query, key, value
            profile_route_density = CUDA_PHASE_PROFILER.enabled
            result = turing_sla_attention_from_prequantized(
                quantized,
                return_stats=debug_route_density or profile_route_density,
            )
            if not (debug_route_density or profile_route_density):
                return result
            output, selected, possible = result
            _profile_route_stats(selected, possible)
            if debug_route_density:
                return collect_stats(
                    (output, selected, possible), sla_call, transformer_options
                )
            return output

        attention_override.container_function = container_function

    attention_override.turing_utils_attention_backend = "sla_sparse_attn"
    attention_override.turing_utils_attention_implementation = "bundled_sla_sparse"
    attention_override.turing_utils_dense_implementation = getattr(
        dense_override,
        "turing_utils_attention_implementation",
        f"inherited:{dense_backend}",
    )
    attention_override.turing_utils_dense_backend = dense_backend
    attention_override.turing_utils_sparse_numeric_backend = (
        "w8a8" if use_w8a8 else "fp16"
    )
    if fused_qk_preprocessing_available():
        attention_override.prepared_attention_executor = prepared_executor
    return attention_override


def attention_base_runtime(model) -> AttentionRuntimeConfig:
    """Resolve the immutable dense base used by a Sol/SLA strategy.

    Models loaded by the ConvRot loader already carry this capability marker.
    Official and third-party loaders are bootstrapped from their current
    override when possible, otherwise from SDPA.
    """
    _bootstrap_attention_integrations()
    transformer_options = model.model_options.setdefault("transformer_options", {})
    config = attention_runtime_config(transformer_options)
    if config is not None:
        return config

    current = transformer_options.get("optimized_attention_override")
    dense_backend = transformer_options.get(
        "turing_utils_attention_base_backend",
        transformer_options.get("turing_utils_attention_backend", "sdpa"),
    )
    if dense_backend not in {"w8a8", "sage", "sdpa"}:
        dense_backend = getattr(current, "turing_utils_attention_backend", "sdpa")
    if dense_backend not in {"w8a8", "sage", "sdpa"}:
        dense_backend = "sdpa"

    if not callable(current):
        current = make_attention_override(dense_backend, device=model.load_device)
    implementation = getattr(
        current,
        "turing_utils_attention_implementation",
        f"inherited:{dense_backend}",
    )
    return AttentionRuntimeConfig(
        dense_backend=dense_backend,
        dense_implementation=implementation,
        dense_override=current,
        native_runtime=False,
    )


def apply_sparse_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
):
    runtime = attention_base_runtime(model)
    override = make_sparse_attention_override(
        model.load_device,
        min_sequence_tokens=min_sequence_tokens,
        routing_threshold=routing_threshold,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        skipped_residual=skipped_residual,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
        dense_backend=runtime.dense_backend,
        dense_override=runtime.dense_override,
    )
    patched = install_attention_strategy(
        model,
        override,
        strategy="Sol sparse",
        backend="sol",
        implementation="bundled_sol_sparse",
        runtime_config=runtime,
    )
    patched = patched.model
    dense_implementation = getattr(
        override,
        "turing_utils_dense_implementation",
        "selected_dense_backend",
    )
    LOG.info("Attention: Sol · threshold=%.2f · dense=%s", routing_threshold, runtime.dense_backend)
    LOG.debug(
        "Sol sparse attention patch enabled: threshold=%.2f "
        "prefix_policy=%s manual_prefix=%d local_radius=1 "
        "skipped_residual=%s sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=%s dense_impl=%s pv_backend=%s debug_route_density=%s",
        routing_threshold,
        prefix_policy,
        manual_prefix_tokens,
        skipped_residual,
        sparse_reference_image,
        sparse_reference_video,
        sparse_reference_audio,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        runtime.dense_backend,
        dense_implementation,
        "u8xs8_tensorcore"
        if getattr(override, "turing_utils_sparse_numeric_backend", "fp16") == "w8a8"
        else "fp16_tensorcore",
        debug_route_density,
    )
    return patched


def apply_sla_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    sparsity_ratio: float = SLA_SPARSITY_RATIO,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SLA_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SLA_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SLA_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SLA_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
):
    runtime = attention_base_runtime(model)
    override = make_sla_attention_override(
        model.load_device,
        min_sequence_tokens=min_sequence_tokens,
        sparsity_ratio=sparsity_ratio,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
        dense_backend=runtime.dense_backend,
        dense_override=runtime.dense_override,
    )
    patched = install_attention_strategy(
        model,
        override,
        strategy="SLA",
        backend="sla",
        implementation="bundled_sla_sparse",
        runtime_config=runtime,
    )
    patched = patched.model
    LOG.info("Attention: SLA · sparsity=%.2f · dense=%s", sparsity_ratio, runtime.dense_backend)
    LOG.debug(
        "SLA sparse attention patch enabled: sparsity_ratio=%.2f "
        "topology=128x64 smooth_k=True prefix_policy=%s manual_prefix=%d "
        "sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=%s dense_impl=%s pv_backend=%s debug_route_density=%s",
        sparsity_ratio,
        prefix_policy,
        manual_prefix_tokens,
        sparse_reference_image,
        sparse_reference_video,
        sparse_reference_audio,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        runtime.dense_backend,
        override.turing_utils_dense_implementation,
        "u8xs8_tensorcore"
        if getattr(override, "turing_utils_sparse_numeric_backend", "fp16") == "w8a8"
        else "fp16_tensorcore",
        debug_route_density,
    )
    return patched


def apply_attention_backend(
    model,
    option: str,
    device: torch.device | None = None,
    *,
    native_runtime: bool = False,
):
    _bootstrap_attention_integrations()
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option, device=device)
    selected = override.turing_utils_attention_backend
    implementation = override.turing_utils_attention_implementation
    config = AttentionRuntimeConfig(
        dense_backend=selected,
        dense_implementation=implementation,
        dense_override=override,
        native_runtime=bool(native_runtime),
    )
    install_attention_runtime(transformer_options, config)
    prepared_executor = getattr(override, "prepared_attention_executor", None)
    if callable(prepared_executor):
        target_device = (
            device if device is not None else getattr(model, "load_device", None)
        )
        if isinstance(target_device, torch.device):
            site_status = ensure_prepared_attention_sites(model, target_device)
            if site_status.matched and site_status.reason is not None:
                LOG.debug(
                    "%s prepared-attention fusion was not installed: %s",
                    site_status.model_kind,
                    site_status.reason,
                )
    LOG.debug(
        "Turing Utils attention runtime: dense=%s via %s requested=%s native=%s",
        selected,
        implementation,
        option,
        native_runtime,
    )
    return model
