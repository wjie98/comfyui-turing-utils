"""Dense attention factories and backend-specific container execution."""

from __future__ import annotations

from collections.abc import Callable
import torch
from ..profiling import CUDA_PHASE_PROFILER
from .protocol import (
    AttentionBackendCapabilities,
    AttentionExecutionOutcome,
    MAPPED_KV_EXECUTOR_ATTR,
    MAPPED_RESIDUAL_CAPABILITY_ATTR,
    MAPPED_RESIDUAL_EXECUTOR_ATTR,
    PreparedAttention,
)
from .sparse import (
    SolAttentionCall,
    prequantize_turing_sol_attention_from_qk,
    turing_sol_attention_from_prequantized,
)
from .stable import (
    LOG,
    AttentionBackend,
    _BACKENDS,
    _select_attention_backend,
    bundled_available,
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
    preflight_bundled_w8a8,
    split_prequantization_available,
    turing_attention_from_prequantized,
    turing_sage_attention,
    turing_w8a8_attention,
)
from .execution import (
    _dtype_compatible_fallback,
    _profiled,
    _profile_route_stats,
    _prepared_call_mismatch,
    _prepared_external_call_reason,
    _prepared_qk_transform,
    _default_attention_fallback,
    _container_fallback,
)


_LOGGED_EXTERNAL_BACKEND_REJECTIONS: set[tuple[str, str]] = set()


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
            "mapped_sparse_kv" if kernel == "w8a8" else "mapped_sparse_fp16_kv"
        ),
    )
    setattr(executor, MAPPED_RESIDUAL_EXECUTOR_ATTR, mapped_residual_executor)
    return executor


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
