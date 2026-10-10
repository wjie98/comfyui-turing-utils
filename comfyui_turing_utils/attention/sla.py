"""SLA semantic sparse attention execution."""

from __future__ import annotations

import math
from collections.abc import Callable
import torch
from ..profiling import CUDA_PHASE_PROFILER
from .protocol import (
    AttentionBackendCapabilities,
    AttentionExecutionOutcome,
    PreparedAttention,
)
from .sparse_runtime import DenseAttentionFallback, SparseSchedule
from .sparse import (
    inspect_sla_attention_call,
    prequantize_turing_sla_attention,
    prequantize_turing_sla_attention_from_qk,
    turing_sla_attention_from_prequantized,
    turing_sla_sparse_attention,
)
from .stable import (
    LOG,
    SPARSE_PREFIX_POLICY,
    SPARSE_REFERENCE_AUDIO,
    SPARSE_REFERENCE_IMAGE,
    SPARSE_REFERENCE_VIDEO,
    SLA_DENSE_PREFIX_LAYERS,
    SLA_DENSE_PREFIX_STEPS,
    SLA_DENSE_SUFFIX_LAYERS,
    SLA_DENSE_SUFFIX_STEPS,
    SLA_SPARSITY_RATIO,
    bundled_sla_available,
    bundled_w8a8_available,
    fused_qk_preprocessing_available,
    is_supported_attention_device,
    normalize_attention_backend,
    prequantize_turing_qk,
    preflight_bundled_sla,
    split_prequantization_available,
)
from .execution import (
    _profiled,
    _profile_route_stats,
    _attention_layer_metadata,
    _prepared_call_mismatch,
    _default_attention_fallback,
    _sparse_options,
)
from .dense import (
    make_attention_override,
)


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
        min_sequence_tokens,
        prefix_policy,
        manual_prefix_tokens,
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
