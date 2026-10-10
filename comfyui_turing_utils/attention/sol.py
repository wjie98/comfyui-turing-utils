"""Sol semantic sparse attention execution."""

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
    inspect_sol_attention_call,
    prequantize_turing_sol_attention,
    prequantize_turing_sol_attention_from_qk,
    turing_sol_attention_from_prequantized,
    turing_sol_sparse_attention,
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
    bundled_sparse_available,
    bundled_w8a8_available,
    fused_qk_preprocessing_available,
    is_supported_attention_device,
    normalize_attention_backend,
    prequantize_turing_qk,
    preflight_bundled_sparse,
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
    _make_dense_prepared_executor,
    _make_dense_container_function,
    make_attention_override,
)


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
        min_sequence_tokens,
        prefix_policy,
        manual_prefix_tokens,
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
