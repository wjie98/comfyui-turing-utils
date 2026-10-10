"""Shared prepared-attention transforms, validation and fallback bridges."""

from __future__ import annotations

from collections.abc import Callable
import torch
from ..profiling import CUDA_PHASE_PROFILER
from .layout import attention_semantic_layout
from .protocol import PreparedAttention
from .stable import _comfy_attention_function


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
