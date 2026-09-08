"""Attention backend selection, bundled Sage, and sparse patch APIs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from ..hardware import (
    is_supported_attention_device as _is_supported_attention_device,
    is_supported_turing_device,
)
from ..kernel_api import load_turing_sage
from ..log import get_logger
from ..runtime.capabilities import kernel_capabilities
from .protocol import QKTransformSpec


LOG = get_logger("attention")
SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
SUPPORTED_INPUT_DTYPES = (*SUPPORTED_KERNEL_DTYPES, torch.float32)
SPARSE_AUTO_MIN_SEQUENCE = 4096
SPARSE_ROUTING_THRESHOLD = 1.0
SPARSE_PREFIX_POLICY = "auto"
SPARSE_SKIPPED_RESIDUAL = "1x64"
SPARSE_REFERENCE_IMAGE = False
SPARSE_REFERENCE_VIDEO = True
SPARSE_REFERENCE_AUDIO = False
SPARSE_USE_W8A8 = True
SPARSE_DENSE_PREFIX_STEPS = 1
SPARSE_DENSE_SUFFIX_STEPS = 0
SPARSE_DENSE_PREFIX_LAYERS = 2
SPARSE_DENSE_SUFFIX_LAYERS = 0
SLA_SPARSITY_RATIO = 0.85
SLA_DENSE_PREFIX_STEPS = 0
SLA_DENSE_SUFFIX_STEPS = 0
SLA_DENSE_PREFIX_LAYERS = 0
SLA_DENSE_SUFFIX_LAYERS = 0
_PREFLIGHTED_DEVICES: set[int] = set()
_PREFLIGHTED_SPARSE_DEVICES: set[int] = set()
_PREFLIGHTED_SLA_DEVICES: set[int] = set()
_PREFLIGHTED_W8A8_DEVICES: set[int] = set()
_LOGGED_FP32_COMPAT = False
_LOGGED_TURING_KERNELS: set[tuple] = set()
_LOGGED_TURING_FALLBACKS: set[str] = set()
_LOGGED_SPARSE_KERNELS: set[tuple] = set()
_LOGGED_SPARSE_DENSE_REASONS: set[str] = set()


def is_supported_attention_device(device: torch.device) -> bool:
    return is_supported_turing_device(device) or _is_supported_attention_device(device)


@dataclasses.dataclass(frozen=True)
class AttentionBackend:
    option: str
    attention_function: str
    label: str
    install_hint: str | None = None
    aliases: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionCall:
    batch: int
    heads: int
    kv_heads: int
    head_dim: int
    query_tokens: int
    key_tokens: int
    tensor_layout: str
    input_dtype: torch.dtype
    skip_output_reshape: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PrequantizedAttentionCall:
    kernel_state: object
    call: AttentionCall


def inspect_turing_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    enable_gqa: bool = False,
    low_precision_attention: bool = True,
    is_causal: bool = False,
    kernel: str = "sage",
    require_long_sequence: bool = False,
) -> tuple[AttentionCall | None, str | None]:
    if kernel not in {"sage", "w8a8", "sol"}:
        raise ValueError(f"unsupported Turing attention kernel: {kernel}")
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        return None, "Q/K/V are not tensors"
    # W8A8 and Sol share the bundled sm75+ integer core.  Architecture-specific
    # MMA/copy variants are selected by the compiled cubin, not by Python.
    device_supported = (
        is_supported_attention_device(q.device)
        if kernel in {"sol", "w8a8"}
        else is_supported_turing_device(q.device)
    )
    if not device_supported:
        return None, "Q/K/V are not on a supported CUDA Tensor Core GPU"
    if mask is not None:
        return None, "an attention mask was supplied"
    if not low_precision_attention:
        return None, "low_precision_attention=False"
    if kernel == "sol" and is_causal:
        return None, f"causal attention is not supported by Turing {kernel}"
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return None, "Q/K/V dtypes do not match"
    if q.dtype not in SUPPORTED_INPUT_DTYPES:
        return None, f"Q/K/V dtype {q.dtype} is unsupported"
    if q.device != k.device or q.device != v.device:
        return None, "Q/K/V devices do not match"

    heads = int(heads)
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or heads <= 0 or q.shape[1] != heads:
            return None, "skip_reshape Q/K/V layout is incompatible"
        batch, _, query_tokens, head_dim = q.shape
        kv_heads = k.shape[1]
        key_tokens = k.shape[2]
        tensor_layout = "HND"
        if (
            k.shape[0] != batch
            or v.shape[0] != batch
            or k.shape != v.shape
            or k.shape[-1] != head_dim
            or kv_heads <= 0
            or heads % kv_heads != 0
        ):
            return None, "Q/K/V shapes or head counts are incompatible"
    else:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or heads <= 0 or q.shape[-1] % heads:
            return None, "unreshaped Q/K/V layout is incompatible"
        batch = q.shape[0]
        head_dim = q.shape[-1] // heads
        if head_dim <= 0 or k.shape[0] != batch or v.shape[0] != batch or k.shape != v.shape:
            return None, "unreshaped Q/K/V shapes are incompatible"
        kv_heads = k.shape[-1] // head_dim if enable_gqa else heads
        if kv_heads <= 0 or k.shape[-1] != kv_heads * head_dim or heads % kv_heads:
            return None, "unreshaped K/V head counts are incompatible"
        query_tokens = q.shape[1]
        key_tokens = k.shape[1]
        tensor_layout = "NHD"

    if query_tokens <= 0 or key_tokens <= 0:
        return None, "empty Q/K sequences are unsupported"
    if not 0 < head_dim <= 128:
        return None, f"head_dim={head_dim} is unsupported by Turing {kernel}"
    if require_long_sequence and (query_tokens < 64 or key_tokens < 64):
        return None, "split attention requires Q/K sequences of at least 64 tokens"
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return None, "the last Q/K/V dimension is not contiguous"
    return AttentionCall(
        batch=batch,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        tensor_layout=tensor_layout,
        input_dtype=q.dtype,
        skip_output_reshape=bool(skip_output_reshape),
    ), None


def normalize_turing_attention_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    call: AttentionCall,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if call.tensor_layout == "NHD":
        q = q.reshape(call.batch, call.query_tokens, call.heads, call.head_dim)
        k = k.reshape(call.batch, call.key_tokens, call.kv_heads, call.head_dim)
        v = v.reshape(call.batch, call.key_tokens, call.kv_heads, call.head_dim)
    if call.input_dtype == torch.float32:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    return q, k, v


def finish_turing_attention_output(output: torch.Tensor, call: AttentionCall) -> torch.Tensor:
    if call.tensor_layout == "HND":
        result = output if call.skip_output_reshape else output.transpose(1, 2).reshape(
            call.batch, -1, call.heads * call.head_dim
        )
    elif call.skip_output_reshape:
        result = output.transpose(1, 2)
    else:
        result = output.reshape(call.batch, -1, call.heads * call.head_dim)
    return result.to(call.input_dtype) if call.input_dtype == torch.float32 else result


_BACKENDS: dict[str, AttentionBackend] = {}
_ALIASES: dict[str, str] = {}


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def register_attention_backend(backend: AttentionBackend) -> None:
    if backend.option in _BACKENDS:
        raise ValueError(f"duplicate attention backend option: {backend.option}")
    normalized_aliases = {
        _normalize_key(alias) for alias in (backend.option, backend.label, *backend.aliases)
    }
    collisions = {
        alias: _ALIASES[alias] for alias in normalized_aliases if alias in _ALIASES
    }
    if collisions:
        details = ", ".join(
            f"{alias!r} is already owned by {option!r}"
            for alias, option in sorted(collisions.items())
        )
        raise ValueError(f"attention backend alias collision: {details}")
    _BACKENDS[backend.option] = backend
    for alias in normalized_aliases:
        _ALIASES[alias] = backend.option


register_attention_backend(
    AttentionBackend(
        option="w8a8",
        attention_function="comfy_kitchen_int8",
        label="w8a8",
        install_hint="Update ComfyUI and comfy-kitchen for INT8 attention support.",
        aliases=(
            "int8_attention",
            "turing_w8a8",
        ),
    )
)
register_attention_backend(
    AttentionBackend(
        option="sage",
        attention_function="sage",
        label="sage",
        install_hint="Install sageattention in the ComfyUI Python environment.",
        aliases=(
            "sage_attn",
            "sage_",
            "sageattention",
            "sage_attention",
            "sage_hybrid",
            "turing_sage",
            "turing_sage_hybrid",
        ),
    )
)
register_attention_backend(
    AttentionBackend(
        option="sdpa",
        attention_function="pytorch",
        label="sdpa",
        aliases=("pytorch", "torch", "torch_sdpa"),
    )
)

def attention_backend_choices() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def normalize_attention_backend(value: str | None) -> str:
    if value is None:
        return "w8a8"
    option = _ALIASES.get(_normalize_key(value))
    if option is None:
        raise ValueError(
            f"Unsupported Turing Utils attention backend {value!r}. "
            f"Supported options: {', '.join(attention_backend_choices())}"
        )
    return option


def _comfy_attention_function(name: str) -> Callable | None:
    from comfy.ldm.modules import attention as comfy_attention

    return comfy_attention.get_attention_function(name, None)


def _resolve_attention_function(backend: AttentionBackend) -> Callable:
    function = _comfy_attention_function(backend.attention_function)
    if function is not None:
        return function
    message = (
        f"Turing Utils attention backend {backend.option!r} requires ComfyUI attention "
        f"function {backend.attention_function!r}, but it is not available."
    )
    if backend.install_hint:
        message = f"{message} {backend.install_hint}"
    raise RuntimeError(message)


def _select_attention_backend(option: str) -> tuple[AttentionBackend, Callable]:
    option = normalize_attention_backend(option)
    backend = _BACKENDS[option]
    return backend, _resolve_attention_function(backend)


def bundled_available() -> bool:
    return kernel_capabilities().supports("stable_sage").supported


def bundled_sparse_available() -> bool:
    return kernel_capabilities().supports("sol").supported


def bundled_sla_available() -> bool:
    return kernel_capabilities().supports("sla").supported


def bundled_w8a8_available() -> bool:
    return kernel_capabilities().supports("dense_w8a8").supported


def _sageattn(*args, **kwargs):
    if kwargs.pop("smooth_k", False):
        raise ValueError("the production Turing Sage backend does not enable smoothing")
    return load_turing_sage().sageattn_compiled(*args, **kwargs)


def _w8a8attn(*args, **kwargs):
    return load_turing_sage().w8a8attn_compiled(*args, **kwargs)


def _sol_sparse_sageattn(*args, **kwargs):
    return load_turing_sage().sol_sparse_sageattn(*args, **kwargs)


def split_prequantization_available() -> bool:
    return kernel_capabilities().supports("split_prequantization").supported


def fused_qk_preprocessing_available() -> bool:
    return kernel_capabilities().supports("fused_qk").supported


def reusable_k_anchor_available() -> bool:
    return kernel_capabilities().supports("reusable_k_anchor").supported


def precompute_turing_k_anchor(
    key: torch.Tensor,
    spec: QKTransformSpec,
):
    return load_turing_sage().precompute_rms_rope_k_anchor(
        key,
        spec.key_norm_weight,
        spec.key_freqs,
        epsilon=spec.epsilon,
        rot_dim=spec.rot_dim,
        tensor_layout="HND",
        norm_scope=spec.norm_scope,
        split_half=spec.split_half,
    )


def prequantize_turing_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    spec: QKTransformSpec,
    *,
    kernel: str,
    k_anchor=None,
    qk_output=None,
    key_source_indices: torch.Tensor | None = None,
):
    if kernel not in {"sage", "w8a8", "sol", "sla"}:
        raise ValueError(f"unsupported fused Q/K target: {kernel}")
    rotate_qk = kernel in {"w8a8", "sol", "sla"}
    stabilize_k = rotate_qk
    return load_turing_sage().prequantize_rms_rope_qk(
        q,
        k,
        spec.query_norm_weight,
        spec.key_norm_weight,
        spec.freqs,
        key_freqs=spec.key_freqs,
        epsilon=spec.epsilon,
        rot_dim=spec.rot_dim,
        tensor_layout="HND",
        norm_scope=spec.norm_scope,
        split_half=spec.split_half,
        rotate_qk=rotate_qk,
        stabilize_k=stabilize_k,
        k_anchor=k_anchor,
        qk_output=qk_output,
        key_source_indices=key_source_indices,
    )


def prequantize_turing_attention_from_qk(
    qk,
    value: torch.Tensor,
    call: AttentionCall,
    *,
    kernel: str,
    scale: float | None,
    is_causal: bool = False,
    value_source_indices: torch.Tensor | None = None,
) -> PrequantizedAttentionCall:
    turing_sage = load_turing_sage()
    if kernel == "sage":
        state = turing_sage.prequantize_sageattn_from_qk(
            qk,
            value,
            is_causal=bool(is_causal),
            sm_scale=scale,
        )
    elif kernel == "w8a8":
        state = turing_sage.prequantize_sol_sageattn_from_qk(
            qk,
            value,
            sm_scale=scale,
            threshold_sigma=0.0,
            residual_subblocks=1,
            use_w8a8=True,
            force_dense=True,
            key_tile_tokens=0,
            is_causal=bool(is_causal),
            value_source_indices=value_source_indices,
        )
    else:
        raise ValueError(f"unsupported split Turing attention kernel: {kernel}")
    return PrequantizedAttentionCall(state, call)


def prequantize_turing_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    call: AttentionCall,
    *,
    kernel: str,
    scale: float | None,
    is_causal: bool = False,
) -> PrequantizedAttentionCall:
    q, k, v = normalize_turing_attention_tensors(q, k, v, call)
    turing_sage = load_turing_sage()
    if kernel == "sage":
        state = turing_sage.prequantize_sageattn(
            q,
            k,
            v,
            tensor_layout=call.tensor_layout,
            is_causal=bool(is_causal),
            sm_scale=scale,
            smooth_k=False,
        )
    elif kernel == "w8a8":
        if call.tensor_layout == "NHD":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
        state = turing_sage.prequantize_sol_sageattn(
            q,
            k,
            v,
            tensor_layout="HND",
            sm_scale=scale,
            threshold_sigma=0.0,
            residual_subblocks=1,
            use_w8a8=True,
            force_dense=True,
            key_tile_tokens=0,
            rotate_qk=True,
            stabilize_k=True,
            is_causal=bool(is_causal),
        )
    else:
        raise ValueError(f"unsupported split Turing attention kernel: {kernel}")
    return PrequantizedAttentionCall(state, call)


def turing_attention_from_prequantized(
    quantized: PrequantizedAttentionCall,
    *,
    kernel: str,
) -> torch.Tensor:
    turing_sage = load_turing_sage()
    if kernel == "sage":
        output = turing_sage.sageattn_from_prequantized(quantized.kernel_state)
    elif kernel == "w8a8":
        output = turing_sage.sol_sparse_sageattn_from_prequantized(
            quantized.kernel_state
        )
        if quantized.call.tensor_layout == "NHD":
            output = output.transpose(1, 2)
    else:
        raise ValueError(f"unsupported split Turing attention kernel: {kernel}")
    return finish_turing_attention_output(output, quantized.call)


def preflight_bundled(device: torch.device) -> None:
    if not is_supported_turing_device(device):
        raise RuntimeError(f"unsupported Turing device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return
    load_turing_sage().preflight(device)
    _PREFLIGHTED_DEVICES.add(index)


def preflight_bundled_sparse(device: torch.device) -> None:
    if not is_supported_attention_device(device):
        raise RuntimeError(f"unsupported Sol attention device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_SPARSE_DEVICES:
        return
    load_turing_sage().preflight_sparse(device)
    _PREFLIGHTED_SPARSE_DEVICES.add(index)


def preflight_bundled_sla(device: torch.device) -> None:
    if not is_supported_attention_device(device):
        raise RuntimeError(f"unsupported SLA attention device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_SLA_DEVICES:
        return
    load_turing_sage().preflight_sla(device)
    _PREFLIGHTED_SLA_DEVICES.add(index)


def preflight_bundled_w8a8(device: torch.device) -> None:
    if not is_supported_attention_device(device):
        raise RuntimeError(f"unsupported integer attention device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_W8A8_DEVICES:
        return
    load_turing_sage().preflight_w8a8(device)
    _PREFLIGHTED_W8A8_DEVICES.add(index)


def _reshape_qkv(q, k, v, heads: int, enable_gqa: bool):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("unreshaped Q/K/V must be three-dimensional")
    batch = q.shape[0]
    if heads <= 0 or q.shape[-1] % heads != 0:
        raise ValueError("Q inner dimension must be divisible by the head count")
    head_dim = q.shape[-1] // heads
    kv_heads = k.shape[-1] // head_dim if enable_gqa else heads
    if kv_heads <= 0 or k.shape[-1] != kv_heads * head_dim or v.shape[-1] != kv_heads * head_dim:
        raise ValueError("K/V inner dimensions do not match the Q head dimension")
    q = q.reshape(batch, -1, heads, head_dim)
    k = k.reshape(batch, -1, kv_heads, head_dim)
    v = v.reshape(batch, -1, kv_heads, head_dim)
    return q, k, v, batch, head_dim


def _bundled_fallback(
    fallback: Callable,
    reason: str,
    fallback_args: tuple,
    fallback_kwargs: dict,
):
    if reason not in _LOGGED_TURING_FALLBACKS:
        LOG.warning(
            "Bundled Turing Sage is falling back to ComfyUI attention (%s); "
            "this message is emitted once per reason",
            reason,
        )
        _LOGGED_TURING_FALLBACKS.add(reason)
    return fallback(*fallback_args, **fallback_kwargs)


def turing_sage_attention(
    fallback: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask=None,
    attn_precision=None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs,
) -> torch.Tensor:
    global _LOGGED_FP32_COMPAT

    turing_kernel = kwargs.pop("_turing_kernel", "sage")
    if turing_kernel not in {"sage", "w8a8"}:
        raise ValueError(f"unsupported bundled Turing attention kernel: {turing_kernel}")

    fallback_args = (q, k, v, heads)
    fallback_kwargs = {
        "mask": mask,
        "attn_precision": attn_precision,
        "skip_reshape": skip_reshape,
        "skip_output_reshape": skip_output_reshape,
        **kwargs,
    }
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
        kernel=turing_kernel,
    )
    if reason is not None:
        return _bundled_fallback(
            fallback,
            reason,
            fallback_args,
            fallback_kwargs,
        )
    input_dtype = call.input_dtype
    head_dim = call.head_dim
    tensor_layout = call.tensor_layout
    q, k, v = normalize_turing_attention_tensors(q, k, v, call)

    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    sequence_axis = 2 if tensor_layout == "HND" else 1
    head_axis = 1 if tensor_layout == "HND" else 2
    kernel_key = (
        turing_kernel,
        index,
        input_dtype,
        tensor_layout,
        head_dim,
        q.shape[sequence_axis],
        k.shape[sequence_axis],
        q.shape[head_axis],
        k.shape[head_axis],
    )
    if kernel_key not in _LOGGED_TURING_KERNELS:
        message = (
            "Bundled Turing W8A8 active: device=%s dtype=%s layout=%s "
            "Q=%s K=%s V=%s heads=%d"
            if turing_kernel == "w8a8"
            else "Bundled Turing Sage active: device=%s dtype=%s layout=%s "
            "Q=%s K=%s V=%s heads=%d"
        )
        LOG.info(
            message,
            q.device,
            input_dtype,
            tensor_layout,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            heads,
        )
        _LOGGED_TURING_KERNELS.add(kernel_key)

    if input_dtype == torch.float32:
        if not _LOGGED_FP32_COMPAT:
            LOG.info(
                "Turing attention FP32 compatibility uses BF16 Q/K/V storage and restores FP32 output"
            )
            _LOGGED_FP32_COMPAT = True
    attention_kernel = _w8a8attn if turing_kernel == "w8a8" else _sageattn
    output = attention_kernel(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        sm_scale=kwargs.get("scale"),
        **(
            {
                "key_tile_tokens": 0,
                "rotate_qk": True,
                "stabilize_k": True,
                "is_causal": bool(kwargs.get("is_causal", False)),
            }
            if turing_kernel == "w8a8"
            else {
                "is_causal": bool(kwargs.get("is_causal", False)),
                "smooth_k": False,
            }
        ),
    )
    return finish_turing_attention_output(output, call)


def turing_w8a8_attention(*args, **kwargs) -> torch.Tensor:
    kwargs["_turing_kernel"] = "w8a8"
    return turing_sage_attention(*args, **kwargs)
