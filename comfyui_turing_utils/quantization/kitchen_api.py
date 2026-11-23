"""Compatibility with Kitchen's CUDA activation and quantized GEMM interfaces."""

from __future__ import annotations

import torch


def backend_status(name: str) -> dict:
    import comfy_kitchen

    return comfy_kitchen.list_backends().get(name, {})


def require_convrot_cuda_api() -> None:
    from comfy_kitchen.backends import cuda

    required = (
        "convrot_w4a4_linear",
        "int8_linear",
        "quantize_int4_rowwise",
        "quantize_int4_rowwise_convrot64",
        "quantize_int8_rowwise_convrot64",
    )
    missing = [name for name in required if not callable(getattr(cuda, name, None))]
    if missing:
        raise RuntimeError(
            "The installed comfy-kitchen does not provide the Turing ConvRot API required by Turing Utils: "
            + ", ".join(missing)
        )


def has_w6_gemv() -> bool:
    try:
        from comfy_kitchen.backends import cuda
    except ImportError:
        return False

    return bool(getattr(cuda, "_W4A8_CHUNKED", False)) and callable(
        getattr(getattr(cuda, "_C", None), "w4a8_codebook_gemv", None)
    )


def apply_input_act(
    x: torch.Tensor,
    input_act: str | None,
    weight: torch.Tensor | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    from comfy_kitchen.backends._activations import apply_input_act as apply

    return apply(x, input_act, weight, eps)


def apply_residual(
    out: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    from comfy_kitchen.backends._activations import apply_residual as apply

    return apply(out, residual, scale)


def quantize_int8_convrot(x: torch.Tensor, group_size: int, *, swiglu: bool):
    from comfy_kitchen.backends import cuda

    if swiglu:
        return cuda.quantize_int8_rowwise_convrot64(x, group_size, input_act="swiglu")
    return cuda.quantize_int8_rowwise_convrot64(x, group_size)


def quantize_int8_convrot_staged(x: torch.Tensor, group_size: int):
    from comfy_kitchen.backends import cuda

    staged = getattr(cuda, "quantize_int8_convrot_staged", None)
    if not callable(staged):
        raise RuntimeError(
            "INT8 activation requires Kitchen staged ConvRot quantization "
            "when Kitchen's default shared-memory launch does not fit"
        )
    return staged(x, group_size)


def quantize_int4_convrot(x: torch.Tensor, group_size: int, *, staged: bool):
    from comfy_kitchen.backends import cuda

    if not staged:
        return cuda.quantize_int4_rowwise_convrot64(x, group_size)
    rotate = getattr(cuda, "rotate_int8_convrot_weight", None)
    if not callable(rotate):
        raise RuntimeError(
            "INT4 activation requires Kitchen grouped ConvRot rotation "
            "when Kitchen's default shared-memory launch does not fit"
        )
    return cuda.quantize_int4_rowwise(rotate(x, group_size))


def try_fused_int8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    avoid_global_workspace: bool,
) -> torch.Tensor | None:
    from comfy_kitchen.backends import cuda

    fused_linear = getattr(cuda, "_int8_linear_turing_quantized", None)
    if not callable(fused_linear):
        return None
    if not avoid_global_workspace:
        prefer_fused = getattr(cuda, "_prefer_turing_fused_int8", None)
        m, k = activation.shape
        if not callable(prefer_fused) or not prefer_fused(m, weight.shape[0], k):
            return None
    return fused_linear(
        activation, weight, activation_scale, weight_scale, bias, out_dtype
    )


def linear_from_int8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from comfy_kitchen.backends import cuda

    # Kitchen also uses this method for unpacked INT4, with per-channel scales.
    linear = getattr(cuda, "_int4_linear_via_int8_values", None)
    if not callable(linear):
        raise RuntimeError("W8A8 requires Kitchen quantized INT8 linear support")
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.expand(weight.shape[0]).contiguous()
    return linear(activation, weight, activation_scale, weight_scale, bias, out_dtype)


def try_cublas_int8_accumulator(
    activation: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    """Return the padded INT32 result for a caller-owned epilogue."""
    from comfy_kitchen.backends import cuda

    required = (
        "_cublas_int8_n_alignment",
        "_pad_2d_cols",
        "_pad_2d_rows",
        "_round_up",
        "_wrap_for_dlpack",
        "get_cublas_workspace",
    )
    gemm = getattr(getattr(cuda, "_C", None), "cublas_gemm_int8", None)
    if not callable(gemm) or not all(
        callable(getattr(cuda, name, None)) for name in required
    ):
        return None

    m, k = activation.shape
    padded_k = cuda._round_up(k, 16)
    padded_n = cuda._round_up(
        weight.shape[0], cuda._cublas_int8_n_alignment(activation)
    )
    cublas_x = cuda._pad_2d_cols(activation, padded_k)
    cublas_weight = cuda._pad_2d_rows(cuda._pad_2d_cols(weight, padded_k), padded_n)
    accumulator = torch.empty(
        (m, padded_n), dtype=torch.int32, device=activation.device
    )
    # Kitchen selects its workspace using the current device.
    with torch.cuda.device(activation.device):
        gemm(
            cuda._wrap_for_dlpack(cublas_x),
            cuda._wrap_for_dlpack(cublas_weight),
            cuda._wrap_for_dlpack(accumulator),
            cuda._wrap_for_dlpack(cuda.get_cublas_workspace()),
            torch.cuda.current_stream().cuda_stream,
        )
    return accumulator
