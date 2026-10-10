"""ConvRot activation quantization and GEMM dispatch for sm75 and newer."""

from __future__ import annotations

import torch
from .formats import grouped_weight_geometry
from ..hardware import device_capabilities, is_supported_tensor_core_device
from ..log import get_logger
from .capabilities import kernel_available as _kernel_available, kernel_op as _kernel_op
from .workspace import codebook_w4a8_workspace_bytes, int8_workspace_bytes


LOG = get_logger("quantization")
KITCHEN_DEFAULT_SHARED_MEMORY_LIMIT = 48 * 1024
TURING_OPTIN_SHARED_MEMORY_LIMIT = 64 * 1024
# Above this point a full MxN INT32 accumulator is more expensive than the
# fixed-workspace fused Turing path. This is a dispatch threshold, never an
# input-size limit.
TURING_INT8_GLOBAL_WORKSPACE_LIMIT = 64 * 1024 * 1024
TURING_CODEBOOK_W4A8_CHUNK_ROWS = 4096


def turing_int8_workspace_bytes(rows: int, output_channels: int) -> int:
    """Return the global INT32 workspace used by the selected W8A8 path."""
    return int8_workspace_bytes(
        rows,
        output_channels,
        global_workspace_limit=TURING_INT8_GLOBAL_WORKSPACE_LIMIT,
    )


def turing_codebook_w4a8_workspace_bytes(
    input_channels: int,
    output_channels: int,
) -> int:
    """Return the bounded decoded-weight workspace used by grouped W4A8."""
    return codebook_w4a8_workspace_bytes(
        input_channels,
        output_channels,
        chunk_rows=TURING_CODEBOOK_W4A8_CHUNK_ROWS,
    )


def convrot_swiglu_channel_sharding_available() -> bool:
    """Return whether exact two-pass channel streaming is in the real ABI."""
    return _kernel_available("turing_swiglu_int8_convrot_quantize_scaled")


def convrot_swiglu_half_width_available() -> bool:
    """Return whether single-pass, lossless half-width FFN staging is available."""
    return all(
        _kernel_available(name)
        for name in (
            "turing_swiglu_convrot_shard_inplace",
            "turing_int8_convrot_quantize_from_partials",
        )
    )


def _convrot_int8_shared_memory_bytes(rows: int, hidden_size: int) -> int:
    if rows == 1:
        block_threads = 512
    elif hidden_size == 256:
        block_threads = 64
    elif hidden_size == 2560:
        block_threads = 640
    elif hidden_size == 6144:
        block_threads = 768
    else:
        block_threads = 1024
    groups_in_flight = block_threads // 64
    return (hidden_size + groups_in_flight * 2 * 256) * 4


def _convrot_int8_bf16_rowbuffer_fits(
    hidden_size: int,
    device: torch.device | str | None = None,
) -> bool:
    shared_memory_limit = TURING_OPTIN_SHARED_MEMORY_LIMIT
    if device is not None:
        detected_limit = device_capabilities(device).optin_shared_memory_per_block
        if detected_limit:
            shared_memory_limit = detected_limit
    for block_threads in (1024, 768, 512):
        groups_in_flight = block_threads // 64
        dynamic_bytes = hidden_size * 2 + groups_in_flight * 2 * 256 * 4
        # ptxas reserves three additional aligned words around the static
        # warp-reduction arrays (80/112/144 bytes for 512/768/1024 threads).
        static_bytes = (block_threads // 32 + 4) * 4
        if dynamic_bytes + static_bytes <= shared_memory_limit:
            return True
    return False


def _convrot_int4_shared_memory_bytes(
    rows: int, hidden_size: int, element_size: int
) -> int:
    if rows != 1 and hidden_size <= 4096:
        block_threads = 256
        scratch_buffers = 2
    elif rows == 1:
        block_threads = 512
        scratch_buffers = 2
    elif hidden_size == 15360:
        block_threads = 640
        scratch_buffers = 1
    else:
        block_threads = 1024
        scratch_buffers = 2
    groups_in_flight = block_threads // 64
    return (hidden_size + groups_in_flight * scratch_buffers * 256) * element_size


def _quantize_turing_int8_activation(
    x2d: torch.Tensor,
    group_size: int,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
):
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if x2d.dtype == torch.float16:
        return _kernel_op("turing_fp16_int8_convrot_quantize")(
            x2d, group_size, input_act, input_act_weight, input_act_eps
        )
    if input_act not in (None, "none", "swiglu", "gelu_tanh"):
        raise ValueError(f"unsupported fused INT8 activation: {input_act!r}")
    if input_act == "swiglu" and x2d.shape[1] % 2:
        raise ValueError("SwiGLU input width must be even")
    hidden_size = x2d.shape[1] // 2 if input_act == "swiglu" else x2d.shape[1]
    if input_act == "gelu_tanh":
        if (
            x2d.dtype == torch.bfloat16
            and is_supported_tensor_core_device(x2d.device)
            and _convrot_int8_bf16_rowbuffer_fits(hidden_size, x2d.device)
            and _kernel_available("turing_bf16_gelu_int8_convrot_quantize")
        ):
            return _kernel_op("turing_bf16_gelu_int8_convrot_quantize")(x2d, group_size)
        if not _kernel_available("turing_gelu_int8_convrot_quantize"):
            raise RuntimeError(
                "W8A8 GELU requires an updated comfyui-turing-utils-kernel; "
                "reinstall the kernel package"
            )
        return _kernel_op("turing_gelu_int8_convrot_quantize")(x2d, group_size)
    if (
        x2d.dtype == torch.bfloat16
        and is_supported_tensor_core_device(x2d.device)
        and _convrot_int8_bf16_rowbuffer_fits(hidden_size, x2d.device)
        and _kernel_available("turing_bf16_int8_convrot_quantize")
    ):
        return _kernel_op("turing_bf16_int8_convrot_quantize")(
            x2d,
            group_size,
            swiglu=input_act == "swiglu",
        )
    requested_shared = _convrot_int8_shared_memory_bytes(x2d.shape[0], hidden_size)
    if requested_shared < KITCHEN_DEFAULT_SHARED_MEMORY_LIMIT:
        if input_act == "swiglu":
            return kitchen_cuda.quantize_int8_rowwise_convrot64(
                x2d, group_size, input_act="swiglu"
            )
        return kitchen_cuda.quantize_int8_rowwise_convrot64(x2d, group_size)
    if input_act == "swiglu":
        if not _kernel_available("turing_swiglu_int8_convrot_quantize"):
            raise RuntimeError(
                "W8A8 SwiGLU requires an updated comfyui-turing-utils-kernel; "
                "reinstall the kernel package"
            )
        return _kernel_op("turing_swiglu_int8_convrot_quantize")(x2d, group_size)
    staged = getattr(kitchen_cuda, "quantize_int8_convrot_staged", None)
    if staged is None:
        raise RuntimeError(
            "INT8 activation requires Kitchen staged ConvRot quantization "
            "when Kitchen's default shared-memory launch does not fit"
        )
    return staged(x2d, group_size)


def quantize_convrot_int8_activation(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a two-dimensional BF16 activation for reusable W8 GEMMs."""
    if x.ndim != 2:
        raise ValueError("ConvRot W8 activation must be two-dimensional")
    if x.dtype is not torch.bfloat16:
        raise ValueError("ConvRot W8 activation must use BF16 storage")
    if not x.is_cuda:
        raise ValueError("ConvRot W8 activation must be on CUDA")
    return _quantize_turing_int8_activation(x.contiguous(), int(group_size))


def quantize_convrot_swiglu_activation(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 ``[gate, up]`` tile with fused SwiGLU+ConvRot."""
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError("SwiGLU ConvRot input must be 2D [M, 2K]")
    if x.dtype is not torch.bfloat16:
        raise ValueError("SwiGLU ConvRot input must use BF16 storage")
    if not x.is_cuda:
        raise ValueError("SwiGLU ConvRot input must be on CUDA")
    operation = _kernel_op("turing_bf16_int8_convrot_quantize")
    return operation(x.contiguous(), int(group_size), swiglu=True)


def quantize_convrot_swiglu_with_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 256,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if not convrot_swiglu_channel_sharding_available():
        raise RuntimeError(
            "exact FFN channel sharding requires an updated "
            "comfyui-turing-utils-kernel; reinstall the kernel package"
        )
    x = x.contiguous()
    scale = scale.reshape(-1).contiguous()
    if output is not None:
        if (
            output.shape != (x.shape[0], x.shape[1] // 2)
            or output.dtype is not torch.int8
            or output.device != x.device
            or output.stride(1) != 1
            or output.stride(0) < output.shape[1]
        ):
            raise ValueError(
                "scaled SwiGLU direct output shape, dtype, device, or stride "
                "is incompatible"
            )
        if not _kernel_available("turing_swiglu_int8_convrot_quantize_scaled_out"):
            temporary = quantize_convrot_swiglu_with_scale(x, scale, group_size)
            output.copy_(temporary)
            return output
        operation = _kernel_op("turing_swiglu_int8_convrot_quantize_scaled_out")
        operation(x, scale, output, int(group_size))
        return output
    operation = _kernel_op("turing_swiglu_int8_convrot_quantize_scaled")
    return operation(x, scale, int(group_size))


def rotate_convrot_swiglu_shard_inplace(
    gate: torch.Tensor,
    up: torch.Tensor,
    partial_absmax: torch.Tensor,
    channel_offset: int,
) -> None:
    """Rotate one aligned SwiGLU shard into the full-width gate buffer."""
    operation = _kernel_op("turing_swiglu_convrot_shard_inplace")
    operation(gate, up, partial_absmax, int(channel_offset))


def quantize_convrot_from_partials(
    rotated: torch.Tensor,
    partial_absmax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce sharded ConvRot maxima and quantize with one whole-row scale."""
    operation = _kernel_op("turing_int8_convrot_quantize_from_partials")
    return operation(rotated, partial_absmax)


def _quantize_turing_int4_activation(
    x2d: torch.Tensor,
    group_size: int,
    input_act: str | None = None,
):
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if input_act not in (None, "none", "swiglu", "gelu_tanh"):
        raise ValueError(f"unsupported fused INT4 activation: {input_act!r}")
    if input_act == "gelu_tanh":
        hidden_size = x2d.shape[1]
        if (
            x2d.dtype == torch.bfloat16
            and is_supported_tensor_core_device(x2d.device)
            and _convrot_int8_bf16_rowbuffer_fits(hidden_size, x2d.device)
            and _kernel_available("turing_bf16_gelu_int4_convrot_quantize")
        ):
            return _kernel_op("turing_bf16_gelu_int4_convrot_quantize")(x2d, group_size)
        if not _kernel_available("turing_gelu_int4_convrot_quantize"):
            raise RuntimeError(
                "W4A4 GELU requires an updated comfyui-turing-utils-kernel; "
                "reinstall the kernel package"
            )
        return _kernel_op("turing_gelu_int4_convrot_quantize")(x2d, group_size)
    if input_act == "swiglu":
        hidden_size = x2d.shape[1] // 2
        if x2d.shape[1] % 2 != 0:
            raise ValueError("SwiGLU input width must be even")
        if (
            x2d.dtype == torch.bfloat16
            and is_supported_tensor_core_device(x2d.device)
            and _convrot_int8_bf16_rowbuffer_fits(hidden_size, x2d.device)
            and _kernel_available("turing_bf16_int4_convrot_quantize")
        ):
            return _kernel_op("turing_bf16_int4_convrot_quantize")(
                x2d,
                group_size,
                swiglu=True,
            )
        if not _kernel_available("turing_swiglu_int4_convrot_quantize"):
            raise RuntimeError(
                "W4A4 SwiGLU requires an updated comfyui-turing-utils-kernel; reinstall the kernel package"
            )
        return _kernel_op("turing_swiglu_int4_convrot_quantize")(x2d, group_size)

    if (
        x2d.dtype == torch.bfloat16
        and is_supported_tensor_core_device(x2d.device)
        and _convrot_int8_bf16_rowbuffer_fits(x2d.shape[1], x2d.device)
        and _kernel_available("turing_bf16_int4_convrot_quantize")
    ):
        return _kernel_op("turing_bf16_int4_convrot_quantize")(
            x2d,
            group_size,
            swiglu=False,
        )

    requested_shared = _convrot_int4_shared_memory_bytes(
        x2d.shape[0], x2d.shape[1], x2d.element_size()
    )
    if requested_shared < KITCHEN_DEFAULT_SHARED_MEMORY_LIMIT:
        return kitchen_cuda.quantize_int4_rowwise_convrot64(x2d, group_size)
    rotate = getattr(kitchen_cuda, "rotate_int8_convrot_weight", None)
    if rotate is None:
        raise RuntimeError(
            "INT4 activation requires Kitchen grouped ConvRot rotation "
            "when Kitchen's default shared-memory launch does not fit"
        )
    rotated = rotate(x2d, group_size)
    return kitchen_cuda.quantize_int4_rowwise(rotated)


def _turing_cublas_int8_bf16(
    qactivation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Run Kitchen's cuBLAS fallback with the bundled BF16 epilogue."""
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if not _kernel_available("turing_dequantize_int8_bf16"):
        return None
    turing_dequantize_int8_bf16 = _kernel_op("turing_dequantize_int8_bf16")

    required = (
        "_C",
        "_cublas_int8_n_alignment",
        "_pad_2d_cols",
        "_pad_2d_rows",
        "_round_up",
        "_wrap_for_dlpack",
        "get_cublas_workspace",
    )
    if not all(hasattr(kitchen_cuda, name) for name in required):
        return None
    if not hasattr(kitchen_cuda._C, "cublas_gemm_int8"):
        return None

    m, k = qactivation.shape
    n = weight.shape[0]
    padded_k = kitchen_cuda._round_up(k, 16)
    padded_n = kitchen_cuda._round_up(
        n, kitchen_cuda._cublas_int8_n_alignment(qactivation)
    )
    cublas_x = kitchen_cuda._pad_2d_cols(qactivation, padded_k)
    cublas_weight = kitchen_cuda._pad_2d_rows(
        kitchen_cuda._pad_2d_cols(weight, padded_k), padded_n
    )
    accumulator = torch.empty(
        (m, padded_n), dtype=torch.int32, device=qactivation.device
    )
    stream_ptr = torch.cuda.current_stream(qactivation.device).cuda_stream
    kitchen_cuda._C.cublas_gemm_int8(
        kitchen_cuda._wrap_for_dlpack(cublas_x),
        kitchen_cuda._wrap_for_dlpack(cublas_weight),
        kitchen_cuda._wrap_for_dlpack(accumulator),
        kitchen_cuda._wrap_for_dlpack(kitchen_cuda.get_cublas_workspace()),
        stream_ptr,
    )
    return turing_dequantize_int8_bf16(
        accumulator,
        activation_scale,
        weight_scale,
        output_columns=n,
    )


def _turing_int8_gemm(
    qactivation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep scalar scales in fused GEMMs and use the fast no-bias BF16 epilogue."""
    from comfy_kitchen.backends import cuda as kitchen_cuda

    m, k = qactivation.shape
    n = weight.shape[0]
    activation_scale = (
        activation_scale.to(device=qactivation.device, dtype=torch.float32)
        .reshape(-1)
        .contiguous()
    )
    weight_scale = (
        weight_scale.to(device=qactivation.device, dtype=torch.float32)
        .reshape(-1)
        .contiguous()
    )
    if activation_scale.numel() != m:
        raise ValueError(
            f"W8A8 activation scale must have {m} values, "
            f"got {activation_scale.numel()}"
        )
    if weight_scale.numel() not in (1, n):
        raise ValueError(
            f"W8A8 weight scale must be scalar or have {n} values, "
            f"got {weight_scale.numel()}"
        )

    if output_dtype == torch.float16 and _kernel_available("turing_fp16_int8_linear"):
        if output is not None and (
            output.shape != (m, n)
            or output.dtype != output_dtype
            or output.device != qactivation.device
        ):
            raise ValueError(
                "W8A8 direct output shape, dtype, or device is incompatible"
            )
        expanded_weight_scale = weight_scale
        if expanded_weight_scale.numel() == 1:
            expanded_weight_scale = expanded_weight_scale.expand(n).contiguous()
        result = _kernel_op("turing_fp16_int8_linear")(
            qactivation, weight, activation_scale, expanded_weight_scale, bias
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    if output is not None:
        if (
            output_dtype != torch.bfloat16
            or output.shape != (m, n)
            or output.dtype != torch.bfloat16
            or output.device != qactivation.device
            or output.stride(1) != 1
            or output.stride(0) < n
        ):
            raise ValueError(
                "W8A8 direct output shape, dtype, device, or stride is incompatible"
            )
        expanded_weight_scale = weight_scale
        if expanded_weight_scale.numel() == 1:
            expanded_weight_scale = expanded_weight_scale.expand(n).contiguous()
        if not _kernel_available("turing_int8_linear_out"):
            temporary = _turing_int8_gemm(
                qactivation,
                weight,
                activation_scale,
                weight_scale,
                bias,
                output_dtype,
            )
            output.copy_(temporary)
            return output
        _kernel_op("turing_int8_linear_out")(
            qactivation,
            weight,
            activation_scale,
            expanded_weight_scale,
            output,
            bias,
        )
        return output

    avoid_global_workspace = m * n * 4 >= TURING_INT8_GLOBAL_WORKSPACE_LIMIT
    if (
        output_dtype == torch.bfloat16
        and is_supported_tensor_core_device(qactivation.device)
        and k % 16 == 0
        and n % 8 == 0
        and _kernel_available("turing_int8_linear")
    ):
        expanded_weight_scale = weight_scale
        if expanded_weight_scale.numel() == 1:
            expanded_weight_scale = expanded_weight_scale.expand(n).contiguous()
        return _kernel_op("turing_int8_linear")(
            qactivation,
            weight,
            activation_scale,
            expanded_weight_scale,
            bias,
        )

    prefer_fused = getattr(kitchen_cuda, "_prefer_turing_fused_int8", None)
    fused_linear = getattr(kitchen_cuda, "_int8_linear_turing_quantized", None)
    if callable(fused_linear) and (
        avoid_global_workspace or (callable(prefer_fused) and prefer_fused(m, n, k))
    ):
        output = fused_linear(
            qactivation,
            weight,
            activation_scale,
            weight_scale,
            bias,
            output_dtype,
        )
        if output is not None:
            return output

    if bias is None and output_dtype == torch.bfloat16:
        output = _turing_cublas_int8_bf16(
            qactivation,
            weight,
            activation_scale,
            weight_scale,
        )
        if output is not None:
            return output

    quantized_linear = getattr(kitchen_cuda, "_int4_linear_via_int8_values", None)
    if quantized_linear is None:
        raise RuntimeError("W8A8 requires Kitchen quantized INT8 linear support")
    expanded_weight_scale = weight_scale
    if expanded_weight_scale.numel() == 1:
        expanded_weight_scale = expanded_weight_scale.expand(n).contiguous()
    return quantized_linear(
        qactivation,
        weight,
        activation_scale,
        expanded_weight_scale,
        bias,
        output_dtype,
    )


def int8_linear_from_quantized(
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run W8 GEMM from an activation quantized once by the caller."""
    if qactivation.ndim != 2 or weight.ndim != 2:
        raise ValueError("quantized W8 linear expects two-dimensional tensors")
    if qactivation.dtype is not torch.int8 or weight.dtype is not torch.int8:
        raise ValueError("quantized W8 linear expects INT8 activation and weight")
    if qactivation.shape[1] != weight.shape[1]:
        raise ValueError("quantized W8 activation and weight widths differ")
    return _turing_int8_gemm(
        qactivation,
        weight.contiguous(),
        activation_scale,
        weight_scale,
        bias,
        out_dtype,
        output,
    )


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    from comfy_kitchen.backends import cuda as kitchen_cuda
    from comfy_kitchen.backends._activations import apply_input_act

    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or not is_supported_tensor_core_device(x.device)
        or not convrot
        or convrot_groupsize != 256
    ):
        result = kitchen_cuda.int8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            out_dtype=out_dtype,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
            input_act=input_act,
            input_act_weight=input_act_weight,
            input_act_eps=input_act_eps,
            residual=residual,
            residual_scale=residual_scale,
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if x2d.dtype == torch.float16:
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize, input_act, input_act_weight, input_act_eps
        )
    elif input_act in ("swiglu", "gelu_tanh"):
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize, input_act=input_act
        )
    else:
        x2d = apply_input_act(
            x2d,
            input_act,
            input_act_weight,
            input_act_eps,
        )
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize
        )

    output_dtype = out_dtype or x.dtype
    output_channels = weight.shape[0]
    gemm_output = None if residual is not None else output
    result = _turing_int8_gemm(
        qactivation,
        weight.contiguous(),
        activation_scale,
        weight_scale,
        bias,
        output_dtype,
        gemm_output,
    )
    result = result.reshape(*original_shape[:-1], output_channels)
    if residual is not None:
        if residual_scale is None:
            raise ValueError("residual_scale is required when residual is provided")
        result = torch.addcmul(residual, result, residual_scale)
        if output is not None:
            output.copy_(result)
            return output
    return result


def codebook_w4a8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
    input_act: str | None = None,
) -> torch.Tensor:
    """Run packed W4/W6 + grouped scales through the shared INT8 family."""
    from comfy_kitchen.backends import cuda as kitchen_cuda
    from comfy_kitchen.backends._activations import apply_input_act

    # The packed W4/W6 tensor describes the post-activation GEMM input. SwiGLU
    # consumes a [gate, up] tensor and halves its last dimension before the
    # linear operation, so compare the weight against K rather than the
    # original 2K input.  Using x.shape[-1] directly rejects the native path
    # for every fused MLP fc2 and sends SM75 to Kitchen's 64-KiB shared-memory
    # fallback, which the 2080 Ti cannot opt into.
    linear_input_channels = x.shape[-1] // 2 if input_act == "swiglu" else x.shape[-1]
    _, logical_k, bits = grouped_weight_geometry(qdata.shape, s_rel.shape, group_size)
    valid_codes = (bits == 4 and codebook is not None and codebook.numel() == 16) or (
        bits == 6 and codebook is None
    )
    fast_path = (
        x.dtype is torch.bfloat16
        and out_dtype is torch.bfloat16
        and is_supported_tensor_core_device(x.device)
        and convrot_groupsize == 256
        and correction is None
        and valid_codes
        and s_rel.dtype in {torch.float8_e4m3fn, torch.float32}
        and qdata.ndim == 2
        and qdata.shape[0] % 8 == 0
        and logical_k == linear_input_channels
    )
    if not fast_path:
        x = apply_input_act(x, input_act)
        return kitchen_cuda.w4a8_int8_linear(
            x,
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            bias=bias,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            out_dtype=out_dtype,
        )

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if input_act not in (None, "none", "swiglu", "gelu_tanh"):
        x2d = apply_input_act(x2d, input_act)
        input_act = None
    qactivation, activation_scale = _quantize_turing_int8_activation(
        x2d,
        convrot_groupsize,
        input_act=input_act,
    )
    operation = _kernel_op("turing_codebook_w4a8_linear")
    output = operation(
        qactivation,
        qdata,
        activation_scale,
        s_rel,
        s_channel,
        codebook,
        bias,
        group_size,
    )
    return output.reshape(*original_shape[:-1], qdata.shape[0])


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    linear_dtype: str = "int4",
    input_act: str | None = None,
) -> torch.Tensor:
    from comfy_kitchen.backends import cuda as kitchen_cuda
    from comfy_kitchen.backends._activations import apply_input_act

    if (
        linear_dtype not in {"int4", "int8"}
        or convrot_groupsize != 256
        or quant_group_size != 64
        or x.dtype != torch.bfloat16
        or not is_supported_tensor_core_device(x.device)
    ):
        x = apply_input_act(x, input_act)
        return kitchen_cuda.convrot_w4a4_linear(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if input_act not in (None, "none", "swiglu", "gelu_tanh"):
        x2d = apply_input_act(x2d, input_act)
        input_act = None
    if linear_dtype == "int8":
        turing_w4a8_linear = _kernel_op("turing_w4a8_linear")

        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d,
            convrot_groupsize,
            input_act=input_act,
        )
        output = turing_w4a8_linear(
            qactivation, qweight, activation_scale, wscales, bias
        )
    else:
        qactivation, activation_scale = _quantize_turing_int4_activation(
            x2d,
            convrot_groupsize,
            input_act=input_act,
        )
        output = kitchen_cuda.int4_linear(
            qactivation,
            qweight.contiguous(),
            activation_scale,
            wscales,
            bias=bias,
            out_dtype=x.dtype,
        )
    output_shape = original_shape[:-1]
    return output.reshape(*output_shape, qweight.shape[0])
