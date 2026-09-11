from __future__ import annotations

import torch

from . import _C


@torch.library.custom_op("turing_utils::w4a8_linear", mutates_args=())
def turing_w4a8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if activation.device.type != "cuda":
        raise RuntimeError("Turing W4A8 requires CUDA tensors")
    if torch.cuda.get_device_capability(activation.device) < (7, 5):
        raise RuntimeError("Turing W4A8 requires sm75 or newer")
    return _C.turing_w4a8_linear(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.contiguous(),
        weight_scale.contiguous(),
        None if bias is None else bias.contiguous(),
    )


@turing_w4a8_linear.register_fake
def _turing_w4a8_linear_fake(activation, weight, activation_scale, weight_scale, bias=None):
    return torch.empty(
        (activation.size(0), weight.size(0)),
        dtype=torch.bfloat16,
        device=activation.device,
    )


@torch.library.custom_op("turing_utils::codebook_w4a8_linear", mutates_args=())
def turing_codebook_w4a8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    group_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    codebook: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    chunk_rows: int = 0,
) -> torch.Tensor:
    """SM75 grouped-codebook W4A8.

    ``chunk_rows=0`` selects the production path, ``-1`` forces inline
    packed-W4 decode for supported long sequences, and a positive multiple of
    eight forces the bounded staged path.
    """
    if activation.device.type != "cuda":
        raise RuntimeError("Turing codebook W4A8 requires CUDA tensors")
    if torch.cuda.get_device_capability(activation.device) < (7, 5):
        raise RuntimeError("Turing codebook W4A8 requires sm75 or newer")
    if group_scale.dtype == torch.float8_e4m3fn:
        group_scale = group_scale.view(torch.uint8)
    if group_scale.dtype != torch.uint8:
        raise TypeError("group_scale must be float8_e4m3fn or its raw uint8 view")
    return _C.turing_codebook_w4a8_linear(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.contiguous(),
        group_scale.contiguous(),
        channel_scale.contiguous(),
        codebook.contiguous(),
        None if bias is None else bias.contiguous(),
        group_size,
        chunk_rows,
    )


@turing_codebook_w4a8_linear.register_fake
def _turing_codebook_w4a8_linear_fake(
    activation,
    weight,
    activation_scale,
    group_scale,
    channel_scale,
    codebook,
    bias=None,
    group_size=16,
    chunk_rows=0,
):
    return torch.empty(
        (activation.size(0), weight.size(0)),
        dtype=torch.bfloat16,
        device=activation.device,
    )


@torch.library.custom_op("turing_utils::int8_linear", mutates_args=())
def turing_int8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw SM75+ W8A8 contraction used by package preflight and comparison."""
    if activation.device.type != "cuda":
        raise RuntimeError("Turing INT8 linear requires CUDA tensors")
    if torch.cuda.get_device_capability(activation.device) < (7, 5):
        raise RuntimeError("Turing INT8 linear requires sm75 or newer")
    return _C.turing_int8_linear(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.contiguous(),
        weight_scale.contiguous(),
        None if bias is None else bias.contiguous(),
    )


@turing_int8_linear.register_fake
def _turing_int8_linear_fake(
    activation, weight, activation_scale, weight_scale, bias=None
):
    return torch.empty(
        (activation.size(0), weight.size(0)),
        dtype=torch.bfloat16,
        device=activation.device,
    )


@torch.library.custom_op(
    "turing_utils::int8_linear_out", mutates_args=("output",)
)
def turing_int8_linear_out(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    """Write W8A8 output into a row-major view with an arbitrary row stride."""
    if activation.device.type != "cuda" or output.device != activation.device:
        raise RuntimeError("Turing INT8 direct output requires one CUDA device")
    if torch.cuda.get_device_capability(activation.device) < (7, 5):
        raise RuntimeError("Turing INT8 direct output requires sm75 or newer")
    _C.turing_int8_linear_out(
        activation.contiguous(),
        weight.contiguous(),
        activation_scale.contiguous(),
        weight_scale.contiguous(),
        None if bias is None else bias.contiguous(),
        output,
    )


@turing_int8_linear_out.register_fake
def _turing_int8_linear_out_fake(
    activation, weight, activation_scale, weight_scale, output, bias=None
):
    return None


@torch.library.custom_op("turing_utils::fp16_int8_linear", mutates_args=())
def turing_fp16_int8_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W8A8 with FP16 output; scales and bias follow eager FP16 rounding."""
    return _C.turing_fp16_int8_linear(
        activation, weight, activation_scale, weight_scale, bias
    )


@turing_fp16_int8_linear.register_fake
def _turing_fp16_int8_linear_fake(activation, weight, activation_scale, weight_scale, bias=None):
    return torch.empty((activation.size(0), weight.size(0)), dtype=torch.float16, device=activation.device)


@torch.library.custom_op("turing_utils::fp16_int8_quantize", mutates_args=())
def turing_fp16_int8_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP16 row quantization with eager's scale and rounding boundaries."""
    return _C.turing_fp16_int8_quantize(x)


@turing_fp16_int8_quantize.register_fake
def _turing_fp16_int8_quantize_fake(x):
    return (
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::dequantize_int8_bf16", mutates_args=())
def turing_dequantize_int8_bf16(
    accumulator: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_columns: int = -1,
) -> torch.Tensor:
    """Dequantize an INT32 GEMM workspace directly into packed BF16 stores."""
    if accumulator.device.type != "cuda":
        raise RuntimeError("Turing BF16 epilogue requires CUDA tensors")
    if torch.cuda.get_device_capability(accumulator.device) < (7, 5):
        raise RuntimeError("Turing BF16 epilogue requires sm75 or newer")
    if accumulator.dtype != torch.int32 or accumulator.ndim != 2:
        raise TypeError("accumulator must be a 2D int32 tensor")
    return _C.turing_dequantize_int8_bf16(
        accumulator.contiguous(),
        activation_scale.contiguous(),
        weight_scale.contiguous(),
        output_columns,
    )


@turing_dequantize_int8_bf16.register_fake
def _turing_dequantize_int8_bf16_fake(
    accumulator, activation_scale, weight_scale, output_columns=-1
):
    columns = accumulator.size(1) if output_columns < 0 else output_columns
    return torch.empty(
        (accumulator.size(0), columns),
        dtype=torch.bfloat16,
        device=accumulator.device,
    )


@torch.library.custom_op("turing_utils::swiglu_int8_convrot_quantize", mutates_args=())
def turing_swiglu_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU into the first pass of staged ConvRot INT8 quantization."""
    if x.device.type != "cuda":
        raise RuntimeError("Turing SwiGLU ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing SwiGLU ConvRot requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SwiGLU ConvRot input must be float16 or bfloat16")
    if x.ndim != 2:
        raise ValueError("SwiGLU ConvRot input must be 2D [M, 2K]")
    return _C.turing_swiglu_int8_convrot_quantize(x.contiguous(), group_size)


@turing_swiglu_int8_convrot_quantize.register_fake
def _turing_swiglu_int8_convrot_quantize_fake(x, group_size=256):
    hidden = x.size(1) // 2
    return (
        torch.empty((x.size(0), hidden), dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(
    "turing_utils::swiglu_int8_convrot_quantize_scaled", mutates_args=()
)
def turing_swiglu_int8_convrot_quantize_scaled(
    x: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 256,
) -> torch.Tensor:
    """Quantize a SwiGLU+ConvRot shard with precomputed whole-row scales."""
    if x.device.type != "cuda":
        raise RuntimeError("scaled Turing SwiGLU ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("scaled Turing SwiGLU ConvRot requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.ndim != 2:
        raise TypeError("scaled SwiGLU ConvRot input must be 2D float16 or bfloat16")
    if scales.dtype != torch.float32 or scales.numel() != x.shape[0]:
        raise TypeError("scaled SwiGLU ConvRot requires one FP32 scale per row")
    return _C.turing_swiglu_int8_convrot_quantize_scaled(
        x.contiguous(), scales.contiguous(), group_size
    )


@turing_swiglu_int8_convrot_quantize_scaled.register_fake
def _turing_swiglu_int8_convrot_quantize_scaled_fake(
    x, scales, group_size=256
):
    return torch.empty(
        (x.size(0), x.size(1) // 2),
        dtype=torch.int8,
        device=x.device,
    )


@torch.library.custom_op(
    "turing_utils::swiglu_int8_convrot_quantize_scaled_out",
    mutates_args=("output",),
)
def turing_swiglu_int8_convrot_quantize_scaled_out(
    x: torch.Tensor,
    scales: torch.Tensor,
    output: torch.Tensor,
    group_size: int = 256,
) -> None:
    """Quantize one channel shard directly into its final INT8 row view."""
    if x.device.type != "cuda" or output.device != x.device:
        raise RuntimeError("scaled Turing SwiGLU direct output requires CUDA")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("scaled Turing SwiGLU direct output requires sm75+")
    _C.turing_swiglu_int8_convrot_quantize_scaled_out(
        x.contiguous(), scales.contiguous(), output, group_size
    )


@turing_swiglu_int8_convrot_quantize_scaled_out.register_fake
def _turing_swiglu_int8_convrot_quantize_scaled_out_fake(
    x, scales, output, group_size=256
):
    return None


@torch.library.custom_op(
    "turing_utils::swiglu_convrot_shard_inplace",
    mutates_args=("gate", "partial_absmax"),
)
def turing_swiglu_convrot_shard_inplace(
    gate: torch.Tensor,
    up: torch.Tensor,
    partial_absmax: torch.Tensor,
    channel_offset: int,
) -> None:
    """Consume one aligned up shard and rotate SwiGLU into ``gate`` in place."""
    if gate.device.type != "cuda" or up.device != gate.device:
        raise RuntimeError("sharded SwiGLU ConvRot requires same-device CUDA tensors")
    if torch.cuda.get_device_capability(gate.device) < (7, 5):
        raise RuntimeError("sharded SwiGLU ConvRot requires sm75 or newer")
    _C.turing_swiglu_convrot_shard_inplace(
        gate, up.contiguous(), partial_absmax, int(channel_offset)
    )


@turing_swiglu_convrot_shard_inplace.register_fake
def _turing_swiglu_convrot_shard_inplace_fake(
    gate, up, partial_absmax, channel_offset
):
    return None


@torch.library.custom_op(
    "turing_utils::int8_convrot_quantize_from_partials", mutates_args=()
)
def turing_int8_convrot_quantize_from_partials(
    rotated: torch.Tensor,
    partial_absmax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finish whole-row INT8 quantization from sharded ConvRot partials."""
    if rotated.device.type != "cuda" or partial_absmax.device != rotated.device:
        raise RuntimeError("ConvRot partial reduction requires same-device CUDA tensors")
    return _C.turing_int8_convrot_quantize_from_partials(
        rotated, partial_absmax
    )


@turing_int8_convrot_quantize_from_partials.register_fake
def _turing_int8_convrot_quantize_from_partials_fake(
    rotated, partial_absmax
):
    return (
        torch.empty_like(rotated, dtype=torch.int8),
        torch.empty(
            (rotated.size(0), 1), dtype=torch.float32, device=rotated.device
        ),
    )


@torch.library.custom_op("turing_utils::swiglu_int4_convrot_quantize", mutates_args=())
def turing_swiglu_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU into staged ConvRot INT4 activation quantization."""
    if x.device.type != "cuda":
        raise RuntimeError("Turing SwiGLU INT4 ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing SwiGLU INT4 ConvRot requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SwiGLU INT4 ConvRot input must be float16 or bfloat16")
    if x.ndim != 2:
        raise ValueError("SwiGLU INT4 ConvRot input must be 2D [M, 2K]")
    return _C.turing_swiglu_int4_convrot_quantize(x.contiguous(), group_size)


@turing_swiglu_int4_convrot_quantize.register_fake
def _turing_swiglu_int4_convrot_quantize_fake(x, group_size=256):
    hidden = x.size(1) // 2
    return (
        torch.empty((x.size(0), hidden // 2), dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::gelu_int8_convrot_quantize", mutates_args=())
def turing_gelu_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse tanh-GELU into staged ConvRot INT8 activation quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing GELU ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.ndim != 2:
        raise TypeError("GELU ConvRot input must be 2D float16 or bfloat16")
    return _C.turing_gelu_int8_convrot_quantize(x.contiguous(), group_size)


@turing_gelu_int8_convrot_quantize.register_fake
def _turing_gelu_int8_convrot_quantize_fake(x, group_size=256):
    return (
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::gelu_int4_convrot_quantize", mutates_args=())
def turing_gelu_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse tanh-GELU into staged ConvRot INT4 activation quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Turing GELU INT4 ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.ndim != 2:
        raise TypeError("GELU INT4 ConvRot input must be 2D float16 or bfloat16")
    return _C.turing_gelu_int4_convrot_quantize(x.contiguous(), group_size)


@turing_gelu_int4_convrot_quantize.register_fake
def _turing_gelu_int4_convrot_quantize_fake(x, group_size=256):
    return (
        torch.empty((x.size(0), x.size(1) // 2), dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::bf16_int8_convrot_quantize", mutates_args=())
def turing_bf16_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
    *,
    swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 ConvRot quantization under the SM75 48 KiB limit."""
    if x.device.type != "cuda":
        raise RuntimeError("BF16 row-buffer ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 row-buffer ConvRot requires sm75 or newer")
    if x.dtype != torch.bfloat16:
        raise TypeError("BF16 row-buffer ConvRot input must be bfloat16")
    if x.ndim != 2:
        raise ValueError("BF16 row-buffer ConvRot input must be 2D")
    return _C.turing_bf16_int8_convrot_quantize(x.contiguous(), group_size, swiglu)


@turing_bf16_int8_convrot_quantize.register_fake
def _turing_bf16_int8_convrot_quantize_fake(x, group_size=256, *, swiglu=False):
    hidden = x.size(1) // 2 if swiglu else x.size(1)
    return (
        torch.empty((x.size(0), hidden), dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::bf16_int4_convrot_quantize", mutates_args=())
def turing_bf16_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
    *,
    swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 ConvRot INT4 quantization under the SM75 48 KiB limit."""
    if x.device.type != "cuda":
        raise RuntimeError("BF16 row-buffer INT4 ConvRot requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 row-buffer INT4 ConvRot requires sm75 or newer")
    if x.dtype != torch.bfloat16:
        raise TypeError("BF16 row-buffer INT4 ConvRot input must be bfloat16")
    if x.ndim != 2:
        raise ValueError("BF16 row-buffer INT4 ConvRot input must be 2D")
    return _C.turing_bf16_int4_convrot_quantize(x.contiguous(), group_size, swiglu)


@turing_bf16_int4_convrot_quantize.register_fake
def _turing_bf16_int4_convrot_quantize_fake(x, group_size=256, *, swiglu=False):
    hidden = x.size(1) // 2 if swiglu else x.size(1)
    return (
        torch.empty((x.size(0), hidden // 2), dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::bf16_gelu_int8_convrot_quantize", mutates_args=())
def turing_bf16_gelu_int8_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 tanh-GELU+ConvRot INT8 quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 GELU ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TypeError("BF16 GELU ConvRot input must be a 2D bfloat16 tensor")
    return _C.turing_bf16_gelu_int8_convrot_quantize(x.contiguous(), group_size)


@turing_bf16_gelu_int8_convrot_quantize.register_fake
def _turing_bf16_gelu_int8_convrot_quantize_fake(x, group_size=256):
    return (
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::bf16_gelu_int4_convrot_quantize", mutates_args=())
def turing_bf16_gelu_int4_convrot_quantize(
    x: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-row BF16 tanh-GELU+ConvRot INT4 quantization."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("BF16 GELU INT4 ConvRot requires an sm75-or-newer CUDA tensor")
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TypeError("BF16 GELU INT4 ConvRot input must be a 2D bfloat16 tensor")
    return _C.turing_bf16_gelu_int4_convrot_quantize(x.contiguous(), group_size)


@turing_bf16_gelu_int4_convrot_quantize.register_fake
def _turing_bf16_gelu_int4_convrot_quantize_fake(x, group_size=256):
    return (
        torch.empty((x.size(0), x.size(1) // 2), dtype=torch.int8, device=x.device),
        torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("turing_utils::segmented_rms_adaln", mutates_args=())
def turing_segmented_rms_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    segments: torch.Tensor,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Fused affine RMSNorm and segmented AdaLN modulation."""
    if x.device.type != "cuda":
        raise RuntimeError("Segmented RMSNorm+AdaLN requires CUDA tensors")
    if torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Segmented RMSNorm+AdaLN requires sm75 or newer")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("segmented RMSNorm+AdaLN input must be float16, bfloat16, or float32")
    if x.ndim != 2:
        raise ValueError("segmented RMSNorm+AdaLN input must be 2D [M, K]")
    if weight.device != x.device or scale.device != x.device or shift.device != x.device:
        raise ValueError("weight, scale, and shift must be on the input device")
    weight = weight.to(dtype=x.dtype).contiguous()
    scale = scale.to(dtype=x.dtype)
    shift = shift.to(dtype=x.dtype)
    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    if segments.device != x.device or segments.dtype != torch.int32:
        raise ValueError("segments must be an int32 tensor on the input device")
    return _C.turing_segmented_rms_adaln(
        x.contiguous(),
        weight,
        scale,
        shift,
        segments.contiguous(),
        epsilon,
    )


@turing_segmented_rms_adaln.register_fake
def _turing_segmented_rms_adaln_fake(x, weight, scale, shift, segments, epsilon=1.0e-5):
    return torch.empty_like(x)


def _prepare_segmented_modulation(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    segments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("Segmented mod-gate requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or x.ndim != 2:
        raise TypeError("segmented mod-gate input must be a 2D floating tensor")
    if not x.is_contiguous():
        raise ValueError("segmented mod-gate mutates x and therefore requires contiguous storage")
    residual = residual.to(device=x.device, dtype=x.dtype).contiguous()
    gate = gate.to(device=x.device, dtype=x.dtype)
    if gate.stride(-1) != 1:
        gate = gate.contiguous()
    if segments.device != x.device or segments.dtype != torch.int32:
        raise ValueError("segments must be int32 on the input device")
    return gate, residual, segments.contiguous()


@torch.library.custom_op("turing_utils::segmented_mod_gate", mutates_args=("x",))
def turing_segmented_mod_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    segments: torch.Tensor,
) -> None:
    """Apply the segmented gated residual directly into ``x``."""
    gate, residual, segments = _prepare_segmented_modulation(x, gate, residual, segments)
    _C.turing_segmented_mod_gate(x, gate, residual, segments)


@torch.library.custom_op(
    "turing_utils::segmented_mod_gate_rms_adaln", mutates_args=("x",)
)
def turing_segmented_mod_gate_rms_adaln(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    segments: torch.Tensor,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Fuse a dtype-rounded gated residual with following RMSNorm+AdaLN."""
    gate, residual, segments = _prepare_segmented_modulation(x, gate, residual, segments)
    weight = weight.to(device=x.device, dtype=x.dtype).contiguous()
    scale = scale.to(device=x.device, dtype=x.dtype)
    shift = shift.to(device=x.device, dtype=x.dtype)
    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    return _C.turing_segmented_mod_gate_rms_adaln(
        x, gate, residual, weight, scale, shift, segments, epsilon
    )


@turing_segmented_mod_gate_rms_adaln.register_fake
def _turing_segmented_mod_gate_rms_adaln_fake(
    x, gate, residual, weight, scale, shift, segments, epsilon=1.0e-5
):
    return torch.empty_like(x)


@torch.library.custom_op("turing_utils::layer_norm_adaln", mutates_args=())
def turing_layer_norm_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Fused FP32-reduction LayerNorm and Wan AdaLN modulation."""
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) < (7, 5):
        raise RuntimeError("LayerNorm+AdaLN requires an sm75-or-newer CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or x.ndim != 3:
        raise TypeError("LayerNorm+AdaLN input must be 3D float16, bfloat16, or float32")
    scale = scale.to(device=x.device, dtype=x.dtype).contiguous()
    shift = shift.to(device=x.device, dtype=x.dtype).contiguous()
    return _C.turing_layer_norm_adaln(x.contiguous(), scale, shift, epsilon)


@turing_layer_norm_adaln.register_fake
def _turing_layer_norm_adaln_fake(x, scale, shift, epsilon=1.0e-5):
    return torch.empty_like(x)
