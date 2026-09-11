"""ConvRot activation quantization and GEMM dispatch for sm75 and newer."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ..hardware import (
    device_capabilities,
    is_supported_tensor_core_device,
    is_supported_turing_device,
)
from ..log import get_logger
from .capabilities import (
    BACKEND_NAME,
    kernel_available as _kernel_available,
    kernel_op as _kernel_op,
    kitchen_backend_available as backend_available,
)
from .operator_scope import use_turing_operator_backend
from .workspace import codebook_w4a8_workspace_bytes, int8_workspace_bytes


LOG = get_logger("quantization")
KITCHEN_DEFAULT_SHARED_MEMORY_LIMIT = 48 * 1024
TURING_OPTIN_SHARED_MEMORY_LIMIT = 64 * 1024
# Above this point a full MxN INT32 accumulator is more expensive than the
# fixed-workspace fused Turing path. This is a dispatch threshold, never an
# input-size limit.
TURING_INT8_GLOBAL_WORKSPACE_LIMIT = 64 * 1024 * 1024
TURING_CODEBOOK_W4A8_CHUNK_ROWS = 4096
_PREFLIGHTED_DEVICES: set[int] = set()
_PREFLIGHTED_CODEBOOK_DEVICES: set[int] = set()
_PREFLIGHTED_KITCHEN: set[tuple[int, bool, bool]] = set()


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




def preflight_w4a8(device: torch.device) -> None:
    if not is_supported_tensor_core_device(device):
        raise RuntimeError(f"unsupported device {device}")
    if not _kernel_available():
        raise RuntimeError("the installed comfyui-turing-utils-kernel does not provide W4A8")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return

    turing_w4a8_linear = _kernel_op("turing_w4a8_linear")

    activation = ((torch.arange(3 * 64, device=device) % 23) - 11).to(torch.int8).reshape(3, 64)
    weight_values = ((torch.arange(5 * 64, device=device) % 15) - 7).to(torch.int8).reshape(5, 64)
    low = weight_values[:, 0::2].to(torch.int32) & 0x0f
    high = weight_values[:, 1::2].to(torch.int32) & 0x0f
    packed_weight = (low | (high << 4)).to(torch.int8)
    activation_scale = torch.linspace(0.01, 0.03, 3, device=device)
    weight_scale = torch.linspace(0.02, 0.06, 5, device=device)
    bias = torch.linspace(-0.2, 0.2, 5, dtype=torch.bfloat16, device=device)
    output = turing_w4a8_linear(
        activation,
        packed_weight,
        activation_scale,
        weight_scale,
        bias,
    )
    reference = (
        activation.float() @ weight_values.float().t()
    ) * activation_scale[:, None] * weight_scale[None, :] + bias.float()
    if output.dtype != torch.bfloat16 or not torch.allclose(output.float(), reference, rtol=0.01, atol=0.01):
        raise RuntimeError("packed W4A8 numerical self-test failed")
    for hidden_size in (256, 8192):
        bf16_input = (
            ((torch.arange(3 * hidden_size, device=device) % 29) - 14)
            .reshape(3, hidden_size)
            .to(torch.bfloat16)
            / 16
        )
        full_weight = torch.zeros((5, hidden_size // 2), dtype=torch.int8, device=device)
        full_output = convrot_w4a4_linear(
            bf16_input,
            full_weight,
            torch.ones((5,), dtype=torch.float32, device=device),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype="int8",
        )
        if full_output.dtype != torch.bfloat16 or not torch.isfinite(full_output).all():
            raise RuntimeError(f"BF16 ConvRot W4A8 self-test failed for K={hidden_size}")
    swiglu_input = torch.zeros((3, 512), dtype=torch.bfloat16, device=device)
    swiglu_output = convrot_w4a4_linear(
        swiglu_input,
        torch.zeros((5, 128), dtype=torch.int8, device=device),
        torch.ones((5,), dtype=torch.float32, device=device),
        convrot_groupsize=256,
        quant_group_size=64,
        linear_dtype="int8",
        input_act="swiglu",
    )
    if swiglu_output.dtype != torch.bfloat16 or not torch.isfinite(swiglu_output).all():
        raise RuntimeError("SwiGLU W4A8 BF16 self-test failed")
    torch.cuda.synchronize(device)
    _PREFLIGHTED_DEVICES.add(index)


def preflight_codebook_w4a8(device: torch.device) -> None:
    """Validate the published grouped-codebook W4A8 contract once per device."""
    if not is_supported_tensor_core_device(device):
        raise RuntimeError(f"unsupported device {device}")
    if not _kernel_available("turing_codebook_w4a8_linear"):
        raise RuntimeError(
            "the installed comfyui-turing-utils-kernel does not provide codebook W4A8"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_CODEBOOK_DEVICES:
        return

    operation = _kernel_op("turing_codebook_w4a8_linear")
    m, n, k, group_size = 3, 8, 64, 16
    activation = ((torch.arange(m * k, device=device) % 23) - 11).to(torch.int8).reshape(m, k)
    codes = (torch.arange(n * k, device=device) % 16).to(torch.int32).reshape(n, k)
    packed = ((codes[:, 0::2] & 0x0f) | ((codes[:, 1::2] & 0x0f) << 4)).to(torch.int8)
    codebook = torch.linspace(-0.95, 0.95, 16, dtype=torch.float32, device=device)
    group_scale = torch.linspace(
        8.0, 32.0, n * (k // group_size), dtype=torch.float32, device=device
    ).reshape(n, k // group_size).to(torch.float8_e4m3fn)
    channel_scale = torch.linspace(0.01, 0.03, n, dtype=torch.float32, device=device)
    activation_scale = torch.linspace(0.02, 0.04, m, dtype=torch.float32, device=device)
    bias = torch.linspace(-0.1, 0.1, n, dtype=torch.bfloat16, device=device)
    output = operation(
        activation,
        packed,
        activation_scale,
        group_scale,
        channel_scale,
        codebook,
        bias,
        group_size,
    )
    decoded = (
        codebook[codes]
        * group_scale.float().repeat_interleave(group_size, dim=1)
    ).round().clamp(-127, 127)
    reference = (
        activation.float() @ decoded.float().t()
    ) * activation_scale[:, None] * channel_scale[None, :] + bias.float()
    if output.dtype is not torch.bfloat16 or not torch.allclose(
        output.float(), reference, rtol=0.01, atol=0.02
    ):
        raise RuntimeError("codebook W4A8 numerical self-test failed")

    # Exercise the H3 MLP contract as well as the raw contraction above.  Its
    # fc2 receives [gate, up] at 2K and therefore exposed a distinct dispatch
    # path that the original small-K preflight did not cover.
    h3_k = 14336
    swiglu_input = (
        ((torch.arange(m * 2 * h3_k, device=device) % 29) - 14)
        .reshape(m, 2 * h3_k)
        .to(torch.bfloat16)
        / 16
    )
    swiglu_output = codebook_w4a8_linear(
        swiglu_input,
        torch.zeros((n, h3_k // 2), dtype=torch.int8, device=device),
        torch.ones(
            (n, h3_k // group_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.ones(n, dtype=torch.float32, device=device),
        codebook=codebook,
        group_size=group_size,
        convrot_groupsize=256,
        out_dtype=torch.bfloat16,
        input_act="swiglu",
    )
    if swiglu_output.dtype is not torch.bfloat16 or not torch.isfinite(
        swiglu_output
    ).all():
        raise RuntimeError("codebook W4A8 H3 SwiGLU self-test failed")
    _PREFLIGHTED_CODEBOOK_DEVICES.add(index)


def preflight_kitchen(device: torch.device, w4a4: bool, w8a8: bool) -> None:
    if not is_supported_tensor_core_device(device):
        raise RuntimeError(f"unsupported device {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, w4a4, w8a8)
    if key in _PREFLIGHTED_KITCHEN:
        return

    import comfy_kitchen

    if w4a4:
        for hidden_size in (256, 16384):
            x = (
                ((torch.arange(16 * hidden_size, device=device) % 31) - 15)
                .reshape(16, hidden_size)
                .to(torch.bfloat16)
                / 16
            )
            packed_weight = torch.zeros((64, hidden_size // 2), dtype=torch.int8, device=device)
            weight_scale = torch.ones((64,), dtype=torch.float32, device=device)
            with use_turing_operator_backend():
                output = comfy_kitchen.convrot_w4a4_linear(
                    x,
                    packed_weight,
                    weight_scale,
                    convrot_groupsize=256,
                    quant_group_size=64,
                    linear_dtype="int4",
                )
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
                raise RuntimeError(f"Kitchen W4A4 BF16 self-test failed for K={hidden_size}")
        swiglu_input = torch.zeros((16, 512), dtype=torch.bfloat16, device=device)
        swiglu_weight = torch.zeros((64, 128), dtype=torch.int8, device=device)
        swiglu_output = convrot_w4a4_linear(
            swiglu_input,
            swiglu_weight,
            torch.ones((64,), dtype=torch.float32, device=device),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype="int4",
            input_act="swiglu",
        )
        if swiglu_output.dtype != torch.bfloat16 or not torch.isfinite(swiglu_output).all():
            raise RuntimeError("SwiGLU W4A4 BF16 self-test failed")
    if w8a8:
        for hidden_size in (256, 5376):
            x = (
                ((torch.arange(16 * hidden_size, device=device) % 31) - 15)
                .reshape(16, hidden_size)
                .to(torch.bfloat16)
                / 16
            )
            weight = torch.zeros((64, hidden_size), dtype=torch.int8, device=device)
            weight_scale = torch.ones((), dtype=torch.float32, device=device)
            with use_turing_operator_backend():
                output = comfy_kitchen.int8_linear(
                    x,
                    weight,
                    weight_scale,
                    out_dtype=torch.bfloat16,
                    convrot=True,
                    convrot_groupsize=256,
                )
            if output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
                raise RuntimeError(f"Kitchen W8A8 BF16 self-test failed for K={hidden_size}")
        swiglu_input = torch.cat((x, x), dim=-1)
        with use_turing_operator_backend():
            swiglu_output = comfy_kitchen.int8_linear(
                swiglu_input,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
                input_act="swiglu",
            )
        if swiglu_output.dtype != torch.bfloat16 or not torch.isfinite(swiglu_output).all():
            raise RuntimeError("Kitchen SwiGLU W8A8 BF16 self-test failed")
        contraction_input = torch.zeros((129, 256), dtype=torch.bfloat16, device=device)
        with use_turing_operator_backend():
            contraction_output = comfy_kitchen.int8_linear(
                contraction_input,
                torch.zeros((64, 256), dtype=torch.int8, device=device),
                torch.ones((), dtype=torch.float32, device=device),
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
            )
        if contraction_output.dtype != torch.bfloat16 or not torch.isfinite(contraction_output).all():
            raise RuntimeError("W8A8 BF16 contraction self-test failed")
    torch.cuda.synchronize(device)
    _PREFLIGHTED_KITCHEN.add(key)


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


def _convrot_int4_shared_memory_bytes(rows: int, hidden_size: int, element_size: int) -> int:
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
):
    from comfy_kitchen.backends import cuda as kitchen_cuda

    if input_act not in (None, "none", "swiglu", "gelu_tanh"):
        raise ValueError(f"unsupported fused INT8 activation: {input_act!r}")
    if input_act == "swiglu" and x2d.shape[1] % 2:
        raise ValueError("SwiGLU input width must be even")
    hidden_size = x2d.shape[1] // 2 if input_act == "swiglu" else x2d.shape[1]
    if x2d.dtype == torch.float16 and input_act in (None, "none", "swiglu"):
        from comfy_kitchen.backends._activations import apply_input_act
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

        activated = apply_input_act(x2d, input_act)
        hadamard = _build_hadamard(group_size, device=x2d.device, dtype=x2d.dtype)
        rotated = _rotate_activation(activated, hadamard, group_size)
        return _kernel_op("turing_fp16_int8_quantize")(rotated)
    if input_act == "gelu_tanh":
        if (
            x2d.dtype == torch.bfloat16
            and is_supported_tensor_core_device(x2d.device)
            and _convrot_int8_bf16_rowbuffer_fits(hidden_size, x2d.device)
            and _kernel_available("turing_bf16_gelu_int8_convrot_quantize")
        ):
            return _kernel_op("turing_bf16_gelu_int8_convrot_quantize")(
                x2d, group_size
            )
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
        return _kernel_op("turing_swiglu_int8_convrot_quantize")(
            x2d, group_size
        )
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
    return _quantize_turing_int8_activation(
        x.contiguous(), int(group_size)
    )


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
    return operation(
        x.contiguous(), int(group_size), swiglu=True
    )


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
        if not _kernel_available(
            "turing_swiglu_int8_convrot_quantize_scaled_out"
        ):
            temporary = quantize_convrot_swiglu_with_scale(
                x, scale, group_size
            )
            output.copy_(temporary)
            return output
        operation = _kernel_op(
            "turing_swiglu_int8_convrot_quantize_scaled_out"
        )
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
            return _kernel_op("turing_bf16_gelu_int4_convrot_quantize")(
                x2d, group_size
            )
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
        return _kernel_op("turing_swiglu_int4_convrot_quantize")(
            x2d, group_size
        )

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
            output.shape != (m, n) or output.dtype != output_dtype
            or output.device != qactivation.device
        ):
            raise ValueError("W8A8 direct output shape, dtype, or device is incompatible")
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
            raise ValueError("W8A8 direct output shape, dtype, device, or stride is incompatible")
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

    avoid_global_workspace = (
        m * n * 4 >= TURING_INT8_GLOBAL_WORKSPACE_LIMIT
    )
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
        avoid_global_workspace
        or (callable(prefer_fused) and prefer_fused(m, n, k))
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
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    original_shape = x.shape
    x2d = x.reshape(-1, original_shape[-1]).contiguous()
    if input_act in ("swiglu", "gelu_tanh"):
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize, input_act=input_act
        )
    else:
        x2d = apply_input_act(x2d, input_act)
        qactivation, activation_scale = _quantize_turing_int8_activation(
            x2d, convrot_groupsize
        )

    output_dtype = out_dtype or x.dtype
    output_channels = weight.shape[0]
    output = _turing_int8_gemm(
        qactivation,
        weight.contiguous(),
        activation_scale,
        weight_scale,
        bias,
        output_dtype,
        output,
    )
    return output.reshape(*original_shape[:-1], output_channels)


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
    """Run Kitchen's grouped-codebook format through the bounded SM75 path."""
    from comfy_kitchen.backends import cuda as kitchen_cuda
    from comfy_kitchen.backends._activations import apply_input_act

    # The packed W4 tensor describes the post-activation GEMM input.  SwiGLU
    # consumes a [gate, up] tensor and halves its last dimension before the
    # linear operation, so compare the weight against K rather than the
    # original 2K input.  Using x.shape[-1] directly rejects the native path
    # for every fused MLP fc2 and sends SM75 to Kitchen's 64-KiB shared-memory
    # fallback, which the 2080 Ti cannot opt into.
    linear_input_channels = (
        x.shape[-1] // 2 if input_act == "swiglu" else x.shape[-1]
    )
    fast_path = (
        x.dtype is torch.bfloat16
        and out_dtype is torch.bfloat16
        and is_supported_tensor_core_device(x.device)
        and convrot_groupsize == 256
        and correction is None
        and codebook is not None
        and codebook.numel() == 16
        and s_rel.dtype is torch.float8_e4m3fn
        and qdata.ndim == 2
        and qdata.shape[0] % 8 == 0
        and qdata.shape[1] * 2 == linear_input_channels
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


# Kitchen resolves backend implementations by the public capability name.
w4a8_int8_linear = codebook_w4a8_linear


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
        output = turing_w4a8_linear(qactivation, qweight, activation_scale, wscales, bias)
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


def register_backend() -> bool:
    try:
        import comfy_kitchen
        from comfy_kitchen.constraints import (
            ExactDims,
            FunctionConstraints,
            MinDims,
            ParamConstraint,
            ShapeRule,
            ValidationResult,
        )
        from comfy_kitchen.registry import registry
    except ImportError:
        return False

    cuda_status = comfy_kitchen.list_backends().get("cuda", {})
    cuda_capabilities = set(cuda_status.get("capabilities", ()))
    if BACKEND_NAME in comfy_kitchen.list_backends():
        return backend_available()

    class SupportedTensorCoreTensor(ShapeRule):
        def check(self, tensor: torch.Tensor) -> bool:
            return is_supported_tensor_core_device(tensor.device)

        def describe(self) -> str:
            return "tensor on a supported NVIDIA sm75+ Tensor Core device"

    cuda_devices = frozenset({"cuda"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    has_w4a8_kernel = _kernel_available()
    has_w4a4_quantizer = _kernel_available(
        "turing_bf16_int4_convrot_quantize"
    )
    has_codebook_w4a8_kernel = _kernel_available("turing_codebook_w4a8_linear")

    def require_convrot_256(kwargs):
        if kwargs.get("convrot") is not True:
            return ValidationResult.fail("convrot", "staged INT8 requires ConvRot")
        if kwargs.get("convrot_groupsize") != 256:
            return ValidationResult.fail("convrot_groupsize", "staged INT8 requires group size 256")
        x = kwargs.get("x")
        weight = kwargs.get("weight")
        input_act = kwargs.get("input_act")
        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            return ValidationResult.fail("x", "Turing W8A8 requires tensor inputs")
        if kwargs.get("out_dtype") != x.dtype:
            return ValidationResult.fail("out_dtype", "local W8A8 preserves the activation dtype")
        hidden = x.shape[-1] // 2 if input_act == "swiglu" else x.shape[-1]
        if hidden % 16 or weight.shape[0] % 8 or weight.shape[1] != hidden:
            return ValidationResult.fail(
                "weight", "Turing W8A8 requires matching K%16=0 and N%8=0"
            )
        rowbuffer = _convrot_int8_bf16_rowbuffer_fits(hidden, x.device)
        if x.dtype == torch.float16:
            if (
                hidden % 256
                or input_act not in (None, "none", "swiglu")
                or not _kernel_available("turing_fp16_int8_quantize")
                or not _kernel_available("turing_fp16_int8_linear")
            ):
                return ValidationResult.fail("x", "no compatible FP16 ConvRot kernel is available")
            return ValidationResult.ok()
        if input_act in (None, "none"):
            local_quantizer = rowbuffer and _kernel_available(
                "turing_bf16_int8_convrot_quantize"
            )
        elif input_act == "swiglu":
            local_quantizer = (
                rowbuffer
                and _kernel_available("turing_bf16_int8_convrot_quantize")
            ) or _kernel_available("turing_swiglu_int8_convrot_quantize")
        elif input_act == "gelu_tanh":
            local_quantizer = (
                rowbuffer
                and _kernel_available("turing_bf16_gelu_int8_convrot_quantize")
            ) or _kernel_available("turing_gelu_int8_convrot_quantize")
        else:
            local_quantizer = False
        if not local_quantizer:
            return ValidationResult.fail(
                "input_act", "no exact Turing Utils activation quantizer is available"
            )
        return ValidationResult.ok()

    def require_w4_convrot_256(kwargs):
        if kwargs.get("convrot_groupsize") != 256:
            return ValidationResult.fail("convrot_groupsize", "W4 requires ConvRot group size 256")
        if kwargs.get("quant_group_size") != 64:
            return ValidationResult.fail("quant_group_size", "W4 requires quantization group size 64")
        if kwargs.get("linear_dtype") not in {"int4", "int8"}:
            return ValidationResult.fail("linear_dtype", "W4 requires int4 or int8 activation")
        if kwargs.get("linear_dtype") == "int8" and not has_w4a8_kernel:
            return ValidationResult.fail("linear_dtype", "W4A8 kernel is unavailable")
        x = kwargs.get("x")
        if not isinstance(x, torch.Tensor) or not (
            has_w4a4_quantizer
            and _convrot_int8_bf16_rowbuffer_fits(x.shape[-1], x.device)
        ):
            return ValidationResult.fail(
                "x", "no exact Turing Utils INT4 activation quantizer is available"
            )
        return ValidationResult.ok()

    def require_codebook_w4a8(kwargs):
        if not has_codebook_w4a8_kernel:
            return ValidationResult.fail("qdata", "codebook W4A8 kernel is unavailable")
        if kwargs.get("convrot_groupsize") != 256:
            return ValidationResult.fail(
                "convrot_groupsize", "codebook W4A8 requires ConvRot group size 256"
            )
        if kwargs.get("correction") is not None:
            return ValidationResult.fail(
                "correction", "codebook W4A8 fast path supports symmetric files only"
            )
        if kwargs.get("codebook") is None:
            return ValidationResult.fail(
                "codebook", "codebook W4A8 requires a 16-entry codebook"
            )
        if kwargs.get("group_size") != 16:
            return ValidationResult.fail(
                "group_size", "codebook W4A8 requires group size 16"
            )
        if kwargs.get("out_dtype") is not torch.bfloat16:
            return ValidationResult.fail(
                "out_dtype", "codebook W4A8 local output must be BF16"
            )
        x = kwargs.get("x")
        qdata = kwargs.get("qdata")
        codebook = kwargs.get("codebook")
        if not all(isinstance(value, torch.Tensor) for value in (x, qdata, codebook)):
            return ValidationResult.fail("x", "codebook W4A8 requires tensor inputs")
        if codebook.numel() != 16:
            return ValidationResult.fail("codebook", "codebook must contain 16 values")
        if qdata.shape[0] % 8 or qdata.shape[1] * 2 != x.shape[-1]:
            return ValidationResult.fail(
                "qdata", "codebook W4A8 requires matching packed K and N%8=0"
            )
        return ValidationResult.ok()

    operations = {}
    implementations = {}
    if (
        "int8_linear" in cuda_capabilities
        and _kernel_available("turing_int8_linear")
    ):
        operations["int8_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16, torch.float16}),
                    shape_rules=(MinDims(2), SupportedTensorCoreTensor()),
                ),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "weight_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "bias": ParamConstraint(dtypes=standard_floats),
                "out_dtype": ParamConstraint(dtypes=frozenset({torch.bfloat16, torch.float16})),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
            },
            default_devices=cuda_devices,
            call_rules=(require_convrot_256,),
        )
        implementations["int8_linear"] = int8_linear
    if "convrot_w4a4_linear" in cuda_capabilities and has_w4a4_quantizer:
        operations["convrot_w4a4_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16}),
                    shape_rules=(MinDims(2), SupportedTensorCoreTensor()),
                ),
                "qweight": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "wscales": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),)
                ),
                "bias": ParamConstraint(dtypes=standard_floats),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                "linear_dtype": ParamConstraint(dtypes=frozenset({str})),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
            },
            default_devices=cuda_devices,
            call_rules=(require_w4_convrot_256,),
        )
        implementations["convrot_w4a4_linear"] = convrot_w4a4_linear
    if (
        "w4a8_int8_linear" in cuda_capabilities
        and has_codebook_w4a8_kernel
    ):
        operations["w4a8_int8_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16}),
                    shape_rules=(MinDims(2), SupportedTensorCoreTensor()),
                ),
                "qdata": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "s_rel": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}), shape_rules=(ExactDims(2),)
                ),
                "s_channel": ParamConstraint(
                    dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)
                ),
                "codebook": ParamConstraint(dtypes=frozenset({torch.float32})),
                "correction": ParamConstraint(dtypes=standard_floats),
                "bias": ParamConstraint(dtypes=standard_floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            call_rules=(require_codebook_w4a8,),
        )
        implementations["w4a8_int8_linear"] = codebook_w4a8_linear
    if not operations:
        return False
    registry.register(BACKEND_NAME, SimpleNamespace(**implementations), operations)
    LOG.info(
        "Registered scoped sm75+ operator backend: name=%s operators=%s global_priority=unchanged",
        BACKEND_NAME,
        ",".join(sorted(operations)),
    )
    return True
