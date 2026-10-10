"""One-time numerical capability checks for installed quantization kernels."""

from __future__ import annotations

import torch
from ..hardware import is_supported_tensor_core_device
from .capabilities import kernel_available as _kernel_available, kernel_op as _kernel_op
from .operator_scope import use_turing_operator_backend
from .dispatch import (
    codebook_w4a8_linear,
    convrot_w4a4_linear,
)


_PREFLIGHTED_DEVICES: set[int] = set()


_PREFLIGHTED_CODEBOOK_DEVICES: set[tuple[int, int]] = set()


_PREFLIGHTED_KITCHEN: set[tuple[int, bool, bool]] = set()


def preflight_w4a8(device: torch.device) -> None:
    if not is_supported_tensor_core_device(device):
        raise RuntimeError(f"unsupported device {device}")
    if not _kernel_available():
        raise RuntimeError(
            "the installed comfyui-turing-utils-kernel does not provide W4A8"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PREFLIGHTED_DEVICES:
        return

    turing_w4a8_linear = _kernel_op("turing_w4a8_linear")

    activation = (
        ((torch.arange(3 * 64, device=device) % 23) - 11).to(torch.int8).reshape(3, 64)
    )
    weight_values = (
        ((torch.arange(5 * 64, device=device) % 15) - 7).to(torch.int8).reshape(5, 64)
    )
    low = weight_values[:, 0::2].to(torch.int32) & 0x0F
    high = weight_values[:, 1::2].to(torch.int32) & 0x0F
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
    reference = (activation.float() @ weight_values.float().t()) * activation_scale[
        :, None
    ] * weight_scale[None, :] + bias.float()
    if output.dtype != torch.bfloat16 or not torch.allclose(
        output.float(), reference, rtol=0.01, atol=0.01
    ):
        raise RuntimeError("packed W4A8 numerical self-test failed")
    for hidden_size in (256, 8192):
        bf16_input = ((torch.arange(3 * hidden_size, device=device) % 29) - 14).reshape(
            3, hidden_size
        ).to(torch.bfloat16) / 16
        full_weight = torch.zeros(
            (5, hidden_size // 2), dtype=torch.int8, device=device
        )
        full_output = convrot_w4a4_linear(
            bf16_input,
            full_weight,
            torch.ones((5,), dtype=torch.float32, device=device),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype="int8",
        )
        if full_output.dtype != torch.bfloat16 or not torch.isfinite(full_output).all():
            raise RuntimeError(
                f"BF16 ConvRot W4A8 self-test failed for K={hidden_size}"
            )
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


def preflight_codebook_w4a8(device: torch.device, *, bits: int = 4) -> None:
    """Validate grouped W4/W6 once per device and packed bit width."""
    if bits not in (4, 6):
        raise ValueError("Grouped preflight supports only W4 and W6")
    if not is_supported_tensor_core_device(device):
        raise RuntimeError(f"unsupported device {device}")
    if not _kernel_available("turing_codebook_w4a8_linear"):
        raise RuntimeError(
            "the installed comfyui-turing-utils-kernel does not provide codebook W4A8"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    if (index, bits) in _PREFLIGHTED_CODEBOOK_DEVICES:
        return

    operation = _kernel_op("turing_codebook_w4a8_linear")
    m, n, k, group_size = 3, 8, 64, 16
    activation = (
        ((torch.arange(m * k, device=device) % 23) - 11).to(torch.int8).reshape(m, k)
    )
    codes = (
        (torch.arange(n * k, device=device) % (1 << bits)).to(torch.int32).reshape(n, k)
    )
    packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int8)
    codebook = (
        torch.linspace(-0.95, 0.95, 16, dtype=torch.float32, device=device)
        if bits == 4
        else None
    )
    if bits == 6:
        upper = codes.reshape(n, k // 4, 4) >> 4
        shifts = torch.arange(4, device=device) * 2
        packed = torch.cat((packed, (upper << shifts).sum(-1).to(torch.int8)), dim=1)
    group_scale = (
        torch.linspace(
            8.0, 32.0, n * (k // group_size), dtype=torch.float32, device=device
        )
        .reshape(n, k // group_size)
        .to(torch.float8_e4m3fn)
    )
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
        (
            (codebook[codes] if bits == 4 else codes - 32)
            * group_scale.float().repeat_interleave(group_size, dim=1)
        )
        .round()
        .clamp(-127, 127)
    )
    reference = (activation.float() @ decoded.float().t()) * activation_scale[
        :, None
    ] * channel_scale[None, :] + bias.float()
    if output.dtype is not torch.bfloat16 or not torch.allclose(
        output.float(), reference, rtol=0.01, atol=0.02
    ):
        raise RuntimeError(f"Grouped W{bits}A8 numerical self-test failed")

    # Exercise the H3 MLP contract as well as the raw contraction above.  Its
    # fc2 receives [gate, up] at 2K and therefore exposed a distinct dispatch
    # path that the original small-K preflight did not cover.
    h3_k = 14336
    swiglu_input = ((torch.arange(m * 2 * h3_k, device=device) % 29) - 14).reshape(
        m, 2 * h3_k
    ).to(torch.bfloat16) / 16
    swiglu_output = codebook_w4a8_linear(
        swiglu_input,
        torch.zeros((n, h3_k * bits // 8), dtype=torch.int8, device=device),
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
    if (
        swiglu_output.dtype is not torch.bfloat16
        or not torch.isfinite(swiglu_output).all()
    ):
        raise RuntimeError("codebook W4A8 H3 SwiGLU self-test failed")
    _PREFLIGHTED_CODEBOOK_DEVICES.add((index, bits))


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
            x = ((torch.arange(16 * hidden_size, device=device) % 31) - 15).reshape(
                16, hidden_size
            ).to(torch.bfloat16) / 16
            packed_weight = torch.zeros(
                (64, hidden_size // 2), dtype=torch.int8, device=device
            )
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
                raise RuntimeError(
                    f"Kitchen W4A4 BF16 self-test failed for K={hidden_size}"
                )
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
        if (
            swiglu_output.dtype != torch.bfloat16
            or not torch.isfinite(swiglu_output).all()
        ):
            raise RuntimeError("SwiGLU W4A4 BF16 self-test failed")
    if w8a8:
        for hidden_size in (256, 5376):
            x = ((torch.arange(16 * hidden_size, device=device) % 31) - 15).reshape(
                16, hidden_size
            ).to(torch.bfloat16) / 16
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
                raise RuntimeError(
                    f"Kitchen W8A8 BF16 self-test failed for K={hidden_size}"
                )
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
        if (
            swiglu_output.dtype != torch.bfloat16
            or not torch.isfinite(swiglu_output).all()
        ):
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
        if (
            contraction_output.dtype != torch.bfloat16
            or not torch.isfinite(contraction_output).all()
        ):
            raise RuntimeError("W8A8 BF16 contraction self-test failed")
    torch.cuda.synchronize(device)
    _PREFLIGHTED_KITCHEN.add(key)
