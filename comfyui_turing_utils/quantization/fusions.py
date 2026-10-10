"""Generic sm75 fusions; model-specific installation lives in adapters."""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence

import torch

from ..kernel_api import load_kernel_package
from .formats import convrot_storage_kind


def convrot_weight_kind(weight: torch.Tensor) -> str | None:
    params = getattr(weight, "_params", None)
    if getattr(params, "orig_dtype", None) is not torch.bfloat16:
        return None
    return convrot_storage_kind(weight)


def is_turing_convrot_linear(linear: torch.nn.Module) -> bool:
    return convrot_weight_kind(getattr(linear, "weight", None)) is not None


def convrot_w8_plain_tensors(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return W8 ConvRot storage without materializing a BF16 weight."""
    if convrot_weight_kind(weight) != "w8a8":
        return None
    import comfy.quant_ops

    return comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)


def convrot_w8_output_slice(
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    start: int,
    stop: int,
    output_dtype: torch.dtype,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a contiguous W8 output-channel interval."""
    from .dispatch import int8_linear_from_quantized

    start, stop = int(start), int(stop)
    if start < 0 or stop <= start or stop > qweight.shape[0]:
        raise ValueError("W8 output slice is outside the weight")
    sliced_scale = (
        weight_scale
        if weight_scale.numel() == 1
        else weight_scale.reshape(-1)[start:stop]
    )
    sliced_bias = None if bias is None else bias[start:stop]
    return int8_linear_from_quantized(
        qactivation,
        activation_scale,
        qweight[start:stop],
        sliced_scale,
        bias=sliced_bias,
        out_dtype=output_dtype,
        output=output,
    )


def convrot_linear_input_act_from_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    x: torch.Tensor,
    input_act: str,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a fused ConvRot activation against an already-cast weight.

    Keeping the cast outside this helper lets activation streaming reuse one
    Dynamic-VRAM transfer for every row tile.  The quantized-linear dispatcher
    owns architecture-specific CUDA selection; the mathematical path is shared
    by every supported Tensor Core generation.
    """
    import comfy.model_management
    import comfy.ops
    import comfy.quant_ops

    from .dispatch import (
        codebook_w4a8_linear,
        convrot_w4a4_linear,
        int8_linear,
    )

    kind = convrot_weight_kind(weight)
    if kind is None or comfy.model_management.in_training:
        activated = comfy.ops.INPUT_ACT_EAGER[input_act](x)
        result = torch.nn.functional.linear(activated, weight, bias)
        if output is not None:
            output.copy_(result)
            return output
        return result
    if kind == "w8a8":
        qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
        return int8_linear(
            x,
            qdata,
            scale,
            bias=bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=weight._params.convrot_groupsize,
            input_act=input_act,
            output=output,
        )
    if kind in {"codebook_w4a8", "w6a8"}:
        qdata, s_rel, s_channel, correction, codebook = (
            comfy.quant_ops.AsymW4A8Int8Layout.get_plain_tensors(weight)
        )
        params = weight._params
        result = codebook_w4a8_linear(
            x,
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            bias=bias,
            group_size=params.group_size,
            convrot_groupsize=params.convrot_groupsize,
            out_dtype=x.dtype,
            input_act=input_act,
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    qdata, scale = comfy.quant_ops.TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
    params = weight._params
    result = convrot_w4a4_linear(
        x,
        qdata,
        scale,
        bias=bias,
        convrot_groupsize=params.convrot_groupsize,
        quant_group_size=params.quant_group_size,
        linear_dtype=params.linear_dtype,
        input_act=input_act,
    )
    if output is not None:
        output.copy_(result)
        return output
    return result


def fused_convrot_linear_input_act(
    linear: torch.nn.Module,
    x: torch.Tensor,
    input_act: str,
) -> torch.Tensor:
    """Fold an activation into a ConvRot quantizer on any sm75+ GPU."""
    import comfy.model_management
    import comfy.ops

    from ..hardware import is_supported_attention_device

    if (
        x.dtype != torch.bfloat16
        or comfy.model_management.in_training
        or not is_supported_attention_device(x.device)
        or not is_turing_convrot_linear(linear)
    ):
        return comfy.ops.linear_input_act(linear, x, input_act)

    weight, bias, offload_stream = comfy.ops.cast_bias_weight(
        linear,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        return convrot_linear_input_act_from_weight(weight, bias, x, input_act)
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, offload_stream)


# Compatibility name retained for existing imports and external workflows.
turing_linear_input_act = fused_convrot_linear_input_act


def _normalized_segments(
    segments: Sequence[tuple[int, int, int]],
    rows: int,
    parameter_rows: int,
) -> tuple[int, ...]:
    if rows <= 0 or parameter_rows <= 0:
        raise ValueError("segmented RMSNorm+AdaLN dimensions must be positive")
    flat: list[int] = []
    cursor = 0
    for segment in segments:
        if len(segment) != 3:
            raise ValueError(
                "each modulation segment must contain start, stop, and row"
            )
        try:
            start, stop, modulation_row = (operator.index(value) for value in segment)
        except TypeError as exc:
            raise ValueError("modulation segment values must be integers") from exc
        if start != cursor or stop <= start:
            raise ValueError("modulation segments must cover the input contiguously")
        if modulation_row < 0 or modulation_row >= parameter_rows:
            raise ValueError("modulation segment row is outside the scale/shift table")
        flat.extend((start, stop, modulation_row))
        cursor = stop
    if cursor != rows:
        raise ValueError("modulation segments must cover every input row")
    return tuple(flat)


@functools.lru_cache(maxsize=32)
def _cached_segment_table(flat: tuple[int, ...], device_index: int) -> torch.Tensor:
    return torch.tensor(
        flat, dtype=torch.int32, device=torch.device("cuda", device_index)
    ).view(-1, 3)


def _segment_table(
    segments: Sequence[tuple[int, int, int | torch.Tensor]] | torch.Tensor,
    rows: int,
    parameter_rows: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(segments, torch.Tensor):
        return segments
    if any(isinstance(row, torch.Tensor) for _, _, row in segments):
        return indexed_modulation_rows(segments, rows, parameter_rows, device)
    flat = _normalized_segments(segments, rows, parameter_rows)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _cached_segment_table(flat, index)


def indexed_modulation_rows(
    segments: Sequence[tuple[int, int, int | torch.Tensor]],
    rows: int,
    parameter_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack GPU row indices once per block, shared by its three fused ops."""
    _normalized_segments(
        [(a, b, 0 if isinstance(row, torch.Tensor) else row) for a, b, row in segments],
        rows,
        parameter_rows,
    )
    parts = []
    for start, stop, row in segments:
        if isinstance(row, torch.Tensor):
            if row.dtype not in (torch.int32, torch.int64):
                raise ValueError("per-token modulation indices must be int32 or int64")
            if row.ndim > 1 or row.numel() not in (1, stop - start):
                raise ValueError(
                    "per-token modulation indices must match their segment length"
                )
            parts.append(row.to(device=device, dtype=torch.int64).expand(stop - start))
        else:
            parts.append(
                torch.full((stop - start,), row, device=device, dtype=torch.int64)
            )
    return torch.cat(parts)


def segmented_rms_adaln(
    norm: torch.nn.Module,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: Sequence[tuple[int, int, int | torch.Tensor]] | torch.Tensor,
) -> torch.Tensor:
    """Run the bundled affine RMSNorm plus segmented AdaLN operator."""
    import comfy.ops

    comfy.ops.run_every_op()
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm, x, offloadable=True)
    try:
        if bias is not None:
            raise RuntimeError("segmented RMSNorm+AdaLN does not support a biased norm")
        if weight is None:
            weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        scale = scale.to(device=x.device, dtype=x.dtype)
        shift = shift.to(device=x.device, dtype=x.dtype)
        table = _segment_table(segments, x.shape[0], scale.shape[0], x.device)
        try:
            turing_segmented_rms_adaln = getattr(
                load_kernel_package(), "turing_segmented_rms_adaln"
            )
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError(
                "RMSNorm+AdaLN fusion requires an updated comfyui-turing-utils-kernel; reinstall the kernel package"
            ) from exc
        return turing_segmented_rms_adaln(
            x, weight, scale, shift, table, float(norm.eps)
        )
    finally:
        comfy.ops.uncast_bias_weight(norm, weight, bias, offload_stream)


def segmented_mod_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    segments: Sequence[tuple[int, int, int | torch.Tensor]] | torch.Tensor,
) -> torch.Tensor:
    """Apply MiniMax's segmented gated residual in-place with one CUDA launch."""
    table = _segment_table(segments, x.shape[0], gate.shape[0], x.device)
    try:
        op = getattr(load_kernel_package(), "turing_segmented_mod_gate")
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            "segmented mod-gate requires an updated comfyui-turing-utils-kernel"
        ) from exc
    op(x, gate, residual, table)
    return x


def segmented_mod_gate_rms_adaln(
    norm: torch.nn.Module,
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: Sequence[tuple[int, int, int | torch.Tensor]] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update ``x`` and normalize the dtype-rounded result in one CUDA kernel."""
    import comfy.ops

    comfy.ops.run_every_op()
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm, x, offloadable=True)
    try:
        if bias is not None:
            raise RuntimeError("fused segmented RMSNorm does not support a biased norm")
        if weight is None:
            weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        table = _segment_table(segments, x.shape[0], scale.shape[0], x.device)
        try:
            op = getattr(load_kernel_package(), "turing_segmented_mod_gate_rms_adaln")
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError(
                "gated residual+RMSNorm fusion requires an updated comfyui-turing-utils-kernel"
            ) from exc
        normalized = op(
            x,
            gate,
            residual,
            weight,
            scale,
            shift,
            table,
            float(norm.eps),
        )
        return x, normalized
    finally:
        comfy.ops.uncast_bias_weight(norm, weight, bias, offload_stream)
