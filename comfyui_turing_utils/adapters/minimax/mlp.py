"""H3 row/channel/half-width MLP execution."""

from __future__ import annotations

import torch
from ..methods import OriginalMethod, weak_method
from ...hardware import is_supported_attention_device
from ...profiling import CUDA_PHASE_PROFILER
from .activation_policy import decide_activation_chunks, decide_ffn_channels
from .memory_state import ensure_dynamic_vram_headroom
from ...quantization.fusions import (
    convrot_linear_input_act_from_weight,
    convrot_w8_output_slice,
    convrot_w8_plain_tensors,
    fused_convrot_linear_input_act,
    is_turing_convrot_linear,
)
from ...quantization.dispatch import (
    int8_linear_from_quantized,
    quantize_convrot_swiglu_activation,
    quantize_convrot_swiglu_with_scale,
    quantize_convrot_from_partials,
    rotate_convrot_swiglu_shard_inplace,
    convrot_swiglu_channel_sharding_available,
    convrot_swiglu_half_width_available,
)
from .execution import (
    quantize_linear_rows,
    _profile_cuda,
    _direct_int8_output_available,
    _runtime_activation_plan,
    _weak_model_reference,
    _RuntimeDispatchAudit,
    _linear_with_cast_weight,
)


def _stream_mlp(mlp: torch.nn.Module, x: torch.Tensor, chunk_rows: int):
    """Evaluate an H3 SwiGLU MLP in row tiles with one weight cast per layer."""
    import comfy.ops

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = _profile_cuda(
        "minimax.mlp.fc1_weight_wait",
        comfy.ops.cast_bias_weight,
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=False,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = _profile_cuda(
            "minimax.mlp.fc2_weight_wait",
            comfy.ops.cast_bias_weight,
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        output_features = int(getattr(mlp.fc2, "out_features", x.shape[-1]))
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for start in range(0, x.shape[0], chunk_rows):
            stop = min(start + chunk_rows, x.shape[0])
            expanded = _profile_cuda(
                "minimax.mlp.fc1_tile",
                _linear_with_cast_weight,
                mlp.fc1,
                x[start:stop],
                fc1_weight,
                fc1_bias,
            )
            if _direct_int8_output_available():
                tile = _profile_cuda(
                    "minimax.mlp.swiglu_fc2_tile",
                    convrot_linear_input_act_from_weight,
                    fc2_weight,
                    fc2_bias,
                    expanded,
                    "swiglu",
                    output[start:stop],
                )
            else:
                tile = _profile_cuda(
                    "minimax.mlp.swiglu_fc2_tile",
                    convrot_linear_input_act_from_weight,
                    fc2_weight,
                    fc2_bias,
                    expanded,
                    "swiglu",
                )
                _profile_cuda(
                    "minimax.mlp.output_store", output[start:stop].copy_, tile
                )
            del expanded, tile
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(mlp.fc2, fc2_weight, fc2_bias, fc2_stream)
        comfy.ops.uncast_bias_weight(mlp.fc1, fc1_weight, fc1_bias, fc1_stream)


def _ffn_expanded_shard(
    mlp,
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    expanded_size: int,
    start: int,
    stop: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Project one aligned gate/up interval without a full fc1 output."""
    width = stop - start
    expanded = torch.empty(
        (qactivation.shape[0], 2 * width),
        dtype=output_dtype,
        device=qactivation.device,
    )
    direct = _direct_int8_output_available()
    gate = convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        start,
        stop,
        output_dtype,
        output=expanded[:, :width] if direct else None,
    )
    if not direct:
        expanded[:, :width].copy_(gate)
    del gate
    up = convrot_w8_output_slice(
        qactivation,
        activation_scale,
        qweight,
        weight_scale,
        bias,
        expanded_size + start,
        expanded_size + stop,
        output_dtype,
        output=expanded[:, width:] if direct else None,
    )
    if not direct:
        expanded[:, width:].copy_(up)
    del up
    return expanded


def _stream_mlp_channels(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    *,
    chunk_rows: int,
    chunk_channels: int,
):
    """Exact two-pass H3 FFN evaluation over ConvRot-aligned channels.

    The first pass finds the same whole-row activation scale as the unsharded
    fused SwiGLU quantizer.  The second pass recomputes each aligned interval,
    uses that common scale, and writes directly into the final compact A8 row.
    The unchanged fused fc2 consumes that row once, so no shard boundary enters
    the contraction or its BF16 epilogue.
    """
    import comfy.ops

    if chunk_channels <= 0 or chunk_channels % 256:
        return None
    if getattr(mlp.fc2, "pre_quant_scale", None) is not None:
        # This is not used by MiniMax H3 today.  Falling back avoids changing
        # the ordering of a future SmoothQuant-style pre-scale.
        return None

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = _profile_cuda(
        "minimax.mlp.fc1_weight_wait",
        comfy.ops.cast_bias_weight,
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = _profile_cuda(
            "minimax.mlp.fc2_weight_wait",
            comfy.ops.cast_bias_weight,
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        fc1_plain = convrot_w8_plain_tensors(fc1_weight)
        fc2_plain = convrot_w8_plain_tensors(fc2_weight)
        if fc1_plain is None or fc2_plain is None or fc2_bias is not None:
            return None
        fc1_qweight, fc1_weight_scale = fc1_plain
        fc2_qweight, fc2_weight_scale = fc2_plain
        expanded_size = int(getattr(mlp.fc2, "in_features", 0))
        if (
            expanded_size <= 0
            or expanded_size % 256
            or fc1_qweight.shape[0] != 2 * expanded_size
            or fc2_qweight.shape[1] != expanded_size
        ):
            return None

        output_features = int(fc2_qweight.shape[0])
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for row_start in range(0, x.shape[0], chunk_rows):
            row_stop = min(row_start + chunk_rows, x.shape[0])
            qactivation, input_scale = _profile_cuda(
                "minimax.mlp.input_quantize",
                quantize_linear_rows,
                mlp.fc1,
                x[row_start:row_stop],
            )

            whole_row_scale = None
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                expanded = _profile_cuda(
                    "minimax.mlp.channel_scale_projection",
                    _ffn_expanded_shard,
                    mlp,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size,
                    start,
                    stop,
                    x.dtype,
                )
                local_quantized, local_scale = quantize_convrot_swiglu_activation(
                    expanded, 256
                )
                del expanded, local_quantized
                if whole_row_scale is None:
                    whole_row_scale = local_scale
                else:
                    torch.maximum(whole_row_scale, local_scale, out=whole_row_scale)
                    del local_scale

            activated = torch.empty(
                (row_stop - row_start, expanded_size),
                dtype=torch.int8,
                device=x.device,
            )
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                expanded = _profile_cuda(
                    "minimax.mlp.channel_output_projection",
                    _ffn_expanded_shard,
                    mlp,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size,
                    start,
                    stop,
                    x.dtype,
                )
                direct = _direct_int8_output_available()
                quantized = quantize_convrot_swiglu_with_scale(
                    expanded,
                    whole_row_scale,
                    256,
                    output=activated[:, start:stop] if direct else None,
                )
                del expanded
                if not direct:
                    activated[:, start:stop].copy_(quantized)
                del quantized

            if _direct_int8_output_available():
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                    output=output[row_start:row_stop],
                )
            else:
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                )
                _profile_cuda(
                    "minimax.mlp.output_store",
                    output[row_start:row_stop].copy_,
                    tile,
                )
            del (
                activated,
                input_scale,
                qactivation,
                tile,
                whole_row_scale,
            )
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(mlp.fc2, fc2_weight, fc2_bias, fc2_stream)
        comfy.ops.uncast_bias_weight(mlp.fc1, fc1_weight, fc1_bias, fc1_stream)


def _stream_mlp_half_width(
    mlp: torch.nn.Module,
    x: torch.Tensor,
    *,
    chunk_rows: int,
    chunk_channels: int,
):
    """Single-pass exact FC1 with one half-width in-place ConvRot buffer.

    Gate channels are projected once into the final rotated workspace. Up
    channels are projected in aligned intervals, consumed immediately by the
    identical FP32 SwiGLU/FHT sequence, and released. The final reduction uses
    all channel partials, so row scale and INT8 values match the full-width
    fused quantizer without a second FC1 pass.
    """
    import comfy.ops

    if chunk_channels <= 0 or chunk_channels % 256:
        return None
    if getattr(mlp.fc2, "pre_quant_scale", None) is not None:
        return None

    comfy.ops.run_every_op()
    fc1_weight, fc1_bias, fc1_stream = _profile_cuda(
        "minimax.mlp.fc1_weight_wait",
        comfy.ops.cast_bias_weight,
        mlp.fc1,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    fc2_weight = fc2_bias = fc2_stream = None
    try:
        fc2_weight, fc2_bias, fc2_stream = _profile_cuda(
            "minimax.mlp.fc2_weight_wait",
            comfy.ops.cast_bias_weight,
            mlp.fc2,
            x,
            offloadable=True,
            compute_dtype=x.dtype,
            want_requant=True,
        )
        fc1_plain = convrot_w8_plain_tensors(fc1_weight)
        fc2_plain = convrot_w8_plain_tensors(fc2_weight)
        if fc1_plain is None or fc2_plain is None or fc2_bias is not None:
            return None
        fc1_qweight, fc1_weight_scale = fc1_plain
        fc2_qweight, fc2_weight_scale = fc2_plain
        expanded_size = int(getattr(mlp.fc2, "in_features", 0))
        if (
            expanded_size <= 0
            or expanded_size % 256
            or fc1_qweight.shape[0] != 2 * expanded_size
            or fc2_qweight.shape[1] != expanded_size
        ):
            return None

        output_features = int(fc2_qweight.shape[0])
        output = torch.empty(
            (x.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        for row_start in range(0, x.shape[0], chunk_rows):
            row_stop = min(row_start + chunk_rows, x.shape[0])
            qactivation, input_scale = _profile_cuda(
                "minimax.mlp.input_quantize",
                quantize_linear_rows,
                mlp.fc1,
                x[row_start:row_stop],
            )
            rotated_gate = _profile_cuda(
                "minimax.mlp.fc1_gate",
                convrot_w8_output_slice,
                qactivation,
                input_scale,
                fc1_qweight,
                fc1_weight_scale,
                fc1_bias,
                0,
                expanded_size,
                x.dtype,
            )
            partial_absmax = torch.empty(
                (row_stop - row_start, expanded_size // 256),
                dtype=torch.float32,
                device=x.device,
            )
            for start in range(0, expanded_size, chunk_channels):
                stop = min(start + chunk_channels, expanded_size)
                up = _profile_cuda(
                    "minimax.mlp.fc1_up_tile",
                    convrot_w8_output_slice,
                    qactivation,
                    input_scale,
                    fc1_qweight,
                    fc1_weight_scale,
                    fc1_bias,
                    expanded_size + start,
                    expanded_size + stop,
                    x.dtype,
                )
                _profile_cuda(
                    "minimax.mlp.swiglu_rotate_tile",
                    rotate_convrot_swiglu_shard_inplace,
                    rotated_gate,
                    up,
                    partial_absmax,
                    start,
                )
                del up

            activated, whole_row_scale = _profile_cuda(
                "minimax.mlp.activation_quantize",
                quantize_convrot_from_partials,
                rotated_gate,
                partial_absmax,
            )
            if _direct_int8_output_available():
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                    output=output[row_start:row_stop],
                )
            else:
                tile = _profile_cuda(
                    "minimax.mlp.fc2_tile",
                    int8_linear_from_quantized,
                    activated,
                    whole_row_scale,
                    fc2_qweight,
                    fc2_weight_scale,
                    bias=None,
                    out_dtype=x.dtype,
                )
                _profile_cuda(
                    "minimax.mlp.output_store",
                    output[row_start:row_stop].copy_,
                    tile,
                )
            del (
                activated,
                input_scale,
                partial_absmax,
                qactivation,
                rotated_gate,
                tile,
                whole_row_scale,
            )
        return output
    finally:
        if fc2_weight is not None:
            comfy.ops.uncast_bias_weight(mlp.fc2, fc2_weight, fc2_bias, fc2_stream)
        comfy.ops.uncast_bias_weight(mlp.fc1, fc1_weight, fc1_bias, fc1_stream)


def _make_mlp_forward(
    mlp: torch.nn.Module,
    audit: _RuntimeDispatchAudit,
    base_model=None,
):
    original = OriginalMethod.capture(mlp.forward, mlp)
    base_model = _weak_model_reference(base_model)

    def forward(self, x: torch.Tensor):
        blocker = None
        if x.dtype != torch.bfloat16:
            blocker = f"dtype={x.dtype}"
        elif not is_turing_convrot_linear(self.fc2):
            blocker = "fc2_not_turing_convrot"
        elif not is_supported_attention_device(x.device):
            blocker = f"device={x.device}"
        audit.record("mlp", blocker is None, x, blocker)
        if blocker is not None:
            return original(self, x)
        expanded_size = int(getattr(self.fc2, "in_features", 0))
        runtime_plan = _runtime_activation_plan(base_model)
        decision = decide_activation_chunks(
            x,
            operation="mlp",
            hidden_size=int(x.shape[-1]),
            expanded_size=expanded_size,
            runtime_plan=runtime_plan,
            base_model=base_model,
        )
        channel_decision = None
        if convrot_swiglu_channel_sharding_available():
            half_width = convrot_swiglu_half_width_available()
            channel_decision = decide_ffn_channels(
                x,
                expanded_size=expanded_size,
                chunk_rows=decision.chunk_rows,
                half_width=half_width,
                runtime_plan=runtime_plan,
                base_model=base_model,
            )
            if channel_decision.sharded:
                profile_shape = (int(x.shape[0]), int(x.shape[-1]))
                CUDA_PHASE_PROFILER.begin_operation(
                    "mlp",
                    profile_shape,
                    adapter="minimax",
                    path=(
                        "half_width_channel_sharded"
                        if half_width
                        else "channel_sharded_two_pass"
                    ),
                    row_chunk=channel_decision.chunk_rows,
                    channel_chunk=channel_decision.chunk_channels,
                )
                if base_model is not None:
                    ensure_dynamic_vram_headroom(
                        base_model,
                        x.device,
                        rows=int(x.shape[0]),
                        operation="ffn_channels",
                        estimated_peak_bytes=channel_decision.estimated_peak_bytes,
                        runtime_plan=runtime_plan,
                    )
                stream_function = (
                    _stream_mlp_half_width if half_width else _stream_mlp_channels
                )
                output = stream_function(
                    self,
                    x,
                    chunk_rows=channel_decision.chunk_rows,
                    chunk_channels=channel_decision.chunk_channels,
                )
                if output is not None:
                    CUDA_PHASE_PROFILER.complete_operation("mlp", profile_shape)
                    return output
                CUDA_PHASE_PROFILER.cancel_operation()
        profile_shape = (int(x.shape[0]), int(x.shape[-1]))
        CUDA_PHASE_PROFILER.begin_operation(
            "mlp",
            profile_shape,
            adapter="minimax",
            path="row_streamed" if decision.streamed else "full",
            row_chunk=decision.chunk_rows,
            channel_chunk=0,
        )
        if base_model is not None:
            ensure_dynamic_vram_headroom(
                base_model,
                x.device,
                rows=int(x.shape[0]),
                operation="mlp",
                estimated_peak_bytes=decision.streamed_peak_bytes,
                runtime_plan=runtime_plan,
            )
        if decision.streamed:
            output = _stream_mlp(self, x, decision.chunk_rows)
        else:
            expanded = _profile_cuda("minimax.mlp.fc1", self.fc1, x)
            output = _profile_cuda(
                "minimax.mlp.swiglu_fc2",
                fused_convrot_linear_input_act,
                self.fc2,
                expanded,
                "swiglu",
            )
            del expanded
        CUDA_PHASE_PROFILER.complete_operation("mlp", profile_shape)
        return output

    return weak_method(forward, mlp)
