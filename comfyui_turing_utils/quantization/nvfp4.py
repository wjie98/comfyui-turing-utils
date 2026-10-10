"""Native NVFP4 storage with bounded S8 staging, using ComfyUI weight ownership."""

import torch
import torch.nn.functional as F
import comfy.ops
import comfy.model_management
from comfy.quant_ops import QuantizedTensor

from ..kernel_api import load_nvfp4_backend


def _input_values(x, pre_scale, swiglu):
    if swiglu:
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up
    return x if pre_scale is None else x * pre_scale


def nvfp4_operations(compute_dtype, device, *, full_precision_mm=False):
    disabled = (
        comfy.ops.get_disabled_quant_formats(device) - {"nvfp4"}
        if device.type == "cuda"
        else set(comfy.ops.QUANT_ALGOS)
    )
    base = comfy.ops.mixed_precision_ops(
        compute_dtype=compute_dtype,
        disabled=disabled,
        full_precision_mm=full_precision_mm,
    )

    class NVFP4Ops(base):
        class Linear(base.Linear):
            def __init__(
                self, in_features, out_features, bias=True, device=None, dtype=None
            ):
                super().__init__(
                    in_features, out_features, bias=bias, device=device, dtype=dtype
                )
                # Text encoders choose their own dtype during construction.
                # Do not force all components of a multi-encoder CLIP to one dtype.
                if compute_dtype is None:
                    self.factory_kwargs["dtype"] = dtype
                    if self.bias is not None:
                        self.bias = torch.nn.Parameter(
                            self.bias.to(dtype=dtype), requires_grad=False
                        )

            def _forward(self, input, weight, bias, pre_scale=None, swiglu=False):
                if not (
                    isinstance(weight, QuantizedTensor)
                    and self.quant_format == "nvfp4"
                    and input.device.type == "cuda"
                    and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and input.ndim >= 1
                    and input.numel() > 0
                    and not self._full_precision_mm_config
                    and not input.requires_grad
                ):
                    return super()._forward(
                        _input_values(input, pre_scale, swiglu), weight, bias
                    )
                params = weight._params
                matrix = input.reshape(-1, input.shape[-1])
                output = load_nvfp4_backend().linear(
                    matrix,
                    weight._qdata,
                    params.block_scale,
                    params.scale,
                    bias,
                    output_columns=self.out_features,
                    pre_scale=pre_scale,
                    swiglu=swiglu,
                )
                return output.reshape(*input.shape[:-1], self.out_features)

            def forward(self, input, *args, **kwargs):
                if self.quant_format != "nvfp4" or input.requires_grad:
                    return super().forward(input, *args, **kwargs)
                return self._forward_input(input)

            def forward_swiglu(self, input):
                return self._forward_input(input, swiglu=True)

            def _forward_input(self, input, swiglu=False):
                if self.quant_format != "nvfp4" or input.requires_grad:
                    return super().forward(_input_values(input, None, swiglu))
                comfy.ops.run_every_op()
                pre_quant_scale = getattr(self, "pre_quant_scale", None)
                if pre_quant_scale is not None:
                    pre_quant_scale = comfy.model_management.cast_to_device(
                        pre_quant_scale, input.device, input.dtype
                    )
                with comfy.ops.CastBiasWeightContext(
                    self,
                    dtype=self.weight.dtype,
                    device=input.device,
                    bias_dtype=input.dtype,
                    compute_dtype=input.dtype,
                    offloadable=True,
                    want_requant=True,
                ) as (weight, bias):
                    # Weight hooks can return a dense patched tensor. Do not
                    # cache it or bypass the hook with a stale prepared weight.
                    if not isinstance(weight, QuantizedTensor):
                        return F.linear(
                            _input_values(input, pre_quant_scale, swiglu),
                            weight.to(input.dtype),
                            bias,
                        )
                    if self._full_precision_mm_config or input.device.type != "cuda":
                        return F.linear(
                            _input_values(input, pre_quant_scale, swiglu),
                            weight.dequantize().to(input.dtype),
                            bias,
                        )
                    return self._forward(input, weight, bias, pre_quant_scale, swiglu)

    return NVFP4Ops
