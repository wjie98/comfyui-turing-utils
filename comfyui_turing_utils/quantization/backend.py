"""Scoped comfy-kitchen registration and operator constraints."""

from __future__ import annotations

from types import SimpleNamespace
import torch
from ..hardware import is_supported_tensor_core_device
from .formats import grouped_weight_geometry
from .capabilities import (
    BACKEND_NAME,
    kernel_available as _kernel_available,
    kitchen_backend_available as backend_available,
)
from .dispatch import (
    LOG,
    _convrot_int8_bf16_rowbuffer_fits,
    codebook_w4a8_linear,
    convrot_w4a4_linear,
    int8_linear,
)


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
    has_w4a4_quantizer = _kernel_available("turing_bf16_int4_convrot_quantize")
    has_codebook_w4a8_kernel = _kernel_available("turing_codebook_w4a8_linear")

    def require_convrot_256(kwargs):
        if kwargs.get("convrot") is not True:
            return ValidationResult.fail("convrot", "staged INT8 requires ConvRot")
        if kwargs.get("convrot_groupsize") != 256:
            return ValidationResult.fail(
                "convrot_groupsize", "staged INT8 requires group size 256"
            )
        x = kwargs.get("x")
        weight = kwargs.get("weight")
        input_act = kwargs.get("input_act")
        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            return ValidationResult.fail("x", "Turing W8A8 requires tensor inputs")
        if kwargs.get("out_dtype") != x.dtype:
            return ValidationResult.fail(
                "out_dtype", "local W8A8 preserves the activation dtype"
            )
        hidden = x.shape[-1] // 2 if input_act == "swiglu" else x.shape[-1]
        if hidden % 16 or weight.shape[0] % 8 or weight.shape[1] != hidden:
            return ValidationResult.fail(
                "weight", "Turing W8A8 requires matching K%16=0 and N%8=0"
            )
        rowbuffer = _convrot_int8_bf16_rowbuffer_fits(hidden, x.device)
        if x.dtype == torch.float16:
            if (
                hidden % 256
                or input_act not in (None, "none", "swiglu", "gelu_tanh", "rms_norm")
                or not _kernel_available("turing_fp16_int8_convrot_quantize")
                or not _kernel_available("turing_fp16_int8_linear")
            ):
                return ValidationResult.fail(
                    "x", "no compatible FP16 ConvRot kernel is available"
                )
            norm_weight = kwargs.get("input_act_weight")
            if input_act == "rms_norm" and (
                not isinstance(norm_weight, torch.Tensor)
                or norm_weight.dtype not in (torch.float16, torch.float32)
            ):
                return ValidationResult.fail(
                    "input_act_weight",
                    "FP16 fused RMSNorm requires an FP16/FP32 weight",
                )
            return ValidationResult.ok()
        if input_act in (None, "none"):
            local_quantizer = rowbuffer and _kernel_available(
                "turing_bf16_int8_convrot_quantize"
            )
        elif input_act == "swiglu":
            local_quantizer = (
                rowbuffer and _kernel_available("turing_bf16_int8_convrot_quantize")
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
            return ValidationResult.fail(
                "convrot_groupsize", "W4 requires ConvRot group size 256"
            )
        if kwargs.get("quant_group_size") != 64:
            return ValidationResult.fail(
                "quant_group_size", "W4 requires quantization group size 64"
            )
        if kwargs.get("linear_dtype") not in {"int4", "int8"}:
            return ValidationResult.fail(
                "linear_dtype", "W4 requires int4 or int8 activation"
            )
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
        if kwargs.get("out_dtype") is not torch.bfloat16:
            return ValidationResult.fail(
                "out_dtype", "codebook W4A8 local output must be BF16"
            )
        x = kwargs.get("x")
        qdata = kwargs.get("qdata")
        codebook = kwargs.get("codebook")
        s_rel = kwargs.get("s_rel")
        if not all(isinstance(value, torch.Tensor) for value in (x, qdata, s_rel)):
            return ValidationResult.fail("x", "codebook W4A8 requires tensor inputs")
        try:
            _, logical_k, bits = grouped_weight_geometry(
                qdata.shape, s_rel.shape, kwargs.get("group_size", 16)
            )
        except ValueError as exc:
            return ValidationResult.fail("qdata", str(exc))
        if (bits == 4 and (codebook is None or codebook.numel() != 16)) or (
            bits == 6 and codebook is not None
        ):
            return ValidationResult.fail(
                "codebook", "W4 requires 16 levels; W6 requires no codebook"
            )
        if qdata.shape[0] % 8 or logical_k != x.shape[-1]:
            return ValidationResult.fail(
                "qdata", "codebook W4A8 requires matching packed K and N%8=0"
            )
        return ValidationResult.ok()

    operations = {}
    implementations = {}
    if "int8_linear" in cuda_capabilities and _kernel_available("turing_int8_linear"):
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
                "out_dtype": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16, torch.float16})
                ),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
                "input_act_weight": ParamConstraint(dtypes=standard_floats),
                "input_act_eps": ParamConstraint(dtypes=frozenset({float})),
                "residual": ParamConstraint(dtypes=standard_floats),
                "residual_scale": ParamConstraint(dtypes=standard_floats),
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
    if "w4a8_int8_linear" in cuda_capabilities and has_codebook_w4a8_kernel:
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
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float32}),
                    shape_rules=(ExactDims(2),),
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
    LOG.debug(
        "Registered scoped sm75+ operator backend: name=%s operators=%s global_priority=unchanged",
        BACKEND_NAME,
        ",".join(sorted(operations)),
    )
    return True
