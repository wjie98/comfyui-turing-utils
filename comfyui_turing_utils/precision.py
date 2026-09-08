"""Turing precision policy and ConvRot runtime preparation."""

from __future__ import annotations

import dataclasses
from importlib.metadata import PackageNotFoundError, version

import torch

from .attention import (
    bundled_available,
    bundled_w8a8_available,
    normalize_attention_backend,
    preflight_bundled,
    preflight_bundled_w8a8,
)
from .hardware import is_supported_tensor_core_device, is_supported_turing_device
from .kernel_api import load_kernel_package
from .log import get_logger
from .quantization.dispatch import (
    backend_available,
    preflight_kitchen,
    preflight_codebook_w4a8,
    preflight_w4a8,
    register_backend,
)


LOG = get_logger("precision")
MIN_KITCHEN_VERSION = (0, 2, 26)
MIN_KERNEL_VERSION = (0, 8, 0)
MIN_CODEBOOK_W4A8_KERNEL_VERSION = (0, 24, 0)
_CONVROT_W4_LAYOUT = "TensorCoreConvRotW4A4Layout"
_CODEBOOK_W4_LAYOUT = "AsymW4A8Int8Layout"
_TENSORWISE_INT8_LAYOUT = "TensorWiseINT8Layout"


def _explicit_dtype_override() -> bool:
    try:
        import comfy.model_management as model_management
    except ImportError:
        return False
    args = getattr(model_management, "args", None)
    return bool(
        getattr(model_management, "FORCE_FP32", False)
        or getattr(args, "fp32_unet", False)
        or getattr(args, "fp64_unet", False)
        or getattr(args, "fp16_unet", False)
        or getattr(args, "fp8_e4m3fn_unet", False)
        or getattr(args, "fp8_e5m2_unet", False)
        or getattr(args, "fp8_e8m0fnu_unet", False)
    )


def _version_tuple(value: str) -> tuple[int, int, int]:
    numeric = []
    for part in value.split(".")[:3]:
        digits = "".join(character for character in part if character.isdigit())
        numeric.append(int(digits) if digits else 0)
    return tuple((numeric + [0, 0, 0])[:3])


def _check_kitchen_contract() -> None:
    try:
        kitchen_version = version("comfy-kitchen")
    except PackageNotFoundError as exc:
        raise RuntimeError("Turing ConvRot execution requires comfy-kitchen") from exc
    if _version_tuple(kitchen_version) < MIN_KITCHEN_VERSION:
        required = ".".join(str(value) for value in MIN_KITCHEN_VERSION)
        raise RuntimeError(
            f"Turing Utils requires comfy-kitchen>={required}, got {kitchen_version}. Update ComfyUI."
        )

    from comfy_kitchen.backends import cuda as kitchen_cuda

    required_cuda_api = (
        "convrot_w4a4_linear",
        "int8_linear",
        "quantize_int4_rowwise",
        "quantize_int4_rowwise_convrot64",
        "quantize_int8_rowwise_convrot64",
    )
    missing = [name for name in required_cuda_api if not hasattr(kitchen_cuda, name)]
    if missing:
        raise RuntimeError(
            "The installed comfy-kitchen does not provide the Turing ConvRot API required by Turing Utils: "
            + ", ".join(missing)
        )


def _check_kernel_contract(minimum_version=MIN_KERNEL_VERSION) -> None:
    try:
        kernel = load_kernel_package()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The independently installed comfyui-turing-utils-kernel is unavailable; reinstall ./kernel"
        ) from exc
    kernel_version = getattr(kernel, "__version__", "0.0.0")
    if _version_tuple(kernel_version) < minimum_version:
        required = ".".join(str(value) for value in minimum_version)
        raise RuntimeError(
            f"Turing Utils requires comfyui-turing-utils-kernel>={required}, "
            f"got {kernel_version}; reinstall the independent kernel package"
        )


def prepare_turing_runtime(
    summary,
    device: torch.device,
    attention_backend: str | None = None,
) -> None:
    """Register and validate the shared sm75+ runtime for one loader."""
    if not is_supported_tensor_core_device(device):
        return

    if attention_backend is not None:
        attention_backend = normalize_attention_backend(attention_backend)
    codebook_w4a8 = bool(getattr(summary, "codebook_w4a8", 0))
    bundled_w8a8 = attention_backend == "w8a8"
    bundled_sage = (
        attention_backend == "sage" and is_supported_turing_device(device)
    )
    needs_kernel = bool(
        summary.w4a4 or summary.w4a8 or codebook_w4a8 or summary.w8a8
    ) or bundled_w8a8 or bundled_sage
    if needs_kernel:
        _check_kernel_contract()
    if codebook_w4a8:
        _check_kernel_contract(MIN_CODEBOOK_W4A8_KERNEL_VERSION)

    if summary.w4a4 or summary.w4a8 or codebook_w4a8 or summary.w8a8:
        _check_kitchen_contract()
        import comfy_kitchen

        cuda_status = comfy_kitchen.list_backends().get("cuda", {})
        if not cuda_status.get("available"):
            reason = cuda_status.get("unavailable_reason") or "not installed"
            raise RuntimeError(f"Kitchen CUDA backend is unavailable: {reason}")
        capabilities = set(cuda_status.get("capabilities", ()))
        if (summary.w4a4 or summary.w4a8) and "convrot_w4a4_linear" not in capabilities:
            raise RuntimeError("Kitchen ConvRot W4 support is unavailable")
        if codebook_w4a8 and "w4a8_int8_linear" not in capabilities:
            raise RuntimeError("Kitchen grouped-codebook W4A8 support is unavailable")
        if summary.w8a8 and "int8_linear" not in capabilities:
            raise RuntimeError("Kitchen W8A8 support is unavailable")
        if not register_backend() or not backend_available():
            raise RuntimeError("the bundled ConvRot backend could not be registered")
        preflight_kitchen(device, bool(summary.w4a4), bool(summary.w8a8))
        if summary.w4a8:
            preflight_w4a8(device)
        if codebook_w4a8:
            preflight_codebook_w4a8(device)

    if bundled_w8a8:
        if not bundled_w8a8_available():
            raise RuntimeError("the bundled sm75+ W8A8 attention extension is unavailable")
        preflight_bundled_w8a8(device)
    elif bundled_sage:
        if not bundled_available():
            raise RuntimeError("the bundled Turing Sage extensions are unavailable")
        preflight_bundled(device)


def select_compute_dtype(
    model_config,
    device: torch.device,
) -> torch.dtype | None:
    """Replace only an exact-sm75 FP32 fallback with BF16 storage/compute."""
    if model_config is None or _explicit_dtype_override():
        return None
    supported = tuple(getattr(model_config, "supported_inference_dtypes", ()) or ())
    if torch.bfloat16 not in supported:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    if capability != (7, 5):
        return None
    if not is_supported_turing_device(device):
        LOG.warning(
            "BF16 forcing is disabled for %s because the bundled SM75 kernels require a Turing GPU with tensor cores",
            torch.cuda.get_device_name(index),
        )
        return None
    if torch.float16 in supported:
        LOG.info("Keeping ComfyUI's native FP16 inference mode on Turing")
        return None
    LOG.info("Replacing the Turing FP32 fallback with BF16 activation storage")
    return torch.bfloat16


def normalize_turing_convrot_weight_dtypes(
    model,
    device: torch.device,
    compute_dtype: torch.dtype | None,
) -> int:
    """Align only ConvRot quantized wrappers with the BF16 execution boundary."""
    if compute_dtype is not torch.bfloat16 or not is_supported_turing_device(device):
        return 0

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError as exc:
        raise RuntimeError("ComfyUI quantized tensor support is unavailable") from exc

    root = getattr(model, "model", model)
    normalized = 0
    matched = 0
    for module_name, module in root.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, QuantizedTensor):
            continue

        layout = getattr(weight, "_layout_cls", None)
        params = getattr(weight, "_params", None)
        is_convrot = layout in {_CONVROT_W4_LAYOUT, _CODEBOOK_W4_LAYOUT} or (
            layout == _TENSORWISE_INT8_LAYOUT and bool(getattr(params, "convrot", False))
        )
        if not is_convrot:
            continue

        matched += 1
        if weight.dtype is not torch.bfloat16 or getattr(params, "orig_dtype", None) is not torch.bfloat16:
            try:
                normalized_params = dataclasses.replace(params, orig_dtype=torch.bfloat16)
                normalized_weight = QuantizedTensor(weight._qdata, weight._layout_cls, normalized_params)
                normalized_parameter = torch.nn.Parameter(normalized_weight, requires_grad=False)
                normalized_parameter._params = normalized_params
                module.register_parameter("weight", normalized_parameter)
            except Exception as exc:
                layer = module_name or "<root>"
                raise RuntimeError(f"Could not normalize Turing ConvRot weight dtype for {layer}") from exc
            normalized += 1

        module.weight_comfy_model_dtype = torch.bfloat16

    if matched:
        LOG.info(
            "Turing ConvRot logical BF16 weights: matched=%d, normalized_from_other_dtype=%d",
            matched,
            normalized,
        )
    return normalized
