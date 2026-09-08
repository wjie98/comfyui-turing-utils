"""ConvRot-aware ComfyUI loading orchestration.

Quantization metadata parsing intentionally lives in ``quantization.convrot``;
this module owns filesystem discovery, ComfyUI model construction, runtime
policy, and adapter installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import folder_paths
import torch

import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.utils

from ..adapters.dynamic_vram import install_dynamic_vram_sample_fence
from ..adapters.registry import apply_model_adapters
from ..attention import apply_attention_backend, normalize_attention_backend
from ..log import get_logger
from ..precision import (
    normalize_turing_convrot_weight_dtypes,
    prepare_turing_runtime,
    select_compute_dtype,
)
from ..quantization.convrot import (
    MODEL_EXTENSIONS,
    ConvRotSummary,
    _convrot_skip_reason,
    _summarize_convrot_modules,
    configure_convrot_activation,
)
from ..quantization.operator_scope import (
    install_clip_operator_scope,
    install_model_operator_scope,
)


LOG = get_logger("loader")
DIFFUSION_FOLDER_NAME = "diffusion_models"
CLIP_FOLDER_NAME = "text_encoders"


def official_clip_loader_inputs() -> dict:
    comfy_nodes = sys.modules.get("nodes")
    if comfy_nodes is None or not hasattr(comfy_nodes, "CLIPLoader"):
        raise RuntimeError("ComfyUI's official CLIPLoader is not available")
    inputs = comfy_nodes.CLIPLoader.INPUT_TYPES()
    if not isinstance(inputs, dict) or not isinstance(inputs.get("required"), dict):
        raise RuntimeError("ComfyUI's official CLIPLoader returned invalid input definitions")
    return inputs


def official_clip_types() -> tuple[str, ...]:
    required = official_clip_loader_inputs()["required"]
    if "type" not in required or not required["type"]:
        raise RuntimeError("ComfyUI's official CLIPLoader does not expose a type input")
    return tuple(required["type"][0])


def convrot_model_names(folder_name: str) -> list[str]:
    names = []
    for name in folder_paths.get_filename_list(folder_name):
        if Path(name).suffix.lower() not in MODEL_EXTENSIONS:
            continue
        model_path = folder_paths.get_full_path(folder_name, name)
        if model_path is None:
            LOG.debug("Skipping ConvRot candidate %s: folder_paths could not resolve it", name)
            continue
        skip_reason = _convrot_skip_reason(model_path)
        if skip_reason is None:
            names.append(name)
        else:
            log = LOG.warning if "convrot" in name.lower() else LOG.debug
            log("Skipping ConvRot candidate %s: %s", name, skip_reason)
    return names


def resolve_convrot_model_path(folder_name: str, model_name: str) -> str:
    model_path = folder_paths.get_full_path_or_raise(folder_name, model_name)
    skip_reason = _convrot_skip_reason(model_path)
    if skip_reason is not None:
        raise ValueError(f"{model_path} is not a supported ConvRot model: {skip_reason}")
    return model_path


def _loaded_convrot_summary(model) -> ConvRotSummary:
    return _summarize_convrot_modules(model.model)


def _loaded_convrot_clip_summary(clip) -> ConvRotSummary:
    return _summarize_convrot_modules(clip.cond_stage_model)


def validate_runtime_support(
    expected: ConvRotSummary,
    device: torch.device | None = None,
) -> None:
    if expected.w4a8 == 0 and expected.codebook_w4a8 == 0:
        return

    try:
        import comfy_kitchen
    except ImportError as exc:
        raise RuntimeError("ConvRot W4A8 requires comfy-kitchen") from exc

    cuda_backend = comfy_kitchen.list_backends().get("cuda", {})
    if not cuda_backend.get("available", False):
        reason = cuda_backend.get("unavailable_reason")
        detail = f": {reason}" if reason else ""
        raise RuntimeError(
            "ConvRot W4A8 requires the comfy-kitchen CUDA extension, but it is unavailable"
            f"{detail}. The eager ConvRot W4 path always computes A4, so this loader will not silently accept A8."
        )

    if device is None:
        device = comfy.model_management.get_torch_device()
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError(f"ConvRot W4A8 requires an NVIDIA CUDA load device, got {device}")
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise RuntimeError(
            f"ConvRot W4A8 requires NVIDIA Turing/sm75 or newer, got sm{capability[0]}{capability[1]}"
        )


def load_convrot_model(
    model_path: str | Path,
    force_int8_gemm: bool = False,
    attention_backend: str | None = "w8a8",
    *,
    disable_dynamic: bool = False,
):
    attention_backend = normalize_attention_backend(attention_backend)
    model_path = Path(model_path)
    state_dict, metadata = comfy.utils.load_torch_file(str(model_path), return_metadata=True)
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    if not any(key.startswith(diffusion_model_prefix) for key in state_dict):
        diffusion_model_prefix = ""
    metadata, expected = configure_convrot_activation(
        state_dict,
        metadata,
        force_int8_gemm,
        model_prefix=diffusion_model_prefix,
    )
    load_device = comfy.model_management.get_torch_device()
    validate_runtime_support(expected, load_device)
    prepare_turing_runtime(expected, load_device, attention_backend)
    model_config = comfy.model_detection.model_config_from_unet(
        state_dict,
        diffusion_model_prefix,
        metadata=metadata,
    )
    compute_dtype = select_compute_dtype(model_config, load_device)
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options={"dtype": compute_dtype} if compute_dtype is not None else {},
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError(f"ComfyUI could not detect a supported model config from {model_path}")

    if compute_dtype is not None:
        model.set_model_compute_dtype(compute_dtype)
        normalize_turing_convrot_weight_dtypes(model, load_device, compute_dtype)

    loaded = _loaded_convrot_summary(model)
    if loaded != expected:
        raise RuntimeError(
            "ConvRot activation selection was not applied to every layer: "
            f"expected {expected}, loaded {loaded}. Update ComfyUI and comfy-kitchen."
        )

    LOG.info(
        "Loaded ConvRot model with force_int8_gemm=%s: "
        "W4A4=%d, legacy_W4A8=%d, codebook_W4A8=%d, W8A8=%d",
        force_int8_gemm,
        loaded.w4a4,
        loaded.w4a8,
        loaded.codebook_w4a8,
        loaded.w8a8,
    )
    install_dynamic_vram_sample_fence(model, load_device)
    apply_model_adapters(model, load_device)
    apply_attention_backend(
        model,
        attention_backend,
        device=load_device,
        native_runtime=True,
    )
    install_model_operator_scope(model)
    model.cached_patcher_init = (
        load_convrot_model,
        (str(model_path), force_int8_gemm, attention_backend),
    )
    return model


def load_convrot_clip_patcher(
    model_path: str | Path,
    embedding_directory,
    clip_type,
    force_int8_gemm: bool,
    model_options: dict,
    disable_dynamic: bool = False,
):
    return load_convrot_clip(
        model_path,
        embedding_directory=embedding_directory,
        clip_type=clip_type,
        force_int8_gemm=force_int8_gemm,
        model_options=model_options,
        disable_dynamic=disable_dynamic,
    ).patcher


def load_convrot_clip(
    model_path: str | Path,
    *,
    embedding_directory=None,
    clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    force_int8_gemm: bool = False,
    model_options: dict | None = None,
    disable_dynamic: bool = False,
):
    model_path = Path(model_path)
    state_dict, metadata = comfy.utils.load_torch_file(
        str(model_path), safe_load=True, return_metadata=True
    )
    metadata, expected = configure_convrot_activation(state_dict, metadata, force_int8_gemm)

    model_options = dict(model_options or {})
    load_device = model_options.get("load_device", comfy.model_management.text_encoder_device())
    validate_runtime_support(expected, load_device)
    prepare_turing_runtime(expected, load_device)

    state_dict, metadata = comfy.utils.convert_old_quants(
        state_dict, model_prefix="", metadata=metadata
    )
    model_options["quantization_metadata"] = {"mixed_ops": True}
    clip = comfy.sd.load_text_encoder_state_dicts(
        [state_dict],
        embedding_directory=embedding_directory,
        clip_type=clip_type,
        model_options=model_options,
        disable_dynamic=disable_dynamic,
    )

    loaded = _loaded_convrot_clip_summary(clip)
    if loaded != expected:
        raise RuntimeError(
            "ConvRot activation selection was not applied to every CLIP layer: "
            f"expected {expected}, loaded {loaded}. Check the CLIP type and update ComfyUI and comfy-kitchen."
        )

    LOG.info(
        "Loaded ConvRot CLIP with force_int8_gemm=%s: "
        "W4A4=%d, legacy_W4A8=%d, codebook_W4A8=%d, W8A8=%d",
        force_int8_gemm,
        loaded.w4a4,
        loaded.w4a8,
        loaded.codebook_w4a8,
        loaded.w8a8,
    )
    install_clip_operator_scope(clip)
    clip.patcher.cached_patcher_init = (
        load_convrot_clip_patcher,
        (
            str(model_path),
            embedding_directory,
            clip_type,
            force_int8_gemm,
            model_options,
        ),
    )
    return clip


# Compatibility for workflows/tests that used the old private spellings.
_official_clip_loader_inputs = official_clip_loader_inputs
_official_clip_types = official_clip_types
_convrot_model_names = convrot_model_names
_resolve_convrot_model_path = resolve_convrot_model_path
_validate_runtime_support = validate_runtime_support
