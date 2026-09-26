"""ComfyUI-managed integration for SeC visual-concept video tracking."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths

from ..log import get_logger


LOG = get_logger("sec")
_MISSING_MODEL = "(No SeC models found in models/sams)"
_SINGLE_FILE_SUFFIXES = {".safetensors"}
_ATTENTION_MODES = {"auto", "sdpa"}


def register_sec_model_folder() -> None:
    """Register the established SeC/SAM model directory with ComfyUI."""

    folder_paths.add_model_folder_path(
        "sams",
        str(Path(folder_paths.models_dir) / "sams"),
    )


register_sec_model_folder()


@dataclass(frozen=True)
class SeCModelSpec:
    name: str
    path: Path
    config_path: Path
    single_file: bool


class _ManagedSeCModule(nn.Module):
    """Give Hugging Face's read-only ``device`` model a Comfy patcher shell."""

    def __init__(self, sec_model: nn.Module):
        super().__init__()
        self.sec_model = sec_model
        self.device = torch.device("cpu")


@dataclass
class SeCModelHandle:
    patcher: comfy.model_patcher.ModelPatcher
    dtype: torch.dtype
    source: str
    attention: "SeCAttentionPlan | None" = None

    @property
    def model(self):
        return self.patcher.model.sec_model


@dataclass(frozen=True)
class SeCVisualPrompt:
    mask: np.ndarray | None
    points: np.ndarray | None
    labels: np.ndarray | None
    box: np.ndarray | None


@dataclass(frozen=True)
class SeCAttentionPlan:
    requested: str
    vision: str
    llm: str
    tracker: str = "sdpa"


def _installed_package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return tuple(int(item) for item in torch.cuda.get_device_capability(device))
    except (AssertionError, RuntimeError, ValueError):
        return None


def _version_major(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value.split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def resolve_sec_attention(
    attention: str | bool,
    device: torch.device,
    dtype: torch.dtype,
) -> SeCAttentionPlan:
    """Resolve the best supported attention independently for each SeC subsystem."""

    if isinstance(attention, bool):
        requested = "auto" if attention else "sdpa"
    else:
        requested = str(attention).strip().lower()
    if requested not in _ATTENTION_MODES:
        raise ValueError(
            f"Unsupported SeC attention mode {attention!r}; expected auto or sdpa"
        )
    fallback = SeCAttentionPlan(requested=requested, vision="sdpa", llm="sdpa")
    if requested == "sdpa" or dtype not in (torch.float16, torch.bfloat16):
        return fallback

    capability = _device_capability(device)
    if capability is None:
        return fallback
    major, _minor = capability

    flash_version = _installed_package_version("flash_attn")
    flash_major = _version_major(flash_version)
    vision = "sdpa"
    llm = "sdpa"

    # On the supported PyTorch 2.9 baseline, SDPA is faster than the external
    # FA2 varlen path for InternViT's fixed-length image-token batches, so keep
    # vision on SDPA for Ampere and newer. Qwen still benefits from FA2. FA1
    # remains useful for InternViT on Turing when a compatible v1 wheel exists.
    if flash_major is not None and flash_major >= 2 and major >= 8:
        llm = "flash_attention_2"
    elif flash_major == 1 and capability >= (7, 5) and dtype == torch.float16:
        vision = "flash_attention_1"

    # Transformers exposes FA3 independently; Qwen can select it on Hopper or
    # newer while InternViT continues to use SDPA.
    if major >= 9 and _module_available("flash_attn_3"):
        llm = "flash_attention_3"

    return SeCAttentionPlan(
        requested=requested,
        vision=vision,
        llm=llm,
    )


def _bundled_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "sec" / "model_config"


def _model_roots() -> tuple[Path, ...]:
    try:
        return tuple(Path(path) for path in folder_paths.get_folder_paths("sams"))
    except KeyError:
        return ()


def available_sec_models() -> tuple[SeCModelSpec, ...]:
    """Find single-file and Hugging Face directory-format SeC checkpoints."""

    found: dict[str, SeCModelSpec] = {}
    bundled_config = _bundled_config_path()
    for root in _model_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _SINGLE_FILE_SUFFIXES:
                if "sec" not in path.relative_to(root).as_posix().lower():
                    continue
                # Shard files belong to their parent directory entry.
                if path.name.startswith("model-") and (path.parent / "config.json").is_file():
                    continue
                name = path.relative_to(root).as_posix()
                found.setdefault(
                    name,
                    SeCModelSpec(name, path, bundled_config, True),
                )
                continue
            if not path.is_dir() or not (path / "config.json").is_file():
                continue
            if "sec" not in path.relative_to(root).as_posix().lower():
                continue
            has_weights = any(
                (path / filename).is_file()
                for filename in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                    "pytorch_model.bin.index.json",
                )
            )
            if has_weights and (path / "tokenizer_config.json").is_file():
                name = f"{path.relative_to(root).as_posix()}/"
                found.setdefault(name, SeCModelSpec(name, path, path, False))
    return tuple(found[name] for name in sorted(found, key=str.casefold))


def sec_model_choices() -> list[str]:
    choices = [spec.name for spec in available_sec_models()]
    return choices or [_MISSING_MODEL]


def _resolve_model(model_name: str) -> SeCModelSpec:
    for spec in available_sec_models():
        if spec.name == model_name:
            return spec
    raise FileNotFoundError(
        f"SeC model {model_name!r} was not found. Put a SeC single-file checkpoint "
        "or Hugging Face model directory under ComfyUI/models/sams."
    )


def _dominant_float_dtype(state_dict: dict[str, Any]) -> torch.dtype:
    counts: dict[torch.dtype, int] = {}
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point():
            counts[value.dtype] = counts.get(value.dtype, 0) + value.numel()
    if not counts:
        raise ValueError("The SeC checkpoint contains no floating-point weights")
    dtype = max(counts, key=counts.__getitem__)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.float16
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported SeC checkpoint dtype: {dtype}")
    return dtype


def _convert_float8_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    # Replace tensors incrementally so a 4B checkpoint does not coexist with a
    # second, fully materialized fp16 state dict in host memory.
    for key in list(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value) and value.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            state_dict[key] = value.to(torch.float16)
    return state_dict


def _config_dtype(config) -> torch.dtype:
    value = getattr(config, "torch_dtype", None)
    if isinstance(value, torch.dtype):
        return value
    value = str(value or "").lower()
    return {
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(value, torch.float16)


def _register_parameter_dtype_input_hooks(model: nn.Module) -> int:
    """Align floating inputs with each parameterized module's weight dtype.

    SeC/SAM2 creates several positional and memory tensors in fp32 even when
    the released checkpoint is fp16 or bf16.  PyTorch's Linear and LayerNorm
    kernels require those inputs to match their parameters.  Keep the repair
    scoped to this model instance: the hooks change dtype only, while ComfyUI
    remains responsible for device placement and model residency.
    """

    handles = []
    for module in model.modules():
        parameter = next(module.parameters(recurse=False), None)
        if parameter is None or not parameter.is_floating_point():
            continue
        target_dtype = parameter.dtype

        def align_dtype(current_module, args, kwargs, *, dtype=target_dtype):
            del current_module

            def convert(value):
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
                    return value.to(dtype=dtype)
                return value

            return tuple(convert(value) for value in args), {
                key: convert(value) for key, value in kwargs.items()
            }

        handles.append(
            module.register_forward_pre_hook(align_dtype, with_kwargs=True)
        )
    # Retain handles for the lifetime of the model and make the installation
    # explicit for diagnostics.  Module hook dictionaries also own them, but
    # this gives us one place to inspect or remove them in future revisions.
    model._sec_dtype_input_hook_handles = handles
    return len(handles)


def load_sec_model(
    model_name: str,
    attention: str | bool = "auto",
) -> SeCModelHandle:
    """Load SeC on the offload device and register it with ComfyUI."""

    try:
        from accelerate import init_empty_weights
        from transformers import Qwen2Tokenizer

        from ..vendor.sec.configuration_sec import SeCConfig
        from ..vendor.sec.modeling_sec import SeCModel
    except ImportError as error:
        raise ImportError(
            "SeC dependencies are incomplete. Reinstall this custom node's "
            "requirements.txt before loading a SeC checkpoint."
        ) from error

    spec = _resolve_model(model_name)
    config = SeCConfig.from_pretrained(str(spec.config_path))
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    cpu_inference = comfy.model_management.is_device_cpu(load_device)

    fast_disk = False
    if spec.single_file:
        state_dict = comfy.utils.load_torch_file(str(spec.path), safe_load=True)
        dtype = _dominant_float_dtype(state_dict)
        if any(
            torch.is_tensor(value)
            and value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            for value in state_dict.values()
        ):
            LOG.warning(
                "SeC float8 checkpoint %s is expanded to fp16 because upstream SeC "
                "float8 inference is numerically unstable",
                spec.name,
            )
            state_dict = _convert_float8_state_dict(state_dict)
        if cpu_inference and dtype != torch.float32:
            LOG.warning(
                "SeC checkpoint %s will use fp32 because ComfyUI is running on CPU",
                spec.name,
            )
            dtype = torch.float32
        attention_plan = resolve_sec_attention(attention, load_device, dtype)
        with init_empty_weights(include_buffers=False):
            model = SeCModel(
                config,
                vision_attention_backend=attention_plan.vision,
                llm_attention_backend=attention_plan.llm,
            )
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        missing = [key for key in incompatible.missing_keys if not key.endswith("num_batches_tracked")]
        if missing or incompatible.unexpected_keys:
            raise ValueError(
                "SeC checkpoint does not match the bundled architecture: "
                f"missing={missing[:20]}, unexpected={incompatible.unexpected_keys[:20]}"
            )
        fast_disk = comfy.storage.state_dict_fast_disk(state_dict)
        del state_dict
    else:
        dtype = _config_dtype(config)
        attention_plan = resolve_sec_attention(attention, load_device, dtype)
        model = SeCModel.from_pretrained(
            str(spec.path),
            config=config,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            vision_attention_backend=attention_plan.vision,
            llm_attention_backend=attention_plan.llm,
        )
        dtype = next(
            (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
            dtype,
        )
        if cpu_inference and dtype != torch.float32:
            LOG.warning(
                "SeC checkpoint %s will use fp32 because ComfyUI is running on CPU",
                spec.name,
            )
            dtype = torch.float32

    resolved_backends = getattr(model, "sec_attention_backends", {})
    attention_plan = SeCAttentionPlan(
        requested=attention_plan.requested,
        vision=str(resolved_backends.get("vision", attention_plan.vision)),
        llm=str(resolved_backends.get("llm", attention_plan.llm)),
        tracker=str(resolved_backends.get("tracker", attention_plan.tracker)),
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(spec.config_path),
        local_files_only=True,
    )
    model.eval()
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=dtype)
    dtype_hook_count = 0
    if not cpu_inference and dtype != torch.float32:
        dtype_hook_count = _register_parameter_dtype_input_hooks(model)

    managed = _ManagedSeCModule(model).eval()
    # DynamicVRAM intentionally keeps parameter storage off-device and moves the
    # active workset at op time.  SeC and SAM2 otherwise infer their compute
    # device from the first stored parameter and incorrectly choose CPU.
    model._comfy_load_device = load_device
    model.grounding_encoder._comfy_load_device = load_device
    if offload_device.type != "cpu":
        managed.to(offload_device)
        managed.device = offload_device
    patcher = comfy.model_patcher.CoreModelPatcher(
        managed,
        load_device=load_device,
        offload_device=offload_device,
        fast_disk=fast_disk,
    )
    LOG.info(
        "Loaded SeC model %s on the ComfyUI offload device: dtype=%s "
        "attention(requested/vision/llm/tracker)=%s/%s/%s/%s "
        "dtype_hooks=%d size=%.2f GiB",
        spec.name,
        dtype,
        attention_plan.requested,
        attention_plan.vision,
        attention_plan.llm,
        attention_plan.tracker,
        dtype_hook_count,
        patcher.model_size() / (1024**3),
    )
    return SeCModelHandle(
        patcher=patcher,
        dtype=dtype,
        source=spec.name,
        attention=attention_plan,
    )


def parse_points(value: str | None, *, width: int, height: int, label: int) -> tuple[np.ndarray, np.ndarray]:
    if value is None or not str(value).strip():
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid point JSON: {error}") from error
    if not isinstance(decoded, list):
        raise ValueError("Point prompts must be a JSON list")
    points = []
    for index, item in enumerate(decoded):
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError(f"Point {index} must contain numeric x and y fields")
        try:
            x, y = float(item["x"]), float(item["y"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Point {index} has non-numeric coordinates") from error
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Point {index} ({x:g}, {y:g}) is outside the {width}x{height} frame"
            )
        points.append((x, y))
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    labels = np.full((len(array),), int(label), dtype=np.int32)
    return array, labels


def parse_bbox(value, *, width: int, height: int) -> np.ndarray | None:
    if value is None:
        return None
    current = value
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]
    if isinstance(current, dict):
        if {"startX", "startY", "endX", "endY"} <= current.keys():
            coords = (
                current["startX"],
                current["startY"],
                current["endX"],
                current["endY"],
            )
        elif {"x", "y", "width", "height"} <= current.keys():
            coords = (
                current["x"],
                current["y"],
                float(current["x"]) + float(current["width"]),
                float(current["y"]) + float(current["height"]),
            )
        else:
            raise ValueError("BBOX dictionaries must use startX/startY/endX/endY or x/y/width/height")
    elif isinstance(current, (list, tuple)) and len(current) == 4:
        coords = current
    else:
        raise ValueError(f"Unsupported BBOX value: {type(value).__name__}")
    try:
        x1, y1, x2, y2 = (float(item) for item in coords)
    except (TypeError, ValueError) as error:
        raise ValueError("BBOX coordinates must be numeric") from error
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBOX is empty after clipping to the {width}x{height} frame")
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _select_mask(mask: torch.Tensor, frames: torch.Tensor, annotation_frame_idx: int) -> np.ndarray:
    if mask.ndim == 2:
        selected = mask
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            selected = mask[0]
        elif mask.shape[0] == frames.shape[0]:
            selected = mask[annotation_frame_idx]
        else:
            raise ValueError(
                "mask must contain one mask or one mask per video frame; "
                f"got {mask.shape[0]} masks for {frames.shape[0]} frames"
            )
    else:
        raise ValueError(f"mask must be [H,W] or [N,H,W], got {tuple(mask.shape)}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    selected = selected.detach().to(device="cpu", dtype=torch.float32)
    if tuple(selected.shape) != (height, width):
        selected = F.interpolate(
            selected[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    mask = selected.numpy() >= 0.5
    if not mask.any():
        raise ValueError("mask contains no foreground on the annotation frame")
    return mask


def prepare_visual_prompt(
    frames: torch.Tensor,
    *,
    annotation_frame_idx: int,
    positive_coords: str | None,
    negative_coords: str | None,
    bounding_box,
    mask: torch.Tensor | None,
) -> SeCVisualPrompt:
    height, width = int(frames.shape[1]), int(frames.shape[2])
    positive, positive_labels = parse_points(
        positive_coords, width=width, height=height, label=1
    )
    negative, negative_labels = parse_points(
        negative_coords, width=width, height=height, label=0
    )
    box = parse_bbox(bounding_box, width=width, height=height)

    if mask is not None:
        selected_mask = _select_mask(mask, frames, annotation_frame_idx)
        if box is not None:
            x1, y1, x2, y2 = box
            roi = np.zeros_like(selected_mask)
            roi[int(np.floor(y1)) : int(np.ceil(y2)), int(np.floor(x1)) : int(np.ceil(x2))] = True
            selected_mask &= roi
            if not selected_mask.any():
                raise ValueError("bounding_box does not overlap the mask foreground")
        for point in positive:
            x, y = int(point[0]), int(point[1])
            if not selected_mask[y, x]:
                raise ValueError(f"Positive point ({x}, {y}) is outside the authoritative mask")
        for point in negative:
            x, y = int(point[0]), int(point[1])
            if selected_mask[y, x]:
                raise ValueError(f"Negative point ({x}, {y}) falls inside the authoritative mask")
        # SAM2 stores mask and point inputs as mutually exclusive prompt states.
        # Keeping one authoritative mask avoids the silent last-prompt-wins behavior.
        return SeCVisualPrompt(mask=selected_mask, points=None, labels=None, box=None)

    if len(positive) == 0 and box is None:
        if len(negative):
            raise ValueError("Negative points require at least one positive point or a BBOX")
        raise ValueError("Provide mask, positive_coords, or bounding_box")
    points = np.concatenate((positive, negative), axis=0)
    labels = np.concatenate((positive_labels, negative_labels), axis=0)
    if len(points) == 0:
        points = labels = None
    return SeCVisualPrompt(mask=None, points=points, labels=labels, box=box)


def estimate_sec_activation_memory(frames: torch.Tensor, semantic_keyframes: int) -> int:
    """Conservative reservation for SAM state plus scene-change MLLM activations."""

    frame_bytes = int(frames.shape[0] * frames.shape[1] * frames.shape[2] * 12)
    base = 1024**3
    semantic = int(max(1, semantic_keyframes) * 128 * 1024**2)
    return base + semantic + frame_bytes


def _seed_predictor(predictor, state, prompt: SeCVisualPrompt, frame_idx: int):
    if prompt.mask is not None:
        _, _, logits = predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=1,
            mask=prompt.mask,
        )
        return prompt.mask
    _, _, logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=1,
        points=prompt.points,
        labels=prompt.labels,
        box=prompt.box,
    )
    return (logits[0].detach().float().cpu().numpy().squeeze() > 0.0)


def _resolve_annotation_frame_idx(annotation_frame_idx: int, frame_count: int) -> int:
    """Resolve absolute or standard Python-style negative frame indexes."""

    index = int(annotation_frame_idx)
    if index < 0:
        index = int(frame_count) + index
    if not 0 <= index < int(frame_count):
        accepted = f"[{-frame_count},-1] or [0,{frame_count - 1}]"
        raise ValueError(
            f"annotation_frame_idx must be in {accepted}; negative values use Python "
            f"indexing where -1 is the final frame, got {annotation_frame_idx}"
        )
    return index


def track_visual_concept(
    handle: SeCModelHandle,
    frames: torch.Tensor,
    *,
    positive_coords: str | None = "",
    negative_coords: str | None = "",
    bounding_box=None,
    mask: torch.Tensor | None = None,
    tracking_direction: str = "forward",
    annotation_frame_idx: int = 0,
    max_frames_to_track: int = -1,
    semantic_keyframes: int = 7,
) -> torch.Tensor:
    if not isinstance(handle, SeCModelHandle):
        raise TypeError("model must come from Load SeC Model")
    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] < 3:
        shape = tuple(frames.shape) if hasattr(frames, "shape") else type(frames).__name__
        raise ValueError(f"frames must be a ComfyUI IMAGE batch [N,H,W,C], got {shape}")
    frame_count = int(frames.shape[0])
    if frame_count < 1:
        raise ValueError("frames must contain at least one image")
    annotation_frame_idx = _resolve_annotation_frame_idx(annotation_frame_idx, frame_count)
    if tracking_direction not in {"forward", "backward", "bidirectional"}:
        raise ValueError(f"Unsupported tracking_direction: {tracking_direction}")
    if int(semantic_keyframes) < 1:
        raise ValueError("semantic_keyframes must be at least 1")

    prompt = prepare_visual_prompt(
        frames,
        annotation_frame_idx=int(annotation_frame_idx),
        positive_coords=positive_coords,
        negative_coords=negative_coords,
        bounding_box=bounding_box,
        mask=mask,
    )
    memory_required = estimate_sec_activation_memory(frames, int(semantic_keyframes))
    comfy.model_management.load_models_gpu(
        [handle.patcher],
        memory_required=memory_required,
        force_full_load=True,
    )

    model = handle.model
    predictor = model.grounding_encoder
    state = None
    output_device = comfy.model_management.intermediate_device()
    masks = torch.zeros(
        (frame_count, int(frames.shape[1]), int(frames.shape[2])),
        dtype=torch.float32,
        device=output_device,
    )
    limit = frame_count if int(max_frames_to_track) < 0 else int(max_frames_to_track)
    progress = comfy.utils.ProgressBar(frame_count)
    completed: set[int] = set()

    def propagate(reverse: bool, init_mask: np.ndarray) -> None:
        for out_frame_idx, _out_obj_ids, mask_logits in model.propagate_in_video(
            state,
            start_frame_idx=int(annotation_frame_idx),
            max_frame_num_to_track=limit,
            reverse=reverse,
            init_mask=init_mask,
            mllm_memory_size=int(semantic_keyframes),
        ):
            comfy.model_management.throw_exception_if_processing_interrupted()
            mask = (mask_logits[0].detach().float().cpu().squeeze() > 0.0).to(torch.float32)
            masks[int(out_frame_idx)].copy_(mask.to(output_device))
            if int(out_frame_idx) not in completed:
                completed.add(int(out_frame_idx))
                progress.update(1)

    try:
        # Frames and tracking state remain on CPU; only the current model workset is
        # transferred to the device.  ComfyUI owns the model's residency lifecycle.
        state = predictor.init_state(
            video_path=frames,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.reset_state(state)
        init_mask = _seed_predictor(
            predictor,
            state,
            prompt,
            int(annotation_frame_idx),
        )
        if tracking_direction == "bidirectional":
            propagate(False, init_mask)
            predictor.reset_state(state)
            init_mask = _seed_predictor(
                predictor,
                state,
                prompt,
                int(annotation_frame_idx),
            )
            propagate(True, init_mask)
        else:
            propagate(tracking_direction == "backward", init_mask)
        return masks
    finally:
        if state is not None:
            try:
                predictor.reset_state(state)
            finally:
                state.clear()


__all__ = [
    "SeCAttentionPlan",
    "SeCModelHandle",
    "available_sec_models",
    "estimate_sec_activation_memory",
    "load_sec_model",
    "parse_bbox",
    "parse_points",
    "prepare_visual_prompt",
    "resolve_sec_attention",
    "sec_model_choices",
    "track_visual_concept",
]
