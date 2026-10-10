"""ConvRot quantization metadata and loaded-module inspection."""

from __future__ import annotations

import dataclasses
import json
import math
import struct
from pathlib import Path

import torch


MODEL_EXTENSIONS = {".safetensors", ".sft"}
W4_FORMAT = "convrot_w4a4"
CODEBOOK_W4_FORMAT = "asym_w4a8_int8"
W8_FORMAT = "int8_tensorwise"
LEGACY_W8_FORMAT = "int8_rowwise"
NVFP4_FORMAT = "nvfp4"
MAX_SAFETENSORS_HEADER_SIZE = 128 * 1024 * 1024
MAX_QUANT_CONFIG_SIZE = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class ConvRotSummary:
    w4a4: int = 0
    w4a8: int = 0
    codebook_w4a8: int = 0
    w8a8: int = 0
    nvfp4: int = 0


def _params(config: dict, layer_name: str) -> dict:
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"ConvRot layer {layer_name} has non-object params metadata")
    return params


def _config_value(config: dict, params: dict, name: str, default):
    return config[name] if name in config else params.get(name, default)


def _normalize_legacy_config(config: dict, layer_name: str) -> None:
    params = _params(config, layer_name)
    quant_format = config.get("format")
    convrot = _config_value(config, params, "convrot", False)
    per_row = _config_value(config, params, "per_row", False)
    if (quant_format is None or quant_format == LEGACY_W8_FORMAT) and convrot is True and per_row is True:
        config["format"] = W8_FORMAT


def _classify_config(config: dict, layer_name: str, force_int8_gemm: bool) -> tuple[str, str] | None:
    if not isinstance(config, dict):
        raise ValueError(f"Quantization metadata for layer {layer_name} must be an object")

    _normalize_legacy_config(config, layer_name)
    quant_format = config.get("format")
    params = _params(config, layer_name)
    if quant_format == NVFP4_FORMAT:
        return "nvfp4", "int8"
    if quant_format == W4_FORMAT:
        activation_dtype = _config_value(config, params, "linear_dtype", "int4")
        if activation_dtype not in {"int4", "int8"}:
            raise ValueError(
                f"ConvRot W4 layer {layer_name} declares unsupported linear_dtype={activation_dtype!r}; "
                "expected 'int4' or 'int8'"
            )
        return "w4", "int8" if force_int8_gemm else activation_dtype

    if quant_format == CODEBOOK_W4_FORMAT:
        convrot = _config_value(config, params, "convrot", True)
        if convrot is not True:
            raise ValueError(
                f"Grouped-codebook W4A8 layer {layer_name} must declare convrot=true"
            )
        group_size = _config_value(config, params, "group_size", 16)
        convrot_groupsize = _config_value(config, params, "convrot_groupsize", 256)
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 4:
            raise ValueError(
                f"Grouped-codebook W4A8 layer {layer_name} has invalid group_size={group_size!r}"
            )
        if convrot_groupsize != 256:
            raise ValueError(
                f"Grouped-codebook W4A8 layer {layer_name} requires convrot_groupsize=256"
            )
        return "w4_codebook", "int8"

    convrot = _config_value(config, params, "convrot", False)
    if not isinstance(convrot, bool):
        raise ValueError(f"Quantized layer {layer_name} has non-boolean convrot={convrot!r}")
    if not convrot:
        return None
    if quant_format != W8_FORMAT:
        raise ValueError(f"ConvRot layer {layer_name} uses unsupported weight format {quant_format!r}")

    activation_dtype = _config_value(config, params, "linear_dtype", "int8")
    if activation_dtype not in {"int4", "int8"}:
        raise ValueError(
            f"ConvRot W8 layer {layer_name} declares unsupported linear_dtype={activation_dtype!r}; "
            "expected 'int8'"
        )
    if activation_dtype != "int8" and not force_int8_gemm:
        raise ValueError(
            f"ConvRot W8 layer {layer_name} cannot use linear_dtype={activation_dtype!r}; "
            "W8 ConvRot supports INT8 activations only"
        )
    return "w8", "int8"


def _decode_quant_tensor(value: torch.Tensor, key: str) -> dict:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a JSON byte tensor")
    try:
        config = json.loads(value.detach().cpu().numpy().tobytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{key} contains invalid quantization JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{key} quantization JSON must be an object")
    return config


def _encode_quant_tensor(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def _read_safetensors_header(model_path: Path) -> tuple[dict, int, int]:
    file_size = model_path.stat().st_size
    with model_path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("file is too short to contain a safetensors header")
        header_size = struct.unpack("<Q", raw_length)[0]
        if header_size == 0 or header_size > MAX_SAFETENSORS_HEADER_SIZE:
            raise ValueError(f"invalid safetensors header size {header_size}")
        data_start = 8 + header_size
        if data_start > file_size:
            raise ValueError(
                f"safetensors header ends at byte {data_start}, beyond file size {file_size}"
            )
        raw_header = handle.read(header_size)

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header contains invalid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, data_start, file_size


def _read_quant_config(
    handle,
    tensor_info: dict,
    key: str,
    data_start: int,
    file_size: int,
) -> dict:
    if not isinstance(tensor_info, dict):
        raise ValueError(f"{key} has an invalid safetensors tensor descriptor")
    if tensor_info.get("dtype") != "U8":
        raise ValueError(f"{key} must use U8 storage")

    shape = tensor_info.get("shape")
    offsets = tensor_info.get("data_offsets")
    if (
        not isinstance(shape, list)
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
    ):
        raise ValueError(f"{key} has invalid shape or data offsets")

    relative_start, relative_end = offsets
    config_size = relative_end - relative_start
    if config_size != math.prod(shape):
        raise ValueError(f"{key} byte length does not match its U8 shape")
    if config_size > MAX_QUANT_CONFIG_SIZE:
        raise ValueError(f"{key} is too large to be quantization JSON")
    absolute_start = data_start + relative_start
    absolute_end = data_start + relative_end
    if relative_start < 0 or absolute_end < absolute_start or absolute_end > file_size:
        raise ValueError(f"{key} data offsets are outside the safetensors file")

    handle.seek(absolute_start)
    raw_config = handle.read(config_size)
    if len(raw_config) != config_size:
        raise ValueError(f"{key} tensor data is truncated")
    try:
        config = json.loads(raw_config)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{key} contains invalid quantization JSON") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{key} quantization JSON must be an object")
    return config


def _convrot_skip_reason(model_path: str | Path) -> str | None:
    model_path = Path(model_path)
    if model_path.suffix.lower() not in MODEL_EXTENSIONS:
        return f"unsupported file extension {model_path.suffix!r}"

    try:
        tensor_index, data_start, file_size = _read_safetensors_header(model_path)
        metadata = tensor_index.get("__metadata__", {})
        if not isinstance(metadata, dict):
            return "safetensors __metadata__ must be an object"

        raw_quantization = metadata.get("_quantization_metadata")
        if raw_quantization is not None:
            quantization = json.loads(raw_quantization)
            if not isinstance(quantization, dict) or not isinstance(quantization.get("layers"), dict):
                return "safetensors _quantization_metadata must contain a layers object"
            for layer_name, config in quantization["layers"].items():
                if _classify_config(config, layer_name, True) is not None:
                    return None

        with model_path.open("rb") as handle:
            for key, tensor_info in tensor_index.items():
                if key == "__metadata__" or not key.endswith(".comfy_quant"):
                    continue
                config = _read_quant_config(handle, tensor_info, key, data_start, file_size)
                if _classify_config(config, key[: -len(".comfy_quant")], True) is not None:
                    return None
    except Exception as exc:
        return f"could not read ConvRot metadata ({exc})"

    return "does not contain supported ConvRot quantization metadata"


def configure_convrot_activation(
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
    force_int8_gemm: bool,
    model_prefix: str | None = None,
) -> tuple[dict[str, str], ConvRotSummary]:
    if not isinstance(force_int8_gemm, bool):
        raise ValueError(f"force_int8_gemm must be boolean, got {force_int8_gemm!r}")

    metadata = dict(metadata or {})
    records: list[tuple[str, dict]] = []
    header_quantization = None
    raw_header = metadata.get("_quantization_metadata")
    if raw_header is not None:
        try:
            header_quantization = json.loads(raw_header)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Safetensors _quantization_metadata contains invalid JSON") from exc
        if not isinstance(header_quantization, dict) or not isinstance(header_quantization.get("layers"), dict):
            raise ValueError("Safetensors _quantization_metadata must contain a layers object")
        records.extend(
            (name, config)
            for name, config in header_quantization["layers"].items()
            if model_prefix is None or name.startswith(model_prefix)
        )

    tensor_configs: dict[str, dict] = {}
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        layer_name = key[: -len(".comfy_quant")]
        if model_prefix is not None and not layer_name.startswith(model_prefix):
            continue
        config = _decode_quant_tensor(value, key)
        tensor_configs[key] = config
        records.append((layer_name, config))

    classified_records: list[tuple[str, dict, str, str]] = []
    layer_types: dict[str, tuple[str, str]] = {}
    for layer_name, config in records:
        classification = _classify_config(config, layer_name, force_int8_gemm)
        if classification is None:
            continue
        previous = layer_types.get(layer_name)
        if previous is not None:
            if previous != classification:
                raise ValueError(
                    f"Conflicting ConvRot metadata for layer {layer_name}: "
                    f"{previous[0]}/{previous[1]} versus {classification[0]}/{classification[1]}"
                )
        else:
            layer_types[layer_name] = classification
        classified_records.append((layer_name, config, *classification))

    if not classified_records:
        raise ValueError("The selected model does not contain supported ConvRot quantization metadata")

    if force_int8_gemm:
        for _, config, weight_dtype, _ in classified_records:
            if weight_dtype not in {"w4_codebook", "nvfp4"}:
                config["linear_dtype"] = "int8"

    if header_quantization is not None:
        metadata["_quantization_metadata"] = json.dumps(header_quantization)
    for key, config in tensor_configs.items():
        state_dict[key] = _encode_quant_tensor(config)

    summary = ConvRotSummary(
        nvfp4=sum(1 for kind, _ in layer_types.values() if kind == "nvfp4"),
        w4a4=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w4", "int4")),
        w4a8=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w4", "int8")),
        codebook_w4a8=sum(
            1
            for weight_dtype, act_dtype in layer_types.values()
            if (weight_dtype, act_dtype) == ("w4_codebook", "int8")
        ),
        w8a8=sum(1 for weight_dtype, act_dtype in layer_types.values() if (weight_dtype, act_dtype) == ("w8", "int8")),
    )
    return metadata, summary


def _summarize_convrot_modules(root: torch.nn.Module) -> ConvRotSummary:
    w4a4 = 0
    w4a8 = 0
    codebook_w4a8 = 0
    w8a8 = 0
    nvfp4 = 0
    for _, module in root.named_modules():
        quant_format = getattr(module, "quant_format", None)
        weight = getattr(module, "weight", None)
        params = getattr(weight, "_params", None)
        if quant_format == NVFP4_FORMAT:
            nvfp4 += 1
        elif quant_format == W4_FORMAT:
            activation_dtype = getattr(params, "linear_dtype", None)
            if activation_dtype == "int4":
                w4a4 += 1
            elif activation_dtype == "int8":
                w4a8 += 1
            else:
                raise RuntimeError(
                    f"Loaded ConvRot W4 layer has unsupported linear_dtype={activation_dtype!r}; "
                    "update ComfyUI and comfy-kitchen"
                )
        elif quant_format == CODEBOOK_W4_FORMAT:
            if getattr(params, "codebook", None) is None:
                raise RuntimeError(
                    "Loaded grouped-codebook W4A8 layer is missing its 16-entry codebook"
                )
            if getattr(params, "correction", None) is not None:
                raise RuntimeError(
                    "Loaded grouped-codebook W4A8 layer uses asymmetric correction, "
                    "which is not supported by the production Turing path"
                )
            codebook_w4a8 += 1
        elif quant_format == W8_FORMAT and getattr(params, "convrot", False):
            w8a8 += 1
    return ConvRotSummary(
        w4a4=w4a4,
        w4a8=w4a8,
        codebook_w4a8=codebook_w4a8,
        w8a8=w8a8,
        nvfp4=nvfp4,
    )
