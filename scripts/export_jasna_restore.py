"""Export local Jasna Restore weights for the Turing Utils ComfyUI node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TRANSFORMER_CONFIG = {
    "in_channels": 128,
    "cross_attention_dim": 4096,
    "caption_channels": 4096,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "num_layers": 48,
    "causal_temporal_positioning": True,
    "rope_type": "split",
    "frequencies_precision": "float64",
    "use_middle_indices_grid": True,
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "cross_attention_adaln": True,
    "ff_bias": False,
}

DISTILLED_INDICES = {
    2: (0, 8, 15),
    4: (0, 4, 8, 12, 15),
    8: (0, 2, 4, 6, 8, 10, 12, 14, 15),
}


def save_export(state, metadata, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jasna-export-") as staging:
        temporary = Path(staging) / destination.name
        save_file(state, temporary, metadata=metadata)
        temporary.chmod(0o600)
        os.replace(temporary, destination)


def export_dit(
    source: Path, destination: Path, schedule_from: Path | None, steps: int | None
):
    with safe_open(source, framework="pt") as weights:
        metadata = weights.metadata() or {}
        if schedule_from is None and steps is None:
            if "sigmas" not in metadata or "stg_scale" not in metadata:
                raise ValueError(
                    "The weights lack a schedule. Specify --schedule-from and --steps."
                )
            sigmas = json.loads(metadata["sigmas"])
            stg_scale = float(metadata["stg_scale"])
        else:
            if schedule_from is None or steps is None:
                raise ValueError(
                    "--schedule-from and --steps must be supplied together."
                )
            if "sigmas" in metadata or "stg_scale" in metadata:
                raise ValueError("The source already defines its own schedule.")
            with safe_open(schedule_from, framework="pt") as base:
                baseline = json.loads(base.metadata()["sigmas"])
            if len(baseline) != 16:
                raise ValueError("The source schedule must contain 15 steps.")
            sigmas = [baseline[i] for i in DISTILLED_INDICES[steps]]
            stg_scale = 0.0

        keys = set(weights.keys())
        state = {}
        layers = {}
        for key in keys:
            name = "transformer_blocks." + key[7:] if key.startswith("blocks.") else key
            if key.endswith(".scales"):
                if key.removesuffix(".scales") + ".qdata" not in keys:
                    raise ValueError(f"Unexpected INT8 scale tensor {key}")
                continue
            tensor = weights.get_tensor(key)
            if key.endswith(".qdata"):
                layer = name.removesuffix(".qdata")
                scale = weights.get_tensor(key.removesuffix(".qdata") + ".scales")
                if (
                    tensor.dtype != torch.int8
                    or tensor.ndim != 2
                    or tensor.shape[1] % 256
                ):
                    raise ValueError(f"Invalid Jasna ConvRot INT8 weight {key}")
                if scale.shape != (tensor.shape[0],) or scale.dtype != torch.float32:
                    raise ValueError(f"Invalid Jasna ConvRot INT8 scale for {key}")
                state[layer + ".weight"] = tensor
                state[layer + ".weight_scale"] = scale
                layers[layer] = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }
            else:
                state[name] = tensor
                if key.endswith(".weight") and tensor.dtype == torch.uint8:
                    # Jasna and Kitchen store high-nibble-first E2M1 with cuBLAS tiled scales.
                    layer = key.removesuffix(".weight")
                    block_scale = weights.get_slice(layer + ".weight_scale")
                    tensor_scale = weights.get_slice(layer + ".weight_scale_2")
                    rows, packed_columns = tensor.shape
                    expected_scale_shape = [
                        (rows + 127) // 128 * 128,
                        (packed_columns + 31) // 32 * 4,
                    ]
                    if (
                        block_scale.get_shape() != expected_scale_shape
                        or block_scale.get_dtype() != "F8_E4M3"
                        or tensor_scale.get_shape() != []
                        or tensor_scale.get_dtype() != "F32"
                    ):
                        raise ValueError(f"Invalid Jasna NVFP4 scales for {key}")
                    layers[name.removesuffix(".weight")] = {"format": "nvfp4"}

        formats = {layer["format"] for layer in layers.values()}
        quantized_per_block = {"int8_tensorwise": 12, "nvfp4": 10}
        if (
            len(formats) != 1
            or "prompt_context" not in state
            or len(layers)
            != TRANSFORMER_CONFIG["num_layers"]
            * quantized_per_block[next(iter(formats))]
        ):
            raise ValueError("Expected a Jasna LTX Restore INT8 or NVFP4 checkpoint.")
        config = json.loads(metadata.get("config", "{}"))
        config.setdefault("transformer", {}).update(TRANSFORMER_CONFIG)
        metadata["config"] = json.dumps(config)
        metadata["_quantization_metadata"] = json.dumps({"layers": layers})
        metadata["jasna_restore"] = json.dumps(
            {
                "version": 1,
                "sampler": "euler",
                "sigmas": sigmas,
                "stg_scale": stg_scale,
                "stg_block": 28,
                "window_frames": 121,
                "overlap_frames": 17,
                "position_fps": 24,
            }
        )
        save_export(state, metadata, destination)


def export_vae(source: Path, overlay: Path, destination: Path):
    with (
        safe_open(source, framework="pt") as base,
        safe_open(overlay, framework="pt") as decoder,
    ):
        metadata = base.metadata() or {}
        config = json.loads(metadata.get("config", "{}")).get("vae", {})
        if config.get("decoder", {}).get("_class_name") != "NADiffusionDecoder":
            raise ValueError(
                "Expected the Jasna base video VAE with its native config."
            )
        state = {key: base.get_tensor(key) for key in base.keys()}
        replacement = {}
        qkv = {}
        for key in decoder.keys():
            name = "decoder." + key
            name = name.replace(
                "t_embedder.timestep_embedder.linear_1.", "t_embedder.mlp.0."
            )
            name = name.replace(
                "t_embedder.timestep_embedder.linear_2.", "t_embedder.mlp.2."
            )
            tensor = decoder.get_tensor(key)
            split = re.fullmatch(r"(.+\.attn\.qkv)\.to_([qkv])\.(weight|bias)", name)
            if split:
                qkv.setdefault(f"{split[1]}.{split[3]}", {})[split[2]] = tensor
            else:
                replacement[name] = tensor
        for name, parts in qkv.items():
            if parts.keys() != {"q", "k", "v"}:
                raise ValueError(f"Incomplete QKV tensors for {name}")
            replacement[name] = torch.cat([parts[part] for part in "qkv"], dim=0)
        for name, tensor in replacement.items():
            if name not in state or tensor.shape != state[name].shape:
                raise ValueError(
                    f"Decoder override does not match the base VAE: {name}"
                )
        state.update(replacement)
        save_export(state, metadata, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="kind", required=True)
    dit = subcommands.add_parser("dit")
    dit.add_argument("source", type=Path)
    dit.add_argument("destination", type=Path)
    dit.add_argument("--schedule-from", type=Path, help="Base 15-step checkpoint")
    dit.add_argument("--steps", type=int, choices=tuple(DISTILLED_INDICES))
    vae = subcommands.add_parser("vae")
    vae.add_argument("source", type=Path)
    vae.add_argument("overlay", type=Path)
    vae.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.kind == "dit":
        export_dit(args.source, args.destination, args.schedule_from, args.steps)
    else:
        export_vae(args.source, args.overlay, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
