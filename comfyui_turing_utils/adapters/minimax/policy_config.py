"""Environment-backed overrides for MiniMax activation scheduling."""

from __future__ import annotations

import os

from ...log import get_logger


LOG = get_logger("minimax.config")


def activation_mode() -> str:
    value = os.environ.get(
        "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE", "auto"
    ).strip().lower()
    aliases = {
        "speed": "throughput",
        "fast": "throughput",
        "safe": "balanced",
        "memory": "balanced",
        "lowvram": "balanced",
    }
    value = aliases.get(value, value)
    return value if value in {"auto", "throughput", "balanced"} else "auto"


def override_chunk_rows(operation: str) -> int | None:
    names = (
        f"COMFYUI_TURING_UTILS_H3_{operation.upper()}_CHUNK_ROWS",
        "COMFYUI_TURING_UTILS_H3_ACTIVATION_CHUNK_ROWS",
    )
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return max(int(raw), 0)
        except ValueError:
            LOG.warning("Ignoring invalid %s=%r", name, raw)
    return None


def override_head_group() -> int | None:
    raw = os.environ.get("COMFYUI_TURING_UTILS_H3_HEAD_GROUP")
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except ValueError:
        LOG.warning("Ignoring invalid COMFYUI_TURING_UTILS_H3_HEAD_GROUP=%r", raw)
        return None


def override_ffn_channels() -> int | None:
    raw = os.environ.get("COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS")
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except ValueError:
        LOG.warning(
            "Ignoring invalid COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS=%r",
            raw,
        )
        return None


__all__ = [
    "activation_mode",
    "override_chunk_rows",
    "override_ffn_channels",
    "override_head_group",
]
