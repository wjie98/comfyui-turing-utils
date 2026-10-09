"""Single user-facing policy choice for automatic MiniMax scheduling."""

from __future__ import annotations

import os

from ...log import get_logger


LOG = get_logger("minimax.config")


def activation_mode() -> str:
    value = os.environ.get(
        "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE", "auto"
    ).strip().lower()
    if value in {"auto", "throughput", "balanced"}:
        return value
    LOG.warning_once("Invalid H3_ACTIVATION_MODE=%r; using auto (expected auto/throughput/balanced)", value)
    return "auto"

__all__ = ["activation_mode"]
