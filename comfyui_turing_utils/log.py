"""Consistent component labels for local runtime diagnostics."""

from __future__ import annotations

import logging
import os
from functools import lru_cache


ROOT_LOGGER = "comfyui-turing-utils"


@lru_cache(maxsize=1)
def profile_level() -> int:
    """Process-start diagnostics: 0=off, 1=summary, 2=detail. Restart to change."""
    retired = (
        "COMFYUI_TURING_UTILS_PROFILE_CALLS", "COMFYUI_TURING_UTILS_PROFILE_BUCKETS",
        "COMFYUI_TURING_UTILS_TIMELINE", "SEC_DEBUG",
        "COMFYUI_TURING_UTILS_H3_ACTIVATION_CHUNK_ROWS",
        "COMFYUI_TURING_UTILS_H3_QKV_CHUNK_ROWS", "COMFYUI_TURING_UTILS_H3_MLP_CHUNK_ROWS",
        "COMFYUI_TURING_UTILS_H3_HEAD_GROUP", "COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS",
    )
    present = [name for name in retired if name in os.environ]
    if present:
        logging.getLogger(ROOT_LOGGER).warning(
            "[Turing/config] Ignoring retired settings: %s. Use COMFYUI_TURING_UTILS_PROFILE=0/1/2; H3 chunking is automatic.",
            ", ".join(present),
        )
    raw = os.environ.get("COMFYUI_TURING_UTILS_PROFILE", "0").strip()
    if raw in {"0", "1", "2"}:
        return int(raw)
    logging.getLogger(ROOT_LOGGER).warning(
        "[Turing/config] Invalid COMFYUI_TURING_UTILS_PROFILE=%r; using 0 (expected 0/1/2)", raw
    )
    return 0


class _ComponentLogger(logging.LoggerAdapter):
    def __init__(self, logger, extra):
        super().__init__(logger, extra)
        self._warned = set()

    def warning_once(self, msg, *args, **kwargs):
        """Bounded per-component deduplication by reason, not changing tensor shapes."""
        if not self.isEnabledFor(logging.WARNING) or msg in self._warned:
            return
        if len(self._warned) < 256:
            self._warned.add(msg)
        self.warning(msg, *args, **kwargs)

    def process(self, msg, kwargs):
        return f"[Turing/{self.extra['component']}] {msg}", kwargs


def get_logger(component: str) -> _ComponentLogger:
    component = str(component).strip(".")
    if not component:
        raise ValueError("log component must not be empty")
    return _ComponentLogger(
        logging.getLogger(f"{ROOT_LOGGER}.{component}"),
        {"component": component},
    )


__all__ = ["ROOT_LOGGER", "get_logger", "profile_level"]
