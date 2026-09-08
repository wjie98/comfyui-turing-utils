"""Consistent component labels for local runtime diagnostics."""

from __future__ import annotations

import logging


ROOT_LOGGER = "comfyui-turing-utils"


class _ComponentLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[Turing/{self.extra['component']}] {msg}", kwargs


def get_logger(component: str) -> logging.LoggerAdapter:
    component = str(component).strip(".")
    if not component:
        raise ValueError("log component must not be empty")
    return _ComponentLogger(
        logging.getLogger(f"{ROOT_LOGGER}.{component}"),
        {"component": component},
    )


__all__ = ["ROOT_LOGGER", "get_logger"]
