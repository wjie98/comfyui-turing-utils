"""Registry for loader-independent, model-specific optimization adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


AdapterInstaller = Callable[[object, torch.device], int]


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    install: AdapterInstaller


_ADAPTERS: list[ModelAdapter] = []


def register_model_adapter(adapter: ModelAdapter) -> None:
    for existing in _ADAPTERS:
        if existing.name == adapter.name:
            if existing != adapter:
                raise ValueError(
                    f"Model adapter {adapter.name!r} is already registered"
                )
            return
    _ADAPTERS.append(adapter)


def apply_model_adapters(model, device: torch.device) -> dict[str, int]:
    """Apply every registered adapter; unmatched adapters return zero."""
    return {
        adapter.name: int(adapter.install(model, device))
        for adapter in tuple(_ADAPTERS)
    }


def registered_model_adapters() -> tuple[str, ...]:
    return tuple(adapter.name for adapter in _ADAPTERS)


__all__ = [
    "ModelAdapter",
    "apply_model_adapters",
    "register_model_adapter",
    "registered_model_adapters",
]
