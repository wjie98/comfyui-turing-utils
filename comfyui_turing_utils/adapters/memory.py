"""Reusable model-adapter memory-planning mechanics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ..quantization.dispatch import (
    turing_codebook_w4a8_workspace_bytes,
    turing_int8_workspace_bytes,
)


@dataclass(frozen=True, slots=True)
class QuantizedWorkspaceProfile:
    formats: tuple[tuple[str, int], ...]
    w8_output_channels: tuple[int, ...]
    fixed_workspaces: tuple[int, ...]

    def transient_bytes(self, rows: int) -> int:
        workspaces = [
            turing_int8_workspace_bytes(rows, output_channels)
            for output_channels in self.w8_output_channels
        ]
        workspaces.extend(self.fixed_workspaces)
        return max(workspaces, default=0)


def scan_quantized_workspaces(
    root,
    classify_weight: Callable[[object], str | None],
) -> QuantizedWorkspaceProfile:
    """Scan serial linear operators once and retain only peak workspace facts."""
    formats = Counter()
    outputs = set()
    fixed = set()
    for module in root.modules():
        weight = getattr(module, "weight", None)
        kind = classify_weight(weight)
        if kind is None:
            continue
        formats[kind] += 1
        if kind == "w8a8" and getattr(weight, "ndim", 0) == 2:
            outputs.add(int(weight.shape[0]))
        elif kind in {"codebook_w4a8", "w6a8"} and getattr(weight, "ndim", 0) == 2:
            fixed.add(
                turing_codebook_w4a8_workspace_bytes(
                    int(weight.shape[1]),
                    int(weight.shape[0]),
                )
            )
    return QuantizedWorkspaceProfile(
        tuple(sorted(formats.items())),
        tuple(sorted(outputs)),
        tuple(sorted(fixed)),
    )


def install_memory_hooks(
    base_model,
    *,
    marker: str,
    condition_key: str,
    extra_conds,
    extra_conds_shapes,
    memory_required,
    required_methods: tuple[str, ...] = (),
) -> bool:
    """Atomically install the three BaseModel memory hooks and factor key."""
    if getattr(base_model, marker, False):
        return False
    if not all(
        callable(getattr(base_model, name, None))
        for name in (
            "extra_conds",
            "extra_conds_shapes",
            "memory_required",
            *required_methods,
        )
    ):
        return False
    base_model.extra_conds = extra_conds
    base_model.extra_conds_shapes = extra_conds_shapes
    base_model.memory_required = memory_required
    factors = tuple(getattr(base_model, "memory_usage_factor_conds", ()))
    if condition_key not in factors:
        base_model.memory_usage_factor_conds = (*factors, condition_key)
    setattr(base_model, marker, True)
    return True


__all__ = [
    "QuantizedWorkspaceProfile",
    "install_memory_hooks",
    "scan_quantized_workspaces",
]
