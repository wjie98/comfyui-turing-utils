"""Shared H3 execution audit, weak references and cast helpers."""

from __future__ import annotations

import weakref
from collections import Counter
import torch
from ...log import get_logger
from ...quantization.dispatch import quantize_convrot_int8_activation
from ...kernel_api import kernel_extension_has_symbol
from ...profiling import CUDA_PHASE_PROFILER
from .layout import RUNTIME_CONTEXT_ATTR


LOG = get_logger("minimax.fusion")


def _profile_cuda(phase: str, function, /, *args, **kwargs):
    if CUDA_PHASE_PROFILER.enabled:
        return CUDA_PHASE_PROFILER.call(phase, function, *args, **kwargs)
    return function(*args, **kwargs)


def _direct_int8_output_available() -> bool:
    return kernel_extension_has_symbol("turing_int8_linear_out")


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _runtime_activation_plan(base_model):
    if base_model is None:
        return None
    try:
        context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    except ReferenceError:
        return None
    return context.get("activation_plan") if isinstance(context, dict) else None


def _weak_model_reference(base_model):
    if base_model is None:
        return None
    try:
        return weakref.proxy(base_model)
    except TypeError:
        # Test doubles and a few third-party wrappers are not weak-referenceable.
        return base_model


class _RuntimeDispatchAudit:
    """Report the first full MiniMax block pass without CUDA timing events."""

    def __init__(self, expected_blocks: int, expected_mlps: int):
        self.expected = {"block": expected_blocks, "mlp": expected_mlps}
        self.counts = {"block": Counter(), "mlp": Counter()}
        self.dtypes = {"block": Counter(), "mlp": Counter()}
        self.shapes = {"block": Counter(), "mlp": Counter()}
        self.reasons = {"block": Counter(), "mlp": Counter()}
        self.logged_phases: set[str] = set()

    def record(
        self,
        phase: str,
        fused: bool,
        x: torch.Tensor,
        reason: str | None = None,
    ) -> None:
        if phase in self.logged_phases or self.expected[phase] == 0:
            return
        self.counts[phase]["fused" if fused else "fallback"] += 1
        self.dtypes[phase][str(x.dtype)] += 1
        self.shapes[phase][str(tuple(x.shape))] += 1
        if reason is not None:
            self.reasons[phase][reason] += 1

        calls = self.counts[phase]["fused"] + self.counts[phase]["fallback"]
        if calls < self.expected[phase]:
            return

        log = LOG.warning if self.counts[phase]["fallback"] else LOG.debug
        log(
            "MiniMax fused runtime dispatch: phase=%s fused=%d fallback=%d "
            "dtypes=[%s] shapes=[%s] reasons=[%s]",
            phase,
            self.counts[phase]["fused"],
            self.counts[phase]["fallback"],
            _format_counts(self.dtypes[phase].elements()),
            _format_counts(self.shapes[phase].elements()),
            _format_counts(self.reasons[phase].elements()) or "none",
        )
        self.logged_phases.add(phase)


def _format_counts(values) -> str:
    counts = Counter(values)
    return ",".join(
        f"{value}:{count}"
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _linear_with_cast_weight(linear, x, weight, bias):
    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        import comfy.model_management

        x = x * comfy.model_management.cast_to_device(
            pre_quant_scale, x.device, x.dtype
        )
    function = getattr(linear, "_forward", None)
    return (
        function(x, weight, bias)
        if callable(function)
        else torch.nn.functional.linear(x, weight, bias)
    )


def scaled_linear_input(linear, x: torch.Tensor) -> torch.Tensor:
    pre_quant_scale = getattr(linear, "pre_quant_scale", None)
    if pre_quant_scale is None:
        return x
    import comfy.model_management

    scale = comfy.model_management.cast_to_device(pre_quant_scale, x.device, x.dtype)
    return x * scale


def quantize_linear_rows(linear, x: torch.Tensor):
    return quantize_convrot_int8_activation(scaled_linear_input(linear, x), 256)
