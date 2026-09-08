"""Sampler-boundary protection for AIMDO DynamicVRAM model patchers."""

from __future__ import annotations

import torch

from ..log import get_logger
from ..profiling import CUDA_PHASE_PROFILER, WORKFLOW_TIMELINE


LOG = get_logger("memory")
DYNAMIC_VRAM_FENCE_WRAPPER_KEY = "turing_utils_dynamic_vram_sample_fence"


def make_dynamic_vram_sample_fence(device: torch.device):
    """Drain CUDA streams before and after one complete sampler invocation.

    DynamicVRAM mappings are shared by ModelPatcher clones.  Sequential model
    branches can therefore hand the same mappings to the next sampler while an
    async prefetch/offload stream is still retiring work.  Synchronizing only
    at the outer sampler boundary preserves all block-level overlap within the
    sampler while preventing the next sampler from faulting VBAR mappings with
    outstanding CUDA work.
    """

    device = torch.device(device)

    def outer_sample_wrapper(executor, *args, **kwargs):
        torch.cuda.synchronize(device)
        timeline = WORKFLOW_TIMELINE.begin(device)
        try:
            return executor(*args, **kwargs)
        finally:
            WORKFLOW_TIMELINE.record_end(timeline)
            torch.cuda.synchronize(device)
            try:
                WORKFLOW_TIMELINE.finish_after_synchronize(timeline)
            except Exception as error:
                LOG.warning("Unable to emit workflow timeline: %s", error)
            try:
                CUDA_PHASE_PROFILER.report_after_synchronize()
            except Exception as error:
                # Optional diagnostics must never replace a sampling failure.
                LOG.warning("Unable to emit deferred CUDA profile: %s", error)

    return outer_sample_wrapper


def install_dynamic_vram_sample_fence(model, device: torch.device) -> bool:
    """Install one keyed fence on a CUDA DynamicVRAM ModelPatcher."""

    is_dynamic = getattr(model, "is_dynamic", None)
    if not callable(is_dynamic) or not is_dynamic():
        return False
    if not callable(getattr(model, "add_wrapper_with_key", None)):
        return False
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return False

    try:
        import comfy.patcher_extension
    except ImportError:
        return False

    wrapper_type = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    get_wrappers = getattr(model, "get_wrappers", None)
    if callable(get_wrappers) and get_wrappers(
        wrapper_type, DYNAMIC_VRAM_FENCE_WRAPPER_KEY
    ):
        CUDA_PHASE_PROFILER.defer_to_sampler_boundary()
        return False

    model.add_wrapper_with_key(
        wrapper_type,
        DYNAMIC_VRAM_FENCE_WRAPPER_KEY,
        make_dynamic_vram_sample_fence(device),
    )
    # The wrapper now owns the synchronization needed to safely read CUDA
    # event timings.  Never insert another fence after an inner attention call:
    # doing so would break ComfyUI's asynchronous weight prefetch pipeline.
    CUDA_PHASE_PROFILER.defer_to_sampler_boundary()
    LOG.info("Enabled DynamicVRAM sampler-boundary CUDA fence")
    return True


__all__ = [
    "DYNAMIC_VRAM_FENCE_WRAPPER_KEY",
    "install_dynamic_vram_sample_fence",
    "make_dynamic_vram_sample_fence",
]
