"""Runtime capability and resource-planning services."""

from .capabilities import (
    CapabilityResult,
    KernelCapabilities,
    RuntimeCapabilities,
    kernel_capabilities,
    runtime_capabilities,
)
from .diagnostics import FEATURES, runtime_diagnostics
from .stage_barrier import (
    BarrierPhase,
    BarrierPlanError,
    BarrierPlanner,
    STAGE_BARRIER_NODE_ID,
    STAGE_PATH_NODE_ID,
    install_stage_barrier_scheduler,
    stage_barrier_candidates,
)
from .stage_barrier_prompt import compile_stage_barrier_prompt

__all__ = [
    "CapabilityResult",
    "KernelCapabilities",
    "RuntimeCapabilities",
    "kernel_capabilities",
    "runtime_capabilities",
    "FEATURES",
    "runtime_diagnostics",
    "BarrierPhase",
    "BarrierPlanError",
    "BarrierPlanner",
    "STAGE_BARRIER_NODE_ID",
    "STAGE_PATH_NODE_ID",
    "compile_stage_barrier_prompt",
    "install_stage_barrier_scheduler",
    "stage_barrier_candidates",
]
