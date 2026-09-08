"""Live CUDA and DynamicVRAM state used by MiniMax scheduling."""

from __future__ import annotations

import dataclasses
import math

import torch

from ...log import get_logger

from ...profiling import WORKFLOW_TIMELINE


LOG = get_logger("minimax.memory")
MIB = 1024**2
VBAR_PAGE_BYTES = 32 * MIB
DEFAULT_WEIGHT_PREFETCH_BYTES = 512 * MIB
MAX_WEIGHT_PREFETCH_BYTES = 1024 * MIB
_LOGGED_DECISIONS: set[tuple] = set()


@dataclasses.dataclass(slots=True)
class ActivationRuntimePlan:
    """Monotonic, sampler-scoped state for memory policy decisions."""

    available_floors: dict[tuple[str, int | None, int, str], int] = (
        dataclasses.field(default_factory=dict)
    )
    reclaim_requests: dict[tuple[str, int | None, int, str], int] = (
        dataclasses.field(default_factory=dict)
    )
    logged_decisions: set[tuple] = dataclasses.field(default_factory=set)

    def observe_available(
        self,
        device: torch.device,
        rows: int,
        operation: str,
        available_bytes: int,
    ) -> int:
        key = (device.type, device.index, int(rows), str(operation))
        previous = self.available_floors.get(key)
        floor = (
            int(available_bytes)
            if previous is None
            else min(previous, int(available_bytes))
        )
        self.available_floors[key] = floor
        return floor

    def should_request_reclaim(
        self,
        device: torch.device,
        rows: int,
        operation: str,
        deficit_bytes: int,
    ) -> bool:
        key = (device.type, device.index, int(rows), str(operation))
        previous = self.reclaim_requests.get(key, 0)
        deficit_bytes = int(deficit_bytes)
        if deficit_bytes <= previous + 64 * MIB:
            return False
        self.reclaim_requests[key] = deficit_bytes
        return True


def dynamic_vbars(
    base_model,
    device: torch.device,
    *,
    include_current: bool = False,
) -> tuple[object, ...]:
    """Return unique inactive DynamicVRAM VBARs on ``device``."""
    if base_model is None:
        return ()
    device = torch.device(device)
    current = getattr(base_model, "current_patcher", None)
    candidates = []
    try:
        import comfy.model_management as model_management

        candidates.extend(model_management.loaded_models())
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass

    result = []
    seen = set()

    def resolve(patcher):
        try:
            if not patcher.is_dynamic():
                return None
            load_device = torch.device(
                getattr(
                    patcher,
                    "load_device",
                    device if patcher is current else None,
                )
            )
            if load_device != device:
                return None
            return patcher._vbar_get()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    current_vbar = resolve(current) if current is not None else None
    current_vbar_id = id(current_vbar) if current_vbar is not None else None
    for patcher in candidates:
        vbar = resolve(patcher)
        if vbar is None or id(vbar) == current_vbar_id or id(vbar) in seen:
            continue
        seen.add(id(vbar))
        result.append(vbar)
    if include_current and current_vbar is not None and current_vbar_id not in seen:
        result.append(current_vbar)
    return tuple(result)


def current_model_is_dynamic(base_model, device: torch.device) -> bool:
    if base_model is None:
        return False
    try:
        patcher = base_model.current_patcher
        return bool(
            patcher.is_dynamic()
            and torch.device(patcher.load_device) == torch.device(device)
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def dynamic_weight_prefetch_reserve(base_model, device: torch.device) -> int:
    """Reserve two average DiT blocks for ComfyUI's offload streams."""
    default = DEFAULT_WEIGHT_PREFETCH_BYTES
    if not current_model_is_dynamic(base_model, device):
        return default
    try:
        patcher = base_model.current_patcher
        vbar = patcher._vbar_get()
        model_size = getattr(patcher, "model_size", None)
        model_bytes = int(
            model_size() if callable(model_size) else vbar.loaded_size()
        )
        layers = max(
            int(getattr(base_model, "_turing_utils_minimax_layer_count", 0)),
            1,
        )
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return default
    if model_bytes <= 0 or layers <= 1:
        return default
    reserve = (
        math.ceil((2 * model_bytes / layers) / VBAR_PAGE_BYTES) * VBAR_PAGE_BYTES
    )
    return min(max(int(reserve), default), MAX_WEIGHT_PREFETCH_BYTES)


def vbar_reclaimable_bytes(vbar) -> int:
    """Count resident, unpinned 32 MiB pages without noisy VBAR analysis."""
    try:
        residency = vbar.get_residency()
        freeable_pages = sum(
            1 for status in residency if (int(status) & 1) and not (int(status) & 2)
        )
        reclaimable = freeable_pages * VBAR_PAGE_BYTES
        loaded_size = int(vbar.loaded_size())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0
    return max(min(reclaimable, loaded_size), 0)


def dynamic_vram_reclaimable(
    base_model,
    device: torch.device,
    *,
    include_current: bool = False,
) -> int:
    return sum(
        vbar_reclaimable_bytes(vbar)
        for vbar in dynamic_vbars(
            base_model,
            device,
            include_current=include_current,
        )
    )


def runtime_memory(device: torch.device, base_model=None) -> tuple[int, int, int]:
    """Return immediately usable bytes, total reserve, and usable ceiling."""
    del base_model
    import comfy.model_management as model_management

    total = int(model_management.get_total_memory(device))
    reserve = int(model_management.extra_reserved_memory())
    usable = max(total - reserve, 0)
    allocated = int(torch.cuda.memory_allocated(device))
    free = int(model_management.get_free_memory(device))
    return max(min(free, usable - allocated), 0), reserve, usable


def planning_available(
    runtime_plan: ActivationRuntimePlan | None,
    device: torch.device,
    rows: int,
    available: int,
    mode: str,
    operation: str,
) -> int:
    if runtime_plan is None or mode == "throughput":
        return available
    return runtime_plan.observe_available(device, rows, operation, available)


def should_log(runtime_plan: ActivationRuntimePlan | None, key: tuple) -> bool:
    logged = runtime_plan.logged_decisions if runtime_plan is not None else _LOGGED_DECISIONS
    if key in logged:
        return False
    logged.add(key)
    return True


def memory_diagnostics(device: torch.device) -> tuple[int, int, int, int]:
    """Best-effort allocator and DynamicVRAM pressure counters."""
    allocated = reserved = raw_free = aimdo_usage = 0
    try:
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        raw_free = int(torch.cuda.mem_get_info(device)[0])
    except (RuntimeError, TypeError):
        pass
    try:
        import comfy.memory_management as memory_management
        import comfy_aimdo.control as aimdo_control

        if memory_management.aimdo_enabled:
            aimdo_usage = int(aimdo_control.get_total_vram_usage())
    except (ImportError, AttributeError, RuntimeError, TypeError):
        pass
    return allocated, reserved, raw_free, aimdo_usage


def log_memory_diagnostics(device: torch.device, base_model=None) -> str:
    allocated, reserved, raw_free, aimdo_usage = memory_diagnostics(device)
    result = (
        f"torch_active={allocated / 1024**3:.2f} GiB "
        f"torch_reserved={reserved / 1024**3:.2f} GiB "
        f"cuda_free={raw_free / 1024**3:.2f} GiB "
        f"aimdo_usage={aimdo_usage / 1024**3:.2f} GiB"
    )
    if base_model is not None:
        reclaimable = dynamic_vram_reclaimable(base_model, device)
        result += f" inactive_reclaimable={reclaimable / 1024**3:.2f} GiB"
    return result


def ensure_dynamic_vram_headroom(
    base_model,
    device: torch.device,
    *,
    rows: int,
    operation: str,
    estimated_peak_bytes: int,
    runtime_plan: ActivationRuntimePlan | None = None,
    _runtime_memory_fn=runtime_memory,
    _dynamic_vbars_fn=dynamic_vbars,
    _diagnostics_fn=log_memory_diagnostics,
) -> int:
    """Release inactive VBAR mappings only when the selected tier cannot fit."""
    device = torch.device(device)
    if base_model is None or device.type != "cuda":
        return 0
    try:
        available, _reserve, usable = _runtime_memory_fn(device)
    except (ImportError, RuntimeError, TypeError):
        return 0
    safety = max(768 * MIB, int(usable * 0.075))
    desired = int(estimated_peak_bytes) + safety + 512 * MIB
    deficit = max(desired - available, 0)
    if deficit <= 0:
        return 0
    if runtime_plan is not None and not runtime_plan.should_request_reclaim(
        device, rows, operation, deficit
    ):
        return 0

    freed = 0
    remaining = deficit
    for vbar in _dynamic_vbars_fn(base_model, device):
        free_memory = getattr(vbar, "free_memory", None)
        if not callable(free_memory):
            continue
        try:
            released = max(int(free_memory(remaining) or 0), 0)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            LOG.debug("DynamicVRAM headroom request was unavailable: %s", error)
            continue
        freed += released
        remaining = max(remaining - released, 0)
        if remaining <= 0:
            break
    if freed > 0:
        WORKFLOW_TIMELINE.sample("dynamic_vram_reclaims")
        WORKFLOW_TIMELINE.sample("dynamic_vram_released_mib", freed / MIB)
        LOG.info(
            "MiniMax H3 DynamicVRAM headroom: op=%s rows=%d requested=%.1f MiB "
            "released=%.1f MiB %s",
            operation,
            rows,
            deficit / MIB,
            freed / MIB,
            _diagnostics_fn(device, base_model),
        )
    return freed


__all__ = [
    "ActivationRuntimePlan",
    "current_model_is_dynamic",
    "dynamic_vram_reclaimable",
    "dynamic_vbars",
    "dynamic_weight_prefetch_reserve",
    "ensure_dynamic_vram_headroom",
    "log_memory_diagnostics",
    "memory_diagnostics",
    "planning_available",
    "runtime_memory",
    "should_log",
    "vbar_reclaimable_bytes",
]
