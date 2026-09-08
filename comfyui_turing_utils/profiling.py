"""Opt-in CUDA phase timing with no events or synchronization by default."""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .kernel_api import (
    attention_kernel_architectures,
    attention_runtime_profile_schema,
    kernel_version,
)
from .log import get_logger


LOG = get_logger("profile")


def _profile_call_limit() -> int:
    value = os.environ.get("COMFYUI_TURING_UTILS_PROFILE_CALLS", "0").strip()
    try:
        return max(int(value), 0)
    except ValueError:
        LOG.warning(
            "Ignoring invalid COMFYUI_TURING_UTILS_PROFILE_CALLS=%r; expected a non-negative integer",
            value,
        )
        return 0


def _profile_bucket_limit() -> int:
    value = os.environ.get("COMFYUI_TURING_UTILS_PROFILE_BUCKETS", "4").strip()
    try:
        return max(int(value), 1)
    except ValueError:
        LOG.warning(
            "Ignoring invalid COMFYUI_TURING_UTILS_PROFILE_BUCKETS=%r; "
            "expected a positive integer",
            value,
        )
        return 4


def _runtime_profile_metadata() -> dict[str, str | int | bool]:
    architectures = attention_kernel_architectures()
    architecture_text = ",".join(architectures) if architectures else "unknown"
    result: dict[str, str | int | bool] = {
        "kernel": kernel_version(),
        "compiled_attention": architecture_text,
        "profile_schema": attention_runtime_profile_schema(),
        "device": "unknown",
        "device_sm": "unknown",
        "native_arch": "unknown",
    }
    try:
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        device_arch = f"sm{major}{minor}"
        result["device"] = str(torch.cuda.get_device_name(index))
        result["device_sm"] = device_arch
        if architectures:
            result["native_arch"] = any(
                architecture.split("+")[0] == device_arch
                for architecture in architectures
            )
    except (AttributeError, RuntimeError):
        pass
    return result


@dataclass(slots=True)
class _ProfileBucket:
    key: str
    kind: str
    shape: tuple[int, ...]
    metadata: tuple[tuple[str, str], ...]
    calls: int = 0
    pending: bool = False
    reported: bool = False
    samples: Counter = field(default_factory=Counter)
    tensor_samples: list[tuple[str, torch.Tensor, float]] = field(
        default_factory=list
    )


class CudaPhaseProfiler:
    """Collect bounded CUDA windows without synchronizing inside a sampler.

    The original profiler stopped forever after the first 50 attention calls,
    which meant a two-resolution H3 workflow only described its first sampler.
    Buckets are now keyed by operation, tensor shape, and execution metadata.
    Attention and MLP windows can therefore be collected together and emitted
    by the already-required DynamicVRAM sampler fence.
    """

    def __init__(self, call_limit: int, bucket_limit: int = 4):
        self.call_limit = int(call_limit)
        self.bucket_limit = max(int(bucket_limit), 1)
        self.calls = 0
        self.records: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._record_buckets: list[str] = []
        self._buckets: dict[str, _ProfileBucket] = {}
        self._current_bucket: str | None = None
        self._suppressed_scope = False
        self._implicit_record_start = 0
        self.shapes = Counter()
        self.reported = False
        self.pending_report = False
        self.defer_sync_to_sampler_boundary = False

    @property
    def enabled(self) -> bool:
        if self.call_limit <= 0:
            return False
        if self._suppressed_scope:
            return False
        if self._current_bucket is not None:
            bucket = self._buckets.get(self._current_bucket)
            return bool(bucket is not None and not bucket.pending and not bucket.reported)
        # Legacy callers may record one implicit bucket before selecting a
        # scope. Once any bucket exists, only begin_operation() may enable a
        # new window; this prevents unrelated CUDA events between samplers.
        return not self._buckets and len(self._buckets) < self.bucket_limit

    def defer_to_sampler_boundary(self) -> None:
        """Use an existing outer sampler fence instead of a mid-model sync."""
        self.defer_sync_to_sampler_boundary = True

    @staticmethod
    def _shape_tuple(value) -> tuple[int, ...]:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else tuple(value)
        return tuple(int(dimension) for dimension in shape)

    @staticmethod
    def _bucket_key(
        kind: str,
        shape: tuple[int, ...],
        metadata: tuple[tuple[str, str], ...],
    ) -> str:
        suffix = ",".join(f"{name}={value}" for name, value in metadata)
        return f"{kind}:{shape}" + (f"[{suffix}]" if suffix else "")

    def begin_operation(self, kind: str, value_or_shape, **metadata) -> bool:
        """Select a profiling bucket for one complete model operation."""
        self._suppressed_scope = False
        if self.call_limit <= 0:
            self._current_bucket = None
            return False
        shape = self._shape_tuple(value_or_shape)
        normalized = tuple(sorted((str(name), str(value)) for name, value in metadata.items()))
        key = self._bucket_key(str(kind), shape, normalized)
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.bucket_limit:
                self._current_bucket = None
                self._suppressed_scope = True
                return False
            bucket = _ProfileBucket(key, str(kind), shape, normalized)
            self._buckets[key] = bucket
        if bucket.pending or bucket.reported:
            self._current_bucket = None
            self._suppressed_scope = True
            return False
        self._current_bucket = key
        return True

    def cancel_operation(self) -> None:
        """Drop the current scope after an unsupported-path fallback."""
        key = self._current_bucket
        if key is not None:
            retained = [
                (record, record_key)
                for record, record_key in zip(self.records, self._record_buckets)
                if record_key != key
            ]
            self.records[:] = [record for record, _record_key in retained]
            self._record_buckets[:] = [
                record_key for _record, record_key in retained
            ]
            bucket = self._buckets.get(key)
            if bucket is not None and bucket.calls == 0:
                self._buckets.pop(key, None)
        self._current_bucket = None
        self._suppressed_scope = False

    def sample(self, name: str, value: int | float) -> None:
        """Attach a synchronization-free scalar diagnostic to this bucket."""
        if self._current_bucket is None:
            return
        bucket = self._buckets.get(self._current_bucket)
        if bucket is not None and not bucket.pending and not bucket.reported:
            bucket.samples[str(name)] += value

    def sample_tensor(
        self,
        name: str,
        value: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> None:
        """Defer a device scalar read until the sampler's existing CUDA fence."""
        if self._current_bucket is None or not torch.is_tensor(value):
            return
        bucket = self._buckets.get(self._current_bucket)
        if bucket is not None and not bucket.pending and not bucket.reported:
            bucket.tensor_samples.append((str(name), value.detach(), float(scale)))

    def call(self, phase: str, function: Callable, /, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.records.append((str(phase), start, end))
        self._record_buckets.append(self._current_bucket or "__implicit__")
        return result

    def complete_operation(self, kind: str | None = None, value_or_shape=None) -> None:
        if self.call_limit <= 0:
            return
        if self._suppressed_scope:
            self._suppressed_scope = False
            self._current_bucket = None
            return
        key = self._current_bucket
        if key is None:
            if value_or_shape is None:
                return
            shape = self._shape_tuple(value_or_shape)
            operation = str(kind or "operation")
            key = self._bucket_key(operation, shape, ())
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.bucket_limit:
                    return
                bucket = _ProfileBucket(key, operation, shape, ())
                self._buckets[key] = bucket
            for index in range(self._implicit_record_start, len(self._record_buckets)):
                if self._record_buckets[index] == "__implicit__":
                    self._record_buckets[index] = key
            self._implicit_record_start = len(self._record_buckets)
        bucket = self._buckets.get(key)
        self._current_bucket = None
        if bucket is None or bucket.pending or bucket.reported:
            return
        bucket.calls += 1
        self.calls += 1
        self.shapes[str(bucket.shape)] += 1
        if bucket.calls < self.call_limit:
            return
        bucket.pending = True
        self.pending_report = True
        bucket_records = [
            record
            for record, record_key in zip(self.records, self._record_buckets)
            if record_key == key
        ]
        if not bucket_records:
            bucket.reported = True
            bucket.pending = False
            self.reported = True
            self.pending_report = any(item.pending for item in self._buckets.values())
            return
        if self.defer_sync_to_sampler_boundary:
            return
        bucket_records[-1][2].synchronize()
        self.report_after_synchronize()

    def complete_attention(self, query_or_shape) -> None:
        self.complete_operation("attention", query_or_shape)

    def report_after_synchronize(self) -> bool:
        """Emit a pending report after the caller has already drained CUDA."""
        pending = [bucket for bucket in self._buckets.values() if bucket.pending]
        if not pending:
            return False
        runtime = _runtime_profile_metadata()
        reported_keys = set()
        for bucket in pending:
            totals = Counter()
            counts = Counter()
            for (phase, start, end), key in zip(self.records, self._record_buckets):
                if key != bucket.key:
                    continue
                totals[phase] += start.elapsed_time(end)
                counts[phase] += 1
            total = sum(totals.values())
            LOG.warning(
                "cuda_phases bucket=%s calls=%d recorded_cuda=%.3f ms "
                "device=%s device_sm=%s kernel=%s compiled_attention=[%s] "
                "native_arch=%s profile_schema=%s",
                bucket.key,
                bucket.calls,
                total,
                runtime["device"],
                runtime["device_sm"],
                runtime["kernel"],
                runtime["compiled_attention"],
                runtime["native_arch"],
                runtime["profile_schema"],
            )
            for phase, elapsed in totals.most_common():
                LOG.warning(
                    "  %-32s calls=%4d total=%10.3f ms avg=%8.3f ms %6.2f%%",
                    phase,
                    counts[phase],
                    elapsed,
                    elapsed / counts[phase],
                    elapsed * 100.0 / total if total else 0.0,
                )
            if bucket.samples:
                LOG.warning(
                    "  counters: %s",
                    " ".join(
                        f"{name}={value}" for name, value in sorted(bucket.samples.items())
                    ),
                )
            if bucket.tensor_samples:
                tensor_totals = Counter()
                for name, value, scale in bucket.tensor_samples:
                    tensor_totals[name] += float(value.item()) * scale
                LOG.warning(
                    "  deferred_counters: %s",
                    " ".join(
                        f"{name}={value:.6g}"
                        for name, value in sorted(tensor_totals.items())
                    ),
                )
                bucket.tensor_samples.clear()
            bucket.pending = False
            bucket.reported = True
            reported_keys.add(bucket.key)
        retained = [
            (record, key)
            for record, key in zip(self.records, self._record_buckets)
            if key not in reported_keys
        ]
        self.records[:] = [record for record, _key in retained]
        self._record_buckets[:] = [key for _record, key in retained]
        self._implicit_record_start = len(self.records)
        self.pending_report = any(bucket.pending for bucket in self._buckets.values())
        self.reported = True
        return True


def _timeline_enabled() -> bool:
    value = os.environ.get("COMFYUI_TURING_UTILS_TIMELINE", "0").strip().lower()
    return value not in ("", "0", "false", "off", "no")


@dataclass(slots=True)
class _TimelineWindow:
    index: int
    label: str
    device: torch.device
    wall_start: float
    cuda_start: torch.cuda.Event
    cuda_end: torch.cuda.Event
    allocated_start: int
    reserved_start: int
    counters: Counter = field(default_factory=Counter)


class WorkflowTimeline:
    """Opt-in sampler timeline piggybacking on the mandatory outer fence."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._next_index = 1
        self._active: _TimelineWindow | None = None

    def begin(
        self, device: torch.device, label: str = "sampler"
    ) -> _TimelineWindow | None:
        if not self.enabled:
            return None
        device = torch.device(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        window = _TimelineWindow(
            self._next_index,
            str(label),
            device,
            time.perf_counter(),
            start,
            end,
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
        )
        self._next_index += 1
        self._active = window
        return window

    def sample(self, name: str, value: int | float = 1) -> None:
        if self._active is not None:
            self._active.counters[str(name)] += value

    def record_end(self, window: _TimelineWindow | None) -> None:
        if window is not None:
            window.cuda_end.record()

    def finish_after_synchronize(self, window: _TimelineWindow | None) -> bool:
        if window is None:
            return False
        wall_ms = (time.perf_counter() - window.wall_start) * 1000.0
        cuda_ms = float(window.cuda_start.elapsed_time(window.cuda_end))
        allocated_end = int(torch.cuda.memory_allocated(window.device))
        reserved_end = int(torch.cuda.memory_reserved(window.device))
        peak_allocated = int(torch.cuda.max_memory_allocated(window.device))
        residual_ms = max(wall_ms - cuda_ms, 0.0)
        counters = " ".join(
            f"{name}={value}" for name, value in sorted(window.counters.items())
        )
        LOG.warning(
            "timeline span=%d label=%s wall=%.3f ms cuda=%.3f ms "
            "host_or_transfer=%.3f ms allocated=%.1f->%.1f MiB "
            "peak=%.1f MiB reserved=%.1f->%.1f MiB%s",
            window.index,
            window.label,
            wall_ms,
            cuda_ms,
            residual_ms,
            window.allocated_start / (1024.0 * 1024.0),
            allocated_end / (1024.0 * 1024.0),
            peak_allocated / (1024.0 * 1024.0),
            window.reserved_start / (1024.0 * 1024.0),
            reserved_end / (1024.0 * 1024.0),
            f" counters=[{counters}]" if counters else "",
        )
        if self._active is window:
            self._active = None
        return True

    def call(
        self,
        label: str,
        device: torch.device,
        function: Callable,
        /,
        *args,
        **kwargs,
    ):
        """Measure a non-sampler GPU phase when timeline diagnostics are enabled."""
        if not self.enabled or torch.device(device).type != "cuda":
            return function(*args, **kwargs)
        torch.cuda.synchronize(device)
        window = self.begin(device, label)
        try:
            return function(*args, **kwargs)
        finally:
            self.record_end(window)
            torch.cuda.synchronize(device)
            self.finish_after_synchronize(window)


CUDA_PHASE_PROFILER = CudaPhaseProfiler(
    _profile_call_limit(),
    _profile_bucket_limit(),
)
WORKFLOW_TIMELINE = WorkflowTimeline(_timeline_enabled())
__all__ = ["CUDA_PHASE_PROFILER", "WORKFLOW_TIMELINE"]
