"""Task-owned snapshots of native progress events; no model or tensor state."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
from statistics import median
import time


@dataclass
class Node:
    id: str
    class_type: str | None
    title: str | None
    started: float
    updated: float
    state: str = "running"
    value: float = 0
    maximum: float = 1
    reported: bool = False
    rates: deque = field(default_factory=lambda: deque(maxlen=5))

    def update(self, value, maximum, now):
        if not (math.isfinite(value) and math.isfinite(maximum)) or maximum <= 0:
            return
        if value < self.value or maximum != self.maximum:
            self.rates.clear()
        elif self.reported and self.value > 0 and value > self.value:
            # Exclude loading/compilation before the first completed unit.
            self.rates.append((now - self.updated) / (value - self.value))
        if not self.reported or value != self.value or maximum != self.maximum:
            self.updated = now
        self.reported, self.value, self.maximum = True, value, maximum

    def snapshot(self, now):
        remaining = None
        if (
            self.state == "running"
            and len(self.rates) >= 2
            and self.value < self.maximum
        ):
            rate = median(self.rates)
            if rate > 0 and max(self.rates) <= min(self.rates) * 2:
                estimate = (self.maximum - self.value) * rate - (now - self.updated)
                if estimate > 0:
                    remaining = round(estimate, 3)
        return {
            "id": self.id,
            "class_type": self.class_type,
            "title": self.title,
            "value": self.value if self.reported else None,
            "max": self.maximum if self.reported else None,
            "fraction": min(1, max(0, self.value / self.maximum))
            if self.reported
            else None,
            "elapsed_seconds": round(
                max(
                    0, (now if self.state == "running" else self.updated) - self.started
                ),
                3,
            ),
            "eta_seconds": remaining,
            "eta_source": "step_rate" if remaining is not None else None,
        }


class Progress:
    def __init__(self):
        self.created = self.changed = time.monotonic()
        self.phase = "receiving"
        self.durations = {}
        self.started_at = self.finished_at = self.finished = None
        self.prompt_id = None
        self.graph = {}
        self.nodes = {}
        self.cached = set()
        self.executions = self.completed_executions = 0
        self.targets, self.completed_targets = set(), set()
        self.signature = None

    def transition(self, phase, now=None):
        now = time.monotonic() if now is None else now
        if self.finished is None:
            self.durations[self.phase] = self.durations.get(self.phase, 0) + max(
                0, now - self.changed
            )
            self.phase, self.changed = phase, now

    def finish(self, status):
        self.transition(status)
        self.finished, self.finished_at = self.changed, time.time()

    def seconds(self, phase, now):
        return self.durations.get(phase, 0) + (
            max(0, now - self.changed)
            if self.phase == phase and self.finished is None
            else 0
        )

    def processing_seconds(self, now):
        return sum(
            self.seconds(phase, now) for phase in ("executing", "finalizing", "cleanup")
        )

    def event(self, event, data, now, graph=None):
        if self.finished is not None:
            return
        if event == "execution_start":
            self.transition("executing", now)
            if self.started_at is None:
                self.started_at = data["timestamp"] / 1000
            self.prompt_id, self.graph = data["prompt_id"], graph
            self.nodes, self.cached = {}, set()
            self.executions += 1
        elif data.get("prompt_id") != self.prompt_id:
            return
        elif event == "execution_cached":
            self.cached.update(data["nodes"])
        elif event == "progress_state":
            for identity, entry in data["nodes"].items():
                state = entry["state"]
                if state not in {"running", "finished"}:
                    continue
                node = self.nodes.get(identity)
                if node is None:
                    kind, title = self.graph.get(
                        identity,
                        self.graph.get(entry.get("real_node_id"), (None, None)),
                    )
                    node = self.nodes[identity] = Node(identity, kind, title, now, now)
                # Registry start/finish values (0/1, 1/1) are not actual steps.
                if state == "running" and (entry["value"] != 0 or entry["max"] != 1):
                    node.update(entry["value"], entry["max"], now)
                if state == "finished" and node.state != "finished":
                    node.state, node.updated = "finished", now
        elif event in {"execution_success", "execution_error", "execution_interrupted"}:
            for node in self.nodes.values():
                if node.state == "running":
                    node.state, node.updated = (
                        "finished" if event == "execution_success" else "failed",
                        now,
                    )
            self.completed_executions += event == "execution_success"
            self.transition(
                "finalizing" if event == "execution_success" else "cleanup", now
            )

    def snapshot(self, history):
        now = self.finished if self.finished is not None else time.monotonic()
        samples = history.get(self.signature, ())
        remaining, estimate_status = None, "estimating"
        if self.finished is not None:
            estimate_status = "complete" if self.phase == "succeeded" else "unavailable"
            remaining = 0 if self.phase == "succeeded" else None
        elif self.phase == "queued":
            estimate_status = "queued"
        elif self.phase not in {"receiving", "preparing"} and len(samples) >= 2:
            prediction = median(samples) - self.processing_seconds(now)
            if prediction > 0 and max(samples) <= min(samples) * 2:
                remaining, estimate_status = round(prediction, 3), "available"
            else:
                estimate_status = "overdue" if prediction <= 0 else "estimating"
        return {
            "progress": {
                "phase": self.phase,
                "prompt_id": self.prompt_id,
                "executions_started": self.executions,
                "executions_completed": self.completed_executions,
                "nodes": {
                    "known": len(self.graph.keys() | self.nodes.keys()),
                    "completed": sum(
                        n.state == "finished" and n.id not in self.cached
                        for n in self.nodes.values()
                    ),
                    "cached": len(self.cached),
                },
                "active_nodes": [
                    node.snapshot(now)
                    for node in self.nodes.values()
                    if node.state == "running"
                ],
                "outputs": {
                    "completed": len(self.completed_targets),
                    "total": len(self.targets),
                },
            },
            "timing": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": round(max(0, now - self.created), 3),
                "queue_seconds": round(self.seconds("queued", now), 3),
                "execution_seconds": round(self.processing_seconds(now), 3),
                "eta_seconds": remaining,
                "estimated_finish": time.time() + remaining
                if estimate_status == "available"
                else None,
                "eta_source": "history" if estimate_status == "available" else None,
                "eta_scope": "processing",
                "confidence": "low" if estimate_status == "available" else None,
                "estimate_status": estimate_status,
                "history_samples": len(samples),
            },
        }


class Observer:
    """Apply execution events and retain per-owner timing samples."""

    def __init__(self, jobs):
        self.jobs = jobs
        self.history = OrderedDict()

    def event(self, identity, event, data, now, graph):
        job = self.jobs.get(identity)
        if (
            job is not None
            and job.progress.finished is None
            and (event != "execution_start" or graph is not None)
        ):
            if event == "execution_start":
                job.status = "running"
            job.progress.event(event, data, now, graph)

    def remember(self, job):
        progress = job.progress
        if (
            job.status == "succeeded"
            and progress.signature is not None
            and progress.executions
        ):
            samples = self.history.setdefault(progress.signature, deque(maxlen=5))
            samples.append(progress.processing_seconds(progress.finished))
            self.history.move_to_end(progress.signature)
            while len(self.history) > 128:
                self.history.popitem(last=False)
