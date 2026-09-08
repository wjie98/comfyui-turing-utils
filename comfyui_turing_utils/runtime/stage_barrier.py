"""Dependency-first Stage Barrier scheduling for ComfyUI.

Stage Barrier values are phase labels, not unconditional global priorities.
The planner collapses the active execution graph to a barrier-only DAG and
assigns every barrier an internal ``(round, stage)`` key. A dependency whose
stage label decreases starts a new round; otherwise it stays in the same
round. This keeps repeated stage sequences intuitive without ever overriding
ComfyUI's real data dependencies.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Iterable, Mapping, NamedTuple

from ..log import get_logger

LOG = get_logger("stage")
STAGE_BARRIER_NODE_ID = "TuringUtilsStageBarrier"
STAGE_PATH_NODE_ID = "TuringUtilsStagePath"
STAGE_SCHEDULING_NODE_IDS = frozenset(
    (STAGE_BARRIER_NODE_ID, STAGE_PATH_NODE_ID)
)
_PATCH_MARKER = "_turing_utils_stage_barrier_scheduler"
_PLANNER_ATTRIBUTE = "_turing_utils_stage_barrier_planner"


class BarrierPhase(NamedTuple):
    """The automatically inferred scheduling key for one barrier."""

    round: int
    stage: int


class BarrierPlanError(RuntimeError):
    """Raised when the active barrier dependency graph cannot be planned."""


def _barrier_stage(dynprompt, node_id: str) -> int | None:
    try:
        node = dynprompt.get_node(node_id)
    except (KeyError, TypeError, AttributeError):
        return None
    if node.get("class_type") not in STAGE_SCHEDULING_NODE_IDS:
        return None
    try:
        return max(0, int(node.get("inputs", {}).get("stage", 0)))
    except (TypeError, ValueError):
        # ComfyUI validates the widget before execution. Keeping malformed
        # workflows in stage zero here preserves a useful execution error at
        # the node instead of failing inside the scheduler.
        return 0


def _graph_predecessors(
    pending: set[str], blocking: Mapping[str, Mapping[str, object]]
) -> dict[str, set[str]]:
    predecessors = {node_id: set() for node_id in pending}
    for source in pending:
        for target in blocking.get(source, {}):
            if target in pending:
                predecessors[target].add(source)
    return predecessors


def _nearest_barrier_predecessors(
    target: str,
    barriers: set[str],
    predecessors: Mapping[str, set[str]],
) -> set[str]:
    """Find the first barrier encountered on every upstream path.

    Stopping at the first barrier produces the direct barrier DAG. Counting
    every transitive barrier as an edge would over-count stage resets on long
    dependency chains.
    """

    found = set()
    visited = {target}
    stack = list(predecessors.get(target, ()))
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in barriers:
            found.add(node_id)
            continue
        stack.extend(predecessors.get(node_id, ()))
    return found


def _barrier_predecessors(
    barriers: set[str],
    predecessors: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    return {
        node_id: _nearest_barrier_predecessors(
            node_id, barriers, predecessors
        )
        for node_id in barriers
    }


def _barrier_topological_order(
    barrier_predecessors: Mapping[str, set[str]],
) -> list[str]:
    successors = {node_id: set() for node_id in barrier_predecessors}
    indegree = {
        node_id: len(sources)
        for node_id, sources in barrier_predecessors.items()
    }
    for target, sources in barrier_predecessors.items():
        for source in sources:
            successors[source].add(target)

    ready = deque(
        sorted(
            (node_id for node_id, degree in indegree.items() if degree == 0),
            key=str,
        )
    )
    order = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        newly_ready = []
        for target in successors[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                newly_ready.append(target)
        ready.extend(sorted(newly_ready, key=str))

    if len(order) != len(barrier_predecessors):
        cyclic = sorted(
            (node_id for node_id, degree in indegree.items() if degree > 0),
            key=str,
        )
        raise BarrierPlanError(
            "Stage Barrier dependency cycle detected among nodes: "
            + ", ".join(map(str, cyclic))
        )
    return order


def _ancestors(
    targets: Iterable[str], predecessors: Mapping[str, set[str]]
) -> set[str]:
    required = set(targets)
    stack = list(required)
    while stack:
        node_id = stack.pop()
        for source in predecessors.get(node_id, ()):
            if source not in required:
                required.add(source)
                stack.append(source)
    return required


def _direct_hidden_stage_inputs(
    dynprompt,
    pending: set[str],
) -> dict[str, set[int]]:
    """Find stage paths currently hidden behind lazy or cached inputs.

    Lazy inputs are deliberately absent from ComfyUI's active topological
    graph until their consumer runs ``check_lazy_status``.  Scheduling that
    cheap decision point is necessary before a hidden low-stage path can join
    the rendezvous.  A cached Stage Path looks the same here and is harmless:
    its consumer simply completes without rematerializing the path.
    """

    consumers: dict[str, set[int]] = {}
    for node_id in pending:
        try:
            inputs = dynprompt.get_node(node_id).get("inputs", {})
        except (KeyError, TypeError, AttributeError):
            continue
        if not isinstance(inputs, Mapping):
            continue
        for value in inputs.values():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            source = value[0]
            if source in pending:
                continue
            stage = _barrier_stage(dynprompt, source)
            if stage is not None:
                consumers.setdefault(node_id, set()).add(stage)
    return consumers


class BarrierPlanner:
    """Persistent, prompt-local barrier plan.

    The plan survives individual node completions, so a newly exposed
    lower-stage descendant cannot jump ahead of peers in the phase that is
    already rendezvousing. Newly materialized lazy or dynamic barriers are
    incorporated without moving any pending barrier backwards in time.
    """

    def __init__(self, dynprompt):
        self.dynprompt = dynprompt
        self._phases: dict[str, BarrierPhase] = {}
        self._visible_barriers: set[str] = set()
        self._floor: BarrierPhase | None = None
        self._logged_initial_plan = False
        self._warned_unavailable_phase: BarrierPhase | None = None

    @property
    def floor(self) -> BarrierPhase | None:
        return self._floor

    def phase_for(self, node_id: str) -> BarrierPhase | None:
        """Return the assigned phase, including completed barriers."""

        return self._phases.get(node_id)

    def _record_completed(self, current_barriers: set[str]) -> None:
        disappeared = self._visible_barriers - current_barriers
        completed = [
            self._phases[node_id]
            for node_id in disappeared
            if node_id in self._phases
        ]
        if completed:
            latest = max(completed)
            if self._floor is None or latest > self._floor:
                self._floor = latest
        self._visible_barriers = set(current_barriers)

    def _minimum_round(self, stage: int) -> int:
        if self._floor is None:
            return 0
        return self._floor.round + int(stage < self._floor.stage)

    def _assign_phases(
        self,
        stages: Mapping[str, int],
        predecessors: Mapping[str, set[str]],
        *,
        dynamic_refresh: bool,
    ) -> None:
        barrier_ids = set(stages)
        direct_predecessors = _barrier_predecessors(
            barrier_ids, predecessors
        )
        order = _barrier_topological_order(direct_predecessors)
        assigned: dict[str, BarrierPhase] = {}

        for node_id in order:
            stage = stages[node_id]
            round_id = self._minimum_round(stage)
            old_phase = self._phases.get(node_id)
            if old_phase is not None:
                # Incremental lazy/dynamic discovery may move work later, but
                # it must never reopen an already passed phase.
                round_id = max(round_id, old_phase.round)
            for source in direct_predecessors[node_id]:
                source_phase = assigned[source]
                round_id = max(
                    round_id,
                    source_phase.round
                    + int(stage < source_phase.stage),
                )
            assigned[node_id] = BarrierPhase(round_id, stage)

        self._phases.update(assigned)
        phase_counts = Counter(assigned.values())
        summary = ",".join(
            f"r{phase.round}/s{phase.stage}:{count}"
            for phase, count in sorted(phase_counts.items())
        )
        if not self._logged_initial_plan or dynamic_refresh:
            LOG.info(
                "Stage Barrier plan%s: barriers=%d phases=[%s]",
                " refreshed" if dynamic_refresh else "",
                len(assigned),
                summary,
            )
            self._logged_initial_plan = True

    def candidates(
        self,
        pending_nodes: Iterable[str],
        blocking: Mapping[str, Mapping[str, object]],
        available_nodes: Iterable[str],
    ) -> list[str]:
        """Return ready nodes allowed to advance the earliest active phase."""

        available = list(available_nodes)
        pending = set(pending_nodes)
        stages = {
            node_id: stage
            for node_id in pending
            if (
                stage := _barrier_stage(self.dynprompt, node_id)
            )
            is not None
        }
        barrier_ids = set(stages)
        self._record_completed(barrier_ids)
        predecessors = _graph_predecessors(pending, blocking)
        hidden_consumers = _direct_hidden_stage_inputs(
            self.dynprompt, pending
        )
        hidden_phases = {
            node_id: min(
                BarrierPhase(self._minimum_round(stage), stage)
                for stage in hidden_stages
            )
            for node_id, hidden_stages in hidden_consumers.items()
        }

        if not barrier_ids:
            if hidden_phases:
                discovery_phase = min(hidden_phases.values())
                discovery_targets = [
                    node_id
                    for node_id, phase in hidden_phases.items()
                    if phase == discovery_phase
                ]
                required = _ancestors(discovery_targets, predecessors)
                candidates = [
                    node_id for node_id in available if node_id in required
                ]
                if candidates:
                    return candidates
            return available

        new_barriers = barrier_ids - set(self._phases)
        stale_phases = {
            node_id
            for node_id in barrier_ids
            if self._phases.get(node_id, BarrierPhase(-1, -1)).stage
            != stages[node_id]
            or self._phases.get(node_id, BarrierPhase(-1, -1))
            < BarrierPhase(
                self._minimum_round(stages[node_id]), stages[node_id]
            )
        }
        if new_barriers or stale_phases or not self._logged_initial_plan:
            self._assign_phases(
                stages,
                predecessors,
                dynamic_refresh=self._logged_initial_plan,
            )

        active_phase = min(self._phases[node_id] for node_id in barrier_ids)
        discoverable = [
            node_id
            for node_id, phase in hidden_phases.items()
            if phase <= active_phase
        ]
        if discoverable:
            discovery_phase = min(
                hidden_phases[node_id] for node_id in discoverable
            )
            discovery_targets = [
                node_id
                for node_id in discoverable
                if hidden_phases[node_id] == discovery_phase
            ]
            required = _ancestors(discovery_targets, predecessors)
            candidates = [
                node_id for node_id in available if node_id in required
            ]
            if candidates:
                return candidates

        targets = [
            node_id
            for node_id in barrier_ids
            if self._phases[node_id] == active_phase
        ]
        required = _ancestors(targets, predecessors)
        candidates = [
            node_id for node_id in available if node_id in required
        ]
        if candidates:
            self._warned_unavailable_phase = None
            return candidates

        # External async blockers or newly materializing lazy inputs can make
        # the selected phase temporarily unable to advance while unrelated
        # work is ready. Preserve liveness, but never hide the loss of strict
        # rendezvous ordering.
        if self._warned_unavailable_phase != active_phase:
            LOG.warning(
                "Stage Barrier phase r%d/s%d has no ready ancestor; "
                "temporarily deferring to ComfyUI scheduling",
                active_phase.round,
                active_phase.stage,
            )
            self._warned_unavailable_phase = active_phase
        return available


def stage_barrier_candidates(
    dynprompt,
    pending_nodes: Iterable[str],
    blocking: Mapping[str, Mapping[str, object]],
    available_nodes: Iterable[str],
) -> list[str]:
    """Plan one scheduler decision without retaining prompt-local state.

    Runtime integration uses :class:`BarrierPlanner` directly. This function
    remains as a convenient compatibility surface for diagnostics and tests.
    """

    return BarrierPlanner(dynprompt).candidates(
        pending_nodes, blocking, available_nodes
    )


def install_stage_barrier_scheduler() -> bool:
    """Wrap ComfyUI's ready-node picker exactly once."""

    try:
        from comfy_execution.graph import ExecutionList
    except (ImportError, AttributeError):
        LOG.warning(
            "Stage Barrier ordering is unavailable: this ComfyUI build has no "
            "compatible ExecutionList scheduler"
        )
        return False

    original = getattr(ExecutionList, "ux_friendly_pick_node", None)
    if original is None:
        LOG.warning(
            "Stage Barrier ordering is unavailable: ComfyUI's ready-node picker "
            "was not found"
        )
        return False
    if getattr(original, _PATCH_MARKER, False):
        return True

    def stage_aware_pick_node(self, node_list):
        planner = getattr(self, _PLANNER_ATTRIBUTE, None)
        if planner is None or planner.dynprompt is not self.dynprompt:
            planner = BarrierPlanner(self.dynprompt)
            setattr(self, _PLANNER_ATTRIBUTE, planner)
        candidates = planner.candidates(
            self.pendingNodes,
            self.blocking,
            node_list,
        )
        return original(self, candidates)

    setattr(stage_aware_pick_node, _PATCH_MARKER, True)
    setattr(stage_aware_pick_node, "_turing_utils_original", original)
    ExecutionList.ux_friendly_pick_node = stage_aware_pick_node
    LOG.info("Enabled dependency-first Stage Barrier scheduling")
    return True


__all__ = [
    "BarrierPhase",
    "BarrierPlanError",
    "BarrierPlanner",
    "STAGE_BARRIER_NODE_ID",
    "STAGE_PATH_NODE_ID",
    "STAGE_SCHEDULING_NODE_IDS",
    "install_stage_barrier_scheduler",
    "stage_barrier_candidates",
]
