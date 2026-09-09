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
_STAGE_PATCH_MARKER = "_turing_utils_stage_barrier_staging"
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
        *,
        fallback_to_available: bool = True,
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

        # External async blockers can make the selected phase temporarily
        # unable to advance while unrelated work is ready.  The async staging
        # wrapper waits in that case.  Stateless callers and genuinely
        # inconsistent graphs retain the old liveness fallback.
        if not fallback_to_available:
            return []
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


def _planner_for(execution_list) -> BarrierPlanner:
    planner = getattr(execution_list, _PLANNER_ATTRIBUTE, None)
    if planner is None or planner.dynprompt is not execution_list.dynprompt:
        planner = BarrierPlanner(execution_list.dynprompt)
        setattr(execution_list, _PLANNER_ATTRIBUTE, planner)
    return planner


async def _wait_for_active_barrier_phase(execution_list) -> None:
    """Wait while only unrelated work is ready for an externally blocked phase.

    ``ExecutionList.ux_friendly_pick_node`` is synchronous.  Restricting its
    candidate list alone therefore cannot distinguish a dependency cycle from
    a barrier ancestor that is temporarily blocked by an asynchronous node.
    ComfyUI exposes exactly that distinction through ``externalBlocks`` and
    ``unblockedEvent``, so wait here before its normal staging method runs.
    """

    planner = _planner_for(execution_list)
    waited = False
    while not execution_list.is_empty():
        available = execution_list.get_ready_nodes()
        if not available:
            # ComfyUI's own staging loop already handles the case where every
            # ready node is externally blocked, including cycle reporting.
            return
        allowed = planner.candidates(
            execution_list.pendingNodes,
            execution_list.blocking,
            available,
            fallback_to_available=False,
        )
        if allowed or execution_list.externalBlocks <= 0:
            if waited:
                LOG.info("Stage Barrier rendezvous resumed after async work")
            return

        if not waited:
            LOG.info(
                "Stage Barrier is waiting for an async ancestor instead of "
                "advancing unrelated work"
            )
            waited = True
        # An unblock racing with this wait leaves the Event set, so there is no
        # lost-wakeup window between the counter check and await.
        await execution_list.unblockedEvent.wait()
        execution_list.unblockedEvent.clear()


def install_stage_barrier_scheduler() -> bool:
    """Wrap ComfyUI's async staging and ready-node picker exactly once."""

    try:
        from comfy_execution.graph import ExecutionList
    except (ImportError, AttributeError):
        LOG.warning(
            "Stage Barrier ordering is unavailable: this ComfyUI build has no "
            "compatible ExecutionList scheduler"
        )
        return False

    current_pick = getattr(ExecutionList, "ux_friendly_pick_node", None)
    current_stage = getattr(ExecutionList, "stage_node_execution", None)
    if current_pick is None or current_stage is None:
        LOG.warning(
            "Stage Barrier ordering is unavailable: ComfyUI's compatible "
            "staging methods were not found"
        )
        return False

    if not getattr(current_pick, _PATCH_MARKER, False):
        original_pick = current_pick

        def stage_aware_pick_node(self, node_list):
            candidates = _planner_for(self).candidates(
                self.pendingNodes,
                self.blocking,
                node_list,
            )
            return original_pick(self, candidates)

        setattr(stage_aware_pick_node, _PATCH_MARKER, True)
        setattr(stage_aware_pick_node, "_turing_utils_original", original_pick)
        ExecutionList.ux_friendly_pick_node = stage_aware_pick_node

    if not getattr(current_stage, _STAGE_PATCH_MARKER, False):
        original_stage = current_stage

        async def stage_aware_stage_node_execution(self):
            await _wait_for_active_barrier_phase(self)
            return await original_stage(self)

        setattr(stage_aware_stage_node_execution, _STAGE_PATCH_MARKER, True)
        setattr(
            stage_aware_stage_node_execution,
            "_turing_utils_original",
            original_stage,
        )
        ExecutionList.stage_node_execution = stage_aware_stage_node_execution

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
