from __future__ import annotations

import asyncio
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_turing_utils.runtime import stage_barrier as stage_barrier_module
BarrierPhase = stage_barrier_module.BarrierPhase
BarrierPlanError = stage_barrier_module.BarrierPlanError
BarrierPlanner = stage_barrier_module.BarrierPlanner
STAGE_BARRIER_NODE_ID = stage_barrier_module.STAGE_BARRIER_NODE_ID
STAGE_PATH_NODE_ID = stage_barrier_module.STAGE_PATH_NODE_ID
stage_barrier_candidates = stage_barrier_module.stage_barrier_candidates
_wait_for_active_barrier_phase = (
    stage_barrier_module._wait_for_active_barrier_phase
)


class _Prompt:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, node_id):
        return self.nodes[node_id]


def _normal():
    return {"class_type": "Normal", "inputs": {}}


def _barrier(stage):
    return {
        "class_type": STAGE_BARRIER_NODE_ID,
        "inputs": {"stage": stage},
    }


def _path(stage, value=None):
    inputs = {"stage": stage}
    if value is not None:
        inputs["value"] = value
    return {"class_type": STAGE_PATH_NODE_ID, "inputs": inputs}


def _blocking(nodes, edges):
    result = {node_id: {} for node_id in nodes}
    for source, target in edges:
        result[source][target] = {}
    return result


def _ready_nodes(pending, edges):
    blocked = {
        target
        for source, target in edges
        if source in pending and target in pending
    }
    return [node_id for node_id in pending if node_id not in blocked]


def _schedule(prompt, nodes, edges):
    pending = list(nodes)
    blocking = _blocking(nodes, edges)
    planner = BarrierPlanner(prompt)
    order = []
    while pending:
        available = _ready_nodes(pending, edges)
        candidates = planner.candidates(pending, blocking, available)
        if not candidates:
            raise AssertionError("scheduler returned no candidate")
        selected = candidates[0]
        order.append(selected)
        pending.remove(selected)
    return order, planner


class StageBarrierSchedulerTest(unittest.TestCase):
    def test_compiled_stage_paths_participate_in_phase_ordering(self):
        nodes = {
            "low_work": _normal(),
            "low": _path(0, ["low_work", 0]),
            "high_work": _normal(),
            "high": _path(1, ["high_work", 0]),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(
                nodes,
                (("low_work", "low"), ("high_work", "high")),
            ),
            ["high_work", "low_work"],
        )
        self.assertEqual(candidates, ["low_work"])

    def test_hidden_low_stage_path_prioritizes_lazy_branch_decision(self):
        nodes = {
            "condition": _normal(),
            "selected_work": _normal(),
            "hidden_low": _path(0, ["selected_work", 0]),
            "switch": {
                "class_type": "LazySwitch",
                "inputs": {
                    "condition": ["condition", 0],
                    "on_true": ["hidden_low", 0],
                },
            },
            "high_work": _normal(),
            "visible_high": _path(1, ["high_work", 0]),
        }
        pending = {"condition", "switch", "high_work", "visible_high"}
        edges = (("condition", "switch"), ("high_work", "visible_high"))
        planner = BarrierPlanner(_Prompt(nodes))

        self.assertEqual(
            planner.candidates(
                pending,
                _blocking(pending, edges),
                ["high_work", "condition"],
            ),
            ["condition"],
        )
        self.assertEqual(
            planner.candidates(
                pending - {"condition"},
                _blocking(
                    pending - {"condition"},
                    (("high_work", "visible_high"),),
                ),
                ["high_work", "switch"],
            ),
            ["switch"],
        )

    def test_hidden_branch_discovery_never_schedules_its_upstream(self):
        nodes = {
            "condition": _normal(),
            "unselected_work": _normal(),
            "hidden": _path(0, ["unselected_work", 0]),
            "switch": {
                "class_type": "LazySwitch",
                "inputs": {
                    "condition": ["condition", 0],
                    "on_false": ["hidden", 0],
                },
            },
            "unrelated": _normal(),
        }
        pending = {"condition", "switch", "unrelated"}
        edges = (("condition", "switch"),)

        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            pending,
            _blocking(pending, edges),
            ["unrelated", "condition"],
        )

        self.assertEqual(candidates, ["condition"])
        self.assertNotIn("unselected_work", candidates)

    def test_lower_stage_is_selected_first_when_dependencies_allow_it(self):
        nodes = {
            "low_work": _normal(),
            "low": _barrier(0),
            "high_work": _normal(),
            "high": _barrier(1),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(
                nodes,
                (("low_work", "low"), ("high_work", "high")),
            ),
            ["high_work", "low_work"],
        )
        self.assertEqual(candidates, ["low_work"])

    def test_same_phase_barriers_hold_unrelated_downstream_work(self):
        nodes = {
            "ready_barrier": _barrier(0),
            "other_work": _normal(),
            "other_barrier": _barrier(0),
            "unrelated": _normal(),
        }
        candidates = stage_barrier_candidates(
            _Prompt(nodes),
            nodes,
            _blocking(nodes, (("other_work", "other_barrier"),)),
            ["unrelated", "ready_barrier", "other_work"],
        )
        self.assertEqual(candidates, ["ready_barrier", "other_work"])

    def test_stage_reset_creates_a_new_round(self):
        nodes = {
            "a_work": _normal(),
            "a_stage_1": _barrier(1),
            "next_work": _normal(),
            "next_stage_0": _barrier(0),
            "b_work": _normal(),
            "b_stage_1": _barrier(1),
        }
        edges = (
            ("a_work", "a_stage_1"),
            ("a_stage_1", "next_work"),
            ("next_work", "next_stage_0"),
            ("b_work", "b_stage_1"),
        )
        order, planner = _schedule(_Prompt(nodes), nodes, edges)

        self.assertEqual(
            planner.phase_for("a_stage_1"), BarrierPhase(0, 1)
        )
        self.assertEqual(
            planner.phase_for("b_stage_1"), BarrierPhase(0, 1)
        )
        self.assertEqual(
            planner.phase_for("next_stage_0"), BarrierPhase(1, 0)
        )
        self.assertLess(order.index("a_stage_1"), order.index("next_stage_0"))
        self.assertLess(order.index("b_stage_1"), order.index("next_stage_0"))

    def test_repeated_workflow_stages_are_grouped_by_automatic_round(self):
        nodes = {
            "a0": _barrier(0),
            "a1": _barrier(1),
            "a2": _barrier(2),
            "b0": _barrier(0),
            "b1": _barrier(1),
            "b2": _barrier(2),
            "c0": _barrier(0),
            "c1": _barrier(1),
            "c2": _barrier(2),
        }
        edges = (
            ("a0", "a1"),
            ("a1", "a2"),
            ("b0", "b1"),
            ("b1", "b2"),
            ("a2", "c0"),
            ("c0", "c1"),
            ("c1", "c2"),
        )
        order, planner = _schedule(_Prompt(nodes), nodes, edges)

        for prefix in ("a", "b"):
            for stage in range(3):
                self.assertEqual(
                    planner.phase_for(f"{prefix}{stage}"),
                    BarrierPhase(0, stage),
                )
        for stage in range(3):
            self.assertEqual(
                planner.phase_for(f"c{stage}"), BarrierPhase(1, stage)
            )
        self.assertLess(order.index("b2"), order.index("c0"))

    def test_downstream_low_stage_does_not_promote_its_prerequisite(self):
        nodes = {
            "high_work": _normal(),
            "high_dependency": _barrier(4),
            "deferred_low": _barrier(0),
            "independent_mid": _barrier(1),
        }
        edges = (
            ("high_work", "high_dependency"),
            ("high_dependency", "deferred_low"),
        )
        planner = BarrierPlanner(_Prompt(nodes))
        candidates = planner.candidates(
            nodes,
            _blocking(nodes, edges),
            ["independent_mid", "high_work"],
        )

        self.assertEqual(candidates, ["independent_mid"])
        self.assertEqual(
            planner.phase_for("high_dependency"), BarrierPhase(0, 4)
        )
        self.assertEqual(
            planner.phase_for("deferred_low"), BarrierPhase(1, 0)
        )

    def test_equal_stage_dependency_remains_in_the_same_phase(self):
        nodes = {
            "first": _barrier(1),
            "dependent": _barrier(1),
            "peer": _barrier(1),
            "later": _barrier(2),
        }
        edges = (("first", "dependent"), ("first", "later"))
        order, planner = _schedule(_Prompt(nodes), nodes, edges)

        self.assertEqual(
            planner.phase_for("first"), BarrierPhase(0, 1)
        )
        self.assertEqual(
            planner.phase_for("dependent"), BarrierPhase(0, 1)
        )
        self.assertLess(order.index("dependent"), order.index("later"))
        self.assertLess(order.index("peer"), order.index("later"))

    def test_late_lower_stage_cannot_reopen_a_completed_phase(self):
        nodes = {"first": _barrier(1), "later": _barrier(2)}
        prompt = _Prompt(nodes)
        planner = BarrierPlanner(prompt)
        blocking = _blocking(nodes, ())

        self.assertEqual(
            planner.candidates(nodes, blocking, ["first", "later"]),
            ["first"],
        )
        self.assertEqual(
            planner.candidates(["later"], blocking, ["later"]),
            ["later"],
        )

        nodes["late"] = _barrier(0)
        blocking["late"] = {}
        self.assertEqual(
            planner.candidates(
                ["later", "late"], blocking, ["late", "later"]
            ),
            ["later"],
        )
        self.assertEqual(planner.phase_for("late"), BarrierPhase(1, 0))

    def test_late_upstream_barrier_can_move_pending_work_later(self):
        nodes = {"target": _barrier(1)}
        prompt = _Prompt(nodes)
        planner = BarrierPlanner(prompt)
        blocking = _blocking(nodes, ())
        planner.candidates(["target"], blocking, ["target"])

        nodes["prerequisite"] = _barrier(2)
        blocking["prerequisite"] = {"target": {}}
        candidates = planner.candidates(
            ["prerequisite", "target"],
            blocking,
            ["prerequisite"],
        )

        self.assertEqual(candidates, ["prerequisite"])
        self.assertEqual(
            planner.phase_for("prerequisite"), BarrierPhase(0, 2)
        )
        self.assertEqual(planner.phase_for("target"), BarrierPhase(1, 1))

    def test_no_barriers_preserves_comfyui_candidate_order(self):
        nodes = {"first": _normal(), "second": _normal()}
        candidates = stage_barrier_candidates(
            _Prompt(nodes), nodes, _blocking(nodes, ()), ["second", "first"]
        )
        self.assertEqual(candidates, ["second", "first"])

    def test_real_graph_cycle_has_an_explicit_barrier_error(self):
        nodes = {"first": _barrier(0), "second": _barrier(1)}
        with self.assertRaisesRegex(BarrierPlanError, "dependency cycle"):
            stage_barrier_candidates(
                _Prompt(nodes),
                nodes,
                _blocking(
                    nodes, (("first", "second"), ("second", "first"))
                ),
                [],
            )

    def test_random_dags_preserve_dependencies_and_monotonic_phases(self):
        randomizer = random.Random(20260830)
        for _ in range(40):
            nodes = {
                f"barrier_{index}": _barrier(randomizer.randrange(4))
                for index in range(10)
            }
            edges = tuple(
                (f"barrier_{source}", f"barrier_{target}")
                for source in range(10)
                for target in range(source + 1, 10)
                if randomizer.random() < 0.18
            )
            order, planner = _schedule(_Prompt(nodes), nodes, edges)
            positions = {
                node_id: index for index, node_id in enumerate(order)
            }

            for source, target in edges:
                self.assertLess(positions[source], positions[target])
                source_phase = planner.phase_for(source)
                target_phase = planner.phase_for(target)
                self.assertIsNotNone(source_phase)
                self.assertIsNotNone(target_phase)
                self.assertLessEqual(source_phase, target_phase)
                source_stage = nodes[source]["inputs"]["stage"]
                target_stage = nodes[target]["inputs"]["stage"]
                if target_stage < source_stage:
                    self.assertGreater(
                        target_phase.round, source_phase.round
                    )


class _ExternallyBlockedExecutionList:
    def __init__(self):
        nodes = {
            "active": _barrier(0),
            "unrelated": _normal(),
        }
        self.dynprompt = _Prompt(nodes)
        self.pendingNodes = dict.fromkeys(nodes, True)
        self.blocking = _blocking(nodes, ())
        self.blockCount = {"active": 1, "unrelated": 0}
        self.externalBlocks = 1
        self.unblockedEvent = asyncio.Event()

    def is_empty(self):
        return not self.pendingNodes

    def get_ready_nodes(self):
        return [
            node_id
            for node_id in self.pendingNodes
            if self.blockCount[node_id] == 0
        ]

    def unblock_active(self):
        self.externalBlocks = 0
        self.blockCount["active"] = 0
        self.unblockedEvent.set()


class StageBarrierAsyncWaitTest(unittest.IsolatedAsyncioTestCase):
    async def test_waits_instead_of_advancing_unrelated_work(self):
        execution_list = _ExternallyBlockedExecutionList()
        wait_task = asyncio.create_task(
            _wait_for_active_barrier_phase(execution_list)
        )
        await asyncio.sleep(0)

        self.assertFalse(wait_task.done())
        execution_list.unblock_active()
        await asyncio.wait_for(wait_task, timeout=1.0)

    async def test_inconsistent_internal_graph_keeps_liveness_fallback(self):
        execution_list = _ExternallyBlockedExecutionList()
        execution_list.externalBlocks = 0

        await asyncio.wait_for(
            _wait_for_active_barrier_phase(execution_list), timeout=1.0
        )


if __name__ == "__main__":
    unittest.main()
