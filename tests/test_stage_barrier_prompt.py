from __future__ import annotations

from pathlib import Path
import sys
import unittest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.runtime.stage_barrier import (  # noqa: E402
    STAGE_BARRIER_NODE_ID,
    STAGE_PATH_NODE_ID,
)
from comfyui_turing_utils.runtime.stage_barrier_prompt import (  # noqa: E402
    compile_stage_barrier_prompt,
    compile_stage_barriers_on_prompt,
)


def _source(name):
    return {"class_type": name, "inputs": {}}


class StageBarrierPromptCompilerTest(unittest.TestCase):
    def test_each_visual_port_becomes_an_independent_unary_node(self):
        prompt = {
            "source_true": _source("Source"),
            "source_false": _source("Source"),
            "barrier": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {
                    "stage": 3,
                    "value_0": ["source_true", 0],
                    "value_1": ["source_false", 0],
                },
                "_meta": {"title": "Characters prepared"},
            },
            "switch": {
                "class_type": "TuringUtilsLazyIfElse",
                "inputs": {
                    "condition": True,
                    "on_true": ["barrier", 0],
                    "on_false": ["barrier", 1],
                },
            },
        }

        compiled, count = compile_stage_barrier_prompt(prompt)

        self.assertEqual(count, 2)
        self.assertNotIn("barrier", compiled)
        true_route = compiled["switch"]["inputs"]["on_true"][0]
        false_route = compiled["switch"]["inputs"]["on_false"][0]
        self.assertNotEqual(true_route, false_route)
        self.assertEqual(compiled[true_route]["class_type"], STAGE_PATH_NODE_ID)
        self.assertEqual(compiled[false_route]["class_type"], STAGE_PATH_NODE_ID)
        self.assertEqual(
            compiled[true_route]["inputs"],
            {"stage": 3, "value": ["source_true", 0]},
        )
        self.assertEqual(
            compiled[false_route]["inputs"],
            {"stage": 3, "value": ["source_false", 0]},
        )
        self.assertNotIn("source_false", repr(compiled[true_route]["inputs"]))
        self.assertNotIn("source_true", repr(compiled[false_route]["inputs"]))

    def test_compilation_is_deterministic_and_does_not_mutate_source(self):
        prompt = {
            "source": _source("Source"),
            "barrier": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {
                    "stage": 1,
                    "value_4": ["source", 0],
                },
            },
            "output": {
                "class_type": "Output",
                "inputs": {"value": ["barrier", 4]},
            },
        }
        original_barrier = dict(prompt["barrier"])
        original_inputs = dict(prompt["barrier"]["inputs"])

        first, first_count = compile_stage_barrier_prompt(prompt)
        second, second_count = compile_stage_barrier_prompt(prompt)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(prompt["barrier"], original_barrier)
        self.assertEqual(prompt["barrier"]["inputs"], original_inputs)

    def test_chained_barriers_rewrite_route_to_route_dependency(self):
        prompt = {
            "source": _source("Source"),
            "first": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {"stage": 0, "value_0": ["source", 0]},
            },
            "second": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {"stage": 1, "value_2": ["first", 0]},
            },
            "output": {
                "class_type": "Output",
                "inputs": {"value": ["second", 2]},
            },
        }

        compiled, count = compile_stage_barrier_prompt(prompt)

        self.assertEqual(count, 2)
        first_route = compiled["second.turing_stage_path.2"]["inputs"]["value"][0]
        self.assertEqual(first_route, "first.turing_stage_path.0")
        self.assertEqual(
            compiled["output"]["inputs"]["value"],
            ["second.turing_stage_path.2", 0],
        )

    def test_consumed_unconnected_output_compiles_to_empty_path(self):
        prompt = {
            "barrier": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {"stage": 5},
            },
            "output": {
                "class_type": "Output",
                "inputs": {"value": ["barrier", 7]},
            },
        }

        compiled, count = compile_stage_barrier_prompt(prompt)

        self.assertEqual(count, 1)
        route = compiled["barrier.turing_stage_path.7"]
        self.assertEqual(route["inputs"], {"stage": 5})
        self.assertEqual(
            compiled["output"]["inputs"]["value"],
            ["barrier.turing_stage_path.7", 0],
        )

    def test_static_api_values_are_supported(self):
        prompt = {
            "source": _source("Source"),
            "barrier": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {
                    "stage": 2,
                    "value_0": ["source", 0],
                },
            },
            "output": {
                "class_type": "Output",
                "inputs": {"value": ["barrier", 0]},
            },
        }

        compiled, count = compile_stage_barrier_prompt(prompt)

        self.assertEqual(count, 1)
        self.assertEqual(
            compiled["barrier.turing_stage_path.0"]["inputs"]["value"],
            ["source", 0],
        )

    def test_request_handler_preserves_non_prompt_fields(self):
        request = {
            "number": 4,
            "extra_data": {"client_id": "client"},
            "prompt": {
                "barrier": {
                    "class_type": STAGE_BARRIER_NODE_ID,
                    "inputs": {"stage": 0},
                },
                "output": {
                    "class_type": "Output",
                    "inputs": {"value": ["barrier", 0]},
                },
            },
        }

        result = compile_stage_barriers_on_prompt(request)

        self.assertEqual(result["number"], 4)
        self.assertEqual(result["extra_data"], {"client_id": "client"})
        self.assertIn("barrier", request["prompt"])
        self.assertNotIn("barrier", result["prompt"])

    def test_comfy_execution_graph_materializes_only_selected_route(self):
        import nodes as comfy_nodes
        from comfy_execution.graph import DynamicPrompt, ExecutionList
        from comfyui_turing_utils.nodes.logic import LazyIfElse, StagePath

        class Source:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            RETURN_TYPES = ("*",)
            FUNCTION = "execute"

        class Output:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": ("*",)}}

            RETURN_TYPES = ()
            FUNCTION = "execute"
            OUTPUT_NODE = True

        class EmptyCache:
            def get_local(self, _node_id):
                return None

            def set_local(self, _node_id, _value):
                return None

        prompt = {
            "source_true": _source("TestStageSource"),
            "source_false": _source("TestStageSource"),
            "barrier": {
                "class_type": STAGE_BARRIER_NODE_ID,
                "inputs": {
                    "stage": 0,
                    "value_0": ["source_true", 0],
                    "value_1": ["source_false", 0],
                },
            },
            "switch": {
                "class_type": "TuringUtilsLazyIfElse",
                "inputs": {
                    "condition": True,
                    "on_true": ["barrier", 0],
                    "on_false": ["barrier", 1],
                },
            },
            "output": {
                "class_type": "TestStageOutput",
                "inputs": {"value": ["switch", 0]},
            },
        }
        compiled, _ = compile_stage_barrier_prompt(prompt)
        original_mappings = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
        try:
            comfy_nodes.NODE_CLASS_MAPPINGS.update(
                {
                    "TestStageSource": Source,
                    "TestStageOutput": Output,
                    "TuringUtilsLazyIfElse": LazyIfElse,
                    STAGE_PATH_NODE_ID: StagePath,
                }
            )
            execution = ExecutionList(DynamicPrompt(compiled), EmptyCache())
            execution.add_node("output")

            true_route = compiled["switch"]["inputs"]["on_true"][0]
            false_route = compiled["switch"]["inputs"]["on_false"][0]
            self.assertNotIn(true_route, execution.pendingNodes)
            self.assertNotIn(false_route, execution.pendingNodes)
            self.assertNotIn("source_true", execution.pendingNodes)
            self.assertNotIn("source_false", execution.pendingNodes)

            execution.make_input_strong_link("switch", "on_true")

            self.assertIn(true_route, execution.pendingNodes)
            self.assertIn("source_true", execution.pendingNodes)
            self.assertNotIn(false_route, execution.pendingNodes)
            self.assertNotIn("source_false", execution.pendingNodes)
        finally:
            comfy_nodes.NODE_CLASS_MAPPINGS.clear()
            comfy_nodes.NODE_CLASS_MAPPINGS.update(original_mappings)


if __name__ == "__main__":
    unittest.main()
