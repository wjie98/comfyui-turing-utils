import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

import execution
import nodes
from comfy_api.latest import io
from comfyui_turing_utils.workspace.cards import describe, flatten
from comfyui_turing_utils.workspace.endpoints import CanvasInputs, parse_ports
from comfyui_turing_utils.workspace.parameters import (
    DYNAMIC_COMBO,
    DYNAMIC_BINDINGS,
    expand_dynamic,
    parameter_default,
    validate_parameter,
)
from comfyui_turing_utils.workspace import projection, workflow_files
from comfyui_turing_utils.workspace.store import Project
from comfyui_turing_utils.workspace.templates import material_template


class DynamicText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CanvasDynamicTextTest",
            inputs=[
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(
                            "plain", [io.String.Input("text", default="plain")]
                        ),
                        io.DynamicCombo.Option(
                            "number",
                            [
                                io.Int.Input("count", default=3),
                                io.DynamicCombo.Input(
                                    "format",
                                    options=[
                                        io.DynamicCombo.Option(
                                            "decimal",
                                            [io.Boolean.Input("positive", default=True)],
                                        ),
                                        io.DynamicCombo.Option(
                                            "hex", [io.String.Input("prefix", default="0x")]
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option("empty", []),
                    ],
                ),
            ],
            outputs=[io.String.Output()],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mode):
        if mode["mode"] == "plain":
            result = mode["text"]
        elif mode["mode"] == "number":
            count = mode["count"]
            fmt = mode["format"]
            if fmt["format"] == "hex":
                result = fmt["prefix"] + format(count, "x")
            else:
                result = str(count if fmt["positive"] else -count)
        else:
            result = "empty"
        return io.NodeOutput(result)


class DynamicParametersTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.folders = mock.patch(
            "folder_paths.get_output_directory", return_value=str(self.output)
        )
        self.folders.start()
        self.addCleanup(self.folders.stop)
        (self.output / "dynamic").mkdir()
        self.project = Project.create("dynamic")
        spec = DynamicText.INPUT_TYPES()["required"]["mode"]
        self.port = dict(
            id="mode", slot=1, name="Mode", type=DYNAMIC_COMBO,
            kind="parameter", options=spec[1],
        )
        self.port["default"] = parameter_default(self.port)
        self.workflow, self.prompt = material_template(("text",))
        self.prompt["1"]["inputs"]["ports"] = json.dumps(
            [*json.loads(self.prompt["1"]["inputs"]["ports"]), self.port]
        )
        self.prompt["compute"] = {
            "class_type": "CanvasDynamicTextTest",
            "inputs": expand_dynamic(self.port, self.port["default"], "mode"),
            "_meta": {DYNAMIC_BINDINGS: {"mode": ["1", 1]}},
        }
        self.prompt["3"]["inputs"]["text"] = ["compute", 0]

    def test_nested_defaults_and_native_input_expansion(self):
        ports = parse_ports(json.dumps([dict(self.port, slot=0)]))
        state = ports[0]["default"]
        self.assertEqual(CanvasInputs().forward(json.dumps(ports)), (state,))
        self.assertNotEqual(
            CanvasInputs.VALIDATE_INPUTS(json.dumps(ports), {"port_0": DYNAMIC_COMBO}),
            True,
        )
        state["selected"] = "number"
        nested = state["branches"]["number"]["format"]
        nested["selected"] = "hex"
        nested["branches"]["hex"]["prefix"] = "HEX:"
        flat = expand_dynamic(self.port, state, "mode")
        self.assertEqual(flat, {
            "mode": "number", "mode.count": 3,
            "mode.format": "hex", "mode.format.prefix": "HEX:",
        })
        state["selected"] = "empty"
        self.assertEqual(expand_dynamic(self.port, state, "mode"), {"mode": "empty"})

    def test_definition_and_all_branches_are_validated(self):
        for spec in (
            ["MODEL", {}], ["FLOAT", {"forceInput": True}],
            ["COMFY_AUTOGROW_V3", {}],
        ):
            port = copy.deepcopy(self.port)
            port["options"]["options"][2]["inputs"]["required"]["bad"] = spec
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "editable"):
                parse_ports(json.dumps([dict(port, slot=0)]))
        for mutate in (
            lambda v: v.update(selected="unknown"),
            lambda v: v["branches"].pop("empty"),
            lambda v: v["branches"]["number"].update(count=True),
            lambda v: v["branches"]["number"]["format"]["branches"]["hex"].update(prefix=3),
            lambda v: v["branches"]["empty"].update(extra=0),
        ):
            state = copy.deepcopy(self.port["default"])
            mutate(state)
            with self.assertRaises(ValueError):
                validate_parameter(self.port, state)
        port = copy.deepcopy(self.port)
        port["options"]["options"][2]["inputs"]["required"]["file"] = [
            "COMBO", {"options": []},
        ]
        port.pop("default")
        self.assertIsNone(parameter_default(port)["branches"]["empty"]["file"])
        parse_ports(json.dumps([dict(port, slot=0)]))

    def test_canvas_persistence_isolation_and_branch_recompilation(self):
        packed = workflow_files.pack(self.workflow, self.prompt)
        a = workflow_files.add_instance(self.project.root, packed, "A")
        b = workflow_files.add_instance(self.project.root, packed, "B")
        state = copy.deepcopy(self.port["default"])
        state["selected"] = "number"
        state["branches"]["number"]["count"] = 31
        state["branches"]["number"]["format"]["selected"] = "hex"
        canvas = workflow_files.load_json(self.project.root / "canvas.json")
        workflow_files.patch(
            self.project.root, canvas["revision"],
            [dict(type="input", id=a + ":parameters", name="mode", value=state)],
        )
        display = projection.document(self.project.root)
        self.assertEqual(display["cards"][0]["fields"][0]["type"], DYNAMIC_COMBO)
        self.assertEqual(display["cards"][0]["ports"], [])
        result = projection.document(self.project.root, execution=True)
        self.assertEqual(result["prompt"][a + ":node:compute"]["inputs"], {
            "mode": "number", "mode.count": 31, "mode.format": "hex", "mode.format.prefix": "0x",
        })
        self.assertEqual(
            result["prompt"][b + ":node:compute"]["inputs"],
            {"mode": "plain", "mode.text": "plain"},
        )
        self.assertNotIn(DYNAMIC_BINDINGS, result["prompt"][a + ":node:compute"]["_meta"])
        opened = workflow_files.read_instance(self.project.root, a)
        workflow_files.save_instance(
            self.project.root, a, self.workflow, self.prompt, opened["revision"],
            parameters={"mode": state},
        )
        self.assertEqual(
            workflow_files.read_instance(self.project.root, a)["parameters"]["mode"], state
        )
        self.assertEqual(packed["extra"]["turing_card"]["prompt"], self.prompt)

    def test_invalid_patch_is_transactional(self):
        identity = workflow_files.add_instance(
            self.project.root, workflow_files.pack(self.workflow, self.prompt), "A"
        )
        path = self.project.root / "canvas.json"
        previous = path.read_bytes()
        invalid = copy.deepcopy(self.port["default"])
        invalid["branches"]["number"]["count"] = "bad"
        with self.assertRaises(ValueError):
            workflow_files.patch(
                self.project.root, json.loads(previous)["revision"],
                [dict(type="input", id=identity + ":parameters", name="mode", value=invalid)],
            )
        self.assertEqual(path.read_bytes(), previous)

    def test_invalid_binding_and_branch_links_fail(self):
        invalid = copy.deepcopy(self.prompt)
        invalid["compute"]["_meta"][DYNAMIC_BINDINGS]["mode"] = ["unknown", 1]
        with self.assertRaisesRegex(ValueError, "binding"):
            describe(invalid)
        invalid = copy.deepcopy(self.prompt)
        invalid["compute"]["inputs"]["mode.text"] = ["2", 0]
        with self.assertRaisesRegex(ValueError, "Disconnect branch"):
            flatten(invalid, describe(invalid))
        direct = copy.deepcopy(self.prompt)
        direct["compute"].pop("_meta")
        direct["compute"]["inputs"]["mode"] = ["1", 1]
        self.assertEqual(
            flatten(direct, describe(direct))["compute"]["inputs"],
            {"mode": "plain", "mode.text": "plain"},
        )

    def test_native_v3_validation_and_execution(self):
        class Server:
            client_id = None
            last_node_id = None

            def send_sync(self, *args):
                pass

        mapping = {"CanvasDynamicTextTest": DynamicText}
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping):
            for selected, expected in (
                ("plain", "plain"), ("number", "0x1f"), ("empty", "empty"),
            ):
                state = copy.deepcopy(self.port["default"])
                state["selected"] = selected
                state["branches"]["number"]["count"] = 31
                state["branches"]["number"]["format"]["selected"] = "hex"
                compiled = flatten(self.prompt, describe(self.prompt), {"mode": state})
                graph = {"compute": compiled["compute"]}
                valid = asyncio.run(
                    execution.validate_prompt("dynamic", graph, ["compute"])
                )
                self.assertTrue(valid[0], valid)
                executor = execution.PromptExecutor(
                    Server(), cache_args={"ram": 0, "ram_inactive": 0}
                )
                executor.execute(graph, "dynamic", {}, ["compute"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(
                    executor.caches.outputs.get_local("compute").outputs, [[expected]]
                )
