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

import nodes
import execution
from comfy_api.latest import io
from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt
from execution import IsChangedCache, validate_prompt, _async_map_node_over_list
from comfyui_turing_utils.workspace.compiler import compile_segment
from comfyui_turing_utils.workspace.nodes import fresh_type, PUBLIC_NODES, INTERNAL_NODES
from comfyui_turing_utils.workspace.store import Project, Conflict, inside
from comfyui_turing_utils.workspace.cache import install_task_cleanup
from comfyui_turing_utils.workspace.endpoints import CanvasInputs, parse_ports
from comfyui_turing_utils.workspace.cards import describe, flatten
from comfyui_turing_utils.workspace import workflow_files


class AddText:
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}
    def run(self, text):
        return (text + "!",)


class AddTextV3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WorkspaceTestV3", inputs=[io.String.Input("text")], outputs=[io.String.Output()])
    @classmethod
    def execute(cls, text):
        return io.NodeOutput(text + "?")


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch("folder_paths.get_output_directory", return_value=self.tmp.name)
        patch.start(); self.addCleanup(patch.stop)
        self.project = Project("test")
        self.mapping = mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {
            **PUBLIC_NODES, **INTERNAL_NODES, "WorkspaceTest": AddText, "WorkspaceTestV3": AddTextV3})
        self.mapping.start(); self.addCleanup(self.mapping.stop)
        self.asset = self.project.text("hello")
        self.project.select("1", {"asset": self.asset})
        self.prompt = {
            "0": {"class_type": "WorkspaceTest", "inputs": {"text": "must not run"}},
            "1": {"class_type": "TuringMaterialText", "inputs": {"value": ["0", 0]}},
            "2": {"class_type": "WorkspaceTest", "inputs": {"text": ["1", 0]}},
            "3": {"class_type": "TuringMaterialText", "inputs": {"value": ["2", 0]}},
        }

    def compile(self):
        return compile_segment(self.prompt, "3", self.project.document()["selections"], "test", self.project.begin_run("3", 0), 0, fresh_type)

    def test_boundary_and_input_immutability(self):
        before = copy.deepcopy(self.prompt)
        compiled = self.compile()
        self.assertNotIn("0", compiled)
        self.assertEqual(compiled["1"]["class_type"], "_TuringMaterialReadText")
        self.assertEqual(self.prompt, before)
        self.assertEqual(compiled["3"]["class_type"], "_TuringMaterialWriteText")

    def test_feedback_reads_previous_material(self):
        self.project.select("3", {"asset": self.asset})
        self.prompt["2"]["inputs"]["text"] = ["3", 0]
        compiled = self.compile()
        source = compiled["2"]["inputs"]["text"][0]
        self.assertNotEqual(source, "3")
        self.assertEqual(compiled[source]["class_type"], "_TuringMaterialReadText")

    def test_connected_model_configuration_stays_cacheable(self):
        self.prompt["model"] = {"class_type": "UNETLoader", "inputs": {"unet_name": ["config", 0]}}
        self.prompt["config"] = {"class_type": "WorkspaceTest", "inputs": {"text": "weights"}}
        self.prompt["2"]["inputs"]["model"] = ["model", 0]
        compiled = self.compile()
        self.assertEqual(compiled["config"]["class_type"], "WorkspaceTest")
        self.assertEqual(compiled["model"]["class_type"], "UNETLoader")

    def test_invalid_trim_preserves_selection(self):
        with self.assertRaises(ValueError):
            self.project.select("1", {"asset": self.asset, "start": 3, "end": 2})
        self.assertEqual(self.project.document()["selections"]["1"]["revision"], 1)

    def test_real_validation(self):
        result = asyncio.run(validate_prompt("test", self.compile(), ["3"]))
        self.assertTrue(result[0], result)

    def test_v3_derived_node_executes_and_is_fresh(self):
        name = fresh_type("WorkspaceTestV3")
        cls = nodes.NODE_CLASS_MAPPINGS[name]
        result = asyncio.run(_async_map_node_over_list("test", "v3", cls, {"text": ["hello"]}, cls.FUNCTION))
        self.assertEqual(result[0].result[0], "hello?")
        self.assertIsNot(cls, AddTextV3)
        self.assertNotIn("fingerprint_inputs", AddTextV3.__dict__)
        from comfyui_turing_utils.nodes import INTERNAL_NODE_NOTE
        schema = cls.GET_SCHEMA()
        self.assertEqual(schema.node_id, name)
        self.assertTrue(schema.is_dev_only)
        self.assertTrue(schema.display_name.endswith("(Internal)"))
        self.assertTrue(schema.description.startswith(INTERNAL_NODE_NOTE))
        self.assertFalse(AddTextV3.GET_SCHEMA().is_dev_only)

    def test_fresh_signatures_differ(self):
        async def signatures():
            # No run-id inputs needed on the wrapped computation node itself.
            name = fresh_type("WorkspaceTest")
            prompt = {"2": {"class_type": name, "inputs": {"text": "same"}}}
            values = []
            for _ in range(2):
                dynamic = DynamicPrompt(copy.deepcopy(prompt))
                cache = IsChangedCache("test", dynamic, None)
                keys = CacheKeySetInputSignature(dynamic, ["2"], cache)
                await keys.add_keys(["2"])
                values.append(keys.get_data_key("2"))
            return values
        first, second = asyncio.run(signatures())
        self.assertNotEqual(first, second)

    def test_conflicting_result_stays_in_history(self):
        run = self.project.begin_run("3", 0)
        self.project.select("3", {"asset": self.asset})
        generated = self.project.text("generated")
        self.assertFalse(self.project.finish_run(run, generated))
        self.assertEqual(self.project.document()["selections"]["3"]["asset"], self.asset)
        self.assertFalse(self.project.finish_run(run, generated))
        self.assertEqual(len(self.project.history("text")), 2)

    def test_document_compare_and_swap(self):
        self.project.patch(0, [])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.project.patch(0, [])

    def test_missing_material_and_cycle(self):
        with self.assertRaisesRegex(ValueError, "no selected"):
            compile_segment(self.prompt, "3", {}, "test", "run", 0, fresh_type)
        self.prompt["2"]["inputs"]["text"] = ["2", 0]
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.compile()

    def test_paths(self):
        for path in ("../escape", "/absolute", "", "folder\\escape"):
            with self.assertRaises(ValueError):
                inside(self.tmp.name, path)

    def test_text_writer(self):
        run = self.project.begin_run("3", 0)
        result = INTERNAL_NODES["_TuringMaterialWriteText"]().write("test", "3", run, 0, "prompt", "new")
        self.assertTrue(result["ui"]["material"][0]["selected"])
        self.assertEqual(self.project.path(result["result"][0]).read_text(), "new")

    def test_real_executor_reuses_model_and_recomputes_processing(self):
        for cache_type in (execution.CacheType.CLASSIC, execution.CacheType.LRU, execution.CacheType.RAM_PRESSURE):
            with self.subTest(cache_type=cache_type):
                self._assert_executor_cache(cache_type)

    def _assert_executor_cache(self, cache_type):
        install_task_cleanup()
        history_before = len(self.project.history("text"))
        counts = {"load": 0, "compute": 0}
        class Loader(AddText):
            def run(self, text):
                counts["load"] += 1
                return (text,)
        class Compute(AddText):
            def run(self, text):
                if counts.get("fail"):
                    raise RuntimeError("Expected workspace failure")
                counts["compute"] += 1
                return (text + "!",)
        class Server:
            client_id = None
            last_node_id = None
            def send_sync(self, *args):
                pass
        self.prompt = {
            "load": {"class_type": "UNETLoader", "inputs": {"text": "model"}},
            "compute": {"class_type": "WorkspaceCompute", "inputs": {"text": ["load", 0]}},
            "3": {"class_type": "TuringMaterialText", "inputs": {"value": ["compute", 0]}},
        }
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {"UNETLoader": Loader, "WorkspaceCompute": Compute}):
            executor = execution.PromptExecutor(Server(), cache_type=cache_type,
                cache_args={"ram": 0, "ram_inactive": 0, "lru": 100})
            executor.execute({"other":{"class_type":"WorkspaceTest","inputs":{"text":"unrelated"}}},"ordinary",{},["other"])
            for index in range(2):
                revision = self.project.document()["selections"].get("3", {}).get("revision", 0)
                run = self.project.begin_run("3", revision)
                prompt = compile_segment(self.prompt, "3", {}, "test", run, revision, fresh_type)
                executor.execute(prompt, run, {}, ["3"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertIsNone(executor.caches.outputs.get_local("compute"))
                self.assertIsNone(executor.caches.objects.get_local("compute"))
                self.assertIsNotNone(executor.caches.outputs.get_local("load"))
            self.assertEqual(counts, {"load": 1, "compute": 2})
            self.assertEqual(len(self.project.history("text")), history_before + 2)
            if cache_type != execution.CacheType.CLASSIC:
                self.assertTrue(any(entry.outputs == [["unrelated!"]] for entry in executor.caches.outputs.cache.values()))
            counts["fail"] = True
            executor.execute(self.compile(),"failure",{},["3"])
            self.assertFalse(executor.success)
            self.assertIsNone(executor.caches.objects.get_local("compute"))
            self.assertIsNotNone(executor.caches.outputs.get_local("load"))

    def test_model_and_lora_cache_shared_between_card_ids(self):
        install_task_cleanup()
        class Server:
            client_id = None
            last_node_id = None
            def send_sync(self, *args):
                pass

        for cache_type in (execution.CacheType.CLASSIC, execution.CacheType.LRU, execution.CacheType.RAM_PRESSURE):
            with self.subTest(cache_type=cache_type):
                counts = {"load": 0, "lora": 0}
                class Loader(AddText):
                    def run(self, text):
                        counts["load"] += 1
                        return (text,)
                class Lora(AddText):
                    @classmethod
                    def INPUT_TYPES(cls):
                        return {"required": {"text": ("STRING",), "strength": ("FLOAT",)}}
                    def run(self, text, strength):
                        counts["lora"] += 1
                        return (text + str(strength),)

                with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {"UNETLoader": Loader, "LoraLoaderModelOnly": Lora}):
                    executor = execution.PromptExecutor(Server(), cache_type=cache_type,
                        cache_args={"ram": 0, "ram_inactive": 0, "lru": 100})
                    # Distinct native node IDs, same weights and LoRA parameters.
                    for card, strength in (("a", 1.0), ("b", 1.0), ("b", 0.5)):
                        target = card + ":result"
                        graph = {
                            card + ":load": {"class_type": "UNETLoader", "inputs": {"text": "model"}},
                            card + ":lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"text": [card + ":load", 0], "strength": strength}},
                            card + ":compute": {"class_type": "WorkspaceTest", "inputs": {"text": [card + ":lora", 0]}},
                            target: {"class_type": "TuringMaterialText", "inputs": {"value": [card + ":compute", 0]}},
                        }
                        revision = self.project.document()["selections"].get(target, {}).get("revision", 0)
                        run = self.project.begin_run(target, revision)
                        prompt = compile_segment(graph, target, {}, "test", run, revision, fresh_type)
                        executor.execute(prompt, run, {}, [target])
                        self.assertTrue(executor.success, executor.status_messages)
                        self.assertEqual(counts["load"], 1)
                        self.assertEqual(counts["lora"], 1 if strength == 1.0 else 2)
                        self.assertIsNotNone(executor.caches.outputs.get_local(card + ":lora"))
                        self.assertIsNone(executor.caches.outputs.get_local(card + ":compute"))

    def test_endpoint_bindings_and_flattening(self):
        ports = [{"id":"parameter","slot":0,"name":"Text","type":"STRING","kind":"value","default":"hello"},
                 {"id":"position","slot":1,"name":"Result","type":"TURING_CANVAS_POSITION","kind":"position"}]
        graph = {"in":{"class_type":"TuringCanvasInputs","inputs":{"ports":json.dumps(ports)}},
                 "compute":{"class_type":"WorkspaceTest","inputs":{"text":["in",0]}},
                 "stub":{"class_type":"TuringMaterialText","inputs":{"stub_id":"stable","position":["in",1],"value":["compute",0]}},
                 "out":{"class_type":"TuringCanvasOutputs","inputs":{"ports":json.dumps([{**ports[0],"id":"result"}]),"port_0":["stub",0]}}}
        interface=describe(graph)
        workflow={"nodes":[],"version":0.4}
        package=workflow_files.pack(workflow,graph)
        a=workflow_files.add_instance(self.project.root,package,"A")
        b=workflow_files.add_instance(self.project.root,package,"B")
        document=self.project.document()
        workflow_files.patch(self.project.root,document["revision"],[{"type":"input","id":a+":parameters","name":"parameter","value":"only A"}])
        document=self.project.document()
        self.assertEqual(document["prompt"][a+":node:compute"]["inputs"]["text"],"only A")
        self.assertEqual(document["prompt"][b+":node:compute"]["inputs"]["text"],"hello")
        self.assertEqual(package["extra"]["turing_card"]["overrides"],{})
        workflow_files.connect(self.project.root,document["revision"],b,"parameter",a+":stable")
        self.assertEqual(self.project.document()["prompt"][b+":node:compute"]["inputs"]["text"],[a+":stable",0])
        original_file = workflow_files.read_instance(self.project.root, b)
        changed = copy.deepcopy(graph)
        changed["in"]["inputs"]["ports"] = json.dumps([{**p,"id":"renamed"} if p["kind"] == "value" else p for p in ports])
        with self.assertRaisesRegex(ValueError, "Disconnect"):
            workflow_files.save_instance(self.project.root,b,{"nodes":[]},changed,original_file["revision"])
        self.assertEqual(workflow_files.read_instance(self.project.root,b), original_file)
        revision = self.project.document()["revision"]
        with self.assertRaisesRegex(ValueError, "type"):
            workflow_files.patch(self.project.root,revision,[{"type":"input","id":b+":parameters","name":"parameter","value":123}])
        with self.assertRaisesRegex(ValueError, "finite"):
            workflow_files.patch(self.project.root,revision,[{"type":"position","id":b,"x":float("nan"),"y":0}])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            workflow_files.patch(self.project.root,revision,[{"type":"input","id":b+":parameters","name":"parameter","value":"pending"},{"type":"invalid"}])
        self.assertEqual(workflow_files.read_instance(self.project.root,b), original_file)
        opened=workflow_files.read_instance(self.project.root,a)
        changed=copy.deepcopy(graph);changed["compute"]["inputs"]["text"]="edited"
        workflow_files.save_instance(self.project.root,a,workflow,changed,opened["revision"])
        with self.assertRaisesRegex(ValueError,"changed"):
            workflow_files.save_instance(self.project.root,a,workflow,changed,opened["revision"])
        external = workflow_files.read_instance(self.project.root,a)["workflow"]
        external["nodes"] = [{"id":123}]
        workflow_files.atomic_json(workflow_files.card_path(self.project.root,a),external)
        self.assertIn("externally",self.project.document()["cards"][0]["error"])
        opened = workflow_files.read_instance(self.project.root,a)
        workflow_files.save_instance(self.project.root,a,workflow,changed,opened["revision"])
        self.assertNotIn("error",self.project.document()["cards"][0])
        compiled=flatten(graph,interface,{"parameter":"override"})
        self.assertEqual(compiled["compute"]["inputs"]["text"],"override")
        self.assertNotIn("position",compiled["stub"]["inputs"])
        self.assertEqual(CanvasInputs().forward(json.dumps(ports)),("hello",None))
        graph["out"]["inputs"]["port_0"]=["compute",0]
        with self.assertRaisesRegex(ValueError,"only accepts material"):
            describe(graph)

    def test_endpoint_and_card_path_validation(self):
        with self.assertRaisesRegex(ValueError, "name"):
            parse_ports('[null]')
        with self.assertRaisesRegex(ValueError, "position type"):
            parse_ports(json.dumps([{"id":"a","slot":0,"name":"marker","kind":"position","type":"STRING"}]))
        identity = "a" * 32
        external = Path(self.tmp.name) / "outside-project"
        external.mkdir()
        parent = self.project.root / "cards"
        parent.mkdir()
        (parent / identity).symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside"):
            workflow_files.card_path(self.project.root, identity)

    def test_native_endpoint_execution(self):
        entries=[{"id":"a","slot":0,"name":"A","type":"STRING","kind":"value","default":"default"},
                 {"id":"b","slot":1,"name":"B","type":"STRING","kind":"value","default":"second"}]
        graph={"source":{"class_type":"WorkspaceTest","inputs":{"text":"source"}},
               "in":{"class_type":"TuringCanvasInputs","inputs":{"ports":json.dumps(entries),"port_0":["source",0]}},
               "out":{"class_type":"TuringCanvasOutputs","inputs":{"ports":json.dumps(entries),"port_0":["in",1],"port_1":["in",0]}}}
        self.assertTrue(asyncio.run(validate_prompt("endpoints",graph,["out"]))[0])
        class Server:
            client_id=None
            last_node_id=None
            def send_sync(self,*args):pass
        executor=execution.PromptExecutor(Server(),cache_args={"ram":0,"ram_inactive":0})
        executor.execute(graph,"endpoints",{},["out"])
        self.assertTrue(executor.success,executor.status_messages)
        self.assertEqual(executor.caches.outputs.get_local("out").outputs,[["second"],["source!"]])
