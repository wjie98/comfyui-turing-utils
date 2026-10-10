import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

from comfyui_turing_utils.runtime.shared_loaders import (
    APPLICATIONS,
    compile_shared_loaders,
)


class SharedLoaderTest(unittest.TestCase):
    def prompt(self, application, options=None):
        return {
            "a": {
                "class_type": application,
                "inputs": {"model_name": "test.safetensors", **(options or {})},
            },
            "b": {
                "class_type": application,
                "inputs": {"model_name": "test.safetensors", **(options or {})},
            },
        }

    def test_shared_stable_loader_and_independent_applications(self):
        for app, (loader, worker, socket, defaults) in APPLICATIONS.items():
            prompt = self.prompt(app)
            prompt["a"]["inputs"]["frames"] = ["frames_a", 0]
            prompt["b"]["inputs"]["frames"] = ["frames_b", 0]
            original = copy.deepcopy(prompt)
            compiled = compile_shared_loaders(prompt)
            self.assertEqual(prompt, original)
            self.assertEqual(
                compiled["a"]["inputs"][socket], compiled["b"]["inputs"][socket]
            )
            loader_id = compiled["a"]["inputs"][socket][0]
            self.assertEqual(
                compiled[loader_id]["inputs"],
                {"model_name": "test.safetensors", **defaults},
            )
            self.assertEqual(compiled["a"]["class_type"], worker)
            self.assertEqual(compile_shared_loaders(compiled), compiled)
            prompt["a"]["inputs"]["frames"] = ["another", 0]
            self.assertEqual(
                compile_shared_loaders(prompt)[loader_id], compiled[loader_id]
            )
            parameter = next(iter(defaults))
            prompt["b"]["inputs"][parameter] = "different"
            different = compile_shared_loaders(prompt)
            self.assertNotEqual(
                different["a"]["inputs"][socket], different["b"]["inputs"][socket]
            )

    def test_removed_model_only_connections_do_not_bypass_application_validation(self):
        for app, (_, worker, socket, _) in APPLICATIONS.items():
            prompt = {"a": {"class_type": app, "inputs": {socket: ["old", 0]}}}
            self.assertEqual(compile_shared_loaders(prompt), prompt)

    def test_core_executor_shares_one_load_even_without_cache(self):
        import nodes
        import execution
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS

        class Server:
            client_id = None
            last_node_id = None

            def send_sync(self, *args, **kwargs):
                pass

        for app, (loader, worker, socket, defaults) in APPLICATIONS.items():
            for cache_type in (execution.CacheType.CLASSIC, execution.CacheType.NONE):
                with self.subTest(app=app, cache=cache_type):
                    sentinel = object()
                    received = []

                    def apply(model, *args, **kwargs):
                        received.append(model)
                        return (1, 1, 1, 1) if socket == "upscale_model" else 1

                    prompt = self.prompt(app)
                    if socket == "upscale_model":
                        prompt["a"]["inputs"].update(latent={}, scale=2.0)
                        prompt["b"]["inputs"].update(latent={}, scale=3.0)
                    else:
                        prompt["a"]["inputs"].update(frames=1, annotation_frame_idx=0)
                        prompt["b"]["inputs"].update(frames=2, annotation_frame_idx=1)
                    module = "comfyui_turing_utils.nodes."
                    if socket == "upscale_model":
                        load_name = module + "minimax.load_h3_latent_upscaler"
                        apply_name = module + "minimax.upscale_h3_latent"
                    else:
                        load_name = module + "sec.load_sec_model"
                        apply_name = module + "sec.track_visual_concept"
                    with (
                        mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, NODE_CLASS_MAPPINGS),
                        mock.patch(load_name, return_value=sentinel) as load,
                        mock.patch(apply_name, side_effect=apply),
                    ):
                        executor = execution.PromptExecutor(
                            Server(),
                            cache_type=cache_type,
                            cache_args={"ram": 0, "ram_inactive": 0},
                        )
                        executor.execute(
                            compile_shared_loaders(prompt),
                            "first",
                            execute_outputs=["a", "b"],
                        )
                        self.assertTrue(executor.success, executor.status_messages)
                        self.assertEqual(load.call_count, 1)
                        self.assertEqual(received, [sentinel, sentinel])
                        if cache_type == execution.CacheType.CLASSIC:
                            key = (
                                "scale"
                                if socket == "upscale_model"
                                else "annotation_frame_idx"
                            )
                            prompt["a"]["inputs"][key] = 4
                            executor.execute(
                                compile_shared_loaders(prompt),
                                "second",
                                execute_outputs=["a", "b"],
                            )
                            self.assertTrue(executor.success, executor.status_messages)
                            self.assertEqual(load.call_count, 1)
                            self.assertEqual(len(received), 3)
                            parameter = next(iter(defaults))
                            prompt["a"]["inputs"][parameter] = (
                                "fp32" if parameter == "precision" else "sdpa"
                            )
                            executor.execute(
                                compile_shared_loaders(prompt),
                                "third",
                                execute_outputs=["a", "b"],
                            )
                            self.assertTrue(executor.success, executor.status_messages)
                            self.assertEqual(load.call_count, 2)

    def test_prompt_handler_preserves_workflow_and_installs_once(self):
        from types import SimpleNamespace
        from server import PromptServer
        from comfyui_turing_utils.runtime.shared_loaders import (
            compile_shared_loaders_on_prompt,
            install_shared_loader_compiler,
        )

        server = SimpleNamespace(add_on_prompt_handler=mock.Mock())
        with mock.patch.object(PromptServer, "instance", server, create=True):
            self.assertTrue(install_shared_loader_compiler())
            self.assertTrue(install_shared_loader_compiler())
        server.add_on_prompt_handler.assert_called_once_with(
            compile_shared_loaders_on_prompt
        )
        data = {
            "prompt": self.prompt(next(iter(APPLICATIONS))),
            "extra_data": {"extra_pnginfo": {"workflow": {"nodes": []}}},
        }
        original = copy.deepcopy(data)
        result = compile_shared_loaders_on_prompt(data)
        self.assertEqual(data, original)
        self.assertEqual(result["extra_data"], original["extra_data"])
        self.assertEqual(len(result["prompt"]), 3)
