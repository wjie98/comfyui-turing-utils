"""Presentation contracts and real ComfyUI dynamic-input execution."""

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

from comfy_api.latest import _io as io
from comfyui_turing_utils.nodes.attention import AttentionStrategy, _ATTENTION_STRATEGIES
from comfyui_turing_utils.nodes.multimodal_chat import MultimodalPromptChat


def finalized(node, values):
    inputs, _, data = io.get_finalized_class_inputs(node.INPUT_TYPES(), values)
    active = {**inputs["required"], **inputs["optional"]}
    return active, io.build_nested_inputs({key: value for key, value in values.items() if key in active}, data)


class NodeConfigurationTest(unittest.TestCase):
    def test_only_chat_options_are_advanced_in_public_ordinary_nodes(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS
        from comfyui_turing_utils.nodes.multimodal_chat import _chat_option_inputs
        expected = {item.id for item in _chat_option_inputs()}
        found = set()
        def walk(inputs):
            for group in ("required", "optional"):
                for key, spec in inputs.get(group, {}).items():
                    metadata = spec[1] if len(spec) > 1 else {}
                    yield key, metadata
                    if spec[0] == "COMFY_DYNAMICCOMBO_V3":
                        for option in metadata["options"]:
                            yield from walk(option["inputs"])
        for name, node in NODE_CLASS_MAPPINGS.items():
            if not name.startswith("TuringUtils") or name == "TuringUtilsStagePath":
                continue
            for key, metadata in walk(node.INPUT_TYPES()):
                if name == "TuringUtilsMultimodalPromptChat":
                    self.assertEqual(bool(metadata.get("advanced")), key in expected, key)
                    if metadata.get("advanced"):
                        found.add(key)
                else:
                    self.assertFalse(metadata.get("advanced"), (name, key))
        self.assertEqual(found, expected)

    def test_public_categories_are_flat_and_internal_nodes_hidden(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        from comfyui_turing_utils.nodes import INTERNAL_NODE_NOTE
        categories = set()
        public = internal = 0
        for name, node in NODE_CLASS_MAPPINGS.items():
            schema = node.GET_SCHEMA() if hasattr(node, "GET_SCHEMA") else None
            category = schema.category if schema else node.CATEGORY
            if (schema and schema.is_dev_only) or getattr(node, "DEV_ONLY", False):
                internal += 1
                self.assertEqual(category, "", name)
                title = schema.display_name if schema else NODE_DISPLAY_NAME_MAPPINGS[name]
                self.assertTrue(title.endswith("(Internal)"), (name, title))
                description = schema.description if schema else node.DESCRIPTION
                self.assertTrue(description.startswith(INTERNAL_NODE_NOTE), name)
            else:
                public += 1
                self.assertEqual(len(category.split("/")), 2, name)
                self.assertTrue(category.startswith("Turing Utils/"), name)
                categories.add(category.split("/")[1])
        self.assertEqual((public, internal), (44, 13))
        self.assertEqual(categories, {"Models", "Prompt", "Video", "Mask", "MiniMax H3", "Bernini", "Krea2", "Workflow", "Materials"})
        self.assertEqual(AttentionStrategy.GET_SCHEMA().category, "Turing Utils/Models")
        for name in ("TuringUtilsSeCModelLoader", "TuringUtilsSeCTrackVisualConceptApply",
                     "TuringUtilsMiniMaxH3LatentUpscaleModelLoader", "TuringUtilsMiniMaxH3LatentUpscaleApply"):
            self.assertNotIn(name, NODE_CLASS_MAPPINGS)

    def test_attention_schema_is_valid_and_has_all_existing_strategies(self):
        schema = AttentionStrategy.GET_SCHEMA()
        strategies = schema.inputs[1].options
        self.assertEqual([option.key for option in strategies], list(_ATTENTION_STRATEGIES))
        for option in strategies:
            with self.subTest(strategy=option.key):
                fields = {item.id: item for item in option.inputs}
                if "dense_prefix_steps" in fields:
                    self.assertFalse(fields["dense_prefix_steps"].advanced)
                if "prefix_policy" in fields:
                    self.assertFalse(fields["prefix_policy"].as_dict()["advanced"])
        veda = next(option for option in strategies if option.key == "veda")
        fields = {item.id: item for item in veda.inputs}
        self.assertFalse(fields["keep_ratio"].advanced)
        self.assertFalse(fields["predictor_precision"].advanced)

    def test_prefix_parameters_only_exist_in_their_active_mode(self):
        for strategy in ("sol", "sla"):
            for policy in ("auto", "none", "manual"):
                values = {"strategy": strategy, "strategy.prefix_policy": policy}
                fields, _ = finalized(AttentionStrategy, values)
                self.assertEqual("strategy.prefix_policy.manual_prefix_tokens" in fields, policy == "manual")
                self.assertEqual("strategy.prefix_policy.sparse_reference_image" in fields, policy == "auto")
                self.assertNotIn("strategy.predictor_name", fields)

    def test_dynamic_settings_delegate_to_existing_implementations(self):
        cases = [
            ({"strategy": "sol", "strategy.routing_threshold": 0.5,
              "strategy.prefix_policy": "manual", "strategy.prefix_policy.manual_prefix_tokens": 128},
             {"routing_threshold": 0.5, "prefix_policy": "manual", "manual_prefix_tokens": 128}),
            ({"strategy": "sla", "strategy.sparsity_ratio": 0.7, "strategy.prefix_policy": "auto",
              "strategy.prefix_policy.sparse_reference_image": True},
             {"sparsity_ratio": 0.7, "prefix_policy": "auto", "sparse_reference_image": True}),
            ({"strategy": "veda", "strategy.predictor_name": "test.safetensors", "strategy.keep_ratio": 0.2},
             {"predictor_name": "test.safetensors", "keep_ratio": 0.2}),
        ]
        for values, expected in cases:
            _, nested = finalized(AttentionStrategy, values)
            model, patched = object(), object()
            apply = mock.Mock(return_value=patched)
            with self.subTest(strategy=values["strategy"]), mock.patch.dict(_ATTENTION_STRATEGIES, {values["strategy"]: apply}):
                self.assertIs(AttentionStrategy.execute(model, **nested).result[0], patched)
                apply.assert_called_once_with(model, **expected)

    def test_png_does_not_have_a_jpeg_quality_input(self):
        fields, nested = finalized(MultimodalPromptChat, {"image_format": "png", "image_format.jpeg_quality": 40})
        self.assertNotIn("image_format.jpeg_quality", fields)
        self.assertEqual(nested["image_format"], {"image_format": "png"})
        fields, nested = finalized(MultimodalPromptChat, {"image_format": "jpeg", "image_format.jpeg_quality": 85})
        self.assertIn("image_format.jpeg_quality", fields)
        self.assertEqual(nested["image_format"], {"image_format": "jpeg", "jpeg_quality": 85})

    def test_existing_node_ids_are_still_registered(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS
        self.assertIs(NODE_CLASS_MAPPINGS["TuringUtilsAttentionStrategy"], AttentionStrategy)
        for name in ["TuringUtilsH3VedaAttentionStrategy","TuringUtilsSolAttentionStrategy","TuringUtilsSlaAttentionStrategy","TuringUtilsMiniMaxH3BlockCachePatch","TuringUtilsWanVideoFramesPadding","TuringUtilsMiniMaxH3VideoFramesPadding","TuringUtilsH3ConcatAVLatent","TuringUtilsH3SeparateAVLatent","TuringUtilsMultimodalChatOptions"]:
            self.assertNotIn(name, NODE_CLASS_MAPPINGS)

    def test_removed_static_image_strategies_have_no_node_aliases(self):
        from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        from comfyui_turing_utils.nodes import attention as attention_nodes
        for name, cls in (("TuringUtilsH3ImageSolAttention", "H3ImageSolAttentionPatch"),
                          ("TuringUtilsH3StaticVirtualKV", "H3StaticVirtualKV")):
            self.assertNotIn(name, NODE_CLASS_MAPPINGS)
            self.assertNotIn(name, NODE_DISPLAY_NAME_MAPPINGS)
            self.assertFalse(hasattr(attention_nodes, cls))

    def test_real_executor_switches_strategy_without_reloading_model(self):
        import execution
        import nodes

        loaded = mock.Mock(return_value=object())

        class ModelSource:
            RETURN_TYPES = ("MODEL",)
            FUNCTION = "load"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            def load(self):
                return (loaded(),)

        class Server:
            client_id = None
            last_node_id = None

            def send_sync(self, *args, **kwargs):
                pass

        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {
            "TestConfigurationModel": ModelSource,
            "TuringUtilsAttentionStrategy": AttentionStrategy,
        }), mock.patch.dict(_ATTENTION_STRATEGIES, {"sol": mock.Mock(return_value=object()), "sla": mock.Mock(return_value=object())}):
            sol, sla = _ATTENTION_STRATEGIES["sol"], _ATTENTION_STRATEGIES["sla"]
            executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC,
                                                cache_args={"ram": 0, "ram_inactive": 0})
            for index, name in enumerate(("sol", "sol", "sla")):
                values = {"model": ["loader", 0], "strategy": name, "strategy.prefix_policy": "manual"}
                fields, _ = finalized(AttentionStrategy, values)
                for key, spec in fields.items():
                    if key not in values and "default" in spec[1]:
                        values[key] = spec[1]["default"]
                values["strategy.prefix_policy.manual_prefix_tokens"] = 128
                prompt = {
                    "loader": {"class_type": "TestConfigurationModel", "inputs": {}},
                    "attention": {"class_type": "TuringUtilsAttentionStrategy", "inputs": values},
                }
                executor.execute(prompt, str(index), execute_outputs=["attention"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(loaded.call_count, 1)
            self.assertEqual(sol.call_count, 1)
            self.assertEqual(sla.call_count, 1)
            self.assertEqual(sla.call_args.kwargs["manual_prefix_tokens"], 128)

    def test_real_executor_reuses_chat_cache_and_tracks_advanced_inputs(self):
        import execution
        import nodes

        class Server:
            client_id = None
            last_node_id = None

            def send_sync(self, *args, **kwargs):
                pass

        inputs = {"prompt": "user", "system_prompt": "system", "base_url": "http://localhost:9200",
                  "model": "test", "api_key": "", "cache_buster": 0,
                  "image_format": "png", "temperature": 0.25}
        prompt = {"chat": {"class_type": "TuringUtilsMultimodalPromptChat", "inputs": inputs}}
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {"TuringUtilsMultimodalPromptChat": MultimodalPromptChat}), mock.patch(
            "comfyui_turing_utils.nodes.multimodal_chat.request_chat_completion",
            return_value={"choices": [{"message": {"content": "response"}}]},
        ) as request:
            executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC,
                                                cache_args={"ram": 0, "ram_inactive": 0})
            for index, temperature in enumerate((0.25, 0.25, 0.75)):
                inputs["temperature"] = temperature
                executor.execute(prompt, str(index), execute_outputs=["chat"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(request.call_count, 1 if index < 2 else 2)
            self.assertEqual(request.call_args.args[2]["temperature"], 0.75)
