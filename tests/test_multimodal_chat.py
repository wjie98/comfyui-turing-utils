from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.multimodal_chat import (  # noqa: E402
    ChatOptions,
    DEFAULT_CHAT_OPTIONS,
    MultimodalChatOptions,
    MultimodalPromptChat,
    build_chat_request,
    build_user_content,
    extract_chat_text,
    normalize_chat_endpoint,
    resolve_api_key,
)


class MultimodalPromptChatTest(unittest.TestCase):
    def test_schema_has_dynamic_image_and_video_inputs(self):
        schema = MultimodalPromptChat.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsMultimodalPromptChat")
        self.assertEqual(schema.outputs[0].id, "enhanced_prompt")
        self.assertEqual(schema.outputs[1].id, "metadata_json")
        self.assertEqual(
            [item.id for item in schema.inputs[:5]],
            ["prompt", "system_prompt", "base_url", "model", "api_key"],
        )
        inputs = {item.id: item for item in schema.inputs}
        self.assertEqual(schema.inputs[5].id, "cache_buster")
        self.assertEqual(inputs["cache_buster"].default, 0)
        self.assertTrue(inputs["cache_buster"].control_after_generate)
        self.assertEqual(inputs["first_frame"].io_type, "IMAGE")
        self.assertEqual(inputs["last_frame"].io_type, "IMAGE")
        self.assertTrue(inputs["first_frame"].optional)
        self.assertTrue(inputs["last_frame"].optional)
        self.assertEqual(inputs["images"].template.input.io_type, "IMAGE")
        self.assertEqual(inputs["videos"].template.input.io_type, "IMAGE")
        self.assertTrue(inputs["images"].optional)
        self.assertTrue(inputs["videos"].optional)
        self.assertTrue(inputs["options"].optional)
        self.assertEqual(inputs["options"].io_type, "TURING_UTILS_CHAT_OPTIONS")
        self.assertNotIn("disable_thinking", inputs)

    def test_options_node_owns_advanced_parameters_and_defaults_to_8k(self):
        schema = MultimodalChatOptions.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsMultimodalChatOptions")
        self.assertEqual(schema.outputs[0].io_type, "TURING_UTILS_CHAT_OPTIONS")
        inputs = {item.id: item for item in schema.inputs}
        self.assertTrue(inputs["disable_thinking"].default)
        self.assertEqual(inputs["max_output_tokens"].default, 8192)
        self.assertNotIn("cache_buster", inputs)
        self.assertEqual(MultimodalChatOptions.execute().result[0], DEFAULT_CHAT_OPTIONS)

    def test_unconnected_and_connected_default_options_build_identical_requests(self):
        connected = MultimodalChatOptions.execute().result[0]
        implicit = build_chat_request("model", "system", "user", DEFAULT_CHAT_OPTIONS)
        explicit = build_chat_request("model", "system", "user", connected)
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit["max_tokens"], 8192)

    def test_endpoint_accepts_root_version_and_complete_urls(self):
        expected = "http://127.0.0.1:9200/v1/chat/completions"
        self.assertEqual(normalize_chat_endpoint("127.0.0.1:9200"), expected)
        self.assertEqual(normalize_chat_endpoint("http://127.0.0.1:9200/"), expected)
        self.assertEqual(normalize_chat_endpoint("http://127.0.0.1:9200/v1"), expected)
        self.assertEqual(normalize_chat_endpoint(expected), expected)
        self.assertEqual(
            normalize_chat_endpoint("https://host.example/api/v2"),
            "https://host.example/api/v2/chat/completions",
        )
        self.assertEqual(
            normalize_chat_endpoint("https://host.example/openai"),
            "https://host.example/openai/v1/chat/completions",
        )

    def test_endpoint_rejects_credentials_query_and_unsupported_scheme(self):
        for value in (
            "ftp://host.example",
            "https://user:pass@host.example",
            "https://host.example?key=value",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_chat_endpoint(value)

    def test_api_key_supports_literal_empty_and_environment_references(self):
        self.assertEqual(resolve_api_key(""), "not-needed")
        self.assertEqual(resolve_api_key("literal-key"), "literal-key")
        with patch.dict(os.environ, {"CHAT_TEST_KEY": "secret"}, clear=False):
            self.assertEqual(resolve_api_key("$CHAT_TEST_KEY"), "secret")
            self.assertEqual(resolve_api_key("${CHAT_TEST_KEY}"), "secret")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "was not found"):
                resolve_api_key("$CHAT_TEST_KEY")

    def test_thinking_is_disabled_through_chat_template_kwargs(self):
        request = build_chat_request(
            "model",
            "system",
            "user",
            ChatOptions(
                max_output_tokens=512,
                extra_body_json='{"top_p":0.9,"chat_template_kwargs":{"custom":1}}',
            ),
        )
        self.assertFalse(request["stream"])
        self.assertNotIn("temperature", request)
        self.assertEqual(
            request["chat_template_kwargs"],
            {"custom": 1, "enable_thinking": False},
        )

    def test_extra_body_cannot_replace_core_request_fields(self):
        with self.assertRaisesRegex(ValueError, "must not override"):
            build_chat_request(
                "model",
                "",
                "user",
                ChatOptions(extra_body_json='{"messages":[]}'),
            )

    def test_images_are_naturally_ordered_and_labeled_for_h3_prompts(self):
        red = torch.zeros((1, 8, 8, 3))
        red[..., 0] = 1.0
        blue = torch.zeros((1, 8, 8, 3))
        blue[..., 2] = 1.0
        content, metadata = build_user_content(
            "edit",
            None,
            None,
            {"image_10": blue, "image_2": red},
            None,
            ChatOptions(max_image_edge=256, video_max_edge=256),
        )
        labels = [item["text"] for item in content if item["type"] == "text"]
        self.assertEqual(labels[1:3], ["<Picture 1>", "<Picture 2>"])
        self.assertEqual(metadata["pictures"], 2)
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_image_batches_are_rejected_instead_of_changing_reference_numbers(self):
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            build_user_content(
                "edit",
                None,
                None,
                {"image_0": torch.zeros((2, 8, 8, 3))},
                None,
                ChatOptions(max_image_edge=256, video_max_edge=256),
            )

    def test_first_and_last_frames_have_explicit_labels_before_pictures(self):
        frame = torch.zeros((1, 8, 8, 3))
        content, metadata = build_user_content(
            "edit",
            frame,
            frame,
            {"image_0": frame},
            None,
            ChatOptions(max_image_edge=256, video_max_edge=256),
        )
        labels = [item["text"] for item in content if item["type"] == "text"]
        self.assertEqual(
            labels[1:4],
            ["<First Frame>", "<Last Frame>", "<Picture 1>"],
        )
        self.assertTrue(metadata["first_frame"])
        self.assertTrue(metadata["last_frame"])
        self.assertEqual(metadata["pictures"], 1)

    def test_image_sequence_video_is_sampled_as_24_fps_and_labeled(self):
        frames = torch.zeros((49, 8, 8, 3))
        frames[..., 1] = 1.0
        content, metadata = build_user_content(
            "describe motion",
            None,
            None,
            None,
            {"video_0": frames},
            ChatOptions(max_image_edge=256, video_max_edge=256),
        )
        labels = [item["text"] for item in content if item["type"] == "text"]
        self.assertTrue(labels[1].startswith("<Video 1>, frame at 0.000s"))
        self.assertEqual(metadata["videos"], 1)
        self.assertEqual(metadata["video_frames"], 5)
        self.assertEqual(metadata["video_timestamps"][0][0], 0.0)
        self.assertEqual(metadata["video_timestamps"][0][-1], 2.0)

    def test_response_extracts_text_without_returning_reasoning(self):
        response = {
            "choices": [
                {
                    "message": {"content": "  final prompt  ", "reasoning_content": "hidden"},
                    "finish_reason": "stop",
                }
            ]
        }
        text, choice = extract_chat_text(response)
        self.assertEqual(text, "final prompt")
        self.assertEqual(choice["finish_reason"], "stop")

    def test_response_accepts_structured_text_blocks(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "output_text", "text": "first"},
                            {"type": "image", "url": "ignored"},
                            {"type": "text", "text": " second"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(extract_chat_text(response)[0], "first second")


if __name__ == "__main__":
    unittest.main()
