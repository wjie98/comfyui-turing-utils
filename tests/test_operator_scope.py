from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.patcher_extension  # noqa: E402

from comfyui_turing_utils.quantization import dispatch  # noqa: E402
from comfyui_turing_utils.quantization.capabilities import BACKEND_NAME  # noqa: E402
from comfyui_turing_utils.quantization.operator_scope import (  # noqa: E402
    CLIP_ATTACHMENT_KEY,
    MODEL_WRAPPER_KEY,
    install_clip_operator_scope,
    install_model_operator_scope,
    make_model_operator_wrapper,
)


class _ModelPatcher:
    def __init__(self):
        self.wrappers = {}

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault((wrapper_type, key), []).append(wrapper)

    def get_wrappers(self, wrapper_type, key):
        return self.wrappers.get((wrapper_type, key), [])


class _ClipPatcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}
        self.attachments = {}

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def get_model_object(self, name):
        return self.object_patches.get(name, getattr(self.model, name))

    def get_attachment(self, name):
        return self.attachments.get(name)

    def set_attachments(self, name, value):
        self.attachments[name] = value


class OperatorScopeTest(unittest.TestCase):
    def test_model_scope_is_keyed_and_installed_once(self):
        model = _ModelPatcher()
        with mock.patch(
            "comfyui_turing_utils.quantization.operator_scope.kitchen_backend_available",
            return_value=True,
        ):
            self.assertTrue(install_model_operator_scope(model))
            self.assertFalse(install_model_operator_scope(model))

        wrappers = model.get_wrappers(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            MODEL_WRAPPER_KEY,
        )
        self.assertEqual(len(wrappers), 1)

    def test_model_scope_does_not_retry_an_execution_failure(self):
        entered = []

        @contextmanager
        def scope(_name):
            entered.append(_name)
            yield

        failure = RuntimeError("kernel failed")
        executor = mock.Mock(side_effect=failure)
        with (
            mock.patch("comfy_kitchen.use_backend", side_effect=scope),
            self.assertRaisesRegex(RuntimeError, "kernel failed"),
        ):
            make_model_operator_wrapper()(executor, "input")

        self.assertEqual(entered, [BACKEND_NAME])
        executor.assert_called_once_with("input")

    def test_clip_scope_uses_model_patcher_object_patch(self):
        calls = []

        def original(tokens):
            calls.append(tokens)
            return "encoded"

        model = SimpleNamespace(encode_token_weights=original)
        patcher = _ClipPatcher(model)
        clip = SimpleNamespace(cond_stage_model=model, patcher=patcher)

        @contextmanager
        def scope(name):
            calls.append(("backend", name))
            yield

        with (
            mock.patch(
                "comfyui_turing_utils.quantization.operator_scope.kitchen_backend_available",
                return_value=True,
            ),
            mock.patch("comfy_kitchen.use_backend", side_effect=scope),
        ):
            self.assertTrue(install_clip_operator_scope(clip))
            self.assertFalse(install_clip_operator_scope(clip))
            result = patcher.object_patches["encode_token_weights"]("tokens")

        self.assertEqual(result, "encoded")
        self.assertEqual(
            calls,
            [("backend", BACKEND_NAME), "tokens"],
        )
        self.assertTrue(patcher.get_attachment(CLIP_ATTACHMENT_KEY))

    def test_backend_registration_does_not_change_global_priority(self):
        fake_registry = mock.Mock()
        fake_registry.register = mock.Mock()
        cuda_status = {
            "cuda": {
                "available": True,
                "disabled": True,
                "capabilities": (
                    "convrot_w4a4_linear",
                    "int8_linear",
                    "w4a8_int8_linear",
                ),
            }
        }
        with (
            mock.patch("comfy_kitchen.list_backends", return_value=cuda_status),
            mock.patch("comfy_kitchen.registry.registry", fake_registry),
            mock.patch.object(dispatch, "_kernel_available", return_value=True),
        ):
            self.assertTrue(dispatch.register_backend())

        fake_registry.register.assert_called_once()
        fake_registry.set_priority.assert_not_called()


if __name__ == "__main__":
    unittest.main()
