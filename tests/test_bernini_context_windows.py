from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.context_windows  # noqa: E402
import comfy.conds  # noqa: E402
from comfyui_turing_utils.adapters import bernini as bernini_nodes  # noqa: E402
from comfyui_turing_utils.nodes.bernini import (  # noqa: E402
    BerniniContextWindowsCore,
    BerniniInpaintCondition,
)


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.added_wrappers = []
        self.removed_wrappers = []

    def clone(self):
        clone = FakeModel()
        clone.model_options = dict(self.model_options)
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.added_wrappers.append((wrapper_type, key, wrapper))

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.removed_wrappers.append((wrapper_type, key))


class FakeVAE:
    def encode(self, image):
        return torch.zeros(
            1,
            16,
            ((image.shape[0] - 1) // 4) + 1,
            image.shape[1] // 8,
            image.shape[2] // 8,
        )


class BerniniContextWindowsTest(unittest.TestCase):
    def test_inpaint_uses_source_latent_and_conservative_mask(self):
        source = torch.zeros(5, 16, 16, 3)
        mask = torch.zeros(5, 16, 16)
        mask[2, 7, 7] = 1.0
        output = BerniniInpaintCondition.execute(
            [[torch.zeros(1), {}]],
            [[torch.zeros(1), {}]],
            FakeVAE(),
            source,
            16,
            16,
            5,
            1,
            source_as_context=True,
            mask=mask,
        )
        latent = output[2]
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 2, 2, 2))
        self.assertGreater(float(latent["noise_mask"].sum()), 0.0)
        self.assertIs(output[0][0][1]["context_latents"][0], latent["samples"])

    def test_inpaint_schema_has_no_removed_reference_hub_inputs(self):
        input_names = [
            item.id for item in BerniniInpaintCondition.define_schema().inputs
        ]
        self.assertNotIn("image_references", input_names)
        self.assertNotIn("video_references", input_names)

    def test_wan_frame_conversion_matches_official_node(self):
        self.assertEqual(bernini_nodes._validate_context_window_frames(81, 30), (21, 7))
        self.assertEqual(bernini_nodes._validate_context_window_frames(1, 0), (1, 0))

        with self.assertRaisesRegex(ValueError, "context_overlap"):
            bernini_nodes._validate_context_window_frames(5, 8)

    def test_input_order_matches_wan_context_window_node(self):
        input_names = list(BerniniContextWindowsCore.INPUT_TYPES()["required"])
        self.assertEqual(
            input_names,
            [
                "model",
                "context_length",
                "context_overlap",
                "position_mode",
                "context_schedule",
                "context_stride",
                "closed_loop",
                "fuse_method",
                "freenoise",
            ],
        )
        self.assertEqual(
            BerniniContextWindowsCore.INPUT_TYPES()["required"]["context_overlap"][1][
                "step"
            ],
            4,
        )

    def test_apply_uses_official_wan_context_options(self):
        original_install = bernini_nodes._install_bernini_absolute_rope_patch
        bernini_nodes._install_bernini_absolute_rope_patch = lambda: None
        try:
            patched = BerniniContextWindowsCore().apply(
                FakeModel(),
                context_length=81,
                context_overlap=28,
                context_schedule=comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                position_mode="absolute",
                context_stride=2,
                closed_loop=True,
                fuse_method=comfy.context_windows.ContextFuseMethods.PYRAMID,
                freenoise=True,
            )[0]
        finally:
            bernini_nodes._install_bernini_absolute_rope_patch = original_install

        handler = patched.model_options["context_handler"]
        self.assertIsInstance(handler, bernini_nodes.BerniniScheduledContextHandler)
        self.assertEqual(
            handler.context_schedule.name,
            comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
        )
        self.assertEqual(handler.context_length, 21)
        self.assertEqual(handler.context_overlap, 7)
        self.assertEqual(handler.context_stride, 2)
        self.assertIs(handler.closed_loop, True)
        self.assertEqual(handler.dim, 2)
        self.assertIs(handler.freenoise, True)
        self.assertEqual(handler.cond_retain_index_list, [])
        self.assertEqual(handler.latent_retain_index_list, [])
        self.assertIs(handler.split_conds_to_windows, False)
        self.assertIs(handler.causal_window_fix, True)
        self.assertIs(handler.turing_utils_absolute_positions, True)

    def test_context_windows_are_marked_for_absolute_rope(self):
        handler = bernini_nodes.BerniniScheduledContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(
                comfy.context_windows.ContextSchedules.STATIC_STANDARD
            ),
            fuse_method=comfy.context_windows.get_matching_fuse_method(
                comfy.context_windows.ContextFuseMethods.PYRAMID
            ),
            context_length=3,
            context_overlap=1,
            dim=2,
            causal_window_fix=True,
        )

        windows = handler.get_context_windows(None, torch.zeros(1, 4, 8), {})
        self.assertTrue(windows)
        self.assertTrue(
            all(
                getattr(window, "turing_utils_use_absolute_indices", False)
                for window in windows
            )
        )

    def test_relative_mode_does_not_install_absolute_rope_wrapper(self):
        with mock.patch.object(
            bernini_nodes,
            "_install_bernini_absolute_rope_patch",
            side_effect=AssertionError("absolute patch must not be installed"),
        ):
            patched = BerniniContextWindowsCore().apply(
                FakeModel(),
                context_length=81,
                context_overlap=28,
                context_schedule=comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                position_mode="relative",
                freenoise=False,
            )[0]
        handler = patched.model_options["context_handler"]
        self.assertIs(handler.turing_utils_absolute_positions, False)
        self.assertFalse(
            any(
                key == bernini_nodes._BERNINI_ROPE_WRAPPER_KEY
                for _, key, _ in patched.added_wrappers
            )
        )

    def test_rope_wrapper_includes_causal_anchor_index(self):
        window = comfy.context_windows.IndexListContextWindow(
            [3, 4, 5], dim=2, total_frames=8
        )
        window.turing_utils_use_absolute_indices = True
        window.causal_anchor_index = 2
        transformer_options = {"context_window": window}
        captured = {}

        def executor(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "ok"

        result = bernini_nodes._bernini_context_rope_wrapper(
            executor,
            None,
            None,
            None,
            None,
            None,
            transformer_options,
        )

        self.assertEqual(result, "ok")
        patched_options = captured["args"][5]
        self.assertIsNot(patched_options, transformer_options)
        self.assertEqual(
            patched_options[bernini_nodes._ABSOLUTE_INDEX_KEY], (2, 3, 4, 5)
        )

    def test_relative_rope_wrapper_keeps_official_window_local_positions(self):
        window = comfy.context_windows.IndexListContextWindow(
            [3, 5, 7], dim=2, total_frames=8
        )
        window.turing_utils_use_absolute_indices = False
        transformer_options = {"context_window": window}
        captured = {}

        def executor(*args, **kwargs):
            captured["args"] = args
            return "ok"

        result = bernini_nodes._bernini_context_rope_wrapper(
            executor, None, None, None, None, None, transformer_options
        )
        self.assertEqual(result, "ok")
        self.assertIs(captured["args"][5], transformer_options)
        self.assertNotIn("rope_options", transformer_options)

    def test_prepare_sampling_budgets_context_and_anchor_without_mutating_conds(self):
        handler = type(
            "Handler",
            (),
            {"dim": 2, "context_length": 3, "causal_window_fix": True},
        )()
        context = torch.zeros(1, 16, 8, 4, 4)
        conds = {"positive": [{"context_latents": [context]}]}
        captured = {}

        def executor(model, noise_shape, estimated_conds, *args, **kwargs):
            captured["noise_shape"] = noise_shape
            captured["conds"] = estimated_conds
            return model, estimated_conds, ["loaded"]

        result = bernini_nodes._bernini_prepare_sampling_wrapper(
            executor,
            "model",
            (1, 16, 8, 4, 4),
            conds,
            model_options={"context_handler": handler},
        )

        self.assertEqual(captured["noise_shape"], [1, 16, 4, 4, 4])
        estimated = captured["conds"]["positive"][0]["context_latents"][0]
        self.assertEqual(estimated.shape, (1, 16, 4, 4, 4))
        self.assertIs(result[1], conds)
        self.assertIs(conds["positive"][0]["context_latents"][0], context)

    def test_prepare_sampling_slices_matching_video_and_keeps_independent_refs(self):
        handler = SimpleNamespace(dim=2, context_length=3, causal_window_fix=True)
        source = torch.zeros(1, 16, 8, 4, 4)
        independent_ref = torch.zeros(1, 16, 3, 6, 6)
        conds = {
            "positive": [
                {
                    "context_latents": [source, independent_ref],
                    bernini_nodes._CONTEXT_ROLES_KEY: ("aligned", "global"),
                }
            ]
        }
        captured = {}

        def executor(model, noise_shape, estimated_conds, *args, **kwargs):
            captured["conds"] = estimated_conds
            return model, estimated_conds, []

        bernini_nodes._bernini_prepare_sampling_wrapper(
            executor,
            object(),
            (1, 16, 8, 4, 4),
            conds,
            model_options={"context_handler": handler},
        )
        estimated = captured["conds"]["positive"][0]["context_latents"]
        self.assertEqual(estimated[0].shape, (1, 16, 4, 4, 4))
        self.assertIs(estimated[1], independent_ref)
        self.assertEqual(source.shape, (1, 16, 8, 4, 4))

    def test_explicit_global_role_keeps_equal_length_reference(self):
        aligned = torch.zeros(1, 16, 8, 4, 4)
        global_ref = torch.zeros(1, 16, 8, 6, 6)
        conds = {
            "positive": [
                {
                    "context_latents": [aligned, global_ref],
                    bernini_nodes._CONTEXT_ROLES_KEY: ("aligned", "global"),
                }
            ]
        }
        estimated = bernini_nodes._estimate_conditioning(conds, 8, 4, 2)
        values = estimated["positive"][0]["context_latents"]
        self.assertEqual(values[0].shape[2], 4)
        self.assertIs(values[1], global_ref)

    def test_context_resize_uses_roles_instead_of_shape_heuristic(self):
        window = comfy.context_windows.IndexListContextWindow(
            [2, 3], dim=2, total_frames=5
        )
        aligned = torch.arange(5).reshape(1, 1, 5, 1, 1)
        global_ref = torch.zeros(1, 1, 5, 2, 2)
        cond = comfy.conds.CONDList([aligned, global_ref])
        resized = bernini_nodes._resize_bernini_context(
            "context_latents",
            cond,
            window,
            aligned,
            torch.device("cpu"),
            {
                bernini_nodes._CONTEXT_ROLES_KEY: comfy.conds.CONDConstant(
                    ("aligned", "global")
                )
            },
        )
        self.assertEqual(resized.cond[0].flatten().tolist(), [2, 3])
        self.assertIs(resized.cond[1], global_ref)

    def test_prepare_sampling_keeps_packed_latent_estimate_conservative(self):
        handler = SimpleNamespace(dim=2, context_length=8, causal_window_fix=True)
        conds = {"positive": []}
        calls = []

        def executor(_model, noise_shape, passed_conds, *args, **kwargs):
            calls.append((noise_shape, passed_conds))
            return "model", passed_conds, (), 0, 0

        result = bernini_nodes._bernini_prepare_sampling_wrapper(
            executor,
            object(),
            [1, 1, 64],
            conds,
            model_options={"context_handler": handler},
        )

        self.assertEqual(calls[0][0], [1, 1, 64])
        self.assertIs(calls[0][1], conds)
        self.assertIs(result[1], conds)


if __name__ == "__main__":
    unittest.main()
