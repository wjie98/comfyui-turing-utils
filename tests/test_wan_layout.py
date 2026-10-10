from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.wan_layout import (  # noqa: E402
    SCAIL_LAYOUT_KIND,
    WAN_LAYOUT_KIND,
    build_scail_attention_layout,
    build_wan_attention_layout,
    ensure_wan_attention_layout_provider,
    publish_scail_attention_layout,
    publish_wan_attention_layout,
)
from comfyui_turing_utils.attention.layout import (  # noqa: E402
    ATTENTION_LAYOUT_KEY,
    attention_semantic_layout,
    has_complete_attention_layout,
)
from comfyui_turing_utils.attention.sparse import _sparse_protected_ranges  # noqa: E402


class WanLayoutTest(unittest.TestCase):
    @staticmethod
    def model():
        return SimpleNamespace(
            patch_size=(1, 2, 2),
            ref_conv=object(),
            blocks=[object(), object(), object()],
        )

    @staticmethod
    def inputs():
        return (
            torch.zeros((1, 16, 3, 8, 10)),
            {
                "reference_latent": torch.zeros((1, 16, 8, 10)),
                "context_latents": [
                    torch.zeros((1, 16, 1, 8, 10)),
                    torch.zeros((1, 16, 4, 8, 10)),
                ],
            },
        )

    def test_sequence_matches_wan_forward_orig_order(self):
        x, kwargs = self.inputs()
        layout = build_wan_attention_layout(self.model(), x, {}, kwargs)
        self.assertEqual(layout.provider, WAN_LAYOUT_KIND)
        self.assertEqual(
            tuple((item.start, item.stop, item.role) for item in layout.query_segments),
            (
                (0, 20, "reference_image"),
                (20, 80, "target_video"),
                (80, 100, "reference_image"),
                (100, 120, "reference_video_anchor"),
                (120, 160, "reference_video"),
                (160, 180, "reference_video_anchor"),
            ),
        )
        self.assertEqual(
            tuple(
                (item.topology_id, item.start, item.stop) for item in layout.topologies
            ),
            (("target_video", 20, 80), ("context_video_1", 100, 180)),
        )
        self.assertIsNone(layout.validate(180, 180))

    def test_versioned_wire_drives_reference_anchor_policy(self):
        x, kwargs = self.inputs()
        options = {ATTENTION_LAYOUT_KEY: {"extension": "preserved"}}
        self.assertTrue(publish_wan_attention_layout(self.model(), x, options, kwargs))
        self.assertEqual(options[ATTENTION_LAYOUT_KEY]["extension"], "preserved")
        self.assertTrue(
            has_complete_attention_layout(options, 180, provider=WAN_LAYOUT_KIND)
        )
        self.assertEqual(attention_semantic_layout(options).protocol_version, 1)
        self.assertEqual(
            _sparse_protected_ranges(
                "auto",
                0,
                options,
                180,
                sparse_reference_image=False,
                sparse_reference_video=True,
                sparse_reference_audio=False,
            ),
            ((0, 20), (80, 120), (160, 180)),
        )

    def test_invalid_context_clears_stale_complete_layout(self):
        x, kwargs = self.inputs()
        options = {}
        self.assertTrue(publish_wan_attention_layout(self.model(), x, options, kwargs))
        kwargs["context_latents"].append(torch.zeros((1, 16, 8, 10)))
        self.assertFalse(publish_wan_attention_layout(self.model(), x, options, kwargs))
        self.assertFalse(has_complete_attention_layout(options, 180))
        self.assertEqual(
            options[ATTENTION_LAYOUT_KEY],
            {"provider": WAN_LAYOUT_KIND},
        )


class ScailLayoutTest(unittest.TestCase):
    @staticmethod
    def model():
        return SimpleNamespace(
            patch_size=(1, 2, 2),
            blocks=[object(), object(), object()],
        )

    @staticmethod
    def inputs(target_frames=3, pose_frames=3):
        return (
            torch.zeros((1, 20, target_frames, 8, 10)),
            {
                "reference_latent": torch.zeros((1, 20, 2, 8, 10)),
                "pose_latents": torch.zeros((1, 20, pose_frames, 4, 6)),
                "ref_mask_latents": torch.zeros((1, 28, target_frames + 2, 8, 10)),
                "sam_latents": torch.zeros((1, 28, pose_frames, 4, 6)),
            },
        )

    def test_sequence_matches_scail_forward_orig_order(self):
        x, kwargs = self.inputs()
        layout = build_scail_attention_layout(self.model(), x, {}, kwargs)
        self.assertEqual(layout.provider, SCAIL_LAYOUT_KIND)
        self.assertEqual(
            tuple((item.start, item.stop, item.role) for item in layout.query_segments),
            (
                (0, 40, "reference_image"),
                (40, 100, "target_video"),
                (100, 118, "pose_video"),
            ),
        )
        self.assertEqual(
            tuple(
                (item.topology_id, item.start, item.stop, item.tokens_per_frame)
                for item in layout.topologies
            ),
            (
                ("reference_images", 0, 40, 20),
                ("target_video", 40, 100, 20),
                ("pose_video", 100, 118, 6),
            ),
        )
        self.assertIsNone(layout.validate(118, 118))

    def test_auto_policy_maps_existing_reference_switches(self):
        x, kwargs = self.inputs()
        options = {}
        self.assertTrue(
            publish_scail_attention_layout(self.model(), x, options, kwargs)
        )
        self.assertEqual(
            _sparse_protected_ranges(
                "auto",
                0,
                options,
                118,
                sparse_reference_image=False,
                sparse_reference_video=True,
                sparse_reference_audio=False,
            ),
            ((0, 40),),
        )
        self.assertEqual(
            _sparse_protected_ranges(
                "auto",
                0,
                options,
                118,
                sparse_reference_image=False,
                sparse_reference_video=False,
                sparse_reference_audio=False,
            ),
            ((0, 40), (100, 118)),
        )

    def test_context_window_republishes_window_local_ranges(self):
        x, kwargs = self.inputs()
        options = {ATTENTION_LAYOUT_KEY: {"extension": "preserved"}}
        self.assertTrue(
            publish_scail_attention_layout(self.model(), x, options, kwargs)
        )
        window_x, window_kwargs = self.inputs(target_frames=2, pose_frames=2)
        self.assertTrue(
            publish_scail_attention_layout(
                self.model(), window_x, options, window_kwargs
            )
        )
        self.assertEqual(options[ATTENTION_LAYOUT_KEY]["extension"], "preserved")
        self.assertTrue(
            has_complete_attention_layout(options, 92, provider=SCAIL_LAYOUT_KIND)
        )
        self.assertEqual(
            tuple(
                (item.start, item.stop, item.role)
                for item in attention_semantic_layout(options).query_segments
            ),
            (
                (0, 40, "reference_image"),
                (40, 80, "target_video"),
                (80, 92, "pose_video"),
            ),
        )

    def test_invalid_window_clears_stale_complete_layout(self):
        x, kwargs = self.inputs()
        options = {}
        self.assertTrue(
            publish_scail_attention_layout(self.model(), x, options, kwargs)
        )
        kwargs["pose_latents"] = torch.zeros((1, 20, 8, 10))
        self.assertFalse(
            publish_scail_attention_layout(self.model(), x, options, kwargs)
        )
        self.assertFalse(has_complete_attention_layout(options, 118))
        self.assertEqual(
            options[ATTENTION_LAYOUT_KEY],
            {"provider": SCAIL_LAYOUT_KIND},
        )

    def test_scail2_installs_its_own_runtime_provider(self):
        from comfy.ldm.wan.model import SCAIL2WanModel

        diffusion = SCAIL2WanModel.__new__(SCAIL2WanModel)
        torch.nn.Module.__init__(diffusion)
        diffusion.patch_size = (1, 2, 2)
        diffusion.blocks = torch.nn.ModuleList([torch.nn.Identity()])

        class Patcher:
            def __init__(self):
                self.model = SimpleNamespace(diffusion_model=diffusion)
                self.object_patches = {}

            def add_object_patch(self, key, value):
                self.object_patches[key] = value

        patcher = Patcher()
        status = ensure_wan_attention_layout_provider(patcher)
        self.assertEqual(status.model_kind, SCAIL_LAYOUT_KIND)
        self.assertTrue(status.installed)
        self.assertEqual(
            set(patcher.object_patches),
            {"diffusion_model.forward_orig"},
        )


if __name__ == "__main__":
    unittest.main()
