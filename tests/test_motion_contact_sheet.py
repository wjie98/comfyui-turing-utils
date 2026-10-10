from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import unittest

import torch
from comfy_api.latest import InputImpl, Types


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.media import VideoMotionContactSheet
from comfyui_turing_utils.media.contact_sheet import render_contact_sheet
from comfyui_turing_utils.media.sampling import (
    motion_weighted_frame_indices,
    uniform_frame_indices,
)


class MotionContactSheetTest(unittest.TestCase):
    def test_uniform_sampling_always_fills_the_grid_and_keeps_endpoints(self):
        self.assertEqual(uniform_frame_indices(5, 9), [0, 0, 1, 2, 2, 2, 3, 4, 4])
        self.assertEqual(uniform_frame_indices(17, 4), [0, 5, 11, 16])

    def test_motion_sampling_keeps_endpoints_and_moves_panels_to_activity(self):
        indices = motion_weighted_frame_indices([0.0, 0.0, 10.0, 10.0, 0.0, 0.0], 4)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 6)
        self.assertGreaterEqual(indices[1], 2)
        self.assertLessEqual(indices[2], 4)

    def test_render_respects_exact_canvas_size_with_film_border(self):
        frames = torch.zeros(4, 48, 80, 3)
        frames[..., 0] = 1.0
        sheet = render_contact_sheet(
            frames,
            [0.0, 1.0, 2.0, 3.0],
            grid_size=2,
            width=161,
            height=97,
            resize_mode="fit",
            gap=1,
            film_border=True,
            annotation="index_timestamp",
        )
        self.assertEqual(tuple(sheet.shape), (1, 97, 161, 3))
        self.assertLess(float(sheet[0, 0, 0].mean()), 0.2)
        self.assertGreater(float(sheet[0, 24, 40, 0]), 0.9)

    def test_clean_grid_does_not_add_a_caption_or_border(self):
        frames = torch.zeros(4, 32, 32, 3)
        frames[..., 1] = 1.0
        sheet = render_contact_sheet(
            frames,
            [0.0, 1.0, 2.0, 3.0],
            grid_size=2,
            width=64,
            height=64,
            resize_mode="stretch",
            gap=0,
            film_border=False,
            annotation="none",
        )
        self.assertTrue(torch.allclose(sheet[..., 1], torch.ones_like(sheet[..., 1])))
        self.assertTrue(torch.equal(sheet[..., 0], torch.zeros_like(sheet[..., 0])))

    def test_schema_exposes_independent_film_border_switch(self):
        schema = VideoMotionContactSheet.define_schema()
        inputs = {item.id: item for item in schema.inputs}
        self.assertTrue(inputs["film_border"].default)
        self.assertEqual(inputs["annotation"].default, "index")
        self.assertTrue(inputs["video"].optional)
        self.assertTrue(inputs["frames"].optional)

    def test_loaded_video_components_use_embedded_frame_rate(self):
        frames = torch.zeros(12, 32, 48, 3)
        frames[:, :, :, 0] = torch.arange(12).reshape(12, 1, 1) / 11.0
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=frames, frame_rate=Fraction(12, 1))
        )
        output = VideoMotionContactSheet.execute(
            grid_size=3,
            sampling="uniform",
            width=96,
            height=64,
            resize_mode="stretch",
            gap=0,
            film_border=False,
            annotation="none",
            image_frame_rate=1.0,
            video=video,
        )
        self.assertEqual(tuple(output[0].shape), (1, 64, 96, 3))
        self.assertEqual(tuple(output[1].shape), (9, 32, 48, 3))
        self.assertIn("chronological motion storyboard", output[2])


if __name__ == "__main__":
    unittest.main()
