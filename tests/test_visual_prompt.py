from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.visual_prompt import MaskToVisualPrompts
from comfyui_turing_utils.media.visual_prompts import visual_prompts_from_mask


class VisualPromptTest(unittest.TestCase):
    def test_schema_exposes_common_prompt_contracts_and_preview(self):
        schema = MaskToVisualPrompts.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsMaskToVisualPrompts")
        self.assertEqual(
            [input_.id for input_ in schema.inputs],
            [
                "image",
                "mask",
                "mask_threshold",
                "positive_point_count",
                "negative_point_count",
                "bbox_padding",
            ],
        )
        self.assertEqual(
            [output.id for output in schema.outputs],
            [
                "preview",
                "positive_coords",
                "negative_coords",
                "bbox",
                "bounding_box",
            ],
        )
        self.assertEqual(schema.outputs[3].io_type, "BBOX")
        self.assertEqual(schema.outputs[4].io_type, "BOUNDING_BOX")

    def test_uses_first_frames_and_builds_both_bbox_formats(self):
        images = torch.rand(3, 20, 30, 3)
        masks = torch.zeros(2, 20, 30)
        masks[0, 5:15, 10:20] = 0.8
        masks[1, 1:3, 1:3] = 1.0
        output = MaskToVisualPrompts.execute(
            image=images,
            mask=masks,
            mask_threshold=0.5,
            positive_point_count=3,
            negative_point_count=4,
            bbox_padding=0.1,
        ).result

        preview, positive_json, negative_json, legacy, canonical = output
        self.assertEqual(tuple(preview.shape), (1, 20, 30, 3))
        self.assertEqual(legacy, [{"startX": 9, "startY": 4, "endX": 21, "endY": 16}])
        self.assertEqual(canonical, [[{"x": 9, "y": 4, "width": 12, "height": 12}]])

        positive = json.loads(positive_json)
        negative = json.loads(negative_json)
        self.assertEqual(len(positive), 3)
        self.assertEqual(len(negative), 4)
        for point in positive:
            self.assertGreaterEqual(point["x"], 10)
            self.assertLess(point["x"], 20)
            self.assertGreaterEqual(point["y"], 5)
            self.assertLess(point["y"], 15)
        for point in negative:
            self.assertEqual(bool(masks[0, point["y"], point["x"]] >= 0.5), False)

    def test_preview_renders_mask_points_and_bbox_distinctly(self):
        images = torch.full((1, 64, 96, 3), 0.12)
        masks = torch.zeros(1, 64, 96)
        masks[:, 16:48, 24:72] = 1.0
        preview, positive_json, negative_json, _, _ = MaskToVisualPrompts.execute(
            image=images,
            mask=masks,
            mask_threshold=0.5,
            positive_point_count=4,
            negative_point_count=4,
            bbox_padding=0.05,
        ).result

        self.assertEqual(tuple(preview.shape), (1, 64, 96, 3))
        self.assertGreaterEqual(float(preview.min()), 0.0)
        self.assertLessEqual(float(preview.max()), 1.0)
        self.assertEqual(len(json.loads(positive_json)), 4)
        self.assertEqual(len(json.loads(negative_json)), 4)
        pixels = preview[0]
        green = (pixels[..., 1] > 0.65) & (pixels[..., 1] > pixels[..., 0] + 0.15)
        red = (pixels[..., 0] > 0.75) & (pixels[..., 0] > pixels[..., 1] + 0.15)
        amber = (
            (pixels[..., 0] > 0.8) & (pixels[..., 1] > 0.55) & (pixels[..., 2] < 0.5)
        )
        self.assertTrue(bool(green.any()))
        self.assertTrue(bool(red.any()))
        self.assertTrue(bool(amber.any()))

    def test_positive_points_cover_distinct_colour_regions(self):
        image = torch.zeros(1, 30, 60, 3)
        image[:, :, :20, 0] = 1.0
        image[:, :, 20:40, 1] = 1.0
        image[:, :, 40:, 2] = 1.0
        mask = torch.zeros(1, 30, 60)
        mask[:, 3:27, 3:57] = 1.0

        positive_json, _, _, _ = visual_prompts_from_mask(
            mask,
            mask_threshold=0.5,
            positive_point_count=4,
            negative_point_count=0,
            bbox_padding=0.0,
            image=image,
        )
        positive = json.loads(positive_json)
        represented = {
            int(torch.argmax(image[0, point["y"], point["x"]]).item())
            for point in positive
        }
        self.assertEqual(represented, {0, 1, 2})

    def test_positive_points_prioritise_a_thin_mask_branch(self):
        image = torch.full((1, 40, 64, 3), 0.5)
        mask = torch.zeros(1, 40, 64)
        mask[:, 8:32, 8:28] = 1.0
        mask[:, 19:21, 28:58] = 1.0

        positive_json, _, _, _ = visual_prompts_from_mask(
            mask,
            mask_threshold=0.5,
            positive_point_count=5,
            negative_point_count=0,
            bbox_padding=0.0,
            image=image,
        )
        positive = json.loads(positive_json)
        self.assertTrue(
            any(point["x"] >= 28 and 19 <= point["y"] < 21 for point in positive),
            positive,
        )

    def test_rejects_empty_mask(self):
        images = torch.zeros(1, 8, 8, 3)
        masks = torch.zeros(1, 8, 8)
        with self.assertRaisesRegex(ValueError, "no foreground"):
            visual_prompts_from_mask(masks[:1], 0.5, 1, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "no foreground"):
            MaskToVisualPrompts.execute(
                image=images,
                mask=masks,
                mask_threshold=0.5,
                positive_point_count=1,
                negative_point_count=0,
                bbox_padding=0.0,
            )

    def test_rejects_empty_or_spatially_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            MaskToVisualPrompts.execute(
                image=torch.zeros(0, 8, 8, 3),
                mask=torch.ones(1, 8, 8),
                mask_threshold=0.5,
                positive_point_count=1,
                negative_point_count=0,
                bbox_padding=0.0,
            )
        with self.assertRaisesRegex(ValueError, "spatial dimensions must match"):
            MaskToVisualPrompts.execute(
                image=torch.zeros(1, 8, 9, 3),
                mask=torch.ones(1, 8, 8),
                mask_threshold=0.5,
                positive_point_count=1,
                negative_point_count=0,
                bbox_padding=0.0,
            )


if __name__ == "__main__":
    unittest.main()
