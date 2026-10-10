from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.media import ResizeImageIfPresent
from comfyui_turing_utils.media.images import resize_image_if_present  # noqa: E402


def _run(image=None, mask=None, **overrides):
    options = {
        "width": 0,
        "height": 0,
        "resize_mode": "crop",
        "upscale_method": "bilinear",
        "crop_position": "center",
        "divisible_by": 1,
        "pad_color": "0, 0, 0",
    }
    options.update(overrides)
    return resize_image_if_present(image, mask, **options)


class ResizeImageIfPresentTest(unittest.TestCase):
    def test_schema_has_optional_image_first(self):
        schema = ResizeImageIfPresent.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsResizeImageIfPresent")
        self.assertEqual(schema.display_name, "Resize Image If Present")
        self.assertEqual(schema.inputs[0].id, "image")
        self.assertTrue(schema.inputs[0].optional)
        self.assertEqual(schema.inputs[1].id, "mask")
        self.assertTrue(schema.inputs[1].optional)

    def test_missing_image_returns_only_absent_outputs(self):
        mask = torch.ones(1, 8, 8)
        self.assertEqual(_run(mask=mask, width=16, height=16), (None, None, 0, 0))

        output = ResizeImageIfPresent.execute(
            width=16,
            height=16,
            resize_mode="crop",
            upscale_method="bilinear",
            crop_position="center",
            divisible_by=1,
            pad_color="0, 0, 0",
        )
        self.assertEqual(output.result, (None, None, 0, 0))

    def test_zero_dimensions_are_an_exact_passthrough(self):
        image = torch.rand(2, 8, 12, 3)
        mask = torch.rand(2, 8, 12)
        output, output_mask, width, height = _run(image, mask)
        self.assertIs(output, image)
        self.assertIs(output_mask, mask)
        self.assertEqual((width, height), (12, 8))

    def test_one_zero_dimension_preserves_aspect(self):
        image = torch.rand(1, 8, 12, 3)
        output, _, width, height = _run(image, width=0, height=16)
        self.assertEqual(tuple(output.shape), (1, 16, 24, 3))
        self.assertEqual((width, height), (24, 16))

    def test_crop_position_changes_the_selected_region(self):
        image = torch.zeros(1, 4, 8, 3)
        image[:, :, :4] = 1.0
        left, _, _, _ = _run(image, width=4, height=4, crop_position="left")
        right, _, _, _ = _run(image, width=4, height=4, crop_position="right")
        self.assertGreater(float(left.mean()), 0.99)
        self.assertLess(float(right.mean()), 0.01)

    def test_crop_keeps_image_and_mask_geometry_aligned(self):
        image = torch.zeros(1, 4, 8, 3)
        mask = torch.zeros(1, 4, 8)
        image[:, :, :4] = 1.0
        mask[:, :, :4] = 1.0
        output, output_mask, width, height = _run(
            image, mask, width=6, height=6, crop_position="left"
        )
        self.assertEqual(tuple(output.shape), (1, 6, 6, 3))
        self.assertEqual(tuple(output_mask.shape), (1, 6, 6))
        self.assertEqual((width, height), (6, 6))
        self.assertGreater(float(output.mean()), 0.99)
        self.assertGreater(float(output_mask.mean()), 0.99)

    def test_fit_reports_actual_content_size(self):
        image = torch.rand(1, 10, 20, 3)
        output, _, width, height = _run(image, width=16, height=16, resize_mode="fit")
        self.assertEqual(tuple(output.shape), (1, 8, 16, 3))
        self.assertEqual((width, height), (16, 8))

    def test_padding_uses_color_and_position(self):
        image = torch.ones(1, 4, 8, 3)
        output, _, width, height = _run(
            image,
            width=8,
            height=8,
            resize_mode="pad",
            crop_position="top",
            pad_color="255, 0, 0",
        )
        self.assertEqual((width, height), (8, 8))
        self.assertTrue(torch.equal(output[:, :4], torch.ones_like(output[:, :4])))
        self.assertTrue(
            torch.equal(output[:, 4:, :, 0], torch.ones_like(output[:, 4:, :, 0]))
        )
        self.assertTrue(
            torch.equal(output[:, 4:, :, 1:], torch.zeros_like(output[:, 4:, :, 1:]))
        )

    def test_rgb_pad_color_keeps_opaque_alpha(self):
        image = torch.ones(1, 4, 8, 4)
        output, _, _, _ = _run(
            image, width=8, height=8, resize_mode="pad", pad_color="255, 0, 0"
        )
        self.assertTrue(torch.equal(output[..., 3], torch.ones_like(output[..., 3])))

    def test_divisibility_rounds_target_down(self):
        image = torch.rand(1, 8, 12, 3)
        output, _, width, height = _run(
            image, width=31, height=23, resize_mode="stretch", divisible_by=8
        )
        self.assertEqual(tuple(output.shape), (1, 16, 24, 3))
        self.assertEqual((width, height), (24, 16))


if __name__ == "__main__":
    unittest.main()
