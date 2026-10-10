from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT.parents[1]))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.nested_tensor  # noqa: E402
import comfy.sampler_helpers  # noqa: E402
from comfyui_turing_utils.nodes.latent import (  # noqa: E402
    VIDEO_MASK_SPECS,
    SetVideoLatentNoiseMask,
    VideoLatentCompositeMasked,
)
from comfyui_turing_utils.registration import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)  # noqa: E402


class VideoLatentCompositeMaskedTest(unittest.TestCase):
    def setUp(self):
        self.original = {
            "samples": torch.randn(2, 24, 7, 2, 3),
            "metadata": {"original": True},
        }
        self.replacement = {
            "samples": torch.randn(2, 24, 7, 2, 3),
            "metadata": {"replacement": True},
        }

    def composite(self, **kwargs):
        return VideoLatentCompositeMasked.execute(
            self.original, self.replacement, **kwargs
        ).result[0]

    def assert_composite(self, output, expected_mask):
        expected_mask = expected_mask.expand(2, 1, 7, 2, 3)
        torch.testing.assert_close(
            output["noise_mask"], expected_mask.float(), rtol=0, atol=0
        )
        expected = torch.where(
            expected_mask.bool(),
            self.replacement["samples"].to(self.original["samples"]),
            self.original["samples"],
        )
        torch.testing.assert_close(output["samples"], expected, rtol=0, atol=0)

    def test_schema_and_registration(self):
        schema = VideoLatentCompositeMasked.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsVideoLatentCompositeMasked")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["original_latent", "replacement_latent", "mask", "type"],
        )
        self.assertTrue(schema.inputs[2].optional)
        self.assertEqual(schema.inputs[3].options, list(VIDEO_MASK_SPECS))
        self.assertEqual(schema.inputs[3].default, "minimax")
        self.assertEqual([item.io_type for item in schema.outputs], ["LATENT"])
        self.assertIs(NODE_CLASS_MAPPINGS[schema.node_id], VideoLatentCompositeMasked)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS[schema.node_id], schema.display_name
        )

    def test_neither_mask_replaces_all_and_ignores_original_mask(self):
        self.original["noise_mask"] = torch.zeros(2, 1, 7, 2, 3)
        self.assert_composite(self.composite(), torch.ones(1, 1, 7, 2, 3))

    def test_only_inherited_mask_preserves_batch_time_and_unions_channels(self):
        inherited = torch.zeros(2, 24, 7, 2, 3)
        inherited[0, 10, 1, 0, 2] = 0.1
        inherited[1, 23, 6, 1, 0] = 1
        self.replacement["noise_mask"] = inherited
        self.assert_composite(self.composite(), inherited.amax(1, keepdim=True) > 0)

    def test_only_explicit_mask_maps_h3_groups_and_keeps_each_batch(self):
        mask = torch.zeros(2, 22, 4, 6)
        mask[0, 16, 3, 5] = 0.001
        mask[1, 17, 0, 0] = 1
        expected = torch.zeros(2, 1, 7, 2, 3)
        expected[0, 0, 4, 1, 2] = 1
        expected[1, 0, 5, 0, 0] = 1
        self.assert_composite(self.composite(mask=mask), expected)

    def test_inherited_and_explicit_regions_are_unioned_not_multiplied(self):
        inherited = torch.zeros(1, 1, 7, 2, 3)
        inherited[0, 0, 0, 0, 0] = 0.2
        self.replacement["noise_mask"] = inherited
        explicit = torch.zeros(7, 2, 3)
        explicit[6, 1, 2] = 0.3
        expected = (inherited > 0) | (explicit[None, None] > 0)
        self.assert_composite(self.composite(mask=explicit), expected)

    def test_zero_explicit_mask_does_not_erase_inherited_region(self):
        inherited = torch.ones(1, 1, 7, 2, 3)
        self.replacement["noise_mask"] = inherited
        self.assert_composite(self.composite(mask=torch.zeros(22, 2, 3)), inherited)

    def test_connected_zero_mask_is_not_treated_as_absent(self):
        for inherited in (False, True):
            with self.subTest(inherited=inherited):
                if inherited:
                    self.replacement["noise_mask"] = torch.zeros(1, 1, 7, 2, 3)
                    output = self.composite()
                else:
                    output = self.composite(mask=torch.zeros(7, 2, 3))
                self.assert_composite(output, torch.zeros(1, 1, 7, 2, 3))

    def test_tiny_positive_values_are_coverage_not_opacity(self):
        for inherited in (False, True):
            with self.subTest(inherited=inherited):
                mask = torch.zeros(7, 2, 3, dtype=torch.float64)
                mask[2, 0, 1] = 1e-50
                if inherited:
                    self.replacement["noise_mask"] = mask[None, None]
                    output = self.composite()
                else:
                    output = self.composite(mask=mask)
                self.assert_composite(output, mask[None, None] > 0)

    def test_all_profiles_reuse_set_mask_mapping(self):
        for model_type, count in (
            ("minimax", 22),
            ("wan", 25),
            ("ltxv", 49),
            ("hunyuan_video", 25),
            ("hunyuan_video_15", 25),
            ("mochi", 37),
        ):
            with self.subTest(model_type=model_type):
                mask = torch.zeros(count, 4, 6)
                mask[-1, -1, -1] = 1
                mapped = SetVideoLatentNoiseMask.execute(
                    self.original, mask, model_type
                ).result[0]["noise_mask"]
                self.assert_composite(
                    self.composite(mask=mask, type=model_type), mapped
                )

    def test_direct_time_mapping_does_not_require_canonical_h3_length(self):
        self.original["samples"] = self.original["samples"][:, :, :3]
        self.replacement["samples"] = self.replacement["samples"][:, :, :3]
        mask = torch.zeros(3, 2, 3)
        mask[1] = 1
        output = self.composite(mask=mask)
        torch.testing.assert_close(
            output["samples"][:, :, 1],
            self.replacement["samples"][:, :, 1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output["samples"][:, :, 0],
            self.original["samples"][:, :, 0],
            rtol=0,
            atol=0,
        )

    def test_original_dtype_and_metadata_and_input_ownership(self):
        self.original["samples"] = self.original["samples"].half()
        mask = torch.zeros(7, 2, 3)
        mask[1] = 1
        old_mask = torch.ones(1, 1, 7, 2, 3)
        self.original["noise_mask"] = old_mask
        original_before = self.original["samples"].clone()
        replacement_before = self.replacement["samples"].clone()
        output = self.composite(mask=mask)
        self.assertEqual(output["samples"].dtype, torch.float16)
        self.assertIs(output["metadata"], self.original["metadata"])
        self.assertIs(self.original["noise_mask"], old_mask)
        self.assert_composite(output, mask[None, None])
        output["samples"].zero_()
        output["noise_mask"].zero_()
        torch.testing.assert_close(
            self.original["samples"], original_before, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.replacement["samples"], replacement_before, rtol=0, atol=0
        )
        self.assertTrue(torch.all(old_mask == 1))
        self.assertTrue(torch.all(mask[1] == 1))

    def test_output_mask_survives_native_sampler_preparation(self):
        mask = torch.zeros(22, 4, 6)
        mask[4, 1, 1] = 1
        output = self.composite(mask=mask)
        prepared = comfy.sampler_helpers.prepare_mask(
            output["noise_mask"], output["samples"].shape, "cpu"
        )
        torch.testing.assert_close(
            prepared, output["noise_mask"].expand_as(prepared), rtol=0, atol=0
        )

    def test_rejects_broadcastable_latent_mismatches(self):
        for shape in (
            (1, 24, 7, 2, 3),
            (2, 1, 7, 2, 3),
            (2, 24, 1, 2, 3),
            (2, 24, 7, 1, 3),
        ):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "identical"),
            ):
                VideoLatentCompositeMasked.execute(
                    self.original, {"samples": torch.zeros(shape)}
                )

    def test_rejects_bad_inherited_mask_shape(self):
        for shape in (
            (7, 2, 3),
            (1, 1, 1, 2, 3),
            (1, 1, 7, 4, 6),
            (3, 1, 7, 2, 3),
            (1, 2, 7, 2, 3),
        ):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "noise_mask"),
            ):
                self.replacement["noise_mask"] = torch.ones(shape)
                self.composite()

    def test_rejects_invalid_values_in_either_mask(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            for inherited in (False, True):
                with (
                    self.subTest(value=value, inherited=inherited),
                    self.assertRaisesRegex(ValueError, r"within \[0,1\]"),
                ):
                    self.replacement.pop("noise_mask", None)
                    mask = torch.full((7, 2, 3), value)
                    if inherited:
                        self.replacement["noise_mask"] = mask[None, None]
                        self.composite()
                    else:
                        self.composite(mask=mask)

    def test_rejects_incompatible_image_frame_count(self):
        with self.assertRaisesRegex(
            ValueError, "Expected 7 latent-frame masks or 22 image-frame masks"
        ):
            self.composite(mask=torch.ones(21, 2, 3))
        with self.assertRaisesRegex(ValueError, "Unknown video mask type"):
            self.composite(type="unknown")

    def test_rejects_nested_av_inputs(self):
        av = {
            "samples": comfy.nested_tensor.NestedTensor(
                (self.original["samples"], torch.zeros(2, 32, 2, 10))
            )
        }
        with self.assertRaisesRegex(ValueError, "Separate AV"):
            VideoLatentCompositeMasked.execute(av, self.replacement)
        with self.assertRaisesRegex(ValueError, "Separate AV"):
            VideoLatentCompositeMasked.execute(self.original, av)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_original_with_cpu_masks_and_replacement(self):
        self.original["samples"] = self.original["samples"].cuda()
        self.replacement["noise_mask"] = torch.zeros(1, 1, 7, 2, 3)
        mask = torch.zeros(22, 4, 6)
        mask[0] = 1
        output = self.composite(mask=mask)
        self.assertEqual(output["samples"].device, self.original["samples"].device)
        self.assertEqual(output["noise_mask"].device, self.original["samples"].device)
        torch.testing.assert_close(
            output["samples"][:, :, 0].cpu(),
            self.replacement["samples"][:, :, 0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output["samples"][:, :, 1:],
            self.original["samples"][:, :, 1:],
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
