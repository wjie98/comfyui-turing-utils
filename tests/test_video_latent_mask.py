from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.nested_tensor  # noqa: E402
import comfy.sampler_helpers  # noqa: E402
from comfy_extras.nodes_hunyuan import (
    EmptyHunyuanLatentVideo,
    EmptyHunyuanVideo15Latent,
)  # noqa: E402
from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVConcatAVLatent,
    LTXVSeparateAVLatent,
)  # noqa: E402
from comfy_extras.nodes_mochi import EmptyMochiLatentVideo  # noqa: E402
from comfyui_turing_utils.nodes.latent import SetVideoLatentNoiseMask  # noqa: E402


MASK_TYPES = ("wan", "minimax", "ltxv", "hunyuan_video", "hunyuan_video_15", "mochi")
CAUSAL_PROFILES = (
    ("wan", 4),
    ("ltxv", 8),
    ("hunyuan_video", 4),
    ("hunyuan_video_15", 4),
    ("mochi", 6),
)


class VideoLatentNoiseMaskTest(unittest.TestCase):
    def apply(
        self,
        mask,
        frames,
        model_type="minimax",
        batch=1,
        height=2,
        width=3,
        channels=24,
    ):
        samples = {"samples": torch.zeros(batch, channels, frames, height, width)}
        return SetVideoLatentNoiseMask.execute(
            samples=samples, mask=mask, type=model_type
        ).result[0]["noise_mask"]

    def test_schema(self):
        schema = SetVideoLatentNoiseMask.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsSetVideoLatentNoiseMask")
        self.assertEqual(
            [item.id for item in schema.inputs], ["samples", "mask", "type"]
        )
        self.assertEqual(schema.inputs[2].options, list(MASK_TYPES))
        self.assertEqual(schema.inputs[2].default, "minimax")
        self.assertEqual(
            [item.io_type for item in schema.inputs], ["LATENT", "MASK", "COMBO"]
        )
        self.assertEqual([item.io_type for item in schema.outputs], ["LATENT"])

    def test_direct_mapping_takes_priority_over_model_grid(self):
        mask = torch.arange(5, dtype=torch.float32)[:, None, None].expand(5, 2, 3) / 4
        for model_type in MASK_TYPES:
            with self.subTest(model_type=model_type):
                output = self.apply(mask, 5, model_type)
                torch.testing.assert_close(output[0, 0], mask, rtol=0, atol=0)

    def test_single_frame_for_every_profile(self):
        for model_type in MASK_TYPES:
            with self.subTest(model_type=model_type):
                mask = torch.full((1, 2, 3), 0.75)
                output = self.apply(mask, 1, model_type)
                torch.testing.assert_close(output[0, 0], mask, rtol=0, atol=0)

    def test_h3_124_pixel_frames_preserve_exactly_two_latent_positions(self):
        mask = torch.ones(124, 2, 3)
        mask[:5] = 0
        output = self.apply(mask, 37)
        self.assertEqual(output.shape, (1, 1, 37, 2, 3))
        self.assertEqual(output[:, :, :2].count_nonzero().item(), 0)
        self.assertTrue(torch.all(output[:, :, 2:] == 1))

    def test_h3_block_phase_and_final_partial_block(self):
        mask = torch.arange(22, dtype=torch.float32)[:, None, None] / 21
        output = self.apply(mask, 7, height=1, width=1)
        expected = torch.tensor([0, 4, 8, 12, 16, 17, 21], dtype=torch.float32) / 21
        torch.testing.assert_close(output.flatten(), expected, rtol=0, atol=0)

    def test_h3_five_frame_clip(self):
        mask = torch.zeros(5, 2, 3)
        mask[4, 1, 2] = 1
        output = self.apply(mask, 2)
        self.assertEqual(output[0, 0, 0].count_nonzero().item(), 0)
        self.assertEqual(output[0, 0, 1, 1, 2].item(), 1)

    def test_wan_first_frame_then_groups_of_four(self):
        mask = torch.arange(9, dtype=torch.float32)[:, None, None] / 8
        output = self.apply(mask, 3, "wan", height=1, width=1)
        torch.testing.assert_close(
            output.flatten(), torch.tensor([0, 0.5, 1.0]), rtol=0, atol=0
        )

    def test_causal_profiles_keep_first_frame_then_merge_fixed_groups(self):
        for model_type, stride in CAUSAL_PROFILES:
            for frames in (2, 3, 17):
                with self.subTest(model_type=model_type, frames=frames):
                    pixel_frames = 1 + stride * (frames - 1)
                    mask = torch.arange(pixel_frames, dtype=torch.float32)[
                        :, None, None
                    ] / (pixel_frames - 1)
                    output = self.apply(mask, frames, model_type, height=1, width=1)
                    expected = mask[::stride, 0, 0]
                    torch.testing.assert_close(
                        output.flatten(), expected, rtol=0, atol=0
                    )

    def test_new_profiles_preserve_prefix_and_sampler_mask_shape(self):
        for model_type, stride, channels in (
            ("ltxv", 8, 128),
            ("hunyuan_video", 4, 16),
            ("hunyuan_video_15", 4, 32),
            ("mochi", 6, 12),
        ):
            with self.subTest(model_type=model_type):
                mask = torch.ones(1 + stride * 4, 4, 6)
                mask[: 1 + stride] = 0
                output = self.apply(mask, 5, model_type, batch=2, channels=channels)
                self.assertEqual(output.shape, (2, 1, 5, 2, 3))
                self.assertEqual(output[:, :, :2].count_nonzero().item(), 0)
                self.assertTrue(torch.all(output[:, :, 2:] == 1))
                prepared = comfy.sampler_helpers.prepare_mask(
                    output, (2, channels, 5, 2, 3), "cpu"
                )
                torch.testing.assert_close(
                    prepared, output.expand_as(prepared), rtol=0, atol=0
                )

    def test_official_empty_video_latents_accept_image_frame_masks(self):
        for model_type, node, length, spatial_stride in (
            ("ltxv", EmptyLTXVLatentVideo, 17, 32),
            ("hunyuan_video", EmptyHunyuanLatentVideo, 9, 8),
            ("hunyuan_video_15", EmptyHunyuanVideo15Latent, 9, 16),
            ("mochi", EmptyMochiLatentVideo, 13, 8),
        ):
            with self.subTest(model_type=model_type):
                samples = node.execute(
                    width=96, height=64, length=length, batch_size=2
                ).result[0]
                mask = torch.zeros(length, 64, 96)
                mask[-1, -1, -1] = 0.75
                result = SetVideoLatentNoiseMask.execute(
                    samples, mask, model_type
                ).result[0]
                self.assertIs(result["samples"], samples["samples"])
                self.assertEqual(
                    result.get("downscale_ratio_spacial"),
                    samples.get("downscale_ratio_spacial"),
                )
                output = result["noise_mask"]
                self.assertEqual(
                    output.shape, (2, 1, 3, 64 // spatial_stride, 96 // spatial_stride)
                )
                self.assertEqual(output.count_nonzero().item(), 2)
                self.assertTrue(torch.all(output[:, 0, -1, -1, -1] == 0.75))

    def test_ltx_av_separate_mask_concat_preserves_audio_and_video_mask(self):
        video = EmptyLTXVLatentVideo.execute(width=96, height=64, length=17).result[0]
        audio = {
            "samples": torch.zeros(1, 8, 12, 16),
            "noise_mask": torch.full((1, 1, 12, 16), 0.25),
        }
        av = LTXVConcatAVLatent.execute(video, audio).result[0]
        video, separated_audio = LTXVSeparateAVLatent.execute(av).result
        mask = torch.ones(17, 64, 96)
        mask[:9] = 0
        masked_video = SetVideoLatentNoiseMask.execute(video, mask, "ltxv").result[0]
        result = LTXVConcatAVLatent.execute(masked_video, separated_audio).result[0]
        self.assertIs(result["samples"].unbind()[1], audio["samples"])
        self.assertIs(result["noise_mask"].unbind()[1], audio["noise_mask"])
        video_mask = result["noise_mask"].unbind()[0]
        torch.testing.assert_close(
            video_mask, masked_video["noise_mask"], rtol=0, atol=0
        )
        self.assertEqual(video_mask[:, :, :2].count_nonzero().item(), 0)
        self.assertTrue(torch.all(video_mask[:, :, 2:] == 1))

    def test_causal_temporal_union_keeps_each_pixel_and_batch_independent(self):
        for model_type, stride in CAUSAL_PROFILES:
            with self.subTest(model_type=model_type):
                mask = torch.zeros(2, 1 + stride * 2, 2, stride)
                for frame in range(stride):
                    mask[0, 1 + frame, 0, frame] = (frame + 1) / stride
                    mask[1, 1 + stride + frame, 1, frame] = (frame + 1) / stride
                output = self.apply(mask, 3, model_type, batch=2, width=stride)
                self.assertEqual(output[:, :, 0].count_nonzero().item(), 0)
                torch.testing.assert_close(
                    output[:, 0, 1], mask[:, 1 : 1 + stride].amax(1), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    output[:, 0, 2], mask[:, 1 + stride :].amax(1), rtol=0, atol=0
                )

    def test_temporal_union_keeps_regions_from_every_frame(self):
        mask = torch.zeros(5, 2, 3)
        mask[1, 0, 0] = 1
        mask[2, 0, 1] = 0.25
        mask[3, 1, 0] = 0.5
        mask[4, 1, 2] = 0.75
        output = self.apply(mask, 2)
        torch.testing.assert_close(output[0, 0, 1], mask[1:].amax(0), rtol=0, atol=0)

    def test_each_image_frame_maps_to_only_its_temporal_group(self):
        profiles = [
            (model_type, (1, stride, stride)) for model_type, stride in CAUSAL_PROFILES
        ]
        profiles.append(("minimax", (1, 4, 4, 4, 4, 1, 4)))
        for model_type, groups in profiles:
            frame_index = 0
            for latent_index, group_size in enumerate(groups):
                for _ in range(group_size):
                    with self.subTest(model_type=model_type, frame=frame_index):
                        mask = torch.zeros(sum(groups), 1, 1)
                        mask[frame_index] = 1
                        output = self.apply(
                            mask, len(groups), model_type, height=1, width=1
                        )
                        self.assertEqual(output.count_nonzero().item(), 1)
                        self.assertEqual(output[0, 0, latent_index, 0, 0].item(), 1)
                    frame_index += 1

    def test_spatial_union_preserves_single_pixels(self):
        mask = torch.zeros(5, 16, 24)
        mask[1, 7, 7] = 1
        mask[4, 15, 23] = 0.5
        output = self.apply(mask, 2)
        expected = torch.zeros(2, 3)
        expected[0, 0] = 1
        expected[1, 2] = 0.5
        torch.testing.assert_close(output[0, 0, 1], expected, rtol=0, atol=0)

    def test_noninteger_spatial_bins_are_conservative(self):
        mask = torch.zeros(1, 3, 5)
        mask[0, 1, 2] = 1
        output = self.apply(mask, 1, height=2, width=2)
        self.assertTrue(torch.all(output == 1))

    def test_smaller_spatial_mask_can_expand_without_averaging(self):
        output = self.apply(torch.tensor([[[0.75]]]), 1)
        self.assertEqual(output.shape, (1, 1, 1, 2, 3))
        self.assertTrue(torch.all(output == 0.75))

    def test_one_sequence_is_shared_across_video_batch(self):
        mask = torch.zeros(5, 2, 3)
        mask[4] = 1
        output = self.apply(mask, 2, batch=2)
        self.assertEqual(output.shape, (2, 1, 2, 2, 3))
        torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)

    def test_explicit_batch_sequences_remain_independent(self):
        mask = torch.zeros(2, 5, 2, 3)
        mask[0, 0] = 1
        mask[1, 4] = 1
        output = self.apply(mask, 2, batch=2)
        self.assertTrue(torch.all(output[0, 0, 0] == 1))
        self.assertTrue(torch.all(output[0, 0, 1] == 0))
        self.assertTrue(torch.all(output[1, 0, 0] == 0))
        self.assertTrue(torch.all(output[1, 0, 1] == 1))

    def test_sampler_preparation_does_not_change_temporal_boundary(self):
        mask = torch.ones(124, 2, 3)
        mask[:5] = 0
        output = self.apply(mask, 37, batch=2)
        prepared = comfy.sampler_helpers.prepare_mask(output, (2, 24, 37, 2, 3), "cpu")
        torch.testing.assert_close(prepared, output.expand_as(prepared), rtol=0, atol=0)

    def test_preserves_samples_metadata_and_input_mask(self):
        video = torch.zeros(1, 24, 2, 2, 3)
        old_mask = torch.ones(1)
        metadata = {"seed": 42}
        samples = {"samples": video, "noise_mask": old_mask, "metadata": metadata}
        mask = torch.zeros(2, 2, 3)
        result = SetVideoLatentNoiseMask.execute(samples, mask, "minimax").result[0]
        self.assertIs(result["samples"], video)
        self.assertIs(result["metadata"], metadata)
        self.assertIs(samples["noise_mask"], old_mask)
        result["noise_mask"].fill_(1)
        self.assertEqual(mask.count_nonzero().item(), 0)

    def test_noncontiguous_and_supported_mask_dtypes(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.bool,
        ):
            with self.subTest(dtype=dtype):
                mask = torch.ones(2, 3, 2, dtype=dtype).transpose(-1, -2)
                output = self.apply(mask, 2)
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.all(output == 1))

    def test_rejects_incompatible_frame_counts_instead_of_resizing(self):
        for count in (1, 5, 36, 38, 123, 125):
            with self.subTest(count=count):
                with self.assertRaisesRegex(
                    ValueError,
                    "Expected 37 latent-frame masks or 124 image-frame masks",
                ):
                    self.apply(torch.zeros(count, 2, 3), 37)
        with self.assertRaisesRegex(
            ValueError, "Expected 3 latent-frame masks or 9 image-frame masks"
        ):
            self.apply(torch.zeros(8, 2, 3), 3, "wan")

    def test_rejects_noncanonical_h3_image_mapping(self):
        with self.assertRaisesRegex(ValueError, r"1 or 5\*n\+2"):
            self.apply(torch.zeros(17, 2, 3), 5)

    def test_causal_profiles_reject_partial_or_extra_image_frames(self):
        for model_type, stride in CAUSAL_PROFILES:
            frames = 5
            pixel_frames = 1 + stride * (frames - 1)
            for count in (
                1,
                frames - 1,
                frames + 1,
                pixel_frames - 1,
                pixel_frames + 1,
            ):
                with self.subTest(model_type=model_type, count=count):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"Expected {frames} latent-frame masks or {pixel_frames} image-frame masks",
                    ):
                        self.apply(torch.zeros(count, 2, 3), frames, model_type)

    def test_selected_type_controls_image_mapping_not_latent_shape(self):
        mask = torch.zeros(17, 2, 3)
        self.assertEqual(self.apply(mask, 3, "ltxv").shape, (1, 1, 3, 2, 3))
        for model_type in ("wan", "hunyuan_video", "hunyuan_video_15", "mochi"):
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "Received 17 mask frames"):
                    self.apply(mask, 3, model_type)

    def test_rejects_unknown_profile_even_when_counts_match(self):
        with self.assertRaisesRegex(ValueError, "Unknown video mask"):
            self.apply(torch.zeros(2, 2, 3), 2, "unknown")

    def test_rejects_invalid_mask_shapes_and_batches(self):
        for mask in (torch.zeros(2, 3), torch.zeros(1, 1, 2, 2, 3)):
            with self.subTest(shape=getattr(mask, "shape", None)):
                with self.assertRaisesRegex(ValueError, "Expected MASK"):
                    self.apply(mask, 2)
        with self.assertRaisesRegex(ValueError, "MASK batch"):
            self.apply(torch.zeros(3, 2, 2, 3), 2, batch=2)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.apply(torch.zeros(0, 2, 3), 2)

    def test_rejects_out_of_range_nonfinite_and_complex_masks(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"finite and within \[0,1\]"):
                    self.apply(torch.full((2, 2, 3), value), 2)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            self.apply(torch.zeros(2, 2, 3, dtype=torch.complex64), 2)

    def test_rejects_audio_images_and_empty_video(self):
        for video in (torch.zeros(1, 32, 2, 9), torch.zeros(1, 4, 2, 3), None):
            with self.subTest(shape=getattr(video, "shape", None)):
                with self.assertRaisesRegex(ValueError, "video latent shaped"):
                    SetVideoLatentNoiseMask.execute(
                        {"samples": video}, torch.ones(2, 2, 3), "minimax"
                    )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            SetVideoLatentNoiseMask.execute(
                {"samples": torch.zeros(1, 24, 0, 2, 3)}, torch.ones(2, 2, 3), "minimax"
            )

    def test_rejects_nested_av_with_actionable_error(self):
        av = comfy.nested_tensor.NestedTensor(
            (torch.zeros(1, 24, 2, 2, 3), torch.zeros(1, 32, 2, 9))
        )
        for model_type in MASK_TYPES:
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "Separate AV Latent"):
                    SetVideoLatentNoiseMask.execute(
                        {"samples": av}, torch.ones(2, 2, 3), model_type
                    )


if __name__ == "__main__":
    unittest.main()
