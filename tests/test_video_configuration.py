from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

from comfyui_turing_utils.nodes.video_padding import VideoFramesPadding, padded_frame_count
from comfyui_turing_utils.nodes.latent import VIDEO_MASK_SPECS, SetVideoLatentNoiseMask, _image_frame_groups
from comfyui_turing_utils.nodes.video_sequence import H3SetAudioPrefixNoiseMask, VideoPrefixContextNoise, _add_prefix_chroma_blocks


class VideoConfigurationTest(unittest.TestCase):
    def test_mask_edits_do_not_reencode_with_real_executor(self):
        import execution
        import nodes
        from comfyui_turing_utils.nodes.minimax_vae import MiniMaxH3VideoVAEEncode

        class Inputs:
            RETURN_TYPES = ("IMAGE", "VAE")
            FUNCTION = "run"
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
            def run(self):
                return torch.zeros(6, 8, 8, 3), object()

        class Masks:
            RETURN_TYPES = ("MASK",)
            FUNCTION = "run"
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": ("FLOAT",)}}
            def run(self, value):
                return (torch.full((6, 8, 8), value),)

        class Server:
            client_id = last_node_id = None
            def send_sync(self, *args, **kwargs):
                pass

        mapping = {"TestPreparationInputs": Inputs, "TestPreparationMasks": Masks,
                   "TuringUtilsVideoFramesPadding": VideoFramesPadding,
                   "TuringUtilsMiniMaxH3VideoVAEEncode": MiniMaxH3VideoVAEEncode,
                   "TuringUtilsSetVideoLatentNoiseMask": SetVideoLatentNoiseMask}
        prompt = {
            "source": {"class_type": "TestPreparationInputs", "inputs": {}},
            "masks": {"class_type": "TestPreparationMasks", "inputs": {"value": 0.0}},
            "pad": {"class_type": "TuringUtilsVideoFramesPadding", "inputs": {"image": ["source", 0], "type": "minimax"}},
            "encode": {"class_type": "TuringUtilsMiniMaxH3VideoVAEEncode", "inputs": {"pixels": ["pad", 0], "vae": ["source", 1]}},
            "mpad": {"class_type": "TuringUtilsVideoFramesPadding", "inputs": {"mask": ["masks", 0], "type": "minimax", "target_frame_count": ["pad", 4]}},
            "output": {"class_type": "TuringUtilsSetVideoLatentNoiseMask", "inputs": {"samples": ["encode", 0], "mask": ["mpad", 1], "type": "minimax"}},
        }
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping), mock.patch.object(
            MiniMaxH3VideoVAEEncode, "encode", return_value=({"samples": torch.zeros(1, 24, 7, 2, 2)},)
        ) as encode:
            executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC,
                                                cache_args={"ram": 0, "ram_inactive": 0})
            for index, value in enumerate((0.0, 1.0, 0.5)):
                prompt["masks"]["inputs"]["value"] = value
                executor.execute(prompt, str(index), execute_outputs=["output"])
                self.assertTrue(executor.success, executor.status_messages)
                self.assertEqual(encode.call_count, 1)

    def test_padding_and_mask_mapping_use_same_types_and_counts(self):
        for kind, stride in VIDEO_MASK_SPECS.items():
            for count in range(1, 130):
                with self.subTest(type=kind, count=count):
                    length = padded_frame_count(count, kind)
                    time = ((length - 5) // 17) * 5 + 2 if stride is None else (length - 1) // stride + 1
                    self.assertEqual(sum(_image_frame_groups(kind, time)), length)
                    self.assertGreaterEqual(length, count)
                    self.assertEqual(padded_frame_count(length, kind), length)

    def test_mask_only_padding_keeps_image_dependency_separate(self):
        images = torch.rand(6, 2, 3, 3)
        mask = torch.rand(6, 2, 3)
        a = VideoFramesPadding.execute(image=images).result
        b = VideoFramesPadding.execute(mask=mask, target_frame_count=a[4]).result
        together = VideoFramesPadding.execute(image=images, mask=mask).result
        self.assertIsNone(b[0])
        self.assertTrue(torch.equal(a[0], together[0]))
        self.assertTrue(torch.equal(b[1], together[1]))
        self.assertEqual(a[4:], (22, 6))
        self.assertTrue(torch.equal(a[0][-1], images[-1]))
        self.assertEqual(VideoFramesPadding.execute().result, (None, None, 0, 0, 0, 0))

    def test_invalid_grid_and_truncation_are_not_silent(self):
        for target in (4, 6, 21):
            with self.assertRaises(ValueError):
                padded_frame_count(6, "minimax", target)
        with self.assertRaises(ValueError):
            VideoFramesPadding.execute(type="unknown")

    def test_missing_mask_preserves_latent_metadata(self):
        samples = {"samples": torch.zeros(1, 24, 2, 2, 2), "noise_mask": torch.ones(1, 1, 2, 2, 2)}
        result = SetVideoLatentNoiseMask.execute(samples).result[0]
        self.assertIsNot(result, samples)
        self.assertIs(result["samples"], samples["samples"])
        self.assertIs(result["noise_mask"], samples["noise_mask"])

    def test_whole_audio_policies_do_not_need_trim_metadata(self):
        samples = {"samples": torch.rand(1, 32, 2, 16)}
        for policy, value in (("protect_all", 0), ("generate_all", 1)):
            for info in (None, {}):
                out = H3SetAudioPrefixNoiseMask.execute(samples, info, policy).result[0]
                self.assertTrue(torch.all(out["noise_mask"] == value))
                self.assertIs(out["samples"], samples["samples"])
        with self.assertRaisesRegex(ValueError, "trim_info"):
            H3SetAudioPrefixNoiseMask.execute(samples)

    def test_noise_dynamic_grid_is_numerically_identical(self):
        images = torch.rand(22, 9, 13, 3)
        for pattern in ("poc_chroma_blocks", "gaussian_rgb", "uniform_rgb"):
            for grid in ("poc_36x64", "block_size"):
                old = _add_prefix_chroma_blocks(images, 0.45, 0, end_strength=0.1,
                    transition_frames=4, tail_protection_frames=5, pattern=pattern,
                    grid_mode=grid, block_size=3)
                new = VideoPrefixContextNoise.execute(images, pattern=pattern, grid_mode={"grid_mode": grid, "block_size": 3}).result[0]
                self.assertTrue(torch.equal(old, new))
        # Omitted optional branch input retains its declared default.
        old = VideoPrefixContextNoise.execute(images, grid_mode={"grid_mode": "block_size", "block_size": 16}).result[0]
        new = VideoPrefixContextNoise.execute(images, grid_mode={"grid_mode": "block_size"}).result[0]
        self.assertTrue(torch.equal(old, new))
