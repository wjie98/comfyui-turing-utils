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


from comfy_extras import nodes_lt as minimax_nodes  # noqa: E402


class MiniMaxH3AVLatentTest(unittest.TestCase):
    def test_official_schema_provides_av_nodes(self):
        concat = minimax_nodes.LTXVConcatAVLatent.define_schema()
        separate = minimax_nodes.LTXVSeparateAVLatent.define_schema()
        self.assertEqual(concat.node_id, "LTXVConcatAVLatent")
        self.assertEqual(separate.node_id, "LTXVSeparateAVLatent")

    def test_concat_and_separate_round_trip(self):
        video = torch.randn(2, 24, 7, 8, 12)
        audio = torch.randn(2, 32, 2, 19)
        video_mask = torch.zeros_like(video)
        audio_mask = torch.ones_like(audio)
        video_latent = {"samples": video, "noise_mask": video_mask, "video_metadata": 1}
        audio_latent = {"samples": audio, "noise_mask": audio_mask, "audio_metadata": 2}

        av_latent = minimax_nodes.LTXVConcatAVLatent.execute(
            video_latent, audio_latent
        ).result[0]
        av_video, av_audio = av_latent["samples"].unbind()
        av_video_mask, av_audio_mask = av_latent["noise_mask"].unbind()
        self.assertIs(av_video, video)
        self.assertIs(av_audio, audio)
        self.assertIs(av_video_mask, video_mask)
        self.assertIs(av_audio_mask, audio_mask)
        self.assertEqual(av_latent["video_metadata"], 1)
        self.assertEqual(av_latent["audio_metadata"], 2)

        separated_video, separated_audio = minimax_nodes.LTXVSeparateAVLatent.execute(
            av_latent
        ).result
        self.assertIs(separated_video["samples"], video)
        self.assertIs(separated_audio["samples"], audio)
        self.assertIs(separated_video["noise_mask"], video_mask)
        self.assertIs(separated_audio["noise_mask"], audio_mask)

    def test_concat_fills_a_missing_stream_mask(self):
        video = torch.randn(1, 24, 7, 8, 12)
        audio = torch.randn(1, 32, 2, 19)
        audio_mask = torch.zeros_like(audio)

        output = minimax_nodes.LTXVConcatAVLatent.execute(
            {"samples": video},
            {"samples": audio, "noise_mask": audio_mask},
        ).result[0]
        video_mask, output_audio_mask = output["noise_mask"].unbind()
        self.assertTrue(torch.equal(video_mask, torch.ones_like(video)))
        self.assertIs(output_audio_mask, audio_mask)

    def test_replacing_existing_audio_fits_length_and_mask(self):
        video = torch.randn(1, 24, 7, 8, 12)
        original_audio = torch.randn(1, 32, 2, 7)
        replacement_audio = torch.randn(1, 32, 2, 5)
        replacement_mask = torch.zeros_like(replacement_audio)
        existing = {
            "samples": comfy.nested_tensor.NestedTensor((video, original_audio)),
        }

        output = minimax_nodes.LTXVConcatAVLatent.execute(
            existing,
            {"samples": replacement_audio, "noise_mask": replacement_mask},
        ).result[0]
        output_video, output_audio = output["samples"].unbind()
        video_mask, audio_mask = output["noise_mask"].unbind()
        self.assertIs(output_video, video)
        self.assertEqual(tuple(output_audio.shape), tuple(original_audio.shape))
        self.assertTrue(torch.equal(output_audio[..., :5], replacement_audio))
        self.assertEqual(int(torch.count_nonzero(output_audio[..., 5:])), 0)
        self.assertTrue(torch.equal(video_mask, torch.ones_like(video)))
        self.assertTrue(torch.equal(audio_mask[..., :5], replacement_mask))
        self.assertTrue(
            torch.equal(audio_mask[..., 5:], torch.ones_like(audio_mask[..., 5:]))
        )


if __name__ == "__main__":
    unittest.main()
