from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.nested_tensor  # noqa: E402

from comfyui_turing_utils.adapters.minimax.latent_upscaler import (  # noqa: E402
    H3LatentResizer3D,
    _resize_video_mask,
    detect_h3_latent_upscaler_architecture,
    load_h3_latent_upscaler,
)
from comfyui_turing_utils.nodes.latent import (
    SetVideoLatentNoiseMask,
    VideoLatentCompositeMasked,
)  # noqa: E402
from comfyui_turing_utils.nodes.minimax import (  # noqa: E402
    _H3UpscaleApply as MiniMaxH3LatentUpscale,
    MiniMaxH3LatentUpscale as UnifiedLatentUpscale,
    _H3UpscaleLoader,
)


class _FakeUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = SimpleNamespace(out_channels=32)
        self.calls = []

    def forward(self, latent, scale, target_size):
        self.calls.append((tuple(latent.shape), float(scale), tuple(target_size)))
        return F.interpolate(
            latent, size=target_size, mode="trilinear", align_corners=False
        )


class _FakePatcher:
    def __init__(self):
        self.model = _FakeUpscaler()
        self.load_device = torch.device("cpu")

    @staticmethod
    def model_dtype():
        return torch.float32


class MiniMaxH3LatentUpscaleTest(unittest.TestCase):
    def test_mask_resize_matches_positive_area_intersections(self):
        for source_size, target_size in (
            ((2, 6), (4, 10)),
            ((3, 5), (2, 2)),
            ((4, 6), (6, 8)),
            ((2, 3), (4, 6)),
        ):
            sh, sw = source_size
            th, tw = target_size
            for sy in range(sh):
                for sx in range(sw):
                    with self.subTest(
                        source=source_size, target=target_size, cell=(sy, sx)
                    ):
                        mask = torch.zeros(1, 1, 2, sh, sw)
                        mask[0, 0, 1, sy, sx] = 1
                        actual = _resize_video_mask(mask, th, tw)
                        expected = torch.zeros(1, 1, 2, th, tw)
                        for dy in range(th):
                            for dx in range(tw):
                                overlaps = (
                                    sy * th < (dy + 1) * sh
                                    and (sy + 1) * th > dy * sh
                                    and sx * tw < (dx + 1) * sw
                                    and (sx + 1) * tw > dx * sw
                                )
                                expected[0, 0, 1, dy, dx] = overlaps
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mask_resize_retains_soft_maxima_dtype_and_batch_time(self):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            for dtype in (
                torch.bool,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ):
                with self.subTest(device=device, dtype=dtype):
                    mask = torch.zeros(2, 1, 2, 2, 6, dtype=dtype, device=device)
                    mask[1, 0, 1, 1, 2] = True if dtype == torch.bool else 0.25
                    actual = _resize_video_mask(mask, 4, 10)
                    expected = torch.zeros(2, 1, 2, 4, 10, dtype=dtype, device=device)
                    expected[1, 0, 1, 2:4, 3:5] = mask[1, 0, 1, 1, 2]
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.device, mask.device)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @mock.patch("comfy.model_management.load_models_gpu")
    def test_low_resolution_mask_upscale_composite_pipeline(self, _load_models_gpu):
        low = {"samples": torch.randn(1, 24, 2, 2, 6)}
        input_mask = torch.zeros(5, 4, 12)
        input_mask[4, 3, 4] = 0.25
        low = SetVideoLatentNoiseMask.execute(low, input_mask, "minimax").result[0]
        replacement = MiniMaxH3LatentUpscale.execute(
            _FakePatcher(), low, scale=2.5
        ).result[0]
        self.assertEqual(replacement["samples"].shape, (1, 24, 2, 4, 10))
        original = {
            "samples": torch.randn_like(replacement["samples"]),
            "noise_mask": torch.ones(1, 1, 2, 4, 10),
        }
        additional = torch.zeros(5, 8, 20)
        additional[0, 0, 0] = 1
        output = VideoLatentCompositeMasked.execute(
            original, replacement, mask=additional
        ).result[0]
        expected = torch.zeros(1, 1, 2, 4, 10)
        expected[0, 0, 0, 0, 0] = 1
        expected[0, 0, 1, 2:4, 3:5] = 1
        torch.testing.assert_close(output["noise_mask"], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            output["samples"],
            torch.where(expected.bool(), replacement["samples"], original["samples"]),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(original["noise_mask"] == 1))
        self.assertEqual(low["noise_mask"].count_nonzero().item(), 1)

    @staticmethod
    def _tiny_model():
        model = H3LatentResizer3D(
            in_channels=24,
            in_blocks=3,
            out_blocks=3,
            channels=32,
            temporal_every=2,
            temporal_kernel=3,
            dtype=torch.float32,
        ).eval()
        for parameter in model.parameters():
            nn.init.uniform_(parameter, -0.02, 0.02)
        return model

    def test_schema_exposes_conditioning_and_multiplier(self):
        loader = _H3UpscaleLoader.define_schema()
        upscale = UnifiedLatentUpscale.define_schema()
        self.assertEqual(loader.node_id, "_TuringUtilsH3UpscaleLoader")
        self.assertEqual(upscale.node_id, "TuringUtilsMiniMaxH3LatentUpscale")
        self.assertEqual(
            [item.id for item in upscale.inputs],
            [
                "model_name",
                "precision",
                "latent",
                "conditioning",
                "scale",
            ],
        )
        self.assertTrue(upscale.inputs[3].optional)
        self.assertTrue(loader.is_dev_only)

    @mock.patch("comfy.model_management.load_models_gpu")
    def test_fl2av_upscales_video_keyframes_and_video_mask(self, load_models_gpu):
        patcher = _FakePatcher()
        video = torch.randn(1, 24, 3, 4, 6)
        audio = torch.randn(1, 32, 2, 11)
        video_mask = torch.zeros_like(video)
        audio_mask = torch.ones_like(audio)
        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "metadata": "preserved",
        }
        first = torch.randn(1, 24, 1, 4, 6)
        last = torch.randn(1, 24, 1, 4, 6)
        refs = [
            {
                "kind": "image",
                "latent_h": 3,
                "latent_w": 5,
                "latent": torch.randn(1, 24, 1, 3, 5),
            }
        ]
        options = {
            "minimax_keyframes": [
                {"resolved_frame_index": 0, "latent": first},
                {"resolved_frame_index": 124, "latent": last},
            ],
            "minimax_refs": refs,
            "layout": object(),
        }
        embedding = torch.randn(1, 5, 8)
        conditioning = [[embedding, options]]

        output_latent, output_conditioning, width, height = (
            MiniMaxH3LatentUpscale.execute(
                patcher,
                latent,
                conditioning,
                2.0,
            ).result
        )

        output_video, output_audio = output_latent["samples"].unbind()
        output_video_mask, output_audio_mask = output_latent["noise_mask"].unbind()
        self.assertEqual(tuple(output_video.shape), (1, 24, 3, 6, 8))
        self.assertIs(output_audio, audio)
        self.assertEqual(tuple(output_video_mask.shape), (1, 24, 3, 6, 8))
        self.assertIs(output_audio_mask, audio_mask)
        self.assertEqual(output_latent["metadata"], "preserved")
        self.assertEqual((width, height), (128, 96))

        synced_options = output_conditioning[0][1]
        self.assertIs(output_conditioning[0][0], embedding)
        self.assertEqual(
            tuple(synced_options["minimax_keyframes"][0]["latent"].shape),
            (1, 24, 1, 6, 8),
        )
        self.assertEqual(
            tuple(synced_options["minimax_keyframes"][1]["latent"].shape),
            (1, 24, 1, 6, 8),
        )
        self.assertIs(synced_options["minimax_refs"], refs)
        self.assertNotIn("layout", synced_options)

        self.assertIs(latent["samples"].unbind()[0], video)
        self.assertIs(options["minimax_keyframes"][0]["latent"], first)
        self.assertIn("layout", options)
        self.assertEqual(
            len(patcher.model.calls), 2
        )  # main video and batched keyframes
        self.assertAlmostEqual(patcher.model.calls[0][1], 2.0**0.5)
        load_models_gpu.assert_called_once()

    @mock.patch("comfy.model_management.load_models_gpu")
    def test_ref2av_references_keep_independent_geometry(self, load_models_gpu):
        patcher = _FakePatcher()
        video = torch.randn(1, 24, 2, 4, 6)
        audio = torch.randn(1, 32, 2, 8)
        reference = torch.randn(1, 24, 7, 10, 14)
        refs = [
            {
                "kind": "video",
                "latent_t": 7,
                "latent_h": 10,
                "latent_w": 14,
                "latent": reference,
                "ref_audio_t": 0,
                "audio_latent": None,
            }
        ]
        conditioning = [[torch.randn(1, 3, 4), {"minimax_refs": refs}]]

        output_latent, output_conditioning, width, height = (
            MiniMaxH3LatentUpscale.execute(
                patcher,
                {"samples": comfy.nested_tensor.NestedTensor((video, audio))},
                conditioning,
                1.5,
            ).result
        )

        output_video, output_audio = output_latent["samples"].unbind()
        self.assertEqual(tuple(output_video.shape), (1, 24, 2, 6, 8))
        self.assertIs(output_audio, audio)
        self.assertIs(output_conditioning[0][1], conditioning[0][1])
        self.assertIs(output_conditioning[0][1]["minimax_refs"][0]["latent"], reference)
        self.assertEqual((width, height), (128, 96))
        self.assertEqual(len(patcher.model.calls), 1)
        load_models_gpu.assert_called_once()

    @mock.patch("comfy.model_management.load_models_gpu")
    def test_scale_one_is_an_exact_passthrough(self, load_models_gpu):
        patcher = _FakePatcher()
        latent = {"samples": torch.randn(1, 24, 1, 4, 6)}
        conditioning = [[torch.randn(1, 2, 3), {}]]
        output = MiniMaxH3LatentUpscale.execute(
            patcher,
            latent,
            conditioning,
            1.0,
        ).result
        self.assertIs(output[0], latent)
        self.assertIs(output[1], conditioning)
        self.assertEqual(output[2:], (96, 64))
        load_models_gpu.assert_not_called()

    @mock.patch("comfy.model_management.load_models_gpu")
    def test_unconnected_conditioning_only_upscales_the_latent(self, load_models_gpu):
        patcher = _FakePatcher()
        video = torch.randn(1, 24, 2, 4, 6)

        output_latent, output_conditioning, width, height = (
            MiniMaxH3LatentUpscale.execute(
                upscale_model=patcher,
                latent={"samples": video},
                scale=1.5,
            ).result
        )

        self.assertEqual(tuple(output_latent["samples"].shape), (1, 24, 2, 6, 8))
        self.assertIsNone(output_conditioning)
        self.assertEqual((width, height), (128, 96))
        self.assertEqual(len(patcher.model.calls), 1)
        load_models_gpu.assert_called_once()

    def test_detected_3d_architecture_round_trips_numerically(self):
        torch.manual_seed(1)
        source = self._tiny_model()
        state_dict = source.state_dict()
        architecture = detect_h3_latent_upscaler_architecture(state_dict)
        self.assertEqual(architecture.in_blocks, 3)
        self.assertEqual(architecture.out_blocks, 3)
        self.assertEqual(architecture.channels, 32)
        self.assertEqual(architecture.temporal_every, 2)
        self.assertEqual(architecture.temporal_kernel, 3)

        restored = H3LatentResizer3D(
            **architecture.__dict__,
            device="meta",
            dtype=torch.float32,
        ).eval()
        restored.load_state_dict(state_dict, strict=True, assign=True)
        latent = torch.randn(1, 24, 2, 4, 6)
        expected = source(latent, 1.5, (2, 6, 10))
        actual = restored(latent, 1.5, (2, 6, 10))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_loader_builds_a_comfy_managed_patcher(self):
        state_dict = self._tiny_model().state_dict()
        with (
            mock.patch(
                "folder_paths.get_full_path_or_raise", return_value="model.safetensors"
            ),
            mock.patch("comfy.utils.load_torch_file", return_value=state_dict),
            mock.patch(
                "comfy.model_management.get_torch_device",
                return_value=torch.device("cpu"),
            ),
            mock.patch(
                "comfy.model_management.unet_offload_device",
                return_value=torch.device("cpu"),
            ),
        ):
            patcher = load_h3_latent_upscaler("model.safetensors", "fp32")
        self.assertEqual(patcher.load_device, torch.device("cpu"))
        self.assertEqual(patcher.offload_device, torch.device("cpu"))
        self.assertEqual(patcher.model_dtype(), torch.float32)
        self.assertFalse(
            any(parameter.is_meta for parameter in patcher.model.parameters())
        )

    def test_rejects_mismatched_fl2av_keyframe_geometry(self):
        patcher = _FakePatcher()
        conditioning = [
            [
                torch.randn(1, 2, 3),
                {
                    "minimax_keyframes": [{"latent": torch.randn(1, 24, 1, 2, 3)}],
                },
            ]
        ]
        with self.assertRaisesRegex(ValueError, "must match the source video latent"):
            MiniMaxH3LatentUpscale.execute(
                patcher,
                {"samples": torch.randn(1, 24, 2, 4, 6)},
                conditioning,
                2.0,
            )


if __name__ == "__main__":
    unittest.main()
