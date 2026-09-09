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

from comfyui_turing_utils.adapters.minimax.conditioning import (  # noqa: E402
    repair_combined_minimax_payload,
)
from comfyui_turing_utils.nodes.minimax_references import (  # noqa: E402
    H3BuildConditioning,
    H3_MAX_KEYFRAME_REFERENCES,
    H3KeyframeReference,
    H3KeyframeReferenceData,
    H3ImageReference,
    H3ImageReferenceData,
    H3LatentInfo,
    H3ReferenceManifest,
    H3SemanticReference,
    H3SemanticReferenceData,
    H3VideoReference,
    _prepare_h3_semantic_encoder,
)


class _FakeVideoVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, pixels):
        self.inputs.append(pixels)
        frames, height, width = pixels.shape[:3]
        latent_t = 1 if frames == 1 else ((frames - 5) // 17) * 5 + 2
        return torch.zeros(1, 24, latent_t, height // 16, width // 16)


class _FakeClip:
    def __init__(self):
        self.calls = []
        self.encoded = None

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if kwargs.get("images"):
            entries = [("keyframe", index) for index, _ in enumerate(kwargs["images"])]
        elif kwargs.get("minimax_ref_items"):
            entries = [
                ("reference", item["type"]) for item in kwargs["minimax_ref_items"]
            ]
        else:
            entries = []
        if prompt:
            entries.append(("prompt", prompt))
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded = tokens
        return [[torch.zeros(1, 1, 1), {"semantic": True}]]


class _FakePatcher:
    def __init__(self, loaded_size=0):
        self.load_device = torch.device("cuda:0")
        self._model_size = 15 * 1024**3
        self._loaded_size = loaded_size

    def model_size(self):
        return self._model_size

    def loaded_size(self):
        return self._loaded_size


class _GuardedFakeClip:
    def __init__(self, patcher, on_load):
        self.patcher = patcher
        self.on_load = on_load
        self.loaded_tokens = None

    def load_model(self, tokens):
        self.loaded_tokens = tokens
        self.on_load()


class _FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.inputs = []

    def encode(self, waveform):
        self.inputs.append(waveform)
        return torch.zeros(1, 32, 2, 10)


class _PayloadHolder:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return _PayloadHolder(cond)


class MiniMaxH3ReferencesTest(unittest.TestCase):
    def test_semantic_encoder_guard_evicts_dynamic_residents_for_workspace(self):
        import comfy.model_management as model_management

        gib = 1024**3
        patcher = _FakePatcher(loaded_size=5 * gib)
        loaded_models = []
        free_state = {"bytes": 4 * gib}

        def on_load():
            patcher._loaded_size = patcher._model_size
            free_state["bytes"] = 5 * gib
            loaded_models[:] = [SimpleNamespace(model=patcher)]

        clip = _GuardedFakeClip(patcher, on_load)
        free_calls = []

        def free_memory(target, device, keep_loaded, for_dynamic):
            free_calls.append((target, device, list(keep_loaded), for_dynamic))
            free_state["bytes"] = int(target)
            return []

        with (
            mock.patch.object(
                model_management,
                "current_loaded_models",
                loaded_models,
            ),
            mock.patch.object(
                model_management,
                "get_total_memory",
                return_value=48 * gib,
            ),
            mock.patch.object(
                model_management,
                "get_free_memory",
                side_effect=lambda _device: free_state["bytes"],
            ),
            mock.patch.object(
                model_management,
                "extra_reserved_memory",
                return_value=400 * 1024**2,
            ),
            mock.patch.object(
                model_management,
                "free_memory",
                side_effect=free_memory,
            ),
        ):
            tokens = {"qwen3vl_32b": [[("prompt", "test")]]}
            _prepare_h3_semantic_encoder(clip, tokens)

        self.assertIs(clip.loaded_tokens, tokens)
        self.assertEqual([call[0] for call in free_calls], [16 * gib, 6 * gib])
        self.assertTrue(all(call[3] is False for call in free_calls))
        self.assertEqual(free_calls[0][2], [])
        self.assertIs(free_calls[1][2][0].model, patcher)

    def test_schema_assigns_keyframe_roles_at_semantic_and_build_boundaries(self):
        frame = H3KeyframeReference.define_schema()
        semantic = H3SemanticReference.define_schema()
        build = H3BuildConditioning.define_schema()

        self.assertEqual(frame.display_name, "H3 Keyframe Reference")
        self.assertEqual(
            [item.id for item in frame.inputs],
            [
                "vae",
                "latent",
                "images",
            ],
        )
        self.assertEqual(frame.inputs[2].template.prefix, "image_")
        self.assertEqual(frame.inputs[2].template.min, 0)
        self.assertEqual(frame.inputs[2].template.max, H3_MAX_KEYFRAME_REFERENCES)
        self.assertEqual(
            [item.id for item in frame.outputs[:3]],
            ["keyframe_0", "keyframe_1", "keyframe_2"],
        )
        self.assertEqual(len(frame.outputs), H3_MAX_KEYFRAME_REFERENCES)
        self.assertEqual(
            [item.id for item in semantic.inputs][2:4],
            ["first_frame", "last_frame"],
        )
        self.assertNotIn("keyframes_reference", [item.id for item in semantic.inputs])
        self.assertEqual(
            [item.id for item in build.inputs][1:4],
            ["latent", "first_frame", "last_frame"],
        )
        self.assertNotIn("keyframes_reference", [item.id for item in build.inputs])

    def test_frame_reference_aligns_to_target_latent_canvas(self):
        vae = _FakeVideoVAE()
        latent = {"samples": torch.zeros(1, 24, 2, 6, 8)}
        image = torch.rand(1, 90, 70, 3)

        outputs = H3KeyframeReference.execute(
            vae,
            latent=latent,
            images={"image_0": image},
        ).result
        reference = outputs[0]

        self.assertEqual(tuple(reference.image.shape), (1, 96, 128, 3))
        self.assertEqual(tuple(reference.latent.shape), (1, 24, 1, 6, 8))
        self.assertEqual(len(outputs), H3_MAX_KEYFRAME_REFERENCES)
        self.assertTrue(all(output is None for output in outputs[1:]))

    def test_keyframe_outputs_follow_dynamic_input_order(self):
        vae = _FakeVideoVAE()
        image_2 = torch.rand(1, 64, 96, 3)
        image_10 = torch.rand(1, 96, 128, 3)

        outputs = H3KeyframeReference.execute(
            vae,
            images={"image_10": image_10, "image_2": image_2},
        ).result

        self.assertEqual(tuple(outputs[0].image.shape), (1, 64, 96, 3))
        self.assertEqual(tuple(outputs[1].image.shape), (1, 96, 128, 3))
        self.assertTrue(all(output is None for output in outputs[2:]))

    def test_image_reference_without_latent_uses_uncropped_megapixel_budget(self):
        vae = _FakeVideoVAE()
        image = torch.rand(1, 67, 99, 3)

        reference = H3ImageReference.execute(vae, images={"image_0": image}).result[0]

        self.assertEqual(tuple(reference.items[0]["image"].shape), (1, 64, 96, 3))
        self.assertEqual(tuple(reference.items[0]["latent"].shape), (1, 24, 1, 4, 6))

    def test_image_reference_megapixel_budget_only_downscales_large_inputs(self):
        vae = _FakeVideoVAE()
        image = torch.rand(1, 256, 512, 3)

        reference = H3ImageReference.execute(
            vae,
            megapixels=0.1,
            images={"image_0": image},
        ).result[0]

        self.assertEqual(tuple(reference.items[0]["image"].shape), (1, 224, 448, 3))
        self.assertEqual(tuple(reference.items[0]["latent"].shape), (1, 24, 1, 14, 28))

    def test_image_reference_with_latent_matches_area_without_upscaling(self):
        vae = _FakeVideoVAE()
        latent = {"samples": torch.zeros(1, 24, 2, 6, 8)}
        image = torch.rand(1, 64, 64, 3)

        reference = H3ImageReference.execute(
            vae,
            latent=latent,
            images={"image_0": image},
        ).result[0]

        self.assertEqual(tuple(reference.items[0]["image"].shape), (1, 64, 64, 3))
        self.assertEqual(tuple(reference.items[0]["latent"].shape), (1, 24, 1, 4, 4))

    def test_image_reference_with_latent_preserves_aspect_in_match_mode(self):
        vae = _FakeVideoVAE()
        latent = {"samples": torch.zeros(1, 24, 2, 6, 8)}
        image = torch.rand(1, 256, 256, 3)

        reference = H3ImageReference.execute(
            vae,
            latent=latent,
            images={"image_0": image},
        ).result[0]

        self.assertEqual(tuple(reference.items[0]["image"].shape), (1, 96, 96, 3))
        self.assertEqual(tuple(reference.items[0]["latent"].shape), (1, 24, 1, 6, 6))

    def test_video_reference_accepts_24_fps_and_pairs_audio_by_index(self):
        video_vae = _FakeVideoVAE()
        audio_vae = _FakeAudioVAE()
        frames = torch.rand(31, 70, 100, 3)
        audio = {
            "waveform": torch.rand(1, 2, 32000),
            "sample_rate": 32000,
        }

        self.assertNotIn(
            "source_fps",
            [item.id for item in H3VideoReference.define_schema().inputs],
        )
        self.assertEqual(H3VideoReference.define_schema().inputs[1].id, "megapixels")

        reference = H3VideoReference.execute(
            video_vae,
            audio_vae=audio_vae,
            videos={"video_2": frames},
            video_audios={"video_audio_2": audio},
        ).result[0]
        item = reference.items[0]

        self.assertEqual(tuple(video_vae.inputs[0].shape), (22, 64, 96, 3))
        self.assertEqual(tuple(item["latent"].shape), (1, 24, 7, 4, 6))
        self.assertEqual(tuple(item["qwen_frames"].shape), (2, 64, 96, 3))
        self.assertEqual(item["timestamps"], [0.0, 0.5])
        self.assertEqual(tuple(item["audio_latent"].shape), (1, 32, 2, 10))

    def test_video_reference_with_latent_matches_area_without_upscaling(self):
        video_vae = _FakeVideoVAE()
        latent = {"samples": torch.zeros(1, 24, 2, 6, 8)}
        frames = torch.rand(22, 64, 64, 3)

        reference = H3VideoReference.execute(
            video_vae,
            latent=latent,
            videos={"video_0": frames},
        ).result[0]

        self.assertEqual(tuple(reference.items[0]["qwen_frames"].shape), (2, 64, 64, 3))
        self.assertEqual(tuple(reference.items[0]["latent"].shape), (1, 24, 7, 4, 4))

    def test_semantic_combines_official_keyframe_and_reference_presentations(self):
        clip = _FakeClip()
        keyframe = H3KeyframeReferenceData(
            image=torch.rand(1, 64, 64, 3),
            latent=torch.zeros(1, 24, 1, 4, 4),
        )
        images = H3ImageReferenceData(
            (
                {
                    "image": torch.rand(1, 32, 32, 3),
                    "latent": torch.zeros(1, 24, 1, 2, 2),
                },
            )
        )

        semantic = H3SemanticReference.execute(
            clip,
            "prompt",
            first_frame=keyframe,
            image_reference=images,
        ).result[0]

        self.assertEqual(len(clip.calls), 2)
        self.assertEqual(clip.calls[0][0], "")
        self.assertIn("images", clip.calls[0][1])
        self.assertEqual(clip.calls[1][0], "prompt")
        self.assertIn("minimax_ref_items", clip.calls[1][1])
        self.assertEqual(
            clip.encoded["qwen3vl_32b"][0],
            [("keyframe", 0), ("reference", "image"), ("prompt", "prompt")],
        )
        self.assertEqual(
            semantic.manifest,
            H3ReferenceManifest(first_frame=True, image_count=1),
        )

    def test_build_conditioning_places_keyframe_and_generic_reference_together(self):
        target = {"samples": torch.zeros(1, 24, 7, 6, 8)}
        first_latent = torch.zeros(1, 24, 1, 6, 8)
        keyframe = H3KeyframeReferenceData(
            image=torch.rand(1, 96, 128, 3),
            latent=first_latent,
        )
        images = H3ImageReferenceData(
            (
                {
                    "image": torch.rand(1, 64, 64, 3),
                    "latent": torch.zeros(1, 24, 1, 4, 4),
                },
            )
        )
        base = [[torch.zeros(1, 2, 3), {"kept": True}]]
        semantic = H3SemanticReferenceData(
            base,
            H3ReferenceManifest(first_frame=True, image_count=1),
        )

        conditioning = H3BuildConditioning.execute(
            semantic,
            target,
            first_frame=keyframe,
            image_reference=images,
        ).result[0]
        options = conditioning[0][1]

        self.assertEqual(options["minimax_frame_count"], 22)
        self.assertEqual(options["minimax_keyframes"][0]["resolved_frame_index"], 0)
        self.assertIs(options["minimax_keyframes"][0]["latent"], first_latent)
        self.assertEqual(options["minimax_refs"][0]["kind"], "image")
        self.assertIs(options["minimax_refs"][0]["latent"], images.items[0]["latent"])
        self.assertTrue(options["kept"])
        self.assertNotIn("minimax_keyframes", base[0][1])

    def test_one_keyframe_reference_can_fill_both_roles(self):
        clip = _FakeClip()
        latent = torch.zeros(1, 24, 1, 6, 8)
        keyframe = H3KeyframeReferenceData(
            image=torch.rand(1, 96, 128, 3),
            latent=latent,
        )

        semantic = H3SemanticReference.execute(
            clip,
            "prompt",
            first_frame=keyframe,
            last_frame=keyframe,
        ).result[0]
        conditioning = H3BuildConditioning.execute(
            semantic,
            {"samples": torch.zeros(1, 24, 7, 6, 8)},
            first_frame=keyframe,
            last_frame=keyframe,
        ).result[0]

        self.assertEqual(
            semantic.manifest,
            H3ReferenceManifest(first_frame=True, last_frame=True),
        )
        self.assertEqual(clip.encoded["qwen3vl_32b"][0][:2], [("keyframe", 0), ("keyframe", 1)])
        keyframes = conditioning[0][1]["minimax_keyframes"]
        self.assertEqual([item["resolved_frame_index"] for item in keyframes], [0, 21])
        self.assertIs(keyframes[0]["latent"], latent)
        self.assertIs(keyframes[1]["latent"], latent)

    def test_build_rejects_different_semantic_structure(self):
        semantic = H3SemanticReferenceData(
            [[torch.zeros(1, 1, 1), {}]],
            H3ReferenceManifest(image_count=1),
        )
        with self.assertRaisesRegex(ValueError, "structures differ"):
            H3BuildConditioning.execute(
                semantic,
                {"samples": torch.zeros(1, 24, 2, 4, 4)},
            )

    def test_latent_info_uses_h3_temporal_grid(self):
        result = H3LatentInfo.execute(
            {"samples": torch.zeros(1, 24, 37, 45, 84)}
        ).result
        self.assertEqual(result, (1344, 720, 124, 24.0))

    def test_combined_payload_repair_matches_packed_layout_order(self):
        keyframe = torch.zeros(1, 24, 1, 4, 4)
        image = torch.ones(1, 24, 1, 2, 2)
        audio = torch.ones(1, 32, 2, 5)
        out = {"minimax_payload": _PayloadHolder({"cond_video_latents": [image]})}
        repaired = repair_combined_minimax_payload(
            out,
            {
                "minimax_keyframes": [{"latent": keyframe}],
                "minimax_refs": [
                    {"kind": "image", "latent": image},
                    {"kind": "audio", "audio_latent": audio},
                ],
            },
        )

        payload = repaired["minimax_payload"].cond
        self.assertIs(payload["cond_video_latents"][0], keyframe)
        self.assertIs(payload["cond_video_latents"][1], image)
        self.assertIs(payload["cond_audio_latents"][0], audio)
        self.assertIsNot(repaired, out)


if __name__ == "__main__":
    unittest.main()
