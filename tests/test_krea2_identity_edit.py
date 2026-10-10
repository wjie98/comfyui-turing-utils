from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.krea2 import (  # noqa: E402
    build_identity_edit_conditioning,
)
from comfyui_turing_utils.nodes.krea2 import (  # noqa: E402
    Krea2IdentityEditConditioning,
)


class FakeModel:
    def __init__(self):
        self.post_input_patch = None
        self.attn1_patch = None
        self.wrappers = []

    def clone(self):
        return FakeModel()

    def set_model_post_input_patch(self, patch):
        self.post_input_patch = patch

    def set_model_attn1_patch(self, patch):
        self.attn1_patch = patch

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))


class FakeVAE:
    def __init__(self):
        self.images = []

    def encode(self, image):
        self.images.append(image)
        batch, height, width, _ = image.shape
        return torch.full(
            (batch, 16, height // 8, width // 8),
            float(image.mean()),
        )


class FakeClip:
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls.append((prompt, kwargs))
        return {"prompt": prompt}

    @staticmethod
    def encode_from_tokens_scheduled(tokens):
        return [[torch.zeros(1), {"tokens": tokens}]]


def image(value: float, height: int, width: int) -> torch.Tensor:
    return torch.full((1, height, width, 3), value)


def image_ids(frame: int, height: int, width: int) -> torch.Tensor:
    ids = torch.zeros(height, width, 3)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(height)[:, None]
    ids[..., 2] = torch.arange(width)[None, :]
    return ids.reshape(1, height * width, 3)


class Krea2IdentityEditTest(unittest.TestCase):
    def test_schema_is_one_conditioning_without_negative_or_latent_output(self):
        schema = Krea2IdentityEditConditioning.define_schema()
        inputs = [item.id for item in schema.inputs]

        self.assertEqual(
            inputs,
            [
                "model",
                "clip",
                "vae",
                "target_latent",
                "character_image",
                "prompt",
                "character_strength",
                "background_strength",
                "grounding_px",
                "background_image",
            ],
        )
        self.assertEqual(len(schema.outputs), 2)
        self.assertNotIn("negative_prompt", inputs)
        self.assertNotIn("negative", [output.id for output in schema.outputs])

    def test_background_precedes_character_in_every_conditioning_path(self):
        vae = FakeVAE()
        clip = FakeClip()
        latent = {"samples": torch.zeros(1, 16, 64, 64)}

        patched, conditioning = build_identity_edit_conditioning(
            FakeModel(),
            clip,
            vae,
            latent,
            image(0.75, 256, 512),
            "place the character in the room",
            background_image=image(0.25, 512, 512),
        )

        self.assertEqual(
            [round(float(value.mean()), 2) for value in vae.images], [0.25, 0.75]
        )
        self.assertEqual(
            [tuple(value.shape[1:3]) for value in vae.images], [(512, 512), (256, 512)]
        )
        grounded = clip.tokenize_calls[0][1]["images"]
        self.assertEqual(
            [round(float(value.mean()), 2) for value in grounded], [0.25, 0.75]
        )

        metadata = conditioning[0][1]
        references = metadata["reference_latents"]
        self.assertEqual(
            [round(float(value.mean()), 2) for value in references], [0.25, 0.75]
        )
        self.assertEqual(metadata["reference_latents_method"], "index")
        self.assertIsNotNone(patched.post_input_patch)

    def test_position_patch_centers_fit_references_on_the_target_grid(self):
        vae = FakeVAE()
        latent = {"samples": torch.zeros(1, 16, 64, 64)}
        patched, _ = build_identity_edit_conditioning(
            FakeModel(),
            FakeClip(),
            vae,
            latent,
            image(0.75, 256, 512),
            "restage",
            background_image=image(0.25, 512, 512),
        )

        target = image_ids(0, 32, 32)
        background = image_ids(1, 32, 32)
        character = image_ids(2, 16, 32)
        ids = torch.cat((target, background, character), dim=1)
        inputs = {
            "img_ids": ids,
            "transformer_options": {
                "reference_image_num_tokens": [32 * 32, 16 * 32],
            },
        }
        patched.post_input_patch(inputs)

        background_ids = ids[:, 32 * 32 : 2 * 32 * 32]
        character_ids = ids[:, 2 * 32 * 32 :]
        self.assertEqual(float(background_ids[..., 1].min()), 0.0)
        self.assertEqual(float(background_ids[..., 2].min()), 0.0)
        self.assertEqual(float(character_ids[..., 1].min()), 8.0)
        self.assertEqual(float(character_ids[..., 1].max()), 23.0)
        self.assertEqual(float(character_ids[..., 2].min()), 0.0)

    def test_reference_strength_bias_only_changes_target_to_reference_attention(self):
        patched, _ = build_identity_edit_conditioning(
            FakeModel(),
            FakeClip(),
            FakeVAE(),
            {"samples": torch.zeros(1, 16, 8, 8)},
            image(0.75, 32, 64),
            "restage",
            background_image=image(0.25, 64, 64),
            character_strength=4.0,
            background_strength=1.0,
        )

        self.assertIsNotNone(patched.attn1_patch)
        self.assertEqual(len(patched.wrappers), 1)
        wrapper = patched.wrappers[0][2]
        captured = {}

        def executor(x, timesteps, context, attention_mask, ref_latents, options):
            q = torch.zeros(1, 1, 42, 4)
            captured.update(
                patched.attn1_patch(
                    q,
                    q,
                    q,
                    extra_options=options,
                )
            )
            return "ok"

        result = wrapper(
            executor,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            None,
            None,
            {},
        )

        self.assertEqual(result, "ok")
        bias = captured["attn_mask"]
        self.assertEqual(tuple(bias.shape), (1, 1, 42, 42))
        self.assertTrue(torch.count_nonzero(bias[:, :, :2]).eq(0))
        self.assertTrue(torch.count_nonzero(bias[:, :, 18:]).eq(0))
        self.assertTrue(torch.count_nonzero(bias[:, :, 2:18, 18:34]).eq(0))
        expected = torch.full((16, 8), math.log(4.0))
        self.assertTrue(torch.allclose(bias[0, 0, 2:18, 34:42], expected))

    def test_unit_strength_keeps_native_attention_unpatched(self):
        patched, _ = build_identity_edit_conditioning(
            FakeModel(),
            FakeClip(),
            FakeVAE(),
            {"samples": torch.zeros(1, 16, 8, 8)},
            image(0.75, 64, 64),
            "restage",
            character_strength=1.0,
        )

        self.assertIsNone(patched.attn1_patch)
        self.assertEqual(patched.wrappers, [])

    def test_character_only_is_one_reference(self):
        conditioning = build_identity_edit_conditioning(
            FakeModel(),
            FakeClip(),
            FakeVAE(),
            {"samples": torch.zeros(1, 16, 64, 64)},
            image(0.75, 512, 512),
            "change the lighting",
        )[1]

        self.assertEqual(len(conditioning[0][1]["reference_latents"]), 1)


if __name__ == "__main__":
    unittest.main()
