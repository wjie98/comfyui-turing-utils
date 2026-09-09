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

import attention  # noqa: E402
from comfyui_turing_utils.adapters.minimax import layout as minimax_layout  # noqa: E402
from comfyui_turing_utils.attention.layout import (  # noqa: E402
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    attention_semantic_layout,
)


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_options = None

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        self.seen_options = transformer_options
        return x


class FakeMiniMaxDiffusion(torch.nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self, blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(FakeBlock() for _ in range(blocks))
        self.forward_callback = None

    def forward(
        self,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        **kwargs,
    ):
        if self.forward_callback is not None:
            return self.forward_callback()
        return x


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = FakeMiniMaxDiffusion()
        self.latent_shapes = None


class FakePatcher:
    def __init__(self, base=None):
        self.model = base if base is not None else FakeBase()
        self.load_device = torch.device("cuda", 0)
        self.model_options = {"transformer_options": {"existing": True}}
        self.object_patches = {}
        self.wrappers = {}

    def clone(self):
        cloned = FakePatcher(self.model)
        cloned.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        cloned.object_patches = self.object_patches.copy()
        cloned.wrappers = self.wrappers.copy()
        return cloned

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def add_wrapper_with_key(self, wrapper_type, key, value):
        self.wrappers[(wrapper_type, key)] = value


class MiniMaxLayoutProviderTest(unittest.TestCase):
    @staticmethod
    def _latent_shapes():
        return [
            torch.Size((1, 24, 7, 8, 10)),
            torch.Size((1, 32, 2, 12)),
        ]

    def _minimax_type_patch(self):
        import comfy.ldm.minimax.model as minimax_model

        return mock.patch.object(
            minimax_model,
            "MiniMaxH3Model",
            FakeMiniMaxDiffusion,
        )

    def test_official_model_patcher_gets_runtime_layout_without_custom_loader(self):
        patcher = FakePatcher()
        with self._minimax_type_patch():
            status = minimax_layout.ensure_minimax_attention_layout_provider(patcher)

        self.assertTrue(status.installed)
        self.assertEqual(status.model_kind, minimax_layout.MINIMAX_H3_LAYOUT_KIND)
        self.assertEqual(len(patcher.object_patches), 3)
        self.assertEqual(len(patcher.wrappers), 1)

        runtime_wrapper = next(iter(patcher.wrappers.values()))
        block_forward = patcher.object_patches["diffusion_model.blocks.1.forward"]
        model_forward = patcher.object_patches["diffusion_model.forward"]
        options = {}
        x = torch.zeros((228, 8), dtype=torch.bfloat16)
        model_input = [
            torch.zeros((1, 24, 7, 8, 10), dtype=torch.bfloat16),
            torch.zeros((1, 32, 2, 12), dtype=torch.bfloat16),
        ]
        context = torch.zeros((1, 64, 8), dtype=torch.bfloat16)

        def executor(*args, **kwargs):
            patcher.model.diffusion_model.forward_callback = lambda: block_forward(
                x, x, [(0, 64, 0), (64, 88, 2), (88, 228, 3)], None,
                transformer_options=options,
            )
            packed_layout = SimpleNamespace(
                signature=(64, 7, 8, 10, 12),
                segments=[
                    (0, 64, "text"),
                    (64, 88, "audio"),
                    (88, 228, "video"),
                ]
            )
            return model_forward(
                model_input, None, context, transformer_options=options,
                minimax_payload={"layout": packed_layout, "refs": []},
            )

        output = runtime_wrapper(
            executor,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            latent_shapes=self._latent_shapes(),
        )

        self.assertIs(output, x)
        wire = options[minimax_layout.ATTENTION_LAYOUT_KEY]
        self.assertEqual(wire["protocol_version"], 1)
        self.assertEqual(wire["dense_prefix_tokens"], 88)
        semantic = attention_semantic_layout(options)
        self.assertEqual(semantic.provider, "minimax_h3")
        self.assertEqual(semantic.layer_index, 1)
        self.assertEqual(semantic.layer_count, 2)
        self.assertEqual(
            tuple((item.start, item.stop, item.role) for item in semantic.query_segments),
            (
                (0, 64, "text"),
                (64, 88, "target_audio"),
                (88, 228, "target_video"),
            ),
        )
        self.assertTrue(
            minimax_layout.has_complete_minimax_attention_layout(options, 228)
        )
        self.assertFalse(
            hasattr(patcher.model, minimax_layout.RUNTIME_CONTEXT_ATTR)
        )

    def test_provider_installation_is_keyed_and_idempotent(self):
        patcher = FakePatcher()
        with self._minimax_type_patch():
            first = minimax_layout.ensure_minimax_attention_layout_provider(patcher)
            first_forwards = dict(patcher.object_patches)
            first_wrappers = dict(patcher.wrappers)
            second = minimax_layout.ensure_minimax_attention_layout_provider(patcher)

        self.assertTrue(first.installed)
        self.assertTrue(second.installed)
        self.assertEqual(set(patcher.object_patches), set(first_forwards))
        self.assertEqual(set(patcher.wrappers), set(first_wrappers))
        for key, value in first_forwards.items():
            self.assertIs(patcher.object_patches[key], value)

    def test_runtime_layout_rebuilds_stale_shape_payload(self):
        import comfy.ldm.minimax.model as minimax_model

        diffusion = FakeMiniMaxDiffusion()
        video = torch.zeros((1, 24, 7, 9, 11), dtype=torch.bfloat16)
        audio = torch.zeros((1, 32, 2, 13), dtype=torch.bfloat16)
        context = torch.zeros((1, 48, 8), dtype=torch.bfloat16)
        stale = SimpleNamespace(signature=(48, 7, 8, 10, 13))
        rebuilt = object()

        with mock.patch.object(
            minimax_model, "PackedLayout", return_value=rebuilt
        ) as constructor:
            result = minimax_layout._resolve_packed_layout(
                diffusion,
                [video, audio],
                context,
                {"layout": stale, "refs": [], "keyframes": None},
            )

        self.assertIs(result, rebuilt)
        constructor.assert_called_once_with(
            48,
            7,
            10,
            12,
            13,
            keyframes=None,
            refs=[],
        )

    def test_h3_image_sol_dense_slice_patterns(self):
        self.assertEqual(
            minimax_layout.h3_image_sol_dense_slices(7, "dense_window"),
            (0, 1),
        )
        self.assertEqual(
            minimax_layout.h3_image_sol_dense_slices(
                7, "dense_anchor_grid"
            ),
            (0, 1, 5),
        )
        self.assertEqual(
            minimax_layout.h3_image_sol_dense_slices(
                12, "dense_anchor_grid"
            ),
            (0, 1, 5, 10),
        )
        self.assertEqual(
            minimax_layout.h3_image_sol_dense_slices(3, "dense_window"),
            (0, 1, 2),
        )

    def test_h3_image_sol_publisher_marks_anchor_grid_exact(self):
        base = FakeBase()
        setattr(
            base,
            minimax_layout.RUNTIME_CONTEXT_ATTR,
            {
                "latent_shapes": [
                    torch.Size((1, 24, 7, 8, 10)),
                    torch.Size((1, 32, 2, 12)),
                ],
                "packed_layout": SimpleNamespace(
                    segments=[
                        (0, 64, "text"),
                        (64, 88, "audio"),
                        (88, 228, "video"),
                    ]
                ),
                "refs": [],
            },
        )
        options = {
            "turing_utils_attention_strategy": "h3_image_sol",
            minimax_layout.H3_IMAGE_SOL_LAYOUT_KEY: "dense_anchor_grid",
        }

        self.assertTrue(
            minimax_layout.publish_minimax_attention_layout(
                options,
                [(0, 64, 0), (64, 88, 2), (88, 228, 3)],
                layer_index=0,
                layer_count=2,
                base_model=base,
                diffusion_model=base.diffusion_model,
            )
        )

        semantic = attention_semantic_layout(options)
        target = [
            segment
            for segment in semantic.query_segments
            if segment.role == "target_video"
        ]
        self.assertEqual(len(target), 7)
        self.assertEqual(
            tuple(
                index
                for index, segment in enumerate(target)
                if not segment.sparse_query_allowed
            ),
            (0, 1, 5),
        )
        self.assertEqual(
            tuple(
                index
                for index, segment in enumerate(target)
                if segment.exact_kv
            ),
            (0, 1, 5),
        )

    def test_publisher_drops_stale_topology_when_current_shapes_do_not_validate(self):
        options = {
            minimax_layout.ATTENTION_LAYOUT_KEY: {
                "provider": "minimax_h3",
                "dense_prefix_tokens": 88,
                "topology_start_tokens": 88,
                "topology_tokens": 140,
                "tokens_per_frame": 20,
                "spatial_tokens_height": 4,
                "spatial_tokens_width": 5,
                "layer_index": 1,
                "layer_count": 2,
                "extension_field": "keep",
            }
        }
        base = FakeBase()
        base.latent_shapes = self._latent_shapes()

        published = minimax_layout.publish_minimax_attention_layout(
            options,
            [(0, 64, 0), (64, 200, 3)],
            layer_index=0,
            layer_count=2,
            base_model=base,
            diffusion_model=base.diffusion_model,
        )

        self.assertFalse(published)
        layout = options[minimax_layout.ATTENTION_LAYOUT_KEY]
        self.assertEqual(layout["extension_field"], "keep")
        self.assertEqual(layout["dense_prefix_tokens"], 64)
        self.assertNotIn("topology_tokens", layout)
        self.assertFalse(
            minimax_layout.has_complete_minimax_attention_layout(options, 200)
        )

    def test_full_modality_segments_cover_multiple_references(self):
        base = FakeBase()
        setattr(
            base,
            minimax_layout.RUNTIME_CONTEXT_ATTR,
            {
                "packed_layout": SimpleNamespace(
                    segments=[
                        (0, 32, "text"),
                        (32, 48, "cond"),
                        (48, 64, "ref_img"),
                        (64, 72, "ref_audio"),
                        (72, 112, "ref_img"),
                        (112, 118, "ref_audio"),
                        (118, 130, "audio"),
                        (130, 258, "video"),
                    ]
                ),
                "refs": [
                    {"kind": "image"},
                    {"kind": "video_audio", "ref_audio_t": 4, "latent_t": 4},
                    {"kind": "audio", "ref_audio_t": 3},
                ],
            },
        )

        self.assertEqual(
            minimax_layout.minimax_attention_segments(base),
            (
                (0, 32, "text"),
                (32, 48, "reference_image"),
                (48, 64, "reference_image"),
                (64, 72, "reference_audio"),
                (72, 82, "reference_video_anchor"),
                (82, 102, "reference_video"),
                (102, 112, "reference_video_anchor"),
                (112, 118, "reference_audio"),
                (118, 130, "target_audio"),
                (130, 258, "target_video"),
            ),
        )

    def test_sparse_patch_installs_layout_provider_on_official_loader_model(self):
        model = FakePatcher()
        override = object()
        with (
            self._minimax_type_patch(),
            mock.patch("attention.make_sparse_attention_override", return_value=override),
        ):
            patched = attention.apply_sparse_attention_patch(model)

        options = patched.model_options["transformer_options"]
        self.assertEqual(
            options[ATTENTION_LAYOUT_REQUIREMENT_KEY],
            minimax_layout.MINIMAX_H3_LAYOUT_KIND,
        )
        runtime = attention.attention_runtime_config(options)
        self.assertEqual(runtime.strategy, "sol")
        self.assertIs(runtime.strategy_override, override)
        self.assertEqual(len(patched.object_patches), 3)
        self.assertEqual(len(patched.wrappers), 1)
        self.assertFalse(model.object_patches)
        self.assertFalse(model.wrappers)

if __name__ == "__main__":
    unittest.main()
