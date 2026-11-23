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

from comfyui_turing_utils.attention import (
    patches as attention_patches,
    runtime as attention_runtime,
)  # noqa: E402
from comfyui_turing_utils.adapters.minimax import layout as minimax_layout  # noqa: E402
from comfyui_turing_utils.attention.layout import (  # noqa: E402
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    attention_semantic_layout,
    has_complete_attention_layout,
)


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_options = None

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
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
    def test_dense_patch_installs_refiner_bridge_on_official_loader_model(self):
        patcher = FakePatcher()
        patcher.model.diffusion_model.token_refiner = torch.nn.Identity()
        with self._minimax_type_patch():
            attention_patches.apply_attention_backend(
                patcher, "sdpa", device=torch.device("cpu")
            )
        self.assertIn(
            "diffusion_model.token_refiner.forward", patcher.object_patches
        )
        self.assertEqual(len(patcher.wrappers), 1)

    def test_native_patcher_does_not_stack_runtime_wrappers_on_reinstall_or_clone(self):
        from comfy.model_patcher import ModelPatcher
        from comfy.patcher_extension import WrappersMP

        patcher = ModelPatcher(
            FakeBase(), load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )
        with self._minimax_type_patch():
            minimax_layout.ensure_minimax_attention_layout_provider(patcher)
            minimax_layout.ensure_minimax_attention_layout_provider(patcher)
            branch = patcher.clone()
            minimax_layout.ensure_minimax_attention_layout_provider(branch)
        for model in (patcher, branch):
            self.assertEqual(len(model.get_wrappers(
                WrappersMP.OUTER_SAMPLE, minimax_layout.RUNTIME_OUTER_WRAPPER_KEY,
            )), 1)

    def test_text_refiner_uses_current_branch_dense_attention_without_sparse_layout(
        self,
    ):
        import comfy.ops
        from comfy.ldm.minimax.model import TokenRefiner

        patcher = FakePatcher()
        refiner = TokenRefiner(
            2,
            16,
            2,
            8,
            32,
            1e-6,
            1e-6,
            1e-6,
            dtype=torch.float32,
            device="cpu",
            operations=comfy.ops.manual_cast,
        )
        for parameter in refiner.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        patcher.model.diffusion_model.token_refiner = refiner
        value = torch.randn(11, 16)
        expected = refiner(value)
        dense_calls = []

        def dense(original, *args, **kwargs):
            dense_calls.append(kwargs["transformer_options"])
            return original(*args, **kwargs)

        def sparse(*args, **kwargs):
            raise AssertionError("Text refiner entered packed video sparse attention")

        config = attention_runtime.AttentionRuntimeConfig(
            "w8a8",
            "test-dense",
            dense,
            strategy="sol",
            strategy_override=sparse,
        )
        attention_runtime.install_attention_runtime(
            patcher.model_options["transformer_options"], config
        )
        with self._minimax_type_patch():
            minimax_layout.ensure_minimax_attention_layout_provider(patcher)
            installed = patcher.object_patches.copy()
            minimax_layout.ensure_minimax_attention_layout_provider(patcher)
            self.assertEqual(patcher.object_patches, installed)
        forward = patcher.object_patches.get(
            "diffusion_model.token_refiner.forward", refiner.forward
        )
        wrapper = next(iter(patcher.wrappers.values()))

        class Executor:
            class_obj = SimpleNamespace(model_options=patcher.model_options)

            def __call__(self):
                return forward(value)

        actual = wrapper(Executor())
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(dense_calls), 2)
        self.assertTrue(
            all(
                set(options) == {"optimized_attention_override"}
                for options in dense_calls
            )
        )
        self.assertFalse(hasattr(patcher.model, minimax_layout.RUNTIME_CONTEXT_ATTR))
        branch = patcher.clone()
        other_calls = []

        def other_dense(original, *args, **kwargs):
            other_calls.append(True)
            return dense(original, *args, **kwargs)

        executor = Executor()
        executor.class_obj = SimpleNamespace(model_options=branch.model_options)
        for strategy in ("dense", "sol", "sla", "spark", "veda"):
            with self.subTest(strategy=strategy):
                attention_runtime.install_attention_runtime(
                    branch.model_options["transformer_options"],
                    attention_runtime.AttentionRuntimeConfig(
                        "sage", "other-dense", other_dense
                    ).with_strategy(strategy, "test", sparse),
                )
                torch.testing.assert_close(wrapper(executor), expected)
        self.assertEqual(len(other_calls), 10)
        self.assertIs(
            attention_runtime.attention_runtime_config(
                patcher.model_options["transformer_options"]
            ).dense_override,
            dense,
        )
        torch.testing.assert_close(forward(value), expected)
        self.assertEqual(len(dense_calls), 12)

        # An official backend node can replace the dispatcher on a clone while
        # the old Turing config remains in model_options. None must use the
        # replacement in preprocessing as well as in the main DiT.
        from comfy.model_patcher import ModelPatcher
        from comfy.ldm.modules import attention as native

        kitchen_calls = []

        def kitchen(*args, **kwargs):
            kitchen_calls.append(True)
            return native.attention_pytorch(*args, **kwargs)

        def kitchen_container(q, k, v, *args, **kwargs):
            return kitchen(q.take(), k.take(), v.take(), *args, **kwargs)

        kitchen.container_function = kitchen_container
        ModelPatcher.set_model_optimized_attention(branch, kitchen)
        with (
            self._minimax_type_patch(),
            mock.patch.dict(
                native.REGISTERED_ATTENTION_FUNCTIONS, {"comfy_kitchen_int8": kitchen}
            ),
        ):
            inherited = attention_patches.inherit_attention_backend(branch)
            executor.class_obj = SimpleNamespace(model_options=inherited.model_options)
            torch.testing.assert_close(wrapper(executor), expected)
        self.assertEqual(len(kitchen_calls), 2)
        self.assertEqual(len(dense_calls), 12)
        self.assertFalse(hasattr(patcher.model, minimax_layout.RUNTIME_CONTEXT_ATTR))

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
                x,
                x,
                [(0, 64, 0), (64, 88, 2), (88, 228, 3)],
                None,
                transformer_options=options,
            )
            packed_layout = SimpleNamespace(
                signature=(64, 7, 8, 10, 12),
                segments=[
                    (0, 64, "text"),
                    (64, 88, "audio"),
                    (88, 228, "video"),
                ],
            )
            return model_forward(
                model_input,
                None,
                context,
                transformer_options=options,
                minimax_payload={"layout": packed_layout, "refs": []},
            )

        executor.class_obj = SimpleNamespace(model_options=patcher.model_options)
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
            tuple(
                (item.start, item.stop, item.role) for item in semantic.query_segments
            ),
            (
                (0, 64, "text"),
                (64, 88, "target_audio"),
                (88, 228, "target_video"),
            ),
        )
        self.assertTrue(
            has_complete_attention_layout(options, 228, provider="minimax_h3")
        )
        self.assertFalse(hasattr(patcher.model, minimax_layout.RUNTIME_CONTEXT_ATTR))

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
        rebuilt = minimax_layout._resolve_packed_layout(
            diffusion,
            [video, audio],
            context,
            {"layout": stale, "refs": [], "keyframes": None},
        )
        self.assertIsInstance(rebuilt, minimax_model.PackedLayout)
        self.assertEqual(rebuilt.signature, (48, 7, 10, 12, 13))
        self.assertEqual(rebuilt.seq_len, 48 + 7 * 5 * 6 + 2 * 13)
        self.assertIs(
            minimax_layout._resolve_packed_layout(
                diffusion, [video, audio], context, {"layout": rebuilt}
            ),
            rebuilt,
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
            has_complete_attention_layout(options, 200, provider="minimax_h3")
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
            mock.patch(
                "comfyui_turing_utils.attention.patches.make_sparse_attention_override",
                return_value=override,
            ),
        ):
            patched = attention_patches.apply_sparse_attention_patch(model)

        options = patched.model_options["transformer_options"]
        self.assertEqual(
            options[ATTENTION_LAYOUT_REQUIREMENT_KEY],
            minimax_layout.MINIMAX_H3_LAYOUT_KIND,
        )
        runtime = attention_runtime.attention_runtime_config(options)
        self.assertEqual(runtime.strategy, "sol")
        self.assertIs(runtime.strategy_override, override)
        self.assertEqual(len(patched.object_patches), 3)
        self.assertEqual(len(patched.wrappers), 1)
        self.assertFalse(model.object_patches)
        self.assertFalse(model.wrappers)


if __name__ == "__main__":
    unittest.main()
