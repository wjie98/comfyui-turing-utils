from __future__ import annotations

import math
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

from comfyui_turing_utils.adapters.minimax import block_cache  # noqa: E402
from comfyui_turing_utils.adapters.minimax.block_cache import (  # noqa: E402
    MiniMaxH3BlockCache,
    MiniMaxH3BlockCacheGroup,
    ResidualStore,
    _PROFILES,
    _profile_block_range,
)
from comfyui_turing_utils.nodes.minimax import MiniMaxH3BlockCachePatch  # noqa: E402


def _options(schedule, sigma, **extra):
    return {
        "sample_sigmas": torch.tensor(schedule, dtype=torch.float32),
        "sigmas": torch.tensor([sigma], dtype=torch.float32),
        **extra,
    }


def _layout():
    return SimpleNamespace(
        segments=(
            (0, 1, "text"),
            (1, 2, "reference_image"),
            (2, 3, "audio"),
            (3, 4, "video"),
        )
    )


def _cache(profile="standard", device="gpu"):
    cache = MiniMaxH3BlockCache(_PROFILES[profile], device, 50)
    cache.set_cache_ranges(_layout())
    return cache


def _complete(cache, before, residual, options):
    cache.complete_middle(before + residual, options)


class MiniMaxH3BlockCacheTest(unittest.TestCase):
    @staticmethod
    def _fake_model_patcher():
        diffusion_model = block_cache.minimax_model.MiniMaxH3Model.__new__(
            block_cache.minimax_model.MiniMaxH3Model
        )
        torch.nn.Module.__init__(diffusion_model)
        diffusion_model.blocks = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        )

        class Patcher:
            def __init__(self):
                self.model_options = {
                    "transformer_options": {
                        "optimized_attention_override": "sol-or-dense",
                    }
                }
                self.object_patches = {}
                self.wrappers = {}
                self.callbacks = {}

            def get_model_object(self, name):
                if name != "diffusion_model":
                    raise KeyError(name)
                return diffusion_model

            def clone(self):
                cloned = Patcher()
                cloned.model_options = {
                    "transformer_options": self.model_options[
                        "transformer_options"
                    ].copy()
                }
                cloned.object_patches = self.object_patches.copy()
                return cloned

            def add_object_patch(self, name, value):
                self.object_patches[name] = value

            def remove_wrappers_with_key(self, wrapper_type, key):
                self.wrappers.pop((wrapper_type, key), None)

            def remove_callbacks_with_key(self, callback_type, key):
                self.callbacks.pop((callback_type, key), None)

            def add_wrapper_with_key(self, wrapper_type, key, value):
                self.wrappers[(wrapper_type, key)] = value

            def add_callback_with_key(self, callback_type, key, value):
                self.callbacks[(callback_type, key)] = value

        return Patcher()

    def test_short_profile_range_keeps_six_edge_blocks_exact(self):
        profile = _PROFILES["4-step LoRA"]
        self.assertEqual(_profile_block_range(profile, 0), (0, 0))
        self.assertEqual(_profile_block_range(profile, 1), (0, 1))
        self.assertEqual(_profile_block_range(profile, 2), (0, 2))
        self.assertEqual(_profile_block_range(profile, 50), (6, 44))

    def test_standard_profile_reuses_only_latest_exact_residual(self):
        cache = _cache()
        schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        first_options = _options(schedule, 1.0)
        first = torch.tensor([[10.0], [15.0], [20.0], [25.0]])
        residual = torch.tensor([[1.0], [1.5], [2.0], [2.5]])

        self.assertFalse(cache.prepare_middle(first, first_options))
        _complete(cache, first, residual, first_options)
        self.assertEqual(cache.full_steps, 1)

        second = first + 0.01
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.8)))
        self.assertTrue(torch.allclose(second, first + 0.01 + residual))
        self.assertEqual(cache.cache_hits, 1)
        self.assertEqual(cache.skipped_blocks, 50)

    def test_four_step_profile_has_one_hit_budget(self):
        cache = _cache("4-step LoRA")
        schedule = [1.0, 0.75, 0.5, 0.25, 0.0]
        first = torch.tensor([[10.0], [15.0], [20.0], [25.0]])
        residual = torch.ones_like(first)

        first_options = _options(schedule, 1.0)
        self.assertFalse(cache.prepare_middle(first, first_options))
        _complete(cache, first, residual, first_options)

        second = first + 0.01
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.75)))
        self.assertEqual((cache.skip_start, cache.skip_end), (6, 44))

        third = first + 0.02
        third_options = _options(schedule, 0.5)
        self.assertFalse(cache.prepare_middle(third, third_options))
        self.assertEqual(cache.reject_counts["hit_budget"], 1)
        cache.complete_middle(third + residual, third_options)
        self.assertIsNone(cache.residual.tensor)

    def test_eight_step_profile_allows_two_nonconsecutive_hits(self):
        cache = _cache("8-step LoRA")
        schedule = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
        base = torch.tensor([[10.0], [15.0], [20.0], [25.0]])
        residual = torch.ones_like(base)

        for sigma in (1.0, 0.875):
            value = base + (1.0 - sigma) * 0.01
            options = _options(schedule, sigma)
            self.assertFalse(cache.prepare_middle(value, options))
            _complete(cache, value, residual, options)

        first_hit = base + 0.02
        self.assertTrue(cache.prepare_middle(first_hit, _options(schedule, 0.75)))

        exact = base + 0.03
        exact_options = _options(schedule, 0.625)
        self.assertFalse(cache.prepare_middle(exact, exact_options))
        self.assertEqual(cache.reject_counts["mcs"], 1)
        _complete(cache, exact, residual * 2.0, exact_options)

        second_hit = base + 0.04
        self.assertTrue(cache.prepare_middle(second_hit, _options(schedule, 0.5)))
        self.assertTrue(torch.allclose(second_hit, base + 0.04 + residual * 2.0))

        exhausted = base + 0.05
        exhausted_options = _options(schedule, 0.375)
        self.assertFalse(cache.prepare_middle(exhausted, exhausted_options))
        self.assertEqual(cache.reject_counts["hit_budget"], 1)

    def test_auto_profile_uses_exact_step_counts(self):
        group = MiniMaxH3BlockCacheGroup("auto", "auto", 50)
        four = group.state_for({"sample_sigmas": torch.ones(5), "uuids": ["p"]})
        eight = group.state_for({"sample_sigmas": torch.ones(9), "uuids": ["p"]})
        six = group.state_for({"sample_sigmas": torch.ones(7), "uuids": ["p"]})
        self.assertEqual(four.profile.name, "4-step LoRA")
        self.assertEqual(eight.profile.name, "8-step LoRA")
        self.assertEqual(six.profile.name, "standard")

    def test_explicit_short_profile_falls_back_above_ten_steps(self):
        group = MiniMaxH3BlockCacheGroup("8-step LoRA", "auto", 50)
        short = group.state_for({"sample_sigmas": torch.ones(9)})
        long = group.state_for({"sample_sigmas": torch.ones(12)})
        self.assertEqual(short.profile.name, "8-step LoRA")
        self.assertEqual(long.profile.name, "standard")

    def test_group_separates_sampler_branches(self):
        group = MiniMaxH3BlockCacheGroup("standard", "auto", 50)
        positive = {"sample_sigmas": torch.ones(6), "uuids": ["positive"]}
        self.assertIs(group.state_for(positive), group.state_for(positive))
        self.assertIsNot(
            group.state_for(positive),
            group.state_for(
                {"sample_sigmas": torch.ones(6), "uuids": ["negative"]}
            ),
        )

    def test_discontinuous_sigma_clears_cached_residual(self):
        cache = _cache()
        schedule = [1.0, 0.8, 0.6, 0.4, 0.0]
        first = torch.tensor([[10.0], [15.0], [20.0], [25.0]])
        options = _options(schedule, 1.0)
        cache.prepare_middle(first, options)
        _complete(cache, first, torch.ones_like(first), options)
        second = first + 0.01
        self.assertTrue(cache.prepare_middle(second, _options(schedule, 0.8)))

        restarted = first + 0.02
        self.assertFalse(cache.prepare_middle(restarted, _options(schedule, 0.8)))
        self.assertEqual(cache.reject_counts["discontinuity"], 1)
        self.assertIsNotNone(cache.residual.tensor)
        self.assertTrue(math.isfinite(cache.last_sigma))

    def test_overlapping_block_patch_disables_storage(self):
        cache = _cache()
        hidden = torch.tensor([[10.0], [15.0], [20.0], [25.0]])
        options = _options([1.0, 0.8, 0.6, 0.0], 1.0)
        self.assertFalse(cache.prepare_middle(hidden, options, force_full=True))
        self.assertEqual(cache.reject_counts["patch_overlap"], 1)
        self.assertIsNone(cache.residual.tensor)
        cache.complete_middle(hidden + 1.0, options)
        self.assertIsNone(cache.residual.tensor)

    def test_residual_store_keeps_one_snapshot_and_applies_exact_residual(self):
        store = ResidualStore("gpu")
        before = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        residual = torch.full_like(before, 2.0)
        store.capture(before, {})
        self.assertNotEqual(store.tensor.data_ptr(), before.data_ptr())
        store.finish_residual(before + residual)
        hidden = torch.ones_like(before)
        store.apply(hidden)
        self.assertTrue(torch.equal(hidden, residual + 1.0))
        store.clear()
        self.assertIsNone(store.tensor)

    def test_auto_storage_obeys_comfy_vram_reserve(self):
        store = ResidualStore("auto")
        source = SimpleNamespace(device=torch.device("cuda:0"))
        with (
            mock.patch.object(store, "_byte_size", return_value=512 << 20),
            mock.patch.object(
                block_cache.comfy.model_management,
                "get_free_memory",
                return_value=3 << 30,
            ),
            mock.patch.object(
                block_cache.comfy.model_management,
                "minimum_inference_memory",
                return_value=2 << 30,
            ),
        ):
            self.assertTrue(
                store._keep_on_gpu(source, {"prefetch_dynamic_vbars": True})
            )

        with (
            mock.patch.object(store, "_byte_size", return_value=512 << 20),
            mock.patch.object(
                block_cache.comfy.model_management,
                "get_free_memory",
                return_value=5 << 29,
            ),
            mock.patch.object(
                block_cache.comfy.model_management,
                "minimum_inference_memory",
                return_value=2 << 30,
            ),
        ):
            self.assertFalse(
                store._keep_on_gpu(source, {"prefetch_dynamic_vbars": True})
            )

        with (
            mock.patch.object(store, "_byte_size", return_value=512 << 20),
            mock.patch.object(
                block_cache.comfy.model_management,
                "get_free_memory",
                return_value=2 << 30,
            ),
            mock.patch.object(
                block_cache.comfy.model_management,
                "minimum_inference_memory",
                return_value=2 << 30,
            ),
        ):
            self.assertFalse(store._keep_on_gpu(source, {}))

    def test_clear_unregisters_comfy_managed_pinned_memory(self):
        store = ResidualStore("cpu")
        store.tensor = torch.zeros(4)
        store.comfy_pinned = True
        with mock.patch.object(
            block_cache.comfy.model_management,
            "unpin_memory",
        ) as unpin:
            store.clear()
        unpin.assert_called_once()
        self.assertIsNone(store.tensor)
        self.assertFalse(store.comfy_pinned)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_store_round_trip_uses_one_exact_chunked_residual(self):
        residual = torch.arange(
            1024 * 64,
            device="cuda",
            dtype=torch.float32,
        ).reshape(1024, 64)
        store = ResidualStore("cpu")
        try:
            store.capture(residual, {})
            self.assertEqual(store.storage_device, "cpu")
            store.finish_residual(residual + 2.0)
            hidden = torch.ones_like(residual)
            store.apply(hidden)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(
                    hidden.cpu(),
                    torch.full_like(hidden, 3.0).cpu(),
                )
            )
        finally:
            store.clear()

    def test_signature_observes_every_layout_segment(self):
        cache = _cache()
        hidden = torch.zeros(4, 64)
        reference = cache._signature(hidden, cache.cache_ranges)
        for row in range(4):
            changed = hidden.clone()
            changed[row] = 1.0
            current = cache._signature(changed, cache.cache_ranges)
            self.assertFalse(torch.equal(current, reference))

    def test_node_exposes_only_profiles_and_storage_policy(self):
        required = MiniMaxH3BlockCachePatch.INPUT_TYPES()["required"]
        self.assertEqual(tuple(required), ("model", "profile", "cache_device"))
        self.assertEqual(required["profile"][1]["default"], "auto")
        self.assertEqual(required["cache_device"][1]["default"], "auto")

    def test_current_comfy_h3_forward_contract_is_supported(self):
        self.assertTrue(
            block_cache._compatible_forward(
                block_cache.minimax_model.MiniMaxH3Model._forward
            )
        )

    def test_current_comfy_full_cache_pass_matches_native_masked_forward(self):
        operations = SimpleNamespace(
            Linear=torch.nn.Linear,
            RMSNorm=torch.nn.RMSNorm,
        )
        torch.manual_seed(123)
        model = block_cache.minimax_model.MiniMaxH3Model(
            hidden_size=128,
            num_layers=2,
            token_refiner_num_layers=1,
            num_attention_heads=1,
            attention_head_dim=128,
            ffn_hidden_size=64,
            latents_dim=2,
            audio_latents_dim=4,
            patch_size=(1, 2, 2),
            text_dim=128,
            timestep_input_dim=32,
            time_embed_hidden_size=64,
            time_embed_dim=32,
            rope_inv_freq_len=16,
            dtype=torch.float32,
            operations=operations,
        ).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim > 1:
                    torch.nn.init.normal_(parameter, std=0.02)
                else:
                    parameter.zero_()
            model.rope.inv_freq.fill_(0.01)

        video = torch.randn(1, 2, 1, 4, 2)
        audio = torch.randn(1, 4, 2, 2)
        context = torch.randn(1, 3, 128)
        timestep = torch.tensor([700.0])
        denoise_mask = torch.tensor(
            [[[[[0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0]]]]]
        )
        audio_denoise_mask = torch.tensor([[[[0.5, 1.0], [0.25, 1.0]]]])
        base_options = {
            "sample_sigmas": torch.tensor([0.7, 0.4, 0.0]),
            "sigmas": torch.tensor([0.7]),
        }

        with (
            mock.patch.object(block_cache.comfy.model_management, "in_training", True),
            torch.no_grad(),
        ):
            expected = model._forward(
                [video, audio],
                timestep,
                context,
                transformer_options=dict(base_options),
                denoise_mask=denoise_mask,
                audio_denoise_mask=audio_denoise_mask,
            )
            options = dict(base_options)
            options[block_cache.CACHE_KEY] = MiniMaxH3BlockCacheGroup(
                "standard", "gpu", 2
            )
            actual = block_cache._cached_forward(
                model,
                [video, audio],
                timestep,
                context,
                transformer_options=options,
                denoise_mask=denoise_mask,
                audio_denoise_mask=audio_denoise_mask,
            )

        for native, cached in zip(expected, actual):
            self.assertTrue(torch.equal(native, cached))

    def test_cleanup_callback_releases_every_branch_cache(self):
        group = MiniMaxH3BlockCacheGroup("standard", "gpu", 50)
        state = group.state_for({"sample_sigmas": torch.ones(6)})
        state.residual.capture(torch.ones(4, 4), {})
        block_cache._CleanupCache(group)()
        self.assertEqual(group.states, {})
        self.assertIsNone(state.residual.tensor)

    def test_installer_preserves_attention_patch_and_registers_cleanup(self):
        patched = block_cache.install_minimax_block_cache(
            self._fake_model_patcher(),
            "auto",
            "auto",
        )
        options = patched.model_options["transformer_options"]
        self.assertEqual(options["optimized_attention_override"], "sol-or-dense")
        self.assertIsInstance(
            options[block_cache.CACHE_KEY],
            MiniMaxH3BlockCacheGroup,
        )
        self.assertIn(block_cache.FORWARD_PATCH_KEY, patched.object_patches)
        self.assertIn(
            (
                block_cache.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                block_cache.PATCH_KEY,
            ),
            patched.wrappers,
        )
        self.assertIn(
            (
                block_cache.comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
                block_cache.PATCH_KEY,
            ),
            patched.callbacks,
        )

    def test_installer_rejects_an_unknown_forward_patch(self):
        model = self._fake_model_patcher()
        model.object_patches[block_cache.FORWARD_PATCH_KEY] = lambda: None
        with self.assertRaisesRegex(RuntimeError, "cannot compose"):
            block_cache.install_minimax_block_cache(model, "auto", "auto")

    def test_run_blocks_prefetches_only_selected_indices(self):
        class Block(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(
                self,
                hidden,
                _t_emb,
                _mod_segments,
                _rope_freqs,
                transformer_options=None,
            ):
                return hidden + self.value

        model = SimpleNamespace(blocks=[Block(i) for i in range(5)])
        completed = []
        with (
            mock.patch.object(
                block_cache.comfy.model_prefetch,
                "make_prefetch_queue",
                return_value=None,
            ) as make_queue,
            mock.patch.object(
                block_cache.comfy.model_prefetch,
                "prefetch_queue_pop",
            ) as pop_queue,
        ):
            result = block_cache._run_blocks(
                model,
                torch.tensor(0),
                (0, 3, 4),
                None,
                None,
                None,
                {},
                {},
                torch.device("cpu"),
                complete_after=3,
                complete_callback=lambda value, _options: completed.append(value.item()),
            )

        self.assertEqual(result.item(), 7)
        self.assertEqual(completed, [3])
        self.assertEqual(
            make_queue.call_args.args[0],
            [model.blocks[0], model.blocks[3], model.blocks[4]],
        )
        self.assertEqual(
            [call.args[2] for call in pop_queue.call_args_list],
            [model.blocks[0], model.blocks[3], model.blocks[4], None],
        )
        if block_cache._PREFETCH_SUPPORTS_MALLOC_SCOPE:
            self.assertTrue(
                all(
                    call.kwargs == {"malloc_scope": "block"}
                    for call in pop_queue.call_args_list
                )
            )


if __name__ == "__main__":
    unittest.main()
