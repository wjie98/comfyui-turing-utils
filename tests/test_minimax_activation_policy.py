from __future__ import annotations

import os
import sys
import unittest
import dataclasses
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from comfyui_turing_utils.adapters.minimax import activation_policy
from comfyui_turing_utils.adapters.minimax import acceleration
from comfyui_turing_utils import precision
from comfyui_turing_utils.quantization import dispatch


_GIB = 1024**3


class _FakeActivation:
    def __init__(self, rows: int):
        self.shape = (rows, 5376)
        self.device = torch.device("cuda", 0)

    @staticmethod
    def element_size() -> int:
        return 2


@dataclasses.dataclass
class _FakePrequantizedQK:
    query_int8: torch.Tensor
    query_scale: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    tensor_layout: str
    input_dtype: torch.dtype
    original_head_dim: int
    route_original_basis: bool


class MiniMaxActivationPolicyTest(unittest.TestCase):
    def test_prepared_attention_fallback_warning_is_bounded_per_shape(self):
        options = {}
        with self.assertLogs("comfyui-turing-utils", level="WARNING") as captured:
            acceleration._warn_attention_fallback(
                options,
                path="head_sharded",
                rows=127_272,
                reason="streamed executor unavailable",
            )
            acceleration._warn_attention_fallback(
                options,
                path="head_sharded",
                rows=127_272,
                reason="streamed executor unavailable",
            )

        self.assertEqual(len(captured.output), 1)
        self.assertIn("path=head_sharded", captured.output[0])
        self.assertIn("rows=127272", captured.output[0])

    def setUp(self):
        self.environment = mock.patch.dict(
            os.environ,
            {
                "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE": "auto",
            },
            clear=False,
        )
        self.environment.start()
        for name in (
            "COMFYUI_TURING_UTILS_H3_QKV_CHUNK_ROWS",
            "COMFYUI_TURING_UTILS_H3_MLP_CHUNK_ROWS",
            "COMFYUI_TURING_UTILS_H3_ACTIVATION_CHUNK_ROWS",
            "COMFYUI_TURING_UTILS_H3_HEAD_GROUP",
            "COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environment.stop()

    @staticmethod
    def _decision(rows: int, operation: str):
        expanded = 7168 if operation == "qkv" else 14336
        return activation_policy.decide_activation_chunks(
            _FakeActivation(rows),
            operation=operation,
            hidden_size=5376,
            expanded_size=expanded,
        )

    @staticmethod
    def _dynamic_base():
        patcher = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: SimpleNamespace(loaded_size=lambda: 4 * _GIB),
            model_size=lambda: 20 * _GIB,
        )
        return SimpleNamespace(
            current_patcher=patcher,
            _turing_utils_minimax_layer_count=50,
        )

    def test_twelve_gib_budget_keeps_low_resolution_throughput(self):
        # 15 s H3 at 480x864: 43,335 video + 1,206 audio target rows.
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            for operation in ("qkv", "mlp"):
                with self.subTest(operation=operation):
                    decision = self._decision(44_541, operation)
                    self.assertFalse(decision.streamed)
                    self.assertEqual(decision.chunk_rows, 0)

    def test_dynamic_qkv_keeps_saturated_tile_when_full_projection_fits(self):
        # A transiently idle desktop used to promote this exact workload back
        # to a slower 60K-row monolithic QKV projection. Four saturated tiles
        # retain the same math while preserving DynamicVRAM weight residency.
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(7.68 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_activation_chunks(
                _FakeActivation(60_186),
                operation="qkv",
                hidden_size=5_376,
                expanded_size=7_168,
                heads=56,
                base_model=self._dynamic_base(),
            )

        self.assertTrue(decision.streamed)
        self.assertEqual(decision.chunk_rows, 16_384)

    def test_dynamic_qkv_does_not_stream_before_four_saturated_tiles(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(7.68 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_activation_chunks(
                _FakeActivation(44_541),
                operation="qkv",
                hidden_size=5_376,
                expanded_size=7_168,
                heads=56,
                base_model=self._dynamic_base(),
            )

        self.assertFalse(decision.streamed)
        self.assertEqual(decision.chunk_rows, 0)

    def test_saturated_qkv_rule_does_not_force_mlp_streaming(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(9.90 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_activation_chunks(
                _FakeActivation(60_186),
                operation="mlp",
                hidden_size=5_376,
                expanded_size=14_336,
                base_model=self._dynamic_base(),
            )

        self.assertFalse(decision.streamed)
        self.assertEqual(decision.chunk_rows, 0)

    def test_twelve_gib_budget_streams_one_megapixel_stage(self):
        # The same clip after 2.5x area upscale to 768x1376 has 111,630 rows.
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            qkv = self._decision(111_630, "qkv")
            mlp = self._decision(111_630, "mlp")

        self.assertTrue(qkv.streamed)
        self.assertEqual(qkv.chunk_rows, 16_384)
        self.assertTrue(mlp.streamed)
        self.assertEqual(mlp.chunk_rows, 16_384)
        self.assertLess(qkv.streamed_peak_bytes, qkv.full_peak_bytes / 2)
        self.assertLess(mlp.streamed_peak_bytes, mlp.full_peak_bytes / 2)

    def test_twelve_gib_budget_includes_typical_reference_rows(self):
        # Two 1 MP images, one 15 s 0.2 MP video, and text bring the packed
        # second-stage sequence close to 135k rows in the supplied workflow.
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            qkv = self._decision(135_000, "qkv")
            mlp = self._decision(135_000, "mlp")

        self.assertEqual(qkv.chunk_rows, 16_384)
        self.assertEqual(mlp.chunk_rows, 16_384)
        self.assertLess(qkv.streamed_peak_bytes, 4.5 * _GIB)
        self.assertLess(mlp.streamed_peak_bytes, 3.9 * _GIB)

    def test_twenty_two_gib_card_releases_full_throughput(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(20 * _GIB, 0, 22 * _GIB),
        ):
            for operation in ("qkv", "mlp"):
                with self.subTest(operation=operation):
                    self.assertFalse(
                        self._decision(111_630, operation).streamed
                    )

    def test_reserve_is_part_of_the_runtime_decision(self):
        # Even if the desktop is temporarily idle, the 12 GiB usable ceiling
        # must keep the one-megapixel stage on the streamed path.
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            decision = self._decision(111_630, "qkv")
        self.assertEqual(decision.reserve_bytes, 4 * _GIB)
        self.assertTrue(decision.streamed)

    def test_attention_heads_keep_full_sequence_when_compact_path_fits(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(135_000),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
            )

        self.assertFalse(decision.sharded)
        self.assertEqual(decision.head_group, 56)
        self.assertFalse(decision.cache_quantized_input)

    def test_attention_heads_shard_at_extreme_budget_on_exact_boundaries(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(6 * _GIB, 0, 6 * _GIB),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(135_000),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
            )

        self.assertTrue(decision.sharded)
        self.assertEqual(decision.head_group % 2, 0)
        self.assertEqual((56 - decision.head_group) % 2, 0)
        self.assertEqual((decision.head_group * 128) % 256, 0)

    def test_sampler_plan_never_promotes_attention_after_free_memory_recovers(self):
        plan = activation_policy.ActivationRuntimePlan()
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            side_effect=(
                (int(6.8 * _GIB), 4 * _GIB, 12 * _GIB),
                (int(7.54 * _GIB), 4 * _GIB, 12 * _GIB),
            ),
        ):
            first = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
                runtime_plan=plan,
            )
            later = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
                runtime_plan=plan,
            )

        self.assertEqual(later.head_group, first.head_group)
        self.assertLess(first.head_group, 56)

    def test_sampler_plan_never_increases_a_streamed_row_tile(self):
        plan = activation_policy.ActivationRuntimePlan()
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            side_effect=(
                (int(5.5 * _GIB), 4 * _GIB, 12 * _GIB),
                (10 * _GIB, 4 * _GIB, 12 * _GIB),
            ),
        ):
            first = activation_policy.decide_activation_chunks(
                _FakeActivation(127_275),
                operation="qkv",
                hidden_size=5376,
                expanded_size=7168,
                runtime_plan=plan,
            )
            later = activation_policy.decide_activation_chunks(
                _FakeActivation(127_275),
                operation="qkv",
                hidden_size=5376,
                expanded_size=7168,
                runtime_plan=plan,
            )

        self.assertTrue(first.streamed)
        self.assertEqual(later.chunk_rows, first.chunk_rows)

    def test_sampler_plan_keeps_independent_operation_low_water_marks(self):
        plan = activation_policy.ActivationRuntimePlan()
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            side_effect=(
                (int(5.75 * _GIB), 4 * _GIB, 12 * _GIB),
                (int(9.90 * _GIB), 4 * _GIB, 12 * _GIB),
                (int(9.90 * _GIB), 4 * _GIB, 12 * _GIB),
            ),
        ):
            qkv = activation_policy.decide_activation_chunks(
                _FakeActivation(60_186),
                operation="qkv",
                hidden_size=5376,
                expanded_size=7168,
                heads=56,
                runtime_plan=plan,
            )
            mlp = activation_policy.decide_activation_chunks(
                _FakeActivation(60_186),
                operation="mlp",
                hidden_size=5376,
                expanded_size=14_336,
                runtime_plan=plan,
            )
            ffn = activation_policy.decide_ffn_channels(
                _FakeActivation(60_186),
                expanded_size=14_336,
                chunk_rows=mlp.chunk_rows,
                runtime_plan=plan,
            )

        self.assertEqual(qkv.chunk_rows, 16_384)
        self.assertFalse(mlp.streamed)
        self.assertFalse(ffn.sharded)
        self.assertEqual(
            plan.available_floors[
                ("cuda", 0, 60_186, "qkv")
            ],
            int(5.75 * _GIB),
        )
        self.assertEqual(
            plan.available_floors[
                ("cuda", 0, 60_186, "mlp")
            ],
            int(9.90 * _GIB),
        )

    def test_attention_peak_includes_w8a8_value_and_summary_lifecycle(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            without_v8 = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=False,
            )
            with_v8 = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
            )

        self.assertGreater(
            with_v8.estimated_peak_bytes,
            without_v8.estimated_peak_bytes + int(0.7 * _GIB),
        )

    def test_virtual_kv_peak_uses_logical_key_rows(self):
        physical = activation_policy.estimate_attention_lifecycle_peak(
            rows=5_996,
            heads=56,
            head_dim=128,
            hidden_size=5_376,
            element_size=2,
            head_group=14,
            compact_qk=False,
            cache_quantized_input=False,
            quantized_value=True,
        )
        virtual = activation_policy.estimate_attention_lifecycle_peak(
            rows=5_996,
            logical_key_rows=16_194,
            heads=56,
            head_dim=128,
            hidden_size=5_376,
            element_size=2,
            head_group=14,
            compact_qk=False,
            cache_quantized_input=False,
            quantized_value=True,
        )

        self.assertGreater(virtual, physical)

    def test_virtual_residual_peak_accounts_for_mapped_sol_summaries(self):
        common = dict(
            rows=5_996,
            logical_key_rows=16_194,
            heads=56,
            head_dim=128,
            hidden_size=5_376,
            element_size=2,
            head_group=14,
            compact_qk=False,
            cache_quantized_input=False,
            quantized_value=True,
        )
        exact = activation_policy.estimate_attention_lifecycle_peak(**common)
        residual = activation_policy.estimate_attention_lifecycle_peak(
            **common,
            residual_subblocks=2,
        )

        self.assertGreater(residual, exact)

    def test_virtual_fp16_residual_retains_only_physical_value_rows(self):
        common = dict(
            rows=5_996,
            logical_key_rows=16_194,
            heads=56,
            head_dim=128,
            hidden_size=5_376,
            element_size=2,
            head_group=14,
            compact_qk=False,
            cache_quantized_input=False,
            residual_subblocks=2,
        )
        fp16_peak = activation_policy.estimate_attention_lifecycle_peak(
            **common,
            quantized_value=False,
        )
        w8a8_peak = activation_policy.estimate_attention_lifecycle_peak(
            **common,
            quantized_value=True,
        )

        self.assertGreater(fp16_peak, 0)
        self.assertLess(fp16_peak, w8a8_peak)

    def test_quiet_vbar_budget_counts_only_resident_unpinned_pages(self):
        vbar = SimpleNamespace(
            get_residency=mock.Mock(return_value=[1, 3, 0, 1]),
            loaded_size=mock.Mock(return_value=3 * 32 * 1024**2),
        )
        patcher = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: vbar,
        )
        base = SimpleNamespace(current_patcher=patcher)

        comfy = ModuleType("comfy")
        comfy.__path__ = []
        model_management = ModuleType("comfy.model_management")
        model_management.loaded_models = mock.Mock(return_value=[])
        comfy.model_management = model_management
        with mock.patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.model_management": model_management,
            },
        ):
            inactive_reclaimable = activation_policy._dynamic_vram_reclaimable(
                base, torch.device("cuda", 0)
            )
            all_reclaimable = activation_policy._dynamic_vram_reclaimable(
                base,
                torch.device("cuda", 0),
                include_current=True,
            )

        self.assertEqual(inactive_reclaimable, 0)
        self.assertEqual(all_reclaimable, 2 * 32 * 1024**2)
        vbar.get_residency.assert_called_once_with()

    def test_runtime_budget_does_not_promote_with_reclaimable_vbar_pages(self):
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        model_management = ModuleType("comfy.model_management")
        model_management.get_total_memory = mock.Mock(return_value=16 * _GIB)
        model_management.extra_reserved_memory = mock.Mock(return_value=4 * _GIB)
        model_management.get_free_memory = mock.Mock(return_value=5 * _GIB)
        comfy.model_management = model_management
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.model_management": model_management,
                },
            ),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=2 * _GIB),
            mock.patch.object(
                activation_policy,
                "_dynamic_vram_reclaimable",
                return_value=4 * _GIB,
            ),
        ):
            available, reserve, usable = activation_policy._runtime_memory(
                torch.device("cuda", 0), object()
            )

        self.assertEqual(available, 5 * _GIB)
        self.assertEqual(reserve, 4 * _GIB)
        self.assertEqual(usable, 12 * _GIB)

    def test_dynamic_vbars_prioritize_inactive_models_and_deduplicate(self):
        inactive_vbar = object()
        current_vbar = object()
        inactive = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: inactive_vbar,
        )
        current = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: current_vbar,
        )
        current_clone = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: current_vbar,
        )
        base = SimpleNamespace(current_patcher=current)
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        model_management = ModuleType("comfy.model_management")
        model_management.loaded_models = mock.Mock(
            return_value=[current_clone, inactive, current]
        )
        comfy.model_management = model_management

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.model_management": model_management,
            },
        ):
            inactive_vbars = activation_policy._dynamic_vbars(
                base, torch.device("cuda", 0)
            )
            all_vbars = activation_policy._dynamic_vbars(
                base,
                torch.device("cuda", 0),
                include_current=True,
            )

        self.assertEqual(inactive_vbars, (inactive_vbar,))
        self.assertEqual(all_vbars, (inactive_vbar, current_vbar))

    def test_high_resolution_head_group_keeps_allocator_margin(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(7.54 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
                base_model=object(),
            )

        self.assertLess(decision.head_group, 56)
        self.assertTrue(decision.sharded)

    def test_observed_high_resolution_budget_shards_without_reclaim(self):
        vbar = SimpleNamespace(
            free_memory=mock.Mock(return_value=896 * 1024**2)
        )
        plan = activation_policy.ActivationRuntimePlan()
        with (
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(int(5.25 * _GIB), 4 * _GIB, 12 * _GIB),
            ),
            mock.patch.object(
                activation_policy,
                "_dynamic_vbars",
                return_value=(vbar,),
            ),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
                runtime_plan=plan,
            )
            released = activation_policy.ensure_dynamic_vram_headroom(
                object(),
                torch.device("cuda", 0),
                rows=127_275,
                operation="attention_heads",
                estimated_peak_bytes=decision.estimated_peak_bytes,
                runtime_plan=plan,
            )

        self.assertEqual(decision.head_group, 14)
        self.assertEqual(decision.saturation_group, 14)
        self.assertEqual(released, 0)
        vbar.free_memory.assert_not_called()

    def test_saturated_head_group_does_not_grow_with_extra_working_memory(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(6.25 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
            )

        self.assertEqual(decision.saturation_group, 14)
        self.assertEqual(decision.head_group, 14)
        self.assertLess(decision.estimated_peak_bytes, 4 * _GIB)

    def test_balanced_saturation_sizes_use_four_equal_groups(self):
        self.assertEqual(
            activation_policy.balanced_saturation_size(
                56,
                alignment=2,
                minimum=8,
            ),
            14,
        )
        self.assertEqual(
            activation_policy.balanced_saturation_size(
                14_336,
                alignment=256,
                minimum=1_024,
            ),
            3_584,
        )

    def test_dynamic_model_residency_disables_full_input_cache(self):
        patcher = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
        )
        base = SimpleNamespace(current_patcher=patcher)
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(6.25 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(127_275),
                heads=56,
                head_dim=128,
                compact_qk=True,
                quantized_input=True,
                quantized_value=True,
                base_model=base,
            )

        self.assertEqual(decision.head_group, 14)
        self.assertFalse(decision.cache_quantized_input)

    def test_dynamic_weight_prefetch_reserve_tracks_two_average_blocks(self):
        vbar = SimpleNamespace(loaded_size=lambda: 4 * _GIB)
        patcher = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: vbar,
            model_size=lambda: 20 * _GIB,
        )
        base = SimpleNamespace(
            current_patcher=patcher,
            _turing_utils_minimax_layer_count=50,
        )

        reserve = activation_policy._dynamic_weight_prefetch_reserve(
            base, torch.device("cuda", 0)
        )

        self.assertEqual(reserve, 26 * 32 * 1024**2)

    def test_dynamic_weight_prefetch_reserve_is_bounded(self):
        patcher = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: SimpleNamespace(loaded_size=lambda: 100 * _GIB),
            model_size=lambda: 100 * _GIB,
        )
        base = SimpleNamespace(
            current_patcher=patcher,
            _turing_utils_minimax_layer_count=2,
        )

        reserve = activation_policy._dynamic_weight_prefetch_reserve(
            base, torch.device("cuda", 0)
        )

        self.assertEqual(reserve, 1024 * 1024**2)

    def test_short_attention_grid_raises_the_saturation_group(self):
        capabilities = SimpleNamespace(multiprocessor_count=46)
        with mock.patch.object(
            activation_policy,
            "device_capabilities",
            return_value=capabilities,
        ):
            group = activation_policy._attention_saturation_group(
                torch.device("cuda", 0),
                rows=151,
                heads=56,
                head_dim=128,
                legal_groups=list(range(56, 1, -2)),
            )

        self.assertEqual(group, 56)

    def test_dynamic_vram_headroom_reclaims_only_the_selected_tier_deficit(self):
        vbar = SimpleNamespace(free_memory=mock.Mock(return_value=384 * 1024**2))
        patcher = SimpleNamespace(
            is_dynamic=lambda: True,
            _vbar_get=lambda: vbar,
        )
        base = SimpleNamespace(current_patcher=patcher)
        plan = activation_policy.ActivationRuntimePlan()
        with (
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(2 * _GIB, 0, 6 * _GIB),
            ),
            mock.patch.object(
                activation_policy,
                "_dynamic_vbars",
                return_value=(vbar,),
            ),
        ):
            first = activation_policy.ensure_dynamic_vram_headroom(
                base,
                torch.device("cuda", 0),
                rows=135_000,
                operation="attention_heads",
                estimated_peak_bytes=2 * _GIB,
                runtime_plan=plan,
            )
            second = activation_policy.ensure_dynamic_vram_headroom(
                base,
                torch.device("cuda", 0),
                rows=135_000,
                operation="attention_heads",
                estimated_peak_bytes=2 * _GIB,
                runtime_plan=plan,
            )

        self.assertEqual(first, 384 * 1024**2)
        self.assertEqual(second, 0)
        vbar.free_memory.assert_called_once()

    def test_headroom_never_evicts_the_current_diffusion_vbar(self):
        current_vbar = SimpleNamespace(
            free_memory=mock.Mock(return_value=512 * 1024**2)
        )
        current = SimpleNamespace(
            load_device=torch.device("cuda", 0),
            is_dynamic=lambda: True,
            _vbar_get=lambda: current_vbar,
        )
        base = SimpleNamespace(current_patcher=current)
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        model_management = ModuleType("comfy.model_management")
        model_management.loaded_models = mock.Mock(return_value=[current])
        comfy.model_management = model_management
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.model_management": model_management,
                },
            ),
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(2 * _GIB, 0, 6 * _GIB),
            ),
        ):
            released = activation_policy.ensure_dynamic_vram_headroom(
                base,
                torch.device("cuda", 0),
                rows=127_275,
                operation="attention_heads",
                estimated_peak_bytes=2 * _GIB,
            )

        self.assertEqual(released, 0)
        current_vbar.free_memory.assert_not_called()

    def test_memory_telemetry_uses_quiet_aimdo_counter(self):
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        memory_management = ModuleType("comfy.memory_management")
        memory_management.aimdo_enabled = True
        comfy.memory_management = memory_management
        comfy_aimdo = ModuleType("comfy_aimdo")
        comfy_aimdo.__path__ = []
        control = ModuleType("comfy_aimdo.control")
        control.get_total_vram_usage = mock.Mock(return_value=1234)
        comfy_aimdo.control = control
        noisy_vbar = ModuleType("comfy_aimdo.model_vbar")
        noisy_vbar.vbars_analyze = mock.Mock(
            side_effect=AssertionError("diagnostic analysis must stay off hot path")
        )
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.memory_management": memory_management,
                    "comfy_aimdo": comfy_aimdo,
                    "comfy_aimdo.control": control,
                    "comfy_aimdo.model_vbar": noisy_vbar,
                },
            ),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=1),
            mock.patch.object(torch.cuda, "memory_reserved", return_value=2),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(3, 4)),
        ):
            counters = activation_policy._memory_diagnostics(
                torch.device("cuda", 0)
            )

        self.assertEqual(counters, (1, 2, 3, 1234))
        control.get_total_vram_usage.assert_called_once_with()
        noisy_vbar.vbars_analyze.assert_not_called()

    def test_attention_head_override_uses_largest_legal_group(self):
        with (
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(20 * _GIB, 0, 22 * _GIB),
            ),
            mock.patch.dict(
                os.environ,
                {"COMFYUI_TURING_UTILS_H3_HEAD_GROUP": "20"},
            ),
        ):
            decision = activation_policy.decide_attention_heads(
                _FakeActivation(10_000),
                heads=56,
                head_dim=128,
                compact_qk=False,
                quantized_input=False,
            )

        self.assertEqual(decision.head_group, 20)

    def test_ffn_channel_sharding_is_reserved_for_tight_or_explicit_use(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
        ):
            row_decision = self._decision(135_000, "mlp")
            normal = activation_policy.decide_ffn_channels(
                _FakeActivation(135_000),
                expanded_size=14_336,
                chunk_rows=row_decision.chunk_rows,
            )
        self.assertFalse(normal.sharded)

        with (
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
            ),
            mock.patch.dict(
                os.environ,
                {"COMFYUI_TURING_UTILS_H3_FFN_CHUNK_CHANNELS": "2300"},
            ),
        ):
            forced = activation_policy.decide_ffn_channels(
                _FakeActivation(135_000),
                expanded_size=14_336,
                chunk_rows=0,
            )
        self.assertTrue(forced.sharded)
        self.assertEqual(forced.chunk_channels, 2_048)
        self.assertEqual(forced.chunk_rows, 4_096)

    def test_ffn_channel_sharding_is_automatic_at_the_extreme_floor(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(2 * _GIB, 0, 6 * _GIB),
        ):
            row_decision = self._decision(135_000, "mlp")
            channel_decision = activation_policy.decide_ffn_channels(
                _FakeActivation(135_000),
                expanded_size=14_336,
                chunk_rows=row_decision.chunk_rows,
            )

        self.assertEqual(row_decision.chunk_rows, 2_048)
        self.assertTrue(channel_decision.sharded)
        self.assertEqual(channel_decision.chunk_channels, 256)
        self.assertEqual(channel_decision.chunk_rows, 2_048)

    def test_ffn_channel_sharding_stops_at_balanced_saturation_width(self):
        with mock.patch.object(
            activation_policy,
            "_runtime_memory",
            return_value=(int(3.95 * _GIB), 4 * _GIB, 12 * _GIB),
        ):
            decision = activation_policy.decide_ffn_channels(
                _FakeActivation(135_000),
                expanded_size=14_336,
                chunk_rows=16_384,
            )

        self.assertTrue(decision.sharded)
        self.assertEqual(decision.chunk_channels, 3_584)

    def test_explicit_modes_and_chunk_override_remain_available(self):
        with (
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(10 * _GIB, 4 * _GIB, 12 * _GIB),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE": "throughput",
                },
            ),
        ):
            self.assertFalse(self._decision(111_630, "qkv").streamed)

        with (
            mock.patch.object(
                activation_policy,
                "_runtime_memory",
                return_value=(20 * _GIB, 0, 22 * _GIB),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE": "auto",
                    "COMFYUI_TURING_UTILS_H3_QKV_CHUNK_ROWS": "8192",
                },
            ),
        ):
            self.assertEqual(self._decision(111_630, "qkv").chunk_rows, 8192)

    def test_streamed_mlp_casts_each_weight_once(self):
        torch.manual_seed(7)
        mlp = SimpleNamespace(
            fc1=torch.nn.Linear(4, 12, bias=True, dtype=torch.bfloat16),
            fc2=torch.nn.Linear(6, 4, bias=True, dtype=torch.bfloat16),
        )
        x = torch.randn(11, 4, dtype=torch.bfloat16)
        expanded = mlp.fc1(x)
        expected = mlp.fc2(
            torch.nn.functional.silu(expanded[:, :6]) * expanded[:, 6:]
        )

        comfy = ModuleType("comfy")
        comfy.__path__ = []
        ops = ModuleType("comfy.ops")
        cast = mock.Mock(
            side_effect=lambda module, *_args, **_kwargs: (
                module.weight,
                module.bias,
                None,
            )
        )
        uncast = mock.Mock()
        ops.run_every_op = mock.Mock()
        ops.cast_bias_weight = cast
        ops.uncast_bias_weight = uncast
        comfy.ops = ops

        def fused(weight, bias, expanded, input_act, output=None):
            self.assertEqual(input_act, "swiglu")
            gate, up = expanded.chunk(2, dim=-1)
            result = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                weight,
                bias,
            )
            if output is not None:
                output.copy_(result)
                return output
            return result

        with (
            mock.patch.dict(
                sys.modules,
                {"comfy": comfy, "comfy.ops": ops},
            ),
            mock.patch.object(
                acceleration,
                "convrot_linear_input_act_from_weight",
                side_effect=fused,
            ),
        ):
            actual = acceleration._stream_mlp(mlp, x, chunk_rows=4)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(cast.call_count, 2)
        self.assertEqual(uncast.call_count, 2)

    def test_ffn_channel_stream_uses_common_scale_and_direct_offsets(self):
        rows, hidden, expanded = 5, 4, 512
        fc1 = SimpleNamespace(
            weight=torch.zeros((2 * expanded, hidden), dtype=torch.int8),
            bias=None,
        )
        fc2 = SimpleNamespace(
            weight=torch.zeros((hidden, expanded), dtype=torch.int8),
            bias=None,
            in_features=expanded,
            pre_quant_scale=None,
        )
        mlp = SimpleNamespace(fc1=fc1, fc2=fc2)
        x = torch.zeros((rows, hidden), dtype=torch.bfloat16)

        comfy = ModuleType("comfy")
        comfy.__path__ = []
        ops = ModuleType("comfy.ops")
        cast = mock.Mock(
            side_effect=lambda module, *_args, **_kwargs: (
                module.weight,
                module.bias,
                None,
            )
        )
        ops.run_every_op = mock.Mock()
        ops.cast_bias_weight = cast
        ops.uncast_bias_weight = mock.Mock()
        comfy.ops = ops

        def plain(weight):
            return weight, torch.ones((), dtype=torch.float32)

        def expanded_shard(*args):
            start, stop = args[7], args[8]
            width = stop - start
            marker = start // 256 + 1
            return torch.full(
                (args[1].shape[0], 2 * width),
                marker,
                dtype=torch.bfloat16,
            )

        def local_quantize(value, _group_size):
            marker = float(value[0, 0])
            scale = torch.full(
                (value.shape[0],), marker, dtype=torch.float32
            )
            return torch.zeros(
                (value.shape[0], value.shape[1] // 2), dtype=torch.int8
            ), scale

        common_scales = []

        def scaled_quantize(value, scale, _group_size, output=None):
            common_scales.append(scale.clone())
            marker = int(value[0, 0])
            result = torch.full(
                (value.shape[0], value.shape[1] // 2),
                marker,
                dtype=torch.int8,
            )
            if output is not None:
                output.copy_(result)
                return output
            return result

        contractions = []

        def contraction(activation, scale, *_args, **kwargs):
            contractions.append((activation.clone(), scale.clone()))
            total = activation.to(torch.int32).sum(dim=-1)
            result = total[:, None].expand(-1, hidden).to(torch.bfloat16)
            if kwargs.get("output") is not None:
                kwargs["output"].copy_(result)
                return kwargs["output"]
            return result

        with (
            mock.patch.dict(sys.modules, {"comfy": comfy, "comfy.ops": ops}),
            mock.patch.object(
                acceleration, "convrot_w8_plain_tensors", side_effect=plain
            ),
            mock.patch.object(
                acceleration,
                "_quantize_qkv_rows",
                side_effect=lambda _linear, tile: (
                    torch.zeros_like(tile, dtype=torch.int8),
                    torch.ones(tile.shape[0], dtype=torch.float32),
                ),
            ),
            mock.patch.object(
                acceleration, "_ffn_expanded_shard", side_effect=expanded_shard
            ),
            mock.patch.object(
                acceleration,
                "quantize_convrot_swiglu_activation",
                side_effect=local_quantize,
            ),
            mock.patch.object(
                acceleration,
                "quantize_convrot_swiglu_with_scale",
                side_effect=scaled_quantize,
            ),
            mock.patch.object(
                acceleration,
                "int8_linear_from_quantized",
                side_effect=contraction,
            ),
        ):
            actual = acceleration._stream_mlp_channels(
                mlp, x, chunk_rows=3, chunk_channels=256
            )

        self.assertEqual(cast.call_count, 2)
        self.assertEqual(ops.uncast_bias_weight.call_count, 2)
        self.assertEqual(len(contractions), 2)
        self.assertEqual(len(common_scales), 4)
        for scale in common_scales:
            self.assertTrue(torch.equal(scale, torch.full_like(scale, 2.0)))
        for activation, scale in contractions:
            self.assertTrue(torch.equal(activation[:, :256], torch.ones_like(activation[:, :256])))
            self.assertTrue(torch.equal(activation[:, 256:], torch.full_like(activation[:, 256:], 2)))
            self.assertTrue(torch.equal(scale, torch.full_like(scale, 2.0)))
        torch.testing.assert_close(
            actual,
            torch.full((rows, hidden), 768, dtype=torch.bfloat16),
        )

    def test_streamed_qkv_keeps_scale_blocks_and_casts_weight_once(self):
        torch.manual_seed(9)
        projection = torch.nn.Linear(
            4, 6, bias=True, dtype=torch.bfloat16
        )
        attention = SimpleNamespace(
            qkv_proj=projection,
            heads=1,
            head_dim=2,
        )
        x = torch.randn(130, 4, dtype=torch.bfloat16)
        expected_value = projection(x)[:, 4:].view(130, 1, 2)
        expected_value = expected_value.transpose(0, 1).unsqueeze(0)

        comfy = ModuleType("comfy")
        comfy.__path__ = []
        ops = ModuleType("comfy.ops")
        cast = mock.Mock(
            return_value=(projection.weight, projection.bias, None)
        )
        uncast = mock.Mock()
        ops.run_every_op = mock.Mock()
        ops.cast_bias_weight = cast
        ops.uncast_bias_weight = uncast
        comfy.ops = ops
        calls = 0

        def prequantize(query, key, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            rows = query.shape[2]
            outputs = _kwargs.get("qk_output")
            if outputs is None:
                query_int8 = torch.full_like(query, calls, dtype=torch.int8)
                query_scale = torch.full(
                    (1, 1, ((rows + 63) // 64) * 4),
                    calls,
                    dtype=torch.float32,
                )
                key_int8 = torch.full_like(key, calls, dtype=torch.int8)
                key_scale = torch.full(
                    (1, 1, (rows + 63) // 64),
                    calls,
                    dtype=torch.float32,
                )
            else:
                query_int8, query_scale, key_int8, key_scale = outputs
                query_int8.fill_(calls)
                query_scale.fill_(calls)
                key_int8.fill_(calls)
                key_scale.fill_(calls)
                self.assertIsNotNone(_kwargs.get("k_anchor"))
            return _FakePrequantizedQK(
                query_int8=query_int8,
                query_scale=query_scale,
                key_int8=key_int8,
                key_scale=key_scale,
                tensor_layout="HND",
                input_dtype=x.dtype,
                original_head_dim=2,
                route_original_basis=False,
            )

        with (
            mock.patch.dict(
                sys.modules, {"comfy": comfy, "comfy.ops": ops}
            ),
            mock.patch.object(
                acceleration,
                "prequantize_turing_qk",
                side_effect=prequantize,
            ),
            mock.patch.object(
                acceleration,
                "precompute_turing_k_anchor",
                return_value=(
                    torch.full((1, 1), -1, dtype=torch.int32),
                    torch.zeros((1, 1, 2), dtype=torch.float32),
                ),
            ),
        ):
            qk, value = acceleration._stream_qkv_projection(
                attention,
                x,
                SimpleNamespace(freqs=None),
                chunk_rows=64,
            )

        self.assertEqual(calls, 3)
        self.assertTrue(torch.equal(qk.query_int8[:, :, :64], torch.ones((1, 1, 64, 2), dtype=torch.int8)))
        self.assertTrue(torch.equal(qk.query_int8[:, :, 64:128], torch.full((1, 1, 64, 2), 2, dtype=torch.int8)))
        self.assertTrue(torch.equal(qk.query_int8[:, :, 128:], torch.full((1, 1, 2, 2), 3, dtype=torch.int8)))
        self.assertEqual(qk.query_scale.flatten().tolist(), [1.0] * 4 + [2.0] * 4 + [3.0] * 4)
        self.assertEqual(qk.key_scale.flatten().tolist(), [1.0, 2.0, 3.0])
        torch.testing.assert_close(value, expected_value)
        cast.assert_called_once()
        uncast.assert_called_once_with(
            projection, projection.weight, projection.bias, None
        )

    def test_head_sharding_preserves_full_sequence_and_feature_order(self):
        sequence, heads, head_dim = 7, 4, 2
        x = torch.randn(sequence, 3, dtype=torch.bfloat16)
        expected = torch.arange(
            sequence * heads * head_dim, dtype=torch.bfloat16
        ).view(sequence, heads, head_dim)
        projection = SimpleNamespace(weight=torch.empty(1), pre_quant_scale=None)
        attention = SimpleNamespace(
            qkv_proj=projection,
            heads=heads,
            head_dim=head_dim,
        )

        comfy = ModuleType("comfy")
        comfy.__path__ = []
        ops = ModuleType("comfy.ops")
        ops.run_every_op = mock.Mock()
        ops.cast_bias_weight = mock.Mock(
            return_value=(torch.empty(1), None, None)
        )
        ops.uncast_bias_weight = mock.Mock()
        comfy.ops = ops
        ldm = ModuleType("comfy.ldm")
        ldm.__path__ = []
        modules = ModuleType("comfy.ldm.modules")
        modules.__path__ = []
        attention_module = ModuleType("comfy.ldm.modules.attention")
        attention_module.optimized_attention = mock.Mock(
            side_effect=lambda _q, _k, value, *_args, **_kwargs: (
                value.transpose(1, 2).reshape(
                    1, sequence, value.shape[1] * value.shape[-1]
                )
            )
        )

        def project(
            _attention,
            _x,
            _qweight,
            _weight_scale,
            _bias,
            head_start,
            head_stop,
            *_args,
        ):
            group = head_stop - head_start
            shape = (sequence, group, head_dim)
            return (
                torch.zeros(shape, dtype=x.dtype),
                torch.zeros(shape, dtype=x.dtype),
                expected[:, head_start:head_stop].clone(),
            )

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy": comfy,
                    "comfy.ops": ops,
                    "comfy.ldm": ldm,
                    "comfy.ldm.modules": modules,
                    "comfy.ldm.modules.attention": attention_module,
                },
            ),
            mock.patch.object(
                acceleration,
                "convrot_w8_plain_tensors",
                return_value=(torch.empty(1), torch.ones(1)),
            ),
            mock.patch.object(
                acceleration,
                "_project_qkv_head_group",
                side_effect=project,
            ),
            mock.patch.object(
                acceleration,
                "_apply_minimax_qk_transform",
                side_effect=lambda _attention, query, key, _rope: (
                    query,
                    key,
                ),
            ),
        ):
            actual = acceleration._head_sharded_attention(
                attention,
                x,
                None,
                SimpleNamespace(),
                {},
                lambda value: value,
                None,
                head_group=2,
                cache_quantized_input=False,
            )

        torch.testing.assert_close(actual, expected.reshape(sequence, -1))
        self.assertEqual(attention_module.optimized_attention.call_count, 2)
        ops.cast_bias_weight.assert_called_once()
        ops.uncast_bias_weight.assert_called_once()

    def test_ampere_capability_uses_shared_fused_w8_path(self):
        x = torch.zeros((4, 12), dtype=torch.bfloat16)
        weight = torch.zeros((8, 6), dtype=torch.int8)
        weight_scale = torch.ones((), dtype=torch.float32)
        qactivation = torch.zeros((4, 6), dtype=torch.int8)
        activation_scale = torch.ones((4, 1), dtype=torch.float32)
        expected = torch.zeros((4, 8), dtype=torch.bfloat16)

        comfy_kitchen = ModuleType("comfy_kitchen")
        comfy_kitchen.__path__ = []
        backends = ModuleType("comfy_kitchen.backends")
        backends.__path__ = []
        cuda = ModuleType("comfy_kitchen.backends.cuda")
        cuda.int8_linear = mock.Mock(return_value="generic fallback")
        activations = ModuleType("comfy_kitchen.backends._activations")
        activations.apply_input_act = lambda value, _name: value
        backends.cuda = cuda
        comfy_kitchen.backends = backends

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy_kitchen": comfy_kitchen,
                    "comfy_kitchen.backends": backends,
                    "comfy_kitchen.backends.cuda": cuda,
                    "comfy_kitchen.backends._activations": activations,
                },
            ),
            mock.patch.object(
                dispatch, "is_supported_tensor_core_device", return_value=True
            ),
            mock.patch.object(
                dispatch,
                "_quantize_turing_int8_activation",
                return_value=(qactivation, activation_scale),
            ) as quantize,
            mock.patch.object(
                dispatch, "_turing_int8_gemm", return_value=expected
            ) as gemm,
        ):
            actual = dispatch.int8_linear(
                x,
                weight,
                weight_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
                input_act="swiglu",
            )

        torch.testing.assert_close(actual, expected)
        quantize.assert_called_once()
        self.assertEqual(quantize.call_args.args[0].data_ptr(), x.data_ptr())
        self.assertEqual(quantize.call_args.args[1], 256)
        self.assertEqual(quantize.call_args.kwargs["input_act"], "swiglu")
        gemm.assert_called_once()
        cuda.int8_linear.assert_not_called()

    def test_large_w8_uses_bundled_fixed_workspace_without_kitchen_symbol(self):
        qactivation = torch.zeros((4, 16), dtype=torch.int8)
        weight = torch.zeros((8, 16), dtype=torch.int8)
        activation_scale = torch.ones(4, dtype=torch.float32)
        weight_scale = torch.ones((), dtype=torch.float32)
        expected = torch.zeros((4, 8), dtype=torch.bfloat16)

        comfy_kitchen = ModuleType("comfy_kitchen")
        comfy_kitchen.__path__ = []
        backends = ModuleType("comfy_kitchen.backends")
        backends.__path__ = []
        cuda = ModuleType("comfy_kitchen.backends.cuda")
        backends.cuda = cuda
        comfy_kitchen.backends = backends
        bundled = mock.Mock(return_value=expected)

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy_kitchen": comfy_kitchen,
                    "comfy_kitchen.backends": backends,
                    "comfy_kitchen.backends.cuda": cuda,
                },
            ),
            mock.patch.object(
                dispatch, "TURING_INT8_GLOBAL_WORKSPACE_LIMIT", 1
            ),
            mock.patch.object(
                dispatch, "_kernel_op", return_value=bundled
            ) as kernel_op,
            mock.patch.object(
                dispatch, "_kernel_available", return_value=True
            ),
            mock.patch.object(
                dispatch, "is_supported_tensor_core_device", return_value=True
            ),
        ):
            actual = dispatch._turing_int8_gemm(
                qactivation,
                weight,
                activation_scale,
                weight_scale,
                bias=None,
                output_dtype=torch.bfloat16,
            )

        torch.testing.assert_close(actual, expected)
        kernel_op.assert_called_once_with("turing_int8_linear")
        bundled.assert_called_once()
        self.assertEqual(bundled.call_args.args[3].shape, (8,))
        self.assertEqual(bundled.call_args.args[3].tolist(), [1.0] * 8)

    def test_w8_direct_output_preserves_destination_stride(self):
        qactivation = torch.zeros((4, 16), dtype=torch.int8)
        weight = torch.zeros((8, 16), dtype=torch.int8)
        activation_scale = torch.ones(4, dtype=torch.float32)
        weight_scale = torch.ones((), dtype=torch.float32)
        storage = torch.empty((4, 12), dtype=torch.bfloat16)
        destination = storage[:, 2:10]

        comfy_kitchen = ModuleType("comfy_kitchen")
        comfy_kitchen.__path__ = []
        backends = ModuleType("comfy_kitchen.backends")
        backends.__path__ = []
        cuda = ModuleType("comfy_kitchen.backends.cuda")
        backends.cuda = cuda
        comfy_kitchen.backends = backends

        def write_output(*args):
            args[4].fill_(3)

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "comfy_kitchen": comfy_kitchen,
                    "comfy_kitchen.backends": backends,
                    "comfy_kitchen.backends.cuda": cuda,
                },
            ),
            mock.patch.object(
                dispatch,
                "_kernel_op",
                return_value=mock.Mock(side_effect=write_output),
            ) as kernel_op,
        ):
            actual = dispatch._turing_int8_gemm(
                qactivation,
                weight,
                activation_scale,
                weight_scale,
                bias=None,
                output_dtype=torch.bfloat16,
                output=destination,
            )

        self.assertIs(actual, destination)
        self.assertEqual(actual.stride(), (12, 1))
        torch.testing.assert_close(actual, torch.full_like(actual, 3))
        kernel_op.assert_called_once_with("turing_int8_linear_out")

    def test_ampere_loader_preflights_the_shared_runtime(self):
        device = torch.device("cuda", 0)
        summary = SimpleNamespace(
            w4a4=0, w4a8=0, codebook_w4a8=0, w8a8=1
        )
        comfy_kitchen = ModuleType("comfy_kitchen")
        comfy_kitchen.list_backends = lambda: {
            "cuda": {
                "available": True,
                "disabled": False,
                "capabilities": ("int8_linear",),
            }
        }

        with (
            mock.patch.dict(sys.modules, {"comfy_kitchen": comfy_kitchen}),
            mock.patch.object(
                precision,
                "is_supported_tensor_core_device",
                return_value=True,
            ),
            mock.patch.object(
                precision, "is_supported_turing_device", return_value=False
            ),
            mock.patch.object(precision, "_check_kernel_contract"),
            mock.patch.object(precision, "_check_kitchen_contract"),
            mock.patch.object(precision, "register_backend", return_value=True),
            mock.patch.object(precision, "backend_available", return_value=True),
            mock.patch.object(precision, "preflight_kitchen") as linear,
            mock.patch.object(
                precision, "bundled_w8a8_available", return_value=True
            ),
            mock.patch.object(
                precision, "preflight_bundled_w8a8"
            ) as attention,
        ):
            precision.prepare_turing_runtime(summary, device, "w8a8")

        linear.assert_called_once_with(device, False, True)
        attention.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
