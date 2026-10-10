from __future__ import annotations

import sys
import unittest
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters import wan as wan_adapter  # noqa: E402
from comfyui_turing_utils.attention.protocol import (  # noqa: E402
    ATTENTION_EXECUTOR_KEY,
    AttentionExecutionOutcome,
)


class WanMemoryPlanningTest(unittest.TestCase):
    def test_prepared_attention_sites_install_without_quantized_weights(self):
        import comfy.ldm.wan.model as wan_model

        class FakeSelfAttention(torch.nn.Module):
            def forward(self, x, freqs, transformer_options={}):
                return x

        class FakeWanModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = FakeSelfAttention()
                self.second = FakeSelfAttention()

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = FakeWanModel()

        class Patcher:
            def __init__(self):
                self.model = Base()
                self.object_patches = {}

            def add_object_patch(self, key, value):
                self.object_patches[key] = value

        patcher = Patcher()
        with (
            mock.patch.object(wan_model, "WanModel", FakeWanModel),
            mock.patch.object(wan_model, "WanSelfAttention", FakeSelfAttention),
            mock.patch.object(
                wan_adapter,
                "is_supported_attention_device",
                return_value=True,
            ),
            mock.patch.object(
                wan_adapter,
                "fused_qk_preprocessing_available",
                return_value=True,
            ),
        ):
            first = wan_adapter.install_wan_attention_sites(
                patcher,
                torch.device("cuda", 0),
            )
            second = wan_adapter.install_wan_attention_sites(
                patcher,
                torch.device("cuda", 0),
            )

        self.assertEqual(first.model_kind, "wan")
        self.assertEqual(first.installed, 2)
        self.assertEqual(second.installed, 0)
        self.assertEqual(
            set(patcher.object_patches),
            {
                "diffusion_model.first.forward",
                "diffusion_model.second.forward",
            },
        )

    def test_self_attention_forward_uses_row_norm_fused_preprocessor(self):
        from comfy.ldm.modules.attention import AttentionTensorContainer

        class FakeAttention(torch.nn.Module):
            num_heads = 2
            head_dim = 64
            qk_norm = True
            eps = 1e-6

            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
                self.k = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
                self.v = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
                self.o = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
                self.norm_q = torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16)
                self.norm_k = torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16)

            def forward(self, x, freqs, transformer_options={}):
                raise AssertionError("the unfused forward should not run")

        attention = FakeAttention()
        seen = {}

        def processor(request):
            seen["shapes"] = tuple(value.shape for value in request.peek_qkv())
            seen["heads"] = request.heads
            seen["spec"] = request.qk_transform
            request.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros((1, 64, 128), dtype=torch.bfloat16)
            )

        patched = wan_adapter._make_self_attention_forward(
            attention, AttentionTensorContainer
        )
        x = torch.randn(1, 64, 128, dtype=torch.bfloat16)
        freqs = torch.randn(1, 64, 1, 32, 2, 2, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = patched(
                x,
                freqs,
                transformer_options={ATTENTION_EXECUTOR_KEY: processor},
            )

        self.assertEqual(output.shape, (1, 64, 128))
        self.assertEqual(
            seen["shapes"],
            (
                torch.Size((1, 2, 64, 64)),
                torch.Size((1, 2, 64, 64)),
                torch.Size((1, 2, 64, 64)),
            ),
        )
        self.assertEqual(seen["heads"], 2)
        self.assertEqual(seen["spec"].norm_scope, "row")
        self.assertFalse(seen["spec"].split_half)
        self.assertEqual(seen["spec"].rot_dim, 64)

    @staticmethod
    def _w4_weight(dtype: torch.dtype, linear_dtype: str):
        from comfy.quant_ops import QuantizedTensor, TensorCoreConvRotW4A4Layout

        params = TensorCoreConvRotW4A4Layout.Params(
            scale=torch.ones(4, dtype=torch.float32),
            orig_dtype=dtype,
            orig_shape=(4, 256),
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype=linear_dtype,
        )
        return QuantizedTensor(
            torch.zeros((4, 128), dtype=torch.uint8),
            "TensorCoreConvRotW4A4Layout",
            params,
        )

    def test_planning_format_is_independent_of_logical_activation_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    wan_adapter._convrot_planning_kind(self._w4_weight(dtype, "int4")),
                    "w4a4",
                )
                self.assertEqual(
                    wan_adapter._convrot_planning_kind(self._w4_weight(dtype, "int8")),
                    "w4a8",
                )

    def test_empty_context_has_no_synthetic_shape(self):
        self.assertIsNone(wan_adapter._context_latents_shape([]))

    def test_context_shape_aggregates_multiple_latents(self):
        latents = [
            torch.empty(1, 16, 5, 8, 8),
            torch.empty(1, 16, 3, 8, 8),
        ]
        self.assertEqual(
            wan_adapter._context_latents_shape(latents),
            [1, 16, 8 * 8 * 8],
        )

    def test_context_shape_sums_per_reference_padding(self):
        latents = [
            torch.empty(1, 16, 1, 3, 3),
            torch.empty(1, 16, 1, 3, 3),
        ]
        shape = wan_adapter._context_latents_shape(latents, (1, 2, 2))
        self.assertEqual(shape, [1, 16, 2 * 1 * 4 * 4])
        self.assertEqual(wan_adapter._shape_token_rows(shape, (1, 2, 2)), 8)

    def test_context_shape_uses_sampler_batch_hint(self):
        latents = [
            torch.empty(1, 16, 5, 8, 8),
            torch.empty(1, 16, 3, 8, 8),
        ]
        self.assertEqual(
            wan_adapter._context_latents_shape(
                latents, (1, 2, 2), estimate_batch_size=3
            ),
            [3, 16, 8 * 8 * 8],
        )

    def test_shape_token_rows_handles_spatial_and_flat_cond_shapes(self):
        self.assertEqual(
            wan_adapter._shape_token_rows([2, 16, 5, 8, 8], (1, 2, 2)),
            160,
        )
        self.assertEqual(
            wan_adapter._shape_token_rows([1, 16, 320], (1, 2, 2)),
            80,
        )

    def test_apply_installs_context_estimator_and_preserves_existing_shapes(self):
        from comfy.ldm.wan.model import WanModel

        diffusion = WanModel.__new__(WanModel)
        torch.nn.Module.__init__(diffusion)
        diffusion.patch_size = (1, 2, 2)

        class Base:
            memory_usage_factor_conds = ("existing",)
            diffusion_model = diffusion

            def extra_conds(self, **kwargs):
                return {}

            def extra_conds_shapes(self, **kwargs):
                return {"existing": [1, 1, 4]}

            def memory_required(self, input_shape, cond_shapes={}):
                return 100.0

        base = Base()
        patcher = SimpleNamespace(model=base)
        context = [torch.empty(1, 16, 3, 8, 8)]
        with (
            mock.patch(
                "comfyui_turing_utils.adapters.wan.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.adapters.wan._quantized_wan_summary",
                return_value=(Counter({"w8a8": 2}), (4096,), ()),
            ),
            mock.patch(
                "comfyui_turing_utils.adapters.wan.turing_int8_workspace_bytes",
                return_value=64.0,
            ),
        ):
            count = wan_adapter.apply_wan_adapter(patcher, torch.device("cuda", 0))

        self.assertEqual(count, 2)
        self.assertEqual(
            base.extra_conds_shapes(context_latents=context),
            {"existing": [1, 1, 4], "context_latents": [1, 16, 3 * 8 * 8]},
        )
        self.assertEqual(
            base.memory_usage_factor_conds, ("existing", "context_latents")
        )
        self.assertEqual(
            base.memory_required(
                [1, 16, 3, 8, 8],
                cond_shapes={"context_latents": [[1, 16, 3 * 8 * 8]]},
            ),
            1_572_964.0,
        )

    def test_workspace_uses_largest_live_accumulator_not_largest_output(self):
        class Base:
            def memory_required(self, input_shape, cond_shapes={}):
                return 0

        base = Base()
        base.memory_required = wan_adapter._make_memory_required(
            base, (1, 2, 2), (2048, 4096)
        )
        # 4096 columns cross the fixed-workspace threshold and need no global
        # accumulator, while 2048 columns still need a 40,960,000-byte buffer.
        self.assertEqual(
            base.memory_required([1, 16, 1, 100, 200]),
            5_000 * 2_048 * 4,
        )

    def test_outer_sample_wrapper_makes_context_estimate_batch_aware(self):
        base = SimpleNamespace()
        seen = []

        class Executor:
            def __call__(self, *args, **kwargs):
                seen.append(getattr(base, wan_adapter._MEMORY_CONTEXT_ATTR))
                return "ok"

        wrapper = wan_adapter._make_outer_sample_wrapper(base)
        result = wrapper(Executor(), torch.empty(3, 16, 5, 8, 8))
        self.assertEqual(result, "ok")
        self.assertEqual(seen, [{"batch_size": 3}])
        self.assertFalse(hasattr(base, wan_adapter._MEMORY_CONTEXT_ATTR))

    def test_runtime_context_condition_reports_padded_batch_shape(self):
        import comfy.conds

        latents = [
            torch.empty(1, 16, 1, 3, 3),
            torch.empty(1, 16, 1, 3, 3),
        ]

        class Base:
            def extra_conds(self, **kwargs):
                return {"context_latents": comfy.conds.CONDList(latents)}

        base = Base()
        base.extra_conds = wan_adapter._make_extra_conds(base, (1, 2, 2))
        cond = base.extra_conds()["context_latents"].process_cond(3)
        self.assertEqual(cond.size(), [1, 16, 3 * 2 * 1 * 4 * 4])


if __name__ == "__main__":
    unittest.main()
