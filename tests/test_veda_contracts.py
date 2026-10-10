import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from comfyui_turing_utils.adapters.minimax.veda.integration import (
    resolve_keep_ratio,
    make_override,
)
from comfyui_turing_utils.adapters.minimax.veda.engine import (
    VedaConfig,
    workspace_per_head,
)
from comfyui_turing_utils.adapters.minimax.veda.plans import PlanTable, TilePlan
from comfyui_turing_utils.adapters.minimax.veda.tiling import TileShape
from .veda_reference import official_logits, route_agreement
from comfyui_turing_utils.adapters.minimax.veda.predictor import (
    convert_projection,
    project_features,
    score_tiles,
)
from comfyui_turing_utils.nodes.attention import (
    veda_inputs,
)


def test_removed_cuda_variants_have_no_declarations_or_dispatch():
    root = Path(__file__).resolve().parents[1] / "kernel/csrc/turing/sage"
    for name in ("sol_sparse_cuda_sm75.cu", "attn_cuda_sm75.h", "pybind_sm75.cpp"):
        source = (root / name).read_text()
        for removed in (
            "veda_sparse_ragged_attn",
            "veda_sparse_heterogeneous_attn",
            "RaggedVeda",
            "heterogeneous_descriptors",
            "tile_metadata_stride",
        ):
            assert removed not in source


def test_ratios_use_bundle_without_changing_explicit_workflows():
    assert resolve_keep_ratio(0, 0.17) == 0.17
    assert resolve_keep_ratio(1, 0.17) == 1
    assert resolve_keep_ratio(0.1, 0.17) == 0.1
    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            resolve_keep_ratio(value, 0.1)


def test_advanced_inputs_use_trained_defaults():
    with mock.patch(
        "comfyui_turing_utils.nodes.attention.predictor_choices", return_value=["x"]
    ):
        inputs = veda_inputs()["required"]
    assert [
        name
        for name, spec in inputs.items()
        if len(spec) < 2 or not spec[1].get("advanced")
    ] == ["model", "predictor_name", "predictor_precision"]
    assert inputs["keep_ratio"][1]["default"] == 0
    assert inputs["reference_keep_ratio"][1]["default"] == 0


def test_w8a8_workspace_includes_full_sequence_value_quantization_overlap():
    from comfyui_turing_utils.adapters.minimax.veda.selection import (
        estimate_workspace_bytes,
    )

    sizes = dict(
        slots=1024,
        video_tiles=7,
        heads=3,
        head_dim=128,
        projection_bytes_per_head=1000,
        element_size=2,
        score_rows=128,
        fused_prepare=True,
        head_major_prepare=True,
    )
    for chunk in (0, 2):
        baseline = estimate_workspace_bytes(**sizes, chunk_tiles=chunk)
        rotated = estimate_workspace_bytes(**sizes, chunk_tiles=chunk, use_w8a8=True)
        assert rotated - baseline == 3 * (1024 * 128 + 128 * 4)


def test_transposed_plan_is_not_claimed_as_native_training():
    plan = TilePlan("landscape", (2, 8, 16), [TileShape(1, 8, 16)], [[0]])
    table = PlanTable([plan])
    assert table.select((2, 8, 16)).exact
    choice = table.select((2, 16, 8))
    assert not choice.exact
    assert "transposed" in choice.how


def test_full_dense_needs_no_predictor_workspace_or_layout():
    config = VedaConfig(None, 1, 1)
    assert workspace_per_head(config, None, 0, 2, (7, 5)) == 0
    dense = mock.Mock(return_value="dense")
    schedule = mock.Mock()
    schedule.is_dense.return_value = True
    override = make_override(config, dense, schedule)
    q = torch.zeros(1, 2, 3, 128)
    with mock.patch(
        "comfyui_turing_utils.adapters.minimax.veda.integration._attention_layer_metadata",
        return_value=(0, 1),
    ):
        assert (
            override(
                None,
                q,
                q,
                q,
                2,
                skip_reshape=True,
                transformer_options={"minimax_h3_layout": SimpleNamespace(seq_len=3)},
            )
            == "dense"
        )
    schedule.is_dense.assert_called_once()


def test_official_oracle_uses_bf16_weights_but_fp32_features_and_residuals():
    torch.manual_seed(71)
    x, y = torch.randn(2, 7, 12), torch.randn(2, 9, 12)
    wq, wk = torch.randn(2, 12, 4), torch.randn(2, 12, 4)
    actual = official_logits(x, y, wq, wk)
    q = x @ wq.bfloat16().float() + x[..., :4]
    k = y @ wk.bfloat16().float() + y[..., :4]
    torch.testing.assert_close(
        actual, q @ k.transpose(1, 2) / math.sqrt(4), rtol=0, atol=0
    )
    bf16 = score_tiles(
        project_features(x, convert_projection(wq, "bf16"), "bf16", capability=(7, 5)),
        project_features(y, convert_projection(wk, "bf16"), "bf16", capability=(7, 5)),
    )
    assert not torch.equal(actual, bf16)


def test_route_recall_excludes_padding_and_is_order_independent():
    ref = torch.tensor([[[1, 2, 99]]])
    valid = torch.tensor([[[True, True, False]]])
    assert route_agreement(ref, valid, torch.tensor([[[2, 1, 0]]]), valid) == 1
    assert route_agreement(ref, valid, torch.tensor([[[2, 3, 0]]]), valid) == 0.5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_once_per_group_scoring_cast_preserves_chunk_logits(dtype):
    torch.manual_seed(42)
    q, k = torch.randn(3, 19, 128).to(dtype), torch.randn(3, 19, 128).to(dtype)
    q32, k32 = q.float(), k.float()
    for start in range(0, 19, 7):
        torch.testing.assert_close(
            score_tiles(q[:, start : start + 7], k),
            score_tiles(q32[:, start : start + 7], k32),
            atol=0,
            rtol=0,
        )
