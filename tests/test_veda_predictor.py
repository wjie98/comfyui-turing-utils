from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from comfyui_turing_utils.adapters.minimax.veda.predictor import (
    FORMAT,
    PRECISIONS,
    Projection,
    convert_projection,
    load_bundle,
    predictor_compute_dtype,
    project_features,
    rotate_features,
)
from .veda_pooling_reference import pool_video_tiles
from .veda_selection_reference import select_tiles
from .veda_tiling_reference import gather_tiles
from comfyui_turing_utils.adapters.minimax.veda.tiling import (
    TileShape,
    TiledSpan,
    build_tile_layout,
)


def test_bf16_sm75_emulates_arithmetic_not_output_dtype():
    gen = torch.Generator().manual_seed(5)
    features = torch.randn(2, 11, 384, generator=gen) * 0.7
    weights = torch.randn(2, 384, 128, generator=gen) * 0.03
    projection = convert_projection(weights, "bf16")
    actual = project_features(features, projection, "bf16", capability=(7, 5))
    x = features.bfloat16()
    projected = torch.bmm(x.float(), weights.bfloat16().float())
    expected = (projected + x[..., :128].float()).bfloat16()
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old = (projected.bfloat16().float() + x[..., :128].float()).bfloat16()
    assert not torch.equal(actual, old)
    assert predictor_compute_dtype("bf16", (7, 5)) == torch.float32
    assert predictor_compute_dtype("bf16", (8, 6)) == torch.bfloat16


def test_fp32_debug_does_not_round_activations_to_bf16():
    x = torch.full((1, 2, 12), 1.001, dtype=torch.float32)
    projection = convert_projection(torch.zeros(1, 12, 4), "fp32")
    actual = project_features(x, projection, "fp32", capability=(7, 5))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, x[..., :4], rtol=0, atol=0)
    assert not torch.equal(actual, actual.bfloat16().float())


def test_predictor_head_selection_keeps_order_and_avoids_contiguous_copy():
    weight = torch.arange(4 * 12 * 4).reshape(4, 12, 4).float()
    projection = Projection(weight, None)
    sliced, scale = projection.select_heads([1, 2])
    assert scale is None
    assert sliced.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    torch.testing.assert_close(sliced, weight[1:3], atol=0, rtol=0)
    selected, _ = projection.select_heads([3, 1, 3])
    torch.testing.assert_close(selected, weight[[3, 1, 3]], atol=0, rtol=0)
    with pytest.raises(IndexError):
        projection.select_heads([3, 4])


def test_rotation_is_kitchen_regular_convrot():
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    h = torch.kron(torch.kron(torch.kron(h4, h4), h4), h4) / 16
    x = torch.randn(3, 512)
    rotated = rotate_features(x)
    expected = (x.reshape(3, 2, 256) @ h).reshape_as(x)
    torch.testing.assert_close(rotated, expected, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(rotate_features(rotated), x, atol=1e-6, rtol=1e-5)


def test_w8a8_conversion_preserves_projection_basis():
    torch.manual_seed(42)
    x = torch.randn(2, 31, 384)
    weight = torch.randn(2, 384, 128) * 0.03
    projection = convert_projection(weight, "w8a8")
    assert projection.weight.shape == (2, 128, 512)
    assert projection.weight.dtype == torch.int8
    rotated = rotate_features(torch.nn.functional.pad(x, (0, 128)))
    actual = torch.bmm(
        rotated, (projection.weight.float() * projection.scale).transpose(1, 2)
    )
    expected = torch.bmm(x, weight)
    relative = (actual - expected).norm() / expected.norm()
    assert relative < 0.012
    assert projection.weight.device.type == "cpu"


def test_pool_partial_tile_excludes_padding_extrema():
    layout = build_tile_layout([TiledSpan(0, (1, 1, 5), TileShape(1, 8, 16))], 5)
    x = torch.arange(1, 6, dtype=torch.float32).reshape(5, 1, 1)
    heads = torch.tensor([0])
    features = pool_video_tiles(gather_tiles(x, layout, heads), layout)
    torch.testing.assert_close(features, torch.tensor([[[3.0, 5.0, 1.0]]]))
    features_negative = pool_video_tiles(gather_tiles(-x, layout, heads), layout)
    torch.testing.assert_close(features_negative, torch.tensor([[[-3.0, -1.0, -5.0]]]))


def test_chunked_routing_preserves_diagonal_and_fractional_budget():
    layout = build_tile_layout([TiledSpan(0, (1, 8, 160), TileShape(1, 8, 16))], 1280)
    scores = torch.randn(2, 10, 10)
    whole_index, whole_keep = select_tiles(scores, layout, 0.23, 1.0)
    chunks = [
        select_tiles(scores[:, a:b], layout, 0.23, 1.0, row_start=a)
        for a, b in ((0, 3), (3, 7), (7, 10))
    ]
    assert torch.equal(whole_index, torch.cat([i for i, _ in chunks], 1))
    assert torch.equal(whole_keep, torch.cat([k for _, k in chunks], 1))
    for row in range(10):
        assert ((whole_index[:, row] == row) & whole_keep[:, row]).any(-1).all()
    assert whole_keep[0].sum() == 23


def test_reference_budget_is_independent():
    layout = build_tile_layout(
        [
            TiledSpan(0, (1, 8, 32), TileShape(1, 8, 16)),
            TiledSpan(256, (1, 8, 64), TileShape(1, 8, 16)),
        ],
        768,
    )
    scores = torch.zeros(1, 6, 6)
    scores[..., 2:] = 100
    index, keep = select_tiles(scores, layout, 0.25, 1.0)
    assert (index[..., :2] == torch.tensor([0, 1])).all()
    assert keep[..., :2].all()
    assert keep.sum(-1).eq(3).all()


@pytest.mark.parametrize("precision", PRECISIONS)
def test_bundle_converts_on_host_and_checks_metadata(tmp_path, precision):
    path = tmp_path / "predictor.safetensors"
    metadata = dict(
        format=FORMAT,
        num_layers="1",
        num_heads="2",
        head_dim="4",
        keep_ratio="0.1",
        dtype="bfloat16",
        plans=json.dumps(
            {
                "test": {
                    "geometry": "test",
                    "grid": [1, 8, 16],
                    "shapes": ["1x8x16"],
                    "head_shape": [[0, 0]],
                }
            }
        ),
    )
    save_file(
        {
            f"layers.0.proj_{name}": torch.randn(2, 12, 4).bfloat16()
            for name in ("q", "k")
        },
        str(path),
        metadata=metadata,
    )
    bundle = load_bundle(str(path), precision)
    assert bundle.precision == precision
    assert bundle.proj_q[0].weight.device.type == "cpu"
    assert bundle.staged_bytes_per_head > 0
    assert bundle.plans.select((1, 8, 16)).exact
    metadata["format"] = "not-veda"
    save_file({}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="Not a Veda"):
        load_bundle(str(path), precision)
