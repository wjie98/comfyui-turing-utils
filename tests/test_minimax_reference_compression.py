"""Reference compression preserves physical time, branch isolation and protection."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))

from comfy.ldm.minimax.model import PackedLayout
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor, WrappersMP
from comfyui_turing_utils.adapters.minimax import layout as h3_layout
from comfyui_turing_utils.adapters.minimax import reference_compression as vpack
from comfyui_turing_utils.adapters.minimax import memory_planning
from comfyui_turing_utils.adapters.minimax.sampling import resize_conditioning
from comfyui_turing_utils.adapters.minimax.veda.h3_layout import describe
from comfyui_turing_utils.attention.layout import attention_semantic_layout
from comfyui_turing_utils.attention.sparse_runtime import SparseProtection
from comfyui_turing_utils.nodes.attention import AttentionStrategy
from comfy_api.latest import _io as io


def video_reference(frames=12, *, audio_frames=0):
    latent = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1)
    latent = latent.expand(1, 24, frames, 16, 16).clone()
    return {
        "kind": "video_audio" if audio_frames else "video",
        "latent": latent,
        "latent_t": frames,
        "latent_h": 16,
        "latent_w": 16,
        "ref_audio_t": audio_frames,
        "audio_latent": torch.zeros(1, 32, 2, audio_frames) if audio_frames else None,
    }


def packed_refs(refs, ratio="3/5"):
    return vpack.compress_conditions({"positive": [{"minimax_refs": refs}]}, ratio)["positive"][0]["minimax_refs"]


def native_layout(refs, keyframes=None):
    return PackedLayout(65, 7, 16, 16, 11, refs=refs, keyframes=keyframes)


@pytest.mark.parametrize("ratio,expected", [
    ("4/5", [0, 1, 2, 3, 5, 6, 7, 8, 10, 11]),
    ("3/5", [0, 1, 3, 5, 6, 8, 10, 11]),
    ("2/5", [0, 3, 5, 8, 10, 11]),
])
def test_exact_retained_content_and_original_anchors(ratio, expected):
    original = video_reference()
    packed = packed_refs([original], ratio)[0]
    assert packed["latent"][0, 0, :, 0, 0].tolist() == expected
    assert packed["latent_t"] == len(expected)
    assert packed["latent"].shape[-2:] == original["latent"].shape[-2:]
    assert original["latent_t"] == 12
    assert "compact_ref2va_dense_indices" not in original
    assert packed_refs([packed], ratio)[0] is packed


def test_shared_positive_negative_reference_is_packed_once_without_changing_inputs():
    reference = video_reference(audio_frames=20)
    image = {"kind": "image", "latent": torch.zeros(1, 24, 1, 16, 16), "latent_h": 16, "latent_w": 16}
    entry = {"minimax_refs": [reference, image], "cross_attn": torch.randn(1, 8, 4)}
    conditions = {"positive": [entry], "negative": [entry]}
    result = vpack.compress_conditions(conditions, "3/5")
    a, b = (result[name][0]["minimax_refs"][0] for name in conditions)
    assert a["latent"] is b["latent"]
    assert a["audio_latent"] is reference["audio_latent"]
    assert result["positive"][0]["cross_attn"] is entry["cross_attn"]
    assert result["positive"][0]["minimax_refs"][1] is image
    assert conditions["positive"][0] is entry
    assert reference["latent_t"] == 12


@pytest.mark.parametrize("ratio", vpack.COMPRESSION_PHASES)
@pytest.mark.parametrize("audio_frames", [0, 25, 250])
def test_multi_reference_rope_matches_original_coordinates_including_target_guides(ratio, audio_frames):
    refs = [video_reference(audio_frames=audio_frames),
            {"kind": "image", "latent_h": 16, "latent_w": 16},
            {"kind": "audio", "ref_audio_t": 9}, video_reference(7)]
    keyframes = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 16, 16)},
                 {"resolved_frame_index": 21, "latent": torch.zeros(1, 24, 1, 16, 16),
                  "audio_latent": torch.zeros(1, 32, 2, 2)}]
    original = native_layout(refs, keyframes)
    packed = packed_refs(refs, ratio)
    layout = native_layout(packed, keyframes)
    vpack.restore_reference_times(layout, packed)
    visual = iter(ref for ref in packed if ref["kind"] in {"image", "video", "video_audio"})
    for (a, b, kind), (c, d, expected_kind) in zip(layout.segments, original.segments):
        assert kind == expected_kind
        expected = original.position_ids[c:d]
        if kind == "ref_img":
            ref = next(visual)
            if ref["kind"] != "image":
                ids = ref["compact_ref2va_dense_indices"]
                expected = expected.view(-1, 64, 3)[ids].reshape(-1, 3)
        torch.testing.assert_close(layout.position_ids[a:b], expected)
    positions = layout.position_ids
    assert vpack.restore_reference_times(layout, packed).position_ids is positions
    description = describe(layout)
    assert not description.skipped
    assert description.references[-1].grid[0] == packed[-1]["latent_t"]


def semantic_options(refs, layout):
    base = SimpleNamespace(
        _turing_utils_minimax_runtime_context={
            "packed_layout": layout, "refs": refs,
            "latent_shapes": [(1, 24, 7, 16, 16), (1, 32, 2, 11)],
        },
    )
    options = {}
    h3_layout.publish_minimax_attention_layout(
        options, layout.segments, layer_index=0, layer_count=2,
        base_model=base, diffusion_model=SimpleNamespace(patch_size=(1, 2, 2)),
    )
    return options


@pytest.mark.parametrize("ratio", vpack.COMPRESSION_PHASES)
def test_all_sparse_protection_switches_use_compressed_segments_and_true_anchors(ratio):
    refs = packed_refs([video_reference(), {
        "kind": "audio", "ref_audio_t": 9,
    }], ratio)
    layout = native_layout(refs)
    vpack.restore_reference_times(layout, refs)
    options = semantic_options(refs, layout)
    semantic = attention_semantic_layout(options)
    for image, video, audio in itertools.product((False, True), repeat=3):
        protection = SparseProtection("auto", 0, image, video, audio)
        for axis in ("query", "key"):
            ranges = protection.ranges(options, layout.seq_len, axis=axis)
            segments = semantic.query_segments if axis == "query" else semantic.key_segments
            allowed = {"reference_video_anchor": image, "reference_video": video, "reference_audio": audio}
            for segment in segments:
                if segment.role in allowed:
                    protected = any(a <= segment.start and segment.stop <= b for a, b in ranges)
                    assert protected is not allowed[segment.role]
    frames = refs[0]["latent"][0, 0, :, 0, 0]
    assert (frames[0].item(), frames[-1].item()) == (0, 11)
    assert SparseProtection("none").ranges(options, layout.seq_len) == ()
    assert SparseProtection("manual", 193).ranges(options, layout.seq_len) == ((0, 193),)


class Block(torch.nn.Module):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        return x


class Diffusion(torch.nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block()])

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        return minimax_payload["layout"]


def patcher():
    base = torch.nn.Module()
    base.diffusion_model = Diffusion()
    return ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))


def test_disabled_is_identity_and_changing_settings_does_not_stack_or_leak():
    model = patcher()
    assert vpack.configure_reference_compression(model) is model
    assert not model.get_all_wrappers(WrappersMP.PREPARE_SAMPLING)
    with mock.patch("comfy.ldm.minimax.model.MiniMaxH3Model", Diffusion):
        enabled = vpack.configure_reference_compression(model, "3/5")
        branch = vpack.configure_reference_compression(enabled, "2/5")
        disabled = vpack.configure_reference_compression(branch)
    assert len(enabled.get_all_wrappers(WrappersMP.PREPARE_SAMPLING)) == 1
    assert len(branch.get_all_wrappers(WrappersMP.PREPARE_SAMPLING)) == 1
    assert not disabled.get_all_wrappers(WrappersMP.PREPARE_SAMPLING)
    assert not model.get_all_wrappers(WrappersMP.PREPARE_SAMPLING)
    original = {"positive": [{"minimax_refs": [video_reference()]}]}
    for configured, expected in ((enabled, 8), (branch, 6), (disabled, 12)):
        executor = WrapperExecutor.new_executor(
            lambda model, shape, conditions: conditions,
            configured.get_all_wrappers(WrappersMP.PREPARE_SAMPLING),
        )
        for _ in range(2):
            result = executor.execute(configured, (1, 24, 7, 16, 16), original)
            assert result["positive"][0]["minimax_refs"][0]["latent_t"] == expected
    assert original["positive"][0]["minimax_refs"][0]["latent_t"] == 12


def test_sampling_preparation_sees_smaller_memory_plan_before_loading():
    model = patcher()
    with mock.patch("comfy.ldm.minimax.model.MiniMaxH3Model", Diffusion):
        enabled = vpack.configure_reference_compression(model, "3/5")
    conditions = {"positive": [{"minimax_refs": [video_reference()], "cross_attn": torch.zeros(1, 65, 8)}]}
    shape = [(1, 24, 7, 16, 16), (1, 32, 2, 11)]
    diffusion = SimpleNamespace(patch_size=(1, 2, 2), hidden_size=128, latents_dim=24, audio_latents_dim=32)
    original = memory_planning._minimax_memory_shape(conditions["positive"][0], shape, diffusion)

    def prepare(model, noise_shape, packed):
        estimate = memory_planning._minimax_memory_shape(packed["positive"][0], shape, diffusion)
        assert estimate.visual_condition_rows == 8 * 64
        assert estimate.full_rows == original.full_rows - 4 * 64
        return estimate

    executor = WrapperExecutor.new_executor(prepare, enabled.get_all_wrappers(WrappersMP.PREPARE_SAMPLING))
    executor.execute(enabled, shape, conditions)


def test_existing_h3_forward_bridge_restores_payload_before_model_and_layout_observers():
    model = patcher()
    refs = packed_refs([video_reference()])
    payload = {"refs": refs}
    with mock.patch("comfy.ldm.minimax.model.MiniMaxH3Model", Diffusion):
        vpack.configure_reference_compression(model, "3/5")
        h3_layout.ensure_minimax_attention_layout_provider(model)
    forward = model.object_patches["diffusion_model.forward"]
    result = forward(
        [torch.zeros(1, 24, 7, 16, 16), torch.zeros(1, 32, 2, 11)],
        torch.ones(1), torch.zeros(1, 65, 8), minimax_payload=payload,
    )
    expected = native_layout([video_reference()])
    torch.testing.assert_close(result.position_ids[-64:, 0], expected.position_ids[-64:, 0])
    assert "layout" not in payload
    assert not hasattr(model.model, h3_layout.RUNTIME_CONTEXT_ATTR)


def test_single_progressive_and_two_pass_conditions_reuse_original_reference_timeline():
    source = video_reference()
    condition = [[torch.zeros(1, 65, 8), {"minimax_refs": [source]}]]
    for _ in range(2):
        for stage in (condition, resize_conditioning(condition, 8, 8)):
            packed = vpack.compress_conditions({"positive": [stage[0][1]]}, "3/5")
            assert packed["positive"][0]["minimax_refs"][0]["latent_t"] == 8
    assert source["latent_t"] == 12


def test_single_vpack_schema_defaults_disabled_for_all_strategies():
    for strategy in ("sol", "sla", "veda", "spark", "none"):
        for ratio in ("5/5", *vpack.COMPRESSION_PHASES):
            values = {"strategy": strategy, "vpack": ratio}
            inputs, _, _ = io.get_finalized_class_inputs(AttentionStrategy.INPUT_TYPES(), values)
            fields = {**inputs["required"], **inputs["optional"]}
            assert "vpack" in fields
            assert not any(name.startswith("reference_video_compression") for name in fields)
    field = AttentionStrategy.GET_SCHEMA().inputs[-1]
    assert field.optional
    assert field.options == ["5/5", "4/5", "3/5", "2/5"]
    assert field.default == "5/5"


@pytest.mark.parametrize("strategy", ["sol", "sla", "veda", "spark", "none"])
def test_common_node_compression_does_not_change_attention_settings(strategy):
    from comfyui_turing_utils.nodes import attention
    model, configured, compressed = object(), object(), object()
    selection = "4/5"
    with mock.patch.dict(attention._ATTENTION_STRATEGIES, {strategy: mock.Mock(return_value=configured)}) as strategies:
        with mock.patch.object(attention, "configure_reference_compression", return_value=compressed) as configure:
            output = AttentionStrategy.execute(model, {"strategy": strategy}, selection)
        strategies[strategy].assert_called_once_with(model)
        configure.assert_called_once_with(configured, selection)
        assert output.result == (compressed,)


@pytest.mark.parametrize("ratio", [None, "5/5"])
def test_disabled_vpack_does_not_probe_model_or_install_wrappers(ratio):
    model = patcher()
    with mock.patch.object(h3_layout, "is_minimax_h3_model", side_effect=AssertionError("disabled VPack probed the model")):
        assert vpack.configure_reference_compression(model, ratio) is model
    assert not model.get_all_wrappers(WrappersMP.PREPARE_SAMPLING)


@pytest.mark.parametrize("ratio", ["1/5", "off", "none", "vpack"])
def test_invalid_keep_ratio_fails_before_configuring_model(ratio):
    with pytest.raises(ValueError, match="H3 vpack"):
        vpack.configure_reference_compression(patcher(), ratio)


@pytest.mark.skipif(os.environ.get("VEDA_TEST_CUDA") != "1", reason="opt-in GPU check")
@pytest.mark.parametrize("strategy", ["sol", "spark", "veda"])
@pytest.mark.parametrize("policy", ["auto", "manual", "none"])
@pytest.mark.parametrize("reference_flags", [(False, False, False), (False, True, False), (True, True, True)])
def test_vpack_cuda_attention_respects_reference_protection(strategy, policy, reference_flags):
    from comfyui_turing_utils.attention.sol import make_sparse_attention_override
    from comfyui_turing_utils.attention.spark import make_spark_attention_override
    from comfyui_turing_utils.adapters.minimax.veda.engine import VedaConfig, attend
    from comfyui_turing_utils.adapters.minimax.veda.predictor import PredictorBundle, convert_projection
    from comfyui_turing_utils.adapters.minimax.veda.plans import PlanTable, TilePlan
    from comfyui_turing_utils.adapters.minimax.veda.tiling import TileShape

    torch.manual_seed(122)
    refs = packed_refs([video_reference(audio_frames=9)], "2/5")
    layout = native_layout(refs)
    vpack.restore_reference_times(layout, refs)
    options = semantic_options(refs, layout)
    device = torch.device("cuda", 0)
    heads, dim = 2, 128
    q, k, v = [torch.randn(1, heads, layout.seq_len, dim, device=device, dtype=torch.float16) for _ in range(3)]

    def no_sdpa(*args, **kwargs):
        raise AssertionError("VPack sparse test fell back to SDPA")

    image, video, audio = reference_flags
    settings = dict(prefix_policy=policy, manual_prefix_tokens=193,
                    sparse_reference_image=image, sparse_reference_video=video,
                    sparse_reference_audio=audio, dense_prefix_steps=0, dense_prefix_layers=0)
    if strategy == "sol":
        override = make_sparse_attention_override(device, min_sequence_tokens=128, **settings)
    elif strategy == "spark":
        override = make_spark_attention_override(device, min_sequence_tokens=128, **settings)
    else:
        projection = [convert_projection(torch.randn(heads, 384, dim) * 0.03, "w8a8") for _ in range(2)]
        plan = TilePlan("vpack-test", (7, 8, 8), [TileShape(1, 8, 16)], [[0, 0]])
        bundle = PredictorBundle("w8a8", 1, heads, dim, 0.5, PlanTable([plan]),
                                 (projection[0],), (projection[1],))
        protection = SparseProtection(policy, 193, image, video, audio)
        config = VedaConfig(bundle, keep_ratio=0.5, reference_keep_ratio=0.5, protection=protection)
    with mock.patch("torch.nn.functional.scaled_dot_product_attention", side_effect=no_sdpa):
        if strategy == "veda":
            result = attend(q, k, v, config=config, packed_layout=layout, layer=0,
                            cache={}, transformer_options=options)
        else:
            result = override(no_sdpa, q, k, v, heads, skip_reshape=True,
                              skip_output_reshape=True, transformer_options=options)
    assert result.shape == q.shape
    assert result.dtype == q.dtype
    assert torch.isfinite(result).all()
