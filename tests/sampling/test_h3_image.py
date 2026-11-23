"""H3 Image runs the real upstream DiT with a small checkpoint and PDD grid."""

import pytest
import torch

from turing_test_support.h3 import native_test_runtime  # noqa: F401

import comfy.model_base as models
import comfy.model_patcher
import comfy.ops
import comfy.supported_models_base
import latent_preview
import nodes
from comfy.latent_formats import MiniMaxH3Video

from comfyui_turing_utils.adapters.minimax import layout as bridge
from comfyui_turing_utils.adapters.minimax.memory_planning import (
    _install_memory_planning,
    _MEMORY_SHAPE_KEY,
)
from comfyui_turing_utils.adapters.minimax.veda.integration import (
    configure as configure_veda,
)
from comfyui_turing_utils.attention.layout import (
    ATTENTION_LAYOUT_KEY,
    has_complete_attention_layout,
)
from comfyui_turing_utils.attention.sparse import _sparse_protected_ranges
from comfyui_turing_utils.nodes.minimax_sampler import (
    MiniMaxH3ImageKSampler,
    MiniMaxH3KSampler,
    _H3KSamplerApply,
)


@pytest.fixture
def image_case():
    image_model = getattr(models, "MiniMaxH3Image", None)
    if image_model is None:
        pytest.skip("Installed ComfyUI does not have MiniMax H3 Image")
    device = torch.device("cpu")
    grid = [1.0, 0.9, 0.6, 0.25, 0.0]
    config = comfy.supported_models_base.BASE(
        {
            "hidden_size": 128,
            "num_layers": 1,
            "token_refiner_num_layers": 0,
            "num_attention_heads": 1,
            "attention_head_dim": 128,
            "ffn_hidden_size": 256,
            "text_dim": 128,
            "timestep_input_dim": 32,
            "time_embed_hidden_size": 64,
            "time_embed_dim": 32,
            "dtype": torch.float32,
            "pdd_steps": len(grid) - 1,
            "pdd_sigmas": grid,
        }
    )
    config.latent_format = MiniMaxH3Video()
    config.custom_operations = comfy.ops.disable_weight_init
    model = image_model(config, device=device)
    generator = torch.Generator().manual_seed(2048)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.01
                )
        model.diffusion_model.rope.inv_freq.copy_(torch.linspace(0.01, 0.8, 16))
    latent = {"samples": torch.zeros(1, 24, 1, 9, 13)}
    refs = [
        {
            "kind": "image",
            "latent_h": h,
            "latent_w": w,
            "latent": torch.randn(1, 24, 1, h, w, generator=generator),
        }
        for h, w in ((4, 6), (6, 8))
    ]
    positive = [[torch.randn(1, 12, 128, generator=generator), {"minimax_refs": refs}]]
    return (
        comfy.model_patcher.ModelPatcher(model, device, device),
        grid,
        latent,
        positive,
    )


def test_image_sampler_uses_checkpoint_steps_and_native_progress(
    image_case, monkeypatch
):
    patcher, grid, latent, positive = image_case
    steps = len(grid) - 1
    observed = []
    native_callback = latent_preview.prepare_callback

    def observe(model, steps):
        callback = native_callback(model, steps)

        def report(step, x0, x, total_steps):
            observed.append((step, total_steps))
            callback(step, x0, x, total_steps)

        return report

    monkeypatch.setattr(latent_preview, "prepare_callback", observe)
    (expected,) = nodes.common_ksampler(
        patcher, 42, steps, 1.0, "euler", "simple", positive, [], latent
    )
    observed.clear()
    (actual,) = MiniMaxH3ImageKSampler.execute(patcher, positive, latent, 42).result
    torch.testing.assert_close(actual["samples"], expected["samples"], rtol=0, atol=0)
    assert observed == [(step, steps) for step in range(steps)]
    (legacy,) = MiniMaxH3KSampler.execute(
        model=patcher,
        conditioning=positive,
        latent=latent,
        seed=42,
        sampler_name="euler",
        sigmas={"sigmas": "scheduler", "steps": steps, "tail_steps": 1},
        mode={"mode": "single"},
    ).result
    torch.testing.assert_close(legacy["samples"], expected["samples"], rtol=0, atol=0)


def test_legacy_image_workflows_reject_av_options_before_loading_upscaler(image_case):
    patcher, _, latent, positive = image_case
    mode = {
        "mode": "progressive",
        "lowres_scale": 0.5,
        "model_name": "missing.safetensors",
    }
    assert _H3KSamplerApply.check_lazy_status(latent, mode=mode, model=patcher) == []
    with pytest.raises(ValueError, match="single sampling only"):
        MiniMaxH3KSampler.execute(
            model=patcher, conditioning=positive, latent=latent, mode=mode
        )
    with pytest.raises(ValueError, match="manual curves"):
        MiniMaxH3KSampler.execute(
            model=patcher,
            conditioning=positive,
            latent=latent,
            sigmas={"sigmas": "manual", "values": "1, 0.5, 0"},
        )
    patcher.add_object_patch(
        "model_sampling",
        comfy.model_sampling.ModelSamplingAV(patcher.model.model_config),
    )
    with pytest.raises(ValueError, match="Remove ModelSamplingMiniMaxH3"):
        MiniMaxH3ImageKSampler.execute(patcher, positive, latent, 0)


def test_image_native_layout_and_memory_plan_include_exact_reference_rows(image_case):
    patcher, grid, latent, positive = image_case
    model = patcher.model
    dit = model.diffusion_model
    x, context, refs = latent["samples"], positive[0][0], positive[0][1]["minimax_refs"]
    shapes = [tuple(x.shape)]
    kwargs = dict(
        minimax_refs=refs,
        noise=x,
        device=x.device,
        cross_attn=context,
        latent_shapes=shapes,
    )
    assert _install_memory_planning(patcher, model, dit)
    sizing = model.extra_conds_shapes(**kwargs)[_MEMORY_SHAPE_KEY]
    encoded = model.extra_conds(**kwargs)
    payload = encoded["minimax_payload"].cond
    assert bridge.ensure_minimax_attention_layout_provider(patcher).installed
    options = {"sample_sigmas": torch.tensor(grid)}
    patcher.patch_model(load_weights=False)
    try:
        with torch.no_grad():
            out = dit(
                x,
                torch.tensor([900.0]),
                context,
                transformer_options=options,
                minimax_payload=payload,
            )
    finally:
        patcher.unpatch_model(unpatch_weights=False)

    native = options["minimax_h3_layout"]
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert has_complete_attention_layout(options, native.seq_len, provider="minimax_h3")
    layout = options[ATTENTION_LAYOUT_KEY]
    spans = [(s["start"], s["stop"], s["role"]) for s in layout["query_segments"]]
    assert spans == [
        (0, 12, "text"),
        (12, 18, "reference_image"),
        (18, 30, "reference_image"),
        (30, 65, "target_video"),
    ]
    for axis in ("query", "key"):
        assert _sparse_protected_ranges(
            "auto",
            0,
            options,
            native.seq_len,
            sparse_reference_image=False,
            sparse_reference_video=True,
            sparse_reference_audio=False,
            axis=axis,
        ) == ((0, 30),)
        assert _sparse_protected_ranges(
            "auto",
            0,
            options,
            native.seq_len,
            sparse_reference_image=True,
            sparse_reference_video=True,
            sparse_reference_audio=False,
            axis=axis,
        ) == ((0, 12),)
    for plan in (sizing, encoded[_MEMORY_SHAPE_KEY].cond):
        assert (
            plan.full_rows,
            plan.target_visual_rows,
            plan.visual_condition_rows,
        ) == (native.seq_len, 35, 18)
        assert (
            plan.target_audio_rows
            == plan.audio_condition_rows
            == plan.audio_row_width
            == 0
        )
    with pytest.raises(ValueError, match="AV only"):
        configure_veda(patcher, predictor_name="video_predictor.safetensors")
