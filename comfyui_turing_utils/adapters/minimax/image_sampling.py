"""Sample MiniMax H3 Image without applying the AV scheduler or upscaler."""

import nodes
from comfy.ldm.minimax.sampling import MiniMaxH3ImageModelSampling

from ...log import get_logger
from .model_contract import is_h3_image_dit


LOG = get_logger("minimax.image")


def sample_h3_image(
    model,
    conditioning,
    latent,
    seed=0,
    denoise=1.0,
    *,
    steps=None,
    sampler_name="euler",
    scheduler="simple",
):
    if not is_h3_image_dit(model.get_model_object("diffusion_model")):
        raise ValueError("MiniMax H3 Image KSampler requires a MiniMax H3 Image MODEL.")

    model_sampling = model.get_model_object("model_sampling")
    if not isinstance(model_sampling, MiniMaxH3ImageModelSampling):
        raise ValueError(
            "MiniMax H3 Image needs its checkpoint's PDD schedule. "
            "Remove ModelSamplingMiniMaxH3 or other sampling overrides."
        )
    samples = latent["samples"]
    if (
        samples.is_nested
        or samples.ndim not in (4, 5)
        or samples.shape[0] != 1
        or (samples.ndim == 5 and samples.shape[2] != 1)
    ):
        raise ValueError("MiniMax H3 Image requires one single-frame image latent.")
    if steps is None:
        steps = model_sampling.grid.numel() - 1

    # ComfyUI owns noise generation, the PDD sigma schedule, the sampler and
    # its preview callback. The distilled image checkpoint uses CFG 1.
    return nodes.common_ksampler(
        model,
        seed,
        steps,
        1.0,
        sampler_name,
        scheduler,
        conditioning,
        [],
        latent,
        denoise=denoise,
    )[0]


def sample_legacy_h3_image(
    model, conditioning, latent, seed, sampler_name, sigmas, mode
):
    """Keep plain Image workflows saved with the shared AV sampler working."""
    if (mode or {}).get("mode", "single") != "single":
        raise ValueError(
            "MiniMax H3 Image supports single sampling only. Use MiniMax H3 Image KSampler."
        )
    settings = sigmas or {}
    if settings.get("sigmas", "scheduler") != "scheduler":
        raise ValueError(
            "MiniMax H3 Image does not use manual curves in MiniMax H3 KSampler. Use the native sampler for a custom curve."
        )
    if settings.get("tail_steps", 0):
        LOG.warning(
            "MiniMax H3 Image ignores AV tail steps. Use MiniMax H3 Image KSampler for the checkpoint schedule."
        )
    return sample_h3_image(
        model,
        conditioning,
        latent,
        seed,
        float(settings.get("denoise", 1.0)),
        steps=settings.get("steps"),
        sampler_name=sampler_name,
        scheduler=settings.get("scheduler", "simple"),
    )
