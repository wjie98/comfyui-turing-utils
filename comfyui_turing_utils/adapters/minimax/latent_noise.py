"""Prepare clean H3 latents for a zero-noise continuation sampler."""

from __future__ import annotations

import math

import torch

from comfy.latent_formats import MiniMaxH3AV
from comfy.model_sampling import CONST, ModelSamplingAV
from comfy.nested_tensor import NestedTensor


def add_h3_noise_for_resampling(model, noise, sigmas, latent):
    latent_format = model.get_model_object("latent_format")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(latent_format, MiniMaxH3AV) or not isinstance(sampling, ModelSamplingAV) or not isinstance(sampling, CONST):
        raise ValueError("H3 Add Noise requires a MiniMax H3 model with FLOW_AV sampling")
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a one-dimensional continuation schedule")
    if sigmas.numel() == 0:
        return latent
    sigma_value = float(sigmas[0])
    if not math.isfinite(sigma_value) or not 0.0 <= sigma_value < 1.0:
        raise ValueError(
            "H3 Add Noise requires 0 <= sigmas[0] < 1. "
            "At sigma=1, DisableNoise discards the input latent; use normal noise initialization instead."
        )
    if sigma_value == 0.0:
        return latent

    samples = latent["samples"]
    nested = samples.is_nested
    streams = list(samples.unbind()) if nested else [samples]
    if nested and len(streams) != 2:
        raise ValueError("H3 AV latent must contain video and audio, in that order")
    audio_streams = []
    for index, stream in enumerate(streams):
        video = stream.ndim == 5 and stream.shape[1] == 24
        audio = stream.ndim == 4 and tuple(stream.shape[1:3]) == (32, 2)
        if not (video or audio) or (nested and audio != (index == 1)):
            raise ValueError("Expected H3 video [B,24,T,H,W] or audio [B,32,2,T]")
        audio_streams.append(audio)
    if nested and streams[0].shape[0] != streams[1].shape[0]:
        raise ValueError("H3 video and audio latent batch sizes must match")
    audio_scale = float(sampling.audio_scale)
    if not math.isfinite(audio_scale) or audio_scale <= 0:
        raise ValueError("H3 audio schedule scale must be finite and positive")

    # Generate once for the whole LATENT: RandomNoise owns seed/batch_index and
    # the RNG sequence across AV streams.
    generated = noise.generate_noise(latent)
    noise_streams = list(generated.unbind()) if generated.is_nested else [generated]
    if generated.is_nested != nested or len(noise_streams) != len(streams):
        raise ValueError("NOISE must return the same video/audio structure as the input latent")

    result = []
    for clean, epsilon, is_audio in zip(streams, noise_streams, audio_streams):
        if clean.shape != epsilon.shape:
            raise ValueError("NOISE must return the same shape as each input latent stream")
        # Match sampler precision; half storage must not round a near-one sigma
        # to one, or overflow the exported latent before the next sampler.
        dtype = torch.promote_types(torch.promote_types(clean.dtype, sigmas.dtype), torch.float32)
        sigma = sigmas[0].to(device=clean.device, dtype=dtype)
        clean = latent_format.process_in(clean.to(dtype=dtype))
        epsilon = epsilon.to(device=clean.device, dtype=dtype)
        carry = audio_scale if is_audio else 1.0

        # H3 carries audio on the video schedule at shift/audio_shift scale.
        # Do not call model.process_latent_in/out here: they depend on mutable
        # latent_shapes left by the last sampler, possibly at another resolution.
        noisy = sampling.noise_scaling(sigma, epsilon, clean * carry)
        prepared = sampling.inverse_noise_scaling(sigma, noisy)
        result.append(latent_format.process_out(prepared / carry))

    output = latent.copy()
    output["samples"] = NestedTensor(result) if nested else result[0]
    return output
