"""H3 single, progressive and independent two-pass sampling on one final canvas."""

from __future__ import annotations

import logging
import math
from time import perf_counter

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
from comfy.latent_formats import MiniMaxH3AV, MiniMaxH3Video
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy.nested_tensor import NestedTensor

from ...media.latent_masks import binary_video_latent_mask, _validate_mask_values
from ...runtime.sampling import NativeStepSampler, SamplingProgress
from .sampling_sigmas import sampling_sigmas
from .sampling_masks import close_patch_mask, low_resolution_mask
from .latent_upscaler import (
    _av_streams,
    upscale_h3_video_to_size,
)


def _latent_samples(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a LATENT dictionary containing samples.")
    return latent["samples"]


def _latent_streams(latent):
    return _av_streams(_latent_samples(latent))


def _masks(latent, video, audio):
    mask = latent.get("noise_mask")
    audio_mask = None
    if getattr(mask, "is_nested", False):
        streams = list(mask.unbind())
        if audio is None or len(streams) != 2:
            raise ValueError("An H3 AV noise_mask must contain video and audio masks.")
        mask, audio_mask = streams
        _validate_mask_values(audio_mask)
        if torch.broadcast_shapes(audio_mask.shape, audio.shape) != audio.shape:
            raise ValueError(
                "Audio noise_mask must broadcast to the first audio latent."
            )
    video_mask = None if mask is None else binary_video_latent_mask(video, mask)
    return video_mask, audio_mask


def low_resolution_size(video, mode):
    scale = mode.get("lowres_scale", 0.75)
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Low-resolution width/height scale must be in (0, 1].")
    return tuple(
        min(size, max(2, round(size * scale / 2) * 2)) for size in video.shape[-2:]
    )


def needs_upscaler(latent, mode=None):
    """Cheap lazy-loader decision; also used when executing a node outside a prompt."""
    mode = mode or {}
    if mode.get("mode", "single") == "single":
        return False
    video, _ = _latent_streams(latent)
    if tuple(video.shape[-2:]) == low_resolution_size(video, mode):
        return False
    # This is a loader probe. Full video/audio mask validation belongs to sampling.
    mask = latent.get("noise_mask")
    if getattr(mask, "is_nested", False):
        mask = mask.unbind()[0]
    return mask is None or bool(mask.any())


def _with_streams(latent, video, audio, video_mask=None, audio_mask=None):
    output = latent.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = video if audio is None else NestedTensor((video, audio))
    output.pop("noise_mask", None)
    if video_mask is not None or audio_mask is not None:
        if video_mask is None:
            video_mask = torch.ones_like(video[:, :1], dtype=torch.float32)
        else:
            video_mask = video_mask.float()
        if audio is None:
            output["noise_mask"] = video_mask
        else:
            if audio_mask is None:
                audio_mask = torch.ones_like(audio[:, :1], dtype=torch.float32)
            output["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return output


def _validate_conditioning(conditioning, video, audio, label):
    if not isinstance(conditioning, (tuple, list)) or not conditioning:
        raise ValueError(f"{label} must contain CONDITIONING entries.")
    frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
    for entry in conditioning:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not isinstance(entry[1], dict)
        ):
            raise ValueError(f"{label} must contain [embedding, options] entries.")
        keyframes = entry[1].get("minimax_keyframes") or ()
        if not isinstance(keyframes, (tuple, list)):
            raise ValueError(f"{label} minimax_keyframes must be a list.")
        for keyframe in keyframes:
            if not isinstance(keyframe, dict):
                raise ValueError(f"Invalid keyframe in {label}.")
            start = keyframe.get("resolved_frame_index", 0)
            if not isinstance(start, int) or not 0 <= start < frame_count:
                raise ValueError(
                    f"{label} keyframe index {start} is outside the first latent's {frame_count} frames."
                )
            guide = keyframe.get("latent")
            if guide is not None:
                guide, _ = _av_streams(guide)
                if tuple(guide.shape[-2:]) != tuple(video.shape[-2:]):
                    raise ValueError(
                        f"{label} keyframe H/W {tuple(guide.shape[-2:])} does not match "
                        f"final canvas H/W {tuple(video.shape[-2:])}. Build conditioning at the final canvas size."
                    )
                if guide.shape[0] not in (1, video.shape[0]):
                    raise ValueError(
                        f"{label} keyframe batch must be 1 or match the first latent."
                    )
                guide_frames = sum(
                    FRAME_PER_TOKEN[k % 5] for k in range(guide.shape[2])
                )
                if start + guide_frames > frame_count:
                    raise ValueError(
                        f"{label} guide clip exceeds the first latent's time axis."
                    )
            guide_audio = keyframe.get("audio_latent")
            if guide_audio is not None:
                if audio is None:
                    raise ValueError(
                        f"{label} includes guide audio but the first latent has no audio stream."
                    )
                if (
                    not torch.is_tensor(guide_audio)
                    or guide_audio.ndim != 4
                    or tuple(guide_audio.shape[1:3]) != (32, 2)
                    or guide_audio.shape[0] not in (1, audio.shape[0])
                    or guide_audio.shape[-1] < 1
                    or guide_audio.shape[-1]
                    > math.floor(audio.shape[-1] - FRAME_RESCALE * start)
                ):
                    raise ValueError(
                        f"{label} guide audio does not fit the first latent's audio timeline."
                    )
        # Ref2VA references intentionally have their own resolution and timeline.


def resize_video(video, height, width):
    """Resize each frame independently; never interpolate the temporal axis."""
    if tuple(video.shape[-2:]) == (height, width):
        return video
    batch, channels, frames, h, w = video.shape
    pixels = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, h, w)
    resized = F.interpolate(
        pixels.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        resized.reshape(batch, frames, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .to(video.dtype)
    )


def resize_conditioning(conditioning, height, width):
    """Derive stage keyframes without touching semantic or Ref2VA references."""
    result = []
    outputs = {}
    changed = False
    for embedding, options in conditioning:
        keyframes = options.get("minimax_keyframes")
        if not keyframes:
            result.append([embedding, options])
            continue
        resized = []
        changed_keyframes = False
        for keyframe in keyframes:
            guide = keyframe.get("latent")
            if guide is not None:
                video, audio = _av_streams(guide)
                if tuple(video.shape[-2:]) != (height, width):
                    if id(video) not in outputs:
                        low = resize_video(video, height, width)
                        # Preserve channel/frame means lost through spatial resampling.
                        outputs[id(video)] = low + (
                            video.float().mean((-2, -1), keepdim=True)
                            - low.float().mean((-2, -1), keepdim=True)
                        ).to(low)
                    low = outputs[id(video)]
                    resized.append(
                        {
                            **keyframe,
                            "latent": (
                                low if audio is None else NestedTensor((low, audio))
                            ),
                        }
                    )
                    changed_keyframes = True
                    continue
            resized.append(keyframe)
        if not changed_keyframes:
            result.append([embedding, options])
            continue
        stage_options = {**options, "minimax_keyframes": resized}
        stage_options.pop("layout", None)
        result.append([embedding, stage_options])
        changed = True
    return result if changed else conditioning


def _restore_protected(result, baseline, video_mask, original_audio, audio_mask):
    if video_mask is None and audio_mask is None:
        return result
    video, audio = _latent_streams(result)
    if video_mask is not None:
        video = torch.where(video_mask.to(video.device), video, baseline.to(video))
    if audio_mask is not None:
        audio = torch.where(
            audio_mask.to(audio.device) > 0, audio, original_audio.to(audio)
        )
    return {**result, "samples": NestedTensor((video, audio))}


def _output_structure(result, video_only):
    if not video_only:
        return result
    output = result.copy()
    output["samples"] = result["samples"].unbind()[0]
    mask = output.get("noise_mask")
    if mask is not None:
        output["noise_mask"] = mask.unbind()[0]
    return output


class H3SamplingTask:
    """One execution's progress; each numerical run remains owned by ComfyUI."""

    def __init__(self, model, seed, sampler_name, total_steps):
        self.model = model
        self.seed = seed
        self.sampler_name = sampler_name
        self.progress = SamplingProgress(total_steps)
        self.completed_steps = 0
        self.preview = self.console = None

    def sample_stage(self, latent, conditioning, sigmas, *, noise=None, capture=None):
        if sigmas.numel() < 2:
            return latent
        if self.preview is None:
            self.preview = latent_preview.prepare_callback(
                self.model, self.progress.total_steps
            )
        offset = self.completed_steps

        def callback(step, x0, x, _stage_steps):
            if capture is not None and step == len(sigmas) - 2:
                capture(x0, x)
            global_step = (
                self.progress.step
                if self.sampler_name in {"dpm_fast", "dpm_adaptive"}
                else offset + step
            )
            if self.preview is not None:
                self.preview(global_step, x0, x, self.progress.total_steps)
            if self.console is not None:
                self.console.update(max(0, global_step + 1 - self.console.n))

        samples = latent["samples"]
        if noise is None:
            noise = comfy.sample.prepare_noise(
                samples, self.seed, latent.get("batch_index")
            )
        started = perf_counter()
        result = comfy.samplers.sample(
            self.model,
            noise,
            conditioning,
            [],
            1.0,
            self.model.load_device,
            NativeStepSampler(self.sampler_name, self.progress, offset),
            sigmas,
            model_options=self.model.model_options,
            latent_image=samples,
            denoise_mask=latent.get("noise_mask"),
            callback=callback,
            disable_pbar=True,
            seed=self.seed,
        )
        if capture is None:
            result = result.to(
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype(),
            )
        self.completed_steps += len(sigmas) - 1
        video, _ = _latent_streams(latent)
        logging.info(
            "H3 stage: %dx%d, %d intervals, %.2fs wall time",
            video.shape[-1] * 16,
            video.shape[-2] * 16,
            len(sigmas) - 1,
            perf_counter() - started,
        )
        return None if capture is not None else {**latent, "samples": result}

    def run(
        self,
        source,
        conditioning,
        curve,
        second_curve,
        mode,
        upscale_model,
        patch_size,
        canonical_mask,
        audio_mask,
    ):
        video, audio = _latent_streams(source)
        kind = mode.get("mode", "single")
        height, width = video.shape[-2:]
        low_h, low_w = (
            (height, width) if kind == "single" else low_resolution_size(video, mode)
        )
        resized = (height, width) != (low_h, low_w)
        redraw_video = (
            not resized or canonical_mask is None or bool(canonical_mask.any())
        )
        if resized and redraw_video and upscale_model is None:
            raise ValueError(
                "Select an H3 upscaler model for the resolution transition."
            )
        grow = mode.get("mask_grow", 0) if kind != "single" else 0
        low_mask = high_mask = None
        if canonical_mask is not None:
            if not redraw_video:
                high_mask = canonical_mask
                low_mask = canonical_mask.new_zeros(
                    (*canonical_mask.shape[:-2], low_h, low_w)
                )
            else:
                high_mask = close_patch_mask(canonical_mask, patch_size, grow)
                if resized:
                    low_mask = low_resolution_mask(high_mask, low_h, low_w, patch_size)
                else:
                    low_mask = (
                        close_patch_mask(canonical_mask, patch_size)
                        if grow
                        else high_mask
                    )
        low_video = resize_video(video, low_h, low_w) if resized else video
        low = _with_streams(source, low_video, audio, low_mask, audio_mask)
        low_condition = (
            conditioning
            if not resized or curve.numel() < 2
            else resize_conditioning(conditioning, low_h, low_w)
        )
        logging.info(
            "H3 plan: %s, %dx%d -> %dx%d, %d+%d intervals, video/audio shift %s/%s",
            kind,
            low_w * 16,
            low_h * 16,
            width * 16,
            height * 16,
            max(0, len(curve) - 1) if kind != "progressive" else mode["lowres_steps"],
            (
                0
                if kind == "single"
                else (
                    max(0, len(second_curve) - 1)
                    if kind == "two_pass"
                    else len(curve) - 1 - mode["lowres_steps"]
                )
            ),
            self.model.get_model_object("model_sampling").shift,
            self.model.get_model_object("model_sampling").audio_shift,
        )
        logging.info(
            "H3 sigmas: main=%s%s",
            curve.detach().cpu().tolist(),
            (
                ""
                if second_curve is None
                else f"; second={second_curve.detach().cpu().tolist()}"
            ),
        )
        self.console = comfy.utils.model_trange(
            self.progress.total_steps, disable=not comfy.utils.PROGRESS_BAR_ENABLED
        )
        try:
            if kind == "progressive":
                return self.progressive(
                    source,
                    low,
                    low_condition,
                    conditioning,
                    curve,
                    mode["lowres_steps"],
                    upscale_model,
                    high_mask,
                    audio_mask,
                    redraw_video,
                )
            clean = self.sample_stage(low, low_condition, curve)
            clean = _restore_protected(clean, low_video, low_mask, audio, audio_mask)
            if kind == "single":
                return clean
            clean_video, clean_audio = _latent_streams(clean)
            high_video = (
                self.lift(clean_video, video, high_mask, upscale_model)
                if redraw_video
                else video
            )
            high = _with_streams(source, high_video, clean_audio, high_mask, audio_mask)
            del clean, clean_video, clean_audio
            result = self.sample_stage(high, conditioning, second_curve)
            return _restore_protected(result, video, high_mask, audio, audio_mask)
        finally:
            self.console.close()
            self.console = self.preview = None

    def lift(self, clean_video, original_video, high_mask, upscale_model):
        if clean_video.shape[-2:] == original_video.shape[-2:]:
            return clean_video
        started = perf_counter()
        high = upscale_h3_video_to_size(
            upscale_model, clean_video, *original_video.shape[-2:]
        )
        if high_mask is not None:
            high = torch.where(high_mask.to(high.device), high, original_video.to(high))
        logging.info(
            "H3 resolution transition: %.2fs wall time", perf_counter() - started
        )
        return high

    def progressive(
        self,
        source,
        low,
        low_condition,
        high_condition,
        curve,
        split,
        upscale_model,
        high_mask,
        audio_mask,
        redraw_video,
    ):
        sampling = self.model.get_model_object("model_sampling")
        latent_format = self.model.get_model_object("latent_format")
        transition = {}
        sigma, next_sigma = curve[split - 1], curve[split]
        same_size = (
            _latent_streams(low)[0].shape[-2:] == _latent_streams(source)[0].shape[-2:]
        )

        def capture(x0, x):
            predicted_video, predicted_audio = _av_streams(x0)
            state_video, state_audio = _av_streams(x)
            if redraw_video:
                transition["video"] = latent_format.process_out(predicted_video.float())
            # The audio state completes the very same Euler interval. Spatial
            # lifting must not replace or re-noise it as a second generation.
            predicted_audio = predicted_audio.float()
            state_audio = state_audio.float()
            ratio = ((next_sigma - sigma) / sigma).to(state_audio)
            transition["audio"] = state_audio + (state_audio - predicted_audio) * ratio
            if same_size:
                transition["video_state"] = state_video.float() + (
                    state_video.float() - predicted_video.float()
                ) * ratio.to(state_video)

        self.sample_stage(low, low_condition, curve[: split + 1], capture=capture)
        if not transition:
            raise RuntimeError(
                "The progressive Euler boundary did not receive its native callback."
            )
        original_video, original_audio = _latent_streams(source)
        high_video = (
            self.lift(transition.pop("video"), original_video, high_mask, upscale_model)
            if redraw_video
            else original_video
        )
        high = _with_streams(
            source,
            high_video,
            original_audio.to(high_video.device),
            high_mask,
            audio_mask,
        )
        # Enlargements lift the clean prediction with a new video noise field.
        # An unchanged grid and audio resume their actual Euler states instead.
        if same_size:
            state = transition.pop("video_state")
            anchor = latent_format.process_in(high_video.to(state))
            s = next_sigma.to(state)
            video_noise = ((state - (1 - s) * anchor) / s).cpu()
        else:
            video_noise = comfy.sample.prepare_noise(
                high_video, (self.seed + 1) % (1 << 64), source.get("batch_index")
            )
        audio_state = transition.pop("audio")
        audio_anchor = (
            latent_format.process_in(original_audio.to(audio_state))
            * sampling.audio_scale
        )
        s = next_sigma.to(audio_state)
        audio_noise = (audio_state - (1 - s) * audio_anchor) / s
        noise = NestedTensor((video_noise, audio_noise.to(device=video_noise.device)))
        result = self.sample_stage(high, high_condition, curve[split:], noise=noise)
        return _restore_protected(
            result, original_video, high_mask, original_audio, audio_mask
        )


def _sample_h3_image(model, conditioning, latent, seed, sampler_name, sigmas):
    """H3-Image branch: plain single-frame latent sampling on the checkpoint's PDD grid.

    The distilled per-step heads live inside the model itself and the stock
    scheduler at the distilled step count lands on their sigma grid, so this
    mirrors the core KSampler path (cfg 1, no negative prompt) for image
    latents. tail_steps and the AV sampling modes do not apply here.
    """
    settings = sigmas or {}
    steps = int(settings.get("steps", 6))
    scheduler = settings.get("scheduler", "simple")
    denoise = float(settings.get("denoise", 1.0))
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    noise = comfy.sample.prepare_noise(
        latent_image, seed, latent.get("batch_index", None)
    )
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        1.0,
        sampler_name,
        scheduler,
        conditioning,
        [],
        latent_image,
        denoise=denoise,
        noise_mask=latent.get("noise_mask"),
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def sample_h3(
    model,
    conditioning,
    latent,
    seed=0,
    sampler_name="euler",
    sigmas=None,
    mode=None,
    upscale_model=None,
):
    """Sample one final-size canvas; modes never accept a second latent/condition."""
    latent_format = model.get_model_object("latent_format")
    if not isinstance(latent_format, MiniMaxH3AV):
        if isinstance(latent_format, MiniMaxH3Video):
            return _sample_h3_image(
                model=model,
                conditioning=conditioning,
                latent=latent,
                seed=seed,
                sampler_name=sampler_name,
                sigmas=sigmas,
            )
        raise ValueError("MiniMax H3 KSampler requires a MiniMax H3 MODEL.")
    mode = {} if mode is None else mode
    kind = mode.get("mode", "single")
    if kind not in {"single", "progressive", "two_pass"}:
        raise ValueError(f"Unknown H3 sampling mode: {kind!r}.")
    if mode.get("mask_grow", 0) < 0:
        raise ValueError("Mask growth must be nonnegative.")
    curve = sampling_sigmas(model, sigmas, sampler_name)
    if curve.numel() < 2 and kind != "two_pass":
        return latent
    second_curve = (
        sampling_sigmas(model, mode.get("second_sigmas"), sampler_name, second=True)
        if kind == "two_pass"
        else None
    )
    if curve.numel() < 2 and second_curve.numel() < 2:
        return latent
    if kind == "progressive":
        if sampler_name != "euler":
            raise ValueError(
                "Progressive H3 currently requires Euler; use single or two_pass for other samplers."
            )
        mode = {"lowres_steps": 3, **mode}
        split = mode["lowres_steps"]
        if not isinstance(split, int) or not 1 <= split < len(curve) - 1:
            raise ValueError(
                "Progressive lowres_steps must leave at least one high-resolution interval."
            )
        if not 0 < float(curve[split]) < 1:
            raise ValueError(
                "Progressive transition sigma must be between zero and one."
            )
    video, audio = _latent_streams(latent)
    video_only = audio is None
    if video_only:
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
        audio = video.new_zeros(
            (video.shape[0], 32, 2, max(1, round(frame_count * FRAME_RESCALE))),
            dtype=torch.float32,
        )
    mask, audio_mask = _masks(latent, video, audio)
    source = (
        {**latent, "samples": NestedTensor((video, audio))} if video_only else latent
    )
    _validate_conditioning(conditioning, video, audio, "conditioning")
    count = max(0, len(curve) - 1) + (
        0 if second_curve is None else max(0, len(second_curve) - 1)
    )
    task = H3SamplingTask(model, seed, sampler_name, count)
    result = task.run(
        source,
        conditioning,
        curve,
        second_curve,
        mode,
        upscale_model,
        tuple(model.get_model_object("diffusion_model").patch_size[1:]),
        mask,
        audio_mask,
    )
    return _output_structure(result, video_only)
