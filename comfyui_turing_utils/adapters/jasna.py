"""Model and window sampling for Jasna's video-only LTX Restore checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import torch
from torch.nn import functional as F

import comfy.model_base
import comfy.model_management as mm
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.storage
import comfy.utils
from comfy.ldm.lightricks.model import LTXVModel
from comfy.ldm.lightricks.vae.na_diffusion_decoder import CausalDiffusionVAE

from ..media.temporal import repeat_last_frame, validate_frame_mask


TEMPORAL_STRIDE = 8
SPATIAL_STRIDE = 32


@dataclass(frozen=True)
class RestoreSettings:
    sigmas: tuple[float, ...]
    stg_scale: float
    stg_block: int
    window_frames: int
    overlap_frames: int
    position_fps: float

    @classmethod
    def from_metadata(cls, metadata):
        data = json.loads(metadata["jasna_restore"])
        if data["version"] != 1 or data["sampler"] != "euler":
            raise ValueError("Unsupported Jasna Restore checkpoint metadata.")
        sigmas = tuple(map(float, data["sigmas"]))
        if (
            len(sigmas) < 2
            or sigmas[0] != 1
            or sigmas[-1] != 0
            or any(not a > b for a, b in zip(sigmas, sigmas[1:]))
        ):
            raise ValueError("Jasna Restore sigmas must decrease from 1 to 0.")
        window = int(data["window_frames"])
        overlap = int(data["overlap_frames"])
        hop = window - overlap
        if (
            (window - 1) % TEMPORAL_STRIDE
            or hop % TEMPORAL_STRIDE
            or not TEMPORAL_STRIDE < overlap <= hop
        ):
            raise ValueError(
                "Jasna Restore windows must overlap on the causal LTX latent grid."
            )
        stg_scale = float(data["stg_scale"])
        stg_block = int(data["stg_block"])
        fps = float(data["position_fps"])
        if (
            not math.isfinite(stg_scale)
            or stg_scale < 0
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(
                "Jasna Restore requires a nonnegative STG scale and positive position FPS."
            )
        return cls(sigmas, stg_scale, stg_block, window, overlap, fps)

    @property
    def hop(self):
        return self.window_frames - self.overlap_frames


class JasnaLTXModel(LTXVModel):
    def _init_transformer_blocks(self, device, dtype, **kwargs):
        super()._init_transformer_blocks(device, dtype, **kwargs)
        for block in self.transformer_blocks:
            for attention in (block.attn1, block.attn2):
                attention.to_gate_logits = self.operations.Linear(
                    self.inner_dim,
                    self.num_attention_heads,
                    bias=True,
                    device=device,
                    dtype=dtype,
                )
                attention.q_norm.eps = attention.k_norm.eps = 1e-6

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        tokens, coords, extra = super()._process_input(x, None, None, **kwargs)
        # Both halves use the same RoPE positions. Only the target has a keyframe marker.
        count = tokens.shape[1] // 2
        coords[:, :, count:] = coords[:, :, :count]
        return tokens, coords, extra

    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        # Embed only the two distinct timesteps before broadcasting to video tokens.
        rows, embedded, prompt = super()._prepare_timestep(
            timestep, batch_size, hidden_dtype, **kwargs
        )
        shape = kwargs["orig_shape"]
        count = math.prod(shape[2:]) // 2
        return (
            rows.repeat_interleave(count, dim=1),
            embedded.repeat_interleave(count, dim=1),
            prompt,
        )


class JasnaLTX(comfy.model_base.BaseModel):
    def __init__(self, model_config, device):
        super().__init__(
            model_config,
            comfy.model_base.ModelType.FLUX,
            device=device,
            unet_model=JasnaLTXModel,
        )

    def process_timestep(self, timestep, x, **kwargs):
        return torch.stack((timestep, torch.zeros_like(timestep)), dim=1)


def _skip_self_attention(args, extra):
    options = {**args["transformer_options"], "stg_skip_self_attn": True}
    return extra["original_block"]({**args, "transformer_options": options})


def build_model(state, metadata, config, device, model_options, disable_dynamic):
    if config is None or config.unet_config["image_model"] != "ltxv":
        raise ValueError("Expected an exported Jasna LTX Restore DiT.")
    settings = RestoreSettings.from_metadata(metadata)
    if "custom_operations" not in model_options:
        state, _ = comfy.utils.convert_old_quants(state, metadata=metadata)
    context = state.pop("prompt_context")
    config.quant_config = comfy.utils.detect_layer_quantization(state, "")
    if context.shape != (1, 1024, config.unet_config["cross_attention_dim"]):
        raise ValueError(
            "The Jasna Restore checkpoint has invalid prompt conditioning."
        )
    if not 0 <= settings.stg_block < config.unet_config["num_layers"]:
        raise ValueError("The Jasna Restore STG block is outside the model.")

    offload = mm.unet_offload_device()
    dtype = model_options.get("dtype") or mm.unet_dtype(
        model_params=comfy.utils.calculate_parameters(state),
        supported_dtypes=config.supported_inference_dtypes,
    )
    compute_dtype = model_options.get("dtype") or mm.unet_manual_cast(
        None,
        device,
        config.supported_inference_dtypes,
    )
    config.set_inference_dtype(dtype, compute_dtype, device=device)
    config.custom_operations = model_options.get("custom_operations")
    model = JasnaLTX(config, offload)
    patcher_type = (
        comfy.model_patcher.ModelPatcher
        if disable_dynamic
        else comfy.model_patcher.CoreModelPatcher
    )
    patcher = patcher_type(
        model,
        load_device=device,
        offload_device=offload,
        fast_disk=comfy.storage.state_dict_fast_disk(state),
    )
    model.diffusion_model.load_state_dict(
        config.process_unet_state_dict(state), strict=True, assign=patcher.is_dynamic()
    )
    patcher.set_attachments("jasna_restore", (context, settings))
    return patcher


def _align_latents(latents, hop, *, initial=False):
    if len(latents) < 2:
        return
    first_shared = hop // TEMPORAL_STRIDE + 1
    count = latents[0].shape[2] - first_shared
    ramp = (torch.arange(1, count + 1, dtype=torch.float32) / (count + 1)).view(
        1, 1, -1, 1, 1
    )
    for left, right in zip(latents, latents[1:]):
        before = left[:, :, first_shared:]
        after = right[:, :, 1 : 1 + count]
        if initial:
            after.copy_(before)
        else:
            overlap = before.float().lerp(after.float(), ramp)
            before.copy_(overlap)
            after.copy_(overlap)


def _sample(patcher, context, settings, references, seed, progress):
    device = patcher.load_device
    model = patcher.model
    latents = []
    generator = torch.Generator(device=device)
    for index, reference in enumerate(references):
        generator.manual_seed((seed + index) & 0xFFFFFFFFFFFFFFFF)
        _, channels, frames, height, width = reference.shape
        tokens = frames * height * width
        noise = torch.randn(
            (1, tokens * 2, channels),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        latents.append(noise[:, :tokens].transpose(1, 2).reshape(reference.shape).cpu())
    _align_latents(latents, settings.hop, initial=True)

    shape = list(references[0].shape)
    shape[2] *= 2
    # Keep GPU capacity for one target/reference window regardless of clip length.
    mm.load_models_gpu([patcher], memory_required=patcher.memory_required(shape))
    model_options = comfy.model_patcher.create_model_options_clone(
        patcher.model_options
    )
    comfy.sampler_helpers.prepare_model_patcher(patcher, {}, model_options)
    options = model_options["transformer_options"]
    weak_options = None
    if settings.stg_scale:
        weak_options = comfy.model_patcher.set_model_options_patch_replace(
            model_options.copy(),
            _skip_self_attention,
            "dit",
            "double_block",
            settings.stg_block,
        )["transformer_options"]

    with mm.cuda_device_context(device):
        try:
            patcher.pre_run()
            context = context.to(device=device, dtype=model.get_dtype_inference())
            conditioning = dict(
                c_crossattn=context,
                attention_mask=None,
                frame_rate=settings.position_fps,
            )
            for sigma, next_sigma in zip(settings.sigmas, settings.sigmas[1:]):
                timestep = torch.tensor([sigma], device=device, dtype=torch.float32)
                patcher.prepare_state(timestep, model_options)
                for index, reference in enumerate(references):
                    mm.throw_exception_if_processing_interrupted()
                    sample = torch.cat((latents[index], reference), dim=2).to(
                        device=device
                    )
                    frame_count = reference.shape[2]
                    denoised = model.apply_model(
                        sample, timestep, transformer_options=options, **conditioning
                    )
                    denoised = denoised[:, :, :frame_count].to(sample.dtype)
                    if weak_options is not None:
                        weak = model.apply_model(
                            sample,
                            timestep,
                            transformer_options=weak_options,
                            **conditioning,
                        )
                        weak = weak[:, :, :frame_count].to(sample.dtype)
                        denoised = (
                            denoised.float()
                            + settings.stg_scale * (denoised.float() - weak.float())
                        ).to(sample.dtype)
                    target = sample[:, :, :frame_count].float()
                    velocity = ((target - denoised.float()) / sigma).to(sample.dtype)
                    target.add_(velocity.float(), alpha=next_sigma - sigma)
                    latents[index].copy_(target)
                    progress.update(1)
                _align_latents(latents, settings.hop)
        finally:
            patcher.cleanup()
    return latents


@torch.inference_mode()
def restore(images, mask, patcher, vae, seed):
    if images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[:3]) < 1:
        raise ValueError(
            "Jasna Restore requires RGB images shaped [frames, height, width, 3]."
        )
    frames, height, width, _ = images.shape
    mask = validate_frame_mask(mask, frames, height, width)
    if mask.ndim != 3:
        raise ValueError(
            "Jasna Restore requires a mask shaped [frames, height, width]."
        )
    mask = mask > 0
    if not mask.any().item():
        return images

    checkpoint = patcher.get_attachment("jasna_restore")
    if checkpoint is None:
        raise ValueError("Load the exported Jasna Restore DiT with Load ConvRot DiT.")
    context, settings = checkpoint
    vae.throw_exception_if_invalid()
    if not isinstance(vae.first_stage_model, CausalDiffusionVAE):
        raise ValueError("Expected the merged Jasna LTX video VAE.")

    window = settings.window_frames
    tail_frames = max(0, frames - window)
    last_start = ((tail_frames + settings.hop - 1) // settings.hop) * settings.hop
    starts = range(0, last_start + 1, settings.hop)
    progress = comfy.utils.ProgressBar((len(settings.sigmas) + 1) * len(starts))
    pad_w, pad_h = (-width) % SPATIAL_STRIDE, (-height) % SPATIAL_STRIDE
    references = []
    for start in starts:
        mm.throw_exception_if_processing_interrupted()
        pixels = images[start : start + window]
        pixels = repeat_last_frame(pixels, window - pixels.shape[0])
        if pad_w or pad_h:
            pixels = F.pad(
                pixels.movedim(-1, 1), (0, pad_w, 0, pad_h), mode="replicate"
            ).movedim(1, -1)
        references.append(vae.encode(pixels).to(device="cpu", dtype=torch.bfloat16))
        progress.update(1)
    del pixels

    sampler = comfy.patcher_extension.WrapperExecutor.new_executor(
        _sample,
        patcher.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE),
    )
    latents = sampler.execute(patcher, context, settings, references, seed, progress)
    del references
    output = torch.empty_like(images)
    for start, latent in zip(starts, latents):
        mm.throw_exception_if_processing_interrupted()
        stop = min(start + window, frames)
        pixels = vae.decode(latent).reshape(-1, height + pad_h, width + pad_w, 3)
        pixels = pixels[: stop - start, :height, :width].to(images)
        if start:
            # The causal decoder's new-window head has no matching latent in the old window.
            overlap = settings.overlap_frames
            ramp = torch.zeros(overlap, device=images.device, dtype=images.dtype)
            ramp[TEMPORAL_STRIDE:] = torch.linspace(
                0,
                1,
                overlap - TEMPORAL_STRIDE,
                device=images.device,
                dtype=images.dtype,
            )
            pixels[:overlap] = output[start : start + overlap].lerp(
                pixels[:overlap], ramp.view(-1, 1, 1, 1)
            )
        torch.where(
            mask[start:stop, ..., None].to(images.device),
            pixels,
            images[start:stop],
            out=output[start:stop],
        )
        progress.update(1)
    return output
