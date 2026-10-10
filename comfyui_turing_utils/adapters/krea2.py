"""Krea2 Identity Edit reference preparation and model patching."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
import node_helpers


_DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)
_REFERENCE_METHOD = "index"
_FIT_CROP_TOLERANCE = 0.08
_BIAS_STATE_KEY = "turing_utils_krea2_reference_bias"
_BIAS_WRAPPER_KEY = "TuringUtilsKrea2IdentityEditReferenceBias"


def _grounding_template(image_count: int) -> str:
    vision = "<|vision_start|><|image_pad|><|vision_end|>" * image_count
    return (
        f"<|im_start|>system\n{_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{vision}{{}}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _prepare_grounding_image(image: torch.Tensor, grounding_px: int) -> torch.Tensor:
    samples = image[..., :3].movedim(-1, 1)
    height, width = samples.shape[-2:]
    longest = max(height, width)
    if grounding_px > 0 and longest > grounding_px:
        scale = grounding_px / longest
        samples = comfy.utils.common_upscale(
            samples,
            round(width * scale),
            round(height * scale),
            "area",
            "disabled",
        )
    return samples.movedim(1, -1)


def _fit_reference_image(
    image: torch.Tensor,
    target_latent_height: int,
    target_latent_width: int,
) -> torch.Tensor:
    """Match the Identity Edit training geometry without latent interpolation."""
    target_height = target_latent_height * 8
    target_width = target_latent_width * 8
    samples = image[..., :3].movedim(-1, 1)
    source_height, source_width = samples.shape[-2:]
    scale = min(target_height / source_height, target_width / source_width)

    if source_height * scale >= target_height * (
        1 - _FIT_CROP_TOLERANCE
    ) and source_width * scale >= target_width * (1 - _FIT_CROP_TOLERANCE):
        fill_scale = max(target_height / source_height, target_width / source_width)
        crop_height = min(source_height, int(round(target_height / fill_scale)))
        crop_width = min(source_width, int(round(target_width / fill_scale)))
        top = (source_height - crop_height) // 2
        left = (source_width - crop_width) // 2
        samples = samples[..., top : top + crop_height, left : left + crop_width]
        fitted_height, fitted_width = target_height, target_width
    else:
        fitted_height = min(
            max(16, int(source_height * scale) // 16 * 16),
            max(16, target_height // 16 * 16),
        )
        fitted_width = min(
            max(16, int(source_width * scale) // 16 * 16),
            max(16, target_width // 16 * 16),
        )
        crop_height = min(source_height, max(1, int(round(fitted_height / scale))))
        crop_width = min(source_width, max(1, int(round(fitted_width / scale))))
        top = (source_height - crop_height) // 2
        left = (source_width - crop_width) // 2
        samples = samples[..., top : top + crop_height, left : left + crop_width]

    fitted = F.interpolate(
        samples.float(),
        size=(fitted_height, fitted_width),
        mode="bicubic",
        antialias=True,
    )
    return fitted.movedim(1, -1).clamp_(0, 1)


def _dit_grid(latent: torch.Tensor) -> tuple[int, int]:
    return ((int(latent.shape[-2]) + 1) // 2, (int(latent.shape[-1]) + 1) // 2)


def _reference_position_patch(
    target_grid: tuple[int, int],
    reference_grids: tuple[tuple[int, int], ...],
):
    expected_counts = tuple(height * width for height, width in reference_grids)
    target_tokens = target_grid[0] * target_grid[1]

    def center_reference_positions(inputs):
        transformer_options = inputs["transformer_options"]
        runtime_counts = tuple(
            int(value)
            for value in transformer_options.get("reference_image_num_tokens", ())
        )
        if not runtime_counts:
            return inputs
        if runtime_counts != expected_counts:
            raise ValueError(
                "Krea2 Identity Edit reference geometry changed during sampling; "
                "use the same target latent for this node and KSampler"
            )

        image_ids = inputs["img_ids"]
        if int(image_ids.shape[1]) != target_tokens + sum(expected_counts):
            raise ValueError(
                "Krea2 Identity Edit target geometry changed during sampling; "
                "use the same target latent for this node and KSampler"
            )

        offset = target_tokens
        for count, (height, width) in zip(expected_counts, reference_grids):
            end = offset + count
            image_ids[:, offset:end, 1].add_((target_grid[0] - height) / 2)
            image_ids[:, offset:end, 2].add_((target_grid[1] - width) / 2)
            offset = end
        return inputs

    return center_reference_positions


def _reference_bias_patches(
    target_tokens: int,
    reference_counts: tuple[int, ...],
    strengths: tuple[float, ...],
):
    reference_tokens = sum(reference_counts)

    def bias_scope(
        executor,
        x,
        timesteps,
        context,
        attention_mask=None,
        ref_latents=None,
        transformer_options=None,
        **kwargs,
    ):
        options = {} if transformer_options is None else transformer_options.copy()
        state = {"bias": None}
        options[_BIAS_STATE_KEY] = state
        try:
            return executor(
                x,
                timesteps,
                context,
                attention_mask,
                ref_latents,
                options,
                **kwargs,
            )
        finally:
            state.clear()

    def apply_reference_bias(
        q, k, v, pe=None, attn_mask=None, extra_options=None, **kwargs
    ):
        state = None if extra_options is None else extra_options.get(_BIAS_STATE_KEY)
        if state is None:
            return {}
        bias = state.get("bias")
        if bias is None:
            sequence = int(q.shape[2])
            text_tokens = sequence - target_tokens - reference_tokens
            if text_tokens < 0:
                raise ValueError(
                    "Krea2 Identity Edit received an invalid attention layout"
                )
            target_end = text_tokens + target_tokens
            bias = q.new_zeros((1, 1, sequence, sequence))
            reference_start = target_end
            for count, strength in zip(reference_counts, strengths):
                reference_end = reference_start + count
                if strength != 1.0:
                    bias[
                        :, :, text_tokens:target_end, reference_start:reference_end
                    ] = math.log(max(strength, 1e-4))
                reference_start = reference_end
            state["bias"] = bias
        if attn_mask is not None:
            bias = bias + attn_mask.to(device=q.device, dtype=q.dtype)
        return {"attn_mask": bias}

    return bias_scope, apply_reference_bias


def build_identity_edit_conditioning(
    model,
    clip,
    vae,
    target_latent,
    character_image: torch.Tensor,
    prompt: str,
    background_image: torch.Tensor | None = None,
    grounding_px: int = 768,
    character_strength: float = 4.0,
    background_strength: float = 1.0,
):
    samples = target_latent["samples"]
    if not torch.is_tensor(samples) or samples.ndim not in (4, 5):
        shape = (
            tuple(samples.shape)
            if hasattr(samples, "shape")
            else type(samples).__name__
        )
        raise ValueError(
            f"Krea2 Identity Edit target latent must be 4D or 5D, got {shape}"
        )

    target_height, target_width = map(int, samples.shape[-2:])
    if background_image is None:
        images = (character_image,)
        strengths = (float(character_strength),)
    else:
        images = (background_image, character_image)
        strengths = (float(background_strength), float(character_strength))

    reference_latents = [
        vae.encode(_fit_reference_image(image, target_height, target_width))
        for image in images
    ]
    grounding_images = [
        _prepare_grounding_image(image, int(grounding_px)) for image in images
    ]
    tokens = clip.tokenize(
        prompt,
        images=grounding_images,
        llama_template=_grounding_template(len(grounding_images)),
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": reference_latents},
        append=True,
    )
    conditioning = node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents_method": _REFERENCE_METHOD},
    )

    target_grid = _dit_grid(samples)
    reference_grids = tuple(_dit_grid(latent) for latent in reference_latents)
    patched = model.clone()
    patched.set_model_post_input_patch(
        _reference_position_patch(target_grid, reference_grids)
    )
    if any(strength != 1.0 for strength in strengths):
        bias_scope, reference_bias = _reference_bias_patches(
            target_grid[0] * target_grid[1],
            tuple(height * width for height, width in reference_grids),
            strengths,
        )
        patched.set_model_attn1_patch(reference_bias)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BIAS_WRAPPER_KEY,
            bias_scope,
        )
    return patched, conditioning


__all__ = ["build_identity_edit_conditioning"]
