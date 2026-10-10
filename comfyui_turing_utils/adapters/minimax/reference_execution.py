"""Reference encoding and conditioning services for MiniMax H3.

Schemas are consumers of these functions, not execution or caching owners.
The semantic memory guard uses ComfyUI's existing model-management lifecycle.
"""

from __future__ import annotations

import torch
import node_helpers

from ...log import get_logger
from .references import (
    H3AudioReferenceData,
    H3ImageReferenceData,
    H3KeyframeReferenceData,
    H3SemanticReferenceData,
    H3VideoReferenceData,
    H3_MODEL_FPS,
    H3_QWEN_VIDEO_FPS,
    _align_keyframe_pixels,
    _align_reference_pixels,
    _dynamic_entries,
    _dynamic_suffix,
    _encode_audio,
    _encode_visual,
    _frame_count_from_latent_t,
    _keyframe_reference,
    _manifest,
    _reference_blocks,
    _reference_presentation,
    _tokenize_semantic,
    _trim_h3_reference_video,
    _validate_pixels,
    _validate_visual_latent,
    _video_latent,
)

LOG = get_logger("minimax")
_GIB = 1024**3
_H3_SEMANTIC_MIN_HEADROOM = 3 * _GIB
_H3_SEMANTIC_MAX_HEADROOM = 6 * _GIB


def _loaded_clip_models(model_management, patcher) -> list[object]:
    """Return ComfyUI LoadedModel records that own this CLIP patcher."""

    matches = []
    for loaded in getattr(model_management, "current_loaded_models", ()):
        loaded_patcher = getattr(loaded, "model", None)
        if loaded_patcher is patcher:
            matches.append(loaded)
            continue
        try:
            if patcher.is_clone(loaded_patcher):
                matches.append(loaded)
        except (AttributeError, RuntimeError, TypeError):
            continue
    return matches


def _prepare_h3_semantic_encoder(clip, tokens) -> None:
    """Load the H3 text encoder with room for quantized-weight expansion.

    Qwen3-VL NVFP4 weights use an eager lookup/dequantization fallback on
    pre-Blackwell GPUs.  ComfyUI's generic CLIP estimate currently accounts
    for neither that transient allocation nor a resident dynamic DiT.  A
    warm rerun can therefore load both models successfully and then OOM on the
    first MLP.  Reserve a bounded fraction of device memory before loading the
    encoder, and keep the encoder while releasing stale residents afterward.
    """

    patcher = getattr(clip, "patcher", None)
    load_model = getattr(clip, "load_model", None)
    device = getattr(patcher, "load_device", None)
    if patcher is None or not callable(load_model) or device is None:
        return
    if getattr(device, "type", None) in ("cpu", "mps"):
        return

    try:
        import comfy.model_management as model_management
    except (ImportError, AttributeError):
        return

    total = int(model_management.get_total_memory(device))
    headroom = min(
        _H3_SEMANTIC_MAX_HEADROOM,
        max(_H3_SEMANTIC_MIN_HEADROOM, total // 8),
    )
    try:
        remaining_weights = max(
            0,
            int(patcher.model_size()) - int(patcher.loaded_size()),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        remaining_weights = 0

    # First make room for weights that are not resident yet plus the actual
    # inference workspace. ``for_dynamic=False`` is intentional: ComfyUI's
    # normal dynamic-to-dynamic path assumes demand paging alone is sufficient
    # and is the source of this warm-rerun OOM.
    reserve_limit = max(0, total - int(model_management.extra_reserved_memory()))
    load_target = min(reserve_limit, remaining_weights + headroom)
    before = int(model_management.get_free_memory(device))
    if before < load_target:
        model_management.free_memory(
            load_target,
            device,
            keep_loaded=_loaded_clip_models(model_management, patcher),
            for_dynamic=False,
        )

    # Encoding will call this a second time; ComfyUI treats an already loaded
    # patcher as a cheap no-op. Loading here lets us make a second, accurate
    # headroom check while protecting the encoder from eviction.
    load_model(tokens)
    after_load = int(model_management.get_free_memory(device))
    if after_load < headroom:
        model_management.free_memory(
            headroom,
            device,
            keep_loaded=_loaded_clip_models(model_management, patcher),
            for_dynamic=False,
        )
    final_free = int(model_management.get_free_memory(device))
    if before < load_target or after_load < headroom:
        LOG.debug(
            "H3 semantic encoder VRAM guard: target=%.2f GiB "
            "free_before=%.2f GiB free_after=%.2f GiB",
            headroom / _GIB,
            before / _GIB,
            final_free / _GIB,
        )


def encode_keyframes(
    vae, latent=None, image_0=None, image_1=None, image_2=None
) -> tuple:
    outputs = []
    for index, image in enumerate((image_0, image_1, image_2)):
        if image is None:
            outputs.append(None)
            continue
        name = f"image_{index}"
        pixels = _align_keyframe_pixels(image[:1], latent, name)
        outputs.append(
            H3KeyframeReferenceData(
                image=pixels,
                latent=_encode_visual(vae, pixels, name),
            )
        )
    return tuple(outputs)


def encode_images(
    vae, megapixels=1.0, latent=None, images=None
) -> H3ImageReferenceData:
    items = []
    for name, image in _dynamic_entries(images):
        pixels = _align_reference_pixels(image[:1], latent, name, float(megapixels))
        items.append(
            {
                "image": pixels,
                "latent": _encode_visual(vae, pixels, name),
            }
        )
    return H3ImageReferenceData(tuple(items))


def encode_videos(
    video_vae,
    megapixels=1.0,
    audio_vae=None,
    latent=None,
    videos=None,
    video_audios=None,
) -> H3VideoReferenceData:
    soundtracks = {
        _dynamic_suffix(name): value for name, value in _dynamic_entries(video_audios)
    }
    items = []
    for name, video in _dynamic_entries(videos):
        frames = _validate_pixels(video, name)
        frames = _trim_h3_reference_video(frames)
        frames = _align_reference_pixels(frames, latent, name, float(megapixels))
        visual_latent = _encode_visual(video_vae, frames, name)
        soundtrack = soundtracks.get(_dynamic_suffix(name))
        audio_latent = None
        if soundtrack is not None:
            if audio_vae is None:
                raise ValueError(
                    f"{name} has a soundtrack but audio_vae is not connected"
                )
            audio_latent = _encode_audio(audio_vae, soundtrack, f"{name} soundtrack")
        sample_stride = int(H3_MODEL_FPS / H3_QWEN_VIDEO_FPS)
        indices = torch.arange(
            0, int(frames.shape[0]), sample_stride, device=frames.device
        )
        qwen_frames = frames.index_select(0, indices)
        items.append(
            {
                "latent": visual_latent,
                "audio_latent": audio_latent,
                "qwen_frames": qwen_frames,
                "timestamps": [
                    index / H3_QWEN_VIDEO_FPS
                    for index in range(int(qwen_frames.shape[0]))
                ],
            }
        )
    return H3VideoReferenceData(tuple(items))


def encode_audios(audio_vae, audios=None) -> H3AudioReferenceData:
    items = tuple(
        {"audio_latent": _encode_audio(audio_vae, audio, name)}
        for name, audio in _dynamic_entries(audios)
    )
    return H3AudioReferenceData(items)


def encode_semantic(
    clip,
    prompt: str,
    first_frame=None,
    last_frame=None,
    image_reference=None,
    video_reference=None,
    audio_reference=None,
) -> H3SemanticReferenceData:
    first_frame = _keyframe_reference(first_frame, "first_frame")
    last_frame = _keyframe_reference(last_frame, "last_frame")
    manifest = _manifest(
        first_frame,
        last_frame,
        image_reference,
        video_reference,
        audio_reference,
    )
    presentation = _reference_presentation(
        image_reference, video_reference, audio_reference
    )
    tokens = _tokenize_semantic(clip, prompt, first_frame, last_frame, presentation)
    _prepare_h3_semantic_encoder(clip, tokens)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return H3SemanticReferenceData(conditioning, manifest)


def build_conditioning(
    semantic_reference,
    latent,
    first_frame=None,
    last_frame=None,
    image_reference=None,
    video_reference=None,
    audio_reference=None,
) -> list:
    if not isinstance(semantic_reference, H3SemanticReferenceData):
        raise ValueError("semantic_reference must come from H3 Semantic Reference")
    first_frame = _keyframe_reference(first_frame, "first_frame")
    last_frame = _keyframe_reference(last_frame, "last_frame")
    target_video = _video_latent(latent)
    frame_count = _frame_count_from_latent_t(int(target_video.shape[2]))
    keyframes = []
    for role, item, frame_index in (
        ("first_frame", first_frame, 0),
        ("last_frame", last_frame, frame_count - 1),
    ):
        if item is None:
            continue
        visual = _validate_visual_latent(item.latent, role)
        if int(visual.shape[2]) != 1 or tuple(visual.shape[-2:]) != tuple(
            target_video.shape[-2:]
        ):
            raise ValueError(
                f"{role} latent {tuple(visual.shape)} does not match target H3 "
                f"spatial grid {tuple(target_video.shape[-2:])}; connect the same "
                "latent to H3 Keyframe Reference or resize upstream"
            )
        keyframes.append({"resolved_frame_index": frame_index, "latent": visual})

    refs = _reference_blocks(image_reference, video_reference, audio_reference)
    values = {"minimax_frame_count": frame_count}
    if keyframes:
        values["minimax_keyframes"] = keyframes
    if refs:
        values["minimax_refs"] = refs
    conditioning = node_helpers.conditioning_set_values(
        semantic_reference.conditioning, values
    )
    return conditioning
