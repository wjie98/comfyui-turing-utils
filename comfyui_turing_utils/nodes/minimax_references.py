"""ComfyUI schemas for composable MiniMax H3 references."""

from __future__ import annotations

import torch

import node_helpers
from comfy_api.latest import io

from ..log import get_logger
from ..adapters.minimax.references import (
    H3AudioReferenceData,
    H3ImageReferenceData,
    H3KeyframeReferenceData,
    H3ReferenceManifest,
    H3SemanticReferenceData,
    H3VideoReferenceData,
    H3_MAX_KEYFRAME_REFERENCES,
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
    h3_latent_info,
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
        LOG.info(
            "H3 semantic encoder VRAM guard: target=%.2f GiB "
            "free_before=%.2f GiB free_after=%.2f GiB",
            headroom / _GIB,
            before / _GIB,
            final_free / _GIB,
        )


H3KeyframeReferenceType = io.Custom("TURING_UTILS_H3_KEYFRAME_REFERENCE")
H3ImageReferenceType = io.Custom("TURING_UTILS_H3_IMAGE_REFERENCE")
H3VideoReferenceType = io.Custom("TURING_UTILS_H3_VIDEO_REFERENCE")
H3AudioReferenceType = io.Custom("TURING_UTILS_H3_AUDIO_REFERENCE")
H3SemanticReferenceType = io.Custom("TURING_UTILS_H3_SEMANTIC_REFERENCE")


class H3KeyframeReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3KeyframeReference",
            display_name="H3 Keyframe Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Encode dynamic reusable H3 keyframes without assigning first/last "
                "roles. Each image_N input has a matching keyframe_N output. With an "
                "optional latent, images are cover-resized and cropped to its decoded "
                "pixel canvas."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=H3_MAX_KEYFRAME_REFERENCES,
                    ),
                ),
            ],
            outputs=[
                H3KeyframeReferenceType.Output(f"keyframe_{index}")
                for index in range(H3_MAX_KEYFRAME_REFERENCES)
            ],
        )

    @classmethod
    def execute(cls, vae, latent=None, images=None) -> io.NodeOutput:
        outputs = []
        for name, image in _dynamic_entries(images):
            pixels = _align_keyframe_pixels(image[:1], latent, name)
            outputs.append(
                H3KeyframeReferenceData(
                    image=pixels,
                    latent=_encode_visual(vae, pixels, name),
                )
            )
        outputs.extend([None] * (H3_MAX_KEYFRAME_REFERENCES - len(outputs)))
        return io.NodeOutput(*outputs)


class H3ImageReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3ImageReference",
            display_name="H3 Image Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Encode dynamic H3 reference images without cropping or upscaling. "
                "A latent applies match-area sizing; without one, the short edge is "
                "limited by the megapixel budget."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Maximum source area when latent is not connected; smaller images are not enlarged.",
                ),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=32,
                    ),
                ),
            ],
            outputs=[H3ImageReferenceType.Output("image_reference")],
        )

    @classmethod
    def execute(cls, vae, megapixels=1.0, latent=None, images=None) -> io.NodeOutput:
        items = []
        for name, image in _dynamic_entries(images):
            pixels = _align_reference_pixels(image[:1], latent, name, float(megapixels))
            items.append(
                {
                    "image": pixels,
                    "latent": _encode_visual(vae, pixels, name),
                }
            )
        return io.NodeOutput(H3ImageReferenceData(tuple(items)))


class H3VideoReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3VideoReference",
            display_name="H3 Video Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Encode dynamic H3 reference videos that were resampled to 24 FPS "
                "upstream, plus index-paired soundtracks. Qwen receives a 2 FPS view; "
                "the DiT receives the full VAE latent. A latent applies match-area "
                "sizing; without one, the megapixel budget limits source area."
            ),
            inputs=[
                io.Vae.Input("video_vae"),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Maximum source area when latent is not connected; smaller videos are not enlarged.",
                ),
                io.Vae.Input("audio_vae", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "video",
                            tooltip="Consecutive frames already resampled to 24 FPS",
                        ),
                        prefix="video_",
                        min=0,
                        max=16,
                    ),
                ),
                io.Autogrow.Input(
                    "video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("video_audio"),
                        prefix="video_audio_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[H3VideoReferenceType.Output("video_reference")],
        )

    @classmethod
    def execute(
        cls,
        video_vae,
        megapixels=1.0,
        audio_vae=None,
        latent=None,
        videos=None,
        video_audios=None,
    ) -> io.NodeOutput:
        soundtracks = {
            _dynamic_suffix(name): value
            for name, value in _dynamic_entries(video_audios)
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
                audio_latent = _encode_audio(
                    audio_vae, soundtrack, f"{name} soundtrack"
                )
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
        return io.NodeOutput(H3VideoReferenceData(tuple(items)))


class H3AudioReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3AudioReference",
            display_name="H3 Audio Reference",
            category="Turing Utils/conditioning/minimax",
            description="Encode a dynamic set of standalone H3 reference audio clips.",
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"),
                        prefix="audio_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[H3AudioReferenceType.Output("audio_reference")],
        )

    @classmethod
    def execute(cls, audio_vae, audios=None) -> io.NodeOutput:
        items = tuple(
            {"audio_latent": _encode_audio(audio_vae, audio, name)}
            for name, audio in _dynamic_entries(audios)
        )
        return io.NodeOutput(H3AudioReferenceData(items))


class H3SemanticReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3SemanticReference",
            display_name="H3 Semantic Reference",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Run one exact Qwen3-VL multimodal encode over the prompt and selected "
                "H3 references. The resulting semantic reference can be reused with "
                "structure-equivalent DiT references at another resolution."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                H3KeyframeReferenceType.Input("first_frame", optional=True),
                H3KeyframeReferenceType.Input("last_frame", optional=True),
                H3ImageReferenceType.Input("image_reference", optional=True),
                H3VideoReferenceType.Input("video_reference", optional=True),
                H3AudioReferenceType.Input("audio_reference", optional=True),
            ],
            outputs=[H3SemanticReferenceType.Output("semantic_reference")],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt: str,
        first_frame=None,
        last_frame=None,
        image_reference=None,
        video_reference=None,
        audio_reference=None,
    ) -> io.NodeOutput:
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
        tokens = _tokenize_semantic(
            clip, prompt, first_frame, last_frame, presentation
        )
        _prepare_h3_semantic_encoder(clip, tokens)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(H3SemanticReferenceData(conditioning, manifest))


class H3BuildConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3BuildConditioning",
            display_name="H3 Build Conditioning",
            category="Turing Utils/conditioning/minimax",
            description=(
                "Combine a reusable Qwen semantic reference with current-resolution "
                "first-last-frame, image, video, and audio VAE references."
            ),
            inputs=[
                H3SemanticReferenceType.Input("semantic_reference"),
                io.Latent.Input("latent"),
                H3KeyframeReferenceType.Input("first_frame", optional=True),
                H3KeyframeReferenceType.Input("last_frame", optional=True),
                H3ImageReferenceType.Input("image_reference", optional=True),
                H3VideoReferenceType.Input("video_reference", optional=True),
                H3AudioReferenceType.Input("audio_reference", optional=True),
            ],
            outputs=[io.Conditioning.Output("conditioning")],
        )

    @classmethod
    def execute(
        cls,
        semantic_reference,
        latent,
        first_frame=None,
        last_frame=None,
        image_reference=None,
        video_reference=None,
        audio_reference=None,
    ) -> io.NodeOutput:
        if not isinstance(semantic_reference, H3SemanticReferenceData):
            raise ValueError("semantic_reference must come from H3 Semantic Reference")
        first_frame = _keyframe_reference(first_frame, "first_frame")
        last_frame = _keyframe_reference(last_frame, "last_frame")
        manifest = _manifest(
            first_frame,
            last_frame,
            image_reference,
            video_reference,
            audio_reference,
        )
        if manifest != semantic_reference.manifest:
            raise ValueError(
                "Semantic and DiT H3 reference structures differ: "
                f"semantic={semantic_reference.manifest}, dit={manifest}"
            )

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
        return io.NodeOutput(conditioning)


class H3LatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3LatentInfo",
            display_name="H3 Latent Info",
            category="Turing Utils/latent",
            description=(
                "Read the decoded pixel width, height, canonical frame count, and "
                "model FPS from an H3 video or nested AV latent without decoding it."
            ),
            inputs=[io.Latent.Input("latent")],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.Float.Output("fps"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        return io.NodeOutput(*h3_latent_info(latent))


__all__ = [
    "H3AudioReference",
    "H3AudioReferenceData",
    "H3BuildConditioning",
    "H3KeyframeReference",
    "H3KeyframeReferenceData",
    "H3ImageReference",
    "H3ImageReferenceData",
    "H3LatentInfo",
    "H3ReferenceManifest",
    "H3SemanticReference",
    "H3SemanticReferenceData",
    "H3VideoReference",
    "H3VideoReferenceData",
    "h3_latent_info",
]
