"""Loader-independent MiniMax H3 attention layout publication.

Sparse attention sees only the flattened packed token sequence.  This module
bridges the official H3 runtime information (nested latent shapes and block
``mod_segments``) into a small, per-forward ``transformer_options`` contract.
It deliberately contains no quantization, Turing, or custom-loader policy.
"""

from __future__ import annotations

import inspect
import math
import weakref

from ...log import get_logger
from ...attention.layout import (
    ATTENTION_LAYOUT_KEY,
    AttentionSegment,
    AttentionSemanticLayout,
    AttentionTopology,
    LayoutProviderStatus,
    has_complete_attention_layout,
)
from ..methods import OriginalMethod, weak_method
from .activation_policy import ActivationRuntimePlan
from .compat import accepts_parameter, make_packed_layout


LOG = get_logger("minimax.layout")
MINIMAX_H3_LAYOUT_KIND = "minimax_h3"
H3_IMAGE_SOL_STRATEGY = "h3_image_sol"
H3_IMAGE_SOL_LAYOUT_KEY = "turing_utils_h3_image_sol_temporal_layout"
H3_IMAGE_SOL_LAYOUTS = ("dense_window", "dense_anchor_grid")
RUNTIME_CONTEXT_ATTR = "_turing_utils_minimax_runtime_context"
RUNTIME_PROVIDER_ATTR = "_turing_utils_minimax_layout_provider"
RUNTIME_OUTER_WRAPPER_KEY = "turing_utils_minimax_runtime_layout"
RUNTIME_FORWARD_PATCH_KEY = "diffusion_model.forward"
_FORWARD_PROVIDER_ATTR = "_turing_utils_minimax_layout_forward"
_MODEL_FORWARD_PROVIDER_ATTR = "_turing_utils_minimax_layout_model_forward"
_BLOCK_FORWARD_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
)
_BLOCK_FORWARD_PARAMETERS_WITH_ATTENTION = (
    *_BLOCK_FORWARD_PARAMETERS,
    "attention",
)
_LAYOUT_FIELDS = {
    "protocol_version",
    "provider",
    "query_segments",
    "key_segments",
    "topologies",
    "dense_prefix_tokens",
    "topology_start_tokens",
    "topology_tokens",
    "tokens_per_frame",
    "spatial_tokens_height",
    "spatial_tokens_width",
    "segments",
    "layer_index",
    "layer_count",
}
_LOGGED_H3_IMAGE_SOL_LAYOUTS = set()


def _root_and_diffusion_model(model):
    root = getattr(model, "model", model)
    return root, getattr(root, "diffusion_model", None)


def _is_minimax_h3_diffusion_model(diffusion_model) -> bool:
    if diffusion_model is None:
        return False
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def is_minimax_h3_model(model) -> bool:
    _, diffusion_model = _root_and_diffusion_model(model)
    return _is_minimax_h3_diffusion_model(diffusion_model)


def _compatible_block_forward(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    return parameters in {
        _BLOCK_FORWARD_PARAMETERS,
        _BLOCK_FORWARD_PARAMETERS_WITH_ATTENTION,
    }


def make_minimax_runtime_context_wrapper(base_model):
    """Expose the current sampler-owned nested shapes for packed H3 blocks."""

    def outer_sample_wrapper(executor, *args, **kwargs):
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is None and len(args) > 8:
            latent_shapes = args[8]
        previous = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
        setattr(
            base_model,
            RUNTIME_CONTEXT_ATTR,
            {
                "latent_shapes": latent_shapes,
                # The same mutable object survives the shallow runtime-context
                # copies made for every diffusion forward and denoising step.
                "activation_plan": ActivationRuntimePlan(),
            },
        )
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(base_model, RUNTIME_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(base_model, RUNTIME_CONTEXT_ATTR, previous)

    return outer_sample_wrapper


def _runtime_latent_shapes(base_model):
    context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    latent_shapes = context.get("latent_shapes") if isinstance(context, dict) else None
    if latent_shapes is None:
        # ComfyUI's sampler also publishes this directly on BaseModel.  Keeping
        # it as a fallback makes the provider tolerant of wrapper ordering.
        latent_shapes = getattr(base_model, "latent_shapes", None)
    return latent_shapes


def _runtime_packed_layout(base_model):
    context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    return context.get("packed_layout") if isinstance(context, dict) else None


def _resolve_packed_layout(diffusion_model, x, context, payload):
    if (
        not isinstance(x, (tuple, list))
        or len(x) < 2
        or not hasattr(x[0], "shape")
        or not hasattr(x[1], "shape")
        or not hasattr(context, "shape")
    ):
        return None
    try:
        from comfy.ldm.minimax.model import PackedLayout

        patch_size = tuple(int(value) for value in diffusion_model.patch_size)
        latent_t, latent_h, latent_w = (
            math.ceil(int(size) / patch) * patch
            for size, patch in zip(x[0].shape[2:5], patch_size)
        )
        signature = (
            int(context.shape[1]),
            latent_t,
            latent_h,
            latent_w,
            int(x[1].shape[-1]),
        )
        layout = payload.get("layout")
        if layout is not None and getattr(layout, "signature", None) == signature:
            return layout
        return make_packed_layout(
            PackedLayout,
            signature[0],
            latent_t,
            latent_h,
            latent_w,
            signature[4],
            payload,
        )
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


def _reference_descriptors(refs):
    """Return PackedLayout reference segments and their temporal metadata."""
    descriptors = []
    for reference in refs or ():
        kind = reference.get("kind")
        if kind == "image":
            descriptors.append(("reference_image", 1))
        elif kind == "audio":
            if int(reference.get("ref_audio_t", 0)) > 0:
                descriptors.append(("reference_audio", None))
        elif kind in ("video", "video_audio"):
            if int(reference.get("ref_audio_t", 0)) > 0:
                descriptors.append(("reference_audio", None))
            descriptors.append(
                ("reference_video", int(reference.get("latent_t", 0) or 0))
            )
    return descriptors


def _append_reference_video_segments(translated, start: int, stop: int, frames: int):
    """Split a reference clip's first/last latent frames from its interior."""
    tokens = stop - start
    if frames <= 0 or tokens % frames:
        translated.append((start, stop, "reference_video"))
        return
    tokens_per_frame = tokens // frames
    first_stop = start + tokens_per_frame
    translated.append((start, first_stop, "reference_video_anchor"))
    if frames > 2:
        translated.append((first_stop, stop - tokens_per_frame, "reference_video"))
    if frames > 1:
        translated.append((stop - tokens_per_frame, stop, "reference_video_anchor"))


def minimax_attention_segments(base_model):
    """Translate H3's PackedLayout into model-independent modality spans."""
    packed_layout = _runtime_packed_layout(base_model)
    packed_segments = getattr(packed_layout, "segments", None)
    if not isinstance(packed_segments, (list, tuple)) or not packed_segments:
        return ()
    context = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
    references = context.get("refs") if isinstance(context, dict) else None
    reference_descriptors = iter(_reference_descriptors(references))
    translated = []
    for start, stop, kind in packed_segments:
        start, stop = int(start), int(stop)
        if kind == "text":
            role = "text"
        elif kind == "cond":
            role = "reference_image"
        elif kind == "cond_audio":
            role = "reference_audio"
        elif kind in ("ref_img", "ref_audio"):
            descriptor = next(reference_descriptors, None)
            role = descriptor[0] if descriptor is not None else None
            expected = "reference_audio" if kind == "ref_audio" else None
            if role is None or (expected is not None and role != expected):
                return ()
            if kind == "ref_img" and role not in {
                "reference_image",
                "reference_video",
            }:
                return ()
            if role == "reference_video":
                _append_reference_video_segments(
                    translated, start, stop, descriptor[1]
                )
                continue
        elif kind == "audio":
            role = "target_audio"
        elif kind == "video":
            role = "target_video"
        else:
            return ()
        translated.append((start, stop, role))
    if next(reference_descriptors, None) is not None:
        return ()
    return tuple(translated)


def minimax_temporal_topology(base_model, diffusion_model, mod_segments):
    """Describe the contiguous target-video tail of the current H3 sequence."""
    if base_model is None or diffusion_model is None or not mod_segments:
        return {}
    latent_shapes = _runtime_latent_shapes(base_model)
    if not isinstance(latent_shapes, (list, tuple)) or not latent_shapes:
        return {}
    video_shape = tuple(int(value) for value in latent_shapes[0])
    if len(video_shape) != 5:
        return {}
    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        return {}
    pt, ph, pw = patch_size
    frames = math.ceil(video_shape[2] / pt)
    spatial_tokens_height = math.ceil(video_shape[3] / ph)
    spatial_tokens_width = math.ceil(video_shape[4] / pw)
    tokens_per_frame = spatial_tokens_height * spatial_tokens_width
    topology_tokens = frames * tokens_per_frame
    topology_start = int(mod_segments[-1][0])
    if int(mod_segments[-1][1]) - topology_start != topology_tokens:
        return {}
    return {
        "topology_start_tokens": topology_start,
        "topology_tokens": topology_tokens,
        "tokens_per_frame": tokens_per_frame,
        "spatial_tokens_height": spatial_tokens_height,
        "spatial_tokens_width": spatial_tokens_width,
    }


def h3_image_sol_dense_slices(latent_frames: int, temporal_layout: str):
    """Return H3 latent-time slices whose Query and K/V stay exact."""
    latent_frames = int(latent_frames)
    temporal_layout = str(temporal_layout).strip().lower()
    if temporal_layout not in H3_IMAGE_SOL_LAYOUTS:
        raise ValueError(
            "H3 image Sol temporal_layout must be dense_window or "
            "dense_anchor_grid"
        )
    if latent_frames < 2 or (latent_frames - 2) % 5:
        return tuple(range(max(latent_frames, 0)))
    dense = {0, 1}
    if temporal_layout == "dense_anchor_grid":
        dense.update(range(0, latent_frames, 5))
    return tuple(sorted(dense))


def _minimax_semantic_segments(raw_segments, topology, temporal_layout):
    topology_id = "target_video"
    tokens_per_frame = int(topology["tokens_per_frame"])
    latent_frames = int(topology["topology_tokens"]) // tokens_per_frame
    dense_slices = set(h3_image_sol_dense_slices(latent_frames, temporal_layout))
    log_key = (temporal_layout, latent_frames, tokens_per_frame)
    if log_key not in _LOGGED_H3_IMAGE_SOL_LAYOUTS:
        residual_slices = tuple(
            frame for frame in range(latent_frames) if frame not in dense_slices
        )
        LOG.info(
            "H3 image Sol layout: temporal_layout=%s latent_t=%d "
            "dense_slices=%s residual_slices=%s",
            temporal_layout,
            latent_frames,
            tuple(sorted(dense_slices)),
            residual_slices,
        )
        _LOGGED_H3_IMAGE_SOL_LAYOUTS.add(log_key)
    segments = []
    for start, stop, role in raw_segments:
        if role != "target_video":
            segments.append(AttentionSegment.for_role(start, stop, role))
            continue
        for frame in range(latent_frames):
            frame_start = int(start) + frame * tokens_per_frame
            frame_stop = frame_start + tokens_per_frame
            if frame in dense_slices:
                segments.append(
                    AttentionSegment(
                        start=frame_start,
                        stop=frame_stop,
                        role=role,
                        sparse_query_allowed=False,
                        sparse_key_allowed=False,
                        exact_kv=True,
                        topology_id=topology_id,
                    )
                )
            else:
                segments.append(
                    AttentionSegment.for_role(
                        frame_start,
                        frame_stop,
                        role,
                        topology_id=topology_id,
                    )
                )
    return tuple(segments)


def publish_minimax_attention_layout(
    transformer_options,
    mod_segments,
    *,
    layer_index: int,
    layer_count: int,
    base_model,
    diffusion_model,
) -> bool:
    """Publish exact H3 packed-sequence semantics for one transformer block."""
    if not isinstance(transformer_options, dict) or not mod_segments:
        return False
    topology = minimax_temporal_topology(base_model, diffusion_model, mod_segments)
    raw_segments = minimax_attention_segments(base_model)
    if not topology or not raw_segments:
        expected_layout = {
            "provider": MINIMAX_H3_LAYOUT_KIND,
            "dense_prefix_tokens": int(mod_segments[-1][0]),
            "layer_index": int(layer_index),
            "layer_count": int(layer_count),
        }
    else:
        topology_id = "target_video"
        temporal_layout = (
            transformer_options.get(H3_IMAGE_SOL_LAYOUT_KEY)
            if transformer_options.get("turing_utils_attention_strategy")
            == H3_IMAGE_SOL_STRATEGY
            else None
        )
        if temporal_layout in H3_IMAGE_SOL_LAYOUTS:
            segments = _minimax_semantic_segments(
                raw_segments, topology, temporal_layout
            )
        else:
            segments = tuple(
                AttentionSegment.for_role(
                    start,
                    stop,
                    role,
                    topology_id=topology_id if role == "target_video" else None,
                )
                for start, stop, role in raw_segments
            )
        semantic = AttentionSemanticLayout(
            provider=MINIMAX_H3_LAYOUT_KIND,
            query_segments=segments,
            key_segments=segments,
            topologies=(
                AttentionTopology(
                    topology_id,
                    "video_grid",
                    int(topology["topology_start_tokens"]),
                    int(topology["topology_start_tokens"])
                    + int(topology["topology_tokens"]),
                    int(topology["tokens_per_frame"]),
                    int(topology["spatial_tokens_height"]),
                    int(topology["spatial_tokens_width"]),
                ),
            ),
            layer_index=int(layer_index),
            layer_count=int(layer_count),
        )
        expected_layout = semantic.to_wire()
    layout = transformer_options.get(ATTENTION_LAYOUT_KEY)
    # Never retain topology from an earlier prompt/window when current shapes
    # cannot be validated. Unknown extension fields may compose, but every
    # provider-owned field is rebuilt atomically for this exact block call.
    preserved = (
        {key: value for key, value in layout.items() if key not in _LAYOUT_FIELDS}
        if isinstance(layout, dict)
        else {}
    )
    updated = {**preserved, **expected_layout}
    if layout != updated:
        transformer_options[ATTENTION_LAYOUT_KEY] = updated
    return "topologies" in expected_layout


def has_complete_minimax_attention_layout(
    transformer_options,
    sequence_length: int | None = None,
) -> bool:
    return has_complete_attention_layout(
        transformer_options,
        sequence_length,
        provider=MINIMAX_H3_LAYOUT_KIND,
    )


def _forward_has_provider(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _FORWARD_PROVIDER_ATTR, False))


def _model_forward_has_provider(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _MODEL_FORWARD_PROVIDER_ATTR, False))


def _make_model_forward(base_model, diffusion_model, original):
    original = OriginalMethod.capture(original, diffusion_model)
    base_model = weakref.proxy(base_model)

    def forward(
        self,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        **kwargs,
    ):
        previous = getattr(base_model, RUNTIME_CONTEXT_ATTR, None)
        runtime = dict(previous) if isinstance(previous, dict) else {}
        payload = minimax_payload if isinstance(minimax_payload, dict) else {}
        runtime.update(
            packed_layout=_resolve_packed_layout(
                self, x, context, payload
            ),
            refs=payload.get("refs"),
        )
        setattr(base_model, RUNTIME_CONTEXT_ATTR, runtime)
        try:
            return original(
                self,
                x,
                timestep,
                context,
                transformer_options=transformer_options,
                minimax_payload=minimax_payload,
                **kwargs,
            )
        finally:
            if previous is None:
                try:
                    delattr(base_model, RUNTIME_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(base_model, RUNTIME_CONTEXT_ATTR, previous)

    setattr(forward, _MODEL_FORWARD_PROVIDER_ATTR, True)
    return weak_method(forward, diffusion_model)


def _make_layout_forward(
    block,
    original,
    *,
    layer_index: int,
    layer_count: int,
    base_model,
    diffusion_model,
):
    original = OriginalMethod.capture(original, block)
    supports_attention = accepts_parameter(original.function, "attention")
    base_model = weakref.proxy(base_model)
    diffusion_model = weakref.proxy(diffusion_model)

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        attention=None,
    ):
        publish_minimax_attention_layout(
            transformer_options,
            mod_segments,
            layer_index=layer_index,
            layer_count=layer_count,
            base_model=base_model,
            diffusion_model=diffusion_model,
        )
        kwargs = {"transformer_options": transformer_options}
        if supports_attention:
            kwargs["attention"] = attention
        return original(
            self, x, t_emb, mod_segments, rope_freqs, **kwargs
        )

    setattr(forward, _FORWARD_PROVIDER_ATTR, True)
    return weak_method(forward, block)


def mark_forward_as_minimax_layout_provider(forward):
    """Mark a fused replacement that already calls publish_minimax_attention_layout."""
    function = getattr(forward, "__func__", forward)
    setattr(function, _FORWARD_PROVIDER_ATTR, True)
    return forward


def ensure_minimax_attention_layout_provider(model) -> LayoutProviderStatus:
    """Install the H3 layout bridge on any standard ComfyUI ModelPatcher."""
    base_model, diffusion_model = _root_and_diffusion_model(model)
    if not _is_minimax_h3_diffusion_model(diffusion_model):
        return LayoutProviderStatus(None, False, "not_minimax_h3")
    if not callable(getattr(model, "add_object_patch", None)) or not callable(
        getattr(model, "add_wrapper_with_key", None)
    ):
        return LayoutProviderStatus(
            MINIMAX_H3_LAYOUT_KIND, False, "model_patcher_api_unavailable"
        )
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return LayoutProviderStatus(MINIMAX_H3_LAYOUT_KIND, False, "blocks_unavailable")
    if any(not _compatible_block_forward(block.forward) for block in blocks):
        return LayoutProviderStatus(
            MINIMAX_H3_LAYOUT_KIND, False, "block_forward_contract_changed"
        )

    try:
        import comfy.patcher_extension
    except ImportError:
        return LayoutProviderStatus(
            MINIMAX_H3_LAYOUT_KIND, False, "patcher_extension_unavailable"
        )

    model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        RUNTIME_OUTER_WRAPPER_KEY,
        make_minimax_runtime_context_wrapper(base_model),
    )
    object_patches = getattr(model, "object_patches", {})
    current_model_forward = object_patches.get(
        RUNTIME_FORWARD_PATCH_KEY,
        diffusion_model.forward,
    )
    if not _model_forward_has_provider(current_model_forward):
        model.add_object_patch(
            RUNTIME_FORWARD_PATCH_KEY,
            _make_model_forward(base_model, diffusion_model, current_model_forward),
        )
    for layer_index, block in enumerate(blocks):
        key = f"diffusion_model.blocks.{layer_index}.forward"
        current = object_patches.get(key, block.forward)
        if _forward_has_provider(current):
            continue
        model.add_object_patch(
            key,
            _make_layout_forward(
                block,
                current,
                layer_index=layer_index,
                layer_count=len(blocks),
                base_model=base_model,
                diffusion_model=diffusion_model,
            ),
        )

    setattr(base_model, RUNTIME_PROVIDER_ATTR, True)
    LOG.info(
        "Enabled loader-independent MiniMax H3 attention layout provider on %d blocks",
        len(blocks),
    )
    return LayoutProviderStatus(MINIMAX_H3_LAYOUT_KIND, True)
