"""Semantic self-attention layout publication for Wan model families.

The provider describes only the flattened self-attention sequence.  It does
not depend on quantization or a Turing device, so sparse attention can use it
with an official ComfyUI loader as well as with this plugin's Wan adapter.
"""

from __future__ import annotations

import inspect
import math

import torch

from ..log import get_logger
from .methods import OriginalMethod, weak_method
from ..attention.layout import (
    ATTENTION_LAYOUT_KEY,
    AttentionSegment,
    AttentionSemanticLayout,
    AttentionTopology,
    LayoutProviderStatus,
)


LOG = get_logger("wan.layout")
WAN_LAYOUT_KIND = "wan_self_attention"
SCAIL_LAYOUT_KIND = "scail_self_attention"
_FORWARD_ORIG_PATCH_KEY = "diffusion_model.forward_orig"
_FORWARD_PROVIDER_ATTR = "_turing_utils_wan_layout_forward"
_FORWARD_ORIG_PARAMETERS = (
    "x",
    "t",
    "context",
    "clip_fea",
    "freqs",
    "transformer_options",
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


def _root_and_diffusion_model(model):
    root = getattr(model, "model", model)
    return root, getattr(root, "diffusion_model", None)


def _supported_layout_kind(diffusion_model) -> str | None:
    if diffusion_model is None:
        return None
    try:
        from comfy.ldm.wan.model import SCAILWanModel, WanModel
    except ImportError:
        return None
    if isinstance(diffusion_model, SCAILWanModel) and (
        type(diffusion_model).forward_orig is SCAILWanModel.forward_orig
    ):
        return SCAIL_LAYOUT_KIND
    if isinstance(diffusion_model, WanModel) and (
        type(diffusion_model).forward_orig is WanModel.forward_orig
    ):
        return WAN_LAYOUT_KIND
    return None


def _compatible_forward_orig(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    return parameters[: len(_FORWARD_ORIG_PARAMETERS)] == _FORWARD_ORIG_PARAMETERS


def _grid_from_video(latent, patch_size) -> tuple[int, int, int] | None:
    if not torch.is_tensor(latent) or latent.ndim != 5:
        return None
    sizes = tuple(int(value) for value in latent.shape[-3:])
    patches = tuple(int(value) for value in patch_size)
    if len(patches) != 3 or any(value <= 0 for value in patches):
        return None
    return tuple(math.ceil(size / patch) for size, patch in zip(sizes, patches))


def _image_tokens(latent, patch_size) -> int | None:
    if not torch.is_tensor(latent) or latent.ndim != 4:
        return None
    height, width = (int(value) for value in latent.shape[-2:])
    patch_height, patch_width = (int(value) for value in patch_size[-2:])
    if min(height, width, patch_height, patch_width) <= 0:
        return None
    tokens_height = height // patch_height
    tokens_width = width // patch_width
    return tokens_height * tokens_width if min(tokens_height, tokens_width) > 0 else None


def _append_video_reference(
    segments: list[AttentionSegment],
    topologies: list[AttentionTopology],
    *,
    start: int,
    grid: tuple[int, int, int],
    topology_id: str,
) -> int:
    frames, height, width = grid
    frame_tokens = height * width
    stop = start + frames * frame_tokens
    topologies.append(
        AttentionTopology(
            topology_id,
            "video_grid",
            start,
            stop,
            frame_tokens,
            height,
            width,
        )
    )
    first_stop = start + frame_tokens
    segments.append(
        AttentionSegment.for_role(
            start,
            first_stop,
            "reference_video_anchor",
            topology_id=topology_id,
        )
    )
    if frames > 2:
        segments.append(
            AttentionSegment.for_role(
                first_stop,
                stop - frame_tokens,
                "reference_video",
                topology_id=topology_id,
            )
        )
    if frames > 1:
        segments.append(
            AttentionSegment.for_role(
                stop - frame_tokens,
                stop,
                "reference_video_anchor",
                topology_id=topology_id,
            )
        )
    return stop


def build_wan_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
) -> AttentionSemanticLayout | None:
    """Build the exact token order used by ``WanModel.forward_orig``."""
    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    target_grid = _grid_from_video(x, patch_size)
    if target_grid is None:
        return None

    segments: list[AttentionSegment] = []
    topologies: list[AttentionTopology] = []
    cursor = 0

    reference = kwargs.get("reference_latent")
    if getattr(diffusion_model, "ref_conv", None) is not None and reference is not None:
        count = _image_tokens(reference, patch_size)
        if count is None:
            return None
        segments.append(
            AttentionSegment.for_role(cursor, cursor + count, "reference_image")
        )
        cursor += count

    frames, height, width = target_grid
    target_tokens = frames * height * width
    target_start = cursor
    cursor += target_tokens
    segments.append(
        AttentionSegment.for_role(
            target_start,
            cursor,
            "target_video",
            topology_id="target_video",
        )
    )
    topologies.append(
        AttentionTopology(
            "target_video",
            "video_grid",
            target_start,
            cursor,
            height * width,
            height,
            width,
        )
    )

    context_latents = kwargs.get("context_latents")
    if context_latents is not None:
        if not isinstance(context_latents, (tuple, list)):
            return None
        for index, latent in enumerate(context_latents):
            grid = _grid_from_video(latent, patch_size)
            if grid is None:
                return None
            if grid[0] == 1:
                count = grid[1] * grid[2]
                segments.append(
                    AttentionSegment.for_role(
                        cursor,
                        cursor + count,
                        "reference_image",
                    )
                )
                cursor += count
            else:
                cursor = _append_video_reference(
                    segments,
                    topologies,
                    start=cursor,
                    grid=grid,
                    topology_id=f"context_video_{index}",
                )

    layer_count = len(getattr(diffusion_model, "blocks", ()))
    if layer_count <= 0:
        return None
    layout = AttentionSemanticLayout(
        provider=WAN_LAYOUT_KIND,
        query_segments=tuple(segments),
        key_segments=tuple(segments),
        topologies=tuple(topologies),
        layer_index=0,
        layer_count=layer_count,
    )
    return layout if layout.validate(cursor, cursor) is None else None


def build_scail_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
) -> AttentionSemanticLayout | None:
    """Build the reference, target, pose order used by SCAIL self-attention."""
    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    target_grid = _grid_from_video(x, patch_size)
    if target_grid is None:
        return None

    segments: list[AttentionSegment] = []
    topologies: list[AttentionTopology] = []
    cursor = 0

    reference = kwargs.get("reference_latent")
    if reference is not None:
        reference_grid = _grid_from_video(reference, patch_size)
        if reference_grid is None:
            return None
        ref_frames, ref_height, ref_width = reference_grid
        ref_frame_tokens = ref_height * ref_width
        ref_stop = cursor + ref_frames * ref_frame_tokens
        segments.append(
            AttentionSegment.for_role(
                cursor,
                ref_stop,
                "reference_image",
                topology_id="reference_images",
            )
        )
        topologies.append(
            AttentionTopology(
                "reference_images",
                "video_grid",
                cursor,
                ref_stop,
                ref_frame_tokens,
                ref_height,
                ref_width,
            )
        )
        cursor = ref_stop

    frames, height, width = target_grid
    frame_tokens = height * width
    target_stop = cursor + frames * frame_tokens
    segments.append(
        AttentionSegment.for_role(
            cursor,
            target_stop,
            "target_video",
            topology_id="target_video",
        )
    )
    topologies.append(
        AttentionTopology(
            "target_video",
            "video_grid",
            cursor,
            target_stop,
            frame_tokens,
            height,
            width,
        )
    )
    cursor = target_stop

    pose = kwargs.get("pose_latents")
    if pose is not None:
        pose_grid = _grid_from_video(pose, patch_size)
        if pose_grid is None:
            return None
        pose_frames, pose_height, pose_width = pose_grid
        pose_frame_tokens = pose_height * pose_width
        pose_stop = cursor + pose_frames * pose_frame_tokens
        segments.append(
            AttentionSegment.for_role(
                cursor,
                pose_stop,
                "pose_video",
                topology_id="pose_video",
            )
        )
        topologies.append(
            AttentionTopology(
                "pose_video",
                "video_grid",
                cursor,
                pose_stop,
                pose_frame_tokens,
                pose_height,
                pose_width,
            )
        )
        cursor = pose_stop

    layer_count = len(getattr(diffusion_model, "blocks", ()))
    if layer_count <= 0:
        return None
    layout = AttentionSemanticLayout(
        provider=SCAIL_LAYOUT_KIND,
        query_segments=tuple(segments),
        key_segments=tuple(segments),
        topologies=tuple(topologies),
        layer_index=0,
        layer_count=layer_count,
    )
    return layout if layout.validate(cursor, cursor) is None else None


def _publish_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
    *,
    provider: str,
    builder,
) -> bool:
    if not isinstance(transformer_options, dict):
        return False
    semantic = builder(diffusion_model, x, transformer_options, kwargs)
    previous = transformer_options.get(ATTENTION_LAYOUT_KEY)
    preserved = (
        {key: value for key, value in previous.items() if key not in _LAYOUT_FIELDS}
        if isinstance(previous, dict)
        else {}
    )
    if semantic is None:
        transformer_options[ATTENTION_LAYOUT_KEY] = {
            **preserved,
            "provider": provider,
        }
        return False
    transformer_options[ATTENTION_LAYOUT_KEY] = semantic.to_wire(
        extensions=preserved
    )
    return True


def publish_wan_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
) -> bool:
    return _publish_attention_layout(
        diffusion_model,
        x,
        transformer_options,
        kwargs,
        provider=WAN_LAYOUT_KIND,
        builder=build_wan_attention_layout,
    )


def publish_scail_attention_layout(
    diffusion_model,
    x,
    transformer_options,
    kwargs,
) -> bool:
    return _publish_attention_layout(
        diffusion_model,
        x,
        transformer_options,
        kwargs,
        provider=SCAIL_LAYOUT_KIND,
        builder=build_scail_attention_layout,
    )


def _forward_has_provider(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _FORWARD_PROVIDER_ATTR, False))


def _make_layout_forward(diffusion_model, original, publisher):
    original = OriginalMethod.capture(original, diffusion_model)

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        publisher(
            self,
            x,
            transformer_options,
            kwargs,
        )
        return original(
            self,
            x,
            t,
            context,
            clip_fea=clip_fea,
            freqs=freqs,
            transformer_options=transformer_options,
            **kwargs,
        )

    setattr(forward_orig, _FORWARD_PROVIDER_ATTR, True)
    return weak_method(forward_orig, diffusion_model)


def ensure_wan_attention_layout_provider(model) -> LayoutProviderStatus:
    """Install a loader-independent layout provider for Wan model families."""
    _, diffusion_model = _root_and_diffusion_model(model)
    layout_kind = _supported_layout_kind(diffusion_model)
    if layout_kind is None:
        return LayoutProviderStatus(None, False, "not_compatible_wan")
    if not callable(getattr(model, "add_object_patch", None)):
        return LayoutProviderStatus(
            layout_kind,
            False,
            "model_patcher_api_unavailable",
        )
    if not _compatible_forward_orig(diffusion_model.forward_orig):
        return LayoutProviderStatus(
            layout_kind,
            False,
            "forward_orig_contract_changed",
        )
    object_patches = getattr(model, "object_patches", {})
    current = object_patches.get(
        _FORWARD_ORIG_PATCH_KEY,
        diffusion_model.forward_orig,
    )
    if not _forward_has_provider(current):
        publisher = (
            publish_scail_attention_layout
            if layout_kind == SCAIL_LAYOUT_KIND
            else publish_wan_attention_layout
        )
        model.add_object_patch(
            _FORWARD_ORIG_PATCH_KEY,
            _make_layout_forward(diffusion_model, current, publisher),
        )
    LOG.info("Enabled loader-independent %s attention layout provider", layout_kind)
    return LayoutProviderStatus(layout_kind, True)


__all__ = [
    "SCAIL_LAYOUT_KIND",
    "WAN_LAYOUT_KIND",
    "build_scail_attention_layout",
    "build_wan_attention_layout",
    "ensure_wan_attention_layout_provider",
    "publish_scail_attention_layout",
    "publish_wan_attention_layout",
]
