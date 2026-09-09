"""Small compatibility helpers for supported ComfyUI MiniMax H3 layouts."""

from __future__ import annotations

import inspect


def accepts_parameter(callable_object, name: str) -> bool:
    """Return whether ``callable_object`` exposes a named parameter."""

    try:
        return name in inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False


def packed_layout_is_legacy(packed_layout_type) -> bool:
    """Identify the pre-guide-audio PackedLayout contract."""

    return accepts_parameter(packed_layout_type, "frame_count")


def make_packed_layout(
    packed_layout_type,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    payload,
):
    """Build PackedLayout across the old first/last and current guide APIs."""

    kwargs = {
        "keyframes": payload.get("keyframes"),
        "refs": payload.get("refs"),
    }
    if packed_layout_is_legacy(packed_layout_type):
        kwargs["frame_count"] = payload.get("frame_count")
    return packed_layout_type(
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        **kwargs,
    )


def keyframe_condition_rows(packed_layout_type, keyframes, frame_rows: int):
    """Return visual/audio rows using the active PackedLayout semantics."""

    if packed_layout_is_legacy(packed_layout_type):
        return len(keyframes or ()) * int(frame_rows), 0

    visual_rows = 0
    audio_rows = 0
    for keyframe in keyframes or ():
        latent = keyframe.get("latent")
        if latent is not None and getattr(latent, "ndim", 0) >= 3:
            visual_rows += int(latent.shape[2]) * int(frame_rows)
        audio_latent = keyframe.get("audio_latent")
        if audio_latent is not None and getattr(audio_latent, "ndim", 0) >= 1:
            audio_rows += int(audio_latent.shape[-1]) * 2
    return visual_rows, audio_rows


__all__ = [
    "accepts_parameter",
    "keyframe_condition_rows",
    "make_packed_layout",
    "packed_layout_is_legacy",
]
