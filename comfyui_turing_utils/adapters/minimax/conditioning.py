"""MiniMax H3 conditioning compatibility."""

from __future__ import annotations

from functools import wraps


def repair_combined_minimax_payload(out, kwargs):
    """Keep visual rows in the same keyframe-then-reference order as PackedLayout."""
    keyframes = kwargs.get("minimax_keyframes") or ()
    refs = kwargs.get("minimax_refs") or ()
    if not keyframes or not refs or not isinstance(out, dict):
        return out
    holder = out.get("minimax_payload")
    payload = getattr(holder, "cond", None)
    if not isinstance(payload, dict):
        return out

    repaired = dict(payload)
    repaired["cond_video_latents"] = [
        *(item["latent"] for item in keyframes),
        *(item["latent"] for item in refs if "latent" in item),
    ]
    repaired["cond_audio_latents"] = [
        item["audio_latent"] for item in refs if item.get("audio_latent") is not None
    ]
    updated = dict(out)
    updated["minimax_payload"] = holder._copy_with(repaired)
    return updated


def install_combined_minimax_conditioning_support() -> bool:
    """Fix the upstream keyframe/reference payload overwrite once per process."""
    import comfy.model_base

    model_class = comfy.model_base.MiniMaxH3
    current = model_class.extra_conds
    if getattr(current, "_turing_utils_combined_h3_references", False):
        return False

    @wraps(current)
    def extra_conds(self, **kwargs):
        return repair_combined_minimax_payload(current(self, **kwargs), kwargs)

    extra_conds._turing_utils_combined_h3_references = True
    model_class.extra_conds = extra_conds
    return True


__all__ = [
    "install_combined_minimax_conditioning_support",
    "repair_combined_minimax_payload",
]
