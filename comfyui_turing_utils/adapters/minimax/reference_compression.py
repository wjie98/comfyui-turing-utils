"""H3 reference-only temporal compression before sampling preparation.

The schedule and physical-time correction follow Spark-VPack (Apache-2.0),
Spark-H3 f58ecced3ff45fb32673839578fc9d19d78fdad8. See
kernel/LICENSES/spark-h3-LICENSE.txt. This implementation also retains the
original final frame so existing reference-anchor protection remains valid.
"""

from __future__ import annotations

import torch

from ...log import get_logger


LOG = get_logger("minimax.vpack")
COMPRESSION_PHASES = {"4/5": (0, 1, 2, 3), "3/5": (0, 1, 3), "2/5": (0, 3)}
_WRAPPER_KEY = "turing_utils_h3_vpack"
_DENSE_T = "compact_ref2va_dense_latent_t"
_DENSE_INDICES = "compact_ref2va_dense_indices"
_TIME_MODE = "compact_ref2va_rope_time_mode"
_LAYOUT_MARKER = "_compact_ref2va_physical_time_applied"


def migrate_reference_compression_inputs(inputs):
    """Translate the pre-keep-ratio VPack fields in a native prompt."""
    field = "reference_video_compression"
    if field not in inputs:
        return inputs
    mode = inputs[field]
    ratio_field = f"{field}.compression_ratio"
    drop = inputs.get(ratio_field, "2/5")
    if isinstance(mode, dict):
        drop = mode.get("compression_ratio", drop)
        mode = mode.get(field)
    if mode == "off":
        keep = "5/5"
    elif mode == "vpack" and isinstance(drop, str):
        keep = {"1/5": "4/5", "2/5": "3/5", "3/5": "2/5"}.get(drop)
    else:
        return inputs
    if keep is None:
        return inputs
    migrated = dict(inputs)
    del migrated[field]
    migrated.pop(ratio_field, None)
    migrated.setdefault("vpack", keep)
    return migrated


def retained_frames(frames: int, ratio: str) -> list[int]:
    phases = COMPRESSION_PHASES[ratio]
    indices = [index for index in range(frames) if index % 5 in phases]
    if frames and indices[-1] != frames - 1:
        indices.append(frames - 1)
    return indices


def compress_conditions(conditions, ratio: str):
    """Copy only changed native sampler conditions; never modify node outputs."""
    packed_conditions = {}
    packed_latents = {}
    original_frames = kept_frames = 0
    for name, entries in conditions.items():
        packed_entries = []
        for entry in entries:
            refs = entry.get("minimax_refs")
            if not refs:
                packed_entries.append(entry)
                continue
            packed_refs = []
            changed = False
            for ref in refs:
                if ref.get("kind") not in {"video", "video_audio"} or _DENSE_INDICES in ref:
                    packed_refs.append(ref)
                    continue
                latent = ref["latent"]
                frames = int(ref["latent_t"])
                if latent.ndim != 5 or latent.shape[2] != frames:
                    raise ValueError("H3 VPack reference latent does not match latent_t")
                indices = retained_frames(frames, ratio)
                if len(indices) == frames:
                    packed_refs.append(ref)
                    continue
                key = id(latent)
                if key not in packed_latents:
                    selection = torch.tensor(indices, device=latent.device, dtype=torch.long)
                    packed_latents[key] = latent.index_select(2, selection)
                    original_frames += frames
                    kept_frames += len(indices)
                packed_refs.append({
                    **ref,
                    "latent": packed_latents[key],
                    "latent_t": len(indices),
                    _DENSE_T: frames,
                    _DENSE_INDICES: indices,
                    _TIME_MODE: "chunk_phase",
                    "compact_ref2va_rope_time_scale": 1.0,
                })
                changed = True
            packed_entries.append(dict(entry, minimax_refs=packed_refs) if changed else entry)
        packed_conditions[name] = packed_entries
    if original_frames:
        LOG.info("H3 VPack keep %s: reference latent frames %d -> %d (anchors retained)",
                 ratio, original_frames, kept_frames)
    else:
        LOG.info("H3 VPack skipped: no uncompressed reference video to reduce")
    return packed_conditions


def restore_reference_times(layout, refs):
    """Restore retained frames' native times before RoPE or sparse layout use."""
    if layout is None or getattr(layout, _LAYOUT_MARKER, False):
        return layout
    if not any(ref.get(_TIME_MODE) == "chunk_phase" for ref in refs or ()):
        return layout
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

    visual_refs = [ref for ref in refs if ref.get("kind") in {"image", "video", "video_audio"}]
    segments = [(start, stop) for start, stop, kind in layout.segments if kind == "ref_img"]
    if len(visual_refs) != len(segments):
        raise ValueError("H3 VPack reference/layout segments do not match")
    positions = layout.position_ids.clone()
    target_shift = 0.0
    for ref, (start, stop) in zip(visual_refs, segments):
        if ref.get(_TIME_MODE) != "chunk_phase":
            continue
        indices = ref[_DENSE_INDICES]
        frames = int(ref["latent_t"])
        dense_frames = int(ref[_DENSE_T])
        if (len(indices) != frames or not indices or indices[0] < 0
                or indices[-1] >= dense_frames
                or any(a >= b for a, b in zip(indices, indices[1:]))):
            raise ValueError("H3 VPack retained time indices do not match the reference")
        rows_per_frame, remainder = divmod(stop - start, frames)
        if remainder:
            raise ValueError("H3 VPack reference segment is not frame-aligned")
        dense_times = []
        dense_span = 0.0
        for index in range(dense_frames):
            dense_times.append(dense_span)
            dense_span += FRAME_RESCALE * FRAME_PER_TOKEN[index % 5]
        times = positions.new_tensor([dense_times[index] for index in indices])
        origin = positions[start, 0].clone()
        positions[start:stop, 0] = origin + times.repeat_interleave(rows_per_frame)
        compact_span = sum(FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(frames))
        audio_span = float(ref.get("ref_audio_t", 0))
        delta = max(audio_span, dense_span) - max(audio_span, compact_span)
        positions[stop:, 0] += delta
        target_shift += delta
    # FL2VA keyframes pack before references but share the target timeline.
    for start, stop, kind in layout.segments:
        if kind in {"cond", "cond_audio"}:
            positions[start:stop, 0] += target_shift
    layout.position_ids = positions
    setattr(layout, _LAYOUT_MARKER, True)
    return layout


def configure_reference_compression(model, ratio="5/5"):
    """Keep one branch-local preparation wrapper, independent of attention."""
    ratio = "5/5" if ratio is None else ratio
    if ratio not in {"5/5", *COMPRESSION_PHASES}:
        raise ValueError(f"H3 vpack must be one of {('5/5', *COMPRESSION_PHASES)}")
    if ratio == "5/5" and not getattr(model, "model_options", {}).get(_WRAPPER_KEY):
        return model
    import comfy.patcher_extension
    from .layout import ensure_minimax_attention_layout_provider, is_minimax_h3_model

    if ratio != "5/5" and not is_minimax_h3_model(model):
        LOG.warning_once("VPack skipped: reference compression requires MiniMax H3")
        return model
    patched = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING
    patched.remove_wrappers_with_key(wrapper_type, _WRAPPER_KEY)
    patched.model_options.pop(_WRAPPER_KEY, None)
    if ratio == "5/5":
        return patched
    status = ensure_minimax_attention_layout_provider(patched)
    if not status.installed:
        raise RuntimeError(f"H3 VPack cannot restore reference times: {status.reason}")

    def prepare(executor, model, noise_shape, conditions, *args, **kwargs):
        return executor(model, noise_shape, compress_conditions(conditions, ratio), *args, **kwargs)

    patched.model_options[_WRAPPER_KEY] = ratio
    patched.add_wrapper_with_key(wrapper_type, _WRAPPER_KEY, prepare)
    return patched
