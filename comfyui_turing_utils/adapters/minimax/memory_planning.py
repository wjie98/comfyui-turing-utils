"""MiniMax packed-sequence memory estimation and ComfyUI hooks."""

from __future__ import annotations

import dataclasses
import math

import torch

from ...log import get_logger
from ..memory import install_memory_hooks, scan_quantized_workspaces
from ..methods import OriginalMethod, weak_method
from ...quantization.dispatch import turing_int8_workspace_bytes
from ...quantization.fusions import convrot_weight_kind
from .activation_policy import (
    balanced_saturation_size,
    estimate_attention_lifecycle_peak,
)
from .layout import RUNTIME_CONTEXT_ATTR, make_minimax_runtime_context_wrapper


LOG = get_logger("minimax.memory")
_MEMORY_SHAPE_KEY = "turing_utils_minimax_packed_sequence"
_MEMORY_CONTEXT_ATTR = RUNTIME_CONTEXT_ATTR
_MEMORY_ADAPTER_ATTR = "_turing_utils_minimax_memory_adapter"


class _MiniMaxMemoryShape(list):
    """BaseModel-compatible synthetic shape carrying exact H3 row metadata."""

    def __init__(
        self,
        equivalent_area: int,
        *,
        full_rows: int,
        target_rows: int,
        target_visual_rows: int,
        target_audio_rows: int,
        visual_condition_rows: int,
        audio_condition_rows: int,
        hidden_size: int,
        video_row_width: int,
        audio_row_width: int,
    ):
        super().__init__((1, 1, max(int(equivalent_area), 0)))
        self.equivalent_area = max(int(equivalent_area), 0)
        self.full_rows = int(full_rows)
        self.target_rows = int(target_rows)
        self.target_visual_rows = int(target_visual_rows)
        self.target_audio_rows = int(target_audio_rows)
        self.visual_condition_rows = int(visual_condition_rows)
        self.audio_condition_rows = int(audio_condition_rows)
        self.hidden_size = int(hidden_size)
        self.video_row_width = int(video_row_width)
        self.audio_row_width = int(audio_row_width)

    def explicit_condition_bytes(self, dtype_size: int) -> int:
        """Known incremental buffers not allowed to fall below the heuristic."""
        condition_rows = max(self.full_rows - self.target_rows, 0)
        packed_hidden = condition_rows * self.hidden_size * int(dtype_size)

        visual_fp32 = self.visual_condition_rows * self.video_row_width * 4
        if self.visual_condition_rows:
            # With visual conditions H3 keeps the individual condition rows and
            # also assembles one target+condition FP32 row buffer.
            visual_fp32 += (
                self.target_visual_rows + self.visual_condition_rows
            ) * self.video_row_width * 4

        audio_fp32 = self.audio_condition_rows * self.audio_row_width * 4
        if self.audio_condition_rows:
            audio_fp32 += (
                self.target_audio_rows + self.audio_condition_rows
            ) * self.audio_row_width * 4
        return packed_hidden + visual_fp32 + audio_fp32


class _MiniMaxMemoryCond:
    """Lightweight model condition exposing the synthetic shape at runtime."""

    def __init__(self, plan: _MiniMaxMemoryShape):
        self.cond = plan

    def process_cond(self, batch_size, **kwargs):
        return self

    def can_concat(self, other):
        return (
            isinstance(other, _MiniMaxMemoryCond)
            and self.cond.full_rows == other.cond.full_rows
            and self.cond.target_rows == other.cond.target_rows
            and self.cond.equivalent_area == other.cond.equivalent_area
            and self.cond.visual_condition_rows
            == other.cond.visual_condition_rows
            and self.cond.audio_condition_rows == other.cond.audio_condition_rows
        )

    def concat(self, others):
        return self.cond

    def size(self):
        return self.cond


@dataclasses.dataclass(frozen=True, slots=True)
class _MiniMaxActivationProfile:
    hidden_size: int
    heads: int
    head_dim: int
    expanded_size: int

    def conservative_floor_bytes(self, rows: int, element_size: int) -> int:
        """Peak of the first safe streamed tiers, not a sum of serial ops."""
        rows = int(rows)
        element_size = int(element_size)
        block_heads = 256 // math.gcd(256, self.head_dim)
        group = balanced_saturation_size(
            self.heads,
            alignment=block_heads,
            minimum=math.ceil(self.heads / 4),
        )
        attention = estimate_attention_lifecycle_peak(
            rows=rows,
            heads=self.heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            element_size=element_size,
            head_group=group,
            compact_qk=True,
            cache_quantized_input=False,
            quantized_value=True,
        )

        qkv_scales = ((rows + 63) // 64) * self.heads * 5 * 4
        qkv_persistent = (
            rows
            * (
                2 * self.heads * self.head_dim
                + self.heads * self.head_dim * element_size
            )
            + qkv_scales
        )
        qkv_tile = min(rows, 16_384) * (
            3 * self.heads * self.head_dim * element_size
            + self.hidden_size
            + 4
        )
        qkv = qkv_persistent + qkv_tile

        mlp = rows * self.hidden_size * element_size + min(rows, 16_384) * (
            2 * self.expanded_size * element_size
            + self.expanded_size
            + self.hidden_size * element_size
        )
        return max(attention, qkv, mlp)


def _activation_profile(diffusion_model) -> _MiniMaxActivationProfile | None:
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return None
    block = blocks[0]
    attention = getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None)
    fc2 = getattr(mlp, "fc2", None)
    try:
        profile = _MiniMaxActivationProfile(
            hidden_size=int(getattr(diffusion_model, "hidden_size")),
            heads=int(getattr(attention, "heads")),
            head_dim=int(getattr(attention, "head_dim")),
            expanded_size=int(getattr(fc2, "in_features")),
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if min(dataclasses.astuple(profile)) <= 0:
        return None
    return profile


def _minimax_memory_shape(kwargs, latent_shapes, diffusion_model):
    """Return an H3 memory proxy whose row count matches PackedLayout."""
    if not isinstance(latent_shapes, (list, tuple)) or len(latent_shapes) < 2:
        return None
    video_shape = tuple(int(value) for value in latent_shapes[0])
    audio_shape = tuple(int(value) for value in latent_shapes[1])
    if len(video_shape) != 5 or len(audio_shape) != 4:
        return None
    if video_shape[0] != 1 or audio_shape[0] != 1 or audio_shape[2] != 2:
        return None

    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    # PackedLayout and H3's modality rows currently define this exact patch
    # contract. If upstream changes it, do not estimate with different rules.
    if patch_size != (1, 2, 2):
        return None
    pt, ph, pw = patch_size
    latent_t = math.ceil(video_shape[2] / pt)
    latent_h = math.ceil(video_shape[3] / ph) * ph
    latent_w = math.ceil(video_shape[4] / pw) * pw
    frame_rows = (latent_h // ph) * (latent_w // pw)
    target_visual_rows = latent_t * frame_rows
    target_audio_rows = int(audio_shape[2]) * int(audio_shape[3])
    target_rows = target_visual_rows + target_audio_rows
    if target_rows <= 0:
        return None

    cross_attn = kwargs.get("cross_attn")
    text_rows = (
        int(cross_attn.shape[1])
        if torch.is_tensor(cross_attn) and cross_attn.ndim >= 2
        else 0
    )

    keyframes = kwargs.get("minimax_keyframes") or ()
    visual_condition_rows = len(keyframes) * frame_rows
    audio_condition_rows = 0
    for ref in kwargs.get("minimax_refs") or ():
        kind = ref.get("kind")
        if kind == "image":
            visual_condition_rows += (
                int(ref["latent_h"]) // ph
            ) * (int(ref["latent_w"]) // pw)
        elif kind == "audio":
            audio_condition_rows += int(ref.get("ref_audio_t", 0)) * 2
        elif kind in ("video", "video_audio"):
            visual_condition_rows += (
                int(ref["latent_t"])
                * (int(ref["latent_h"]) // ph)
                * (int(ref["latent_w"]) // pw)
            )
            audio_condition_rows += int(ref.get("ref_audio_t", 0)) * 2

    condition_rows = text_rows + visual_condition_rows + audio_condition_rows
    full_rows = target_rows + condition_rows
    target_area = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    equivalent_area = math.ceil(target_area * condition_rows / target_rows)
    return _MiniMaxMemoryShape(
        equivalent_area,
        full_rows=full_rows,
        target_rows=target_rows,
        target_visual_rows=target_visual_rows,
        target_audio_rows=target_audio_rows,
        visual_condition_rows=visual_condition_rows,
        audio_condition_rows=audio_condition_rows,
        hidden_size=int(diffusion_model.hidden_size),
        video_row_width=int(diffusion_model.latents_dim) * math.prod(patch_size),
        audio_row_width=int(diffusion_model.audio_latents_dim),
    )


def _make_extra_conds_shapes(base_model, diffusion_model):
    original = OriginalMethod.capture(base_model.extra_conds_shapes, base_model)

    def extra_conds_shapes(self, **kwargs):
        out = dict(original(self, **kwargs))
        context = getattr(self, _MEMORY_CONTEXT_ATTR, None)
        latent_shapes = kwargs.get("latent_shapes")
        if latent_shapes is None and isinstance(context, dict):
            latent_shapes = context.get("latent_shapes")
        shape = _minimax_memory_shape(kwargs, latent_shapes, diffusion_model)
        if shape is not None:
            out[_MEMORY_SHAPE_KEY] = shape
        return out

    return weak_method(extra_conds_shapes, base_model)


def _make_extra_conds(base_model, diffusion_model):
    original = OriginalMethod.capture(base_model.extra_conds, base_model)

    def extra_conds(self, **kwargs):
        out = original(self, **kwargs)
        latent_shapes = kwargs.get("latent_shapes")
        plan = _minimax_memory_shape(kwargs, latent_shapes, diffusion_model)
        if plan is not None:
            out[_MEMORY_SHAPE_KEY] = _MiniMaxMemoryCond(plan)
        return out

    return weak_method(extra_conds, base_model)


def _dtype_size(dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (TypeError, RuntimeError):
        return 2


def _make_memory_required(
    base_model,
    w8_output_channels: tuple[int, ...],
    fixed_workspaces: tuple[int, ...],
    activation_profile: _MiniMaxActivationProfile | None = None,
):
    original = OriginalMethod.capture(base_model.memory_required, base_model)

    def memory_required(self, input_shape, cond_shapes={}):
        required = original(self, input_shape, cond_shapes=cond_shapes)
        plans = [
            shape
            for shape in cond_shapes.get(_MEMORY_SHAPE_KEY, ())
            if isinstance(shape, _MiniMaxMemoryShape)
        ]
        if not plans:
            return required

        dtype_size = _dtype_size(self.get_dtype_inference())
        heuristic_extra = sum(
            original(self, [1, 1, plan.equivalent_area], cond_shapes={})
            for plan in plans
        )
        explicit_extra = sum(
            plan.explicit_condition_bytes(dtype_size) for plan in plans
        )
        required += max(explicit_extra - heuristic_extra, 0)

        rows = max(plan.full_rows for plan in plans)
        transient_workspaces = [
            turing_int8_workspace_bytes(rows, output_channels)
            for output_channels in w8_output_channels
        ]
        transient_workspaces.extend(fixed_workspaces)
        if activation_profile is not None:
            transient_workspaces.append(
                activation_profile.conservative_floor_bytes(rows, dtype_size)
            )
        if transient_workspaces:
            # Linear layers execute serially, so only the largest transient
            # workspace is live at once. Summing them would over-reserve VRAM.
            required += max(transient_workspaces)
        return required

    return weak_method(memory_required, base_model)


_make_outer_sample_wrapper = make_minimax_runtime_context_wrapper


def _linear_workspace_requirements(
    root: torch.nn.Module,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    profile = scan_quantized_workspaces(root, convrot_weight_kind)
    return profile.w8_output_channels, profile.fixed_workspaces


def _install_memory_planning(model, base_model, diffusion_model) -> bool:
    outputs, fixed_workspaces = _linear_workspace_requirements(base_model)
    activation_profile = _activation_profile(diffusion_model)
    installed = install_memory_hooks(
        base_model,
        marker=_MEMORY_ADAPTER_ATTR,
        condition_key=_MEMORY_SHAPE_KEY,
        extra_conds=_make_extra_conds(base_model, diffusion_model),
        extra_conds_shapes=_make_extra_conds_shapes(base_model, diffusion_model),
        memory_required=_make_memory_required(
            base_model,
            outputs,
            fixed_workspaces,
            activation_profile,
        ),
        required_methods=("get_dtype_inference",),
    )
    if not installed:
        return False
    LOG.info(
        "Enabled MiniMax packed-sequence VRAM planning: W8 outputs=[%s] "
        "fixed_workspaces=[%s MiB] activation_profile=%s",
        ",".join(map(str, outputs)) or "none",
        ",".join(f"{value / 1024**2:.1f}" for value in fixed_workspaces) or "none",
        (
            f"H{activation_profile.hidden_size}/A{activation_profile.heads}x"
            f"{activation_profile.head_dim}/F{activation_profile.expanded_size}"
            if activation_profile is not None
            else "none"
        ),
    )
    return True
__all__ = [
    "_MEMORY_ADAPTER_ATTR",
    "_MEMORY_CONTEXT_ATTR",
    "_MEMORY_SHAPE_KEY",
    "_MiniMaxActivationProfile",
    "_MiniMaxMemoryCond",
    "_MiniMaxMemoryShape",
    "_activation_profile",
    "_install_memory_planning",
    "_linear_workspace_requirements",
    "_make_extra_conds",
    "_make_extra_conds_shapes",
    "_make_memory_required",
    "_make_outer_sample_wrapper",
    "_minimax_memory_shape",
]
