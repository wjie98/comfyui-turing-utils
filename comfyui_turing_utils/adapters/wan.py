"""Wan/Bernini memory planning for generic Turing kernels."""

from __future__ import annotations

import inspect
import math
from collections import Counter

import torch

from ..log import get_logger
from .methods import OriginalMethod, weak_method
from .memory import install_memory_hooks, scan_quantized_workspaces
from ..attention.integration import AttentionSiteStatus, execute_projected_attention
from ..attention.protocol import (
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
    prepared_attention_executor,
)
from ..attention.stable import fused_qk_preprocessing_available
from ..hardware import is_supported_attention_device
from ..profiling import CUDA_PHASE_PROFILER
from ..quantization.dispatch import (
    turing_int8_workspace_bytes,
)


LOG = get_logger("wan")
_CONTEXT_SHAPE_KEY = "context_latents"
_MEMORY_CONTEXT_ATTR = "_turing_utils_wan_memory_context"
_OUTER_SAMPLE_WRAPPER_KEY = "turing_utils_wan_memory_context"
_W4_LAYOUT = "TensorCoreConvRotW4A4Layout"
_W8_LAYOUT = "TensorWiseINT8Layout"
_CODEBOOK_W4_LAYOUT = "AsymW4A8Int8Layout"
_SELF_ATTENTION_FORWARD_PARAMETERS = (
    "x",
    "freqs",
    "transformer_options",
)
_PREPARED_ATTENTION_FORWARD_ATTR = "_turing_utils_wan_prepared_attention"


def _compatible_self_attention_forward(attention_type) -> bool:
    parameters = tuple(inspect.signature(attention_type.forward).parameters)
    return parameters == ("self", *_SELF_ATTENTION_FORWARD_PARAMETERS)


def _make_self_attention_forward(attention, attention_container, original=None):
    original = OriginalMethod.capture(
        attention.forward if original is None else original,
        attention,
    )

    def forward(self, x, freqs, transformer_options={}):
        executor = prepared_attention_executor(transformer_options)
        patches = (
            transformer_options.get("patches", {})
            if isinstance(transformer_options, dict)
            else {}
        )
        if (
            executor is None
            or "attn1_patch" in patches
            or not self.qk_norm
            or x.ndim != 3
            or x.shape[1] < 64
            or x.dtype not in (torch.float16, torch.bfloat16)
            or self.head_dim not in (64, 128)
            or not torch.is_tensor(freqs)
            or freqs.ndim != 6
            or (
                torch.is_grad_enabled()
                and (
                    x.requires_grad
                    or any(parameter.requires_grad for parameter in self.parameters())
                )
            )
        ):
            return original(
                self,
                x,
                freqs,
                transformer_options=transformer_options,
            )

        import comfy.model_management

        batch, sequence = x.shape[:2]
        profile_shape = (batch, self.num_heads, sequence, self.head_dim)
        CUDA_PHASE_PROFILER.begin_operation(
            "attention", profile_shape, adapter="wan", path="full"
        )
        query = CUDA_PHASE_PROFILER.call("wan.q_projection", self.q, x)
        key = CUDA_PHASE_PROFILER.call("wan.k_projection", self.k, x)
        value = CUDA_PHASE_PROFILER.call("wan.v_projection", self.v, x)
        query = query.view(batch, sequence, self.num_heads, self.head_dim)
        key = key.view(batch, sequence, self.num_heads, self.head_dim)
        value = value.view(batch, sequence, self.num_heads, self.head_dim)
        query_norm = comfy.model_management.cast_to(
            self.norm_q.weight, device=x.device, dtype=query.dtype
        )
        key_norm = comfy.model_management.cast_to(
            self.norm_k.weight, device=x.device, dtype=key.dtype
        )
        transform = QKTransformSpec(
            query_norm=RMSNormSpec(query_norm, float(self.eps), "row"),
            key_norm=RMSNormSpec(key_norm, float(self.eps), "row"),
            rotary=RotaryEmbeddingSpec(freqs, self.head_dim, "interleaved"),
        )
        outcome = execute_projected_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            heads=self.num_heads,
            qk_transform=transform,
            transformer_options=transformer_options,
            container_factory=attention_container,
        )
        if not outcome.supported:
            del query, key, value
            CUDA_PHASE_PROFILER.cancel_operation()
            return original(
                self,
                x,
                freqs,
                transformer_options=transformer_options,
            )
        output = outcome.output
        output = CUDA_PHASE_PROFILER.call("wan.out_projection", self.o, output)
        CUDA_PHASE_PROFILER.complete_operation("attention", profile_shape)
        return output

    setattr(forward, _PREPARED_ATTENTION_FORWARD_ATTR, True)
    return weak_method(forward, attention)


def _has_prepared_attention_forward(forward) -> bool:
    function = getattr(forward, "__func__", forward)
    return bool(getattr(function, _PREPARED_ATTENTION_FORWARD_ATTR, False))


def install_wan_attention_sites(model, device: torch.device) -> AttentionSiteStatus:
    """Install only the Wan/Bernini handoff to generic attention backends."""
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer
        from comfy.ldm.wan.model import WanModel, WanSelfAttention
    except ImportError:
        return AttentionSiteStatus(None, 0, "wan_unavailable")
    base_model = getattr(model, "model", model)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if not isinstance(diffusion_model, WanModel):
        return AttentionSiteStatus(None, 0, "not_wan")
    if not is_supported_attention_device(device):
        return AttentionSiteStatus("wan", 0, "not_supported_tensor_core")
    if not callable(getattr(model, "add_object_patch", None)):
        return AttentionSiteStatus("wan", 0, "model_patcher_api_unavailable")
    if not fused_qk_preprocessing_available():
        return AttentionSiteStatus("wan", 0, "fused_qk_unavailable")
    if not _compatible_self_attention_forward(WanSelfAttention):
        return AttentionSiteStatus("wan", 0, "attention_contract_changed")

    object_patches = getattr(model, "object_patches", {})
    installed = 0
    for name, module in base_model.named_modules():
        if not name or type(module) is not WanSelfAttention:
            continue
        key = f"{name}.forward"
        current = object_patches.get(key, module.forward)
        if _has_prepared_attention_forward(current):
            continue
        model.add_object_patch(
            key,
            _make_self_attention_forward(
                module,
                AttentionTensorContainer,
                current,
            ),
        )
        installed += 1
    return AttentionSiteStatus("wan", installed, None)


def _context_latents_from_kwargs(kwargs):
    context_latents = kwargs.get(_CONTEXT_SHAPE_KEY)
    if context_latents is not None:
        return context_latents
    model_conds = kwargs.get("model_conds")
    if isinstance(model_conds, dict):
        cond = model_conds.get(_CONTEXT_SHAPE_KEY)
        return getattr(cond, "cond", None)
    return None


def _context_latents_shape(
    context_latents,
    patch_size=(1, 2, 2),
    estimate_batch_size: int | None = None,
) -> list[int] | None:
    if not isinstance(context_latents, (list, tuple)) or not context_latents:
        return None
    tensors = [latent for latent in context_latents if torch.is_tensor(latent) and latent.ndim >= 3]
    if not tensors:
        return None
    channels = int(tensors[0].shape[1])
    if channels <= 0 or any(int(latent.shape[1]) != channels for latent in tensors):
        return None
    patch_size = tuple(int(value) for value in patch_size)
    patch_volume = math.prod(patch_size)
    padded_per_sample = []
    for latent in tensors:
        spatial = tuple(int(value) for value in latent.shape[2:])
        if len(spatial) == len(patch_size):
            padded = math.prod(
                math.ceil(size / patch) * patch
                for size, patch in zip(spatial, patch_size)
            )
        else:
            padded = math.ceil(math.prod(spatial) / patch_volume) * patch_volume
        padded_per_sample.append(padded)

    if estimate_batch_size is not None:
        batch = max(int(estimate_batch_size), 1)
        return [batch, channels, sum(padded_per_sample)]

    # Without the sampler batch hint, preserve the total volume already present
    # in each tensor. CONDList.size() uses the same flattened representation.
    total = sum(
        max(int(latent.shape[0]), 1) * padded
        for latent, padded in zip(tensors, padded_per_sample)
    )
    return [1, channels, total]


def _shape_token_rows(shape, patch_size) -> int:
    if len(shape) < 3:
        return 0
    batch = int(shape[0])
    spatial = [int(value) for value in shape[2:]]
    if len(spatial) == len(patch_size):
        tokens = math.prod(
            math.ceil(size / patch) for size, patch in zip(spatial, patch_size)
        )
    else:
        tokens = math.ceil(math.prod(spatial) / math.prod(patch_size))
    return batch * tokens


def _make_extra_conds_shapes(base_model, patch_size):
    original = OriginalMethod.capture(base_model.extra_conds_shapes, base_model)

    def extra_conds_shapes(self, **kwargs):
        out = dict(original(self, **kwargs))
        context = getattr(self, _MEMORY_CONTEXT_ATTR, None)
        estimate_batch_size = (
            context.get("batch_size") if isinstance(context, dict) else None
        )
        shape = _context_latents_shape(
            _context_latents_from_kwargs(kwargs),
            patch_size,
            estimate_batch_size=estimate_batch_size,
        )
        if shape is not None:
            out[_CONTEXT_SHAPE_KEY] = shape
        return out

    return weak_method(extra_conds_shapes, base_model)


def _make_extra_conds(base_model, patch_size):
    original = OriginalMethod.capture(base_model.extra_conds, base_model)

    class PaddedContextLatents:
        def __init__(self, cond):
            self.cond = cond

        def _copy_with(self, cond):
            return PaddedContextLatents(cond)

        def process_cond(self, batch_size, **kwargs):
            import comfy.utils

            out = [
                comfy.utils.repeat_to_batch_size(latent, batch_size)
                for latent in self.cond
            ]
            return self._copy_with(out)

        def can_concat(self, other):
            return (
                isinstance(other, PaddedContextLatents)
                and len(self.cond) == len(other.cond)
                and all(
                    left.shape == right.shape
                    for left, right in zip(self.cond, other.cond)
                )
            )

        def concat(self, others):
            out = []
            for index in range(len(self.cond)):
                out.append(
                    torch.cat(
                        [self.cond[index]]
                        + [other.cond[index] for other in others]
                    )
                )
            return out

        def size(self):
            return _context_latents_shape(self.cond, patch_size)

    def extra_conds(self, **kwargs):
        out = original(self, **kwargs)
        context = out.get(_CONTEXT_SHAPE_KEY)
        values = getattr(context, "cond", None)
        if isinstance(values, list):
            out[_CONTEXT_SHAPE_KEY] = PaddedContextLatents(values)
        return out

    return weak_method(extra_conds, base_model)


def _make_memory_required(
    base_model,
    patch_size,
    w8_output_channels: tuple[int, ...],
    fixed_workspaces: tuple[int, ...] = (),
):
    original = OriginalMethod.capture(base_model.memory_required, base_model)

    def memory_required(self, input_shape, cond_shapes={}):
        required = original(self, input_shape, cond_shapes=cond_shapes)
        if not w8_output_channels and not fixed_workspaces:
            return required

        rows = _shape_token_rows(input_shape, patch_size)
        for shape in cond_shapes.get(_CONTEXT_SHAPE_KEY, ()):
            rows += _shape_token_rows(shape, patch_size)
        workspaces = [
            turing_int8_workspace_bytes(rows, output_channels)
            for output_channels in w8_output_channels
        ]
        workspaces.extend(fixed_workspaces)
        workspace = max(workspaces)
        return required + workspace

    return weak_method(memory_required, base_model)


def _make_outer_sample_wrapper(base_model):
    def outer_sample_wrapper(executor, *args, **kwargs):
        noise = args[0] if args else kwargs.get("noise")
        batch_size = int(noise.shape[0]) if torch.is_tensor(noise) else 1
        previous = getattr(base_model, _MEMORY_CONTEXT_ATTR, None)
        setattr(base_model, _MEMORY_CONTEXT_ATTR, {"batch_size": batch_size})
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(base_model, _MEMORY_CONTEXT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(base_model, _MEMORY_CONTEXT_ATTR, previous)

    return outer_sample_wrapper


def _convrot_planning_kind(weight) -> str | None:
    """Classify storage for planning without imposing a compute dtype."""
    params = getattr(weight, "_params", None)
    if (
        getattr(params, "transposed", False)
        or getattr(params, "convrot_groupsize", None) != 256
    ):
        return None
    layout = getattr(weight, "_layout_cls", None)
    if layout == _W8_LAYOUT and bool(getattr(params, "convrot", False)):
        return "w8a8"
    if (
        layout == _CODEBOOK_W4_LAYOUT
        and getattr(params, "codebook", None) is not None
        and getattr(params, "correction", None) is None
    ):
        return "codebook_w4a8"
    if layout != _W4_LAYOUT or getattr(params, "quant_group_size", None) != 64:
        return None
    linear_dtype = getattr(params, "linear_dtype", None)
    return {"int4": "w4a4", "int8": "w4a8"}.get(linear_dtype)


def _quantized_wan_summary(
    diffusion_model,
) -> tuple[Counter, tuple[int, ...], tuple[int, ...]]:
    profile = scan_quantized_workspaces(diffusion_model, _convrot_planning_kind)
    return (
        Counter(dict(profile.formats)),
        profile.w8_output_channels,
        profile.fixed_workspaces,
    )


def apply_wan_adapter(model, device: torch.device) -> int:
    """Install Wan-specific planning without imposing input-size restrictions."""
    if not is_supported_attention_device(device):
        return 0

    try:
        from comfy.ldm.wan.model import WanModel
    except ImportError:
        return 0

    base_model = getattr(model, "model", model)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if not isinstance(diffusion_model, WanModel):
        return 0
    if getattr(base_model, "_turing_utils_wan_adapter", False):
        return 0

    formats, w8_output_channels, fixed_workspaces = _quantized_wan_summary(
        diffusion_model
    )

    patch_size = tuple(int(value) for value in diffusion_model.patch_size)
    installed = install_memory_hooks(
        base_model,
        marker="_turing_utils_wan_adapter",
        condition_key=_CONTEXT_SHAPE_KEY,
        extra_conds=_make_extra_conds(base_model, patch_size),
        extra_conds_shapes=_make_extra_conds_shapes(base_model, patch_size),
        memory_required=_make_memory_required(
            base_model,
            patch_size,
            w8_output_channels,
            fixed_workspaces,
        ),
    )
    if not installed:
        return 0

    attention_fusions = install_wan_attention_sites(model, device).installed

    if hasattr(model, "add_wrapper_with_key"):
        import comfy.patcher_extension

        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _OUTER_SAMPLE_WRAPPER_KEY,
            _make_outer_sample_wrapper(base_model),
        )

    LOG.info(
        "Enabled Wan tensor-core adapter: formats=[%s], context-aware VRAM planning, "
        "w8_outputs=[%s], fixed_workspaces=[%s MiB], fused_qk_attention=%d",
        ",".join(f"{kind}:{count}" for kind, count in sorted(formats.items())),
        ",".join(map(str, w8_output_channels)) or "none",
        ",".join(f"{value / 1024**2:.1f}" for value in fixed_workspaces) or "none",
        attention_fusions,
    )
    return max(sum(formats.values()), attention_fusions, 1)
