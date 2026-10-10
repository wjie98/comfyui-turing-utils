"""H3 block fusion installation and model adapter integration."""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Sequence
import torch
from ..methods import OriginalMethod, weak_method
from ...attention.layout import ATTENTION_LAYOUT_REQUIREMENT_KEY
from ...hardware import is_supported_attention_device
from ...kernel_api import load_kernel_package, segmented_modulation_schema
from .layout import (
    ATTENTION_LAYOUT_KEY,
    RUNTIME_OUTER_WRAPPER_KEY,
    ensure_minimax_attention_layout_provider,
    mark_forward_as_minimax_layout_provider,
    minimax_temporal_topology,
    publish_minimax_attention_layout,
)
from ...quantization.fusions import (
    convrot_weight_kind,
    is_turing_convrot_linear,
    indexed_modulation_rows,
    segmented_mod_gate,
    segmented_mod_gate_rms_adaln,
    segmented_rms_adaln,
)
from .memory_planning import _install_memory_planning
from .execution import (
    LOG,
    _SUPPORTED_DTYPES,
    _RuntimeDispatchAudit,
    _format_counts,
)
from .attention_ops import (
    install_minimax_attention_sites,
)
from .mlp import (
    _make_mlp_forward,
)


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


_OUTER_SAMPLE_WRAPPER_KEY = RUNTIME_OUTER_WRAPPER_KEY


_ATTENTION_LAYOUT_KEY = ATTENTION_LAYOUT_KEY


def _audit_fc2(blocks: Sequence[torch.nn.Module]) -> int:
    linears = []
    for block in blocks:
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "fc2"):
            linears.append(mlp.fc2)
    if not linears:
        return 0
    kinds = [
        convrot_weight_kind(getattr(linear, "weight", None)) or "other"
        for linear in linears
    ]
    eligible = sum(kind != "other" for kind in kinds)
    LOG.debug(
        "MiniMax fused fc2 dispatch: blocks=%d eligible=%d formats=[%s]",
        len(linears),
        eligible,
        _format_counts(kinds),
    )
    return eligible


def _compatible_block_forward(block_type: type[torch.nn.Module]) -> bool:
    parameters = tuple(inspect.signature(block_type.forward).parameters)
    return parameters in {
        ("self", *_BLOCK_FORWARD_PARAMETERS),
        ("self", *_BLOCK_FORWARD_PARAMETERS_WITH_ATTENTION),
    }


def _block_fusion_blocker(
    x: torch.Tensor,
    t_emb: torch.Tensor,
    device_index: int,
) -> str | None:
    if x.device.type != "cuda" or x.dtype not in _SUPPORTED_DTYPES or x.ndim != 2:
        return f"input={x.device.type}/{x.dtype}/ndim{x.ndim}"
    index = (
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    )
    if index != device_index:
        return f"device_index={index},expected={device_index}"
    if torch.is_grad_enabled() and (x.requires_grad or t_emb.requires_grad):
        return "grad_enabled"
    return None


_minimax_temporal_topology = minimax_temporal_topology


def _make_block_forward(
    block: torch.nn.Module,
    device_index: int,
    mod_gate,
    audit: _RuntimeDispatchAudit,
    layer_index: int = 0,
    layer_count: int = 0,
    base_model=None,
    diffusion_model=None,
):
    original = OriginalMethod.capture(block.forward, block)
    supports_attention = "attention" in inspect.signature(original.function).parameters
    supports_indexed_modulation = segmented_modulation_schema() >= 2
    if base_model is not None:
        base_model = weakref.proxy(base_model)
    if diffusion_model is not None:
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
        blocker = _block_fusion_blocker(x, t_emb, device_index)
        indexed = any(isinstance(row, torch.Tensor) for _, _, row in mod_segments)
        if blocker is None and indexed and not supports_indexed_modulation:
            blocker = "per_token_modulation"
        audit.record("block", blocker is None, x, blocker)
        if blocker is not None:
            kwargs = {"transformer_options": transformer_options}
            if supports_attention:
                kwargs["attention"] = attention
            return original(self, x, t_emb, mod_segments, rope_freqs, **kwargs)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(t_emb)
        )
        if indexed:
            mod_segments = indexed_modulation_rows(
                mod_segments, x.shape[0], scale_msa.shape[0], x.device
            )
        h = segmented_rms_adaln(self.norm1, x, shift_msa, scale_msa, mod_segments)
        attention_impl = self.attn if attention is None else attention
        x, h = segmented_mod_gate_rms_adaln(
            self.norm2,
            x,
            gate_msa,
            attention_impl(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            ),
            shift_mlp,
            scale_mlp,
            mod_segments,
        )
        return segmented_mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    return mark_forward_as_minimax_layout_provider(weak_method(forward, block))


def apply_minimax_adapter(model, device: torch.device) -> int:
    """Install MiniMax-only forward substitutions through the ModelPatcher."""
    if not is_supported_attention_device(device):
        return 0
    if not hasattr(model, "add_object_patch"):
        raise RuntimeError("MiniMax CUDA integration requires a ComfyUI ModelPatcher")

    try:
        from comfy.ldm.minimax.model import DiTBlock, _mod_gate
    except ImportError:
        return 0
    if not _compatible_block_forward(DiTBlock):
        LOG.warning(
            "MiniMax CUDA fusions are disabled because the DiTBlock forward contract changed"
        )
        return 0

    root = getattr(model, "model", model)
    candidates = [
        (name, block)
        for name, block in root.named_modules()
        if name and isinstance(block, DiTBlock)
    ]
    if not candidates:
        return 0
    try:
        setattr(root, "_turing_utils_minimax_layer_count", len(candidates))
    except (AttributeError, TypeError):
        pass

    layout_status = ensure_minimax_attention_layout_provider(model)
    if layout_status.installed:
        transformer_options = model.model_options.setdefault("transformer_options", {})
        transformer_options[ATTENTION_LAYOUT_REQUIREMENT_KEY] = layout_status.model_kind
    if layout_status.required and not layout_status.installed:
        LOG.warning(
            "MiniMax H3 attention layout provider could not be installed: %s",
            layout_status.reason,
        )

    diffusion_model = getattr(root, "diffusion_model", None)
    if diffusion_model is not None:
        _install_memory_planning(model, root, diffusion_model)

    eligible_fc2 = _audit_fc2([block for _, block in candidates])
    index = device.index if device.index is not None else torch.cuda.current_device()
    block_fusions = 0
    mlp_fusions = 0
    attention_fusions = install_minimax_attention_sites(model, device).installed
    try:
        kernel_package = load_kernel_package()
        segmented_block_ops = (
            getattr(kernel_package, "turing_segmented_rms_adaln"),
            getattr(kernel_package, "turing_segmented_mod_gate"),
            getattr(kernel_package, "turing_segmented_mod_gate_rms_adaln"),
        )
    except (ImportError, OSError, AttributeError):
        segmented_block_ops = ()

    block_ops_available = bool(segmented_block_ops) and all(
        callable(op) for op in segmented_block_ops
    )
    expected_blocks = len(candidates) if block_ops_available else 0
    audit = _RuntimeDispatchAudit(expected_blocks, eligible_fc2)

    for layer_index, (name, block) in enumerate(candidates):
        if hasattr(block.mlp, "fc2") and is_turing_convrot_linear(block.mlp.fc2):
            model.add_object_patch(
                f"{name}.mlp.forward",
                _make_mlp_forward(block.mlp, audit, root),
            )
            mlp_fusions += 1
        if block_ops_available:
            model.add_object_patch(
                f"{name}.forward",
                _make_block_forward(
                    block,
                    index,
                    _mod_gate,
                    audit,
                    layer_index,
                    len(candidates),
                    root,
                    diffusion_model,
                ),
            )
            block_fusions += 1

    if block_fusions:
        LOG.debug(
            "Enabled MiniMax segmented RMSNorm+AdaLN on %d CUDA blocks", block_fusions
        )
    if mlp_fusions:
        LOG.debug(
            "Enabled MiniMax fused/streamed ConvRot SwiGLU on %d MLP layers",
            mlp_fusions,
        )
    if attention_fusions:
        LOG.debug(
            "Enabled MiniMax fused Q/K RMSNorm+RoPE+INT8 preprocessing on %d attention layers",
            attention_fusions,
        )
    if eligible_fc2 and mlp_fusions != eligible_fc2:
        raise RuntimeError(
            "MiniMax fused fc2 adapter did not patch every eligible layer"
        )
    return max(block_fusions, mlp_fusions, attention_fusions)
