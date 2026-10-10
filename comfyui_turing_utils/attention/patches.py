"""Loader-independent attention strategy installation."""

from __future__ import annotations

import torch
from .integration import ensure_prepared_attention_sites
from .orchestration import install_attention_strategy
from .runtime import (
    AttentionRuntimeConfig,
    attention_runtime_config,
    install_attention_runtime,
)
from .stable import (
    LOG,
    SPARSE_DENSE_PREFIX_LAYERS,
    SPARSE_DENSE_PREFIX_STEPS,
    SPARSE_DENSE_SUFFIX_LAYERS,
    SPARSE_DENSE_SUFFIX_STEPS,
    SPARSE_PREFIX_POLICY,
    SPARSE_REFERENCE_AUDIO,
    SPARSE_REFERENCE_IMAGE,
    SPARSE_REFERENCE_VIDEO,
    SPARSE_ROUTING_THRESHOLD,
    SPARSE_SKIPPED_RESIDUAL,
    SLA_DENSE_PREFIX_LAYERS,
    SLA_DENSE_PREFIX_STEPS,
    SLA_DENSE_SUFFIX_LAYERS,
    SLA_DENSE_SUFFIX_STEPS,
    SLA_SPARSITY_RATIO,
    normalize_attention_backend,
)
from .dense import (
    make_attention_override,
)
from .sol import (
    make_sparse_attention_override,
)
from .sla import (
    make_sla_attention_override,
)


def _bootstrap_attention_integrations() -> None:
    # Applying a public patch is an explicit integration action. Keep ordinary
    # package imports light while preserving direct-API behavior for callers
    # that do not enter through ComfyUI's plugin root.
    from ..bootstrap import bootstrap_builtin_integrations

    bootstrap_builtin_integrations()


def attention_base_runtime(model) -> AttentionRuntimeConfig:
    """Resolve the immutable dense base used by a Sol/SLA strategy.

    Models loaded by the ConvRot loader already carry this capability marker.
    Official and third-party loaders are bootstrapped from their current
    override when possible, otherwise from SDPA.
    """
    _bootstrap_attention_integrations()
    transformer_options = model.model_options.setdefault("transformer_options", {})
    config = attention_runtime_config(transformer_options)
    if config is not None:
        return config

    current = transformer_options.get("optimized_attention_override")
    dense_backend = transformer_options.get(
        "turing_utils_attention_base_backend",
        transformer_options.get("turing_utils_attention_backend", "sdpa"),
    )
    if dense_backend not in {"w8a8", "sage", "sdpa"}:
        dense_backend = getattr(current, "turing_utils_attention_backend", "sdpa")
    if dense_backend not in {"w8a8", "sage", "sdpa"}:
        dense_backend = "sdpa"

    if not callable(current):
        current = make_attention_override(dense_backend, device=model.load_device)
    implementation = getattr(
        current,
        "turing_utils_attention_implementation",
        f"inherited:{dense_backend}",
    )
    return AttentionRuntimeConfig(
        dense_backend=dense_backend,
        dense_implementation=implementation,
        dense_override=current,
        native_runtime=False,
    )


def apply_sparse_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    routing_threshold: float = SPARSE_ROUTING_THRESHOLD,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    skipped_residual: str = SPARSE_SKIPPED_RESIDUAL,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SPARSE_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SPARSE_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SPARSE_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SPARSE_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
):
    runtime = attention_base_runtime(model)
    override = make_sparse_attention_override(
        model.load_device,
        min_sequence_tokens=min_sequence_tokens,
        routing_threshold=routing_threshold,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        skipped_residual=skipped_residual,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
        dense_backend=runtime.dense_backend,
        dense_override=runtime.dense_override,
    )
    patched = install_attention_strategy(
        model,
        override,
        strategy="Sol sparse",
        backend="sol",
        implementation="bundled_sol_sparse",
        runtime_config=runtime,
    )
    patched = patched.model
    dense_implementation = getattr(
        override,
        "turing_utils_dense_implementation",
        "selected_dense_backend",
    )
    LOG.info(
        "Attention: Sol · threshold=%.2f · dense=%s",
        routing_threshold,
        runtime.dense_backend,
    )
    LOG.debug(
        "Sol sparse attention patch enabled: threshold=%.2f "
        "prefix_policy=%s manual_prefix=%d local_radius=1 "
        "skipped_residual=%s sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=%s dense_impl=%s pv_backend=%s debug_route_density=%s",
        routing_threshold,
        prefix_policy,
        manual_prefix_tokens,
        skipped_residual,
        sparse_reference_image,
        sparse_reference_video,
        sparse_reference_audio,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        runtime.dense_backend,
        dense_implementation,
        "u8xs8_tensorcore"
        if getattr(override, "turing_utils_sparse_numeric_backend", "fp16") == "w8a8"
        else "fp16_tensorcore",
        debug_route_density,
    )
    return patched


def apply_sla_attention_patch(
    model,
    min_sequence_tokens: int = 0,
    sparsity_ratio: float = SLA_SPARSITY_RATIO,
    prefix_policy: str = SPARSE_PREFIX_POLICY,
    manual_prefix_tokens: int = 0,
    sparse_reference_image: bool = SPARSE_REFERENCE_IMAGE,
    sparse_reference_video: bool = SPARSE_REFERENCE_VIDEO,
    sparse_reference_audio: bool = SPARSE_REFERENCE_AUDIO,
    dense_prefix_steps: int = SLA_DENSE_PREFIX_STEPS,
    dense_suffix_steps: int = SLA_DENSE_SUFFIX_STEPS,
    dense_prefix_layers: int = SLA_DENSE_PREFIX_LAYERS,
    dense_suffix_layers: int = SLA_DENSE_SUFFIX_LAYERS,
    debug_route_density: bool = False,
):
    runtime = attention_base_runtime(model)
    override = make_sla_attention_override(
        model.load_device,
        min_sequence_tokens=min_sequence_tokens,
        sparsity_ratio=sparsity_ratio,
        prefix_policy=prefix_policy,
        manual_prefix_tokens=manual_prefix_tokens,
        sparse_reference_image=sparse_reference_image,
        sparse_reference_video=sparse_reference_video,
        sparse_reference_audio=sparse_reference_audio,
        dense_prefix_steps=dense_prefix_steps,
        dense_suffix_steps=dense_suffix_steps,
        dense_prefix_layers=dense_prefix_layers,
        dense_suffix_layers=dense_suffix_layers,
        debug_route_density=debug_route_density,
        dense_backend=runtime.dense_backend,
        dense_override=runtime.dense_override,
    )
    patched = install_attention_strategy(
        model,
        override,
        strategy="SLA",
        backend="sla",
        implementation="bundled_sla_sparse",
        runtime_config=runtime,
    )
    patched = patched.model
    LOG.info(
        "Attention: SLA · sparsity=%.2f · dense=%s",
        sparsity_ratio,
        runtime.dense_backend,
    )
    LOG.debug(
        "SLA sparse attention patch enabled: sparsity_ratio=%.2f "
        "topology=128x64 smooth_k=True prefix_policy=%s manual_prefix=%d "
        "sparse_reference=(image=%s,video=%s,audio=%s) "
        "dense_prefix_steps=%d dense_suffix_steps=%d "
        "dense_prefix_layers=%d dense_suffix_layers=%d "
        "dense_backend=%s dense_impl=%s pv_backend=%s debug_route_density=%s",
        sparsity_ratio,
        prefix_policy,
        manual_prefix_tokens,
        sparse_reference_image,
        sparse_reference_video,
        sparse_reference_audio,
        dense_prefix_steps,
        dense_suffix_steps,
        dense_prefix_layers,
        dense_suffix_layers,
        runtime.dense_backend,
        override.turing_utils_dense_implementation,
        "u8xs8_tensorcore"
        if getattr(override, "turing_utils_sparse_numeric_backend", "fp16") == "w8a8"
        else "fp16_tensorcore",
        debug_route_density,
    )
    return patched


def apply_attention_backend(
    model,
    option: str,
    device: torch.device | None = None,
    *,
    native_runtime: bool = False,
):
    _bootstrap_attention_integrations()
    option = normalize_attention_backend(option)
    transformer_options = model.model_options.setdefault("transformer_options", {})
    override = make_attention_override(option, device=device)
    selected = override.turing_utils_attention_backend
    implementation = override.turing_utils_attention_implementation
    config = AttentionRuntimeConfig(
        dense_backend=selected,
        dense_implementation=implementation,
        dense_override=override,
        native_runtime=bool(native_runtime),
    )
    install_attention_runtime(transformer_options, config)
    prepared_executor = getattr(override, "prepared_attention_executor", None)
    if callable(prepared_executor):
        target_device = (
            device if device is not None else getattr(model, "load_device", None)
        )
        if isinstance(target_device, torch.device):
            site_status = ensure_prepared_attention_sites(model, target_device)
            if site_status.matched and site_status.reason is not None:
                LOG.debug(
                    "%s prepared-attention fusion was not installed: %s",
                    site_status.model_kind,
                    site_status.reason,
                )
    LOG.debug(
        "Turing Utils attention runtime: dense=%s via %s requested=%s native=%s",
        selected,
        implementation,
        option,
        native_runtime,
    )
    return model
