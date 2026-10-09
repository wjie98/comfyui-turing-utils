"""Shared installation mechanics for attention strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .integration import AttentionSiteStatus, ensure_prepared_attention_sites
from .layout import (
    ATTENTION_LAYOUT_REQUIREMENT_KEY,
    LayoutProviderStatus,
    ensure_attention_layout_provider,
)
from .protocol import ATTENTION_EXECUTOR_KEY
from .runtime import AttentionRuntimeConfig, install_attention_runtime
from .stable import LOG


@dataclass(frozen=True, slots=True)
class AttentionStrategyInstallation:
    model: object
    layout: LayoutProviderStatus
    attention_sites: AttentionSiteStatus | None


def install_attention_strategy(
    model,
    override,
    *,
    strategy: str,
    backend: str,
    implementation: str,
    runtime_config: AttentionRuntimeConfig | None = None,
) -> AttentionStrategyInstallation:
    """Clone and install one strategy without owning its algorithm."""
    patched = model.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    existing_requirement = transformer_options.get(ATTENTION_LAYOUT_REQUIREMENT_KEY)
    if (
        runtime_config is not None
        and runtime_config.native_runtime
        and isinstance(existing_requirement, str)
        and existing_requirement
    ):
        # The ConvRot loader already installed the provider and the prepared
        # attention sites.  A ModelPatcher clone carries those keyed patches;
        # reinstalling them here can stack sample wrappers in some Comfy builds.
        layout_status = LayoutProviderStatus(existing_requirement, True)
    else:
        layout_status = ensure_attention_layout_provider(patched)
    if layout_status.required:
        transformer_options[ATTENTION_LAYOUT_REQUIREMENT_KEY] = layout_status.model_kind
        if not layout_status.installed:
            LOG.warning_once(
                "%s %s attention will stay dense because its runtime layout "
                "provider could not be installed: %s",
                layout_status.model_kind,
                strategy,
                layout_status.reason,
            )

    if runtime_config is not None:
        runtime_config = runtime_config.with_strategy(
            backend,
            implementation,
            override,
        )
        install_attention_runtime(transformer_options, runtime_config)
    else:
        # Compatibility for callers using the old orchestration API directly.
        transformer_options["optimized_attention_override"] = override
    prepared_executor = getattr(override, "prepared_attention_executor", None)
    site_status = None
    if callable(prepared_executor) and not (
        runtime_config is not None and runtime_config.native_runtime
    ):
        if runtime_config is None:
            transformer_options[ATTENTION_EXECUTOR_KEY] = prepared_executor
        site_status = ensure_prepared_attention_sites(patched, patched.load_device)
        if site_status.matched and site_status.reason is not None:
            LOG.debug(
                "%s prepared-attention fusion was not installed: %s",
                site_status.model_kind,
                site_status.reason,
            )
    else:
        if runtime_config is None:
            transformer_options.pop(ATTENTION_EXECUTOR_KEY, None)
    if runtime_config is None:
        transformer_options["turing_utils_attention_backend"] = backend
        transformer_options["turing_utils_attention_implementation"] = implementation
    return AttentionStrategyInstallation(patched, layout_status, site_status)


# Compatibility aliases for integrations written before orchestration became
# useful to non-sparse strategies such as virtual K/V expansion.
SparsePatchInstallation = AttentionStrategyInstallation
install_sparse_attention_override = install_attention_strategy


__all__ = [
    "AttentionStrategyInstallation",
    "SparsePatchInstallation",
    "install_attention_strategy",
    "install_sparse_attention_override",
]
