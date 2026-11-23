"""Startup for standalone model adapters and the ComfyUI custom node plugin."""

from __future__ import annotations

import threading


_BOOTSTRAP_LOCK = threading.Lock()
_ADAPTERS_REGISTERED = False


def bootstrap_builtin_integrations() -> None:
    """Register the built-in adapters exactly once for this interpreter."""
    global _ADAPTERS_REGISTERED
    if _ADAPTERS_REGISTERED:
        return
    with _BOOTSTRAP_LOCK:
        if _ADAPTERS_REGISTERED:
            return

        from .adapters.minimax.acceleration import apply_minimax_adapter
        from .adapters.minimax.acceleration import install_minimax_attention_sites
        from .adapters.minimax.layout import ensure_minimax_attention_layout_provider
        from .adapters.registry import ModelAdapter, register_model_adapter
        from .adapters.qwen_image21 import apply_qwen_image21_adapter
        from .adapters.wan import apply_wan_adapter, install_wan_attention_sites
        from .adapters.wan_layout import ensure_wan_attention_layout_provider
        from .attention.integration import register_attention_site_installer
        from .attention.layout import register_attention_layout_provider

        register_attention_layout_provider(ensure_minimax_attention_layout_provider)
        register_attention_layout_provider(ensure_wan_attention_layout_provider)
        register_attention_site_installer(install_minimax_attention_sites)
        register_attention_site_installer(install_wan_attention_sites)
        register_model_adapter(ModelAdapter("minimax_h3", apply_minimax_adapter))
        register_model_adapter(ModelAdapter("wan", apply_wan_adapter))
        register_model_adapter(ModelAdapter("qwen_image21", apply_qwen_image21_adapter))
        _ADAPTERS_REGISTERED = True


def initialize_plugin() -> None:
    """Install ComfyUI integrations for this process and its current server."""
    from server import PromptServer

    from .runtime.stage_barrier import install_stage_barrier_scheduler

    bootstrap_builtin_integrations()
    with _BOOTSTRAP_LOCK:
        install_stage_barrier_scheduler()

        server = getattr(PromptServer, "instance", None)
        if server is None:
            return
        from .job_api.routes import install_routes
        from .runtime.workflow_prompt import install_workflow_prompt_compiler

        install_workflow_prompt_compiler(server)
        install_routes(server)
