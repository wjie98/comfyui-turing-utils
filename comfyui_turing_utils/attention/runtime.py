"""Unified dense/strategy attention runtime dispatch.

The model-side attention hand-off is installed once. Subsequent strategy
nodes replace only the immutable runtime configuration carried in
``transformer_options``; the dispatcher and model object patches stay stable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .protocol import (
    ATTENTION_EXECUTOR_KEY,
    AttentionExecutionOutcome,
    PreparedAttention,
)


ATTENTION_RUNTIME_PROTOCOL_VERSION = 2
ATTENTION_RUNTIME_CONFIG_KEY = "turing_utils_attention_runtime_v2"
ATTENTION_RUNTIME_DISPATCHER_ATTR = "turing_utils_attention_runtime_dispatcher"
_STREAMED_QKV_EXECUTOR_ATTR = "turing_utils_streamed_qkv_executor"


@dataclass(frozen=True, slots=True)
class AttentionRuntimeConfig:
    """Immutable strategy selection for one ModelPatcher branch."""

    dense_backend: str
    dense_implementation: str
    dense_override: Callable
    strategy: str = "dense"
    strategy_implementation: str = "dense"
    strategy_override: Callable | None = None
    native_runtime: bool = False
    protocol_version: int = ATTENTION_RUNTIME_PROTOCOL_VERSION

    @property
    def active_override(self) -> Callable:
        return self.strategy_override or self.dense_override

    def with_strategy(
        self,
        strategy: str,
        implementation: str,
        override: Callable | None,
    ) -> "AttentionRuntimeConfig":
        strategy = str(strategy).strip().lower()
        if strategy not in {
            "dense",
            "sol",
            "sla",
            "veda",
        }:
            raise ValueError(f"unsupported attention strategy: {strategy}")
        if strategy == "dense":
            override = None
        return replace(
            self,
            strategy=strategy,
            strategy_implementation=str(implementation),
            strategy_override=override,
        )


def attention_runtime_config(transformer_options) -> AttentionRuntimeConfig | None:
    if not isinstance(transformer_options, dict):
        return None
    config = transformer_options.get(ATTENTION_RUNTIME_CONFIG_KEY)
    if not isinstance(config, AttentionRuntimeConfig):
        return None
    if config.protocol_version != ATTENTION_RUNTIME_PROTOCOL_VERSION:
        return None
    return config


def _active_override(transformer_options, fallback: Callable) -> Callable:
    config = attention_runtime_config(transformer_options)
    return config.active_override if config is not None else fallback


def make_attention_runtime_dispatcher(dense_override: Callable) -> Callable:
    """Create one stable dispatcher for every strategy using a dense base."""

    def attention_override(original: Callable, *args, **kwargs):
        target = _active_override(kwargs.get("transformer_options"), dense_override)
        return target(original, *args, **kwargs)

    def prepared_executor(request: PreparedAttention) -> AttentionExecutionOutcome:
        target = _active_override(request.transformer_options, dense_override)
        executor = getattr(target, "prepared_attention_executor", None)
        if not callable(executor):
            return AttentionExecutionOutcome.unsupported(
                "the active attention strategy has no prepared executor"
            )
        return executor(request)

    def container_function(q, k, v, heads: int, *args, **kwargs):
        target = _active_override(kwargs.get("transformer_options"), dense_override)
        container = getattr(target, "container_function", None)
        if not callable(container):
            # This is only a compatibility path for third-party dense
            # overrides which expose neither prepared nor container execution.
            from .execution import _default_attention_fallback

            q.peek(), k.peek(), v.peek()
            query, key, value = q.take(), k.take(), v.take()
            return target(
                _default_attention_fallback(),
                query,
                key,
                value,
                heads,
                *args,
                **kwargs,
            )
        return container(q, k, v, heads, *args, **kwargs)

    dense_prepared = getattr(dense_override, "prepared_attention_executor", None)
    dense_streamed = getattr(dense_prepared, _STREAMED_QKV_EXECUTOR_ATTR, None)
    if callable(dense_streamed):

        def streamed_qkv_executor(
            qk,
            value,
            *,
            heads: int,
            qk_transform,
            transformer_options,
        ) -> AttentionExecutionOutcome:
            target = _active_override(transformer_options, dense_override)
            executor = getattr(target, "prepared_attention_executor", None)
            streamed = getattr(executor, _STREAMED_QKV_EXECUTOR_ATTR, None)
            if not callable(streamed):
                return AttentionExecutionOutcome.unsupported(
                    "the active attention strategy has no streamed QKV executor"
                )
            return streamed(
                qk,
                value,
                heads=heads,
                qk_transform=qk_transform,
                transformer_options=transformer_options,
            )

        setattr(prepared_executor, _STREAMED_QKV_EXECUTOR_ATTR, streamed_qkv_executor)

    capabilities = getattr(dense_prepared, "capabilities", None)
    if capabilities is not None:
        prepared_executor.capabilities = capabilities
    attention_override.container_function = container_function
    attention_override.prepared_attention_executor = prepared_executor
    attention_override.turing_utils_attention_backend = getattr(
        dense_override, "turing_utils_attention_backend", "external"
    )
    attention_override.turing_utils_attention_implementation = (
        "turing_attention_runtime_v2"
    )
    attention_override.turing_utils_dense_override = dense_override
    setattr(attention_override, ATTENTION_RUNTIME_DISPATCHER_ATTR, True)
    return attention_override


def is_attention_runtime_dispatcher(value) -> bool:
    return bool(getattr(value, ATTENTION_RUNTIME_DISPATCHER_ATTR, False))


def install_attention_runtime(
    transformer_options: dict,
    config: AttentionRuntimeConfig,
    *,
    dispatcher: Callable | None = None,
) -> Callable:
    """Install or update runtime configuration without stacking overrides."""

    if dispatcher is None:
        current = transformer_options.get("optimized_attention_override")
        if (
            is_attention_runtime_dispatcher(current)
            and getattr(current, "turing_utils_dense_override", None)
            is config.dense_override
        ):
            dispatcher = current
        else:
            dispatcher = make_attention_runtime_dispatcher(config.dense_override)
    transformer_options[ATTENTION_RUNTIME_CONFIG_KEY] = config
    transformer_options["optimized_attention_override"] = dispatcher

    active_executor = getattr(
        config.active_override, "prepared_attention_executor", None
    )
    if callable(active_executor):
        # Prepared model sites already execute inside one ModelPatcher branch.
        # Bind that branch's resolved executor directly so head/row streaming
        # cannot silently change strategy if runtime metadata is copied or
        # merged by ComfyUI between sampler stages.  The stable dispatcher is
        # still used by optimized_attention's legacy/container entry point.
        transformer_options[ATTENTION_EXECUTOR_KEY] = active_executor
    else:
        transformer_options.pop(ATTENTION_EXECUTOR_KEY, None)

    transformer_options["turing_utils_attention_base_backend"] = config.dense_backend
    transformer_options["turing_utils_attention_strategy"] = config.strategy
    transformer_options["turing_utils_attention_backend"] = (
        config.dense_backend if config.strategy == "dense" else config.strategy
    )
    transformer_options["turing_utils_attention_implementation"] = (
        config.dense_implementation
        if config.strategy == "dense"
        else config.strategy_implementation
    )
    return dispatcher


__all__ = [
    "ATTENTION_RUNTIME_CONFIG_KEY",
    "ATTENTION_RUNTIME_PROTOCOL_VERSION",
    "AttentionRuntimeConfig",
    "attention_runtime_config",
    "install_attention_runtime",
    "is_attention_runtime_dispatcher",
    "make_attention_runtime_dispatcher",
]
