"""Shared control-plane mechanics for semantic sparse attention strategies."""

from __future__ import annotations

from collections.abc import Callable

from .protocol import AttentionExecutionOutcome, PreparedAttention
from .sparse import _sparse_dense_layer, _sparse_dense_schedule


class SparseSchedule:
    """Own sampler-local dense-step state for one installed strategy."""

    __slots__ = (
        "_key",
        "_standalone",
        "dense_prefix_steps",
        "dense_suffix_steps",
        "dense_prefix_layers",
        "dense_suffix_layers",
    )

    def __init__(
        self,
        *,
        dense_prefix_steps: int,
        dense_suffix_steps: int,
        dense_prefix_layers: int,
        dense_suffix_layers: int,
    ) -> None:
        self._key = object()
        self._standalone: dict[str, object] = {}
        self.dense_prefix_steps = int(dense_prefix_steps)
        self.dense_suffix_steps = int(dense_suffix_steps)
        self.dense_prefix_layers = int(dense_prefix_layers)
        self.dense_suffix_layers = int(dense_suffix_layers)

    def state_for(self, transformer_options) -> dict[str, object]:
        if not isinstance(transformer_options, dict):
            return self._standalone
        state = transformer_options.setdefault(self._key, {})
        if isinstance(state, dict):
            return state
        state = {}
        transformer_options[self._key] = state
        return state

    def dense_step(self, transformer_options) -> bool:
        return _sparse_dense_schedule(
            transformer_options,
            self.dense_prefix_steps,
            self.dense_suffix_steps,
            self.state_for(transformer_options),
        )

    def dense_layer(self, transformer_options) -> bool:
        return _sparse_dense_layer(
            transformer_options,
            self.dense_prefix_layers,
            self.dense_suffix_layers,
        )

    def is_dense(self, transformer_options, *, force_dense: bool = False) -> bool:
        return bool(
            force_dense
            or self.dense_step(transformer_options)
            or self.dense_layer(transformer_options)
        )


class DenseAttentionFallback:
    """Prepared/container/streamed views of one immutable dense backend."""

    __slots__ = (
        "override",
        "prepared_executor",
        "container",
        "streamed_qkv_executor",
        "_default_fallback",
    )

    def __init__(
        self,
        override: Callable,
        *,
        default_fallback: Callable,
        prepared_executor: Callable | None = None,
        container: Callable | None = None,
    ) -> None:
        self.override = override
        self.prepared_executor = (
            prepared_executor
            if prepared_executor is not None
            else getattr(override, "prepared_attention_executor", None)
        )
        self.container = (
            container
            if container is not None
            else getattr(override, "container_function", None)
        )
        self.streamed_qkv_executor = getattr(
            self.prepared_executor,
            "turing_utils_streamed_qkv_executor",
            None,
        )
        self._default_fallback = default_fallback

    def run_prepared(self, request: PreparedAttention) -> AttentionExecutionOutcome:
        if callable(self.prepared_executor):
            return self.prepared_executor(request)
        return AttentionExecutionOutcome.unsupported(
            "the selected dense backend does not expose prepared attention"
        )

    def run_container(
        self,
        q,
        k,
        v,
        heads,
        *,
        mask,
        attn_precision,
        skip_reshape,
        skip_output_reshape,
        **kwargs,
    ):
        if callable(self.container):
            return self.container(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        return self.override(
            self._default_fallback(),
            q.take(),
            k.take(),
            v.take(),
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )


__all__ = ["DenseAttentionFallback", "SparseSchedule"]
