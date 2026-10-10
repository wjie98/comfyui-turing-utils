"""Versioned contracts between model adapters and attention backends.

The protocol intentionally describes tensors and mathematical transforms, not
model classes.  Model adapters remain responsible for projections and output
epilogues; attention backends own preflight, tensor consumption, quantization,
and attention execution.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch


ATTENTION_PROTOCOL_VERSION = 2
ATTENTION_EXECUTOR_KEY = "turing_utils_attention_executor_v1"
MAPPED_KV_EXECUTOR_ATTR = "turing_utils_mapped_kv_executor"
MAPPED_RESIDUAL_EXECUTOR_ATTR = "turing_utils_mapped_residual_executor"
MAPPED_RESIDUAL_CAPABILITY_ATTR = "turing_utils_mapped_residual_capability"


@runtime_checkable
class AttentionTensorOwner(Protocol):
    """Single-owner tensor accepted by ComfyUI optimized attention."""

    def peek(self) -> torch.Tensor: ...

    def take(self) -> torch.Tensor: ...


@dataclasses.dataclass(frozen=True, slots=True)
class RMSNormSpec:
    weight: torch.Tensor
    epsilon: float
    scope: str

    def validate(self) -> str | None:
        if not torch.is_tensor(self.weight):
            return "RMSNorm weight is not a tensor"
        if self.scope not in {"head", "row"}:
            return f"RMSNorm scope {self.scope!r} is unsupported"
        if self.epsilon <= 0.0:
            return "RMSNorm epsilon must be positive"
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class RotaryEmbeddingSpec:
    freqs: torch.Tensor | None
    rot_dim: int
    pairing: str
    key_freqs: torch.Tensor | None = None

    def validate(self, head_dim: int) -> str | None:
        if self.pairing not in {"none", "split_half", "interleaved"}:
            return f"RoPE pairing {self.pairing!r} is unsupported"
        if self.pairing == "none":
            if (
                self.freqs is not None
                or self.key_freqs is not None
                or self.rot_dim != 0
            ):
                return "RoPE pairing 'none' requires query/key freqs=None and rot_dim=0"
            return None
        if not torch.is_tensor(self.freqs):
            return "query RoPE frequencies are unavailable"
        if self.key_freqs is not None and not torch.is_tensor(self.key_freqs):
            return "key RoPE frequencies are invalid"
        if self.rot_dim <= 0 or self.rot_dim > int(head_dim) or self.rot_dim % 2:
            return (
                f"RoPE rot_dim={self.rot_dim} is incompatible with head_dim={head_dim}"
            )
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class QKTransformSpec:
    """Q/K transforms required before the attention score contraction."""

    query_norm: RMSNormSpec
    key_norm: RMSNormSpec
    rotary: RotaryEmbeddingSpec

    def validate(self, head_dim: int) -> str | None:
        query_reason = self.query_norm.validate()
        if query_reason is not None:
            return f"query {query_reason}"
        key_reason = self.key_norm.validate()
        if key_reason is not None:
            return f"key {key_reason}"
        if self.query_norm.scope != self.key_norm.scope:
            return "query and key RMSNorm scopes differ"
        if self.query_norm.epsilon != self.key_norm.epsilon:
            return "query and key RMSNorm epsilons differ"
        return self.rotary.validate(head_dim)

    # Flat compatibility properties keep the CUDA binding independent from the
    # richer protocol representation.
    @property
    def query_norm_weight(self) -> torch.Tensor:
        return self.query_norm.weight

    @property
    def key_norm_weight(self) -> torch.Tensor:
        return self.key_norm.weight

    @property
    def freqs(self) -> torch.Tensor | None:
        return self.rotary.freqs

    @property
    def key_freqs(self) -> torch.Tensor | None:
        return (
            self.rotary.key_freqs
            if self.rotary.key_freqs is not None
            else self.rotary.freqs
        )

    @property
    def epsilon(self) -> float:
        return self.query_norm.epsilon

    @property
    def rot_dim(self) -> int:
        return self.rotary.rot_dim

    @property
    def norm_scope(self) -> str:
        return self.query_norm.scope

    @property
    def split_half(self) -> bool:
        return self.rotary.pairing == "split_half"


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionBackendCapabilities:
    """Feature set accepted by a prepared-attention executor."""

    protocol_version: int = ATTENTION_PROTOCOL_VERSION
    head_dims: frozenset[int] = frozenset((64, 128))
    tensor_layouts: frozenset[str] = frozenset(("HND",))
    norm_scopes: frozenset[str] = frozenset(("head", "row"))
    rope_pairings: frozenset[str] = frozenset(("none", "split_half", "interleaved"))
    supports_gqa: bool = True
    supports_asymmetric_qk: bool = True
    supports_mask: bool = False
    supports_causal: bool = False
    supports_semantic_sparse: bool = False
    preserves_qk_for_observers: bool = False

    def unsupported_reason(self, request: "PreparedAttention") -> str | None:
        if request.protocol_version != self.protocol_version:
            return (
                f"attention protocol v{request.protocol_version} is unsupported; "
                f"backend requires v{self.protocol_version}"
            )
        if request.tensor_layout not in self.tensor_layouts:
            return f"tensor layout {request.tensor_layout!r} is unsupported"
        if request.head_dim not in self.head_dims:
            return f"head_dim={request.head_dim} is unsupported"
        if request.heads != request.kv_heads and not self.supports_gqa:
            return "GQA is unsupported"
        if (
            request.query_tokens != request.key_tokens
            and not self.supports_asymmetric_qk
        ):
            return "asymmetric Q/K lengths are unsupported"
        if request.mask is not None and not self.supports_mask:
            return "an attention mask was supplied"
        if request.is_causal and not self.supports_causal:
            return "causal attention is unsupported"
        if request.observer_requirements and not self.preserves_qk_for_observers:
            return "the model requires post-transform Q/K observers"
        if request.qk_transform.query_norm.scope not in self.norm_scopes:
            return (
                f"RMSNorm scope {request.qk_transform.query_norm.scope!r} "
                "is unsupported"
            )
        if request.qk_transform.rotary.pairing not in self.rope_pairings:
            return (
                f"RoPE pairing {request.qk_transform.rotary.pairing!r} is unsupported"
            )
        reason = request.qk_transform.validate(request.head_dim)
        if reason is not None:
            return reason
        scope = request.qk_transform.query_norm.scope
        query_norm_elements = request.qk_transform.query_norm.weight.numel()
        key_norm_elements = request.qk_transform.key_norm.weight.numel()
        expected_query = (
            request.head_dim if scope == "head" else request.heads * request.head_dim
        )
        expected_key = (
            request.head_dim if scope == "head" else request.kv_heads * request.head_dim
        )
        if query_norm_elements != expected_query:
            return (
                f"query RMSNorm has {query_norm_elements} elements, "
                f"expected {expected_query}"
            )
        if key_norm_elements != expected_key:
            return (
                f"key RMSNorm has {key_norm_elements} elements, expected {expected_key}"
            )
        return None


@dataclasses.dataclass(slots=True)
class PreparedAttention:
    """A model-independent, transactionally consumed attention request."""

    query: AttentionTensorOwner
    key: AttentionTensorOwner
    value: AttentionTensorOwner
    heads: int
    kv_heads: int
    head_dim: int
    query_tokens: int
    key_tokens: int
    tensor_layout: str
    qk_transform: QKTransformSpec
    transformer_options: dict | None = None
    scale: float | None = None
    mask: torch.Tensor | None = None
    is_causal: bool = False
    low_precision_attention: bool = True
    skip_output_reshape: bool = False
    observer_requirements: frozenset[str] = frozenset()
    protocol_version: int = ATTENTION_PROTOCOL_VERSION

    @classmethod
    def from_hnd(
        cls,
        query: AttentionTensorOwner,
        key: AttentionTensorOwner,
        value: AttentionTensorOwner,
        *,
        heads: int,
        qk_transform: QKTransformSpec,
        transformer_options: dict | None = None,
        scale: float | None = None,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
        low_precision_attention: bool = True,
        skip_output_reshape: bool = False,
        observer_requirements: frozenset[str] = frozenset(),
    ) -> "PreparedAttention":
        q, k, v = query.peek(), key.peek(), value.peek()
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("prepared HND Q/K/V must be four-dimensional")
        if k.shape != v.shape:
            raise ValueError("prepared K/V shapes differ")
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError("prepared Q/K batch or head dimensions differ")
        if int(heads) <= 0 or q.shape[1] != int(heads):
            raise ValueError("prepared Q head count does not match heads")
        if k.shape[1] <= 0 or int(heads) % int(k.shape[1]):
            raise ValueError("prepared Q/K head counts do not form valid GQA")
        if q.shape[2] <= 0 or k.shape[2] <= 0 or q.shape[-1] <= 0:
            raise ValueError("prepared Q/K/V dimensions must be positive")
        return cls(
            query=query,
            key=key,
            value=value,
            heads=int(heads),
            kv_heads=int(k.shape[1]),
            head_dim=int(q.shape[-1]),
            query_tokens=int(q.shape[2]),
            key_tokens=int(k.shape[2]),
            tensor_layout="HND",
            qk_transform=qk_transform,
            transformer_options=transformer_options,
            scale=scale,
            mask=mask,
            is_causal=bool(is_causal),
            low_precision_attention=bool(low_precision_attention),
            skip_output_reshape=bool(skip_output_reshape),
            observer_requirements=frozenset(observer_requirements),
        )

    def peek_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query.peek(), self.key.peek(), self.value.peek()

    def consume_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Preflight every owner first so an already-consumed input cannot cause
        # a partial ownership transfer.
        self.peek_qkv()
        return self.query.take(), self.key.take(), self.value.take()

    @property
    def expected_output_shape(self) -> tuple[int, ...]:
        query = self.query.peek()
        if self.skip_output_reshape:
            return (
                int(query.shape[0]),
                self.heads,
                self.query_tokens,
                self.head_dim,
            )
        return (
            int(query.shape[0]),
            self.query_tokens,
            self.heads * self.head_dim,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionExecutionOutcome:
    output: torch.Tensor | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.output is None) == (self.reason is None):
            raise ValueError(
                "attention outcome must contain exactly one of output or reason"
            )

    @property
    def supported(self) -> bool:
        return self.reason is None and self.output is not None

    @classmethod
    def unsupported(cls, reason: str) -> "AttentionExecutionOutcome":
        return cls(None, str(reason))


PreparedAttentionExecutor = Callable[[PreparedAttention], AttentionExecutionOutcome]


def prepared_attention_executor(
    transformer_options,
) -> PreparedAttentionExecutor | None:
    if not isinstance(transformer_options, dict):
        return None
    executor = transformer_options.get(ATTENTION_EXECUTOR_KEY)
    return executor if callable(executor) else None


def execute_prepared_attention(
    request: PreparedAttention,
) -> AttentionExecutionOutcome:
    executor = prepared_attention_executor(request.transformer_options)
    if executor is None:
        return AttentionExecutionOutcome.unsupported(
            "no prepared-attention executor is installed"
        )
    expected_shape = request.expected_output_shape
    outcome = executor(request)
    if not isinstance(outcome, AttentionExecutionOutcome):
        raise TypeError("prepared-attention executor returned an invalid outcome")
    if not outcome.supported:
        try:
            request.peek_qkv()
        except RuntimeError as error:
            raise RuntimeError(
                "prepared-attention backend rejected a request after consuming its inputs"
            ) from error
        return outcome
    if not torch.is_tensor(outcome.output):
        raise TypeError("prepared-attention executor output is not a tensor")
    if tuple(outcome.output.shape) != expected_shape:
        raise RuntimeError(
            "prepared-attention executor returned shape "
            f"{tuple(outcome.output.shape)}, expected {expected_shape}"
        )
    return outcome


__all__ = [
    "ATTENTION_EXECUTOR_KEY",
    "ATTENTION_PROTOCOL_VERSION",
    "MAPPED_KV_EXECUTOR_ATTR",
    "MAPPED_RESIDUAL_CAPABILITY_ATTR",
    "MAPPED_RESIDUAL_EXECUTOR_ATTR",
    "AttentionBackendCapabilities",
    "AttentionExecutionOutcome",
    "AttentionTensorOwner",
    "PreparedAttention",
    "PreparedAttentionExecutor",
    "QKTransformSpec",
    "RMSNormSpec",
    "RotaryEmbeddingSpec",
    "execute_prepared_attention",
    "prepared_attention_executor",
]
