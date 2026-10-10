from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.attention.dense import _make_dense_prepared_executor  # noqa: E402
from comfyui_turing_utils.attention.execution import (  # noqa: E402
    _prepared_external_call_reason,
    _prepared_qk_transform,
)
from comfyui_turing_utils.attention.layout import (  # noqa: E402
    ATTENTION_LAYOUT_KEY,
    AttentionSegment,
    AttentionSemanticLayout,
    attention_semantic_layout,
    has_complete_attention_layout,
)
from comfyui_turing_utils.attention.protocol import (  # noqa: E402
    ATTENTION_EXECUTOR_KEY,
    AttentionBackendCapabilities,
    AttentionExecutionOutcome,
    PreparedAttention,
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
    execute_prepared_attention,
)


class Owner:
    def __init__(self, tensor):
        self.tensor = tensor
        self.takes = 0

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("already consumed")
        return self.tensor

    def take(self):
        value = self.peek()
        self.tensor = None
        self.takes += 1
        return value


def transform(head_dim=64):
    return QKTransformSpec(
        RMSNormSpec(torch.ones(head_dim), 1e-6, "head"),
        RMSNormSpec(torch.ones(head_dim), 1e-6, "head"),
        RotaryEmbeddingSpec(None, 0, "none"),
    )


class AttentionProtocolTest(unittest.TestCase):
    def request(self, *, query_tokens=64, key_tokens=64, heads=4, kv_heads=4, **kwargs):
        q = Owner(torch.zeros((1, heads, query_tokens, 64), dtype=torch.bfloat16))
        k = Owner(torch.zeros((1, kv_heads, key_tokens, 64), dtype=torch.bfloat16))
        v = Owner(torch.zeros((1, kv_heads, key_tokens, 64), dtype=torch.bfloat16))
        request = PreparedAttention.from_hnd(
            q,
            k,
            v,
            heads=heads,
            qk_transform=transform(),
            **kwargs,
        )
        return request, (q, k, v)

    def test_capability_rejection_never_consumes_inputs(self):
        request, owners = self.request(mask=torch.ones((1, 1)))
        outcome = _make_dense_prepared_executor("sage")(request)
        self.assertFalse(outcome.supported)
        self.assertIn("mask", outcome.reason)
        self.assertTrue(all(owner.tensor is not None for owner in owners))
        self.assertEqual([owner.takes for owner in owners], [0, 0, 0])

    def test_owner_preflight_prevents_partial_transfer(self):
        request, owners = self.request()
        owners[-1].tensor = None
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            request.consume_qkv()
        self.assertEqual([owner.takes for owner in owners], [0, 0, 0])
        self.assertIsNotNone(owners[0].tensor)
        self.assertIsNotNone(owners[1].tensor)

    def test_gqa_and_unequal_query_key_lengths_are_explicit_capabilities(self):
        request, _ = self.request(
            query_tokens=96,
            key_tokens=128,
            heads=8,
            kv_heads=2,
        )
        self.assertIsNone(AttentionBackendCapabilities().unsupported_reason(request))
        reason = AttentionBackendCapabilities(
            supports_gqa=False,
        ).unsupported_reason(request)
        self.assertEqual(reason, "GQA is unsupported")
        reason = AttentionBackendCapabilities(
            supports_asymmetric_qk=False,
        ).unsupported_reason(request)
        self.assertEqual(reason, "asymmetric Q/K lengths are unsupported")

    def test_asymmetric_qk_accepts_independent_rope_tables(self):
        query_tokens = 6
        key_tokens = 9
        identity = torch.eye(2, dtype=torch.float32).reshape(1, 1, 1, 1, 2, 2)
        query_freqs = identity.expand(1, query_tokens, 1, 32, 2, 2).clone()
        key_freqs = identity.expand(1, key_tokens, 1, 32, 2, 2).clone()
        spec = QKTransformSpec(
            RMSNormSpec(torch.ones(64), 1e-6, "head"),
            RMSNormSpec(torch.ones(64), 1e-6, "head"),
            RotaryEmbeddingSpec(
                query_freqs,
                64,
                "split_half",
                key_freqs=key_freqs,
            ),
        )
        query = Owner(torch.randn((1, 2, query_tokens, 64)))
        key = Owner(torch.randn((1, 2, key_tokens, 64)))
        value = Owner(torch.randn((1, 2, key_tokens, 64)))
        request = PreparedAttention.from_hnd(
            query,
            key,
            value,
            heads=2,
            qk_transform=spec,
        )
        self.assertIsNone(_prepared_external_call_reason(request))
        transformed_query, transformed_key = _prepared_qk_transform(
            query.peek(), key.peek(), spec
        )
        self.assertEqual(transformed_query.shape[2], query_tokens)
        self.assertEqual(transformed_key.shape[2], key_tokens)

        invalid_spec = dataclasses.replace(
            spec,
            rotary=dataclasses.replace(spec.rotary, key_freqs=query_freqs),
        )
        invalid = PreparedAttention.from_hnd(
            query,
            key,
            value,
            heads=2,
            qk_transform=invalid_spec,
        )
        self.assertEqual(
            _prepared_external_call_reason(invalid),
            "prepared key RoPE token count does not match key",
        )

    def test_observer_requirement_fails_closed(self):
        request, owners = self.request(
            observer_requirements=frozenset(("post_rope_query",)),
        )
        outcome = _make_dense_prepared_executor("sage")(request)
        self.assertFalse(outcome.supported)
        self.assertIn("observer", outcome.reason)
        self.assertTrue(all(owner.tensor is not None for owner in owners))

    def test_backend_cannot_reject_after_consuming_inputs(self):
        def invalid_executor(request):
            request.consume_qkv()
            return AttentionExecutionOutcome.unsupported("late rejection")

        options = {ATTENTION_EXECUTOR_KEY: invalid_executor}
        request, _ = self.request(transformer_options=options)
        with self.assertRaisesRegex(RuntimeError, "after consuming"):
            execute_prepared_attention(request)

    def test_layout_protocol_describes_unequal_query_and_key_sequences(self):
        semantic = AttentionSemanticLayout(
            provider="test_cross_attention",
            query_segments=(AttentionSegment.for_role(0, 128, "target_video"),),
            key_segments=(
                AttentionSegment.for_role(0, 64, "text"),
                AttentionSegment.for_role(64, 192, "target_video"),
            ),
            topologies=(),
            layer_index=3,
            layer_count=8,
        )
        options = {ATTENTION_LAYOUT_KEY: semantic.to_wire()}
        self.assertTrue(
            has_complete_attention_layout(
                options,
                128,
                key_sequence_length=192,
                provider="test_cross_attention",
            )
        )
        parsed = attention_semantic_layout(options)
        self.assertEqual(parsed.query_segments[-1].stop, 128)
        self.assertEqual(parsed.key_segments[-1].stop, 192)


if __name__ == "__main__":
    unittest.main()
