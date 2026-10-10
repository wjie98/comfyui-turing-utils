"""Veda attention packing; predictor features must be pooled before this step.

W8A8 uses shared Q/K Hadamard rotation, without chunk-local K centering.
V scales span the complete sequence so compact and whole preparation agree.
"""

from dataclasses import dataclass, replace
from importlib import import_module

import torch

from ....kernel_api import load_kernel_extension, load_turing_sage


@dataclass(frozen=True)
class AttentionPack:
    query_int8: torch.Tensor
    key_int8: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    value: torch.Tensor
    sm_scale: float
    value_int8: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None


def prequantize(q, k, v, *, use_w8a8=False):
    sage = load_turing_sage()
    if use_w8a8:
        quant = import_module(sage.__name__ + ".quant")
        qi, qs, ki, ks = quant.per_warp_int8_hadamard(q, k, stabilize_k=False)
        return AttentionPack(qi, ki, qs, ks, v, q.shape[-1] ** -0.5)
    return sage.prequantize_sageattn(q, k, v)


def finish_value(packed):
    if not isinstance(packed, AttentionPack):
        return packed
    b, h, n, d = packed.value.shape
    vi = torch.empty(
        (b, h, d, (n + 63) // 64 * 64), device=packed.value.device, dtype=torch.int8
    )
    vs = torch.empty((b, h, d), device=packed.value.device, dtype=torch.float32)
    load_kernel_extension("_sage_qattn_sm75").quantize_v_int8_sm75(packed.value, vi, vs)
    # Native W8A8 only needs the logical V dtype/head dimension, not its storage.
    return replace(
        packed,
        value=packed.value.new_empty((b, h, 0, d)),
        value_int8=vi,
        value_scale=vs,
    )
