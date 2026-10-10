"""Independent official-arithmetic oracle for tests only.

Pass decoded source weights, NOT already converted W8A8 weights. Upstream
stores host projections in BF16, but keeps pooled features, projection results,
residual additions and route logits in FP32. Disable TF32 in CUDA comparisons.
"""

import math

import torch


@torch.no_grad()
def official_logits(features_q, features_k, source_q, source_k):
    dim = source_q.shape[-1]
    q = features_q.float()
    k = features_k.float()
    qhat = torch.bmm(q, source_q.bfloat16().float()) + q[..., :dim]
    khat = torch.bmm(k, source_k.bfloat16().float()) + k[..., :dim]
    return torch.bmm(qhat, khat.transpose(1, 2)) / math.sqrt(dim)


@torch.no_grad()
def route_agreement(reference_indices, reference_keep, indices, keep):
    """Mean per-row recall of the reference selected set (diagnostics only)."""
    matched = (
        (reference_indices[..., :, None] == indices[..., None, :]) & keep[..., None, :]
    ).any(-1) & reference_keep
    count = reference_keep.sum(-1)
    recall = matched.sum(-1).float() / count.clamp_min(1)
    return torch.where(count > 0, recall, 1.0).mean()
