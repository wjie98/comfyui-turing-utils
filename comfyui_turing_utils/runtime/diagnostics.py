"""Serializable runtime diagnostics for support reports and release gates."""

from __future__ import annotations

from dataclasses import asdict

import torch

from .capabilities import runtime_capabilities


FEATURES = (
    "core_fusions",
    "ffn_channel_sharding",
    "grouped_w6a8",
    "stable_sage",
    "dense_w8a8",
    "sol",
    "sla",
    "split_prequantization",
    "fused_qk",
    "reusable_k_anchor",
    "overlap_accumulate",
)


def runtime_diagnostics(device: torch.device | str) -> dict:
    runtime = runtime_capabilities(device)
    device_data = asdict(runtime.device)
    device_data["device"] = str(runtime.device.device)
    device_data["architecture"] = runtime.device.architecture
    result = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "device": device_data,
        "kernel": {
            "installed": runtime.kernel.installed,
            "version": ".".join(map(str, runtime.kernel.version)),
            "features": sorted(runtime.kernel.features),
            "reason": runtime.kernel.reason,
        },
        "support": {},
    }
    for feature in FEATURES:
        support = runtime.supports(feature)
        result["support"][feature] = {
            "supported": support.supported,
            "reason": support.reason,
        }
    if runtime.device.cuda:
        try:
            free, total = torch.cuda.mem_get_info(runtime.device.device)
        except (AttributeError, RuntimeError):
            free = total = 0
        result["memory"] = {
            "driver_free": int(free),
            "driver_total": int(total),
            "torch_allocated": int(torch.cuda.memory_allocated(runtime.device.device)),
            "torch_reserved": int(torch.cuda.memory_reserved(runtime.device.device)),
        }
    return result


__all__ = ["FEATURES", "runtime_diagnostics"]
