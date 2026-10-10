"""Unified hardware, binary, and operator capability resolution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..hardware import DeviceCapabilities, device_capabilities
from ..kernel_api import (
    kernel_extension_has_symbol,
    kernel_version,
    load_kernel_package,
    load_turing_sage,
    qk_preprocess_protocol_schema,
)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = []
    for raw in str(value).split(".")[:3]:
        digits = "".join(character for character in raw if character.isdigit())
        if not digits:
            return (0, 0, 0)
        parts.append(int(digits))
    return tuple((parts + [0, 0, 0])[:3])


def _probe(module, name: str) -> bool:
    function = getattr(module, name, None)
    if not callable(function):
        return False
    try:
        return bool(function())
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    supported: bool
    reason: str | None = None

    def require(self, operation: str) -> None:
        if not self.supported:
            raise RuntimeError(
                f"{operation} is unavailable: {self.reason or 'unsupported'}"
            )


@dataclass(frozen=True, slots=True)
class KernelCapabilities:
    installed: bool
    version: tuple[int, int, int]
    features: frozenset[str]
    reason: str | None = None

    def supports(self, feature: str) -> CapabilityResult:
        if not self.installed:
            return CapabilityResult(
                False, self.reason or "kernel package is not installed"
            )
        if feature not in self.features:
            return CapabilityResult(False, f"kernel feature {feature!r} is unavailable")
        return CapabilityResult(True)


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    device: DeviceCapabilities
    kernel: KernelCapabilities

    def supports(self, feature: str) -> CapabilityResult:
        if feature == "stable_sage":
            if self.device.compute_capability != (7, 5) or not self.device.tensor_core:
                return CapabilityResult(
                    False, "bundled stable Sage requires an sm75 Tensor Core GPU"
                )
        elif feature in {
            "dense_w8a8",
            "sol",
            "sla",
            "fused_qk",
            "asymmetric_qk_rope",
            "mapped_kv",
            "mapped_sparse_kv",
            "mapped_sparse_fp16_kv",
            "reusable_k_anchor",
            "overlap_accumulate",
            "core_fusions",
            "ffn_channel_sharding",
            "ffn_half_width",
            "grouped_w6a8",
        }:
            if not self.device.tensor_core:
                return CapabilityResult(
                    False, "an NVIDIA sm75+ Tensor Core GPU is required"
                )
        return self.kernel.supports(feature)


_MINIMUM_VERSIONS = {
    "split_prequantization": (0, 20, 0),
    "fused_qk": (0, 22, 0),
    "dense_w8a8": (0, 23, 0),
    "sol": (0, 23, 0),
    "sla": (0, 29, 1),
    "reusable_k_anchor": (0, 30, 0),
    "ffn_channel_sharding": (0, 30, 0),
    "grouped_w6a8": (0, 45, 0),
}


def kernel_capabilities() -> KernelCapabilities:
    """Describe the installed ABI without executing a CUDA kernel."""
    version = _version_tuple(kernel_version())
    try:
        package = load_kernel_package()
        sage = load_turing_sage()
    except (ImportError, OSError) as error:
        return KernelCapabilities(False, version, frozenset(), str(error))

    features = set()
    if version >= _MINIMUM_VERSIONS["grouped_w6a8"] and kernel_extension_has_symbol(
        "turing_codebook_w4a8_linear"
    ):
        features.add("grouped_w6a8")
    if all(
        kernel_extension_has_symbol(symbol)
        for symbol in (
            "turing_segmented_rms_adaln",
            "turing_segmented_mod_gate",
            "turing_segmented_mod_gate_rms_adaln",
        )
    ):
        features.add("core_fusions")
    if version >= _MINIMUM_VERSIONS[
        "ffn_channel_sharding"
    ] and kernel_extension_has_symbol("turing_swiglu_int8_convrot_quantize_scaled"):
        features.add("ffn_channel_sharding")
    if all(
        kernel_extension_has_symbol(symbol)
        for symbol in (
            "turing_swiglu_convrot_shard_inplace",
            "turing_int8_convrot_quantize_from_partials",
        )
    ):
        features.add("ffn_half_width")
    if _probe(sage, "available"):
        features.add("stable_sage")
    probes = {
        "dense_w8a8": "w8a8_available",
        "sol": "sparse_available",
        "sla": "sla_available",
        "split_prequantization": "split_prequantization_available",
        "fused_qk": "fused_qk_preprocessing_available",
        "overlap_accumulate": "overlap_accumulate_available",
    }
    for feature, probe in probes.items():
        minimum = _MINIMUM_VERSIONS.get(feature, (0, 0, 0))
        if version >= minimum and _probe(sage, probe):
            features.add(feature)
    if (
        version >= _MINIMUM_VERSIONS["reusable_k_anchor"]
        and "fused_qk" in features
        and callable(getattr(sage, "precompute_rms_rope_k_anchor", None))
    ):
        features.add("reusable_k_anchor")
    if "fused_qk" in features and qk_preprocess_protocol_schema() >= 2:
        features.add("asymmetric_qk_rope")
    mapped_qk = bool(
        "fused_qk" in features
        and qk_preprocess_protocol_schema() >= 3
        and kernel_extension_has_symbol(
            "quant_qk_rms_rope_int8_mapped_cuda", "_sage_fused_sm75"
        )
    )
    if mapped_qk and kernel_extension_has_symbol(
        "gather_value_int8_mapped_cuda", "_sage_fused_sm75"
    ):
        features.add("mapped_kv")
    if "mapped_kv" in features and kernel_extension_has_symbol(
        "sol_w8a8_precompute_mapped_summaries", "_sage_qattn_sm75"
    ):
        features.add("mapped_sparse_kv")
    if mapped_qk and kernel_extension_has_symbol(
        "sol_sparse_online_int8_f16_mapped_attn", "_sage_qattn_sm75"
    ):
        features.add("mapped_sparse_fp16_kv")
    return KernelCapabilities(True, version, frozenset(features))


def runtime_capabilities(device: torch.device | str) -> RuntimeCapabilities:
    return RuntimeCapabilities(device_capabilities(device), kernel_capabilities())


__all__ = [
    "CapabilityResult",
    "KernelCapabilities",
    "RuntimeCapabilities",
    "kernel_capabilities",
    "runtime_capabilities",
]
