"""Hardware capability discovery shared by every CUDA feature.

The historical public predicates remain available, but policy code should use
``device_capabilities`` when it needs more than a yes/no architecture gate.
This keeps hardware facts separate from independently installed kernel support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_TURING_WITHOUT_TENSOR_CORES = (
    "GTX 1630",
    "GTX 1650",
    "GTX 1660",
    "T500",
    "T550",
    "T600",
    "MX450",
    "MX550",
    "CMP 30HX",
    "T1000",
    "T1200",
    "T2000",
)


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    device: torch.device
    index: int | None
    name: str
    compute_capability: tuple[int, int] | None
    total_memory: int
    multiprocessor_count: int
    shared_memory_per_block: int
    optin_shared_memory_per_block: int
    shared_memory_per_multiprocessor: int
    tensor_core: bool
    native_bf16: bool
    async_copy: bool

    @property
    def cuda(self) -> bool:
        return self.compute_capability is not None

    @property
    def architecture(self) -> str | None:
        if self.compute_capability is None:
            return None
        major, minor = self.compute_capability
        return f"sm{major}{minor}"


def _property_int(properties, *names: str) -> int:
    for name in names:
        value = getattr(properties, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(value, 0)
    return 0


def device_capabilities(device: torch.device | str) -> DeviceCapabilities:
    """Return immutable hardware facts without probing any custom operator."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return DeviceCapabilities(
            device, None, "", None, 0, 0, 0, 0, 0, False, False, False
        )

    index = device.index if device.index is not None else torch.cuda.current_device()
    normalized = torch.device("cuda", index)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    try:
        properties = torch.cuda.get_device_properties(index)
    except (AttributeError, RuntimeError):
        properties = None
    try:
        name = str(torch.cuda.get_device_name(index))
    except (AttributeError, RuntimeError):
        name = str(getattr(properties, "name", ""))

    exact_turing_tensor_core = capability == (7, 5) and not any(
        model in name for model in _TURING_WITHOUT_TENSOR_CORES
    )
    tensor_core = exact_turing_tensor_core or capability >= (8, 0)
    shared = _property_int(
        properties,
        "shared_memory_per_block",
        "sharedMemPerBlock",
    )
    optin = _property_int(
        properties,
        "shared_memory_per_block_optin",
        "sharedMemPerBlockOptin",
    )
    return DeviceCapabilities(
        normalized,
        index,
        name,
        capability,
        _property_int(properties, "total_memory", "totalGlobalMem"),
        _property_int(
            properties,
            "multi_processor_count",
            "multiProcessorCount",
        ),
        shared,
        max(optin, shared),
        _property_int(
            properties,
            "shared_memory_per_multiprocessor",
            "sharedMemPerMultiprocessor",
        ),
        tensor_core,
        capability >= (8, 0),
        capability >= (8, 0),
    )


def is_supported_turing_device(device: torch.device) -> bool:
    """Return whether ``device`` is exact sm75 with usable Tensor Cores."""
    capabilities = device_capabilities(device)
    return bool(capabilities.tensor_core and capabilities.compute_capability == (7, 5))


def is_supported_tensor_core_device(device: torch.device) -> bool:
    """Return whether the shared low-precision CUDA kernels target ``device``.

    The runtime contract is capability based: exact sm75 devices must have
    Tensor Cores, while sm80 and newer devices use the same operator APIs with
    architecture-specific cubins selected by CUDA.
    """
    return device_capabilities(device).tensor_core


def is_supported_attention_device(device: torch.device) -> bool:
    """Compatibility alias for the original public predicate."""
    return is_supported_tensor_core_device(device)


__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "is_supported_attention_device",
    "is_supported_tensor_core_device",
    "is_supported_turing_device",
]
