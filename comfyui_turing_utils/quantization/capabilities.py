"""Low-cost ConvRot kernel and comfy-kitchen capability probes."""

from __future__ import annotations

from ..kernel_api import load_kernel_extension, load_kernel_package
from .kitchen_api import backend_status


BACKEND_NAME = "turing_utils_sm75"


def kernel_available(name: str = "turing_w4a8_linear") -> bool:
    """Return whether both the Python API and compiled ABI export *name*."""
    try:
        extension = load_kernel_extension("_C")
        package = load_kernel_package()
    except (ImportError, OSError):
        return False
    return callable(getattr(extension, name, None)) and callable(
        getattr(package, name, None)
    )


def kernel_op(name: str):
    """Resolve one public kernel-package operation or raise a stable error."""
    try:
        return getattr(load_kernel_package(), name)
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(f"bundled CUDA operation {name!r} is unavailable") from exc


def kitchen_backend_available() -> bool:
    """Return whether comfy-kitchen selected the shared sm75+ backend."""
    try:
        status = backend_status(BACKEND_NAME)
    except ImportError:
        return False
    return bool(status.get("available") and not status.get("disabled"))


__all__ = [
    "BACKEND_NAME",
    "kernel_available",
    "kernel_op",
    "kitchen_backend_available",
]
