"""Quantized storage facts, independent of device and compute-dtype policy."""

from __future__ import annotations

from dataclasses import dataclass

W4_FORMAT = "convrot_w4a4"
GROUPED_INT8_FORMAT = "asym_w4a8_int8"
W8_FORMAT = "int8_tensorwise"
LEGACY_W8_FORMAT = "int8_rowwise"
NVFP4_FORMAT = "nvfp4"
W4_LAYOUT = "TensorCoreConvRotW4A4Layout"
GROUPED_INT8_LAYOUT = "AsymW4A8Int8Layout"
W8_LAYOUT = "TensorWiseINT8Layout"
NVFP4_LAYOUT = "TensorCoreNVFP4Layout"


def grouped_weight_geometry(packed_shape, scale_shape, group_size: int):
    """Return logical (N, K, bits) without unpacking or transferring weights."""
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size < 4
        or (16 % group_size and group_size % 16)
    ):
        raise ValueError("group_size must divide 16 or be a multiple of 16, and be >=4")
    if len(packed_shape) != 2 or len(scale_shape) != 2:
        raise ValueError("Grouped INT8 weights and scales must be two-dimensional")
    n, packed_k = packed_shape
    scale_n, groups = scale_shape
    k = groups * group_size
    if n <= 0 or packed_k <= 0 or groups <= 0 or n != scale_n or k % 16:
        raise ValueError("Grouped INT8 weight/scale geometry is inconsistent")
    bits, remainder = divmod(packed_k * 8, k)
    if remainder or bits not in (4, 6):
        raise ValueError("Grouped INT8 packed rows must store 4 or 6 bits per weight")
    if bits == 6 and (k % 32 or group_size < 16 or group_size % 16):
        raise ValueError("W6A8 requires K%32=0 and group_size a multiple of 16")
    return n, k, bits


@dataclass(frozen=True, slots=True)
class WeightStorage:
    kind: str
    rotation_group: int | None
    group_size: int | None = None
    transposed: bool = False
    correction: bool = False


def describe_weight_storage(weight) -> WeightStorage | None:
    params = getattr(weight, "_params", None)
    layout = getattr(weight, "_layout_cls", None)
    rotation = getattr(params, "convrot_groupsize", None)
    transposed = bool(getattr(params, "transposed", False))
    if layout == NVFP4_LAYOUT:
        # NVFP4 is stored in the original basis; its activation-time ConvRot
        # bridge does not turn the resident checkpoint into a ConvRot weight.
        return WeightStorage("nvfp4", None, 16, transposed)
    if layout == W8_LAYOUT and getattr(params, "convrot", False):
        return WeightStorage("w8a8", rotation, transposed=transposed)
    if layout == W4_LAYOUT:
        kind = {"int4": "w4a4", "int8": "w4a8"}.get(
            getattr(params, "linear_dtype", None)
        )
        if kind is not None:
            return WeightStorage(kind, rotation, params.quant_group_size, transposed)
    if layout == GROUPED_INT8_LAYOUT:
        group_size = params.group_size
        _, _, bits = grouped_weight_geometry(
            weight._qdata.shape, params.scale.shape, group_size
        )
        codebook = params.codebook
        if codebook is not None and (bits != 4 or codebook.numel() != 16):
            raise ValueError("Only grouped W4A8 accepts a 16-entry codebook")
        kind = (
            "w6a8"
            if bits == 6
            else "codebook_w4a8"
            if codebook is not None
            else "uniform_w4a8"
        )
        return WeightStorage(
            kind, rotation, group_size, transposed, params.correction is not None
        )
    return None


def convrot_storage_kind(weight) -> str | None:
    """Formats supported by the local ConvRot family; no dtype requirement."""
    storage = describe_weight_storage(weight)
    if storage is None or storage.transposed or storage.rotation_group != 256:
        return None
    if storage.kind in {"w4a4", "w4a8"} and storage.group_size != 64:
        return None
    if storage.correction or storage.kind == "uniform_w4a8":
        return None
    return storage.kind
