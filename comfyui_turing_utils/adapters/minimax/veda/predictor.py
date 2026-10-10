"""Host-owned Veda projections; only the requested heads are staged on CUDA.

The predictor sees post-RoPE Q/K in their original basis. Its optional ConvRot
is local to the 3D pooled features, not a rotation of the attention Q/K.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ....kernel_api import load_kernel_extension
from .plans import PlanTable, TilePlan


PRECISIONS = ("w8a8", "bf16", "fp16", "fp32")
FLOAT_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
FORMAT = "miowtion-veda-predictor-v1"
CONVROT_GROUP = 256


def predictor_compute_dtype(precision: str, capability: tuple[int, int] | None):
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported Veda predictor precision: {precision}")
    if precision == "w8a8":
        return torch.int8
    if precision == "bf16" and capability is not None and capability < (8, 0):
        return torch.float32
    return FLOAT_DTYPES[precision]


def rotate_features(x: torch.Tensor) -> torch.Tensor:
    # Kitchen's regular, normalized H4 Kronecker rotation, without a persistent
    # device tensor cache. This also serves as the CPU conversion/reference path.
    original = x.shape
    y = x.float().reshape(*original[:-1], -1, CONVROT_GROUP)
    stride = 1
    while stride < CONVROT_GROUP:
        y = y.reshape(*original[:-1], -1, 4, stride)
        a, b, c, d = y.unbind(-2)
        y = (
            torch.stack(
                (a + b + c - d, a + b - c + d, a - b + c + d, -a + b + c + d), dim=-2
            )
            * 0.5
        )
        stride *= 4
    return y.reshape(original)


@dataclass(frozen=True)
class Projection:
    # [heads, D, padded_3D] for W8A8, [heads, 3D, D] otherwise.
    weight: torch.Tensor
    scale: torch.Tensor | None

    def select_heads(self, heads: list[int]):
        if heads and heads == list(range(heads[0], heads[0] + len(heads))):
            if heads[0] < 0 or heads[-1] >= self.weight.shape[0]:
                raise IndexError("Veda predictor head index out of range")
            return tuple(
                None if t is None else t[heads[0] : heads[-1] + 1]
                for t in (self.weight, self.scale)
            )
        index = torch.tensor(heads, dtype=torch.long)
        return tuple(
            None if t is None else t.index_select(0, index)
            for t in (self.weight, self.scale)
        )

    def stage(self, heads: list[int], device: torch.device) -> Projection:
        return Projection(
            *(
                None if t is None else t.to(device, copy=True)
                for t in self.select_heads(heads)
            )
        )

    @property
    def bytes_per_head(self) -> int:
        size = self.weight[0].numel() * self.weight.element_size()
        return size + (
            0
            if self.scale is None
            else self.scale[0].numel() * self.scale.element_size()
        )


class ProjectionTransfer:
    """One bounded head group's Q/K upload; never a GPU weight cache.

    Keep pinned sources alive until consumption, and record the consumer stream
    on copied allocations. No global streams or allocations survive a forward.
    """

    def __init__(self, projections, heads, device, stream):
        self.sources = []
        self.outputs = []
        for projection in projections:
            self.sources.append(
                tuple(
                    None if t is None else t.pin_memory()
                    for t in projection.select_heads(heads)
                )
            )
        with torch.cuda.stream(stream):
            for pair in self.sources:
                self.outputs.append(
                    Projection(
                        *(
                            None if t is None else t.to(device, non_blocking=True)
                            for t in pair
                        )
                    )
                )
            self.ready = torch.cuda.Event()
            self.ready.record(stream)

    def consume(self, device):
        current = torch.cuda.current_stream(device)
        current.wait_event(self.ready)
        for projection in self.outputs:
            projection.weight.record_stream(current)
            if projection.scale is not None:
                projection.scale.record_stream(current)
        return self.outputs


def convert_projection(weight: torch.Tensor, precision: str) -> Projection:
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported Veda predictor precision: {precision}")
    if weight.device.type != "cpu":
        raise ValueError("Veda bundle conversion must run on the host")
    if not torch.isfinite(weight.float()).all():
        raise ValueError("Veda predictor contains non-finite weights")
    if precision != "w8a8":
        return Projection(weight.to(FLOAT_DTYPES[precision]).contiguous(), None)
    w = weight.float().transpose(1, 2)
    w = F.pad(w, (0, (-w.shape[-1]) % CONVROT_GROUP))
    w = rotate_features(w)
    scale = w.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    return Projection(
        (w / scale).round().clamp(-127, 127).to(torch.int8).contiguous(), scale
    )


@dataclass(frozen=True)
class PredictorBundle:
    precision: str
    num_layers: int
    num_heads: int
    head_dim: int
    keep_ratio: float
    plans: PlanTable
    proj_q: tuple[Projection, ...]
    proj_k: tuple[Projection, ...]

    def __deepcopy__(self, memo):
        # ModelPatcher branches may deep-copy transformer_options. Converted
        # host weights are read-only; stage() always makes a separate copy.
        memo[id(self)] = self
        return self

    @property
    def staged_bytes_per_head(self) -> int:
        return self.proj_q[0].bytes_per_head + self.proj_k[0].bytes_per_head


def load_bundle(path: str, precision: str = "w8a8") -> PredictorBundle:
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported Veda predictor precision: {precision}")
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(f"Not a Veda predictor bundle: expected {FORMAT}")
        layers, heads, dim = (
            int(metadata[k]) for k in ("num_layers", "num_heads", "head_dim")
        )
        if min(layers, heads, dim) <= 0:
            raise ValueError("Veda predictor dimensions must be positive")
        stored = metadata.get("dtype", "float32")
        storage_types = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        if stored not in storage_types:
            raise ValueError(f"Unsupported Veda bundle storage dtype: {stored}")
        table = PlanTable(
            [TilePlan.from_json(p) for p in json.loads(metadata["plans"]).values()]
        )
        for plan in table.plans.values():
            if len(plan.grid) != 3 or any(
                not isinstance(n, int) or n <= 0 for n in plan.grid
            ):
                raise ValueError(f"Veda tile plan {plan.name} has an invalid grid")
            if plan.num_layers != layers or any(
                len(row) != heads for row in plan.head_shape
            ):
                raise ValueError(
                    f"Veda tile plan {plan.name} disagrees with predictor dimensions"
                )
            if any(
                i < 0 or i >= len(plan.shapes) for row in plan.head_shape for i in row
            ):
                raise ValueError(
                    f"Veda tile plan {plan.name} contains an invalid shape index"
                )
        remaining = set(source.keys())
        projections = {"proj_q": [], "proj_k": []}
        for layer in range(layers):
            for name, output in projections.items():
                key = f"layers.{layer}.{name}"
                value = source.get_tensor(key)
                remaining.remove(key)
                if (
                    tuple(value.shape) != (heads, 3 * dim, dim)
                    or value.dtype != storage_types[stored]
                ):
                    raise ValueError(f"Veda {key}: invalid shape or storage dtype")
                if stored == "float8_e4m3fn":
                    scale_key = key + ".__scale"
                    scale = source.get_tensor(scale_key).float()
                    remaining.remove(scale_key)
                    if (
                        tuple(scale.shape) != (heads,)
                        or not torch.isfinite(scale).all()
                        or (scale < 0).any()
                    ):
                        raise ValueError(f"Veda {key}: invalid per-head FP8 scale")
                    value = value.float() * scale[:, None, None]
                output.append(convert_projection(value, precision))
                del value
        if remaining:
            raise ValueError(f"Unexpected Veda bundle tensors: {sorted(remaining)[:4]}")
    ratio = float(metadata["keep_ratio"])
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("Veda bundle keep_ratio must be in (0, 1]")
    return PredictorBundle(
        precision,
        layers,
        heads,
        dim,
        ratio,
        table,
        tuple(projections["proj_q"]),
        tuple(projections["proj_k"]),
    )


def project_features(
    features: torch.Tensor,
    projection: Projection,
    precision: str,
    *,
    capability: tuple[int, int] | None = None,
) -> torch.Tensor:
    if capability is None and features.device.type == "cuda":
        capability = torch.cuda.get_device_capability(features.device)
    compute_dtype = predictor_compute_dtype(precision, capability)
    dim = features.shape[-1] // 3
    if precision == "w8a8":
        # Use the shared ConvRot GEMM with per-head weights and fused residual.
        ops = load_kernel_extension("ops")
        x = features.to(torch.bfloat16)
        padded = F.pad(x, (0, projection.weight.shape[-1] - x.shape[-1]))
        quantized, scales = ops.turing_bf16_int8_convrot_quantize(
            padded.flatten(0, 1).contiguous()
        )
        quantized = quantized.reshape(*x.shape[:2], -1)
        scales = scales.reshape(*x.shape[:2], 1)
        return ops.turing_int8_batched_residual(
            quantized, projection.weight, scales, projection.scale, x.contiguous()
        )
    output_dtype = FLOAT_DTYPES[precision]
    # SM75 emulates BF16 operands in FP32, but still returns BF16. Add the
    # residual before the final conversion, without a rounded GEMM temporary.
    x = features.to(output_dtype)
    weight = projection.weight.to(output_dtype)
    x = x.to(compute_dtype)
    return torch.baddbmm(x[..., :dim], x, weight.to(compute_dtype)).to(output_dtype)


def score_tiles(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Route comparison uses FP32 logits. BF16/FP16 operands retain their
    # activation rounding; FP32 mode keeps unrounded diagnostic activations.
    return torch.bmm(q.float(), k.float().transpose(1, 2)) / math.sqrt(q.shape[-1])
