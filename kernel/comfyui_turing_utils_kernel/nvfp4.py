"""Compact native NVFP4 storage, paired ConvRot256 and bounded INT8 GEMM.

Only this algorithm is used. Padding is mathematical zero padding on BOTH
operands; serialized padding is never interpreted as model weight values.
No decoded weight or graph cache survives a call.
"""
import torch
import torch.nn.functional as F
from comfy_kitchen.backends import cuda as ck
from comfy_kitchen.registry import registry
from .ops import (
    turing_nvfp4_convrot_quantize, turing_int8_linear_out,
    turing_bf16_int8_convrot_quantize,
    turing_nvfp4_convrot_quantize_out,
)


def _linear_pipeline(qx, sx, weight, blocks, scale, output, bias, rows, logical_k, n):
    main = torch.cuda.current_stream(qx.device)
    producer = torch.cuda.Stream(device=qx.device)
    buffers = [torch.empty((rows, qx.shape[1]), device=qx.device, dtype=torch.int8) for _ in range(2)]
    scales = [torch.empty(rows, device=qx.device, dtype=torch.float32) for _ in range(2)]
    ready = [torch.cuda.Event() for _ in range(2)]
    consumed = [torch.cuda.Event() for _ in range(2)]
    producer.wait_stream(main)

    def prepare(index):
        slot = index % 2
        start = index * rows
        count = min(rows, n - start)
        with torch.cuda.stream(producer):
            if index >= 2:
                producer.wait_event(consumed[slot])
            turing_nvfp4_convrot_quantize_out(
                weight, blocks, scale, buffers[slot][:count], scales[slot][:count], start, logical_k)
            ready[slot].record()

    # Joining before return also makes allocator reuse and offloading safe;
    # no decoded weights or tensor-owning stream state survive this call.
    try:
        prepare(0)
        for index, start in enumerate(range(0, n, rows)):
            slot = index % 2
            count = min(rows, n - start)
            if start + rows < n:
                prepare(index + 1)
            main.wait_event(ready[slot])
            turing_int8_linear_out(qx, buffers[slot][:count], sx, scales[slot][:count],
                                  output[:, start:start + count],
                                  None if bias is None else bias[start:start + count])
            consumed[slot].record()
    finally:
        main.wait_stream(producer)


def quantize_activation(x, *, pre_scale=None, swiglu=False):
    if swiglu:
        if x.shape[-1] % 2:
            raise ValueError("SwiGLU requires an even input width")
        k = x.shape[-1] // 2
        if pre_scale is None and x.dtype != torch.float16 and x.is_cuda and x.shape[0] > 0 and k > 0 and k % 256 == 0:
            # Keep activation + rotation in one pass where shared memory fits.
            # FP32 activations are never rounded through a half-precision buffer.
            if k <= 16384 and ck._convrot_fused_shared_memory_fits(x, k, 256):
                return ck.quantize_int8_rowwise_convrot64(
                    x.contiguous(), 256, input_act="swiglu")
            # The existing BF16 row-buffer primitive needs less shared memory
            # than Kitchen's FP32 staging, including on SM75.
            if x.dtype == torch.bfloat16 and k <= 16384:
                return turing_bf16_int8_convrot_quantize(x, 256, swiglu=True)
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up
    if pre_scale is not None:
        x = x * pre_scale
    k = x.shape[-1]
    x = F.pad(x, (0, (-k) % 256)) if k % 256 else x
    if x.dtype == torch.float16:
        # Half-precision divisors underflow for tiny rows. Quantize in FP32,
        # after activation/pre-scale, and bound the temporary by row chunks.
        q = torch.empty_like(x, dtype=torch.int8)
        s = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        rows = max(1, (16 * 1024**2) // (4 * x.shape[1]))
        for start in range(0, x.shape[0], rows):
            end = min(start + rows, x.shape[0])
            chunk_q, chunk_s = ck.quantize_int8_convrot_weight(x[start:end].float().contiguous(), 256)
            q[start:end].copy_(chunk_q)
            s[start:end].copy_(chunk_s.reshape(-1, 1))
        return q, s
    return ck.quantize_int8_convrot_weight(x.contiguous(), 256)


def linear(x, weight, blocks, scale, bias=None, *, output_columns=None,
           pre_scale=None, swiglu=False):
    if x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NVFP4 linear requires a FP16/BF16/FP32 CUDA matrix")
    k = x.shape[1] // 2 if swiglu else x.shape[1]
    n = weight.shape[0] if output_columns is None else output_columns
    if k <= 0 or k > weight.shape[1] * 2 or n <= 0 or n > weight.shape[0]:
        raise ValueError("logical NVFP4 dimensions exceed packed storage")
    padded_n = (n + 7) // 8 * 8
    if padded_n > weight.shape[0]:
        raise ValueError("NVFP4 stored rows must include N padding to 8")
    if bias is not None and (bias.ndim != 1 or bias.numel() != n):
        raise ValueError("bias must have one value per logical output column")
    if x.shape[0] == 0:
        return x.new_empty((0, n))
    qx, sx = quantize_activation(x, pre_scale=pre_scale, swiglu=swiglu)
    # At most 16 MiB of decoded S8 per chunk (minimum 8 rows).
    rows = min(2048, max(8, (16 * 1024**2 // qx.shape[1]) // 8 * 8))
    if rows >= 256:
        rows = rows // 256 * 256
    # Keep row starts on 128-byte boundaries without decoding/computing extra
    # weight rows. Native H3 widths already satisfy this alignment.
    alignment = 128 // x.element_size()
    output_stride = (padded_n + alignment - 1) // alignment * alignment
    output = x.new_empty((x.shape[0], output_stride))
    if bias is not None:
        bias = F.pad(bias.float(), (0, padded_n - n))
    # Same algorithm and tile sizes; overlap only in the measured SM86 range.
    # A second <=16 MiB staging buffer is temporary, never a weight cache.
    if (x.dtype == torch.bfloat16 and 8192 <= x.shape[0] <= 24576
            and 8192 <= qx.shape[1] <= 16384 and 4096 <= padded_n < 16384
            and torch.cuda.get_device_capability(x.device) == (8, 6)
            and not torch.cuda.is_current_stream_capturing()):
        _linear_pipeline(qx, sx, weight, blocks, scale, output, bias, rows, k, padded_n)
        return output[:, :n]
    # Kitchen's prequantized GEMV wrapper allocates exactly one output row.
    use_kitchen_gemv = (
        x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and bias is None
        and qx.shape[1] <= 16384
        and torch.cuda.get_device_capability(x.device) >= (8, 0)
        and registry.is_available("cuda")
        and callable(getattr(getattr(ck, "_C", None), "int8_gemv_dequant", None))
        and callable(getattr(ck, "int8_gemv_dequant", None))
        and not torch.cuda.is_current_stream_capturing()
    )
    for start in range(0, padded_n, rows):
        end = min(start + rows, padded_n)
        qw, sw = turing_nvfp4_convrot_quantize(weight, blocks, scale, start, end-start, k)
        tile = output[:, start:end]
        if use_kitchen_gemv:
            with torch.cuda.device(x.device):
                tile.copy_(ck.int8_gemv_dequant(qx, qw, sx.reshape(-1, 1), sw, None, x.dtype))
        else:
            turing_int8_linear_out(
                qx, qw, sx, sw, tile, None if bias is None else bias[start:end]
            )
        del qw, sw
    return output[:, :n]


def validate_runtime(device):
    from . import _C
    if getattr(_C, "nvfp4_runtime_schema", 0) != 6:
        raise RuntimeError("Rebuild Turing Utils: NVFP4 runtime schema 6 required")
    with torch.cuda.device(device):
        for n in (13, 16384):
            stored_n = (n + 127) // 128 * 128
            w = torch.zeros((stored_n, 16), device=device, dtype=torch.uint8)
            b = torch.zeros((stored_n, 4), device=device, dtype=torch.uint8)
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                y = linear(torch.zeros((1, 17), device=device, dtype=dtype),
                           w, b, torch.ones(1, device=device),
                           torch.ones(n, device=device), output_columns=n)
                if not bool((y == 1).all()):
                    raise RuntimeError("NVFP4 runtime validation failed")
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            q, s = quantize_activation(
                torch.zeros((2, 512), device=device, dtype=dtype), swiglu=True)
            if not bool((q == 0).all()) or not bool(s.isfinite().all()):
                raise RuntimeError("NVFP4 fused activation validation failed")
