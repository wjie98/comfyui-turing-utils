# MiniMax H3 Video VAE

The H3 VAE nodes accept normal ComfyUI `VAE`, `LATENT`, and `IMAGE` types.
They call the official ComfyUI VAE entry points and add scoped fused operators,
decoder attention selection, and lightweight submitted-tile progress in tqdm.
Use the official `VAELoader`; no separate Turing Utils VAE loader is required
or provided. These optimizations activate only while the dedicated H3 node is
executing. Other nodes using the same VAE object retain normal dispatch.

## Decode

`MiniMax H3 Video VAE Decode` delegates to `vae.decode(samples)`. The official
H3 model evaluates each spatial window independently:

- native H3 window geometry (normally 256px with at least 64px overlap);
- independent image/register tokens and local RoPE for all decoder blocks;
- native pixel projection and ordered vertical/horizontal linear blending;
- native temporal padding, overlap, trimming, and pixel normalization.

Shared hidden states, query pruning, custom overlap accumulation, and multiband
pixel stitching are no longer used. The `overlap_query_threshold` and
`final_full_overlap_blocks` inputs have been removed. Existing graphs keep the
same node type and `samples`, `vae`, and `attention` inputs; recreate the node
if an older frontend retains the deleted widgets.

The attention selector applies to every decoder Transformer block:

- `native`: leave the upstream attention forward and backend selection intact.
  This inherits upstream decoder attention fixes without a plugin override.
  Turing's scoped linear-operator dispatch and tile progress remain enabled;
  use the official decode node for a completely unmodified baseline.
- `sdpa`: PyTorch SDPA; Turing BF16 inputs compute in FP16 to avoid its slow
  math fallback;
- `sage`: bundled Turing Sage attention;
- `w8a8`: quantized QK attention when the installed kernel supports it.

This selector does not change VAE weight quantization. Eligible INT8 decoder
FFNs use ComfyUI's fused SwiGLU/input-quantization path. Fusion and lower-precision
attention can still introduce numerical differences; neither changes the
independent-window reconstruction policy.

Kernel 0.42 adds an FP16-input/FP16-output INT8 ConvRot path (SM75+). It consumes
the existing `[N,K]` INT8 weight storage directly as the column-major GEMM
operand, without a transposed weight copy, and fuses row quantization and the
scale/bias epilogue. Activation and Hadamard rotation retain Kitchen's native
implementation: replacing rotation with a butterfly reduction changed a few
rounding-threshold values and amplified differences through 36 decoder blocks.
FP32 calls and unsupported shapes retain Kitchen's normal
dispatch. The existing BF16 kernels are not used to lower VAE compute precision.

Input latent storage may be FP16, BF16, or FP32. Compute follows `vae.vae_dtype`:
the [ComfyUI H3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd.py)
advertises FP16/FP32 and defaults its quantized decoder to FP16; the
[Diffusers VAE example](https://huggingface.co/docs/diffusers/main/en/api/models/autoencoderkl_minimax_h3)
uses FP32. Do not infer VAE compute precision from the DiT checkpoint's BF16 label.

## Encode

`MiniMax H3 Video VAE Encode` delegates to `vae.encode(pixels)`, retaining the
official cropping, tiled CNN encoder, linear blending, temporal layout, and
latent normalization. It has no attention
selector because the encoder is convolutional. The same execution-local
operator scope is enabled, but only compatible quantized operations use it;
ordinary convolutions retain their native implementation and dtype.

## Execution and validation

The plugin does not merge spatial tiles into larger batches, create pixel-copy
streams or pinned double buffers, reserve a non-evicting memory budget, or create
block-level weight-prefetch queues. ComfyUI owns model loading/unloading,
DynamicVRAM/aimdo paging, official batching, input/output transfers, dtype and
OOM recovery. The plugin does not retry allocator or kernel errors on its own.
Keeping other models resident is not promised; ComfyUI may offload them as needed.

The selected tiled decoder receives a temporary attention method override.
Current ComfyUI calls `attend`, and older versions call `forward`.
The `native` option keeps those methods as provided by ComfyUI. Sol-H3 uses its
native window attention and streamed decoding. Linear fusion is selected
through operator dispatch.
The official decoder block loop, spatial blending and temporal
reconstruction are not replaced. These instance overrides and progress hooks are
restored on success, errors and cancellation. No process-global attention method
is patched, and no device tensors are persistently cached on the VAE.

For normal single-video inputs, progress precomputes the exact native H3 spatial
and temporal tile plan from the model's own `split_tiles` and temporal-chunk
helpers. tqdm can therefore show `current/total`, percentage and ETA. The counter
still measures native tile forwards submitted by the host, not GPU completion or
end-to-end time. Nonstandard latent batches stay open-ended because ComfyUI may
split them according to live memory. If an official retry exceeds the planned
total, the display falls back to open-ended progress rather than showing a false
percentage. No CUDA events, synchronization, background threads or additional
ComfyUI UI progress hooks are used.

Regression tests cover direct delegation to the official VAE, native output
parity, tile/input-batch preservation, dtype handling, official OOM fallback,
attention/fusion dispatch, and restoration after errors or cancellation.

### Upstream maintenance

Pre-attention RMSNorm weights stay inside ComfyUI's cast context until the fused
QKV linear returns, including exception cleanup. Newer norm modules own this
context through `cast_weight`; the current core uses `CastBiasWeightContext`.
No norm weight or device copy is cached on the model. The tensor form of
`linear_input_act` remains usable across these core interfaces.

Upstream spatial/temporal blending, encoder kernels and tile batching should
never be copied into this plugin. Test after core upgrades instead. Attention
overrides and the custom Kitchen backend still require interface tests: choosing
a custom kernel intentionally bypasses improvements to the equivalent Kitchen
kernel. `native` is an explicit upstream-attention path, not silent recovery
from arbitrary CUDA errors.

The 2026-10-08 upstream audit identified norm offload fixes (#16698), linear
wrapper changes (#16861), light VAE loading (#16657), and H3 embedding temporary
release (2d6b73283a). Core/Kitchen upgrades must be tested separately from plugin
changes; this plugin change does not install those updates. Reduced-layer VAE
support belongs to the official loader, not a hard-coded plugin layer count.

For long-video regressions, compare actual token counts and model evaluations,
not seconds alone. Hold checkpoint, LoRAs, resolution, references, sparse ratios,
seed and sigmas fixed, then test 8/10/12/15-second workloads. Record separate
encode/text-encode/DiT/decode times, per-step latency, head/row chunk policy,
VRAM/RAM peaks and PCIe transfer activity. A sudden jump accompanied by smaller
chunks or weight transfers is different from the smooth quadratic growth of
fixed-density attention. No universal 10-second threshold is assumed.

Historical operator measurements (before the native-lifecycle simplification):
on A40/cu128, a synthetic full-width 36-block INT8/FP16/SDPA decoder window
dropped from 386.96 ms to 110.09 ms. A synthetic 864×480, 22-frame decode at
tile batch 1 dropped from 5855.08 ms to 1675.47 ms, with bitwise-equal output
in both comparisons. These are random-weight, resident-model measurements,
not real-checkpoint quality or end-to-end generation results. Batch 4 saved
another 103 ms but used 417 MiB more peak memory and was not bitwise equal to
single-tile execution (maximum pixel difference 0.000895). Custom tile batching
is now removed; these timings are not performance claims for the new lifecycle.
Different attention backends must not be described as universally lossless.
Actual SM75/Windows and trained-checkpoint validation are still required.
