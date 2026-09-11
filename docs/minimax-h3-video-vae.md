# MiniMax H3 Video VAE

The H3 VAE nodes accept normal ComfyUI `VAE`, `LATENT`, and `IMAGE` types.
They retain native H3 spatial/temporal reconstruction and add fused operators,
decoder attention selection, and completed-tile progress in the UI and tqdm.
Use the official `VAELoader`; no separate Turing Utils VAE loader is required
or provided. These optimizations activate only while the dedicated H3 node is
executing. Other nodes using the same VAE object retain normal dispatch.

## Decode

`MiniMax H3 Video VAE Decode` evaluates each spatial window independently:

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

`MiniMax H3 Video VAE Encode` retains the native 256px/64px tiled CNN encoder,
linear blending, temporal layout, and latent normalization. It has no attention
selector because the encoder is convolutional. The same execution-local
operator scope is enabled, but only compatible quantized operations use it;
ordinary convolutions retain their native implementation and dtype.

## Execution and validation

Independent windows can be batched within the current memory budget (up to
sixteen). Only the one-tile requirement is passed to ComfyUI model loading.
After loading, larger batches are selected from idle/reusable allocator memory
minus the configured reserve and not-yet-resident VAE weights. Evictable
DiT/CLIP weights are not counted as optional batching capacity. One tile can
still require unavoidable offloading on a small card; this is not a guarantee
that every model remains resident or that concurrent GPU allocations cannot OOM.
ComfyUI owns model loading and short-lived weight prefetch queues;
pixel transfers retain asynchronous buffering. Output tensors use
`vae.vae_output_dtype()`, and progress advances as tile work completes.

Regression tests compare the decoder directly with native `ViT3DDecoder` and
`MiniMaxH3VideoVAE.tiled_decode`, including multiple windows, input batches,
tile-batch sizes, FP16 CUDA execution, and differing pixels in overlap regions.
Temporal and encoder reference comparisons remain covered separately.

On A40/cu128, a synthetic full-width 36-block INT8/FP16/SDPA decoder window
dropped from 386.96 ms to 110.09 ms. A synthetic 864×480, 22-frame decode at
tile batch 1 dropped from 5855.08 ms to 1675.47 ms, with bitwise-equal output
in both comparisons. These are random-weight, resident-model measurements,
not real-checkpoint quality or end-to-end generation results. Batch 4 saved
another 103 ms but used 417 MiB more peak memory and was not bitwise equal to
single-tile execution (maximum pixel difference 0.000895). Larger batches and
different attention backends must not be described as universally lossless.
Actual SM75/Windows and trained-checkpoint validation are still required.
