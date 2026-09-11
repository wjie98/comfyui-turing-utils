# Operator and runtime support

This document is the source of truth for the production operator surface. It
separates public custom operators from Python dispatch and model-specific
adapters; supporting a quantization format does not imply that every GEMM is
implemented locally.

## Production stack

```text
ComfyUI nodes
    -> generic attention / quantization services
    -> optional MiniMax, Wan, or Bernini adapter
    -> kernel_api (the only independent-kernel import boundary)
    -> comfyui-turing-utils-kernel
```

The plugin and kernel package remain independently installable. A Python-only
plugin update does not rebuild CUDA. Sparse attention remains an explicit patch.
Kernel builds target all unique visible supported CUDA architectures unless a
cross-compilation list is supplied explicitly.
The loader defaults to bundled W8A8 on sm75 and newer Tensor Core GPUs; Sage
and SDPA remain explicit choices. Native cubins provide compile-time device
specialization without a separate Python model path.
ConvRot MODEL and CLIP loaders install a model-local Kitchen selection scope.
H3 VAE Encode/Decode nodes also enter a scope for the duration of their call,
including VAEs loaded by the official loader; they do not change VAE precision.
Exact local contracts have priority inside that scope; unsupported calls return
to Kitchen's normal dispatcher before execution. The plugin does not change the
global Kitchen backend priority, and it does not catch a selected operator's
runtime failure to continue with another implementation.
Attention has a separate single-owner container preflight: a shape or ABI that
fails capability checks is delegated before Q/K/V are consumed. A failure from
an attention kernel that has already started is not retried either.

## Public custom operators

Every operator below is registered with `torch.library.custom_op` and a fake
implementation.

| Family | `turing_utils::` operator | Purpose |
|---|---|---|
| Linear | `w4a8_linear` | Packed INT4 weights with INT8 activations on sm75+ Tensor Cores; BF16 output |
| Linear | `codebook_w4a8_linear` | Grouped-codebook INT4 storage with E4M3 group scales, inline packed-to-shared decode for long sequences, bounded staged fallback, sm75+ INT8 Tensor Core contraction, and BF16 output |
| Linear | `int8_linear` | Raw prequantized sm75+ W8A8 contraction used by the grouped-codebook path and backend regression gates |
| Linear | `fp16_int8_linear` | W8A8 with FP16 scale/bias rounding and FP16 output; native `[N,K]` weight layout |
| Epilogue | `dequantize_int8_bf16` | INT32 GEMM workspace to packed BF16 output |
| Activation quantization | `swiglu_int8_convrot_quantize`, `swiglu_int4_convrot_quantize` | Fused SwiGLU and ConvRot activation quantization |
| Activation quantization | `gelu_int8_convrot_quantize`, `gelu_int4_convrot_quantize` | Fused tanh-GELU and ConvRot activation quantization |
| Activation quantization | `bf16_int8_convrot_quantize`, `bf16_int4_convrot_quantize` | BF16 row-buffer ConvRot quantization, optionally with SwiGLU |
| Activation quantization | `fp16_int8_quantize` | FP16 row quantization after native activation/rotation; eager scale and rounding boundaries |
| Activation quantization | `swiglu_int8_convrot_quantize_scaled` | Quantize one aligned FFN channel interval with a precomputed whole-row scale |
| Activation quantization | `swiglu_convrot_shard_inplace`, `int8_convrot_quantize_from_partials` | Single-pass half-width FC1 staging with in-place SwiGLU+ConvRot and exact whole-row quantization |
| Activation quantization | `bf16_gelu_int8_convrot_quantize`, `bf16_gelu_int4_convrot_quantize` | BF16 row-buffer GELU and ConvRot quantization |
| Normalization | `segmented_rms_adaln` | RMSNorm plus segmented AdaLN modulation |
| Normalization | `segmented_mod_gate`, `segmented_mod_gate_rms_adaln` | In-place segmented residual gate, optionally fused with the following RMSNorm+AdaLN |
| Normalization | `layer_norm_adaln` | LayerNorm plus AdaLN modulation |
| Attention preprocessing | `qk_rms_rope_int8` | Fused RMSNorm, RoPE and production Q/K INT8 quantization |
| Attention | `sage_attention` | Stable dense INT8-QK, FP16/BF16-PV SM75 attention |
| Attention | `sage_attention_varlen` | Packed variable-length stable Sage attention |
| Attention | `w8a8_attention` | Dense INT8-QK and INT8-PV sm75+ attention |
| Attention | `w8a8_attention_varlen` | Packed variable-length dense W8A8 attention |
| Attention | `sol_attention` | Native sm75+ online Sol routing with FP16/BF16-PV or INT8-PV |
| Attention | `sla_attention` | Native sm75+ 128x64 fixed-Top-K SLA routing with FP16/BF16-PV or INT8-PV |

W4A4 contraction deliberately reuses Comfy Kitchen. W8A8 uses the local sm75+
contraction for its aligned BF16/FP16 contracts and otherwise returns to Kitchen's
normal dispatcher. The local package also supplies sm75+-capable quantization,
the BF16 epilogue, activation fusions, dispatch, and W4A8 contraction; it does
not duplicate the full W4A4 GEMM.

## Attention feature matrix

| Feature | Stable Sage | Dense W8A8 | Sol FP16-PV | Sol W8A8 |
|---|---:|---:|---:|---:|
| Production status | explicit | loader default | production patch option | production patch default |
| Input storage | FP16/BF16 | FP16/BF16 | FP16/BF16 | FP16/BF16 |
| QK score domain | INT8, FP32 accumulation | INT8, FP32 accumulation | INT8-consistent routing and score | INT8-consistent routing and score |
| PV path | FP16 Tensor Core, FP32 accumulation | U8 x S8 Tensor Core | FP16 Tensor Core | U8 x S8 Tensor Core |
| Output dtype | input V dtype | input V dtype | input V dtype | input V dtype |
| Head dimension | 1--128, native D64/D128 | 1--128, native D64/D128 | 1--128, native D64/D128 | 1--128, native D64/D128 |
| HND / NHD | yes / yes | yes / yes | HND only | HND only |
| GQA | yes | yes | yes | yes |
| Unequal Q/K length | yes | yes | yes | yes |
| Causal | yes | yes (upper-left) | no | no |
| Arbitrary attention mask | no | no | no | no |
| Variable length | yes | yes (packed) | no | no |
| Split prequantization | Q/K; V retained | Q/K/V | Q/K; V retained | Q/K/V plus summaries |
| Q/K Hadamard rotation | no | optional, default on | optional, default on | optional, default on |
| Adaptive K anchor | no | optional, default on | optional, default on | optional, default on |
| Exact modal KV ranges | n/a | n/a | yes | yes |
| Bundled architecture | exact sm75 | sm75 / Ampere / Ada / Hopper | sm75 / Ampere / Ada / Hopper | sm75 / Ampere / Ada / Hopper |

FP32 is accepted at the plugin boundary only for ComfyUI's Turing BF16
fallback case; it is converted to BF16 storage before entering these kernels.
The kernels do not advertise native FP32 attention.

SLA has the same fixed-shape, unmasked, non-causal, GQA, unequal-Q/K, D64/D128,
semantic exact-KV, and split-prequantization support as Sol. Its routing policy
is intentionally different: every 128-token Query block keeps a fixed fraction
of 64-token K/V blocks, adjacent Q64 execution CTAs share that route, and
unselected blocks contribute nothing. The retained route is a compressed
bitset; the centroid score tensor exists only while building Top-K and is
released before exact attention. Smooth-K's global K subtraction is omitted
mathematically because it adds the same constant to every candidate score in a
Query row and therefore cannot change Top-K ordering.

The D64 attention specializations use 16 KiB shared memory and D128 uses 32
KiB. The long-key 128-token schedule reuses a 64-token shared tile rather than
doubling the CTA allocation. Exact occupancy and throughput require a profile
on every accepted native architecture; one GPU's result is never reused as
another GPU's launch policy.

## Quantized linear support

| Format | Weight storage | Activation | Contraction owner | Local Turing additions |
|---|---|---|---|---|
| W8A8 | INT8 | INT8 | local sm75+ kernel for its exact contract; otherwise Kitchen/cuBLAS | sm75+ ConvRot quantization, fused SwiGLU/GELU, BF16 epilogue, workspace policy |
| W4A4 | packed INT4 | INT4 | Kitchen | sm75+ BF16 input/output compatibility and fused activation quantization |
| Legacy W4A8 | signed packed INT4 | INT8 | local sm75+ kernel | packed-weight shared-tile expansion, INT8 Tensor Core MMA, BF16 output |
| Grouped-codebook W4A8 | 4-bit codebook indices + E4M3 g16 scales + FP32 channel scales | INT8 | local sm75+ kernel | register decode directly into the shared W8A8 tile for long sequences, bounded staged fallback, BF16 output; covers the symmetric `asym_w4a8_int8` MiniMax-H3 files |

ConvRot group size 256 is the optimized H3 path. Unsupported layouts, group
sizes, devices, or dtypes are rejected or delegated to Kitchen according to the
format contract; dense checkpoint weights are never silently quantized at
runtime.

## MiniMax H3 activation scheduling

H3 QKV and FFN activation policy is based on current allocatable VRAM, not the
GPU marketing generation. It combines ComfyUI's live free/reclaimable value
with the hard `--reserve-vram` ceiling and leaves explicit allocator, output,
and per-layer weight headroom. The same decision function runs on every
supported Tensor Core GPU.

For the 15-second portrait workflow, the target sequence is 44,541 rows at
480x864 and 111,630 rows after the 2.5x-area latent upscale to 768x1376. Before
reference rows, the relevant operator-local BF16/INT8 estimates are:

| Target stage | Full QKV | Streamed QKV | Full FFN | Streamed FFN |
|---|---:|---:|---:|---:|
| 0.4 MP / 44,541 rows | 3.57 GiB | not selected | 3.87 GiB | not selected |
| 1.06 MP / 111,630 rows | 8.94 GiB | 3.86 GiB at 16,384 rows/tile | 9.69 GiB | 3.63 GiB at 32,768 rows/tile |

These are incremental operator buffers, not total process VRAM. Packed text,
keyframes, image/video/audio references add rows, while the model, current
hidden/residual, CUDA allocator, and compositor consume the rest. Because the
policy measures memory again inside each layer, those live allocations reduce
the selected tile automatically. Long or multiple reference videos can still
raise the irreducible INT8-Q/K plus BF16-V floor; cap reference duration or
resolution when the log reports a minimum tile near the budget.

The streamed FFN casts each weight once and processes all row tiles before
uncasting. Streamed QKV similarly casts the projection once, emits 64-row-
aligned INT8 Q/K scale blocks, and retains BF16 V. Bundled dense W8A8, Sol-W8A8,
and SLA-W8A8 consume this state directly, so the memory reduction does not
materialize a second full floating Q/K pair.

If the compact attention state is still over budget, the policy selects the
largest fitting whole-head group whose start and end feature offsets are both
divisible by the ConvRot-256 block. The group width need not divide the total
head count; a smaller aligned tail is valid. Each group contains the full
Query/Key/Value sequence;
there is no token-window approximation and no repeated global attention work.
The adapter calls the workflow-selected backend for every group, so an explicit
SDPA selection stays SDPA. Kernel 0.30's reusable global K-anchor and direct
Q/K output ABI keep row-streamed W8A8/Sol preprocessing equivalent to the full
sequence contract.

If even the minimum FFN row tile is constrained, W8 ConvRot fc1/fc2 can split
the 14,336-wide intermediate on aligned 256-channel boundaries. The first pass
computes the maximum of the per-interval scales, exactly recovering the full
row scale. The second pass quantizes every interval with that common scale and
writes it by offset into the final compact INT8 row. The original fused fc2
then performs one complete contraction and BF16 epilogue. This is the last
automatic tier because it recomputes fc1 intervals;
normal 12 GiB runs prefer the materially faster row-only path.

The resulting order is not a GPU-generation table:

| Ladder | Selected when | Preserved contract |
|---|---|---|
| L0 full | full activation has headroom | original throughput path |
| L1 rows | transient projection/FFN activation is too large | exact per-row linear and quantization math |
| L2 compact Q/K | prepared W8A8/Sol/SLA is active | global sequence anchor, INT8 Q/K and BF16/INT8 V |
| L3 heads | complete attention state is too large | full non-causal sequence and selected backend per head |
| L4 FFN channels | minimum practical row tile is still constrained | common scale, direct A8 offsets, unchanged fused fc2 |

Every decision uses current reclaimable memory after ComfyUI's
`--reserve-vram` ceiling. Native cubins specialize instructions for sm75,
sm86, and later architectures, while Python uses this one ladder for all of
them. The ladder has no Triton dependency; on Windows, a missing optional
Kitchen fixed-workspace entry point falls through to the bundled CUTLASS W8
contraction rather than a sequence-sized INT32 workspace.

The retained overlap accumulator follows the same sm75+ capability contract;
the H3 VAE decoder now uses native linear stitching instead.
Legacy `turing` and `_sm75` names are retained only as public/package ABI; CUDA
selects the architecture-specific cubin and instruction sequence internally.

## Production classification

"Production" means that the public shape/dtype contract is checked before any
input is consumed, fake/meta registration is available, Linux and Windows
build paths are gated, numerical references pass, and the exact-sm75 cubin has
no stack/local spill. It does not mean that one backend is fastest for every
sequence length or that an A40 PTX result can choose a Turing launch policy.

| Tier | Operator families | Runtime policy |
|---|---|---|
| Production default | Dense W8A8 attention; ConvRot W8A8/W4A8/W4A4 activation paths; BF16 epilogue; normalization fusions | Selected by the loader/adapter after preflight; tile choices are cached per device and contraction shape while the native cubin supplies architecture specialization |
| Production alternative | Stable Sage; SDPA FP16 bridge; legacy and grouped-codebook W4A8 | Explicit backend/weight-format choice with deterministic fallback |
| Production patch | Sol FP16-PV and Sol W8A8 | Enabled only by the independent Sol node so workflow-specific routing and modality controls remain explicit; W8A8 is the node default; sm75+ uses the same prepared sparse and bundled dense interfaces with a native cubin for the installed architecture |
| Production patch | SLA FP16-PV and SLA W8A8 | Independent fixed-Top-K node for SLA-trained weights; shares Sol's safety controls and the same bundled sm75+ dense path without changing SLA route semantics |
| Compatibility only | Staged codebook decode | Preserves grouped-codebook formats that cannot use the inline g16 decoder |

The resource release gate covers all compiled production and compatibility
families, not only attention: 51 core kernels, 48 native D64/D128 attention
variants, and 24 D64/D128 Q/K preprocessing variants in the current exact-sm75
image. All report zero stack/local memory. Register count and shared-memory use
are recorded rather than constrained to the obsolete "two CTAs per SM" rule;
the final choice is based on measured latency on the target device.

## Model-specific integration

- MiniMax H3 publishes packed text/image/video/audio token ranges, estimates
  packed dynamic-VRAM allocations, fuses segmented RMSNorm+AdaLN plus the
  preceding dtype-rounded gated residual (with a standalone fused final gate), and fuses
  fc1 -> SwiGLU -> fc2-input quantization. Supported attention calls also fuse
  per-head RMSNorm+split-half RoPE directly into INT8 Q/K.
- Wan publishes packed-context memory estimates and fuses its whole-row Q/K
  RMSNorm plus interleaved RoPE into Sage/W8A8/Sol Q/K quantization.
- Bernini provides conditioning, context-window memory estimation, and
  absolute-position integration. Its Wan self-attention inherits the same
  generic fused preprocessing and tensor-lifetime path, including explicitly
  selected Sol; it does not own
  quantized CUDA kernels.
- Video padding and H3 AV latent utilities are media operations, not CUDA
  kernel families.

## Optimization status

| Work item | Status | Decision |
|---|---|---|
| AttentionTensorContainer lifetime | complete | Dense Sage, dense W8A8, and Sol consume containers only after preflight; quantized Q/K replace floating Q/K before output allocation, W8A8 also releases floating V, while FP16-PV correctly retains V until the attention CTA consumes it |
| Fused Q/K RMSNorm + RoPE + INT8 quantization | complete | H3, Wan and Bernini publish model semantics through one adapter contract; dense Sage/W8A8 and explicit Sol share the same D64/D128 custom op |
| fc2/out_proj gate + residual epilogue | not implemented | high-value H3 adapter work, but requires a contraction-owned epilogue rather than another post-GEMM kernel |
| Head/layer/step-specific Sol policy | not implemented | useful only behind calibration and visual/audio gates; do not replace the stable global policy without H3 data |
| Route reuse and hysteresis | not implemented | potentially useful, but must justify route-state memory and avoid cross-prompt state leakage |
| Kitchen versus bundled native A/B | directionally tested on A40 | final policy requires matched sm75 and sm86 runs with native cubins |
| H3 trajectory block cache | experimental node | profile-driven standard, 4-step, and 8-step policies reuse one exact residual; sampler branches are isolated, skipped weights are omitted from Dynamic VRAM prefetch, and storage honors ComfyUI's GPU reserve/pinned-memory lifecycle; it remains opt-in because block skipping is inherently approximate |
| Spectrum-style transformer skipping | not integrated | replay memory and audio workflow cost conflict with the Turing/dynamic-VRAM target |

The next production-oriented order is: gated fc2/out-projection epilogues,
calibration tooling for head/layer/step Sol
budgets, then matched native sm75/sm86 backend A/B. The first two target tensor traffic in
the MLP/projection-heavy part of H3 and therefore have a higher whole-model
ceiling than another dense attention variant.

The current A40 environment could not load the installed Kitchen CUDA binary
because that binary requires a newer CUDA driver. The repeatable benchmark
reports this as an unavailable comparison instead of substituting a different
kernel. Bundled Sage was compared against external Sage and the historical
pre-refactor bundled image; the stable main loop stayed within about 0.1% of
the historical image. Matched native sm75/sm86 Kitchen/bundled A/B therefore
remains an explicit release-machine task, not an inferred result.

`kernel/scripts/benchmark_backends.py` is the repeatable backend regression
gate. It reports prequantized and end-to-end scopes separately for H3 QKV/fc1/
fc2 shapes, compares the two W4 formats with raw W8A8, Kitchen when its binary
is compatible, bundled/external attention implementations, and SDPA. Its
preprocess suite separately measures A8/A4 ConvRot quantization, fused
SwiGLU/tanh-GELU input activation, RMSNorm/LayerNorm+AdaLN, and the BF16
epilogue, so Python/eager regressions cannot be misreported as GEMM or attention
regressions. Build with `COMFYUI_TURING_UTILS_ARCH_LIST` set to every target
architecture before using the numbers for release decisions.
`benchmark_arch_matrix.py` runs identical arguments serially on all listed
devices and stores one JSON comparison artifact.
