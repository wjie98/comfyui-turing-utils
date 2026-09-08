# Capability-based CUDA runtime flow

The historical Turing path is now a generic runtime capability. Model adapters only connect
MiniMax or Wan/Bernini block structure to those capabilities.

```text
checkpoint metadata
  -> quantization summary
  -> resolve hardware facts and the compiled operator ABI
  -> preflight only the selected supported kernels
  -> on exact sm75, replace an unsupported-FP16 model's FP32 fallback with BF16
  -> ComfyUI constructs the ModelPatcher
  -> when BF16 was selected, normalize ConvRot logical dtype without copying packed data
  -> install optional model adapter object patches
  -> install attention override
  -> install model-local Turing Utils -> Kitchen operator selection
```

Diffusion and CLIP loaders prepare their own runtime. Loading a
text encoder therefore does not depend on a diffusion loader having run first.
An explicit ComfyUI dtype flag wins over automatic BF16 selection, but it does
not skip attention or quantized-kernel preflight.

## Ownership

| Module | Scope |
|---|---|
| `comfyui_turing_utils/precision.py` | BF16 selection and the exact-sm75 precision compatibility rule |
| `comfyui_turing_utils/runtime/` | device facts, real compiled-symbol capability resolution, and diagnostics |
| `comfyui_turing_utils/loading/` | ConvRot discovery, Comfy construction, preflight, and adapter installation |
| `comfyui_turing_utils/attention/` | prepared-attention protocol, stable Sage, sparse policies, semantic layout, and patches |
| `comfyui_turing_utils/nodes/attention.py` | thin production attention-strategy node schemas |
| `comfyui_turing_utils/quantization/` | sm75+ W8/W4 format inspection, dispatch, and generic fusions |
| `comfyui_turing_utils/adapters/minimax/` | MiniMax layout, packed-sequence planning, fusions, and H3 virtual-K/V policy |
| `comfyui_turing_utils/adapters/wan.py` | Wan/Bernini context-aware planning and Q/K preprocessing hooks |
| `comfyui_turing_utils/adapters/wan_layout.py` | Wan/Bernini semantic attention-layout provider |
| `comfyui_turing_utils/kernel_api.py` | sole lazy boundary to the independently installed kernel package |
| `kernel/csrc/turing` | historical source path for separately installed sm75+ kernels and exact-sm75 Sage |

The quantization scope is attached through ModelPatcher. It never changes
Kitchen's process-wide priority: a matching local implementation is selected
only while the loaded MODEL or CLIP is executing. Capability rejection happens
before the operation starts; execution errors propagate without retry.

## Linear matrix

| Weight/activation | Plain input | Fused activation input | Output |
|---|---|---|---|
| W8A8 | bundled BF16 row-buffer rotation and local aligned contraction; Kitchen handles calls outside the local contract | SwiGLU and tanh-GELU are folded into the same rotation/quantization decision | BF16 on the local contract; Kitchen owns other requested dtypes |
| W4A4 | bundled BF16 row-buffer rotation with Kitchen contraction; Kitchen owns calls outside the local contract | bundled staged/row-buffer SwiGLU or tanh-GELU produces packed A4 directly | original BF16 boundary |
| Legacy W4A8 | shares the W8 activation quantizer and consumes signed packed W4 directly | shares fused W8 SwiGLU/tanh-GELU quantization | BF16 |
| Grouped-codebook W4A8 | shares the W8 activation quantizer; long sequences decode packed g16 codebook values directly into the W8A8 shared tile, with a bounded staged fallback | shares fused W8 SwiGLU/tanh-GELU quantization | BF16 |

The BF16 row-buffer stores one completed rotated row as BF16 and keeps only
active FHT groups in FP32. Launch selection uses the device's opt-in dynamic
shared-memory limit rather than a fixed 48 KiB policy. Among the 512/768/1024
thread geometries that fit, dispatch estimates resident CTAs from the device's
shared-memory and thread limits and selects the geometry with the most active
warps for the actual row count. Resident CTA count is an input to that choice,
not a fixed acceptance rule. It does not allocate the full activated or rotated BF16
intermediate. A40 compatibility validation shows identical packed INT8/INT4
values versus the staged operators across K=256, 5376, 7168, and 14336; INT4
scale differences are at most normal FP32 reduction-order roundoff.

For the real H3 fc2 boundary (`M=4096`, raw SwiGLU width 28672, contracted
width 14336), an A40 JITing compute_75 PTX measured 0.589 ms for the row-buffer
A8 path versus 0.967 ms for the former staged path, and 0.530 versus 0.888 ms
for A4. Forced 512/768/1024 A8 geometries measured 0.599/0.775/0.767 ms and
produced identical packed values and scales. A40 selects 512 because its
100 KiB shared-memory pool can keep two such CTAs resident; exact Turing has a
different 64 KiB residency calculation, so its final geometry and throughput
still require an exact-sm75 measurement.

W4A8 reads packed W4 tiles directly, expands each vector once while filling
CUTLASS's SM75 crosswise INT8 shared-memory layout, and executes the contraction
with `m8n8k16` INT8 Tensor Core instructions. Scaling, optional bias, and BF16
storage stay in the epilogue. It never creates a persistent W8 weight copy and
rejects only tiles that exceed the actual per-device launch limit.
Legacy W4A8 edge dimensions no longer fall back to a full-matrix DP4A kernel.
K values divisible by four use relaxed, predicated global iterators while the
contraction remains `m8n8k16` Tensor Core MMA. For N tails, the aligned output
channels use the normal long-sequence kernel and only the final one-to-seven
packed rows are zero-padded into an eight-row temporary before a small Tensor
Core launch. No persistent expanded weight or full padded-weight copy is made.
On an A40 JITing compute-75 PTX, `M=4096,N=5376,K=5372` fell from 49.52 ms on
the former DP4A fallback to 1.21 ms on the predicated Tensor Core path. This is
a directional result; exact-sm75 acceptance still requires real Turing.

Since kernel 0.24, the runtime also accepts the symmetric `asym_w4a8_int8`
layout used by the current MiniMax-H3 experimental W4A8 checkpoints. Unlike
the legacy signed nibble format, each stored nibble is a codebook index and
must be combined with an E4M3 per-group relative scale before INT8 MMA. The
SM75 implementation loads one packed 16-value group per thread, combines it
with the E4M3 scale and codebook in registers, and writes the decoded values
directly into CUTLASS's normal crosswise S8 shared-memory tile. The
long-sequence path therefore has no decoded-weight workspace; short sequences
and non-g16 compatibility cases keep the bounded staged implementation.
Neither path expands the checkpoint at load time or creates an MxN INT32
accumulator. Asymmetric correction files remain a Kitchen fallback and are not
silently accepted by this path.

The benchmark uses H3's actual fc2 contracted width 14336 rather than treating
the full 28672-wide fc1 output as the fc2 contraction. On an A40 running the
compute-75/PTX schedule, the latest 4,096-row prequantized W8/legacy-W4/
codebook-W4 timings were 4.38/4.78/5.54 ms for QKV, 6.21/6.40/7.33 ms for
fc1, and 2.70/3.04/3.48 ms for fc2. At 8,192 rows they were
9.95/9.85/10.54 ms, 13.32/13.22/14.05 ms, and 6.13/6.29/6.75 ms. Codebook W4
is therefore a checkpoint-size and quantization-quality feature, not a claim
of higher contraction throughput than W8A8. Inline decode still removes the
112 MiB staged workspace at long H3 sequence lengths without creating a
persistent expanded weight. A synthetic checkpoint-format comparison measured
relative L2 0.07314 and cosine 0.99732 for grouped-codebook g16, versus relative
L2 0.16035 and cosine 0.98738 for the legacy row-scaled signed W4 format. These
are direction and format tests, not final exact-sm75 or model-quality acceptance
results. Final throughput acceptance still requires an exact-sm75 run.

The inline and raw-W8 long-sequence kernels use the same fixed 128x256x64,
256-thread CTA and shared-memory tile. The SM75 cubin reports zero local-memory
spill and one resident CTA; inline decode does not lower CTA density. The
staged fallback uses 4,096-output-channel chunks,
matching Kitchen's production chunk policy, and is bounded at 112 MiB for H3
fc2 and 21 MiB for qkv/fc1. MiniMax and Wan planning remain conservative for
short or non-g16 inputs and reserve only the largest mutually exclusive staged
workspace.

The staged codebook decoder packs all 16 output bytes in registers. Its exact
SM75 image uses 28 registers, 64 bytes of static shared storage, and zero stack
or local memory; the previous addressable temporary arrays required a 16-byte
per-thread stack frame. The compatibility path remains bit exact with inline
g16 decode, including predicated N edges.

Raw W8, signed W4, and long-sequence codebook W4 use deterministic dispatch.
SM75 uses the fixed 128x256x64 two-stage Tensor Core schedule. Native sm80+
uses the matching three-stage schedule for wide QKV/FC1 outputs and the proven
lower-overhead SM75 schedule for narrow/deep FC2 and output projections. The
runtime performs no first-use probes, synchronization, process caching, file
I/O, or persistent schedule selection, so CUDA Graph capture follows the same
path as normal execution.

The stable Sage main loop was also rebuilt from historical commit `4255f3c`
in an isolated worktree and compared with the current compute-75 image on the
same A40 tensors. Old/current timings were 0.7685/0.7677 ms at N=4096,
2.8179/2.8197 ms at N=8192, and 10.9173/10.9179 ms at N=16384. The maximum
difference is about 0.1%, so an observed whole-workflow regression should be
profiled in projection, Q/K/V preparation, model patching, or VRAM movement;
it is not explained by a slower bundled stable-Sage CUDA main loop.

For W8A8 GEMM, the bundled CUTLASS contraction is first choice for its aligned
BF16 sm75+ contract. Calls outside that contract are handed to Kitchen before
execution; Kitchen may select its fused CUDA path, cuBLAS, Triton, or eager
implementation according to its own registered capabilities. This selection is
not an exception-driven fallback and has no effect on models loaded without a
Turing Utils loader.

Wan projections, block normalization, attention dispatch, and feed-forward
execution stay on ComfyUI's native path; current ComfyUI folds tanh-GELU through
`linear_input_act` without replacing the block forward. The adapter only adds
context-aware memory planning and therefore does not alter Wan model numerics.
Bernini context latents are included in ComfyUI's memory estimate; context
windows budget both the context tokens and the possible causal anchor without
changing the conditioning used at runtime.

MiniMax planning receives the sampler's original nested video/audio shapes
before ComfyUI flattens them. It mirrors `PackedLayout` row accounting for text,
first/last keyframes, reference images, reference video, reference audio, and
the target streams. The normal ComfyUI heuristic is scaled by the complete
packed sequence, with a lower bound from the known FP32 condition-row buffers
and BF16 packed hidden state. W8A8 planning checks every distinct output width
and reserves the largest live INT32 accumulator; it does not assume that the
widest layer owns the largest workspace after fixed-workspace dispatch.

H3 activation execution follows the same live-memory accounting. It first
retains the unmodified full path, then reduces only transient state in this
order: row-streamed QKV/FFN, prepared compact Q/K, whole-head attention groups,
and finally aligned FFN intermediate groups. This is a resource ladder rather
than a Turing/Ampere branch. ComfyUI's immediately usable bytes and
`--reserve-vram` ceiling select the highest-throughput fitting rung at every
operator call; inactive-model residency is reserved for emergency headroom,
not tier promotion.

DynamicVRAM budgeting distinguishes immediately usable memory from evictable
model residency. Evictable pages do not promote a faster execution tier:
releasing current DiT weights to reduce exact head sharding causes those pages
to be loaded again by the next layer and loses more time than it saves.
Resident, unpinned pages from inactive model VBARs are therefore only an
emergency reserve for a tier already selected from immediately usable memory;
the current diffusion VBAR is never explicitly reclaimed by this policy. This
uses the quiet residency bitmap instead of `vbars_analyze`, so normal runs do
not emit one warning per pinned page.

The activation budget also protects the two asynchronous weight-offload
streams. For a dynamic MiniMax model it estimates two average transformer
blocks from the current patcher's staged model bytes and installed layer
count, aligns that value to the 32 MiB VBAR page, and bounds it to
512 MiB--1 GiB. This keeps
the current/prefetched weights resident after an activation shard has already
reached the modeled MFU plateau without introducing an Ampere/Turing policy
branch.

The sampler low-water mark is keyed by operation as well as sequence shape.
Attention, QKV projection, and MLP observe different live-buffer boundaries;
a low pre-attention reading must not force the later MLP to retain row
streaming after QKV and attention temporaries have retired.

Automatic splitting is saturation-first rather than capacity-first. For
attention, the minimum group must expose at least four CTA scheduling waves,
at least 1024 QKV output channels, and no more than four balanced head passes.
The smallest legal ConvRot-aligned group meeting those constraints becomes the
`saturation_group`; a larger fitting group is not selected unless the complete
unsharded lifecycle fits with its normal safety margin. QKV and MLP row tiles
cap at 16K, and FFN channel splitting similarly targets no more than four
balanced, 256-aligned groups. Once a dynamically paged QKV projection spans
four such tiles, `auto` retains the saturated 16K path even if a momentary
free-memory reading says the monolithic projection fits. The full QKV tensor
adds no useful parallelism at that size, makes V preparation more expensive,
and can displace the two-stream weight prefetch window. This rule depends on
sequence geometry and DynamicVRAM residency, not on an Ampere/Turing device
name; explicit `throughput` mode still forces the full path for profiling.
These tile sizes already expose much more Tensor Core work than can run
concurrently on current supported GPUs, so preserving hot model residency
takes precedence over enlarging their transient state.
The packed-sequence memory estimate uses the same 16K MLP tile and balanced
head group, preventing ComfyUI from pre-evicting weights for an obsolete,
larger transient estimate.

When the current DiT uses DynamicVRAM, head sharding also omits the optional
full-sequence quantized-input cache. Row quantization is repeated inside the
balanced groups, but the smaller working set retains roughly one additional
hidden tensor of hot weights and avoids a substantially costlier PCIe reload.
Explicit chunk/head/channel overrides remain available for measurement.

QKV row tiles write directly into their final INT8 Q/K and BF16 V storage.
Kernel 0.32 accepts a reusable K anchor computed from the same nine positions
in the complete sequence, so streaming no longer disables adaptive anchoring.
If attention must be split further, only whole heads are grouped; every group
still contains every token and calls the workflow-selected backend. Legal group
widths are common boundaries of the model head layout and ConvRot's 256-value
blocks. This preserves non-causal global attention and does not repeat any Q/K
pair within a head.

The last FFN rung is restricted to W8 ConvRot fc1/fc2. Each 256-aligned channel
interval is projected twice: the first pass reduces its local maximum into the
same whole-row activation scale used by the complete 14,336-channel tensor;
the second quantizes with that common scale and writes directly into its offset
in the complete compressed INT8 activation. The original fused fc2 then runs
once, including its normal BF16 epilogue. The A8 row is smaller than an INT32
output accumulator for H3, so this removes full fc1/SwiGLU intermediates
without changing the contraction, but costs extra fc1 work and is selected only
after row streaming cannot satisfy the budget or when explicitly diagnosed.

Kernel 0.32 extends the W8 and scaled-SwiGLU ABIs with row-strided output
views. Gate/up channel intervals, the final compressed A8 interval, and the
fc2 row interval therefore land at their final offsets instead of allocating
then copying per-shard tensors. Older kernels retain the same numerical path
through an allocate-and-copy compatibility fallback.

Wan/Bernini reference shapes are padded per stream before aggregation. The
outer sampling wrapper also supplies the real sampler batch, so repeated
context conditions are represented during initial model loading as well as in
the per-step batching check.

Wan patch embedding deliberately retains ComfyUI's FP32 convolution boundary.
An A40 test with TF32 disabled found the prospective FP16 path slower and
observed a non-zero BF16 output delta, so it is not installed as a speculative
Turing optimization. The reproducible comparison is available through
`validate_wan_fusions.py --patch-embedding`.

## Attention matrix

The loader exposes exactly `w8a8`, `sage`, and `sdpa`, with W8A8 selected by
default. W8A8 uses the bundled implementation on sm75 and newer GPUs. Sage is
bundled on exact sm75 and delegates to ComfyUI's registered SageAttention
function elsewhere. `auto` is no longer accepted. Legacy serialized
`sage_attn`, `sage_`, `sage_hybrid`, and `turing_sage` values normalize
invisibly to `sage` and are not displayed by the loader.

| Option | Q/K path | Smoothing | PV path |
|---|---|---|---|
| `sage` on Turing | INT8, per-16-token Q-warp scales | disabled | FP16 V tiles with direct FP32 accumulation |
| `w8a8` on sm75+ | stable-Sage INT8 score domain | optional adaptive K anchor | channel-wise signed INT8 V and unsigned INT8 probabilities, INT32 Tensor Core PV, FP32 online state |
| `Configure Sol Sparse Attention` | fused 64-token centroid routing; selected tiles reuse stable Sage INT8 QK | input-adaptive `mean + tau * std` threshold | inherited W8A8 exact PV, or FP16 exact V tiles for Sage/SDPA; skipped-block V centroids and FP32 online accumulation |
| `Configure SLA Sparse Attention` | one route shared by adjacent Q64 CTAs (logical Q128 x K64); selected tiles reuse stable Sage INT8 QK | fixed Top-K budget; Smooth-K-invariant ordering | inherited W8A8 PV or FP16 selected tiles; no skipped-block residual |
| `Configure H3 Image Sol Attention` | native H3 Query/FFN rows with either an initial Dense window or the H3 VAE Dense anchor grid | H3 semantic prefix plus fixed `1x64` residuals outside Exact/local ranges | inherited W8A8 exact PV or FP16 exact V tiles for Sage/SDPA; reference sparsity and Dense step/layer schedules match Sol |
| `Configure H3 Static Virtual KV` | physical Query remains two H3 latent-time slices; conservative materializes seven BF16 K/V slices, kernel 0.39 fast maps two physical slices into seven exact logical INT8 K/V slices, and kernel 0.41 residual keeps the first two slices exact while routing the five virtual slices as Sol residuals | all seven real temporal RoPE phases; residual uses 2x32 summaries with exact physical-context ranges | W8A8 maps INT8 V; inherited Sage/SDPA map physical FP16/BF16 V through the bundled Sol residual kernel; an upstream Sol/SLA strategy is replaced |

Integer Q/K MMA accumulates into INT32. The stable facade supports FP16 and
BF16 Q/K/V, HND/NHD, GQA, causal mode, unequal Q/KV lengths, head dimensions
through 128, and variable-length batches. BF16 V is converted tile-by-tile
while loading shared memory, so no full V conversion tensor exists. FP32 Q/K/V
use one BF16 boundary conversion and restore FP32 output. Explicit SDPA instead
consumes BF16 containers and converts Q, K, then V to FP16 before PyTorch
dispatch on exact sm75, avoiding the BF16 math implementation while bounding
overlapping storage; its output is restored to BF16.

When either logical sequence is shorter than the 64-token SM75 CTA, the facade
uses a bounded exact FP32 SDPA path. It contains fewer than 4096 scores per head
and cannot reproduce the large-sequence SDPA allocation failure.

The ConvRot loader installs a stable legacy/container runtime dispatcher and
records its dense backend as an immutable capability. Each ModelPatcher branch
binds the resolved prepared executor directly, so fused row/head streaming does
not depend on a second runtime lookup. Independent `Configure Sol/SLA Sparse
Attention` and model-adapter strategy nodes change the strategy configuration
only; their routing,
modality, step, layer, and residual-quality controls do not invalidate the
large loader node. A model from another loader is bootstrapped from its current
attention override, with SDPA as the conservative default.

The same configured model may feed multiple full- or partial-denoise samplers.
Dense prefix/suffix counts are evaluated against each sampler's own sigma list,
so every invocation starts at its local step zero. No global invocation counter
is used and Sol/SLA alternation does not stack attention overrides.

SLA is likewise installed only by its independent patch node and requires
kernel 0.29.1. It shares the Sol layout contract, dense schedules, fused Q/K
preprocessing, GQA and unequal-Q/K handling. It differs at the route itself:
128-token Query centroids select a fixed Top-K set of 64-token K blocks and the
two corresponding Q64 execution CTAs consume one compressed bitset. There is no
local-radius union and no centroid contribution for skipped values. This keeps
the runtime aligned with SLA-trained weights rather than turning SLA into a
second Sol profile.
The official-style SLA defaults apply the 85% sparse route to every denoising
step and transformer layer: all four dense prefix/suffix step/layer counts are
zero. Dense scheduling remains available only as an explicit compatibility or
quality safeguard.
Dispatch depends only on the attention call: matching
FP16, BF16, or FP32 Q/K/V; head dimensions 1--128; unmasked non-causal attention;
and both Q and K meeting the configurable minimum sequence length. HND and
ComfyUI's unreshaped layout, GQA, unequal Q/K lengths, and incomplete final
blocks are supported. Other calls use bundled stable Sage without model-family,
sampling-step checks. A model adapter may publish semantic layer and topology
metadata; unknown models remain fully generic.

The bundled `w8a8` backend and the inherited Sol/SLA W8A8 path require kernel 0.23.0.
They support sm75 and newer native cubins and head dimensions 1--128. Dense W8A8
supports fixed HND/NHD and native packed-varlen inputs, GQA, unequal Q/K
lengths, and an upper-left causal diagonal; arbitrary masks remain unsupported.
Sol remains fixed-shape, unmasked, and non-causal. The W8A8 path
keeps the same Q64 tile shape as Sol: native D64 uses 16 KiB shared memory and
native D128 uses 32 KiB. V is quantized once per call into a channel-major,
16-token-permuted signed-INT8 tensor;
softmax probabilities are packed to unsigned INT8; PV uses SM75 U8xS8 Tensor
Core MMA and the output remains FP32 until normalization and dtype writeback.
The route-free dense specialization omits centroid summaries and route state.
Short calls can lose to stable Sage because the extra V scan is not amortized;
H3's long packed sequences are the intended W8A8 workload.

For packed varlen, Q/K/output and cumulative sequence metadata remain compact;
the implementation does not pad every sequence to the batch maximum or launch
one attention kernel per sequence. Q/K Hadamard rotation is fused into their
packed quantizers. V uses a per-sequence, per-head, per-channel signed-INT8
scale and pads each sequence internally by at most 63 tokens, preserving
aligned 128-bit V tile reads without a `batch * max_length` allocation.
Adaptive K-anchor subtraction is intentionally unavailable in this contract
because finding an anchor would add a separate per-sequence scan.

Kernel 0.20.0 provides the split prequantize/execute ABI used by current ComfyUI's
`AttentionTensorContainer`. Q/K quantization, optional V quantization, and Sol
correction summaries are completed before allocating the output. The original
Q/K/V tensors are then released; stable Sage and FP16-PV sparse paths retain
only the contiguous V buffer required by the main kernel, while W8A8 retains
only quantized V. Older kernels remain supported through the one-call ABI.
All bundled attention kernels launch on PyTorch's current CUDA stream, which
also makes the graph-leaf dense Sage/W8A8 operations safe for CUDA Graph
capture. Logical CTA-K scheduling is selected automatically from device
resources, while fused Hadamard Q/K and adaptive K anchoring stay enabled on
quantized production paths.

Kernel 0.22 extends that lifetime contract upstream into model-owned Q/K
preprocessing. MiniMax H3 passes raw projected Q/K together with per-head
RMSNorm and partial split-half RoPE semantics; Wan and Bernini pass whole-row
RMSNorm and full interleaved RoPE semantics. One custom op reads each raw Q/K
tile once, keeps it in CTA shared memory, and emits the exact INT8/scales layout
consumed by dense Sage, dense W8A8, Sol FP16-PV, and Sol W8A8 for supported H3
and Wan/Bernini self-attention calls. It therefore
avoids materializing normalized/rotated BF16 Q/K. Protected Sol steps and layers
use the corresponding dense finalizer without repeating preprocessing.
Exact-sm75 uses the bundled quantizing finalizer. Ampere and newer external
Sage, plus explicit SDPA, use a floating prepared finalizer which applies the
same protocol RMSNorm/RoPE transform to the existing projected Q/K and hands
those tensors directly to the registered ComfyUI backend. Rejection happens
before tensor ownership transfer, so protocol-incompatible causal calls and
unsupported dtype, device, or RoPE layouts retain the original-model fallback
without partial consumption. Masks are forwarded to the selected dense backend.
The handoff is installed through a model-neutral attention-site registry, so
an explicit Sol patch on an official ComfyUI-loaded H3 or Bernini model can use
the same fused path as the ConvRot loader. Capability rejection occurs before
Q/K/V ownership transfers; a rejecting backend is forbidden from consuming a
tensor and the model safely retains its original attention path.

Kernel 0.38 generalizes that preprocessing ABI so Query and Key may have
different sequence lengths and independent RoPE tables. The CUDA operation has
no H3 frame arithmetic or duplication policy. MiniMax's adapter supplies the
H3 target segment and virtual temporal positions; Bernini or another model can
reuse the same ABI with its own semantic-layout adapter.

Kernel 0.39 adds an optional logical-to-physical source map. The fused Q/K
preprocessor reads physical K through that map while applying the logical RoPE
table, and the W8A8 value path gathers the already-quantized channel-major V
container with its native 16-token permutation. Thus fast virtual K/V avoids
expanded BF16 storage without changing the seven-slice attention problem.

Kernel 0.41 completes Sol's logical-to-physical V map. H3's `residual` mode
marks the prefix, both physical target slices, and any suffix/reference context
exact. The remaining five virtual target slices are represented by two
32-token residuals per skipped K block and merged into the exact blocks by
Sol's existing online softmax. W8A8 maps the prequantized INT8 V container;
Sage/SDPA-derived policies construct the same summaries from physical
FP16/BF16 V and map every selected exact-V tile load. `sdpa` is the inherited
numeric and exact-fallback policy, not a claim that PyTorch SDPA implements
Sol residuals. The route still retains its one-block local safety neighborhood;
there is no post-attention averaging or separate dense-plus-residual output
blend.

The compute_75 cubin reports at most 21,128 bytes static shared memory and 76
registers/thread for D128 preprocessing, or 10,632 bytes and 59
registers/thread for D64, with no local spill. An A40 PTX direction test at
BF16 B1/H8/N8192/D128 measured 0.25 ms and about 16 MiB peak allocation for the
fused path versus 2.44 ms and 112 MiB for an unfused PyTorch reference. This is
not a comparison against Kitchen's fused model op and does not replace an
exact-SM75 end-to-end profile.

Kernel 0.23 separates Sol's route-statistics basis from its exact score basis.
The selected-block QK and skipped-block correction MMA remain post-Hadamard and
therefore reuse W8A8's exact INT8 representation. Only the Q/K block centroids
used by the diagonal `mean + tau * std` threshold are inverse-transformed to
the pre-Hadamard basis. D64 needs one cross-warp butterfly and D128 needs two;
they reuse the existing route scratch. No complete Q/K tensor is reread, no
route map is materialized, and dynamic shared memory remains 16/32 KiB. One
separate FP16 K centroid is retained for `1x64`, because the post-Hadamard copy
continues to serve skipped-block correction.
At N=4096, Hq=8/Hkv=4, D128 BF16 on the A40 compute_75 path, five 500-call
runs measured -0.45% for FP16-PV and +0.19% for W8A8 versus the old rotated
threshold domain, with identical selected-block counts; both are measurement
noise rather than observable overhead. Selected-block outputs are unchanged
when every block is exact. The route words are explicitly scalarized into four
registers per lane; the resource gate verifies zero local/stack spill after
enabling the transform.

Optional phase timing is process-local and disabled unless
`COMFYUI_TURING_UTILS_PROFILE_CALLS` is a positive integer. The disabled path
creates no CUDA events and performs no synchronization.
On DynamicVRAM, reaching the call limit only marks the report pending. Event
timings are read after the existing outer sampler fence, avoiding a mid-model
synchronization that would otherwise interrupt asynchronous weight prefetch.
Non-DynamicVRAM execution retains the bounded end-of-window synchronization.

Profiling is bucketed by operation, shape, and execution path. The default
four buckets can capture both H3 resolutions and both attention/MLP phases in
one run; `COMFYUI_TURING_UTILS_PROFILE_BUCKETS` changes that bound. Sparse
selected/possible block counts stay as device scalars until the same outer
fence, so route-density diagnostics do not add a hot-path `.item()`.

`COMFYUI_TURING_UTILS_TIMELINE=1` adds an outer-sampler timeline without an
extra sampler synchronization: it records CUDA elapsed time, wall time, their
host/transfer residual, allocator start/end/peak, reserved memory, and
DynamicVRAM reclaim counts at the already-required sampler fence. Combined
with bounded phase profiling, this separates attention/MLP kernel time from
weight waits and host or storage stalls across the low- and high-resolution
samplers. It also emits named latent-upscale and visual/audio reference-encode
spans; because these operations have no existing outer fence, timeline mode
deliberately synchronizes around those diagnostic-only spans. The default path
still creates no events or synchronization.

When channel pressure requires FFN sharding and ABI 0.33 is available, the
runtime uses a single-pass half-width path. FC1 gate channels are written once
into an `F`-wide buffer; aligned up-channel shards are consumed immediately by
the same FP32 SwiGLU and ConvRot arithmetic and overwrite the corresponding
gate interval. A final reduction over all channel partials produces the same
whole-row scale and INT8 activation as the full `2F` fused path. This replaces
the older exact two-pass FC1 recomputation with roughly `F + channel_chunk`
BF16 staging. The two-pass implementation remains the compatibility fallback
for older kernel packages.

Kernel 0.31 makes architecture validation part of the same bounded report. The
Python summary publishes the architectures embedded by the wheel build and
whether the active device has an exact entry. The CUDA launch report queries
the specialization selected for the real attention call and prints
`binary_sm`, `ptx_compute`, registers per thread, static/dynamic shared memory,
local memory, active CTAs/warps per SM, and occupancy. This distinguishes an
exact sm86 cubin from a stale sm75-only/PTX install and exposes resource-limited
long-sequence schedules without changing the mathematical path. These resource
queries run once per selected specialization and only while the profiler is
enabled.

The node keeps the measured 4096-token crossover internally; shorter calls use
stable Sage. `routing_threshold=1.0` matches the official mean-plus-one-standard-
deviation policy. Lower values preserve more exact blocks. The local safeguard
is fixed to +/- one 64-token block and is no longer exposed. Density bounds and
frame-distance temporal protection were removed from the complete Python/CUDA
path.

`skipped_residual=1x64` is the official-style fast default. `2x32` changes only
the skipped-block reconstruction; it deliberately shares the identical route.
`dense_prefix_steps=1`, `dense_suffix_steps=0`, `dense_prefix_layers=2`, and
`dense_suffix_layers=0` match the default protection policy for every sampler
invocation. One Sol-configured model branch can therefore feed both stages of a
4+2, 6+2, or 8+2 graph without pass-specific controls.
Every dense step or layer calls the loader-selected protected backend directly:
route-free bundled W8A8, stable bundled Sage, or SDPA. Sage/SDPA select Sol's
floating FP16 sparse core, while W8A8 selects its integer PV core. If the prefix
and suffix layer counts sum to at least the runtime layer count, every valid
layer takes this direct dense path and no Sol summaries or routing are built.

The common layout contract contains contiguous semantic segments. MiniMax H3's
adapter publishes text, keyframe/reference image, reference-video first/last
latent-frame anchors, reference-video interior, reference audio, target audio,
and target video ranges from the runtime `PackedLayout`.
The three reference switches independently decide whether those reference
Query and KV blocks may be sparse. Defaults are image=false, video=true, and
audio=false. Reference-video anchors follow the image switch, while the clip
interior follows the video switch. A disabled switch makes that modality's Query block exact and its
KV block an exact sink for every sparse Query. Target video is sparse; text and
target audio remain protected. Non-aligned boundaries conservatively round
outward to complete 64-token blocks. Missing or inconsistent required H3 layout
metadata selects stable Sage.

The CUDA kernel builds no global route map. Query/key/value summaries remain
separate compact preprocessing tensors, while threshold routing executes inside
each sparse Query CTA immediately before skipped-residual and selected-block
online-softmax updates. The temporary route occupies CTA-local shared memory. A
single lane counts and compacts selected blocks into ascending 16-bit indices
for bounded diagnostics, while the W8A8 exact path walks the same shared bitset
in ascending order. This removes the former four route words per lane and their
sm86 local-memory spill without changing selected-block or online-softmax
order. The FP16-PV path keeps its route in registers because its 16 KiB value
tile reuses the route arena after routing. The normal D64 16 KiB or D128 32 KiB
shared-memory tiles are reused. The kernel accepts at
most 4096 K/V blocks
(262144 tokens) per call.

Sol derives Q/K centroids from the same prequantized INT8 tensors and scales as
selected-block Sage. The K/V preprocessing scan produces one or two such K
centroids and matching original V means for skipped-block reconstruction.
Selected blocks use stable Sage's per-16-row Q and per-64-row K INT8 scales and
SM75 integer Tensor Core QK. By default PV retains the established FP16/FP32
behavior. Optional W8A8 uses signed INT8 V, unsigned INT8 probabilities, INT32
Tensor Core PV, and FP32 online state; skipped residual V centroids remain
original-value FP16. K and, when enabled, V are quantized once per call and
shared by sparse and dense Query blocks.

The Query CTA derives the route threshold from its INT8 Q tile and expands that
tile into resident FP16 shared storage for skipped-block correction. One
Q-to-K-centroid Tensor Core traversal supplies both the routing score and the
online-softmax correction, with conflict-free per-warp shared partials instead
of shared atomics. The route count remains in shared memory while the W8A8
exact traversal reads the shared bitset and the arena is reused for exact K/V
tiles. Keeping both the FP16
correction operand and the INT8 exact operand live would raise D128 shared
storage above 32 KiB, so the production kernel deliberately re-reads the small
INT8 Q tile instead of reducing CTA residency. Exact-Q staging is only 8 KiB
per CTA and is normally L2-hot after routing.

The official-style `1x64` and quality `2x32` residual paths, plus the 64- and
128-token exact-K staging paths, are separate compile-time specializations.
This removes runtime loop bounds and second-stage state from the default
long-sequence kernel without changing the selected route or arithmetic order.
A correctness gate requires bitwise-identical output between K64 and K128
staging. The 0.33 compute_75 image contained 24 variants for each native head
dimension with unchanged 16 KiB D64 / 32 KiB D128 dynamic shared memory. The
0.37 dual-architecture build contains 38 reachable/compatibility variants per
dimension because the host dispatcher must carry both baseline and SM80+
specializations in one fat binary. The resource gate allows only the known
16-byte CUDA argument frame, rejects true LOCAL storage, and reports register
residency rather than prescribing occupancy. Final resident-CTA throughput
still requires real Turing profiling.

Kernel 0.37 keeps that mathematical contract while removing two routing-side
bottlenecks. Aligned residual ranges are formed with warp ballots instead of
per-block shared atomics, and route words are compacted by stable warp prefix
scans rather than a serial lane. The resulting K-block list is still strictly
ascending, so exact attention and online-softmax accumulation keep their prior
order. Native SM80+ builds additionally select a 20/40-KiB D64/D128
precomputed-summary W8A8 specialization that pipelines summary and selected-V
loads with `cp.async`; exact SM75 continues through the common synchronous
fallback and unchanged 16/32-KiB layout. Dispatch is capability-based and does
not introduce model, GPU-product, or tuning-cache policy branches.

On an A40 exact-sm86 A/B run using identical BF16 inputs (`N=32768`, `H=8`,
`D=128`, 16.12% selected route), the same current kernel with the SM80+
pipeline compile-disabled measured a 7.29 ms median, versus 6.82 ms with the
pipeline enabled (about 6.5% faster). K64 pipelined and K128 baseline staging
remained bitwise identical. This is a directional Ampere kernel measurement,
not a claim about full H3 workflow time or GA104 clock/residency behavior.

With the 0.23 causal/varlen specializations, compute_75 reports zero local and
stack storage for all dense W8A8 variants. D128 uses 180 registers for fixed
non-causal, 244 for fixed causal, 245 for packed non-causal, and 246--247 for
packed causal; D64 stays at or below 175. All retain the existing 32/16 KiB
dynamic shared-memory budgets. On an A40 executing the compute_75 path, a BF16
GQA batch with Q lengths 3072/4096, K lengths 3201/4096, Hq=8, Hkv=4, D128
measured 1.145 ms packed versus 1.253 ms for two fixed calls (non-causal), and
0.700 versus 0.765 ms (causal). Packed V quantization measured 0.075 ms, down
from 1.146 ms in the discarded serial prototype. These are direction tests,
not a substitute for exact-sm75 profiling.

On an A40 JITing compute_75 PTX, four-head FP16 synthetic tests at threshold 1.0
selected 19.4%, 17.8%, and 16.6% of blocks at 4096, 8192, and 16384 tokens.
After residual and K-stage specialization, Sol-W8A8 `1x64` measured 0.235,
0.519, and 1.526 ms versus dense Sage at 0.405, 1.431, and 5.352 ms (1.72x,
2.76x, and 3.51x). `2x32` FP16-PV measured 0.232, 0.607, and 1.996 ms. An
H3-like BF16 shape with 56 heads and 52,842 tokens measured 176.3 ms at 16.2%
route density, down from about 186.8 ms before both compile-time
specializations. These are directional A40 compute_75/PTX compatibility
results, not Turing end-to-end or visual-quality measurements.

Kernel 0.21 adds native D64 dense W8A8 and Sol kernels rather than padding D64
to D128. On the same A40 compute_75/PTX directional run at BF16, N=4096,
Hq/Hkv=8/4, the prequantized dense core measured 0.409 ms for native D64 versus
0.866 ms for the same input zero-padded to D128; Sol-W8A8 measured 0.166 versus
0.257 ms. D64 uses 16 KiB shared memory and remains below the D128 register
footprint. Final speed and CTA residency still require exact-sm75 measurement.

For the new W8A8 path, an H3-like BF16 shape (`N=52,842`, 56 heads,
`threshold=1.0`, 15.9% route density) measured 715.5 ms for route-free dense
W8A8 versus 765.2 ms for stable Sage, and 220.9 ms for Sol-W8A8 versus 282.8 ms
for Sol with FP16 PV. Thus V quantization is amortized at the intended long
sequence: dense W8A8 was 1.07x faster and Sol-W8A8 was 1.28x faster than its
FP16-PV counterpart. At 4k--16k tokens the extra V scan can instead make W8A8
slower; this is the main reason to retain Sage as an explicit alternative to
the W8A8 default.

Kernel 0.22.3 removes an unintended runtime two-stage loop from route-free
dense W8A8 while retaining CTA-K64/128 staging for Sol. On the same A40
compute_75 direction check at BF16, `N=53,192`, and 56 heads, the prequantized
W8A8 core measured 696.3 ms versus 774.1 ms for stable Sage (1.11x). Before
the fix the dense core measured about 946 ms. This is still an A40 directional
test; exact-sm75 end-to-end throughput remains the acceptance criterion.

Stable Sage deliberately remains CTA-K64. A CTA-K128 experiment kept roughly
the same theoretical active-warp count, but the upstream kernel has no
cross-K-warp merge for its online-softmax state: its output cosine versus
CTA-K64 was only about 0.50, and it was 10--20% slower in A40 compute_75/PTX
tests. The kernel now rejects any future multi-K-warp instantiation at compile
time. Route-free dense W8A8 also deliberately uses one compile-time K stage;
its public 64/128 route tile setting applies to Sol routing and does not create
a slower dense runtime staging loop. The common benchmark reports both Sage
and W8A8 prequantized cores so preprocessing cannot hide this distinction.

The Sage1 and Sage2 adaptations produced severe block artefacts and black
flicker in local Turing tests. They are unstable experiments, not production
fallbacks. The loader, public package, default bindings, and default template
instantiations exclude them. Their complete checkpoint and reproduction steps
are documented in
[`kernel/experiments/turing_sage_variants`](../kernel/experiments/turing_sage_variants/README.md).

On sm75 and newer GPUs, W8A8 uses the same bundled prepared-attention path and
the loaded cubin supplies the architecture specialization. Sage remains exact-
sm75 and uses the registered SageAttention function elsewhere; SDPA uses
ComfyUI's PyTorch implementation. Flash Attention is not a loader option. An
all-FP32 call that cannot enter external Sage uses ComfyUI's PyTorch attention
implementation deterministically. Sol and SLA share bundled W8A8 for protected
dense steps/layers, so Q/K preprocessing and activation-streaming ownership do
not fork at the Turing/Ampere boundary.

The loader log reports `w8a8 via bundled_turing_w8a8` or
`sage via bundled_turing_sage` for local SM75 implementations. Sol logs the
resolved protected ranges, three reference switches, threshold, residual
profile, and fixed local radius. MiniMax additionally emits its fused block/MLP
dispatch counters.

`debug_route_density` is disabled by default. With kernel package 0.23.0 or
newer, the already-running sparse CTA accumulates one selected-block counter;
there is no route allocation or popcount kernel. Counts remain on-device across
layers and synchronize once for the end-of-step log. The log reports selected
and possible blocks, min/mean/max layer density, sampling step, layer range,
protected Query count, and residual profile. Debug-off adds no counter atomic,
event, synchronization, or allocation.

The final A40 compute-75/PTX regression sweep keeps preprocessing and core
attention separate. At N=4096/H56/D128, end-to-end Sage/W8A8/SDPA/external-
Sage measured 4.67/5.47/4.39/4.63 ms, while the already-prequantized bundled
cores measured 4.54/4.32 ms. At N=8192 they measured
18.68/19.65/18.04/17.10 ms end to end and 18.39/16.51 ms prequantized. This
confirms that W8A8's INT8 PV core is faster but its extra V quantization is not
amortized at short sequences; the long H3 measurements above are the intended
default workload. It also prevents a core win from being mislabeled as an
end-to-end win.

The same sweep measured fused H3-like fc2 SwiGLU ConvRot input preparation at
0.65 versus 1.04 ms staged for 4,096 rows and 1.17 versus 1.92 ms for 8,192
rows (about 1.6x). The packed BF16 epilogue measured about 5x faster than its
eager reference. These are the retained optimizations. Grouped-codebook W4 is
retained for checkpoint size/quality despite being 6--29% slower than raw W8
in the tested contractions; no production documentation claims otherwise.

## Validation boundary

Local builds detect and deduplicate all visible supported GPU capabilities. A
mixed Turing/Ampere host therefore emits both cubins automatically. An explicit
`COMFYUI_TURING_UTILS_ARCH_LIST` remains available for cross-compilation;
GPU-less builders use a conservative sm75 fallback. Static tests validate
dispatch, fallbacks, loader independence, shapes, dtypes, spill-free SM75
resources for every compiled core/attention/preprocessing family, the public
symbol boundary, and exclusion of the retired Sage1/Sage2 variants. For
compatible A40 validation, build with:

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --sol
python kernel/scripts/validate_wan_fusions.py --device cuda:0
python kernel/scripts/audit_attention_resources.py
```

An A40 run validates numerical behavior, allocation shapes, and the absence of
Ampere-only source dependencies. It JITs compute_75 PTX and selects the same
CTA schedule used on sm75. This does not replace the final exact-sm75 occupancy
and end-to-end test.

For native Ampere acceptance, use `COMFYUI_TURING_UTILS_ARCH_LIST="8.6"`.
The resulting attention cubins use the `__CUDA_ARCH__ >= 800` async-copy and
INT8 MMA paths; A40 preflight covers BF16 D64/D128 GQA for both Sol and W8A8.
Sol's initial Q, exact Q/K, and W8 value tiles use the shared async-copy
abstraction. During exact sparse attention, sm80+ starts the next K transfer
after current QK releases the K buffer and overlaps it with online-softmax/PV;
V is replaced only after current PV completes. The same source emits a safe
synchronous K-then-V schedule on sm75 and allocates no second shared tile. W8
projections use CUTLASS SM80 `m16n8k32` kernels with a per-shape
three-schedule cache; explicit policies remain available for controlled
benchmarking. The common scale/bias/BF16 visitor is shared with sm75, so this
is a schedule substitution rather than a model-specific numerical path.
The historical `_sage_*_sm75` module names remain stable ABI identifiers.
On the initial native-sm86 direction check (`N=4096`, H8, D128, BF16), dense
W8A8 measured 0.840 ms, Sol FP16-PV 0.428 ms, and Sol W8A8 0.417 ms at 19.7%
route density. Their output cosine against dense W8A8 was 0.9985 and 0.9982,
respectively. These figures verify native dispatch and arithmetic; they are not
an H3 end-to-end or visual-quality claim.
