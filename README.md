# ComfyUI Turing Utils

Compatibility and performance extensions for CUDA Tensor Core GPUs. The plugin
currently provides ConvRot W8A8/W4A8/W4A4 support, exact-sm75 BF16 activation
storage, bundled sm75+ W8A8 attention, exact-sm75 Sage, native sm75+ Sol and
fixed-Top-K SLA sparse attention, and focused Krea2/Wan/Bernini utilities.

## Requirements

- NVIDIA GPU with CUDA support
- Turing, Ampere, Ada, or newer architecture
- Python 3.10 or newer
- PyTorch with CUDA and ComfyUI
- `comfy-kitchen>=0.2.26` for ConvRot model integration
- the independently installed `comfyui-turing-utils-kernel>=0.29.1` for SLA;
  native sm75+ W8A8/Sol requires a cubin (or PTX) for the target GPU, while
  exact-sm75 installs additionally provide bundled Sage and BF16 compatibility

Grouped-codebook `asym_w4a8_int8` checkpoints require kernel 0.24.0. Existing
W8A8, W4A4, legacy W4A8, and attention paths keep their earlier minimum.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wjie98/comfyui-turing-utils.git
cd comfyui-turing-utils
python -m pip install -r requirements.txt
python -m pip install -v --no-build-isolation -e ./kernel
```

The kernel build detects every visible supported CUDA architecture and removes
duplicates. A machine with a 2080 Ti and a 3070 therefore builds `7.5;8.6` in
one install. Set `COMFYUI_TURING_UTILS_ARCH_LIST` only for cross-compilation or
to override the visible-device set. GPU-less build hosts fall back to `7.5`.

The custom node and CUDA package have separate installation lifecycles.
Python-only plugin updates never invoke a compiler or JIT; rebuild the kernel
only after its CUDA sources or required version change.

## Nodes

### Shared application models

`MiniMax H3 Latent Upscale` and `SeC Track Visual Concept` include their
model selectors. When a workflow is submitted through ComfyUI's prompt server,
applications with identical model names and loading settings (or identical
links supplying those settings) share one internal loader. Application inputs
such as videos, points, masks and scale do not enter the loader's cache key.
This also works across frontend subgraphs, which submit a flattened prompt.
Within a prompt the shared loader executes once, even with caching disabled.
Across prompts, reuse follows ComfyUI's normal cache policy; eviction or
disabling caching permits reloading. GPU offload/reload remains ComfyUI-managed.

The old loader interfaces remain registered as development-only internal nodes,
not additional normal workflow steps. Existing explicit model connections are
accepted by the prompt compiler. Direct Python callers can still call the
original loading functions; callers bypassing the prompt server must invoke
`compile_shared_loaders` before handing a graph to `PromptExecutor`.

- `Load ConvRot DiT` and `Load ConvRot CLIP` also accept ComfyUI-native `nvfp4`
  layers alongside `int8_tensorwise`, `convrot_w4a4`, and `asym_w4a8_int8`.
  Mixed formats dispatch per layer; official format identifiers are preserved.
  The former standalone `Load NVFP4 DiT` node is removed: replace it with
  `Load ConvRot DiT` in saved workflows. The NVFP4 CUDA path retains packed
  NVFP4 weights, and uses A8/S8 GEMM with at most 16 MiB per temporary S8 staging
  buffer for normal model widths (two buffers in the measured overlap range).
  It does not use native Blackwell A4 arithmetic.
  DiT attention retains the ConvRot loader's `w8a8` default; `sdpa` and `sage`
  remain available. ConvRot layers keep their existing dispatch; other formats
  and explicit full-precision layers retain ComfyUI execution. CLIP's CPU
  option uses a dense NVFP4 fallback. `force_int8_gemm` does not rewrite NVFP4.
  LoRA uses ComfyUI's patch/requantization lifecycle; patching may temporarily
  expand a layer, and weight hooks can take the dense fallback. Full H3 video
  quality and 2080 Ti runtime validation are still pending. Rebuild the current
  kernel sources before trying this node.
  The single runtime path applies paired ConvRot256 to activations and decoded
  NVFP4 weights, then runs row-scaled INT8 Tensor Core GEMM. NVFP4 decode,
  FP32 weight rotation and S8 quantization are fused. Only bounded call-local
  S8 tiles are expanded (normally at most 16 MiB); native NVFP4 storage remains
  unchanged. No rotated/requantized model copy or backend selector is retained.
  Logical K need not be divisible by 256: both operands are zero-padded before
  rotation, ignoring serialized padding. Logical N tails and FP16/BF16/FP32
  outputs are supported. K above 16384 uses a streaming two-pass specialization
  of the same algorithm. Pre-scales and SwiGLU run before rotation; ComfyUI
  remains responsible for LoRA hooks and requantization.
  `kernel/scripts/benchmark_nvfp4_routes.py --profile` compares the one runtime
  implementation against preexpanded W8A8/FP16/BF16 controls and separates its
  activation, weight conversion, and GEMM costs. Real weights plus synthetic
  activations are not a substitute for video/LLM quality validation.
  A40 / M=19989 convergence benchmark (milliseconds, synthetic BF16 inputs):
  QKV 25.08 vs expanded W8A8 23.40 / FP16 39.88 / BF16 39.00;
  FFN-down 19.49 vs 18.47 / 28.37 / 25.90. W8A8 controls use the same
  NVFP4-derived rotated S8 weights, not independently quantized checkpoints.
  Added relative L2 error versus decoded NVFP4 is about 1.31% / 1.39%.
  Standalone component timings are QKV activation 0.95, weight conversion 1.42,
  chunked GEMM 22.81; FFN-down 2.85, 1.03, 16.10. These isolated measurements
  are not strictly additive. Further work should prioritize GEMM utilization
  and FFN activation/rotation fusion; even eliminating weight conversion alone
  cannot deliver a large long-sequence speedup.
  Follow-up optimization on the same A40/M=19989 workload: QKV 24.92 ms
  (essentially unchanged), FFN-down 18.54 ms versus the prior 19.49 ms.
  SM86 BF16 long/wide-K GEMMs use a four-stage asynchronous mainloop; other
  shapes retain their baseline schedule. SwiGLU is fused with rotation/A8
  quantization for BF16/FP32 when shared memory and input layout allow it, with a compact
  BF16 row-buffer fallback. FP16 uses bounded row-chunk FP32 rotation/quantization
  after activation to avoid half-divisor underflow on zero/tiny rows; it does
  not use the fused half-divisor path. Channel pre-scales retain the correct unfused
  ordering. Including SwiGLU, FFN-down measured 19.11 ms versus 22.95 ms with
  unfused activation under the same final GEMM schedule; incremental peak
  allocated memory (including output, excluding inputs/resident weights) fell
  from 1093 to 492 MiB. Fusion slightly improved error against FP32 SwiGLU in
  these synthetic tests, but is not bit-identical to BF16 intermediate rounding.
  Register-resident weight quantization reuses a rounded FP32 scale reciprocal;
  rare S8 boundary values can change by one code, with regression error bounds.
  Output row strides are aligned to 128 bytes without computing extra weight
  rows. Non-aligned N tests gained about 1–2%; native H3 widths already align.
  Additional K padding beyond ConvRot256 was slower and is not enabled.
  Nsight Compute counters were unavailable due to driver permissions, so these
  are event-timing measurements, not measured MFU/occupancy claims.
  Bounded two-stream staging is now used for the measured SM86 BF16 range
  (M=8192..24576, padded K=8192..16384, padded N=4096..16383). It keeps the
  original GEMM chunk size and reuses two <=16 MiB S8 buffers with explicit
  producer/consumer events; no decoded weight cache persists. Other shapes,
  devices/dtypes and CUDA Graph capture retain serial staging. This is overlap,
  not decode-inside-GEMM: the intermediate S8 write still exists. Paired A40
  M=19989 FFN timings were 18.15 -> 17.79 ms, or 18.79 -> 17.88 ms including
  fused SwiGLU, with bit-identical serial/pipelined results and ~14 MiB extra
  temporary storage for K=14336. Expanded resident S8 controls were 16.53 and
  17.24 ms respectively. Halving chunks to avoid the second-buffer cost hurt
  some long sequences and is not enabled. These remain synthetic-activation,
  real-weight operator tests, not full-video or SM75 results.

  `kernel/scripts/benchmark_nvfp4_h3_block.py` adds a whole core DiT-block
  comparison using real checkpoint weights and synthetic tokens. It excludes
  RoPE, SOL, model offloading, and the full video pipeline; expanded weights
  remain resident only as benchmark controls. For A40/M=19989, compact NVFP4
  measured 203.69 ms, expanded S8 200.13 ms, and decoded BF16 256.46 ms.
  Disabling SwiGLU fusion took 210.20 ms; fusion reduced incremental peak
  allocation from 2597 to 2054 MiB. Relative L2 versus the decoded NVFP4 BF16
  block was 0.61%; this is not a perceptual-quality guarantee. At M=8192 the
  corresponding compact/S8/BF16 timings were 62.88/54.34/79.69 ms.
  Further half-width MLP staging reduced memory but regressed latency by
  7--15%; row-chunking repeated weight conversion and was also slower.
  Serial call-local S8 buffer reuse did not materially improve long-sequence
  latency. These candidates are not enabled and add no runtime selectors.

  SM75 portability was cross-compiled with CUDA 13.0, not inferred from SM86:
  the 128x256x64 INT8 GEMM uses 208 registers/thread, 256 threads and 48 KiB
  dynamic shared memory, with no ptxas spills. Both registers and shared memory
  limit it to one CTA/SM (25% theoretical warp occupancy). Weight conversion
  uses 54 registers/thread at K=5376 and 72 at K=14336: respectively four and
  three CTAs/SM (100%/75%). The BF16 SwiGLU row-buffer fallback at K=14336
  selects 1024 threads on SM75 for long sequences: 30 registers/thread,
  61440 dynamic + 144 static shared bytes, one CTA/SM and 100% warp occupancy.
  These kernels have zero reported spill loads/stores. SM86 retains its own
  schedules; the two architectures need not have matching register counts.

  `kernel/scripts/audit_nvfp4_resources.py` reads exact SM75/SM86 cubins using
  cuobjdump and the standalone NVIDIA occupancy calculator. Build its helper
  `kernel/scripts/occupancy_model.cpp` with a host C++ compiler and
  `-I "$CUDA_HOME/include"`, placing the executable in the instance's temporary
  directory, then pass `--binary PATH_TO_CORE_EXTENSION --calculator PATH_TO_HELPER`.
  Optional `--build-log PATH` reads spill counters from a build made with
  `NVCC_APPEND_FLAGS='--ptxas-options=-v'`. Build both architectures with
  `COMFYUI_TURING_UTILS_ARCH_LIST='7.5;8.6'`. All 19 reported SM86 residency
  predictions matched the A40 driver occupancy API. These are resource ceilings,
  not achieved occupancy or speed predictions; SM75/Windows runtime and visual
  validation remain required. Linux binaries cannot be reused on Windows.

  A diagnostic 128x128x64 CTA / 32x64 warp candidate reduced SM75 register
  usage from 208 to 124/thread and shared memory from 48 to 32 KiB, giving
  two CTAs/SM with no spills. SM86 compiled to 126 registers/thread and also
  supported two CTAs/SM, confirmed by the driver. However, A40 real-weight
  tests at M=8192/19989/32768 regressed despite bit-identical BF16 outputs.
  At M=19989, compact QKV/out-proj/FFN-up/FFN-down took respectively
  40.49/14.73/53.66/31.08 ms versus 25.80/9.58/34.89/19.49 ms using the
  original SM75 algorithm on the same GPU (FFN-down includes SwiGLU).
  This candidate is not enabled or retained in runtime dispatch. Its SM75
  resource improvement is proven; its speed on an actual 2080 Ti is not.

  LoRA merging is inherited from ComfyUI, not replaced by this GEMM kernel:
  dequantize the original NVFP4 weight, accumulate the complete patch list in
  the selected merge dtype, then requantize once with recalculated tensor and
  block scales. ComfyUI's positive-seed path uses its stochastic NVFP4 rounding;
  Kitchen's standalone default quantizer is deterministic. Neither guarantees
  lossless LoRA merging, including on Blackwell. Tests compare packed bytes and
  both scales against official Linear/requantize paths for multiple LoRAs,
  FP32/BF16 merges, stochastic/deterministic modes and padded dimensions.
  Weight preparation for INT8 GEMM never modifies those stored NVFP4 bytes.
- `Load ConvRot DiT` loads ComfyUI ConvRot diffusion models. It supports W8A8,
  W4A8, and W4A4 dispatch. Its attention choices are `w8a8`, `sage`, and
  `sdpa`; W8A8 is the default.
- `Load ConvRot CLIP` loads a ConvRot text encoder independently of the DiT.
- `Bernini Inpaint Condition` starts sampling from the source-video latent,
  supports local or global repainting, and optionally adds the source as aligned
  context tokens.
- `Krea2 Identity Edit Conditioning` combines the Identity Edit appearance and
  Qwen3-VL semantic paths in one node. It accepts one required character image
  and one optional background/edit canvas, always orders them as
  `[background, character]`, fits and VAE-encodes both before sampling, and
  returns the patched model plus one conditioning. Connect the same target
  latent to this node and KSampler; the latent continues directly to KSampler.
  Character/background strength defaults remain `4/1`; setting both to `1`
  disables the extra attention bias and keeps the fastest native attention path.
  For the recommended Turbo/CFG 1 path, connect the one conditioning output to
  both positive and negative sockets.
- `Bernini Context Windows` applies reference-aware Wan context windows with
  selectable absolute or official relative temporal positions.
- `Video Frames Padding` selects the model frame grid through `type`, including
  Wan and H3's `17*n+5` frame grid.
- `Load/Save Indexed Video Segment` read and atomically write six-digit MP4
  segments such as `000324.mp4`. Relative root directories resolve below the
  active ComfyUI instance's output directory, while absolute roots are used
  directly. Loader index `0` returns empty, positive index `i` reads segment
  `i-1`, and `-1` selects the highest existing segment number. The loader can
  return only the final frames with synchronized audio, defaulting to the H3
  `17+5=22`-frame continuation prefix; a missing segment returns empty outputs.
  The saver normalizes audio to the exact video-frame duration, supports explicit
  overwrite, and deliberately produces no preview.
- `Merge Indexed Video Segments` verifies that `000000.mp4` through the selected
  inclusive maximum index are contiguous and video-compatible, then stream-copies
  the compressed video into one MP4. Maximum index `-1` (the default) selects all
  segments through the highest existing index; `0` selects only `000000.mp4`. It
  decodes audio per segment, removes each
  segment's independent AAC padding, fits sample counts to exact cumulative
  video-frame boundaries, and performs one continuous AAC encode. This avoids
  cumulative timestamp rounding and encoder-delay seams without holding the
  complete video as an IMAGE batch. Legacy segments that decode up to one AAC
  frame short are padded automatically; larger mismatches remain errors. Segments
  are joined exactly as stored, so
  continuation context must pass through `Trim Video Continuation Prefix`
  before each segment is saved. The merger is atomic and has no preview.
- `Video Continuation Concat` prepends optional image/audio context, supplies
  zero masks for an unmasked prefix and one masks for an unmasked generated
  body, and returns exact rational boundary metadata. `concat` mode prepends
  the prefix for continuation generation; `replace` mode fills leading context
  slots already included in the body timeline without increasing its duration.
  In either mode the prefix is temporary context and the trim node removes it
  from the generated images and synchronized audio. Image masks and audio follow
  the same boundary.
- `Video Prefix Context Noise` is an independent IMAGE processor intended for
  a complete prefix sequence. Its default recipe first preserves the final five
  frames, tapers the preceding four frames from alpha `0.45` toward `0.10`, and
  applies full six-colour block noise to every earlier frame. This preserves the
  same `13 full + 4 transition + 5 clean` layout for a 22-frame H3 prefix without
  requiring its total length as an input. Short batches allocate the clean tail
  first, then as much transition as fits, and only then add a full-noise region.
  Advanced controls expose the end alpha, pattern, grid mode, and block size. A
  missing image passes through as absent. These empirical defaults follow MacroSony's
  [H3 chained-character-swap recipe](https://github.com/MacroSony/minimax-h3-chained-character-swap)
  and its [ComfyUI context-noise implementation](https://github.com/beijinren/ComfyUI-H3-Context-Noise).
  Missing audio spans are represented by duration-matched silence.
- `Trim Video Continuation Prefix` removes that exact image/audio prefix after
  generation. `H3 Set Audio Prefix Noise Mask` maps the recorded waveform
  boundary onto a standalone `[B,32,2,T]` H3 audio latent: its default mode
  protects the prefix and generates the body before core `Concat AV Latent`.
- `Set Video Latent Noise Mask` accepts image-frame masks for a standalone
  video latent. Its `type` choices follow CLIP Loader: `wan`, `minimax`
  (H3 video only, default), `ltxv` (LTX-Video/LTX-2 video), `hunyuan_video`,
  `hunyuan_video_15`, and `mochi`. Equal mask/latent time counts map directly.
  Otherwise Wan and HunyuanVideo keep the first frame separate and merge
  subsequent groups of 4 frames; LTX uses groups of 8 and Mochi groups of 6.
  H3 instead merges `[1,4,4,4,4]*n+[1,4]` into latent positions. Temporal
  groups and spatial resize bins use maximum/union, preserving small masked
  regions and soft-mask strengths. For a 124-frame H3 video, 124 masks with
  the first five black become 37 masks with the first two black. Black
  preserves and white redraws. Other frame counts fail without interpolation,
  padding, or truncation; a single mask is not broadcast across time.
  Input masks are `[frames,H,W]` (shared across video batches) or
  `[B,frames,H,W]`; output `noise_mask` is `[B,1,T,H_lat,W_lat]`. Existing
  noise masks are replaced; samples and other metadata stay unchanged.
  Only native unpacked `[B,C,T,H,W]` video latents are supported. Separate AV
  latents (including LTX-2) before this node and concatenate afterward. This maps
  temporal groups, not the VAE's full receptive field, and does not guarantee
  pixel-identical decoded boundaries or make H3 Add Noise mask-aware.
- `Video Latent Composite Masked` takes matching standalone `[B,C,T,H,W]`
  `original_latent` (clean original high-resolution video) and
  `replacement_latent` (e.g. upscaled first-pass `denoised_output`). It unions
  the replacement's inherited `noise_mask` with an optional image/latent-frame
  `mask`, using the same `type` profiles and conservative mapping as Set Video
  Latent Noise Mask. Every positive mask value means full replacement, not
  opacity; threshold noisy/soft input masks upstream if needed. The exact same
  binary region selects replacement samples and becomes the output `noise_mask`.
  Outside it, original samples are retained exactly. With neither mask, the
  whole video is replaced. The original's metadata is kept but its mask is
  ignored. Inherited masks must match the replacement T/H/W and have batch 1/B
  and channels 1/C (channel coverage is unioned). There is no latent resizing,
  retiming, or feathering. Align frame sequences/crops as well as tensor shapes.
  For masked second-pass sampling, use the clean composite with native
  RandomNoise and the remaining sigmas; the existing H3 Add Noise is not made
  mask-aware by this node. Decoded boundaries need not be pixel-identical.
- `Resize Image If Present` resizes, crops, or pads an optional image and mask.
  With no image connected it returns no image, so one graph can safely feed
  optional first- or last-frame conditioning sockets without making a black
  placeholder frame.
- `Video Mask Guided Crop` converts one target mask per video frame into a
  temporally smoothed, fixed-aspect crop sequence. `width` and `height` are both
  the exact output resolution and the source-frame crop-box ratio; the box is
  shifted and, when necessary, scaled down so it always stays entirely inside
  the source frame without padding. Bounded missing-mask runs can interpolate
  position and logarithmic size, while leading/trailing runs hold the nearest
  observation; `hold` mode instead keeps the last valid crop. The cropped mask
  remains empty on missing frames even though the image crop stays continuous.
  `crop_info` stores the per-frame floating-point transforms.
- `Video Mask Guided Stitch` maps regenerated crops and their masks back through
  those exact transforms and composites only the masked pixels over the source
  video. Its optional feathering is measured in source-video pixels. Extra
  regenerated tail frames are ignored, while missing frames or mismatched source
  geometry are rejected.
- `Video Pad For Outpaint` creates final-resolution video frames and hard repaint
  masks without requiring a paired stitch node. Its dynamic layout control shows
  either explicit source-pixel margins or a current-aspect expansion ratio plus
  normalized X/Y placement. Relative placement can clamp the complete source
  inside the canvas or deliberately let it cross the frame boundary. The canvas
  is scaled isotropically toward a target megapixel count; its final dimensions
  and the inward-protected source rectangle align to a configurable multiple
  (32 by default for MiniMax H3). Padding can extend edge pixels, use neutral
  gray, or use black; every output mask remains binary.
- `Mask to Visual Prompts` uses the first frame from its IMAGE and MASK inputs
  and returns reusable visual prompts plus a rendered preview. Positive points
  combine perceptually distinct colour-region representatives, mask-medial
  points biased toward thin structures, and spatially spread interior
  fallbacks. Optional negative points are spread through a nearby exterior
  ring. The preview overlays a subtle thresholded-mask tint and contour, an
  amber bounding box, green positive markers, and red negative markers. Point
  outputs use the JSON coordinate format shared by KJNodes, SeC, and built-in
  SAM3. The legacy `BBOX` output targets older KJNodes consumers, while
  `BOUNDING_BOX` targets SeC and current ComfyUI nodes such as SAM3. Input batch
  lengths may differ, but their first frames must have matching spatial dimensions.
- `SeC Track Visual Concept` includes model selection and loads single-file or Hugging Face directory-format SeC
  checkpoints from `ComfyUI/models/sams`. The checkpoint starts on ComfyUI's
  offload device and is registered through a model patcher; there is no manual
  device selector or private unload lifecycle. Its `auto` attention mode selects
  a compatible Flash Attention implementation independently for InternViT and
  the language model, with SDPA as the guaranteed fallback; `sdpa` forces the
  portable backend throughout SeC.
- `SeC Track Visual Concept` accepts the JSON coordinates and canonical
  `BOUNDING_BOX` emitted by `Mask to Visual Prompts`, plus an optional direct
  mask. Frames are consumed
  from the IMAGE tensor without temporary JPEG files. With a mask connected,
  that mask is authoritative, the bounding box limits its region, and clicks are
  checked for consistency. Without a mask, the box and all clicks are submitted in one
  SAM2 prompt so one prompt type cannot silently erase another. Video frames and
  per-run tracking state remain on CPU while ComfyUI owns model loading,
  retention, and eviction; only the tracked mask batch is returned. The advanced
  `semantic_keyframes` control limits the labelled scene-change memories supplied
  to the MLLM recovery path. `annotation_frame_idx` accepts absolute indexes or
  standard Python-style negative indexes: `-1` selects the final frame and `-2`
  selects the penultimate frame.
- `Is Input Present` accepts an optional value of any type and reports whether
  it is connected and non-empty; scalar `0` and `false` still count as present.
  Its second output forwards that value or lazily evaluates an optional fallback.
- `Lazy If / Else` switches values of any ComfyUI type while lazily evaluating
  only the selected branch, unless another workflow output also needs the
  unselected branch.
- `Stage Barrier` forwards eight fixed, optional arbitrary-value pairs and treats its
  non-negative `stage` widget as a reusable phase label. The scheduler derives
  dependency rounds automatically: increasing or equal labels stay in the
  current round, while a dependency whose label decreases starts the next
  round. Barriers with the same inferred `(round, stage)` rendezvous before
  downstream work is released. Each visual input/output pair is compiled into
  an independent cache and execution path, so several Lazy If / Else branches
  can share one Barrier without evaluating the unselected branches. Use stage
  0 after reference/VAE preparation, stage 1 after semantic conditioning or
  sampling, and stage 2 after decode; dependent branches may reuse the same
  labels without manual renumbering.
- Use core `Concat AV Latent` / `Separate AV Latent` for H3 AV streams.
  The plugin-specific duplicate nodes have been removed.
- `H3 Add Noise` prepares **clean x0** for a continuation sampler using
  `DisableNoise` (or `add_noise=disable`). Connect the continuation H3 `MODEL`,
  `RandomNoise`, the remaining `SIGMAS`, and a clean `LATENT`; the first sigma
  is the target level, not the difference between the schedule endpoints.
  It processes every supplied stream: standalone video, standalone audio, or
  both in a native AV latent. H3's current audio/video shift ratio is honored
  even for standalone audio, using the same **video** schedule as the sampler.
  For a 6+2 upscale workflow, split the first sampler's `denoised_output`,
  upscale its clean video, then use this node on that video. Join it with the
  unchanged audio from the first sampler's **`output`**, and resume with the
  remaining two-step schedule and noise disabled. Do not re-noise that audio
  unless you deliberately restart it from clean x0 as well.
  Empty schedules and sigma zero are no-ops; sigma one cannot be represented
  for `DisableNoise` and is rejected. Masks and metadata are preserved for the
  next sampler, not applied by this node. Half-precision latents are promoted
  to float32 for continuation arithmetic. This prepares a new noisy state;
  it does not preserve multistep solver history across two sampler nodes.
- `H3 Latent Info` reports the decoded pixel width, height, frame count, and
  H3's 24 FPS model rate without running the VAE.
- `H3 Keyframe Reference` dynamically adds `image_N` inputs and matching
  `keyframe_N` outputs. Every output is role-free and reusable: it can connect
  to either the independent first- or last-frame socket on the semantic/build
  nodes, including both roles across different sampling branches.
  `H3 Image/Video/Audio Reference` encode dynamic
  generic reference sets without allocating a target latent. Visual references
  use match-area sizing when a `latent` is connected and a configurable
  megapixel area budget otherwise; neither mode crops or deliberately enlarges
  the source. The default unbound reference budget is 1.0 megapixel.
  Video reference inputs must be resampled to 24 FPS by their upstream loaders.
- `H3 Semantic Reference` performs the Qwen3-VL presentation encode once from
  the prompt and reference objects. `H3 Build Conditioning` combines that
  reusable semantic result with independently selected VAE references and an
  H3 target latent. Build references may be omitted, added, or replaced without
  rerunning Qwen or matching the semantic reference counts/modalities. Connected
  first/last keyframes must still be single-frame latents matching the target
  spatial grid. Semantic embeddings and their token tags are kept unchanged;
  reference meaning and ordering are controlled by the workflow.
- `MiniMax H3 Latent Upscale` includes model/precision selection and loads the attention-free 3D learned latent
  upscaler through ComfyUI's normal offload lifecycle. Place compatible weights
  from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
  in `models/latent_upscale_models/`.
- `MiniMax H3 Latent Upscale` enlarges only the video stream by a continuous
  1x--4x multiplier and passes the audio stream through exactly. Its optional
  `CONDITIONING` input enlarges FL2AV first/last keyframe latents with the same
  learned model, while Ref2AV image/video/audio references retain their
  independent geometry. Without it, only the AV latent is processed. No text
  or VAE conditioning stage is rerun. Existing video noise masks are enlarged
  with conservative spatial maximum coverage, not nearest interpolation:
  every intersecting source cell contributes, keeping binary masks binary and
  retaining soft maxima. Time and audio masks are unchanged; absent masks stay
  absent. This covers grid footprints, not the learned upscaler's full receptive
  field. Composite onto the original high-resolution latent to restore the
  area outside the inherited mask; do not overwrite the union with Set Mask.
- `MiniMax H3 Video VAE Decode/Encode` call the official ComfyUI VAE entry points,
  retaining scoped fused operators, decoder attention selection, and lightweight
  tqdm progress with a planned tile total and ETA for normal single-video inputs.
  ComfyUI owns tiling, batching, dtype, transfers, model residency and OOM recovery.
  Custom tile batching, async pixel buffers,
  block prefetch and non-evicting memory budgets are removed, along with the
  earlier shared-state decoding and experimental overlap controls.
- `Multimodal Prompt Chat` sends one non-streaming system/user turn to an
  OpenAI-compatible Chat Completions endpoint using only Python's standard HTTP
  client. Either `prompt` or `system_prompt` may be empty, but not both.
  Optional first/last-frame images receive explicit `<First Frame>` and
  `<Last Frame>` labels; dynamic images are labeled `<Picture N>`. Dynamic
  `IMAGE` frame sequences follow H3's 24 FPS reference convention and are
  sampled into timestamped `<Video N>` frames. A root URL automatically gains
  `/v1/chat/completions`, while versioned and complete endpoint URLs are kept.
  API keys may be literal, empty for a local placeholder, or `$NAME`/`${NAME}`
  environment references. Thinking, sampling, media, and retry controls now live
  in the node's advanced inputs. JPEG quality appears only for JPEG encoding.
  The standalone Options node and socket have been removed. Defaults are unchanged,
  including an 8192-token output limit and
  `chat_template_kwargs.enable_thinking=false`.
- `Video Motion Contact Sheet (Experimental)` samples an `N x N` chronological
  storyboard from a loaded `VIDEO` or decoded `IMAGE` frame batch. It can use
  uniform or motion-weighted sampling and optionally wraps each panel in
  annotated film rails so frame numbers and timestamps stay outside the image.
- **Material Workspace** opens in ComfyUI's native workflow tabs from the Turing Utils menu.
  Four material boundary nodes persist text, images, video and audio. Ordinary
  workflows with Canvas Inputs/Outputs become independent card copies; executing a material runs only the segment back
  to the preceding saved materials through ComfyUI's normal queue. Videos load a
  poster only until playback is requested. Creating a project requires an empty
  folder; opening one requires its valid `canvas.json`. The standalone page is removed.
  See [Material Workspace](docs/material-canvas.md) for usage and validation limits.
- `Configure Attention Strategy` is the unified entry for Sol, SLA, and Veda.
  H3 Image Sol and Static Virtual KV have been removed, including their node IDs.
  Its dynamic selector exposes only
  the selected strategy; advanced inputs contain reference protection and
  dense-step/layer safeguards. Manual prefix length appears only for manual
  protection. The three previous Sol/SLA/Veda Configure node IDs are removed. Dense backend selection stays in the loader, and this
  change does not alter kernels or introduce additional model copies/caches.
  See [node configuration and reusable presets](docs/node-configuration.md)
  and the [complete node inventory](docs/node-inventory.md).
- `Video Frames Padding` shares the model `type` choices of Set Video Latent
  Noise Mask: Wan, MiniMax H3, LTXV, Hunyuan Video, Hunyuan Video 1.5, and Mochi.
  It can pad images and masks independently, keeping image encoding cached on
  mask edits. Old Wan/H3 padding node IDs are removed.
- The Sol strategy applies the production model-generic,
  loader-independent
  long-sequence sparse backend. It uses an input-adaptive statistical threshold,
  keeps one 64-token skipped-block centroid by default, accepts semantic
  multimodal layout metadata, and exposes integer dense-step safeguards,
  dense first/last-layer protection, and an internal automatic short-sequence
  crossover. It inherits `w8a8`, `sage`, or `sdpa` from `Load ConvRot DiT`:
  W8A8 selects integer sparse PV, while Sage/SDPA select the floating FP16
  sparse core. Dense prefix/suffix counts are local to every sampler invocation,
  so one configured model can feed both stages without pass-specific controls.
- The SLA strategy implements the MiniMax H3 Turbo-SLA runtime as
  fixed-budget 128-query by 64-key Top-K routing. It shares Sol's semantic
  reference protection, dense step/layer scheduling, fused Q/K preprocessing,
  tensor lifetime, and inherited W8A8/FP16 numeric path, but deliberately does
  not add Sol's local blocks or skipped-block residual. Use it with the SLA-trained
  LoRA; `sparsity_ratio=0.85` matches the published runtime hyperparameter.
  The legacy `Patch Sol/SLA Sparse Attention` nodes have been removed.
  Replace them with the `Configure` nodes in existing workflows; the dense
  backend is selected by the loader, not by the old `use_w8a8` widget.

The shared asymmetric-Q/K and independent-Q/K-RoPE protocol and kernel ABI
remain model-independent; the removed H3 static-image adapter is not required.

## Krea2 Identity Edit wiring

Apply the compatible Identity Edit LoRA before the node. The target latent is
an input because reference VAE encoding must finish before KSampler loads the
DiT, and because mismatched reference aspect ratios need the target grid for
centered RoPE positions. It is not copied to an output:

```text
Load Diffusion Model -> Load LoRA -> Krea2 Identity Edit Conditioning.model
Load CLIP (krea2) ----------------> Krea2 Identity Edit Conditioning.clip
Load VAE -------------------------> Krea2 Identity Edit Conditioning.vae
Load Image (character) -----------> Krea2 Identity Edit Conditioning.character_image
Load Image (background, optional) -> Krea2 Identity Edit Conditioning.background_image
Empty SD3 Latent -----------------+> Krea2 Identity Edit Conditioning.target_latent
                                  +> KSampler.latent_image

Krea2 Identity Edit Conditioning.model --------> KSampler.model
Krea2 Identity Edit Conditioning.conditioning -+> KSampler.positive
                                                +> KSampler.negative (CFG 1)
```

When both images are connected, Qwen3-VL and the DiT always receive
`[background, character]`. The character-only path remains a single-reference
edit; there are no numbered reference sockets or hidden role changes.

## MiniMax H3 automatic activation memory

The H3 adapter uses one capability-based path on Turing, Ampere, Ada, Hopper,
and newer Tensor Core GPUs. CUDA selects the cubin compiled for the installed
card; Python does not maintain a per-generation H3 algorithm. At each QKV or
FFN call, the adapter reads ComfyUI's immediately usable memory and the
`--reserve-vram` ceiling:

- if the complete activation fits with safety headroom, it keeps the normal
  full-row path for maximum throughput;
- otherwise QKV projection is streamed by rows while retaining only INT8 Q/K
  and BF16 V, and the SwiGLU FFN is streamed into its final hidden output;
- if that compact state still does not fit, attention is evaluated in legal
  whole-head groups. Every group still attends over the complete sequence and
  keeps the selected backend (including explicit SDPA); ConvRot groups are
  split only on their 256-value boundary;
- at the extreme FFN floor, the intermediate width is split on the same
  256-value boundary. A first pass obtains the original whole-row scale, a
  second pass writes directly into the final compressed INT8 activation, and
  the original fused fc2 performs the complete contraction once;
- each layer's weights are cast/transferred once and reused by every row tile,
  so activation savings do not multiply Dynamic VRAM traffic;
- under AIMDO DynamicVRAM, immediately usable memory selects the execution
  tier. Resident model pages never promote a faster tier: evicting hot DiT
  weights only to reload them on the next layer costs more than an exact head
  shard. Resident, unpinned pages from inactive models remain an emergency
  reserve for an already-selected tier, without using the noisy
  `vbars_analyze` diagnostic path;
- activation low-water marks are scoped to the operation. A low pre-QKV
  reading therefore cannot unnecessarily force the later MLP to stream after
  QKV/attention buffers have retired;
- once splitting is required, automatic shard sizes stop at the MFU plateau
  instead of consuming every free byte. Attention requires at least four CTA
  waves, a 1024-channel QKV projection, and at most four balanced head groups;
  FFN channel shards use the same four-way balance, while QKV and MLP row
  tiles cap at 16K. Larger activations offer little additional utilization but
  displace hot DynamicVRAM weight pages;
- DynamicVRAM additionally reserves two average transformer blocks (bounded
  to 512 MiB--1 GiB and aligned to AIMDO's page size) for the active and next
  asynchronous weight stream. This reserve is derived from the active model's
  VBAR size and layer count, not from a GPU-generation name;
- a dynamically resident DiT does not retain the optional full-sequence INT8
  input cache during head sharding. Recomputing the inexpensive row
  quantization preserves roughly one hidden tensor of weight residency and
  avoids PCIe page churn.

No workflow socket or node changes are required. For a 16 GiB display card
that must leave 4 GiB to Windows and the compositor, launch ComfyUI with
`--reserve-vram 4`. The default `auto` mode then treats 12 GiB as a hard
inference ceiling even while the desktop is temporarily idle.

The runtime exposes one activation policy setting:

```text
COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE=auto|throughput|balanced
```

`auto` is the default. Row/head/channel limits are selected automatically;
explicit overrides exist only as internal policy-function arguments for tests
and benchmarks, not environment variables. QKV
streaming is available through bundled W8A8, Sol-W8A8, and SLA-W8A8 prepared
attention. Kernel 0.32 precomputes the adaptive K anchor from the same nine
global sequence locations and reuses it while writing every row tile directly
into the final Q/K storage. Row and head splitting therefore do not discard the
global anchor, RMSNorm, RoPE, orthogonal rotation, scale blocks, or any K/V row.
For a dynamically paged model, `auto` keeps a 16K QKV row tile after a sequence
reaches four tiles even when transient free VRAM would permit the full
projection. That tile is already compute-saturated; retaining it avoids
whole-sequence V preparation and preserves prefetched weight pages. The rule is
based on workload geometry and residency rather than the GPU architecture.
Policy logs report both `head_group` and `saturation_group`; equality means the
selected group has reached the modeled MFU plateau without growing its working
set further.

The automatic ladder is: full throughput, row streaming, compact prepared
Q/K, whole-head grouping, then two-pass FFN-channel grouping. It is selected
independently for each live operator, so a 12 GiB budget normally stops at row
streaming while a much tighter run can descend further. The final hidden output
and the chosen attention backend's irreducible state still have to fit; `auto`
cannot make an arbitrarily reference-heavy 15-second workflow fit 6 GiB.
None of these rungs requires Triton. If a Windows Kitchen build lacks its
optional fixed-workspace W8 entry point, large aligned contractions use the
bundled CUTLASS BF16-output kernel instead of allocating a full INT32 matrix.
On sm80+, that kernel uses native `m16n8k32` INT8 Tensor Core instructions and
caches the fastest of three CUTLASS tile schedules per exact M/N/K shape. Its
epilogue and the scaled SwiGLU quantizer can write directly into row-strided
destination views, removing row/channel-shard copy buffers without changing
rounding.

Some Python symbols and extension filenames still contain `turing`/`sm75` for
backward ABI compatibility. They do not select a separate H3 implementation;
device-specific MMA/copy instructions are compile-time CUDA specializations.

## Turing behavior

When a model declares BF16 inference support but ComfyUI would otherwise fall
back to FP32 on exact sm75 Tensor Core GPUs, the plugin keeps activation storage
and bundled-kernel boundaries in BF16. Reductions and other precision-sensitive
internal arithmetic remain FP32. Explicit ComfyUI dtype flags still win.

The ConvRot path reuses comfy-kitchen W8A8 and W4A4 operators and supplies a
packed W4A8 SM75 Tensor Core kernel. Its row-buffer quantizers retain completed
rows in BF16 and use FP32 only for active rotation/reduction scratch. Launches
select the largest useful tile that fits the device's opt-in shared-memory
limit; shared-memory size or resident CTA count is not an acceptance target.
MiniMax-specific integration is isolated
under `comfyui_turing_utils/adapters/minimax/`, including packed-sequence VRAM
planning for text, keyframes, and multimodal references. Wan/Bernini integration
is isolated under `comfyui_turing_utils/adapters/`; it adds batch-aware,
per-reference-padded VRAM planning and, for supported Tensor Core attention calls,
the same single-owner Q/K/V lifetime and fused RMSNorm+RoPE+INT8 preprocessing
used by H3. This includes explicitly selected Sol calls; Sol remains opt-in.
Generic dtype, attention, and fused operators remain model-independent.

The bundled Sage backend accepts FP16/BF16 Q/K/V, GQA, causal attention,
unequal sequence lengths, HND/NHD layouts, and head dimensions up to 128. FP32
callers use BF16 boundary storage and receive FP32 output. On non-Turing GPUs,
the explicit `sage` choice uses ComfyUI's registered SageAttention backend.

The default `w8a8` attention backend uses the same bundled prepared-attention
path on sm75 and newer Tensor Core GPUs. Native builds select compile-time
architecture specializations; sm80+ uses asynchronous shared-memory copies and
the matching INT8 MMA implementation without Triton.
The bundled kernel retains stable Sage's INT8 Q/K score domain,
quantizes V channel-wise to signed INT8, packs online-softmax probabilities to
unsigned INT8, and evaluates both QK and PV with Turing Tensor Cores. It
supports FP16/BF16 storage, GQA, unequal sequence lengths, head dimensions
1--128, fixed HND/NHD layouts, upper-left causal masking, and native packed
varlen `[total_tokens, heads, dim]` inputs. Arbitrary masks remain unsupported.
The dense/sparse core has native D64 and D128 specializations, pads 1--63 only to
D64 and 65--127 only to D128, and retains the original softmax scale and output
width. The dense kernel uses a route-free specialization of the Sol exact-token
core; unsupported calls fall back through the pre-existing attention override.

Sol remains an independent patch rather than a loader option because its
quality/performance policy is intentionally configurable. Connect the model
through the Sol patch node to enable it explicitly. The
kernel accepts FP16/BF16/FP32 Q/K/V, GQA, head dimensions 1--128, unequal Q/K,
and unmasked non-causal sequences; incompatible or short calls use the selected
architecture-native dense backend. Automatic semantic protection requires separate Query/K layout
metadata for unequal sequences; ambiguous single-sequence metadata falls back
instead of applying the wrong ranges.

The bundled Sol core and its protected dense W8A8 path are native on sm75,
Ampere, Ada, and Hopper when those architectures are included in the kernel
build. Explicit Sage remains exact-sm75 and uses installed SageAttention on
newer GPUs. The extension filenames retain their historical `_sm75` suffix as
a Python ABI name; it no longer describes the only cubin that can be built.

Online Sol routing on Ampere or newer requires kernel package 0.28.0.
Adapter-protected Query blocks run through the
selected exact dense backend, while every sparse Query keeps protected modality
blocks as exact K/V sinks. Selected blocks reuse stable Sage's INT8 Tensor Core
QK path. Routing and exact selected-block QK both derive from the
same prequantized INT8 Q/K tensors and scales. Exact proxy/correction scores
remain in the post-Hadamard INT8 score domain, while route centroids are
inverse-transformed to the pre-Hadamard basis before diagonal threshold
statistics are formed. The orthogonal transform preserves centroid dot
products without estimating per-channel variance in the mixed basis. Each
Q-to-K-centroid Tensor Core score is reused for route selection and
skipped-block online-softmax correction, while original V means remain in the
value approximation.
Official-style `1x64` is the default; optional `2x32` improves bimodal
skipped-block fidelity without changing routing.

Kernel 0.23.0 retains the current ComfyUI attention tensor-container lifecycle
and adds adapter-owned fused Q/K preprocessing. H3 per-head RMSNorm plus
split-half RoPE, and Wan/Bernini whole-row RMSNorm plus interleaved RoPE, feed
the production INT8 Q/K representation without materializing normalized BF16
Q/K. Dense Sage, dense W8A8, Sol, and Sol-W8A8 share this path for H3 and
Wan/Bernini self-attention; protected Sol
steps/layers use the matching dense finalizer. Raw Q/K are released after the
fused preprocessing launch, and W8A8 releases raw V after V quantization. The
D128 preprocessing CTA uses at most about 21.1 KiB static shared memory; D64
uses about 10.6 KiB.

Diagnostics use `COMFYUI_TURING_UTILS_PROFILE=0/1/2`: off (default),
one-line span summaries, or detailed profiling. Restart after changing it.
Level 0 creates no diagnostic CUDA events or synchronization. Level 2 samples
two calls in each of at most four attention/MLP buckets, including weight-wait
phases and deferred sparse-route counters. See [diagnostics](docs/diagnostics.md)
for the complete environment-variable and logging contract. Kernel 0.31
also embeds the wheel's exact CUDA architecture set and, while this profiler is
enabled at level 2, reports the specialization CUDA selected for dense/Sol attention.
With DynamicVRAM, the report reuses the existing outer sampler fence instead
of synchronizing after an inner attention call, so profiling does not break
the asynchronous weight-prefetch pipeline.
`compiled_attention=[sm75,sm86] native_arch=True` proves that the wheel contains
an exact cubin for the active device. The native report additionally includes
`binary_sm`, `ptx_compute`, registers, shared/local memory, active CTAs and
occupancy; the historical `_sm75` extension filename remains only an ABI name.

The loader now installs one stable legacy/container attention dispatcher. Sol
and SLA change only its immutable strategy configuration, while every
ModelPatcher branch binds its resolved prepared executor directly for the fused
QKV hot path. They do not stack another model-side attention implementation or
require separate patched model branches for full and partial denoise samplers.
The selected dense backend is inherited:
`w8a8` uses the signed-V/unsigned-probability Tensor Core path for selected
exact blocks, while `sage` and `sdpa` use the FP16 sparse core. Dense protected
steps/layers remain on the selected Sage or SDPA backend. Skipped-block
correction keeps original V centroids and FP32 online state, so the numeric
path does not alter routing. Both variants keep a 64-query
tile. Native D64 uses 16 KiB dynamic shared memory, while D128 uses 32 KiB. The
automatic logical K schedule uses 64 tokens for short K and two sequential
64-token stages for K above 1024.
These are the current production geometries, not global resource limits.

On architectures where `sage` delegates to ComfyUI's registered SageAttention,
the runtime exposes a floating prepared-attention finalizer. It consumes the
already-projected Q/K/V exactly once, applies the model-published RMSNorm/RoPE
contract, and calls Sage directly. Dense Sol/SLA protection therefore does not
fall back through the original model attention or repeat QKV projection. The
same bridge covers explicit SDPA and third-party-loader fallback paths.

Sol keeps the first step of every sampler invocation and the first two
transformer layers dense by default. If
`dense_prefix_layers + dense_suffix_layers` reaches or exceeds
the runtime layer count, every layer dispatches directly to the selected dense
W8A8, Sage, or SDPA backend and skips all Sol preprocessing.

The `sdpa` option keeps ComfyUI's `AttentionTensorContainer` ownership path.
On exact-sm75, BF16 Q/K/V are consumed and converted one at a time to FP16 before
calling PyTorch SDPA, avoiding its slow BF16 math fallback while bounding the
period where floating input copies overlap. The output is restored to BF16.

The threshold and fixed +/- one-block local neighborhood execute inside each
attention CTA. Only compact, head-independent dense-Query and exact-KV policy
masks are stored globally; no full route map or follow-up popcount kernel is
materialized. MiniMax H3 publishes complete text, reference-image, reference-
video-anchor, reference-video-interior, reference-audio, target-audio, and
target-video spans. Reference-video first/last latent frames follow the image
sparsity switch; its interior follows the video switch. Defaults remain
image=false, video=true, audio=false. The current D64/D128 variants use 16/32
KiB respectively; larger candidates are accepted only when they improve real
SM75 latency without spilling. A40 compute_75 direction
tests validate numerical behavior and speed; final quality, occupancy, and
throughput still require an actual Turing GPU.

See [`docs/operator-support.md`](docs/operator-support.md) for the operator and
feature matrix, [`docs/turing-runtime.md`](docs/turing-runtime.md) for dispatch
and validation details, [`docs/architecture.md`](docs/architecture.md) for the
Python/kernel layering, and [`docs/logging.md`](docs/logging.md) for stable log
components and severity rules. Experimental Sage1/Sage2 sources are not
installed or exposed by loader nodes.

## Kernel validation

```bash
COMFYUI_TURING_UTILS_ARCH_LIST="7.5+PTX" \
python -m pip install -v --no-build-isolation -e ./kernel
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark
python kernel/scripts/validate_compatible.py --device cuda:0 --benchmark --sol
python kernel/scripts/release_gate.py --build --device cuda:0
python kernel/scripts/benchmark_backends.py --device cuda:0 --suite all
python kernel/scripts/diagnose_runtime.py --device cuda:0
python kernel/scripts/benchmark_arch_matrix.py --devices 0,1 --suite all
```

Compatible A40 runs validate numerical behavior and allocation shapes but do
not replace final exact-sm75 occupancy and end-to-end testing.
For native A40 validation, build with
`COMFYUI_TURING_UTILS_ARCH_LIST="8.6"`; this emits sm86 cubins and enables the
Ampere async-copy and INT8 MMA specializations rather than JITing compute_75.

`diagnose_runtime.py` reports hardware shared-memory limits, installed kernel
ABI features, and live allocator state as JSON. `benchmark_arch_matrix.py`
runs identical arguments serially on multiple local GPUs and writes one JSON
artifact, so a 2080 Ti and 3070 build can be compared without mixing warmups,
shapes, or backend scope. Capability checks inspect the compiled extension's
real symbols as well as its Python version, so a stale editable-build binary
is reported and safely excluded from scheduling instead of failing mid-run.

## License

Apache-2.0. See `kernel/LICENSE`, `kernel/NOTICE`, and `kernel/LICENSES/`.
