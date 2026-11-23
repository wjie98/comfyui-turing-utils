# Plugin architecture

The repository has two intentionally independent lifecycles:

- `comfyui_turing_utils/` is the ComfyUI plugin implementation. Python-only
  updates do not compile or JIT CUDA code.
- `kernel/` builds and installs `comfyui-turing-utils-kernel`. Its extension
  names and public Python API are an ABI boundary for the plugin.

The maintained source tree is intentionally shallow:

```text
comfyui-turing-utils/
├── __init__.py                  # ComfyUI entry point
├── comfyui_turing_utils/
│   ├── attention/               # dense/sparse backends, layout and patches
│   ├── quantization/            # ConvRot formats, dispatch and fusions
│   ├── loading/                 # ComfyUI model/CLIP construction orchestration
│   ├── adapters/                # MiniMax, Wan and Bernini integration
│   ├── nodes/                   # thin ComfyUI schemas
│   ├── media/                   # sampling, geometry, masks, timelines and files
│   ├── prompt/                  # multimodal request construction and transport
│   ├── job_api/                 # temporary API jobs and native queue integration
│   ├── runtime/                 # device/kernel capability resolution and diagnostics
│   ├── hardware.py
│   ├── kernel_api.py            # independent-kernel boundary
│   ├── log.py                   # component logger hierarchy and console labels
│   ├── precision.py
│   ├── profiling.py             # disabled-by-default bounded CUDA timing
│   └── registration.py          # sole node mapping table
├── kernel/
│   ├── comfyui_turing_utils_kernel/
│   │   ├── ops.py               # linear/fusion custom ops
│   │   └── turing_sage/         # attention API, quantization and scheduling
│   ├── csrc/turing/             # Turing fusions plus shared sm75+ attention
│   ├── scripts/                 # validation and benchmarks
│   └── setup.py
├── docs/
├── skills/turing-workflow/       # portable agent client and project capabilities
└── tests/
```

There are no model or node implementation modules at repository root. New
functionality belongs in one existing package above unless it introduces a
genuinely new responsibility.

## Dependency direction

```text
ComfyUI registration and nodes
        ↓
media / prompt services
        ↓
loading orchestration ─────────────→ model adapters
        ↓                              ↓
attention / quantization services ←────┘
        ↓
runtime capabilities / hardware / kernel_api
                         ↓
          comfyui-turing-utils-kernel
```

Concrete model adapters never belong in the generic attention or quantization
layers. `comfyui_turing_utils.bootstrap` is the explicit, idempotent
composition root. The ComfyUI plugin entry point invokes it before node
registration; importing an implementation submodule has no registration side
effects.

## Python packages

| Package | Responsibility |
|---|---|
| `attention/stable.py` | backend registry, stable bundled Sage/W8A8, dtype/layout facade |
| `attention/sparse.py` | production Sol policy, exact modality protection, and W8A8/FP16 PV dispatch |
| `attention/protocol.py` | versioned tensor ownership, transform, capability, and execution contract |
| `attention/integration.py` | model-neutral projected-QKV handoff and attention-site registry |
| `attention/orchestration.py` | shared ModelPatcher/layout/executor strategy installation mechanics |
| `attention/layout.py` | versioned Query/KV modality topology contract and provider registry |
| `attention/sparse_runtime.py` | common Sol/SLA schedule state and dense fallback ownership |
| `attention/patches.py` | thin Sol/SLA strategy composition and loader-independent ModelPatcher installation |
| `attention/execution.py` | common transforms, validation, profiling, metadata and dtype fallback |
| `attention/dense.py` | dense executor construction and prepared/container/external calls |
| `attention/sol.py`, `sla.py` | distinct routing policies over the shared execution contract |
| `quantization/convrot.py` | ConvRot metadata parsing and loaded-module format inspection |
| `quantization/formats.py` | shared storage geometry, packed bit width and ConvRot format facts; no compute-dtype/device policy |
| `quantization/capabilities.py` | independent-kernel symbol and comfy-kitchen backend probes |
| `quantization/operator_scope.py` | MODEL/CLIP-local Turing Utils-first Kitchen selection scope |
| `quantization/workspace.py` | pure, model-independent workspace formulas |
| `quantization/dispatch.py` | W8A8/W6A8/W4A8/W4A4 activation quantization and GEMM execution |
| `quantization/backend.py` | scoped Kitchen registration and operator constraints |
| `quantization/preflight.py` | bounded, device-specific numerical capability checks |
| `quantization/fusions.py` | model-independent fused activation and normalization operations |
| `loading/convrot.py` | filesystem discovery, Comfy model construction, runtime preparation, and adapter installation |
| `runtime/capabilities.py` | immutable device facts plus independently versioned kernel feature probes |
| `runtime/stage_barrier.py` | prompt-local dependency rounds, phase-label rendezvous and incremental lazy/dynamic barrier planning |
| `runtime/stage_barrier_prompt.py` | pre-validation expansion of visual multi-port barriers into independent unary cache/execution paths |
| `runtime/sampling.py` | execution-local task step progress and native solver-loop bridge; no MODEL/global mutation or duplicated solver math |
| `adapters/minimax/sampling.py` | H3 task preflight, complete first sampling, clean upscale/inpaint handoff and second restart; one attention/preview timeline |
| `adapters/minimax/sampling_sigmas.py`, `sampling_masks.py` | curve policy and conservative spatial edit coverage, separate from task execution |
| `adapters/memory.py` | common quantized workspace scan and BaseModel memory-hook installation |
| `adapters/minimax/policy_config.py` | environment/config parsing only; no live CUDA state |
| `adapters/minimax/memory_state.py` | live allocator/DynamicVRAM observation and bounded reclaim requests |
| `adapters/minimax/memory_planning.py` | ComfyUI packed-shape and staged-workspace memory hooks |
| `adapters/minimax/activation_policy.py` | tier/chunk/head/channel decisions using the memory-state service |
| `adapters/minimax/acceleration.py` | adapter installation and block-level orchestration |
| `adapters/minimax/execution.py` | shared casting, scaled-linear quantization, runtime-plan ownership and audit |
| `adapters/minimax/attention_ops.py`, `mlp.py` | H3 attention and MLP execution respectively |
| `adapters/minimax/references.py` | reference records, shape validation and presentation helpers |
| `adapters/minimax/reference_execution.py` | visual/audio/semantic encoding and conditioning assembly |
| `adapters/krea2.py` | Krea2 Identity Edit reference fitting, grounded conditioning and centered reference RoPE patching |
| `adapters/wan.py` | Wan/Bernini packed-context planning and supported self-attention preprocessing |
| `adapters/wan_layout.py` | loader-independent Wan/Bernini self-attention sequence semantics |
| `adapters/bernini.py` | Bernini context-window and absolute-RoPE integration |
| `nodes/` | thin ComfyUI schemas and calls into the implementation packages |
| `media/` | reusable image/video sampling, geometry, mask mapping, file and timeline operations |
| `prompt/chat.py` | chat options, media, request/response handling and transport |
| `job_api/` | native workflow requests, temporary artifacts, progress and discovery |

`hardware.py` owns architecture facts. `runtime/capabilities.py` combines those
facts with operator-level ABI probes without launching CUDA. `kernel_api.py` is the only module
allowed to import the independently installed kernel package. `registration.py`
is the only node mapping table.

Testing entry points, quick selection, full matrices, and coverage ownership are
documented in [testing.md](testing.md). Tests do not initialize the plugin root
implicitly; scoped fixtures own native runtime setup.

H3 sampling separates task progress from numerical solver state. One
`H3SamplingTask` owns both stages and their clean upscale/mask handoff;
`NativeStepSampler` delegates model loading, hooks, inpainting and solver math
to ComfyUI. A task-local solver namespace observes the native iteration loop
because native preview callbacks have different timing across solver families.
It never changes ComfyUI module globals. Sparse strategies read Python step
metadata, while H3 retains each stage's real `sample_sigmas`. All native sampler
families are regression-tested; changed loop contracts fail explicitly.

Inside the kernel package, `turing_sage/records.py` owns immutable prepared
tensor contracts, `sparse_policy.py` owns cached route construction, and
`scheduling.py` owns resource-based tile selection. CUDA remains a single
translation unit for the production sparse kernel; textual `.cuh` components
under `csrc/turing/sage/sparse/` separate route machinery without changing
template instantiation, launch ABI, or generated architecture coverage.

## Compatibility

ComfyUI workflow compatibility is governed by the stable
`NODE_CLASS_MAPPINGS` keys, input names, and defaults rather than Python
filenames. Implementation and tests import canonical `comfyui_turing_utils`
paths. No top-level Attention facade, dynamic module proxy or ConvRot loading
alias is maintained. Tests patch dependencies in the module that actually uses
them rather than synchronizing globals through a compatibility facade.

Sol/SLA inherit one dense-backend configuration from the model runtime; there
is no second `use_w8a8` override at the strategy boundary. Their scheduling and
dense fallback contracts remain shared, while their distinct routing policies
stay explicit. Ordinary node input names, order and defaults are unchanged.

Sparse attention remains explicit and is never selected by a loader backend.
Model-specific topology is installed through the attention-layout provider
registry, so the official ComfyUI loader and ConvRot loader follow the same path.
Model-side fused Q/K handoff is installed through a separate attention-site
registry. This lets dense and Sol backends request the same H3 or Wan/Bernini
integration without importing either model family into generic attention code.

## Media and prompt ownership

The same sampling and resizing code serves contact sheets, multimodal chat and
ordinary media nodes. Native model-specific encoding remains in its adapter.
Do not force distinct mask policies into one implicit broadcast rule:

- ROI crop/stitch validates per-frame geometry and its own resize policy.
- Timeline concatenation may explicitly repeat one image mask across frames.
- Latent masks map pixel frames onto model-specific temporal groups.

These share validation primitives where their semantics agree, not a blanket
"resize and repeat" implementation that can hide incorrect frame counts.

Chat transport belongs in `prompt/chat.py`; its schema and NodeOutput assembly
belong in `nodes/multimodal_chat.py`. H3 reference encoding follows the same
service/schema separation. Import data records from their owning adapter, not
from the node module that happens to consume them.

## Maintaining the structure

`job_api/workflow.py` binds native API prompts to job-owned uploads and saver paths.
The service submits directly to the native queue and observes its history and progress events.
`skills/turing-workflow/scripts/turing_client/` owns the portable HTTP client,
local typed capability rendering, schema checks and resumable run snapshots, without
importing ComfyUI. The server's `job_api` package resolves its existing lightweight
module imports to that implementation; there is no second client to maintain.
Project assets live below one `turing/` subdirectory. Usage notes evolve per
capability and are snapshotted with each run; execution verification tracks the
source graph/contract hash. Remote status and local download/cleanup activity are
recorded separately. The skill guides authoring and result review using this client.

`job_api/discovery.py` owns the small server manifest and searchable node
summaries. `/turing/server` includes usage guidance; full node schemas and model
lists stay on native ComfyUI routes. Legacy input enumeration is excluded from
catalog searches, V3 metadata comes from the native schema, and runtime facts
reuse capability resolution without running models or kernels. Protocol limits
are shared with submission handling, and discovery remains readable when job
cleanup prevents execution.

Use the owning implementation module directly. Do not add compatibility
re-exports of private helpers to make old tests pass; patch a dependency where
its consumer actually looks it up. A coherent numerical operation may remain
large: splitting a CUDA mainloop or H3 attention executor by arbitrary line count
is not a useful abstraction. Python ownership moves alone do not require a kernel
rebuild. The subsequent FP16 numerical-contract convergence (0.44.0) and grouped
W6 support (0.45.0) do require updating the independent kernel.

Weight storage is identified once in `quantization/formats.py`. Memory planning
uses storage facts; fusion eligibility adds its BF16 and device requirements.
Do not reuse a compute-eligibility predicate as a memory-format classifier.
Grouped W4 stores K/2 bytes and a 16-entry codebook. Uniform W6 stores
3K/4 bytes without a codebook. The loader reads logical K from the group-scale
grid and normalizes legacy W6 labels to the format supported by ComfyUI.
Both reuse the same ConvRot256 activation and INT8 contraction family. New
formats should extend that geometry/dispatch boundary, not duplicate loaders
or add model-specific tests inside generic services.

Continuation composition belongs in `media/timeline.py`, hard latent composition
in `media/latent_masks.py`, and H3-specific audio-prefix masking in
`adapters/minimax/latent_noise.py`. Node methods only adapt these services to
`NodeOutput`; their schemas, ordered ports and defaults remain unchanged.

Python formatting and unambiguous static checks are defined in `ruff.toml`.
Vendored SeC source and independent third-party repositories are excluded. Run:

```text
ruff format --check comfyui_turing_utils tests
ruff check comfyui_turing_utils tests
ops/test-dev.sh -q --tb=short
```

`tests/fixtures/node_contracts.json` records existing node interfaces,
including ordered input names, DynamicCombo/Autogrow branches and defaults.
The baseline schema bodies were compared as ASTs before recording order.
`test_package_architecture.py` enforces service dependencies and the kernel
import boundary. Python tests cover local execution and model/LoRA reuse.

The pre-refactor GPU baseline had four failures: two incomplete device mocks,
a FP32 fallback test comparing CUDA against eager, and the 36-block FP16 VAE
bitwise native-rotation comparison. The first three are corrected without
changing runtime math or loosening tolerances. Following a separate numerical
contract review, kernel 0.44.0 replaces the split FP16 activation path with fused
FP32 activation, rotation, scale and quantization. Input/output storage remains
FP16 and the INT8 GEMM/weight layout is unchanged. Tests now check the agreed
contract against an independent dense FP64 rotation oracle, including zero and
subnormal rows; they no longer claim bitwise identity with Kitchen's intermediate
FP16 rounding. The 36-block decoder test checks the high-precision contract and
retains the strict native-result comparison after the operator scope exits.
