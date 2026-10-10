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
│   ├── workspace/               # material storage, card projection and local execution
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
├── web/                         # native-editor extensions and small shared UI helpers
└── tests/
```

There are no model or node implementation modules at repository root. New
functionality belongs in one existing package above unless it introduces a
genuinely new responsibility.

## Dependency direction

```text
ComfyUI registration and nodes
        ↓
media / prompt / workspace services
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
| `prompt/chat.py` | chat options, presentation, request/response handling and transport |
| `workspace/` | project-owned material storage, native graph projection, card validation and segment compilation |
| `web/workspace/` | request feedback, project save ordering and material-control resource ownership |
| `web/lib/` | native combo adaptation, directory selection, endpoint interaction and port reordering |

`hardware.py` owns architecture facts. `runtime/capabilities.py` combines those
facts with operator-level ABI probes without launching CUDA. `kernel_api.py` is the only module
allowed to import the independently installed kernel package. `registration.py`
is the only node mapping table.

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

Canvas retains only the native-editor API. Opening a project derives lightweight
card descriptors from one request-local project snapshot; it does not flatten
computation graphs. Insertion reads only the new card. Execution expands only
the target card and reads other cards through their published material selections.
Layout saves validate connected interfaces without expanding computation graphs.
File-format hashes, revision checks and server-side types remain authoritative.
No persistent execution-graph cache or additional model-weight cache is introduced.

Sparse attention remains explicit and is never selected by a loader backend.
Model-specific topology is installed through the attention-layout provider
registry, so the official ComfyUI loader and ConvRot loader follow the same path.
Model-side fused Q/K handoff is installed through a separate attention-site
registry. This lets dense and Sol backends request the same H3 or Wan/Bernini
integration without importing either model family into generic attention code.

## Material and prompt ownership

Media operations no longer import ComfyUI node schemas or Canvas services.
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

## Canvas boundaries

| Module | Owns |
|---|---|
| `workflow_files.py` | ordinary workflow serialization, instance files and atomic project updates |
| `projection.py` | lightweight display descriptors or explicitly requested execution expansion |
| `compiler.py`, `tasks.py` | material-boundary graph cuts and validated task creation |
| `execution.py` | fresh ordinary-node execution adapters; no model preparation cache |
| `store.py`, `runs.py` | media index, selection revisions and recoverable publication |
| `materials.py` | native media loading, import copying, serialization and selected-content decoding |
| `previews.py` | lazy still posters with bounded concurrent decoding and request deduplication |
| `directories.py`, `routes.py` | instance-owned filesystem operations and HTTP adapters respectively |
| `web/workspace/project_state.js` | per-project save queue and request-local native graph snapshots |
| `web/workspace/material_controls.js` | native controls, posters and one active on-demand player |
| `web/workspace/api.js` | API transport and native feedback |

Task compilation completes before a prepared run is recorded. Publication
records a SQLite intent before updating `canvas.json`; retry or reopening can
finish an interrupted publication. Selection revisions prevent a delayed run
from replacing a newer manual choice. Text remains inline in the canvas file;
the temporary publication payload is cleared on completion.

Model and LoRA preparation still use ComfyUI's existing owners and shared loader
identities. `workspace/cache.py` is the isolated, version-sensitive integration
with the executor's intermediate cache. Do not spread private executor calls or
global cache clearing into media, UI or model services.

## Maintaining the structure

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
The `asym_w4a8_int8` checkpoint label covers two different packings: W4 stores
K/2 bytes and a 16-entry codebook, whereas uniform W6 stores 3K/4 bytes without
a codebook. Logical K comes from the group-scale grid, not the packed width.
Both reuse the same ConvRot256 activation and INT8 contraction family. New
formats should extend that geometry/dispatch boundary, not duplicate loaders
or add model-specific tests inside generic services.

Continuation composition belongs in `media/timeline.py`, hard latent composition
in `media/latent_masks.py`, and H3-specific audio-prefix masking in
`adapters/minimax/latent_noise.py`. Node methods only adapt these services to
`NodeOutput`; their schemas, ordered ports and defaults remain unchanged.

Python formatting and unambiguous static checks are defined in `ruff.toml`.
Frontend formatting is defined in `.prettierrc.json`; formatting tools are
development tooling, not runtime dependencies.
Vendored SeC source and independent third-party repositories are excluded. Run:

```text
ruff format --check comfyui_turing_utils tests
ruff check comfyui_turing_utils tests
prettier --check 'web/**/*.js' 'tests/browser/*.mjs'
ops/test-dev.sh -q --tb=short
```

`tests/fixtures/node_contracts.json` records all 57 maintained node interfaces,
including ordered input names, DynamicCombo/Autogrow branches and defaults.
The baseline schema bodies were compared as ASTs before recording order.
`test_package_architecture.py` enforces service dependencies and the kernel
import boundary. Python tests cover local execution and model/LoRA reuse.
Classic UI geometry, connections, subgraph reloads, project saves and local
execution are also exercised in `tests/browser/`.

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
