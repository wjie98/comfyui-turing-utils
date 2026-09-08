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
├── attention.py                 # sole legacy Python compatibility facade
├── comfyui_turing_utils/
│   ├── attention/               # dense/sparse backends, layout and patches
│   ├── quantization/            # ConvRot formats, dispatch and fusions
│   ├── loading/                 # ComfyUI model/CLIP construction orchestration
│   ├── adapters/                # MiniMax, Wan and Bernini integration
│   ├── nodes/                   # thin ComfyUI schemas
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
└── tests/
```

There are no model or node implementation modules at repository root. New
functionality belongs in one existing package above unless it introduces a
genuinely new responsibility.

## Dependency direction

```text
ComfyUI registration and nodes
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
| `quantization/convrot.py` | ConvRot metadata parsing and loaded-module format inspection |
| `quantization/capabilities.py` | independent-kernel symbol and comfy-kitchen backend probes |
| `quantization/operator_scope.py` | MODEL/CLIP-local Turing Utils-first Kitchen selection scope |
| `quantization/workspace.py` | pure, model-independent workspace formulas |
| `quantization/dispatch.py` | compatibility facade plus W8A8/W4A8/W4A4 quantization/GEMM dispatch |
| `quantization/fusions.py` | model-independent fused activation and normalization operations |
| `loading/convrot.py` | filesystem discovery, Comfy model construction, runtime preparation, and adapter installation |
| `runtime/capabilities.py` | immutable device facts plus independently versioned kernel feature probes |
| `runtime/stage_barrier.py` | prompt-local dependency rounds, phase-label rendezvous and incremental lazy/dynamic barrier planning |
| `runtime/stage_barrier_prompt.py` | pre-validation expansion of visual multi-port barriers into independent unary cache/execution paths |
| `adapters/memory.py` | common quantized workspace scan and BaseModel memory-hook installation |
| `adapters/minimax/policy_config.py` | environment/config parsing only; no live CUDA state |
| `adapters/minimax/memory_state.py` | live allocator/DynamicVRAM observation and bounded reclaim requests |
| `adapters/minimax/memory_planning.py` | ComfyUI packed-shape and staged-workspace memory hooks |
| `adapters/minimax/activation_policy.py` | pure tier/chunk/head/channel decisions with compatibility exports |
| `adapters/minimax/acceleration.py` | H3 attention/MLP hot-path installation and execution |
| `adapters/minimax/image_sol.py` | H3 image-mode static Sol policy and strategy composition |
| `adapters/minimax/virtual_kv.py` | H3-only five-frame validation, virtual temporal mapping, and strategy composition |
| `adapters/krea2.py` | Krea2 Identity Edit reference fitting, grounded conditioning and centered reference RoPE patching |
| `adapters/wan.py` | Wan/Bernini packed-context planning and supported self-attention preprocessing |
| `adapters/wan_layout.py` | loader-independent Wan/Bernini self-attention sequence semantics |
| `adapters/bernini.py` | Bernini context-window and absolute-RoPE integration |
| `nodes/` | thin ComfyUI schemas and calls into the implementation packages |

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
paths. The sole top-level compatibility module is `attention.py`; it preserves
the old monkey-patchable attention facade while downstream integrations migrate
to `comfyui_turing_utils.attention`.

Sparse attention remains explicit and is never selected by a loader backend.
Model-specific topology is installed through the attention-layout provider
registry, so the official ComfyUI loader and ConvRot loader follow the same path.
Model-side fused Q/K handoff is installed through a separate attention-site
registry. This lets dense and Sol backends request the same H3 or Wan/Bernini
integration without importing either model family into generic attention code.
