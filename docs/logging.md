# Runtime logging

Every plugin-owned message uses the `comfyui-turing-utils.<component>` logger
hierarchy and carries a visible `[Turing/<component>]` prefix. This keeps the
plain ComfyUI console readable while preserving standard Python filtering.

| Prefix | Owner |
|---|---|
| `loader` | checkpoint discovery, construction, and format summaries |
| `precision` | dtype selection and logical weight normalization |
| `dispatch` | model-local operator policy installation |
| `quantization` | backend registration and quantized execution preparation |
| `attention` | dense and sparse attention decisions |
| `memory` | DynamicVRAM boundaries and residency behavior |
| `minimax.*` | H3 layout, policy, fusion, cache, VAE, and upscaler details |
| `wan`, `wan.layout`, `bernini` | model adapter decisions |
| `stage` | Stage Barrier compilation, planning, and execution |
| `profile` | opt-in CUDA phase and workflow timeline reports |

INFO records one-time configuration and policy decisions. WARNING identifies a
degraded path, invalid configuration, or diagnostic failure. Per-call details
belong at DEBUG unless an explicit profiler is enabled. Decision messages use
stable `name=value` fields so separate runs can be compared.

The `dispatch` installation record means only that a MODEL or CLIP loaded by a
Turing Utils loader has the scoped policy. It does not change Kitchen's global
priority. Within that scope, Kitchen selects a Turing Utils operation only when
its complete dtype, device, shape, layout, and ABI constraints pass. An
unsupported call returns to Kitchen's normal dispatcher before execution. An
exception raised after a selected implementation starts is never rerouted.

Kernel-side profile lines come from the independently installed CUDA package
and retain their existing `[Turing kernel profile]` prefix.
