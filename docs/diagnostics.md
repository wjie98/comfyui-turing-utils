# Runtime diagnostics and environment variables

Use ComfyUI's logging configuration. The plugin installs no handlers, log files,
or independent log-level control. Python messages use the
`comfyui-turing-utils` logger and a short `[Turing/component]` prefix.

## Runtime settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `COMFYUI_TURING_UTILS_PROFILE` | `0` | `0`: off; `1`: concise timing summaries; `2`: detailed timings, native attention resources and SeC diagnostics |
| `COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE` | `auto` | Automatic H3 scheduling, or explicit `throughput` / `balanced` policy |
| Chat API-key environment reference | None | A user-selected `$NAME` or `${NAME}`; credentials must not be logged |

Set the profile level before starting ComfyUI; restart after changing it.
Invalid levels warn once and disable profiling. Level 0 creates no diagnostic
CUDA events, memory queries or synchronization. Algorithm-required operations
and explicitly enabled node diagnostics are unaffected.

Level 1 emits one line per measured span, not per layer. Level 2 additionally
samples two calls per attention/MLP bucket, up to four buckets per process.
It may be verbose and is intended for short diagnostic runs. Native CUDA
resource reports are written directly to stderr, only at level 2. Rebuild the
kernel when updating from the old profiling interface.

Sampler reports reuse the existing outer CUDA fence where available.
Diagnostics around reference encoding and upscaling intentionally synchronize
at span boundaries. Profiling can therefore affect performance; benchmark
normal throughput with level 0. Wall-minus-CUDA time is a residual estimate,
not a measurement of PCIe transfer time.

## Log policy

| Level | Content |
| --- | --- |
| INFO | Concise model/configuration summaries; explicitly requested profiling |
| DEBUG | Dispatch shapes, memory planning, fusion installation and stage details |
| WARNING | Actual unsupported paths, failures or invalid settings |
| ERROR | Operations that cannot complete |

Repeated attention fallback warnings are deduplicated by reason rather than
tensor shape. Exceptions retain their diagnostic context. Upstream ComfyUI,
Torch, Transformers and other dependencies retain their own logging behavior.

## Removed runtime settings

`PROFILE_CALLS`, `PROFILE_BUCKETS`, `TIMELINE`,
`H3_ACTIVATION_CHUNK_ROWS`, `H3_QKV_CHUNK_ROWS`, `H3_MLP_CHUNK_ROWS`,
`H3_HEAD_GROUP`, and `H3_FFN_CHUNK_CHANNELS` (all previously prefixed
`COMFYUI_TURING_UTILS_`), and `SEC_DEBUG`, are no longer controls.
Their presence produces one migration warning listing names, never values.
H3 row/head/channel splitting is automatic; tests and benchmarks can pass
explicit overrides to the internal policy functions.

## Build-only settings

These are not inference knobs:

- `COMFYUI_TURING_UTILS_ARCH_LIST`: target SM architectures; falls back to
  standard `TORCH_CUDA_ARCH_LIST`, then detected devices.
- `COMFYUI_TURING_UTILS_CUTLASS_AUTO_DOWNLOAD` (default enabled),
  `COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR`,
  `COMFYUI_TURING_UTILS_CCCL_INCLUDE_DIR`: build dependency locations.
- `COMFYUI_TURING_UTILS_HOST_CXX_STANDARD`,
  `COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD`,
  `COMFYUI_TURING_UTILS_CUDAHOSTCXX`: compiler overrides.
  The redundant `COMFYUI_TURING_UTILS_CXX_STANDARD` alias was removed.
- Standard toolchain variables such as `CUDA_HOME`, `CUDA_PATH`,
  `CONDA_PREFIX`, `CUDAHOSTCXX`, `MAX_JOBS` and Windows `VSLANG` remain
  owned by the build environment. Instance launchers also own device/cache
  variables; these are not plugin user settings.

## Maintenance

New diagnostics must use the shared logger and profile level. Guard expensive
statistics before computing them, not just before printing. Do not add a
separate environment variable for every tuning experiment. Keep stable node
parameters on nodes and benchmark overrides inside benchmark/policy APIs.
