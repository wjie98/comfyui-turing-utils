# Development validation

Run tests through the owning instance's `ops/test-dev.sh`. The launcher selects
the `didv` environment, development GPU, CUDA headers, driver preload, and
instance-local temporary/cache directories. It never starts or changes the
production instance. The plugin and installed kernel remain separate packages.

## Daily work

```bash
./ops/test-dev.sh -q                         # quick, based on working-tree changes
./ops/test-dev.sh --area api -q        # explicitly selected quick area
./ops/test-dev.sh tests/test_job_progress.py -q
./ops/test-dev.sh --list                     # selection only; no runtime imports
```

Explicit pytest files, directories, and node IDs run exactly those targets.
They do not add the whole `tests` directory. `-k` filters the selected targets;
it does not expand a quick selection into the full suite. Quick areas are
`api`, `media`, `prompt`, `attention`, `sampling`, `quantization`,
and `kernel`; `--area` may be repeated. Explicit areas override Git-based
selection. Git selection includes staged, unstaged, renamed, and untracked
files. Unknown runtime modules broaden the quick selection.

The selector chooses paths before pytest collection. The pytest root and
configuration boundary are `tests`, so a pure unit test does not initialize
the ComfyUI plugin entry point. Runtime integrations must initialize their own
dependencies instead of relying on another test's import side effects.

Quick selection includes shared interface contracts, related domain tests,
and small CUDA boundary checks when quantization/kernel code changes. Large
native-queue scenarios use representative cases in quick mode and retain all
cases in full mode. Changed test files run themselves unless an explicit quick
sample is declared in `scripts/test_suite.py`. The report prints the selected
paths and why they were chosen.

## Full validation and release gates

```bash
./ops/test-dev.sh --full -q
./ops/test-dev.sh --kitchen -q -k TestNVFP4Bounds
```

Full validation runs all Python cases and enables the opt-in Veda CUDA cases.
Hardware coverage is recorded for the device and operating system used.

Run full validation after cross-domain refactoring, serialized-interface or
execution/cache ownership changes, dependency/ABI upgrades, and before a
production deployment. Ordinary edits use related quick tests. Record timings
with the necessary validation run instead of rerunning everything for timing.

CUDA source, allocator/ownership, or compiler changes additionally require
Compute Sanitizer and the relevant numerical/architecture checks. Performance
benchmarks are separate from correctness tests. The NVFP4 tail sentinels and
independent CPU reference remain in `tests/gpu/test_nvfp4_bounds.py`; an ordinary
successful model output cannot replace these checks.

## Coverage ownership

HTTP tests exercise the native queue, lifecycle, output downloads and cleanup.
H3 curve tests use their policy owner separately from numerical task-execution
tests. The root conftest exposes shared fixtures without starting ComfyUI.

Keep one serialized-interface contract and direct tests for races, path/file
ownership, cleanup, sparse protection, masks, and numeric buffer boundaries.
Remove a source-writing or forwarding assertion when an existing behavioral
test covers the same contract. Do not remove a boundary check merely because
an end-to-end success case also uses the operation.
