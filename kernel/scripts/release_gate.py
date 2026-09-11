#!/usr/bin/env python3
"""Cross-platform release gate for the plugin and independent kernel package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "kernel"
COMFYUI_ROOT = ROOT.parents[1]
EXPECTED_VERSION = "0.42.0"


def _run(command: list[str], *, cwd: Path = ROOT, env=None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _static_gate() -> None:
    version_sources = {
        KERNEL / "pyproject.toml": r'version\s*=\s*"([^"]+)"',
        KERNEL / "setup.py": r'version\s*=\s*"([^"]+)"',
        KERNEL / "comfyui_turing_utils_kernel" / "__init__.py": r'__version__\s*=\s*"([^"]+)"',
    }
    for path, pattern in version_sources.items():
        matches = re.findall(pattern, path.read_text(encoding="utf-8"))
        if EXPECTED_VERSION not in matches:
            found = ", ".join(matches) if matches else "missing"
            raise RuntimeError(f"kernel version mismatch in {path}: {found}")

    production_files = tuple((ROOT / "comfyui_turing_utils").rglob("*.py")) + tuple(
        (KERNEL / "comfyui_turing_utils_kernel").rglob("*.py")
    )
    retired = ("frame_sparse", "FrameSparse", "SageFrameSparse")
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        for marker in retired:
            if marker in text:
                raise RuntimeError(f"retired frame-sparse marker {marker!r} remains in {path}")

    maintained_text = production_files + tuple((ROOT / "docs").rglob("*.md")) + (
        ROOT / "README.md",
        KERNEL / "README.md",
    )
    for path in maintained_text:
        text = path.read_text(encoding="utf-8").lower()
        if re.search(r"svd[ _-]?int4|svdlora", text):
            raise RuntimeError(f"retired SVDInt4 marker remains in {path}")

    cuda = (KERNEL / "csrc/turing/sage/sol_sparse_cuda_sm75.cu").read_text(encoding="utf-8")
    route_compaction = (
        KERNEL / "csrc/turing/sage/sparse/route_compaction.cuh"
    ).read_text(encoding="utf-8")
    if '#include "sparse/route_compaction.cuh"' not in cuda:
        raise RuntimeError("sparse attention does not include route compaction component")
    cuda_components = cuda + "\n" + route_compaction
    for marker in (
        "AttentionGeometry<128>::kAttentionSharedBytes <= 64 * 1024",
        "key_tile_tokens == 64 || key_tile_tokens == 128",
        "key_tile_tokens / kBlockTokens",
        "quantize_varlen_value_kernel",
        "build_varlen_value_offsets_kernel",
        "compact_route_words<S::kSelectedCapacity>",
        "next_compact_route_block<S::kSelectedCapacity>",
        "next_shared_route_block",
        "shared_selected_value_int8_next",
        "value_ping_pong",
        "kCompactionScratchWords",
        "__shfl_up_sync",
        "__ballot_sync",
        "AttentionStorage<HeadDim, SparseValuePipeline>",
        "sparse_v_pipeline=",
        "current_cuda_device_major() >= 8",
        "union ScoreStorage",
        "ResidualSubblocks == 1 || ResidualSubblocks == 2",
        "KeyStages == 1 || KeyStages == 2",
        "key_tile_tokens / kBlockTokens == KeyStages",
    ):
        if marker not in cuda_components:
            raise RuntimeError(f"missing SM75 attention resource/ABI gate: {marker}")
    gemm = (KERNEL / "csrc/turing/w4a8.cu").read_text(encoding="utf-8")
    for marker in (
        "properties->major >= 8 && n >= 16384",
        "run_ampere_int8_tile<128, 256, 64, 64, 64, 64, 3>",
        "run_int8_tile<128, 256, 64, 64>",
    ):
        if marker not in gemm:
            raise RuntimeError(f"missing deterministic GEMM dispatch gate: {marker}")
    for marker in ("run_auto_tuned", "GEMM_TUNE", "GEMM_CACHE"):
        if marker in gemm:
            raise RuntimeError(f"retired GEMM tuning marker remains: {marker}")
    cp_async = (KERNEL / "csrc/turing/sage/cp_async.cuh").read_text(encoding="utf-8")
    fallback = cp_async.split("#else", 1)[1].split("#endif", 1)[0]
    if "__threadfence_block" in fallback or "__syncthreads" in fallback:
        raise RuntimeError("SM75 synchronous-copy fallback contains a duplicate barrier")
    header = (KERNEL / "csrc/turing/sage/attn_cuda_sm75.h").read_text(encoding="utf-8")
    if header.count("int key_tile_tokens") != 4:
        raise RuntimeError(
            "C++/pybind attention ABI does not expose all key-tile arguments"
        )
    binding = (KERNEL / "csrc/turing/sage/pybind_sm75.cpp").read_text(
        encoding="utf-8"
    )
    mapped_summary = "sol_w8a8_precompute_mapped_summaries"
    if mapped_summary not in header or mapped_summary not in cuda or mapped_summary not in binding:
        raise RuntimeError("mapped Sol summary ABI is incomplete")
    mapped_fp16 = "sol_sparse_online_int8_f16_mapped_attn"
    if mapped_fp16 not in header or mapped_fp16 not in cuda or mapped_fp16 not in binding:
        raise RuntimeError("mapped FP16/BF16 Sol attention ABI is incomplete")
    if "load_half_tile_mapped" not in cuda:
        raise RuntimeError("mapped FP16/BF16 exact-V tile load is missing")
    setup_source = (KERNEL / "setup.py").read_text(encoding="utf-8")
    for marker in (
        "CUDA_TOOLKIT_VERSION = _cuda_toolkit_version()",
        "def _normalize_cxx_standard(",
        "def _visible_cuda_arches()",
        'os.environ.get("TORCH_CUDA_ARCH_LIST")',
        'os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH_LIST',
        'return "c++20" if cuda_version is None or cuda_version >= (12, 0) else "c++17"',
        '"COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD", DEFAULT_CXX_STANDARD',
    ):
        if marker not in setup_source:
            raise RuntimeError(
                "CUDA-aware CUTLASS C++ language selection is incomplete"
            )
    wheel_builder = (KERNEL / "scripts/build_wheel.py").read_text(encoding="utf-8")
    for marker in (
        'default=None',
        'defaults to all visible supported GPUs',
        "def configure_cuda_home(",
        'prefix / "Library" if platform.system() == "Windows" else prefix',
        'env.setdefault("CUDA_HOME", str(root.resolve()))',
    ):
        if marker not in wheel_builder:
            raise RuntimeError(
                "the standalone wheel builder cannot discover Conda NVCC"
            )
    fused_source = (KERNEL / "csrc/turing/sage/fused.cu").read_text(
        encoding="utf-8"
    )
    fused_header = (KERNEL / "csrc/turing/sage/fused.h").read_text(
        encoding="utf-8"
    )
    for retired_fp8_helper in (
        "TransposePadPermuteKernel",
        "MeanScaleKernel",
        "transpose_pad_permute_cuda",
        "scale_fuse_quant_cuda",
        "mean_scale_fuse_quant_cuda",
    ):
        if retired_fp8_helper in fused_source or retired_fp8_helper in fused_header:
            raise RuntimeError(
                f"retired Sage2/FP8 helper remains in the production build: {retired_fp8_helper}"
            )
    fused_binding = (KERNEL / "csrc/turing/sage/pybind_fused.cpp").read_text(
        encoding="utf-8"
    )
    qk_preprocess = (
        KERNEL / "csrc/turing/sage/qk_preprocess.cu"
    ).read_text(encoding="utf-8")
    overlap_source = (KERNEL / "csrc/turing/sage/overlap_blend.cu").read_text(
        encoding="utf-8"
    )
    for path, source, marker in (
        (KERNEL / "setup.py", setup_source, '"csrc/turing/sage/qk_preprocess.cu"'),
        (KERNEL / "setup.py", setup_source, '"csrc/turing/sage/overlap_blend.cu"'),
        (
            KERNEL / "csrc/turing/sage/fused.h",
            fused_header,
            "quant_qk_rms_rope_int8_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/pybind_fused.cpp",
            fused_binding,
            "quant_qk_rms_rope_int8_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/pybind_fused.cpp",
            fused_binding,
            "quant_qk_rms_rope_int8_mapped_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/pybind_fused.cpp",
            fused_binding,
            "gather_value_int8_mapped_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/pybind_fused.cpp",
            fused_binding,
            'm.attr("qk_preprocess_protocol_schema") = 3',
        ),
        (
            KERNEL / "csrc/turing/sage/qk_preprocess.cu",
            qk_preprocess,
            "at::Tensor query_freqs,",
        ),
        (
            KERNEL / "csrc/turing/sage/qk_preprocess.cu",
            qk_preprocess,
            "at::Tensor key_freqs,",
        ),
        (
            KERNEL / "csrc/turing/sage/fused.h",
            fused_header,
            "overlap_blend_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/pybind_fused.cpp",
            fused_binding,
            "overlap_blend_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/fused.h",
            fused_header,
            "overlap_accumulate_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/pybind_fused.cpp",
            fused_binding,
            "overlap_accumulate_cuda",
        ),
        (
            KERNEL / "csrc/turing/sage/overlap_blend.cu",
            overlap_source,
            "overlap_accumulate_kernel",
        ),
        (
            KERNEL / "csrc/turing/sage/fused.h",
            fused_header,
            "at::Tensor anchor_values",
        ),
    ):
        if marker not in source:
            raise RuntimeError(
                f"missing fused preprocessing/overlap ABI marker in {path}"
            )
    core_api = (KERNEL / "csrc/kernel_api.h").read_text(encoding="utf-8")
    core_binding = (KERNEL / "csrc/bindings.cpp").read_text(encoding="utf-8")
    for path, source, marker in (
        (
            KERNEL / "csrc/kernel_api.h",
            core_api,
            "turing_swiglu_int8_convrot_quantize_scaled",
        ),
        (
            KERNEL / "csrc/bindings.cpp",
            core_binding,
            'm.def("turing_swiglu_int8_convrot_quantize_scaled"',
        ),
        (
            KERNEL / "csrc/bindings.cpp",
            core_binding,
            'm.def("turing_swiglu_int8_convrot_quantize_scaled_out"',
        ),
        (
            KERNEL / "csrc/bindings.cpp",
            core_binding,
            'm.def("turing_int8_linear_out"',
        ),
    ):
        if marker not in source:
            raise RuntimeError(f"missing activation-streaming ABI marker in {path}")
    sparse_policy = (ROOT / "comfyui_turing_utils/attention/sparse.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        'segment.role in {"reference_image", "reference_video_anchor"}',
    ):
        if marker not in sparse_policy:
            raise RuntimeError(f"missing production Sol quality invariant: {marker}")
    sparse_core = (
        KERNEL / "comfyui_turing_utils_kernel/turing_sage/core.py"
    ).read_text(encoding="utf-8")
    for marker in (
        "* sparse_block_count\n        * key_block_count",
        "possible_blocks=possible_blocks",
        "def _sla_fixed_topk_indices(",
        "route_words=route_words",
    ):
        if marker not in sparse_core:
            raise RuntimeError(f"missing full-block Sol route statistic: {marker}")
    sparse_patch = (ROOT / "comfyui_turing_utils/attention/patches.py").read_text(
        encoding="utf-8"
    )
    if "bundled_turing_sol_sparse_experimental" in sparse_patch:
        raise RuntimeError("the production Sol patch still advertises an experimental ABI")
    for marker in (
        "sla_qk_block_summaries",
        "sla_build_route_words",
        "sla_sparse_online_attn",
    ):
        if marker not in cuda or marker not in header:
            raise RuntimeError(f"missing production SLA CUDA ABI marker: {marker}")
    print(f"static release gate passed (kernel ABI {EXPECTED_VERSION})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--build", action="store_true", help="Build extensions in place")
    parser.add_argument("--device", help="Run the CUDA/A40-compatible correctness gate")
    args = parser.parse_args()
    _static_gate()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(KERNEL), str(COMFYUI_ROOT), env.get("PYTHONPATH", "")),
        )
    )
    if not args.skip_tests:
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], env=env)
    if args.build:
        env.setdefault("COMFYUI_TURING_UTILS_ARCH_LIST", "7.5+PTX")
        _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=KERNEL, env=env)
        _run([sys.executable, "kernel/scripts/audit_attention_resources.py"], env=env)
    if args.device:
        _run(
            [
                sys.executable,
                "kernel/scripts/validate_compatible.py",
                "--device",
                args.device,
                "--sol",
                "--sla",
            ],
            env=env,
        )
    print("release gate passed")


if __name__ == "__main__":
    main()
