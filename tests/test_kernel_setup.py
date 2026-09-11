from __future__ import annotations

import os
import platform
import runpy
import sys
import tempfile
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 build environments
    import tomli as tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PLUGIN_ROOT / "kernel" / "setup.py"


class KernelSetupTest(unittest.TestCase):
    @staticmethod
    def _extension(*, name, **kwargs):
        return SimpleNamespace(name=name, kwargs=kwargs)

    def _run_windows_setup(
        self,
        conda_prefix: Path,
        *,
        arch_list: str = "8.6",
        cuda_version: str = "12.8",
        extra_environment: dict[str, str] | None = None,
    ):
        environment = {
            "CONDA_PREFIX": str(conda_prefix),
            "COMFYUI_TURING_UTILS_ARCH_LIST": arch_list,
        }
        environment.update(extra_environment or {})
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch("torch.utils.cpp_extension.CUDA_HOME", None),
            mock.patch("torch.version.cuda", cuda_version),
            mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
            mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
            mock.patch("shutil.which", return_value=None),
            mock.patch("setuptools.setup") as setup,
        ):
            runpy.run_path(str(SETUP_PATH), run_name="__turing_utils_windows_setup_test__")
        return setup.call_args.kwargs["ext_modules"]

    @staticmethod
    def _make_cutlass_headers(include_dir: Path) -> None:
        for relative in (
            "cutlass/cutlass.h",
            "cutlass/gemm/device/gemm_universal_adapter.h",
            "cutlass/epilogue/threadblock/fusion/visitors.hpp",
            "cute/tensor.hpp",
        ):
            header = include_dir / relative
            header.parent.mkdir(parents=True, exist_ok=True)
            header.touch()

    def test_attention_compute75_ptx_uses_requested_architecture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "cutlass" / "include"
            self._make_cutlass_headers(include_dir)
            environment = {
                "COMFYUI_TURING_UTILS_ARCH_LIST": "7.5+PTX",
                "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR": str(include_dir),
            }
            with (
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                namespace = runpy.run_path(
                    str(SETUP_PATH), run_name="__turing_utils_linux_setup_test__"
                )

        extensions = setup.call_args.kwargs["ext_modules"]
        self.assertEqual(
            [extension.name for extension in extensions],
            ["comfyui_turing_utils_kernel._C", "comfyui_turing_utils_kernel._sage_qattn_sm75", "comfyui_turing_utils_kernel._sage_fused_sm75"],
        )
        flags = extensions[1].kwargs["extra_compile_args"]["nvcc"]
        self.assertEqual(namespace["ARCH_LIST"], "7.5+PTX")
        self.assertEqual(
            extensions[1].kwargs["define_macros"],
            [
                ("COMFYUI_TURING_UTILS_BUILD_SM75", "1"),
                ("COMFYUI_TURING_UTILS_BUILD_SM75_PTX", "1"),
            ],
        )
        self.assertNotIn("-gencode", flags)
        self.assertNotIn("-DCOMFYUI_TURING_UTILS_EXPERIMENTAL_SAGE_VARIANTS", flags)
        self.assertIn("-std=c++17", flags)
        self.assertIn(
            "-std=c++17",
            extensions[0].kwargs["extra_compile_args"]["cxx"],
        )
        self.assertIn("csrc/turing/sage/sol_sparse_cuda_sm75.cu", extensions[1].kwargs["sources"])
        self.assertIn("csrc/turing/sage/quant_v_int8_cuda_sm75.cu", extensions[1].kwargs["sources"])
        self.assertIn(
            "csrc/turing/sage/qk_preprocess.cu", extensions[2].kwargs["sources"]
        )
        self.assertIn(
            "csrc/turing/sage/overlap_blend.cu", extensions[2].kwargs["sources"]
        )
        self.assertEqual(setup.call_args.kwargs["version"], "0.42.0")
        self.assertEqual(set(setup.call_args.kwargs["packages"]), {
            "comfyui_turing_utils_kernel",
            "comfyui_turing_utils_kernel.turing_sage",
        })

    def test_visible_gpu_arches_are_default_and_explicit_multiarch_is_supported(self):
        setup_source = SETUP_PATH.read_text(encoding="utf-8")
        wheel_source = (
            PLUGIN_ROOT / "kernel" / "scripts" / "build_wheel.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "def _visible_cuda_arches()",
            setup_source,
        )
        self.assertIn('{"75", "80", "86", "89", "90"}', setup_source)
        self.assertIn('default=None', wheel_source)
        self.assertIn('defaults to all visible supported GPUs', wheel_source)

        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "cutlass" / "include"
            self._make_cutlass_headers(include_dir)
            environment = {
                "COMFYUI_TURING_UTILS_ARCH_LIST": "8.0;8.6;8.9;90;8.6",
                "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR": str(include_dir),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch(
                    "torch.utils.cpp_extension.CUDAExtension",
                    side_effect=self._extension,
                ),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                namespace = runpy.run_path(
                    str(SETUP_PATH), run_name="__turing_utils_multiarch_setup_test__"
                )

        self.assertEqual(namespace["ARCH_LIST"], "8.0;8.6;8.9;9.0")
        self.assertEqual(
            [extension.name for extension in setup.call_args.kwargs["ext_modules"]],
            [
                "comfyui_turing_utils_kernel._C",
                "comfyui_turing_utils_kernel._sage_qattn_sm75",
                "comfyui_turing_utils_kernel._sage_fused_sm75",
            ],
        )
        self.assertIn(
            "csrc/turing/sage/sol_sparse_cuda_sm75.cu",
            setup.call_args.kwargs["ext_modules"][1].kwargs["sources"],
        )
        self.assertEqual(
            setup.call_args.kwargs["ext_modules"][1].kwargs["define_macros"],
            [
                ("COMFYUI_TURING_UTILS_BUILD_SM80", "1"),
                ("COMFYUI_TURING_UTILS_BUILD_SM86", "1"),
                ("COMFYUI_TURING_UTILS_BUILD_SM89", "1"),
                ("COMFYUI_TURING_UTILS_BUILD_SM90", "1"),
            ],
        )

    def test_default_arch_detects_all_unique_visible_gpu_capabilities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "cutlass" / "include"
            self._make_cutlass_headers(include_dir)
            environment = {
                "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR": str(include_dir),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.device_count", return_value=3),
                mock.patch(
                    "torch.cuda.get_device_capability",
                    side_effect=((8, 6), (7, 5), (8, 6)),
                ),
                mock.patch(
                    "torch.utils.cpp_extension.CUDAExtension",
                    side_effect=self._extension,
                ),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                namespace = runpy.run_path(
                    str(SETUP_PATH), run_name="__turing_utils_autoarch_setup_test__"
                )

        self.assertEqual(namespace["ARCH_LIST"], "7.5;8.6")
        self.assertEqual(len(setup.call_args.kwargs["ext_modules"]), 3)

    def test_default_arch_falls_back_to_turing_without_a_visible_gpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "cutlass" / "include"
            self._make_cutlass_headers(include_dir)
            environment = {
                "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR": str(include_dir),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                namespace = runpy.run_path(
                    str(SETUP_PATH), run_name="__turing_utils_no_gpu_setup_test__"
                )

        self.assertEqual(namespace["ARCH_LIST"], "7.5")
        self.assertEqual(len(setup.call_args.kwargs["ext_modules"]), 3)

    def test_plugin_arch_setting_overrides_stale_torch_arch_setting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "cutlass" / "include"
            self._make_cutlass_headers(include_dir)
            environment = {
                "COMFYUI_TURING_UTILS_ARCH_LIST": "8.6",
                "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR": str(include_dir),
                "TORCH_CUDA_ARCH_LIST": "7.5",
            }
            with (
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup"),
            ):
                namespace = runpy.run_path(
                    str(SETUP_PATH), run_name="__turing_utils_arch_precedence_test__"
                )
                torch_arch_list = os.environ["TORCH_CUDA_ARCH_LIST"]

        self.assertEqual(namespace["ARCH_LIST"], "8.6")
        self.assertEqual(torch_arch_list, "8.6")

    def test_sm90a_arch_suffix_builds_integer_attention(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "cutlass" / "include"
            self._make_cutlass_headers(include_dir)
            environment = {
                "COMFYUI_TURING_UTILS_ARCH_LIST": "9.0a",
                "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR": str(include_dir),
            }
            with (
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                runpy.run_path(
                    str(SETUP_PATH), run_name="__turing_utils_sm90a_setup_test__"
                )

        self.assertEqual(len(setup.call_args.kwargs["ext_modules"]), 3)

    def test_retired_sage2_fp8_helpers_are_not_compiled(self):
        sage_dir = PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage"
        production_source = (sage_dir / "fused.cu").read_text(encoding="utf-8")
        production_header = (sage_dir / "fused.h").read_text(encoding="utf-8")
        for marker in (
            "TransposePadPermuteKernel",
            "MeanScaleKernel",
            "transpose_pad_permute_cuda",
            "scale_fuse_quant_cuda",
            "mean_scale_fuse_quant_cuda",
        ):
            self.assertNotIn(marker, production_source)
            self.assertNotIn(marker, production_header)

    def test_retired_sage_variants_are_absent_from_production_sources(self):
        csrc = PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage"
        production_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in csrc.rglob("*")
            if path.is_file()
        )
        self.assertNotIn(
            "COMFYUI_TURING_UTILS_EXPERIMENTAL_SAGE_VARIANTS",
            production_source,
        )

    def test_setup_and_pyproject_versions_match(self):
        metadata = tomllib.loads(
            (PLUGIN_ROOT / "kernel" / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["project"]["version"], "0.42.0")

    def test_overlap_epilogue_is_self_contained_and_deterministic_by_design(self):
        source = (
            PLUGIN_ROOT
            / "kernel"
            / "csrc"
            / "turing"
            / "sage"
            / "overlap_blend.cu"
        ).read_text(encoding="utf-8")
        self.assertNotIn("ATen/cuda/CUDAContext", source)
        self.assertNotIn("cusparse", source.lower())
        self.assertNotIn("atomic", source.lower())
        self.assertIn("float accumulated", source)
        self.assertIn("getCurrentCUDAStream", source)

    def test_sparse_source_does_not_require_optional_cuda_library_headers(self):
        source = (
            PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage" / "sol_sparse_cuda_sm75.cu"
        ).read_text(encoding="utf-8")
        route_source = (
            PLUGIN_ROOT
            / "kernel"
            / "csrc"
            / "turing"
            / "sage"
            / "sparse"
            / "route_compaction.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn('#include "torch_compat.h"', source)
        self.assertNotIn("ATen/cuda/CUDAContext", source)
        self.assertNotIn("CUDAGuard", source)
        self.assertNotIn("cusparse", source.lower())
        self.assertIn("key_score_summary", source)
        self.assertIn("dequantize_int8_tile", source)
        self.assertIn("compute_fp16_qk<HeadDim, 1>", source)
        self.assertIn("shared_proxy", source)
        self.assertIn("kRouteStorageOffset", source)
        self.assertIn("key_int8", source)
        self.assertIn(
            "AttentionGeometry<128>::kAttentionSharedBytes <= 64 * 1024", source
        )
        self.assertIn("HeadDim / 32", source)
        self.assertNotIn("__global__ void block_summary_kernel(", source)
        self.assertNotIn("query_threshold_kernel", source)
        self.assertNotIn("at::Tensor thresholds", source)
        self.assertNotIn("query.data_ptr", source)
        self.assertNotIn("key.data_ptr", source)
        self.assertNotIn("minimum_route_density", source)
        self.assertNotIn("temporal_neighbor_frames", source)
        self.assertIn("sparse_query_blocks", source)
        self.assertIn("exact_kv_blocks", source)
        self.assertIn("shared_route", source)
        self.assertIn("residual_subblocks", source)
        self.assertIn("UseW8A8", source)
        self.assertIn("ForceDense", source)
        self.assertIn("compute_int8_sv_permuted", source)
        self.assertIn("kCompactionScratchWords", source)
        self.assertIn('#include "sparse/route_compaction.cuh"', source)
        self.assertIn("__shfl_up_sync", route_source)
        self.assertIn("__ballot_sync", source)
        self.assertIn("AttentionStorage<HeadDim, SparseValuePipeline>", source)
        self.assertIn("sparse_v_pipeline=", source)
        self.assertIn("current_cuda_device_major() >= 8", source)
        self.assertIn("load_half_tile_mapped", source)
        self.assertIn("sol_sparse_online_int8_f16_mapped_attn", source)

    def test_mapped_fp16_sol_abi_is_complete(self):
        sage_dir = PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage"
        marker = "sol_sparse_online_int8_f16_mapped_attn"
        for path in (
            sage_dir / "sol_sparse_cuda_sm75.cu",
            sage_dir / "attn_cuda_sm75.h",
            sage_dir / "pybind_sm75.cpp",
        ):
            self.assertIn(marker, path.read_text(encoding="utf-8"))

    def test_attention_launches_use_pytorch_current_cuda_stream(self):
        sage_dir = PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage"
        fixed = (sage_dir / "qk_int_sv_f16_cuda_sm75.cu").read_text(encoding="utf-8")
        varlen = (sage_dir / "qk_int_sv_f16_varlen_cuda_sm75.cu").read_text(
            encoding="utf-8"
        )
        fused = (sage_dir / "fused.cu").read_text(encoding="utf-8")

        current_stream = "c10::cuda::getCurrentCUDAStream()"
        self.assertIn(current_stream, fixed)
        self.assertIn(current_stream, varlen)
        # Retired Sage1/Sage2 launch sites are not part of the production
        # source count; every remaining fused family still names the current
        # PyTorch stream explicitly.
        self.assertEqual(fused.count(current_stream), 8)
        self.assertNotIn("<<<grid, block>>>", fused)

    def test_fused_qk_abi_exposes_independent_rope_schema(self):
        sage_dir = PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage"
        source = (sage_dir / "qk_preprocess.cu").read_text(encoding="utf-8")
        binding = (sage_dir / "pybind_fused.cpp").read_text(encoding="utf-8")
        self.assertIn("at::Tensor query_freqs,", source)
        self.assertIn("at::Tensor key_freqs,", source)
        self.assertIn("at::Tensor key_source_indices,", source)
        self.assertIn("quant_qk_rms_rope_int8_mapped_cuda", binding)
        self.assertIn("gather_value_int8_mapped_cuda", binding)
        self.assertIn('m.attr("qk_preprocess_protocol_schema") = 3', binding)

    def test_sol_summary_abi_supports_mapped_physical_values(self):
        sage_dir = PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "sage"
        source = (sage_dir / "sol_sparse_cuda_sm75.cu").read_text(
            encoding="utf-8"
        )
        header = (sage_dir / "attn_cuda_sm75.h").read_text(encoding="utf-8")
        binding = (sage_dir / "pybind_sm75.cpp").read_text(encoding="utf-8")
        marker = "sol_w8a8_precompute_mapped_summaries"
        self.assertIn("bool MappedValue", source)
        self.assertIn(marker, source)
        self.assertIn(marker, header)
        self.assertIn(marker, binding)

    def test_stable_sage_locks_single_k_warp_and_benchmarks_core(self):
        fixed = (
            PLUGIN_ROOT
            / "kernel"
            / "csrc"
            / "turing"
            / "sage"
            / "qk_int_sv_f16_cuda_sm75.cu"
        ).read_text(encoding="utf-8")
        benchmark = (
            PLUGIN_ROOT / "kernel" / "scripts" / "benchmark_backends.py"
        ).read_text(encoding="utf-8")
        self.assertIn("num_warps_k == 1", fixed)
        self.assertIn("cross-warp online-softmax reduction", fixed)
        self.assertIn("prequantize_sageattn", benchmark)
        self.assertIn('"bundled Sage",\n                "prequantized"', benchmark)

    def test_bf16_convrot_uses_device_optin_shared_memory_and_tunable_geometry(self):
        bindings = (PLUGIN_ROOT / "kernel" / "csrc" / "bindings.cpp").read_text(
            encoding="utf-8"
        )
        source = (
            PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "convrot_quant.cu"
        ).read_text(encoding="utf-8")
        benchmark = (
            PLUGIN_ROOT / "kernel" / "scripts" / "benchmark_backends.py"
        ).read_text(encoding="utf-8")
        self.assertIn("sharedMemPerBlockOptin", bindings)
        self.assertIn("{512, 768, 1024}", bindings)
        self.assertIn("sharedMemPerMultiprocessor", bindings)
        self.assertIn("maxThreadsPerMultiProcessor", bindings)
        self.assertIn("forced_threads", bindings)
        self.assertIn("cudaFuncAttributeMaxDynamicSharedMemorySize", source)
        self.assertIn('(\"fc2\", 5376, 14336, \"swiglu\")', benchmark)
        self.assertIn(
            '(\"H3 fc2 fused SwiGLU+ConvRot A8\", 28672, \"swiglu\")',
            benchmark,
        )

    def test_linear_tiles_use_deterministic_architecture_dispatch(self):
        bindings = (PLUGIN_ROOT / "kernel" / "csrc" / "bindings.cpp").read_text(
            encoding="utf-8"
        )
        source = (
            PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "w4a8.cu"
        ).read_text(encoding="utf-8")
        self.assertNotIn("tile_policy", bindings)
        self.assertNotIn("run_auto_tuned", source)
        self.assertNotIn("GEMM_TUNE", source)
        self.assertNotIn("GEMM_CACHE", source)
        self.assertNotIn("PersistentTileTuneCache", source)
        self.assertIn("properties->major >= 8 && n >= 16384", source)
        self.assertIn(
            "run_ampere_int8_tile<128, 256, 64, 64, 64, 64, 3>",
            source,
        )
        self.assertIn("run_int8_tile<128, 256, 64, 64>", source)

    def test_legacy_w4a8_edges_remain_on_tensor_cores(self):
        bindings = (PLUGIN_ROOT / "kernel" / "csrc" / "bindings.cpp").read_text(
            encoding="utf-8"
        )
        source = (
            PLUGIN_ROOT / "kernel" / "csrc" / "turing" / "w4a8.cu"
        ).read_text(encoding="utf-8")
        self.assertNotIn("__dp4a", source)
        self.assertNotIn("w4a8_compatibility_kernel", source)
        self.assertIn("run_k_tail_tile", source)
        self.assertIn("InputAlignment", source)
        self.assertIn("tail_weight", bindings)
        self.assertIn("tensor_core_channels", bindings)

    def test_windows_conda_target_specific_cccl_path_is_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            conda_include = prefix / "Library" / "include"
            self._make_cutlass_headers(conda_include)

            extensions = self._run_windows_setup(prefix)

        self.assertEqual(
            [extension.name for extension in extensions],
            [
                "comfyui_turing_utils_kernel._C",
                "comfyui_turing_utils_kernel._sage_qattn_sm75",
                "comfyui_turing_utils_kernel._sage_fused_sm75",
            ],
        )
        self.assertEqual(extensions[0].kwargs["include_dirs"][1], str(conda_include.resolve()))
        self.assertIn(str(cccl.resolve()), extensions[0].kwargs["include_dirs"])
        self.assertIn("/std:c++20", extensions[0].kwargs["extra_compile_args"]["cxx"])
        self.assertIn("-std=c++20", extensions[0].kwargs["extra_compile_args"]["nvcc"])

    def test_windows_sm75_build_includes_fused_qk_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            self._make_cutlass_headers(prefix / "Library" / "include")

            extensions = self._run_windows_setup(prefix, arch_list="7.5")

        self.assertEqual(
            [extension.name for extension in extensions],
            [
                "comfyui_turing_utils_kernel._C",
                "comfyui_turing_utils_kernel._sage_qattn_sm75",
                "comfyui_turing_utils_kernel._sage_fused_sm75",
            ],
        )
        fused = extensions[2]
        self.assertIn("csrc/turing/sage/qk_preprocess.cu", fused.kwargs["sources"])
        self.assertIn(str(cccl.resolve()), fused.kwargs["include_dirs"])
        self.assertIn("-std=c++20", fused.kwargs["extra_compile_args"]["nvcc"])

    def test_cuda_11_selects_cxx17_unless_explicitly_overridden(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            self._make_cutlass_headers(prefix / "Library" / "include")

            extensions = self._run_windows_setup(
                prefix,
                cuda_version="11.8",
            )

        flags = extensions[0].kwargs["extra_compile_args"]
        self.assertIn("/std:c++17", flags["cxx"])
        self.assertIn("-std=c++17", flags["nvcc"])

    def test_conda_toolkit_metadata_precedes_torch_cuda_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            (prefix / "version.txt").write_text("CUDA Version 11.8.0", encoding="utf-8")
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            self._make_cutlass_headers(prefix / "Library" / "include")

            extensions = self._run_windows_setup(prefix, cuda_version="12.8")

        flags = extensions[0].kwargs["extra_compile_args"]
        self.assertIn("/std:c++17", flags["cxx"])
        self.assertIn("-std=c++17", flags["nvcc"])

    def test_cxx_standard_overrides_are_normalized_and_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            self._make_cutlass_headers(prefix / "Library" / "include")
            extensions = self._run_windows_setup(
                prefix,
                extra_environment={
                    "COMFYUI_TURING_UTILS_HOST_CXX_STANDARD": "/std:c++20",
                    "COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD": "20",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "must select c\\+\\+17 or c\\+\\+20"):
                self._run_windows_setup(
                    prefix,
                    extra_environment={
                        "COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD": "c++23",
                    },
                )

        flags = extensions[0].kwargs["extra_compile_args"]
        self.assertEqual(flags["cxx"].count("/std:c++20"), 1)
        self.assertEqual(flags["nvcc"].count("-std=c++20"), 1)

    def test_wheel_builder_finds_windows_conda_nvcc(self):
        script = PLUGIN_ROOT / "kernel" / "scripts" / "build_wheel.py"
        namespace = runpy.run_path(str(script), run_name="__turing_utils_build_wheel_test__")
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            nvcc = prefix / "Library" / "bin" / "nvcc.exe"
            nvcc.parent.mkdir(parents=True)
            nvcc.touch()
            environment = {"CONDA_PREFIX": str(prefix)}
            with mock.patch.object(platform, "system", return_value="Windows"):
                selected = namespace["configure_cuda_home"](environment)

        self.assertEqual(selected, nvcc.resolve())
        self.assertEqual(environment["CUDA_HOME"], str((prefix / "Library").resolve()))

    def test_nvidia_cutlass_python_package_is_auto_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            site_root = Path(temp_dir)
            cutlass_include = site_root / "cutlass_library" / "source" / "include"
            self._make_cutlass_headers(cutlass_include)
            environment = {"COMFYUI_TURING_UTILS_ARCH_LIST": "8.6"}
            with (
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(sys, "path", [str(site_root), *sys.path]),
                mock.patch("torch.utils.cpp_extension.CUDAExtension", side_effect=self._extension),
                mock.patch("torch.utils.cpp_extension.BuildExtension", object()),
                mock.patch("setuptools.setup") as setup,
            ):
                runpy.run_path(str(SETUP_PATH), run_name="__turing_utils_cutlass_package_test__")

        core = setup.call_args.kwargs["ext_modules"][0]
        self.assertEqual(core.kwargs["include_dirs"][1], str(cutlass_include.resolve()))

    def test_windows_uses_cached_portable_cutlass_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prefix = temp_root / "conda"
            cccl = prefix / "Library" / "include" / "targets" / "x64"
            (cccl / "nv").mkdir(parents=True)
            (cccl / "nv" / "target").touch()
            cutlass_include = (
                temp_root / "comfyui-turing-utils-cutlass-4.2.0.0" / "include"
            )
            self._make_cutlass_headers(cutlass_include)

            with (
                mock.patch.object(sys, "path", [str(temp_root / "empty-site")]),
                mock.patch("tempfile.gettempdir", return_value=temp_dir),
            ):
                extensions = self._run_windows_setup(prefix)

        core = extensions[0]
        self.assertEqual(core.kwargs["include_dirs"][1], str(cutlass_include.resolve()))
        self.assertIn(str(cccl.resolve()), core.kwargs["include_dirs"])

    def test_windows_missing_cccl_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            # Isolate the CCCL diagnostic: CUTLASS is present, but nv/target
            # deliberately is not. The test must never depend on network
            # access to the portable-header fallback.
            self._make_cutlass_headers(prefix / "Library" / "include")
            with self.assertRaisesRegex(RuntimeError, "cuda-cccl=12.8.90"):
                self._run_windows_setup(prefix)


if __name__ == "__main__":
    unittest.main()
