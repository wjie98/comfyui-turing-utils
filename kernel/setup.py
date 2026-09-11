import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    # Keep MSVC diagnostics in English so PyTorch's compiler probe can decode
    # cl.exe output reliably on non-English Windows installations.
    os.environ.setdefault("VSLANG", "1033")

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def _parse_cuda_version(value) -> tuple[int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)", str(value or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _cuda_toolkit_version() -> tuple[int, int] | None:
    """Resolve the toolkit used by NVCC, not merely the runtime driver."""
    nvcc_candidates: list[Path] = []
    for root in (CUDA_HOME, os.environ.get("CUDA_PATH")):
        if root:
            nvcc_candidates.append(
                Path(root) / "bin" / ("nvcc.exe" if IS_WINDOWS else "nvcc")
            )
    discovered = shutil.which("nvcc")
    if discovered:
        nvcc_candidates.append(Path(discovered))
    seen: set[str] = set()
    for candidate in nvcc_candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout + result.stderr)
        if match:
            return int(match.group(1)), int(match.group(2))

    # Conda CUDA packages may expose version metadata even when nvcc is not on
    # PATH during PEP-517 isolation.
    for root in (CUDA_HOME, os.environ.get("CUDA_PATH"), os.environ.get("CONDA_PREFIX")):
        if not root:
            continue
        root_path = Path(root)
        for metadata in (root_path / "version.json", root_path / "version.txt"):
            if metadata.is_file():
                try:
                    version = _parse_cuda_version(metadata.read_text(encoding="utf-8"))
                except OSError:
                    version = None
                if version is not None:
                    return version

    try:
        import torch

        return _parse_cuda_version(torch.version.cuda)
    except (AttributeError, ImportError):
        return None


def _default_cxx_standard(cuda_version: tuple[int, int] | None) -> str:
    # CUTLASS's EVT-based W4A8 epilogue needs C++20 specifically on the
    # NVCC/MSVC path. Linux remains on PyTorch's portable C++17 baseline so a
    # CUDA 12 toolkit does not accidentally require GCC 10+ merely because
    # NVCC itself understands C++20. Older Windows toolkits also stay on C++17.
    if not IS_WINDOWS:
        return "c++17"
    return "c++20" if cuda_version is None or cuda_version >= (12, 0) else "c++17"


def _normalize_cxx_standard(value: str, variable: str) -> str:
    """Accept concise overrides while rejecting malformed compiler flags."""
    normalized = str(value).strip().lower()
    for prefix in ("-std=", "/std:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    aliases = {
        "17": "c++17",
        "c++17": "c++17",
        "20": "c++20",
        "c++20": "c++20",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise RuntimeError(
            f"{variable} must select c++17 or c++20, got {value!r}"
        ) from error


def _is_cutlass_include_dir(candidate: Path) -> bool:
    required = (
        "cutlass/cutlass.h",
        "cutlass/gemm/device/gemm_universal_adapter.h",
        "cutlass/epilogue/threadblock/fusion/visitors.hpp",
        "cute/tensor.hpp",
    )
    return all((candidate / relative).is_file() for relative in required)


def _download_cutlass_include_dir() -> Path | None:
    """Fetch the pinned, platform-independent NVIDIA wheel without its Python deps."""
    enabled = os.environ.get("COMFYUI_TURING_UTILS_CUTLASS_AUTO_DOWNLOAD", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    version = "4.2.0.0"
    filename = f"nvidia_cutlass-{version}-py3-none-any.whl"
    url = (
        "https://files.pythonhosted.org/packages/21/6b/"
        "e86fc2fd536dd4b8bd2209aa31d7002610c501b0802337787eac9c9b328c/"
        + filename
    )
    expected_sha256 = "f9599ada45c7bcb6bf53f7d3f0a7d154183e53451c01bfa59eecd37dd4a693f6"
    cache_root = Path(tempfile.gettempdir()) / f"comfyui-turing-utils-cutlass-{version}"
    include_dir = cache_root / "include"
    if _is_cutlass_include_dir(include_dir):
        return include_dir.resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    wheel = cache_root / filename
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_sha256:
        partial = cache_root / f"{filename}.{os.getpid()}.part"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            digest = hashlib.sha256(partial.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise RuntimeError(
                    f"downloaded NVIDIA CUTLASS wheel has SHA256 {digest}, expected {expected_sha256}"
                )
            partial.replace(wheel)
        finally:
            if partial.exists():
                partial.unlink()

    staging = cache_root / f"include.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    prefix = "cutlass_library/source/include/"
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = Path(member.filename[len(prefix):])
            if not relative.parts or ".." in relative.parts:
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if not _is_cutlass_include_dir(staging):
        raise RuntimeError("the pinned NVIDIA CUTLASS wheel did not contain the required C++ headers")
    if include_dir.exists():
        shutil.rmtree(include_dir)
    staging.replace(include_dir)
    return include_dir.resolve()


def _cutlass_include_dir() -> Path:
    """Find portable CUTLASS C++ headers on Linux and Windows."""
    override = os.environ.get("COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR")
    if override:
        root = Path(override)
        for candidate in (root, root / "include"):
            if _is_cutlass_include_dir(candidate):
                return candidate.resolve()
        raise RuntimeError(
            "COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR must name CUTLASS's include directory "
            "or its parent (the resolved directory must contain cutlass/cutlass.h)."
        )

    candidates: list[Path] = []
    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.insert(0, Path(conda_prefix))
    for prefix in prefixes:
        candidates.extend(
            [
                prefix / "Library" / "include",  # Conda on Windows
                prefix / "include",              # Conda on Linux
            ]
        )

    cuda_roots: list[Path] = []
    for value in (CUDA_HOME, os.environ.get("CUDA_PATH")):
        if value:
            cuda_roots.append(Path(value))
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_roots.append(Path(nvcc).resolve().parent.parent)
    for cuda_root in cuda_roots:
        candidates.extend(
            [
                cuda_root / "include",
                cuda_root / "targets" / "x86_64-linux" / "include",
                cuda_root / "targets" / "x64" / "include",
            ]
        )

    # NVIDIA's platform-independent nvidia-cutlass wheel installs the C++
    # sources below cutlass_library/source. Keep additional conventional paths
    # for future NVIDIA packages without importing their Python runtime.
    for entry in sys.path:
        if not entry:
            continue
        site_root = Path(entry)
        candidates.extend(
            [
                site_root / "cutlass_library" / "source" / "include",
                site_root / "nvidia" / "cutlass" / "include",
                site_root / "nvidia_cutlass" / "include",
                site_root / "cutlass" / "include",
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if _is_cutlass_include_dir(candidate):
            return candidate.resolve()

    try:
        downloaded = _download_cutlass_include_dir()
    except Exception as error:
        raise RuntimeError(
            "CUTLASS C++ headers were not found locally and the pinned NVIDIA "
            f"header download failed: {error}"
        ) from error
    if downloaded is not None:
        return downloaded

    raise RuntimeError(
        "CUTLASS C++ headers are missing or undiscoverable. Install the official "
        "portable package with `python -m pip install nvidia-cutlass==4.2.0.0`, "
        "install a Conda CUTLASS package, or set COMFYUI_TURING_UTILS_CUTLASS_INCLUDE_DIR "
        "to the directory containing "
        "cutlass/cutlass.h. Set COMFYUI_TURING_UTILS_CUTLASS_AUTO_DOWNLOAD=1 to allow the "
        "default pinned-header download."
    )


def _windows_cccl_include_dirs() -> list[str]:
    """Locate the target-specific CCCL headers omitted from some Conda nvcc paths."""
    if not IS_WINDOWS:
        return []

    candidates: list[Path] = []
    override = os.environ.get("COMFYUI_TURING_UTILS_CCCL_INCLUDE_DIR")
    if override:
        candidates.append(Path(override))

    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.insert(0, Path(conda_prefix))
    for prefix in prefixes:
        candidates.extend(
            [
                prefix / "Library" / "include" / "targets" / "x64",
                prefix / "Library" / "include" / "cccl",
                prefix / "Library" / "include",
                prefix / "include" / "cccl",
                prefix / "include",
            ]
        )

    cuda_roots: list[Path] = []
    for value in (CUDA_HOME, os.environ.get("CUDA_PATH")):
        if value:
            cuda_roots.append(Path(value))
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_roots.append(Path(nvcc).resolve().parent.parent)
    for cuda_root in cuda_roots:
        candidates.extend(
            [
                cuda_root / "include" / "targets" / "x64",
                cuda_root / "include" / "cccl",
                cuda_root / "include",
                cuda_root / "targets" / "x64" / "include",
            ]
        )

    for entry in sys.path:
        if entry:
            candidates.append(Path(entry) / "nvidia" / "cuda_cccl" / "include")

    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "nv" / "target").is_file():
            found.append(str(candidate.resolve()))
    if found:
        return found

    raise RuntimeError(
        "CUDA CCCL headers are missing or undiscoverable: nv/target was not found. "
        "For a CUDA 12.8 Conda environment on Windows, install them with "
        "`conda install -c nvidia cuda-cccl=12.8.90`, or set "
        "COMFYUI_TURING_UTILS_CCCL_INCLUDE_DIR to the directory containing nv/target."
    )


def _visible_cuda_arches() -> tuple[str, ...]:
    try:
        if not torch.cuda.is_available():
            return ()
        capabilities = {
            tuple(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        }
    except (OSError, RuntimeError):
        return ()
    return tuple(
        f"{major}.{minor}"
        for major, minor in sorted(capabilities)
        if (major, minor) >= (7, 5)
    )


def _arch_list() -> str:
    explicit = os.environ.get("COMFYUI_TURING_UTILS_ARCH_LIST")
    if explicit is None:
        explicit = os.environ.get("TORCH_CUDA_ARCH_LIST")
    value = explicit or ";".join(_visible_cuda_arches()) or "7.5"
    arches = []
    for raw in value.replace(",", ";").split(";"):
        arch = raw.strip()
        if not arch:
            continue
        if arch in {"75", "80", "86", "89", "90"}:
            arch = f"{arch[0]}.{arch[1]}"
        if arch not in arches:
            arches.append(arch)
    return ";".join(arches)


ARCH_LIST = _arch_list()
# The plugin-specific setting is authoritative.  Mirroring the normalized
# value also prevents a stale global Torch setting from silently producing a
# wheel for a different architecture than setup.py advertises.
os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH_LIST
print(f"comfyui-turing-utils-kernel: CUDA architectures {ARCH_LIST}")


def _architecture_macros() -> list[tuple[str, str]]:
    """Embed the exact wheel architecture set in every native extension."""
    macros: list[tuple[str, str]] = []
    for raw in ARCH_LIST.split(";"):
        arch = raw.strip()
        base = arch.split("+")[0]
        match = re.match(r"^(\d+)\.(\d+)(a?)$", base)
        if not match:
            continue
        suffix = "A" if match.group(3) else ""
        name = (
            "COMFYUI_TURING_UTILS_BUILD_SM"
            f"{match.group(1)}{match.group(2)}{suffix}"
        )
        macros.append((name, "1"))
        if "+PTX" in arch.upper():
            macros.append((f"{name}_PTX", "1"))
    return macros


ARCHITECTURE_MACROS = _architecture_macros()
CUTLASS_INCLUDE_DIR = _cutlass_include_dir()
CCCL_INCLUDE_DIRS = _windows_cccl_include_dirs()

COMMON_DEFINES = [
    "-DENABLE_BF16=1",
]

CUDA_TOOLKIT_VERSION = _cuda_toolkit_version()
DEFAULT_CXX_STANDARD = _default_cxx_standard(CUDA_TOOLKIT_VERSION)
HOST_CXX_STANDARD = _normalize_cxx_standard(
    os.environ.get(
        "COMFYUI_TURING_UTILS_HOST_CXX_STANDARD",
        os.environ.get("COMFYUI_TURING_UTILS_CXX_STANDARD", DEFAULT_CXX_STANDARD),
    ),
    "COMFYUI_TURING_UTILS_HOST_CXX_STANDARD",
)
NVCC_CXX_STANDARD = _normalize_cxx_standard(
    os.environ.get("COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD", DEFAULT_CXX_STANDARD),
    "COMFYUI_TURING_UTILS_NVCC_CXX_STANDARD",
)

NVCC_FLAGS = [
    *COMMON_DEFINES,
    f"-std={NVCC_CXX_STANDARD}",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=--allow-expensive-optimizations=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_HALF2_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
]

if IS_WINDOWS:
    NVCC_FLAGS.extend(
        [
            "--use-local-env",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/O2",
            "-Xcompiler",
            "/EHsc",
            "-Xcompiler",
            "/bigobj",
            "-Xcompiler",
            "/FS",
            "-Xcompiler",
            "/DNOMINMAX",
            "-Xcompiler",
            "/DWIN32_LEAN_AND_MEAN",
        ]
    )

CUDAHOSTCXX = os.environ.get("COMFYUI_TURING_UTILS_CUDAHOSTCXX") or os.environ.get("CUDAHOSTCXX")
if CUDAHOSTCXX:
    NVCC_FLAGS.extend(["-ccbin", CUDAHOSTCXX])

if IS_WINDOWS:
    CXX_FLAGS = [
        "/DENABLE_BF16=1",
        f"/std:{HOST_CXX_STANDARD}",
        "/O2",
        "/EHsc",
        "/MD",
        "/bigobj",
        "/FS",
        "/DNOMINMAX",
        "/DWIN32_LEAN_AND_MEAN",
        "/wd4251",
        "/wd4275",
        "/wd4819",
    ]
else:
    CXX_FLAGS = [
        *COMMON_DEFINES,
        f"-std={HOST_CXX_STANDARD}",
        "-O3",
        "-fvisibility=hidden",
    ]


core_ext = CUDAExtension(
    name="comfyui_turing_utils_kernel._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/turing/bf16_epilogue.cu",
        "csrc/turing/convrot_quant.cu",
        "csrc/turing/segmented_rms_adaln.cu",
        "csrc/turing/w4a8.cu",
    ],
    define_macros=ARCHITECTURE_MACROS,
    include_dirs=[
        str((ROOT / "csrc").resolve()),
        str(CUTLASS_INCLUDE_DIR.resolve()),
        *CCCL_INCLUDE_DIRS,
    ],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)


def _includes_integer_attention_arch() -> bool:
    for arch in ARCH_LIST.split(";"):
        base = arch.split("+")[0]
        match = re.match(r"^(\d+)\.(\d+)", base)
        if match and (int(match.group(1)), int(match.group(2))) >= (7, 5):
            return True
    return False


ext_modules = [core_ext]
if _includes_integer_attention_arch():
    attention_nvcc_flags = [
        *NVCC_FLAGS,
        "--use_fast_math",
    ]
    sage_include_dirs = [
        str((ROOT / "csrc" / "turing").resolve()),
        str((ROOT / "csrc" / "turing" / "sage").resolve()),
        *CCCL_INCLUDE_DIRS,
    ]
    ext_modules.extend(
        [
            CUDAExtension(
                name="comfyui_turing_utils_kernel._sage_qattn_sm75",
                sources=[
                    "csrc/turing/sage/pybind_sm75.cpp",
                    "csrc/turing/sage/qk_int_sv_f16_cuda_sm75.cu",
                    "csrc/turing/sage/qk_int_sv_f16_varlen_cuda_sm75.cu",
                    "csrc/turing/sage/sol_sparse_cuda_sm75.cu",
                    "csrc/turing/sage/quant_v_int8_cuda_sm75.cu",
                ],
                define_macros=ARCHITECTURE_MACROS,
                include_dirs=sage_include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": attention_nvcc_flags},
            ),
            CUDAExtension(
                name="comfyui_turing_utils_kernel._sage_fused_sm75",
                sources=[
                    "csrc/turing/sage/pybind_fused.cpp",
                    "csrc/turing/sage/fused.cu",
                    "csrc/turing/sage/qk_preprocess.cu",
                    "csrc/turing/sage/overlap_blend.cu",
                ],
                define_macros=ARCHITECTURE_MACROS,
                include_dirs=sage_include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": attention_nvcc_flags},
            ),
        ]
    )


setup(
    name="comfyui-turing-utils-kernel",
    version="0.42.0",
    packages=find_packages(where=str(ROOT)),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
