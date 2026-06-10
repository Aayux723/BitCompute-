import os
import subprocess
from pathlib import Path

from setuptools import setup


CUDA_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
VS2022_HOME = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
MSVC_VERSION = "14.44.35207"
MSVC_BIN = VS2022_HOME / "VC" / "Tools" / "MSVC" / MSVC_VERSION / "bin" / "Hostx64" / "x64"


def _prepend_path(*paths):
    entries = [str(path) for path in paths if path and Path(path).exists()]
    os.environ["PATH"] = os.pathsep.join(entries + [os.environ.get("PATH", "")])


def _load_vs2022_x64_environment():
    vcvarsall = VS2022_HOME / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.exists():
        raise RuntimeError(f"VS2022 vcvarsall.bat was not found: {vcvarsall}")

    command = f'"{vcvarsall}" x64 >nul && set'
    output = subprocess.check_output(command, shell=True, text=True)

    for line in output.splitlines():
        key, _, value = line.partition("=")
        if key and value:
            os.environ[key] = value


def _configure_toolchain():
    if not CUDA_HOME.exists():
        raise RuntimeError(f"CUDA 12.8 was not found: {CUDA_HOME}")
    if not (MSVC_BIN / "cl.exe").exists():
        raise RuntimeError(f"VS2022 x64 cl.exe was not found: {MSVC_BIN / 'cl.exe'}")

    os.environ["CUDA_HOME"] = str(CUDA_HOME)
    os.environ["CUDA_PATH"] = str(CUDA_HOME)
    os.environ["CUDACXX"] = str(CUDA_HOME / "bin" / "nvcc.exe")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["VSCMD_ARG_HOST_ARCH"] = "x64"
    os.environ["VSCMD_ARG_TGT_ARCH"] = "x64"

    # Make the compiler environment deterministic before torch.cpp_extension
    # snapshots CUDA_HOME and before setuptools probes Visual Studio.
    _load_vs2022_x64_environment()
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    _prepend_path(CUDA_HOME / "bin", MSVC_BIN)


_configure_toolchain()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bitnet_engine",
    ext_modules=[
        CUDAExtension(
            name="BitNet_engine",
            sources=["Binding.cpp", "kernel.cu"],
            extra_compile_args={
                "cxx": ["/O2", "/GL", "/std:c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-ccbin",
                    str(MSVC_BIN),
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
