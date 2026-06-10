import os
import subprocess
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths

# ---------------------------------------------------------------------------
# CUDA 12.8's cudafe++ crashes when CUDAExtension injects PyTorch include
# paths + -std=c++17 into the nvcc command on MSVC 14.44. We work around it:
#   1. Compile kernel.cu ourselves with a clean nvcc invocation via vcvars64
#   2. Use CppExtension (not CUDAExtension) for Binding.cpp + link kernel.obj
# ---------------------------------------------------------------------------

CUDA_HOME = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', '')).strip()
if not CUDA_HOME and os.name == 'nt':
    # Fallback to checking default NVIDIA paths on Windows
    default_cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1',
    ]
    for p in default_cuda_paths:
        if os.path.exists(p):
            CUDA_HOME = p
            break

if not CUDA_HOME:
    # On Linux or unknown, nvcc might just be in PATH
    CUDA_INC = ''
    CUDA_LIB = ''
    NVCC = 'nvcc'
else:
    CUDA_INC = os.path.join(CUDA_HOME, 'include')
    # On Linux, lib64 is common. On Windows, lib/x64
    CUDA_LIB = os.path.join(CUDA_HOME, 'lib64') if os.name != 'nt' else os.path.join(CUDA_HOME, 'lib', 'x64')
    NVCC = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if os.name == 'nt' else 'nvcc')

KERNEL_SRC = os.path.join('BitNet_Engine', 'kernel.cu')
# Use .o on Linux, .obj on Windows
KERNEL_OBJ = os.path.join('BitNet_Engine', 'kernel.obj' if os.name == 'nt' else 'kernel.o')

def _find_vcvars():
    """Find vcvars64.bat dynamically on Windows."""
    if os.name != 'nt':
        return None
    base_path = r'C:\Program Files\Microsoft Visual Studio\2022'
    base_path_x86 = r'C:\Program Files (x86)\Microsoft Visual Studio\2022'
    editions = ['Enterprise', 'Professional', 'Community', 'BuildTools']
    
    for base in (base_path_x86, base_path):
        for edition in editions:
            vcvars = os.path.join(base, edition, r'VC\Auxiliary\Build\vcvars64.bat')
            if os.path.exists(vcvars):
                return vcvars
    return None

def _compile_cuda_kernel(arch):
    """Compile kernel.cu natively or via MSVC env on Windows."""
    nvcc_args = (
        f'"{NVCC}" '
        f'-c "{KERNEL_SRC}" '
        f'-o "{KERNEL_OBJ}" '
        f'-O3 --use_fast_math '
        f'-gencode=arch=compute_{arch},code=sm_{arch} '
    )
    
    if os.name == 'nt':
        nvcc_args += f'-Xcompiler /MD -Xcompiler /O2 '
        if CUDA_INC:
            nvcc_args += f'-I"{CUDA_INC}"'
        vcvars = _find_vcvars()
        if vcvars:
            full_cmd = f'"{vcvars}" x64 >nul 2>&1 && {nvcc_args}'
        else:
            full_cmd = nvcc_args
    else:
        nvcc_args += f'-Xcompiler -fPIC -Xcompiler -O3 '
        if CUDA_INC:
            nvcc_args += f'-I"{CUDA_INC}"'
        full_cmd = nvcc_args
        
    print(f"Running CUDA compilation: {full_cmd}")
    print(f'\n>>> Pre-compiling CUDA kernel (compute_{arch}) ...')
    print(f'>>> {nvcc_args}\n')
    subprocess.check_call(full_cmd, shell=True)
    print('>>> kernel.obj ready.\n')


class BuildWithCUDA(BuildExtension):
    """Custom build step: compile kernel.cu before building the C++ extension."""

    def build_extensions(self):
        needs_build = (
            not os.path.exists(KERNEL_OBJ)
            or os.path.getmtime(KERNEL_SRC) > os.path.getmtime(KERNEL_OBJ)
        )
        if needs_build:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                arch = f'{major}{minor}'
            else:
                arch = '89'
            _compile_cuda_kernel(arch)

        # The user's PowerShell is often an x86 Dev Shell, which breaks the 64-bit Python link.
        # We must clear the VC environment variables so setuptools auto-detects the x64 compiler natively!
        os.environ.pop('DISTUTILS_USE_SDK', None)
        os.environ.pop('MSSdk', None)
        os.environ.pop('VSCMD_ARG_TGT_ARCH', None)
        os.environ.pop('VSCMD_ARG_HOST_ARCH', None)
        os.environ.pop('INCLUDE', None)
        os.environ.pop('LIB', None)
        os.environ.pop('LIBPATH', None)
        
        super().build_extensions()


# Get all include paths for CUDA from PyTorch's helper
cuda_include_dirs = include_paths(device_type='cuda')
# Also add our explicit CUDA include dir
if CUDA_INC not in cuda_include_dirs:
    cuda_include_dirs.append(CUDA_INC)

setup(
    name='BitCore',
    version='0.1.0',
    description='1.58-bit LLM Execution Engine and Benchmark Suite',
    author='Aayush Anand',
    packages=find_packages(),
    py_modules=['BitNet_bench'],

    ext_modules=[
        CppExtension(
            name='BitNet_engine',
            sources=['BitNet_Engine/Binding.cpp'],
            include_dirs=cuda_include_dirs,
            extra_objects=[KERNEL_OBJ],
            library_dirs=[CUDA_LIB],
            libraries=['cudart', 'c10_cuda', 'torch_cuda'],
            extra_compile_args=[
                '-O2', '-GL',
                '/Zc:alignedNew-',
                '/Zc:__cplusplus',
                '-D_DISABLE_EXTENDED_ALIGNED_STORAGE',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildWithCUDA
    },

    entry_points={
        'console_scripts': [
            'bitcore-bench=BitNet_bench:main',
        ],
    },
)
