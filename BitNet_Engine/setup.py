from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="bitnet_engine",
    ext_modules=[
        CppExtension(
            name="BitNet_engine",
            sources=["Binding.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)