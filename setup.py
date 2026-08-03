from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CUTLASS = ROOT / "third_party" / "cutlass" / "include"

setup(
    name="low_latency_mxfp4",
    ext_modules=[
        CUDAExtension(
            name="low_latency_mxfp4",
            sources=[
                str(ROOT / "torch" / "low_latency_mxfp4_torch.cu"),
                str(ROOT / "low_latency_grouped_gemm" / "src" / "low_latency_grouped_gemm.cu"),
                str(ROOT / "low_latency_grouped_gemm" / "src" / "low_latency_mxfp4_fp8.cu"),
            ],
            include_dirs=[str(ROOT), str(CUTLASS)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "--threads=4"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
