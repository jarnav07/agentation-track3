"""Builds the compiled engine extension `fastsim._engine` (used by the Dockerfile)."""
from setuptools import Extension, setup

setup(
    name="fastsim",
    version="1.0.0",
    packages=["fastsim"],
    ext_modules=[
        Extension(
            "fastsim._engine",
            sources=["csrc/engine.cpp", "csrc/module.cpp"],
            language="c++",
            # No fast-math and no FMA contraction: the floating-point sequence must match NumPy's.
            extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off", "-fno-fast-math"],
            extra_link_args=["-static-libstdc++", "-static-libgcc"],
        )
    ],
)
