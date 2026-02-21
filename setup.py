"""
Cython build script for find_mfs extensions.

Extensions are built automatically via `pip install -e .` thanks to
the [build-system] in pyproject.toml including Cython.
"""
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

compile_args = ["-O3", "-ffast-math"]

extensions = [
    Extension(
        "find_mfs.core._algorithms",
        sources=["find_mfs/core/_algorithms.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "find_mfs.core._light_formula",
        sources=["find_mfs/core/_light_formula.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "find_mfs.core._pipeline",
        sources=["find_mfs/core/_pipeline.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "find_mfs.isotopes._isospec",
        sources=["find_mfs/isotopes/_isospec.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args + ["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": 3,
        },
    ),
    packages=find_packages(),
    include_dirs=[np.get_include()],
)
