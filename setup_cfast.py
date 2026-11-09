"""Build script for Cython board acceleration module.

Usage:
    python setup_cfast.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "alphigo.mcts._cboard",
        sources=["alphigo/mcts/_cboard.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }),
)
