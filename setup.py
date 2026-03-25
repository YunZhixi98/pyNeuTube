from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - optional at build time
    cythonize = None


ROOT = Path(__file__).parent.resolve()
COMPILER_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "embedsignature": True,
    "profile": False,
}


def _compiled_source(stem: str) -> str:
    for suffix in (".c", ".cpp"):
        candidate = ROOT / f"{stem}{suffix}"
        if candidate.exists():
            return f"{stem}{suffix}".replace("\\", "/")
    raise FileNotFoundError(f"Missing generated C/C++ source for {stem!r}.")


def _extension_source(stem: str, *, use_cython: bool) -> str:
    if use_cython:
        return f"{stem}.pyx".replace("\\", "/")
    return _compiled_source(stem)


def _extra_compile_args() -> list[str]:
    if sys.platform == "win32":
        return ["/O2"]
    return ["-O3"]


def build_extensions(*, use_cython: bool = False) -> list[Extension]:
    extensions = [
        Extension(
            "pyneutube.core.processing.local_maximum",
            [_extension_source("pyneutube/core/processing/local_maximum", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.core.processing.transform",
            [_extension_source("pyneutube/core/processing/transform", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.core.processing.sampling",
            [_extension_source("pyneutube/core/processing/sampling", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.filters",
            [_extension_source("pyneutube/tracers/pyNeuTube/filters", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.seg_utils",
            [_extension_source("pyneutube/tracers/pyNeuTube/seg_utils", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.stack_graph_utils",
            [_extension_source("pyneutube/tracers/pyNeuTube/stack_graph_utils", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.geometry_accel",
            [_extension_source("pyneutube/tracers/pyNeuTube/geometry_accel", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
    ]

    if use_cython:
        if cythonize is None:
            raise RuntimeError(
                "PYNEUTUBE_USE_CYTHON=1 requires Cython to be installed in the build environment."
            )
        return cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES)

    return extensions


def build_setup_kwargs(*, use_cython: bool = False) -> dict[str, object]:
    return {"ext_modules": build_extensions(use_cython=use_cython)}


if __name__ == "__main__":
    setup(**build_setup_kwargs(use_cython=False))
