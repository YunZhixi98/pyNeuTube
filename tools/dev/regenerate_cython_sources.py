"""Regenerate tracked C sources from the project's Cython modules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cython.Build import cythonize
from setuptools import Extension

from setup import COMPILER_DIRECTIVES

MODULE_PATHS = [
    "pyneutube/core/processing/local_maximum.pyx",
    "pyneutube/core/processing/transform.pyx",
    "pyneutube/core/processing/sampling.pyx",
    "pyneutube/tracers/pyNeuTube/filters.pyx",
    "pyneutube/tracers/pyNeuTube/geometry_accel.pyx",
    "pyneutube/tracers/pyNeuTube/seg_utils.pyx",
    "pyneutube/tracers/pyNeuTube/stack_graph_utils.pyx",
]


def main() -> None:
    extensions = [Extension("*", [str(ROOT / relative_path)]) for relative_path in MODULE_PATHS]
    cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES, force=True)


if __name__ == "__main__":
    main()
