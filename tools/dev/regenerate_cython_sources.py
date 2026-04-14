"""Regenerate tracked C sources from the project's Cython modules."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Cython.Build import cythonize
from setuptools import Extension

_setup_spec = importlib.util.spec_from_file_location("pyneutube_setup", ROOT / "setup.py")
if _setup_spec is None or _setup_spec.loader is None:
    raise RuntimeError("Unable to load setup.py for compiler directives.")
_setup_module = importlib.util.module_from_spec(_setup_spec)
_setup_spec.loader.exec_module(_setup_module)
COMPILER_DIRECTIVES = _setup_module.COMPILER_DIRECTIVES

MODULE_PATHS = [
    "pyneutube/core/processing/local_maximum.pyx",
    "pyneutube/core/processing/transform.pyx",
    "pyneutube/core/processing/sampling.pyx",
    "pyneutube/tracers/pyNeuTube/filters.pyx",
    "pyneutube/tracers/pyNeuTube/geometry_accel.pyx",
    "pyneutube/tracers/pyNeuTube/seg_utils.pyx",
    "pyneutube/tracers/pyNeuTube/shortest_path_accel.pyx",
    "pyneutube/tracers/pyNeuTube/stack_graph_utils.pyx",
]


def main() -> None:
    extensions = [Extension("*", [str(ROOT / relative_path)]) for relative_path in MODULE_PATHS]
    cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES, force=True)


if __name__ == "__main__":
    main()
