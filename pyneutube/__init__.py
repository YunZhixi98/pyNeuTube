"""Public package interface for PyNeuTube."""

from __future__ import annotations

from ._version import __version__
from .io import ImageParser, Neuron
from .processing import (
    connectivity_filter,
    local_max_filter,
    refine_local_max_threshold,
    subtract_background,
    threshold_filter,
    triangle_threshold,
)
from .tracing import (
    SUPPORTED_IMAGE_SUFFIXES,
    preprocess_volume,
    trace_directory,
    trace_file,
    trace_files,
    trace_volume,
)
from .visualization import save_overlay_figure

__all__ = [
    "__version__",
    "trace_file",
    "trace_files",
    "trace_directory",
    "trace_volume",
    "ImageParser",
    "Neuron",
    "SUPPORTED_IMAGE_SUFFIXES",
    "preprocess_volume",
    "subtract_background",
    "threshold_filter",
    "triangle_threshold",
    "refine_local_max_threshold",
    "local_max_filter",
    "connectivity_filter",
    "save_overlay_figure",
]
