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
    connect_trace_chains,
    extract_trace_seeds,
    generate_trace_chains,
    preprocess_volume,
    trace_file,
    trace_volume,
    trace_files,
    trace_directory,
)
from .visualization import save_overlay_figure

__all__ = [
    "__version__",
    "trace_file",
    "trace_volume",
    "trace_files",
    "trace_directory",
    "save_overlay_figure",
    "ImageParser",
    "Neuron",
    "preprocess_volume",
    "extract_trace_seeds",
    "generate_trace_chains",
    "connect_trace_chains",
    "SUPPORTED_IMAGE_SUFFIXES",
    "subtract_background",
    "threshold_filter",
    "triangle_threshold",
    "refine_local_max_threshold",
    "local_max_filter",
    "connectivity_filter",
]
