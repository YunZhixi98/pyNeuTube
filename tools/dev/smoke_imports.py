"""Basic import smoke test for local development and release verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyneutube import (
    ImageParser,
    Neuron,
    SUPPORTED_IMAGE_SUFFIXES,
    connectivity_filter,
    local_max_filter,
    preprocess_volume,
    refine_local_max_threshold,
    save_overlay_figure,
    subtract_background,
    threshold_filter,
    trace_directory,
    trace_file,
    trace_volume,
    triangle_threshold,
)
from pyneutube.cli import main as cli_main
from pyneutube.core.processing import local_maximum, sampling, transform
from pyneutube.tracers.pyNeuTube import filters, geometry_accel, seg_utils, stack_graph_utils
from pyneutube.tracers.pyNeuTube.chains_to_morphology import ChainConnector


def main() -> None:
    print("Imports OK:")
    print(
        ImageParser,
        Neuron,
        SUPPORTED_IMAGE_SUFFIXES,
        subtract_background,
        threshold_filter,
        triangle_threshold,
        refine_local_max_threshold,
        local_max_filter,
        connectivity_filter,
        preprocess_volume,
        trace_volume,
        trace_file,
        trace_directory,
        save_overlay_figure,
        cli_main,
        local_maximum,
        sampling,
        transform,
        filters,
        seg_utils,
        stack_graph_utils,
        geometry_accel,
        ChainConnector,
    )


if __name__ == "__main__":
    main()
