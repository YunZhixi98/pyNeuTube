"""Lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyneutube.core.io.image_parser import ImageParser
from pyneutube.core.io.swc_parser import Neuron


def _load_volume(image: np.ndarray | str | Path) -> tuple[np.ndarray, str | None]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        return ImageParser(path).load(), path.name
    return np.asarray(image), None


def _load_trace(trace: Neuron | np.ndarray | str | Path) -> Neuron | np.ndarray:
    if isinstance(trace, Neuron):
        return trace
    if isinstance(trace, (str, Path)):
        return Neuron().initialize(trace)
    return np.asarray(trace, dtype=float)


def _plot_trace(ax: plt.Axes, trace: Neuron | np.ndarray, *, color: str) -> None:
    if isinstance(trace, Neuron):
        for node in trace.swc:
            parent_idx = trace.nidHash.get(node[6])
            if parent_idx is None:
                continue
            parent = trace.swc[parent_idx]
            ax.plot([node[2], parent[2]], [node[3], parent[3]], "-", color=color, linewidth=1.0)
        return

    coords = np.asarray(trace, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinate traces must have shape (N, 3).")
    if len(coords) == 0:
        raise ValueError("No coordinates are available for visualization.")
    ax.plot(coords[:, 0], coords[:, 1], ".", color=color, markersize=1.5, alpha=0.8)


def save_overlay_figure(
    image: np.ndarray | str | Path,
    trace: Neuron | np.ndarray | str | Path,
    output_path: str | Path,
    *,
    color: str = "tab:orange",
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Save a maximum-intensity projection overlay for a trace on a 3D volume.

    `image` accepts either an in-memory volume or an image path supported by `ImageParser`.
    `trace` accepts a `Neuron`, an SWC path, or an `(N, 3)` coordinate array.
    """

    volume, default_title = _load_volume(image)
    loaded_trace = _load_trace(trace)
    output_path = Path(output_path)

    mip = np.max(np.asarray(volume), axis=0)
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    ax.imshow(mip, cmap="gray", origin="lower")
    _plot_trace(ax, loaded_trace, color=color)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or default_title or output_path.stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
