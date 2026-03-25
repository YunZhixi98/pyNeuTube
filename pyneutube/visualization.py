"""Lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_overlay_figure(
    image: np.ndarray,
    node_coords: np.ndarray,
    output_path: str | Path,
    *,
    color: str = "tab:orange",
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Save a maximum-intensity-projection overlay of SWC nodes on an image."""

    output_path = Path(output_path)
    node_coords = np.asarray(node_coords, dtype=float)
    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError("`node_coords` must have shape (N, 3).")

    mip = np.max(np.asarray(image), axis=0)
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    ax.imshow(mip, cmap="gray", origin="lower")
    ax.plot(node_coords[:, 0], node_coords[:, 1], ".", color=color, markersize=1.5, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
