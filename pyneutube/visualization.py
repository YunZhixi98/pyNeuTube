"""Lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from pyneutube.core.io.image_parser import ImageParser
from pyneutube.core.io.swc_parser import Neuron

if TYPE_CHECKING:
    from pyneutube.tracers.pyNeuTube.seeds import Seeds
    from pyneutube.tracers.pyNeuTube.tracing import SegmentChains


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


def _project_overlay_image(volume: np.ndarray, *, log_transform: bool) -> np.ndarray:
    mip = np.max(np.asarray(volume, dtype=np.float64), axis=0)
    if log_transform:
        mip = np.log1p(np.clip(mip, 0, None))
    return mip


def _make_overlay_axes(
    volume: np.ndarray,
    *,
    title: str,
    dpi: int,
    log_transform: bool,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    mip = _project_overlay_image(volume, log_transform=log_transform)
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True, dpi=dpi)
    ax.imshow(mip, cmap="gray", origin="lower")
    ax.set_xlim(0, mip.shape[1])
    ax.set_ylim(0, mip.shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return fig, ax, mip


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


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


def _iter_seeds(seeds: Any) -> list[Any]:
    if hasattr(seeds, "_seeds"):
        return list(seeds)
    return list(seeds)


def _iter_chains(chains: Any) -> list[Any]:
    if hasattr(chains, "_chains"):
        return list(chains)
    return list(chains)


def save_overlay_figure(
    image: np.ndarray | str | Path,
    trace: Neuron | np.ndarray | str | Path,
    output_path: str | Path,
    *,
    color: str = "tab:orange",
    title: str | None = None,
    dpi: int = 200,
    log_transform: bool = True,
) -> Path:
    """Save a maximum-intensity projection overlay for a trace on a 3D volume.

    `image` accepts either an in-memory volume or an image path supported by `ImageParser`.
    `trace` accepts a `Neuron`, an SWC path, or an `(N, 3)` coordinate array.
    """

    volume, default_title = _load_volume(image)
    loaded_trace = _load_trace(trace)
    fig, ax, _mip = _make_overlay_axes(
        volume,
        title=title or default_title or Path(output_path).stem,
        dpi=dpi,
        log_transform=log_transform,
    )
    _plot_trace(ax, loaded_trace, color=color)
    return _save_figure(fig, output_path)


def save_seed_overlay_figure(
    image: np.ndarray | str | Path,
    seeds: Seeds | Any,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 200,
    log_transform: bool = True,
    cmap: str = "viridis",
) -> Path:
    """Save a MIP overlay for fitted tracing seeds.

    Each seed is drawn as a center point plus the fitted segment projected to the XY plane.
    Seed colors vary with depth along the z axis.
    """

    volume, default_title = _load_volume(image)
    seed_items = _iter_seeds(seeds)
    if not seed_items:
        raise ValueError("No seeds are available for visualization.")

    fig, ax, _mip = _make_overlay_axes(
        volume,
        title=title or default_title or Path(output_path).stem,
        dpi=dpi,
        log_transform=log_transform,
    )

    z_values = np.array([float(seed.seg.center_coord[2]) for seed in seed_items], dtype=np.float64)
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_range = z_max - z_min
    color_map = plt.get_cmap(cmap)

    for seed, z_value in zip(seed_items, z_values, strict=True):
        seg = seed.seg
        if z_range == 0:
            color = color_map(0.5)
        else:
            color = color_map((z_value - z_min) / z_range)
        ax.plot(
            [seg.start_coord[0], seg.end_coord[0]],
            [seg.start_coord[1], seg.end_coord[1]],
            "-",
            color=color,
            linewidth=1.0,
            alpha=0.9,
        )
        ax.scatter(
            [seg.center_coord[0]],
            [seg.center_coord[1]],
            s=10,
            c=[color],
            edgecolors="none",
            alpha=0.95,
        )

    return _save_figure(fig, output_path)


def save_chain_overlay_figure(
    image: np.ndarray | str | Path,
    chains: SegmentChains | Any,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 200,
    log_transform: bool = True,
    random_seed: int = 0,
) -> Path:
    """Save a MIP overlay for tracing chains.

    Each chain is shown with a deterministic random color and includes node markers.
    """

    volume, default_title = _load_volume(image)
    chain_items = [chain for chain in _iter_chains(chains) if len(chain) > 0]
    if not chain_items:
        raise ValueError("No chains are available for visualization.")

    fig, ax, _mip = _make_overlay_axes(
        volume,
        title=title or default_title or Path(output_path).stem,
        dpi=dpi,
        log_transform=log_transform,
    )
    rng = np.random.default_rng(random_seed)

    for chain in chain_items:
        coords = np.asarray(chain.to_coords(), dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
            continue
        color = rng.random(3)
        ax.plot(coords[:, 0], coords[:, 1], "-", color=color, linewidth=1.0, alpha=0.9)
        ax.scatter(coords[:, 0], coords[:, 1], s=6, c=[color], edgecolors="none", alpha=0.95)

    return _save_figure(fig, output_path)
