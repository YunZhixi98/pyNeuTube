"""Micro-benchmark for vectorized voxel sampling."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.ndimage import map_coordinates

from pyneutube.core.processing.sampling import sample_voxels


def pointwise_map_coordinates(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    values = np.empty(coords.shape[0], dtype=np.float64)
    for i, coord in enumerate(coords):
        values[i] = map_coordinates(
            image,
            coord[::-1].reshape(3, 1),
            order=1,
            mode="constant",
            cval=0.0,
        )[0]
    return values


def timed(func, *args) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - start


def main() -> None:
    rng = np.random.default_rng(42)
    image = rng.random((64, 96, 96), dtype=np.float64)
    coords = rng.uniform(low=0.0, high=95.0, size=(25_000, 3))

    baseline, baseline_time = timed(pointwise_map_coordinates, image, coords)
    vectorized, vectorized_time = timed(sample_voxels, image, coords)

    print(f"pointwise_map_coordinates: {baseline_time:.4f}s")
    print(f"sample_voxels:             {vectorized_time:.4f}s")
    print(f"speedup:                   {baseline_time / vectorized_time:.2f}x")
    print(f"max_abs_diff:              {np.max(np.abs(baseline - vectorized)):.6e}")


if __name__ == "__main__":
    main()
