# core/processing/filtering.py

"""
filtering.py

Utilities for filtering image signals.
"""

import time
from typing import Callable, Optional, Union

import numpy as np
from fast_histogram import histogram1d
from scipy.ndimage import convolve, generate_binary_structure, label, maximum_filter
# from cupyx.ndarray import convolve as cupy_convolve

from pyneutube.core.neighbors import check_kernel_and_neighbors, neighbors_26_forward
from pyneutube.core.processing.local_maximum import Stack_Local_Max, Stack_Locmax_Region


_CONNECTIVITY_18_STRUCTURE = generate_binary_structure(3, 2)


def subtract_background(
    image: np.ndarray,
    min_fraction: float = 0.5,
    max_iterations: int = 3,
    *,
    verbose: int = 0,
) -> np.ndarray:
    """
    Subtract a common background intensity from a grayscale image.

    We build the intensity histogram of the image, then look for the
    largest contiguous “background” peak such that its relative frequency
    falls below `min_fraction`, iterating at most `max_iterations` times.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D array of integers (int8/16/32, etc.), representing your image.
    min_fraction : float
        Minimum fraction of total voxels allowed in the background peak.
    max_iterations : int
        Maximum number of re-histogramming steps to peel away successive peaks.

    Returns
    -------
    np.ndarray
        A new array, same shape and dtype as `image`, with background
        intensity subtracted and values clipped at zero.
    """

    flat = image.ravel()
    total = flat.size

    # Build histogram over [minV…maxV]
    imin, imax = int(flat.min()), int(flat.max())
    #counts, _ = np.histogram(flat,                     # original version
    #                         bins=(imax - imin + 1),
    #                         range=(imin, imax + 1))
    #counts = np.bincount(flat - imin, minlength=imax - imin + 1)   # faster
    counts = histogram1d(flat, bins=imax - imin + 1, range=(imin, imax + 1)).astype('int64')   # fastest

    common_intensity = 0

    # Initial “early accept” test:
    if counts[0] / total > 0.9:
        common_intensity = imin
    else:
        for _ in range(max_iterations):
            low = common_intensity + 1
            if low > imax:
                break

            # Find mode in [low … imax]
            sub = counts[(low - imin):]        # counts at intensities ≥ low
            if sub.size == 0:
                break

            offset = int(np.argmax(sub))
            mode = low + offset
            common_intensity = mode

            # If mode is at the very top, stop
            if common_intensity == imax:
                break

            # Compute foreground fraction ABOVE the new common intensity
            upper = counts[(common_intensity - imin + 1):]
            fg_fraction = upper.sum() / total

            if fg_fraction < min_fraction:
                break

    if verbose:
        print(f"Subtracting background: {common_intensity}")
    if common_intensity <= imin:    # escape calculation if not necessary
        out = image - common_intensity
        return out
    else:
        out = image.astype(np.int16)    # broadcast to int16 is enough for this case
        out -= common_intensity
        np.clip(out, 0, None, out=out)

        return out.astype(image.dtype)


def threshold_filter(image: np.ndarray, threshold: Union[int, float]) -> np.ndarray:
    """
    Apply a threshold filter to the image, keeping only values above the threshold.
    
    Parameters
    ----------
    image : np.ndarray
        The input image to filter.
    threshold : float
        The threshold value.

    Returns
    -------
    np.ndarray
        The binary mask where values above the threshold are retained.
    """

    if threshold < 0:
        raise ValueError("`threshold` must be non-negative.")
    
    binary_image = (image > threshold).astype(np.uint8)

    return binary_image


def triangle_threshold(image: np.ndarray, 
                       min_value: Optional[int] = None, 
                       max_value: Optional[int] = None,
                       max_height_value: Optional[int] = None) -> np.ndarray:
    """
    Apply a triangle threshold filter to the image or signal

    Parameters
    ----------
    image : np.ndarray
        The input image or indexed signal to filter.
    min_value : Optional[float]
        Minimum value when computing histogram. Default is the minimum value in the image.
    max_value : Optional[float]
        Maximum value when computing histogram. Default is the maximum value in the image.
    max_height_value: Optional[float]
        Maximum value to find the peak.

    Returns
    -------
    np.ndarray
        The threshold value determined by the triangle method.
    """
    values = np.asarray(image, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("`image` must contain at least one value.")

    if min_value is None:
        min_value = int(values.min())
    if max_value is None:
        max_value = int(values.max())
    if min_value is None or max_value is None:
        raise ValueError("`min_value` and `max_value` must not be None.")

    min_value = int(min_value)
    max_value = int(max_value)

    if min_value < 0 or max_value < 0:
        raise ValueError("`min_value` and `max_value` must be non-negative.")
    if min_value > max_value:
        raise ValueError("`min_value` must be less than or equal to `max_value`.")

    hist = histogram1d(
        values,
        bins=max_value - min_value + 1,
        range=(min_value, max_value + 1),
    ).astype(np.int64)

    low = min_value
    high = max_value if max_height_value is None else min(max_value, int(max_height_value))
    if high < low:
        raise ValueError("`max_height_value` must be greater than or equal to `min_value`.")

    hist_view = hist[(low - min_value):(high - min_value + 1)]
    if hist_view.size == 0:
        raise ValueError("No histogram bins remain in the requested range.")

    max_index = int(np.argmax(hist_view))
    tail_hist = hist_view[max_index:]
    tail_nonzero = np.flatnonzero(tail_hist)
    min_tail_value = int(tail_hist[tail_nonzero].min())
    min_index = max_index + int(tail_nonzero[tail_hist[tail_nonzero] == min_tail_value][-1])

    if max_index == min_index:
        return float(low - 1)

    hist_segment = hist_view[max_index:min_index + 1]
    tail_value = int(hist_segment[-1])
    norm_factor = (hist_segment.size - 1) / float(hist_segment[0] - tail_value)
    normalized_hist = (hist_segment.astype(np.float64) - tail_value) * norm_factor

    threshold_offset = 0
    best_score = normalized_hist[0]
    for index in range(1, hist_segment.size):
        if hist_segment[index] <= 0:
            continue
        score = normalized_hist[index] + float(index)
        if score < best_score:
            best_score = score
            threshold_offset = index

    return float(low + max_index + threshold_offset)


def rc_threshold(image: np.ndarray, 
                 min_value: Optional[int] = None, 
                 max_value: Optional[int] = None) -> np.ndarray:
    """
    Riddler-Calvard Method thresholding
    """

    def centroid(counts: np.ndarray) -> float:
        """
        Compute centroid index of a 1D histogram counts array.
        Returns a floating-point centroid in [0, len(counts)-1].
        If counts.sum() == 0, returns the midpoint (len(counts)-1)/2.0.
        """
        counts = np.asarray(counts, dtype=np.float64)
        total = counts.sum()
        if total == 0:
            # fallback to middle of the range (same-sized behavior as a "neutral" centroid)
            return (len(counts) - 1) / 2.0
        indices = np.arange(len(counts), dtype=np.float64)
        return (counts * indices).sum() / total

    flat = image.ravel()
    total = flat.size

    # Build histogram over [minV…maxV]
    if min_value is not None:
        imin = min_value
    else:
        imin = int(flat.min())
    if max_value is not None:
        imax = max_value
    else:
        imax = int(flat.max())

    counts = histogram1d(flat, bins=imax - imin + 1, range=(imin, imax + 1)).astype('int64')   # fastest

    thresh = int((imin + imax) / 2.0 - imin)
    c1, c2 = 0, 0  # define center

    prev_thresh = thresh
    
    while True:
        # if threshold equals last index, break (no second class)
        if thresh >= total - 1:
            break

        # class1: indices 0..thre
        # class2: indices thre+1..length-1
        c1_rel = centroid(counts[: thresh + 1]) if thresh >= 0 else centroid(np.array([]))
        c2_rel = centroid(counts[thresh + 1 :])  # centroid returns relative index in that subarray

        # convert c2 to relative-to-hist_slice coordinates
        c2_rel += (thresh + 1)

        c1, c2 = c1_rel, c2_rel

        new_thresh = int((c1 + c2) / 2.0)

        # Convergence check
        if new_thresh == prev_thresh:
            break

        prev_thresh = new_thresh

    # convert back to gray value
    thresh += imin
    c1 += imin
    c2 += imin

    return thresh, c1, c2


def local_max_filter(image: np.ndarray) -> np.ndarray:
    """
    Re-encapsulate `Stack_Locmax_Region` and keep one seed voxel per plateau.
    """
    image = np.ascontiguousarray(image, dtype=np.float64)
    img_padding = np.pad(image, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0)
    loc_max_mask = (img_padding != 0).astype(np.uint8)
    loc_max_mask = Stack_Locmax_Region(img_padding, loc_max_mask)
    if not np.any(loc_max_mask):
        return loc_max_mask

    labeled, _ = label(loc_max_mask > 0, structure=_CONNECTIVITY_18_STRUCTURE)
    labeled_flat = labeled.ravel()
    nonzero_positions = np.flatnonzero(labeled_flat)
    _, first_indices = np.unique(labeled_flat[nonzero_positions], return_index=True)
    representative_positions = nonzero_positions[first_indices]
    representative_mask = np.zeros_like(loc_max_mask, dtype=np.uint8)
    representative_mask.ravel()[representative_positions] = 1

    return representative_mask

def refine_local_max_threshold(
    image: np.ndarray,
    init_thresh: float,
    *,
    threshold_source: Optional[np.ndarray] = None,
    low_ratio: float = 0.01,
    high_ratio: float = 0.05,
    drop_factor_if_low: float = 0.3,
    drop_factor_if_high: float = 0.5,
    max_iterations: int = 3,
    threshold_finder: Callable[[np.ndarray, float, float], float] = triangle_threshold,
) -> float:
    """
    Refine an initial foreground threshold using the NeuTube retry logic.

    `image` is used for foreground-area ratios, while `threshold_source`
    provides the histogram source for retry thresholds.
    """

    total_voxels = image.size
    threshold_values = image if threshold_source is None else threshold_source
    threshold_values = np.asarray(threshold_values, dtype=np.float64).ravel()
    if threshold_values.size == 0:
        return float(init_thresh)

    upper_bound = float(threshold_values.max())

    def fg_ratio_above(threshold: float) -> float:
        return np.count_nonzero(image > threshold) / total_voxels

    current_thresh = float(init_thresh)
    initial_ratio = fg_ratio_above(current_thresh)

    if low_ratio < initial_ratio <= high_ratio:
        if current_thresh + 1 > upper_bound - 1:
            return current_thresh
        candidate = threshold_finder(threshold_values, current_thresh + 1, upper_bound - 1)
        if fg_ratio_above(candidate) <= initial_ratio * drop_factor_if_low:
            return float(candidate)
        return current_thresh

    if initial_ratio > high_ratio:
        candidate = current_thresh
        ratio = initial_ratio
        prev_ratio = initial_ratio
        retries_remaining = max(0, int(max_iterations))
        while ratio > high_ratio and retries_remaining > 0:
            if candidate + 1 > upper_bound - 1:
                break
            candidate = threshold_finder(threshold_values, candidate + 1, upper_bound - 1)
            ratio = fg_ratio_above(candidate)
            if ratio <= prev_ratio * drop_factor_if_high:
                current_thresh = float(candidate)
            prev_ratio = ratio
            retries_remaining -= 1
        return current_thresh

    return current_thresh


def _actual_neighbor_count(kernel: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    if kernel.shape == (3, 3, 3) and int(np.sum(kernel)) == 26 and kernel[1, 1, 1] == 0:
        depth, height, width = image_shape
        z = np.minimum(np.arange(depth), np.arange(depth)[::-1])[:, None, None]
        y = np.minimum(np.arange(height), np.arange(height)[::-1])[None, :, None]
        x = np.minimum(np.arange(width), np.arange(width)[::-1])[None, None, :]
        z_count = np.where(z == 0, 2, 3)
        y_count = np.where(y == 0, 2, 3)
        x_count = np.where(x == 0, 2, 3)
        return (z_count * y_count * x_count - 1).astype(np.uint8, copy=False)

    ones = np.ones(image_shape, dtype=np.uint8)
    return convolve(ones, kernel, mode="constant", cval=0)


def connectivity_filter(image: np.ndarray, min_neighbors: int, 
                        kernel: Optional[np.ndarray] = None, 
                        n_neighbors: Optional[int] = None) -> np.ndarray:
    """
    Apply a connectivity filter to the image, keeping only regions with at least `min_neighbors` neighbors.
    
    Parameters
    ----------
    image : np.ndarray
        The input image to filter.
    min_neighbors : int
        Minimum number of neighbors required to keep a voxel.
    kernel : Optional[np.ndarray]
        Kernel used for convolution.
    n_neighbors : Optional[int]
        Number of neighbors to consider (e.g., 18 or 26).

    Returns
    -------
    np.ndarray
        The binary mask where only regions with sufficient connectivity are retained.
    """

    if min_neighbors <= 0:
        raise ValueError("`min_neighbors` must be a positive integer.")
    kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)

    binary_image = (image > 0).astype(np.uint8)
    actual_neighbors = _actual_neighbor_count(kernel, binary_image.shape)
    neighbor_count = convolve(binary_image, kernel, mode="constant", cval=0)
    threshold = (min_neighbors * actual_neighbors) / n_neighbors
    mask = ((binary_image > 0) & (neighbor_count >= threshold)).astype(np.uint8)

    return mask

def maximum_filter_mask1(image: np.ndarray, 
                        kernel: Optional[np.ndarray] = None,
                        n_neighbors: Optional[int] = None) -> np.ndarray:
    """
    Apply a maximum filter to the image and return a mask of local maxima.

    Parameters
    ----------
    image : np.ndarray
        The input image to filter.
    kernel : Optional[np.ndarray]
        Kernel used for convolution.
    n_neighbors : Optional[int]
        Number of neighbors to consider (e.g., 18 or 26).

    Returns
    -------
    np.ndarray
        A binary mask where local maxima are set to 1 and others to 0.
    """
    kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)
    # Apply maximum filter
    max_filtered = maximum_filter(image, footprint=kernel, mode='constant', cval=0)

    mask = (image > 0) & (image > max_filtered)  # `image > max_filtered` because the kernel has been excluded the center voxel.

    return mask.astype(np.uint8)


def maximum_filter_mask(image: np.ndarray, *, verbose: int = 0) -> np.ndarray:
    """
    - 13 forward neighbors only
    - center<neighbor → zero center
    - center>neighbor → zero neighbor
    - center==neighbor → zero neighbor
    """
    import time
    t0 = time.time()
    binary_image = Stack_Local_Max(np.ascontiguousarray(image, dtype=np.float64))
    if verbose:
        print(f"Stack_Local_Max: {time.time() - t0:.3f}s")
    return binary_image
