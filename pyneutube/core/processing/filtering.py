# core/processing/filtering.py

"""
filtering.py

Utilities for filtering image signals.
"""

import time
from typing import Callable, Optional, Union

import numpy as np
from fast_histogram import histogram1d
from scipy.ndimage import convolve, label, maximum_filter
# from cupyx.ndarray import convolve as cupy_convolve

from pyneutube.core.neighbors import check_kernel_and_neighbors, neighbors_26_forward
from pyneutube.core.processing.local_maximum import Stack_Local_Max, Stack_Locmax_Region


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
    if min_value is None:
        min_value = np.min(image)
    if max_value is None:
        max_value = np.max(image)
    if min_value is None or max_value is None:
        raise ValueError("`min_value` and `max_value` must not be None.")
    
    # cast to float, in case of implicit out-of-range conversion
    min_value = int(min_value)
    max_value = int(max_value)

    bin_range = (min_value, max_value + 1)
    bins = max_value - min_value + 1

    if min_value < 0 or max_value < 0:
        raise ValueError("`min_value` and `max_value` must be non-negative.")
    if min_value >= max_value:
        raise ValueError("`min_value` must be less than `max_value`.")
    
    # Calculate histogram
    hist = histogram1d(np.asarray(image, dtype=np.float64).ravel(), bins=bins, range=bin_range).astype(np.int64)
    bin_edges = np.arange(min_value, max_value + 2, dtype=np.float64)

    # Find peak, lowest and highest gray levels.
    # To generalize, find out the peak at given region
    if max_height_value is not None:
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
        hist_search_mask = bin_centers <= max_height_value
        hist = hist[hist_search_mask]
        bin_edges = bin_edges[:len(hist)+1]
        bins = len(hist)

    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]

    if arg_low_level == arg_high_level:
        # Image has constant intensity.
        return max_value
    
    # Flip is True if left tail is shorter.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = bins - arg_high_level - 1
        arg_peak_height = bins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del arg_high_level

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    # Normalize. Construct the normalized x and y vectors along the hypotenuse.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm  # y
    width /= norm  # x

    # Maximize the length.
    length = peak_height * x1 - width * y1  # cross product as a measure between histogram and hypotenuse
    length[y1==0] = 0
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = bins - arg_level - 1

    return bin_edges[arg_level]


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
    re-encapsulate Stack_Locmax_Region in Cython.
    """
    image = np.ascontiguousarray(image, dtype=np.float64)
    img_padding = np.pad(image, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    loc_max_mask = (img_padding != 0).astype(np.uint8)
    loc_max_mask = Stack_Locmax_Region(img_padding, loc_max_mask)

    return loc_max_mask


def refine_local_max_threshold(
    image: np.ndarray,
    init_thresh: float,
    *,
    low_ratio: float = 0.01,
    high_ratio: float = 0.05,
    drop_factor_if_low: float = 0.3,
    drop_factor_if_high: float = 0.5,
    threshold_finder: Callable[[np.ndarray, float, float], float] = triangle_threshold
) -> float:
    """
    Refine an initial foreground threshold by “peeling away” peaks until
    the foreground ratio meets stability/drop criteria.

    Parameters
    ----------
    image : np.ndarray
        Input intensity image.
    init_thresh : float
        Initial threshold.
    low_ratio : float
        Lower bound on acceptable foreground fraction.
    high_ratio : float
        Upper bound on acceptable foreground fraction.
    drop_factor_if_low : float
        If initial FG ratio ∈ (low_ratio, high_ratio], we accept a single
        new threshold only if FG ratio drops by at least this factor.
    drop_factor_if_high : float
        If initial FG ratio > high_ratio, we iteratively peel peaks
        until FG ratio ≤ high_ratio; at each peel, we accept a new
        threshold if the drop vs. previous ratio ≥ this factor.
    threshold_finder : Callable
        Function(image, low, high) → new_threshold. Defaults to
        `triangle_threshold(image, low, high)`.

    Returns
    -------
    float
        Refined threshold.
    """

    total_voxels = image.size
    max_val = float(image.max())

    def fg_ratio_above(t: float) -> float:
        """Compute foreground fraction above threshold t."""
        return np.count_nonzero(image > t) / total_voxels

    # Initial ratio
    current_thresh = init_thresh
    initial_ratio = fg_ratio_above(current_thresh)

    # Case 1: “just barely” too much FG → one-shot test
    if low_ratio < initial_ratio <= high_ratio:
        candidate = threshold_finder(image, current_thresh + 1, max_val - 1)
        new_ratio = fg_ratio_above(candidate)
        if new_ratio <= initial_ratio * drop_factor_if_low:
            return candidate
        return current_thresh

    # Case 2: too much FG → iterative peeling
    if initial_ratio > high_ratio:
        prev_ratio = initial_ratio
        while True:
            candidate = threshold_finder(image, current_thresh + 1, max_val - 1)
            ratio = fg_ratio_above(candidate)

            # Stop if we've dipped below the acceptable FG fraction
            if ratio <= high_ratio:
                # Accept candidate if we had a big enough drop
                if ratio <= prev_ratio * drop_factor_if_high:
                    return candidate
                break

            # Accept intermediate peel if drop is big enough
            if ratio <= prev_ratio * drop_factor_if_high:
                current_thresh = candidate
            prev_ratio = ratio

        return current_thresh

    # Case 3: too little FG already — nothing to do
    return current_thresh


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
    
    # Calculate the actual number of neighborhoods (without considering boundaries)
    ones = np.ones_like(binary_image, dtype=np.uint8)
    actual_neighbors = convolve(ones, kernel, mode='constant', cval=0)
    
    # Calculate the number of signals in the neighborhood
    neighbor_count = convolve(binary_image, kernel, mode='constant', cval=0)
    
    # apply boundary threshold：min_neighbors * actual_neighbors / self.neighbors
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
