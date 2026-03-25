# core/processing/segmentation.py

"""
segmentation.py

Utilities for segmentation operations on images.
"""

from typing import Optional, Union

import numpy as np
from scipy.ndimage import label

from pyneutube.core.neighbors import check_kernel_and_neighbors


def label_connected_components(binary_image: np.ndarray, 
                               kernel: Optional[np.ndarray] = None, 
                               n_neighbors: Optional[int] = None) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary image.

    Parameters
    ----------
    image : np.ndarray
        The input binary image to label.
    kernel : np.ndarray, optional
        The structuring element used for connectivity.
    n_neighbors : int, optional
        The number of neighbors for connectivity.

    Returns
    -------
    np.ndarray
        Labeled image where each connected component has a unique label.
    int
        The number of connected components found in the image.
    """
    kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)
    
    labeled_image, num_features = label(binary_image, structure=kernel, output=None)
    
    return labeled_image, num_features
