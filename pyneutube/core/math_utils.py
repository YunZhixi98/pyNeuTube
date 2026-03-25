# core/math_utils.py
"""
math_utils.py

Utilities for mathematical operations.
"""

from typing import Union, Tuple

import numpy as np


def get_bounding_box(coords: np.ndarray, 
                     image_shape: Union[np.ndarray, Tuple, None] = None
                     ) -> np.ndarray:
    """
    Calculate the axis-aligned bounding box for given coordinates.

    Parameters
    ----------
        coords (np.ndarray): 
            Array of coordinates with shape (N, D) where N is number of points and D is dimensionality.
        image_shape (Union[np.ndarray, Tuple, None]): 
            Shape of image.

    Returns
    -------
        np.ndarray: Bounding box as array with shape (2, D) where first row contains
                   minimum coordinates and second row contains maximum coordinates.
    """

    coords = np.asarray(coords, dtype=np.float64)
    image_shape = np.asarray(image_shape, dtype=np.int16)[::-1]

    min_coords = np.floor(np.min(coords, axis=0) - 1.5).astype(int)
    max_coords = np.ceil(np.max(coords, axis=0) + 1.5).astype(int)

    if image_shape is not None:
        min_coords = np.clip(min_coords, 0, image_shape - 1)
        max_coords = np.clip(max_coords, 0, image_shape - 1)


    return min_coords, max_coords
