# core/processing/morphology.py

"""
morphology.py

Utilities for image morphology operations.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion, 
    binary_closing, binary_opening,
    generate_binary_structure
)

from pyneutube.core.neighbors import check_kernel_and_neighbors


class Morphology:
    """
    Class for performing morphological operations on binary images.
    This class is a re-encapsulation of common morphological operations in scipy.ndimage.
    """

    def __init__(self):
        """
        Initialize the Morphology class.
        No instance attributes are required since all methods are static.
        """
        pass

    @staticmethod
    def binary_dilation(image: np.ndarray, kernel: Optional[np.ndarray] = None, 
                        n_neighbors: Optional[int] = None) -> np.ndarray:
        """
        Perform binary dilation on the input image using the specified kernel.

        Parameters
        ----------
        image : np.ndarray
            The input binary image to dilate.
        kernel : np.ndarray
            The structuring element used for dilation.

        Returns
        -------
        np.ndarray
            The dilated binary image.
        """
        kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)
        kernel[tuple(np.array(kernel.shape)//2)] = 1
        
        return binary_dilation(image, structure=kernel)

    @staticmethod
    def binary_erosion(image: np.ndarray, kernel: Optional[np.ndarray] = None, 
                        n_neighbors: Optional[int] = None) -> np.ndarray:
        """
        Perform binary erosion on the input image using the specified kernel.

        Parameters
        ----------
        image : np.ndarray
            The input binary image to erode.
        kernel : np.ndarray
            The structuring element used for erosion.

        Returns
        -------
        np.ndarray
            The eroded binary image.
        """
        kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)
        kernel[tuple(np.array(kernel.shape)//2)] = 1

        return binary_erosion(image, structure=kernel)
    
    @staticmethod
    def binary_opening(image: np.ndarray, kernel: Optional[np.ndarray] = None, 
                        n_neighbors: Optional[int] = None) -> np.ndarray:
        """
        Perform morphological opening on the input image.

        Parameters
        ----------
        image : np.ndarray
            The input binary image to open.
        kernel : np.ndarray
            The structuring element used for opening.

        Returns
        -------
        np.ndarray
            The opened binary image.
        """
        kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)
        kernel[tuple(np.array(kernel.shape)//2)] = 1

        return binary_opening(image, structure=kernel)
    
    @staticmethod
    def binary_closing(image: np.ndarray, kernel: Optional[np.ndarray] = None, 
                        n_neighbors: Optional[int] = None) -> np.ndarray:
        """
        Perform morphological closing on the input image.

        Parameters
        ----------
        image : np.ndarray
            The input binary image to close.
        kernel : np.ndarray
            The structuring element used for closing.

        Returns
        -------
        np.ndarray
            The closed binary image.
        """
        kernel, n_neighbors = check_kernel_and_neighbors(kernel, n_neighbors)
        kernel[tuple(np.array(kernel.shape)//2)] = 1

        return binary_closing(image, structure=kernel, border_value=1)
    