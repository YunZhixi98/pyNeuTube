# core/constants.py

"""
Constants for image processing neighbors/connectivity and kernels.
"""

from typing import Optional, Union, Tuple, List
from itertools import product
import warnings

import numpy as np

__all__ = [
    "neighbors_18",
    "neighbors_26",
    "neighbors_26_forward",
    "kernel_18",
    "kernel_26",
]


neighbors_18 = np.array([
    [ 1,  0,  0], [-1,  0,  0],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
    [ 1,  1,  0], [ 1, -1,  0],
    [-1,  1,  0], [-1, -1,  0],
    [ 1,  0,  1], [ 1,  0, -1],
    [-1,  0,  1], [-1,  0, -1],
    [ 0,  1,  1], [ 0,  1, -1],
    [ 0, -1,  1], [ 0, -1, -1],
], dtype=np.intc)

neighbors_26 = np.array([
    [ 1,  0,  0], [-1,  0,  0],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
    [ 1,  1,  0], [ 1, -1,  0],
    [-1,  1,  0], [-1, -1,  0],
    [ 1,  0,  1], [ 1,  0, -1],
    [-1,  0,  1], [-1,  0, -1],
    [ 0,  1,  1], [ 0,  1, -1],
    [ 0, -1,  1], [ 0, -1, -1],
    [ 1,  1,  1], [ 1,  1, -1],
    [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1],
    [-1, -1,  1], [-1, -1, -1],
], dtype=np.intc)

neighbors_26_forward = np.array([
    (0, 0,  1),  # x+ 
    (0, 1,  0),  # y+
    (1, 0,  0),  # z+
    (0, 1,  1),  # x+ y+
    (0, 1, -1),  # x- y+
    (1, 0,  1),  # x+ z+
    (1, 0, -1),  # x- z+
    (1, 1,  0),  # y+ z+
    (1, -1, 0),  # y- z+
    (1, 1,  1),  # x+ y+ z+
    (1, 1, -1),  # x- y+ z+
    (1, -1, 1),  # x+ y- z+
    (1, -1, -1)  # x- y- z+
], dtype=np.intc)  # 13 forward neighbors

# Kernel for 18 neighbors (3x3x3, center=0, neighbors=1)
kernel_18 = np.zeros((3, 3, 3), dtype=np.intc)
for offset in neighbors_18:
    kernel_18[offset[0]+1, offset[1]+1, offset[2]+1] = 1

# Kernel for 26 neighbors (3x3x3, center=0, neighbors=1)
kernel_26 = np.ones((3, 3, 3), dtype=np.intc)
kernel_26[1, 1, 1] = 0


def get_boundary_indices(image_shape: Union[np.ndarray, List, Tuple]):
    """
    Returns a 1D array of linear indices of all boundary voxels of a 3D volume,
    grouped (in order) as:
    1) corners
    2) edges (excluding corners)
    3) faces (excluding edges and corners)
    and within each group sorted by (z,y,x) lex order.

    Parameters
    ----------
    image_shape: Union[np.ndarray, list, tuple]
        Image shape in zyx-order.

    Returns
    -------
    boundary_idx : np.ndarray, dtype=int
        Flat indices into an array of size D*H*W, in the grouping/order described.
    """
    D, H, W = image_shape
    if not (D >= 2 and H >= 2 and W >= 2):
        raise ValueError("All dimensions must be at least 2 to have a boundary.")

    # helper to convert (z,y,x) to flat index
    def idx(z,y,x):
        return z*(H*W) + y*W + x

    corners = []
    for z, y, x in product((0, D-1), (0, H-1), (0, W-1)):
        corners.append((z,y,x))

    # Edges: exactly two coords on {0, dim-1}, one in (1..dim-2)
    edges = []
    for (fixed_axes, var_axis) in (
        ((0,1), 2),  # x varies
        ((0,2), 1),  # y varies
        ((1,2), 0),  # z varies
    ):
        tmpedges = []
        # fixed_axes are the axes idxs held at boundary
        for fb in product((0,1), repeat=2):
            # fb picks for each fixed axis 0→min, 1→max
            fixed = {}
            fixed[fixed_axes[0]] = fb[0]
            fixed[fixed_axes[1]] = fb[1]
            for v in range(1, image_shape[var_axis]-1):
                coord = [None, None, None]
                coord[ var_axis ] = v
                coord[ fixed_axes[0] ] = (image_shape[ fixed_axes[0] ]-1 if fb[0] else 0)
                coord[ fixed_axes[1] ] = (image_shape[ fixed_axes[1] ]-1 if fb[1] else 0)
                tmpedges.append(tuple(coord))
        tmpedges.sort()
        edges.extend(tmpedges)

    # Faces: exactly one coord on {0, dim-1}, two coords in (1..dim-2)
    faces = []
    for face_axis in (0,1,2):
        tmpfaces = []
        for face_bound in (0, image_shape[face_axis]-1):
            # the two remaining axes vary internally
            axes = [0,1,2]
            axes.remove(face_axis)
            a0,a1 = axes
            for v0 in range(1, image_shape[a0]-1):
                for v1 in range(1, image_shape[a1]-1):
                    coord = [None, None, None]
                    coord[face_axis] = face_bound
                    coord[a0] = v0
                    coord[a1] = v1
                    tmpfaces.append(tuple(coord))
        tmpfaces.sort()
        faces.extend(tmpfaces)

    # # flatten to indices
    # all_idx = [idx(*c) for group in (corners, edges, faces) for c in group]
    # return np.array(all_idx, dtype=int)

    return np.array(corners+edges+faces, dtype=np.intc)


def get_kernel(n_neighbors: int) -> np.ndarray:
    """
    Get the kernel for the specified number of neighbors.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors (18 or 26).

    Returns
    -------
    np.ndarray
        The corresponding kernel.
    """
    if n_neighbors == 18:
        return kernel_18
    elif n_neighbors == 26:
        return kernel_26
    else:
        raise ValueError("n_neighbors must be 18 or 26.")
    

def check_kernel_and_neighbors(kernel: Optional[np.ndarray], n_neighbors: Optional[int]) -> tuple[Optional[np.ndarray], int]:
    """
    Ensure that kernel and n_neighbors are set correctly.
    """
    if kernel is None and n_neighbors is None:
        raise ValueError("Both kernel and n_neighbors cannot be None at the same time.")
    
    if kernel is not None:
        if n_neighbors is not None:
            warnings.warn("Both kernel and n_neighbors are provided. Using kernel will override n_neighbors.")
        n_neighbors = int(np.abs(kernel).sum())
        if kernel[tuple(np.array(kernel.shape)//2)]!=0:
            n_neighbors -= 1  # Exclude the center voxel
    elif n_neighbors is not None:
        kernel = get_kernel(n_neighbors)
    else:
        raise ValueError("Either kernel or n_neighbors must be provided.")
        
    return kernel, n_neighbors
