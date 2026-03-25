# transform_simple.pyx
# Simplified version to avoid compilation issues
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Simplified transform.pyx that avoids compilation issues while maintaining performance.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, fmod, M_PI

# Type definitions
np.import_array()
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rotate_by_theta_psi_fast(double[:, ::1] vectors,
                            double theta,
                            double psi,
                            center=None,
                            bint inverse=False):
    """
    Fast rotation function using direct matrix multiplication.
    """
    cdef:
        Py_ssize_t n_points = vectors.shape[0]
        Py_ssize_t i, j
        double[:, ::1] result = np.zeros((n_points, 3), dtype=np.float64)
        double cos_theta = cos(theta)
        double sin_theta = sin(theta)
        double cos_psi = cos(psi)
        double sin_psi = sin(psi)
        double x, y, z, x_rot, y_rot, z_rot
        double cx = 0.0, cy = 0.0, cz = 0.0
        bint has_center = False
        
    # Handle center
    if center is not None:
        center_array = np.asarray(center, dtype=np.float64)
        if center_array.shape[0] == 3:
            cx = center_array[0]
            cy = center_array[1]
            cz = center_array[2]
            has_center = True
    
    # Build rotation matrix elements
    cdef double r00, r01, r02, r10, r11, r12, r20, r21, r22
    
    if not inverse:
        # Rz(psi) * Rx(theta)
        r00 = cos_psi
        r01 = -sin_psi * cos_theta
        r02 = sin_psi * sin_theta
        
        r10 = sin_psi
        r11 = cos_psi * cos_theta
        r12 = -cos_psi * sin_theta
        
        r20 = 0
        r21 = sin_theta
        r22 = cos_theta
    else:
        # Inverse: Rx(-theta) * Rz(-psi)
        r00 = cos_psi
        r01 = sin_psi
        r02 = 0
        
        r10 = -sin_psi * cos_theta
        r11 = cos_psi * cos_theta
        r12 = sin_theta
        
        r20 = sin_psi * sin_theta
        r21 = -cos_psi * sin_theta
        r22 = cos_theta
    
    # Apply rotation
    for i in range(n_points):
        if has_center:
            x = vectors[i, 0] - cx
            y = vectors[i, 1] - cy
            z = vectors[i, 2] - cz
        else:
            x = vectors[i, 0]
            y = vectors[i, 1]
            z = vectors[i, 2]
        
        # Matrix multiplication
        x_rot = r00 * x + r01 * y + r02 * z
        y_rot = r10 * x + r11 * y + r12 * z
        z_rot = r20 * x + r21 * y + r22 * z
        
        if has_center:
            result[i, 0] = x_rot + cx
            result[i, 1] = y_rot + cy
            result[i, 2] = z_rot + cz
        else:
            result[i, 0] = x_rot
            result[i, 1] = y_rot
            result[i, 2] = z_rot
    
    return np.asarray(result)

# Python-compatible wrapper
def rotate_by_theta_psi(vectors, theta, psi, center=None, inverse=False):
    """
    Python-compatible rotation function.
    
    Parameters
    ----------
    vectors : array_like
        A single 3D point (shape (3,)) or an array of points (shape (N,3)).
    theta : float
        Rotation angle about the X-axis (counter-clockwise), in radians.
    psi : float
        Rotation angle about the Z-axis (counter-clockwise), in radians.
    center : array_like, optional
        A 3-vector specifying the center of rotation.
    inverse : bool
        If True, apply inverse rotation.
        
    Returns
    -------
    rotated : ndarray
        The rotated point(s), same shape as `vectors`.
    """
    # Handle input
    input_1d = False
    vecs = np.asarray(vectors, dtype=np.float64)
    
    if vecs.ndim == 1:
        input_1d = True
        vecs = vecs.reshape(1, -1)
    
    if vecs.shape[1] != 3:
        raise ValueError("`vectors` must be shape (3,) or (N,3).")
    
    # Call optimized function
    result = rotate_by_theta_psi_fast(vecs, theta, psi, center, inverse)
    
    # Return same shape as input
    if input_1d:
        return result[0]
    return result

# Alternative scipy-based implementation for validation
def rotate_by_theta_psi_scipy(vectors, theta, psi, center=None, inverse=False):
    """Fallback implementation using scipy."""
    from scipy.spatial.transform import Rotation as R
    
    vecs = np.atleast_2d(vectors).astype(float)
    if vecs.shape[1] != 3:
        raise ValueError("`vectors` must be shape (3,) or (N,3).")
    
    if center is not None:
        c = np.asarray(center, dtype=float)
        if c.shape != (3,):
            raise ValueError("`center` must be a length-3 vector.")
        vecs = vecs - c
    
    if not inverse:
        rot_x = R.from_euler('x', theta)
        rot_z = R.from_euler('z', psi)
        rot = rot_z * rot_x
    else:
        rot_x = R.from_euler('x', -theta)
        rot_z = R.from_euler('z', -psi)
        rot = rot_x * rot_z
    
    rotated = rot.apply(vecs)
    
    if center is not None:
        rotated = rotated + c
    
    if np.asarray(vectors).ndim == 1:
        return rotated[0]
    return rotated


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple normalize_euler_zx(double theta, double psi):
    """Cython implementation of normalize_euler_zx."""
    cdef double twopi = 2 * M_PI

    # Wrap angles into [0, 2π)
    theta = fmod(theta, twopi)
    if theta < 0.0:
        theta += twopi

    psi = fmod(psi, twopi)
    if psi < 0.0:
        psi += twopi

    # If theta is in (π, 2π), reflect it back
    if theta > M_PI:
        theta = twopi - theta
        psi += M_PI
        psi = fmod(psi, twopi)
        if psi < 0.0:
            psi += twopi

    return theta, psi
