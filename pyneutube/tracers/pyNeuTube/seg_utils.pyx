# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

from pyneutube.core.processing.transform cimport rotate_by_theta_psi_fast

np.import_array()
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def set_orientation(double theta, double psi):
    """
    Set orientation in-place (modifies the provided direction vector).
    
    Parameters
    ----------
    theta : float
        Rotation angle theta (radians)
    psi : float
        Rotation angle psi (radians)
    dir_v : array, shape (3,)
        Direction vector to modify in-place
    """
    cdef int i
    cdef double[:, ::1] dir_v = np.array([[0,0,1]], dtype=np.float64)

    # Rotate and normalize
    dir_v = rotate_by_theta_psi_fast(dir_v, theta, psi)
    
    # Normalize
    cdef double norm = 0.0
    for i in range(3):
        norm += dir_v[0,i] * dir_v[0,i]
    norm = sqrt(norm)
    
    if norm > 0.0:
        for i in range(3):
            dir_v[0,i] /= norm

    return np.asarray(dir_v[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def set_coordinates(np.ndarray[DTYPE_t, ndim=1] coord_np,
                   np.ndarray[DTYPE_t, ndim=1] dir_v_np,
                   double length,
                   str alignment):
    """
    Fast Cython helper returning (start, center, end) as new NumPy arrays,
    implemented with simple np.empty + memoryviews.
    """
    cdef:
        DTYPE_t[::1] coord = np.ascontiguousarray(coord_np, dtype=np.float64)
        DTYPE_t[::1] dir_v  = np.ascontiguousarray(dir_v_np, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] start = np.empty(3, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] center = np.empty(3, dtype=np.float64)
        np.ndarray[DTYPE_t, ndim=1] end = np.empty(3, dtype=np.float64)
        DTYPE_t[::1] s = start
        DTYPE_t[::1] c = center
        DTYPE_t[::1] e = end
        double length_1 = length - 1.0
        double half = length_1 * 0.5
        int i

    if alignment == 'center':
        for i in range(3):
            s[i] = coord[i] - half  * dir_v[i]
            c[i] = coord[i]
            e[i] = coord[i] + half  * dir_v[i]

    elif alignment == 'start':
        for i in range(3):
            s[i] = coord[i]
            c[i] = coord[i] + half  * dir_v[i]
            e[i] = coord[i] + length_1 * dir_v[i]

    elif alignment == 'end':
        for i in range(3):
            s[i] = coord[i] - length_1 * dir_v[i]
            c[i] = coord[i] - half  * dir_v[i]
            e[i] = coord[i]

    else:
        raise ValueError("alignment must be 'center', 'start', or 'end'")

    return start, center, end