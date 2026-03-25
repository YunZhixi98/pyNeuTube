# transform.pxd
# Header file for transform module to allow cross-module optimization

cimport numpy as np

# Expose the fast rotation function to other Cython modules
cpdef rotate_by_theta_psi_fast(double[:, ::1] vectors,
                              double theta,
                              double psi,
                              center=*,
                              bint inverse=*)
