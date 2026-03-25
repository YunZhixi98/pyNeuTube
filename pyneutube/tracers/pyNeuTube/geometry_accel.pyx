# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from math import floor

import numpy as np
cimport cython
cimport numpy as cnp
from libc.math cimport sqrt

from pyneutube.core.processing.transform import rotate_by_theta_psi


ctypedef cnp.float64_t DTYPE_f64

cnp.import_array()


cdef inline double _norm3(double x, double y, double z) noexcept nogil:
    return sqrt(x * x + y * y + z * z)


@cython.boundscheck(False)
@cython.wraparound(False)
def point_to_seg_surface(coord, seg2):
    cdef double radius = float(seg2.radius)
    cdef double scale = float(seg2.scale)
    cdef double seg_length = float(seg2.length)
    cdef double top_z = seg_length - 1.0
    cdef cnp.ndarray[DTYPE_f64, ndim=1] coord_arr = np.asarray(coord, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] coord_local
    cdef double x
    cdef double y
    cdef double z
    cdef double d2
    cdef double nx
    cdef double ny
    cdef double norm_r
    cdef double z_s
    cdef cnp.ndarray[DTYPE_f64, ndim=1] intersection_point
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double dist

    coord_local = np.asarray(coord_arr - seg2.start_coord, dtype=np.float64)
    coord_local = np.asarray(
        rotate_by_theta_psi(coord_local, seg2.theta, seg2.psi, inverse=True),
        dtype=np.float64,
    )

    x = coord_local[0]
    y = coord_local[1]
    z = coord_local[2]

    if z >= -0.5 and z <= seg_length:
        d2 = (x / scale) ** 2 + y ** 2
        if d2 <= radius ** 2:
            return 0.0, coord_arr

    nx = x / (radius * scale)
    ny = y / radius
    norm_r = sqrt(nx * nx + ny * ny)

    if z < 0.0:
        z_s = 0.0
    elif z > top_z:
        z_s = top_z
    else:
        z_s = z

    if norm_r <= 1.0:
        intersection_point = np.array([x, y, z_s], dtype=np.float64)
    else:
        intersection_point = np.array([x / norm_r, y / norm_r, z_s], dtype=np.float64)

    dx = coord_local[0] - intersection_point[0]
    dy = coord_local[1] - intersection_point[1]
    dz = coord_local[2] - intersection_point[2]
    dist = _norm3(dx, dy, dz)

    intersection_point = np.asarray(
        rotate_by_theta_psi(intersection_point, seg2.theta, seg2.psi),
        dtype=np.float64,
    ) + seg2.start_coord

    return dist, intersection_point


@cython.boundscheck(False)
@cython.wraparound(False)
def seg_to_seg_surface(seg1, seg2, int discrete_step=1):
    cdef int max_step = int(floor(seg1.length))
    cdef cnp.ndarray[DTYPE_f64, ndim=1] start_coord = np.asarray(seg1.start_coord, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] dir_v = np.asarray(seg1.dir_v, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] coord = np.empty(3, dtype=np.float64)
    cdef double min_dist = np.inf
    cdef object intersection_point = None
    cdef double dist
    cdef object tmp_point
    cdef int step

    for step in range(0, max_step, discrete_step):
        coord[0] = start_coord[0] + step * dir_v[0]
        coord[1] = start_coord[1] + step * dir_v[1]
        coord[2] = start_coord[2] + step * dir_v[2]
        dist, tmp_point = point_to_seg_surface(coord, seg2)
        if dist < min_dist:
            min_dist = dist
            intersection_point = tmp_point
            if min_dist == 0.0:
                break

    return min_dist, intersection_point


@cython.boundscheck(False)
@cython.wraparound(False)
def segment_segment_distance(P0, P1, Q0, Q1, double eps=1e-12):
    cdef cnp.ndarray[DTYPE_f64, ndim=1] p0 = np.asarray(P0, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] p1 = np.asarray(P1, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] q0 = np.asarray(Q0, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] q1 = np.asarray(Q1, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64, ndim=1] u = p1 - p0
    cdef cnp.ndarray[DTYPE_f64, ndim=1] v = q1 - q0
    cdef cnp.ndarray[DTYPE_f64, ndim=1] w = p0 - q0
    cdef double a = float(np.dot(u, u))
    cdef double b = float(np.dot(u, v))
    cdef double c = float(np.dot(v, v))
    cdef double d = float(np.dot(u, w))
    cdef double e = float(np.dot(v, w))
    cdef double den = a * c - b * b
    cdef double s
    cdef double t
    cdef cnp.ndarray[DTYPE_f64, ndim=1] cpP
    cdef cnp.ndarray[DTYPE_f64, ndim=1] cpQ
    cdef list candidates
    cdef tuple best

    def point_to_segment_closest(P, A, B):
        cdef cnp.ndarray[DTYPE_f64, ndim=1] p = np.asarray(P, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_f64, ndim=1] a_local = np.asarray(A, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_f64, ndim=1] b_local = np.asarray(B, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_f64, ndim=1] ab = b_local - a_local
        cdef double ab_len2 = float(np.dot(ab, ab))
        cdef double t_local
        cdef cnp.ndarray[DTYPE_f64, ndim=1] cp
        if ab_len2 == 0.0:
            cp = a_local.copy()
            return float(np.linalg.norm(p - cp)), cp
        t_local = float(np.dot(p - a_local, ab) / ab_len2)
        t_local = np.clip(t_local, 0.0, 1.0)
        cp = a_local + t_local * ab
        return float(np.linalg.norm(p - cp)), cp

    if den > eps:
        s = (b * e - c * d) / den
        t = (a * e - b * d) / den
    else:
        s = 0.0
        t = (e / c) if c > eps else 0.0

    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        cpP = p0 + s * u
        cpQ = q0 + t * v
        return float(np.linalg.norm(cpP - cpQ)), cpP, cpQ

    candidates = []
    dist0, cp0 = point_to_segment_closest(p0, q0, q1)
    candidates.append((dist0, p0.copy(), cp0))
    dist1, cp1 = point_to_segment_closest(p1, q0, q1)
    candidates.append((dist1, p1.copy(), cp1))
    dist2, cp2 = point_to_segment_closest(q0, p0, p1)
    candidates.append((dist2, cp2, q0.copy()))
    dist3, cp3 = point_to_segment_closest(q1, p0, p1)
    candidates.append((dist3, cp3, q1.copy()))

    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    return float(best[0]), best[1], best[2]


@cython.boundscheck(False)
@cython.wraparound(False)
def seg_to_seg_dist(seg1, seg2):
    cdef object mindist
    mindist, _, _ = segment_segment_distance(
        seg1.start_coord,
        seg1.end_coord,
        seg2.start_coord,
        seg2.end_coord,
    )
    return mindist


@cython.boundscheck(False)
@cython.wraparound(False)
def seg_chain_dist_upper_bound(chain, seg):
    cdef double min_dist = np.inf
    cdef cnp.ndarray[DTYPE_f64, ndim=1] seg_center_coord = np.asarray(seg.center_coord, dtype=np.float64)
    cdef object tmpseg
    cdef cnp.ndarray[DTYPE_f64, ndim=1] center_coord
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double dist

    for tmpseg in chain._segments:
        center_coord = np.asarray(tmpseg.center_coord, dtype=np.float64)
        dx = center_coord[0] - seg_center_coord[0]
        dy = center_coord[1] - seg_center_coord[1]
        dz = center_coord[2] - seg_center_coord[2]
        dist = _norm3(dx, dy, dz)
        if dist < min_dist:
            min_dist = dist

    return min_dist


@cython.boundscheck(False)
@cython.wraparound(False)
def point_to_chain_surface(point, chain):
    cdef double min_dist = np.inf
    cdef object intersection_point = None
    cdef int min_idx = 0
    cdef int i
    cdef object seg
    cdef double dist
    cdef object tmp_point

    for i, seg in enumerate(chain._segments):
        dist, tmp_point = point_to_seg_surface(point, seg)
        if dist < min_dist:
            min_dist = dist
            intersection_point = tmp_point
            min_idx = i
            if min_dist == 0.0:
                break

    return min_dist, intersection_point, min_idx
