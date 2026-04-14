# cython: language_level=3

from libc.math cimport INFINITY

import numpy as np
cimport numpy as cnp


cdef inline void _heap_swap(
    cnp.int32_t[:] heap_nodes,
    cnp.float64_t[:] heap_dists,
    Py_ssize_t a,
    Py_ssize_t b,
) noexcept nogil:
    cdef cnp.int32_t node_tmp = heap_nodes[a]
    cdef cnp.float64_t dist_tmp = heap_dists[a]
    heap_nodes[a] = heap_nodes[b]
    heap_dists[a] = heap_dists[b]
    heap_nodes[b] = node_tmp
    heap_dists[b] = dist_tmp


cdef inline void _heap_push(
    cnp.int32_t[:] heap_nodes,
    cnp.float64_t[:] heap_dists,
    Py_ssize_t* heap_size,
    cnp.int32_t node,
    cnp.float64_t dist,
) noexcept nogil:
    cdef Py_ssize_t pos = heap_size[0]
    cdef Py_ssize_t parent

    heap_nodes[pos] = node
    heap_dists[pos] = dist
    heap_size[0] += 1

    while pos > 0:
        parent = (pos - 1) // 2
        if heap_dists[parent] <= heap_dists[pos]:
            break
        _heap_swap(heap_nodes, heap_dists, parent, pos)
        pos = parent


cdef inline void _heap_pop(
    cnp.int32_t[:] heap_nodes,
    cnp.float64_t[:] heap_dists,
    Py_ssize_t* heap_size,
    cnp.int32_t* out_node,
    cnp.float64_t* out_dist,
) noexcept nogil:
    cdef Py_ssize_t size = heap_size[0]
    cdef Py_ssize_t pos = 0
    cdef Py_ssize_t left
    cdef Py_ssize_t right
    cdef Py_ssize_t smallest

    out_node[0] = heap_nodes[0]
    out_dist[0] = heap_dists[0]

    size -= 1
    heap_size[0] = size
    if size <= 0:
        return

    heap_nodes[0] = heap_nodes[size]
    heap_dists[0] = heap_dists[size]

    while True:
        left = pos * 2 + 1
        right = left + 1
        smallest = pos

        if left < size and heap_dists[left] < heap_dists[smallest]:
            smallest = left
        if right < size and heap_dists[right] < heap_dists[smallest]:
            smallest = right
        if smallest == pos:
            break

        _heap_swap(heap_nodes, heap_dists, pos, smallest)
        pos = smallest


cpdef tuple graph_shortest_path_dijkstra_cy(
    cnp.ndarray[cnp.int32_t, ndim=2] edges,
    cnp.ndarray[cnp.float64_t, ndim=1] weights,
    int nvertex,
    int start,
    int end,
):
    cdef Py_ssize_t nedge = edges.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] predecessors
    cdef cnp.ndarray[cnp.float64_t, ndim=1] distances
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] visited
    cdef cnp.ndarray[cnp.int32_t, ndim=1] degree
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] cursor
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] data
    cdef cnp.ndarray[cnp.int32_t, ndim=1] heap_nodes
    cdef cnp.ndarray[cnp.float64_t, ndim=1] heap_dists
    cdef cnp.int32_t[:] edges_u
    cdef cnp.int32_t[:] edges_v
    cdef cnp.float64_t[:] weights_view
    cdef cnp.int32_t[:] pred_view
    cdef cnp.float64_t[:] dist_view
    cdef cnp.uint8_t[:] visited_view
    cdef cnp.int32_t[:] degree_view
    cdef cnp.int32_t[:] indptr_view
    cdef cnp.int32_t[:] cursor_view
    cdef cnp.int32_t[:] indices_view
    cdef cnp.float64_t[:] data_view
    cdef cnp.int32_t[:] heap_nodes_view
    cdef cnp.float64_t[:] heap_dists_view
    cdef Py_ssize_t i
    cdef Py_ssize_t heap_size = 0
    cdef Py_ssize_t edge_pos
    cdef cnp.int32_t u
    cdef cnp.int32_t v
    cdef cnp.int32_t cur_node
    cdef cnp.float64_t cur_dist
    cdef cnp.float64_t next_dist
    cdef Py_ssize_t adj_start
    cdef Py_ssize_t adj_end
    cdef Py_ssize_t total_slots

    predecessors = np.full(nvertex, -1, dtype=np.int32)
    distances = np.full(nvertex, np.inf, dtype=np.float64)
    if nvertex <= 0:
        return predecessors, distances

    if start < 0 or start >= nvertex or end < 0 or end >= nvertex:
        raise IndexError("start/end vertex is out of range")

    visited = np.zeros(nvertex, dtype=np.uint8)
    degree = np.zeros(nvertex, dtype=np.int32)

    pred_view = predecessors
    dist_view = distances
    visited_view = visited
    degree_view = degree

    if nedge > 0:
        edges_u = edges[:, 0]
        edges_v = edges[:, 1]
        weights_view = weights

        for i in range(nedge):
            degree_view[edges_u[i]] += 1
            degree_view[edges_v[i]] += 1

        indptr = np.empty(nvertex + 1, dtype=np.int32)
        indptr_view = indptr
        indptr_view[0] = 0
        for i in range(nvertex):
            indptr_view[i + 1] = indptr_view[i] + degree_view[i]

        total_slots = indptr_view[nvertex]
        indices = np.empty(total_slots, dtype=np.int32)
        data = np.empty(total_slots, dtype=np.float64)
        cursor = indptr[:-1].copy()
        indices_view = indices
        data_view = data
        cursor_view = cursor

        for i in range(nedge):
            u = edges_u[i]
            v = edges_v[i]

            edge_pos = cursor_view[u]
            indices_view[edge_pos] = v
            data_view[edge_pos] = weights_view[i]
            cursor_view[u] += 1

            edge_pos = cursor_view[v]
            indices_view[edge_pos] = u
            data_view[edge_pos] = weights_view[i]
            cursor_view[v] += 1
    else:
        indptr = np.zeros(nvertex + 1, dtype=np.int32)
        indptr_view = indptr
        indices = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.float64)
        indices_view = indices
        data_view = data

    heap_nodes = np.empty(max(1, 2 * nedge + 1), dtype=np.int32)
    heap_dists = np.empty(max(1, 2 * nedge + 1), dtype=np.float64)
    heap_nodes_view = heap_nodes
    heap_dists_view = heap_dists

    dist_view[start] = 0.0
    with nogil:
        _heap_push(heap_nodes_view, heap_dists_view, &heap_size, start, 0.0)

        while heap_size > 0:
            _heap_pop(heap_nodes_view, heap_dists_view, &heap_size, &cur_node, &cur_dist)
            if visited_view[cur_node]:
                continue
            if cur_dist > dist_view[cur_node]:
                continue

            visited_view[cur_node] = 1
            if cur_node == end:
                break

            adj_start = indptr_view[cur_node]
            adj_end = indptr_view[cur_node + 1]
            for i in range(adj_start, adj_end):
                v = indices_view[i]
                if visited_view[v]:
                    continue
                next_dist = cur_dist + data_view[i]
                if next_dist < dist_view[v]:
                    dist_view[v] = next_dist
                    pred_view[v] = cur_node
                    _heap_push(heap_nodes_view, heap_dists_view, &heap_size, v, next_dist)

    return predecessors, distances
