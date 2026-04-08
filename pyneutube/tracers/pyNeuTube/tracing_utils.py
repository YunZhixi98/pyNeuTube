

import numpy as np

from pyneutube.core.processing.transform import rotate_by_theta_psi_fast

from .tracing_base import BaseTracingSegment


def _dense_segment_coords(seg: BaseTracingSegment) -> np.ndarray:
    radius_y = max(float(seg.radius), 1e-8)
    radius_x = max(float(seg.radius * seg.scale), 1e-8)
    z_max = max(float(seg.length - 1.0), 0.0)
    radius_x_voxels = int(np.ceil(radius_x))
    radius_y_voxels = int(np.ceil(radius_y))
    z_voxels = int(np.floor(z_max))

    x_coords = np.arange(-radius_x_voxels, radius_x_voxels + 1, dtype=np.int16)
    y_coords = np.arange(-radius_y_voxels, radius_y_voxels + 1, dtype=np.int16)
    z_coords = np.arange(0, z_voxels + 1, dtype=np.int16)

    xy_grid = np.stack(np.meshgrid(x_coords, y_coords, indexing="xy"), axis=-1).reshape(-1, 2)
    xy_mask = (xy_grid[:, 0] / radius_x) ** 2 + (xy_grid[:, 1] / radius_y) ** 2 <= 1.0
    xy_points = xy_grid[xy_mask]
    if xy_points.size == 0 or z_coords.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    local_coords = np.empty((xy_points.shape[0] * z_coords.size, 3), dtype=np.float64)
    block = xy_points.shape[0]
    xy_points = xy_points.astype(np.float64, copy=False)
    for index, z_value in enumerate(z_coords):
        start = index * block
        stop = start + block
        local_coords[start:stop, :2] = xy_points
        local_coords[start:stop, 2] = float(z_value)

    if seg.theta != 0 or seg.psi != 0:
        local_coords = rotate_by_theta_psi_fast(local_coords, seg.theta, seg.psi, None)

    local_coords += np.asarray(seg.start_coord, dtype=np.float64)
    return local_coords

def label_tracing_mask(seg: BaseTracingSegment, trace_mask: np.ndarray, dilate: bool) -> np.ndarray:
    """
    label traced voxels by dilated segment on a tracing mask
    """
    seg_dilated = seg.copy()
    if dilate:
        seg_dilated._dilate_segment()

    coords_3d = _dense_segment_coords(seg_dilated)
    if coords_3d.size == 0:
        return trace_mask
 
    coords_int = np.rint(coords_3d).astype(np.intp)  # use np.intp for indexing
    xs = coords_int[:, 0]
    ys = coords_int[:, 1]
    zs = coords_int[:, 2]

    sz, sy, sx = trace_mask.shape
    valid = (xs >= 0) & (xs < sx) & (ys >= 0) & (ys < sy) & (zs >= 0) & (zs < sz)
    if not np.any(valid):
        return trace_mask

    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]

    trace_mask[zs, ys, xs] = 1

    return trace_mask
