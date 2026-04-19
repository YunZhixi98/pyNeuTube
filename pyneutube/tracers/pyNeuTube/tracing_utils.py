

import numpy as np

from pyneutube.core.processing.transform import rotate_by_theta_psi_fast

from .tracing_base import BaseTracingSegment


def _dense_segment_coords(seg: BaseTracingSegment) -> np.ndarray:
    radius_y = max(float(seg.radius), 1e-8)
    radius_x = max(float(seg.radius * seg.scale), 1e-8)
    z_max = max(float(seg.length - 1.0), 0.0)
    local_corners = np.array(
        [
            [x, y, z]
            for x in (-radius_x, radius_x)
            for y in (-radius_y, radius_y)
            for z in (0.0, z_max)
        ],
        dtype=np.float64,
    )
    if seg.theta != 0 or seg.psi != 0:
        world_corners = rotate_by_theta_psi_fast(local_corners, seg.theta, seg.psi, None)
    else:
        world_corners = local_corners
    world_corners += np.asarray(seg.start_coord, dtype=np.float64)

    bbox_min = np.floor(world_corners.min(axis=0)).astype(np.intp)
    bbox_max = np.ceil(world_corners.max(axis=0)).astype(np.intp)

    x_coords = np.arange(bbox_min[0], bbox_max[0] + 1, dtype=np.intp)
    y_coords = np.arange(bbox_min[1], bbox_max[1] + 1, dtype=np.intp)
    z_coords = np.arange(bbox_min[2], bbox_max[2] + 1, dtype=np.intp)
    if x_coords.size == 0 or y_coords.size == 0 or z_coords.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    world_coords = np.stack(
        np.meshgrid(x_coords, y_coords, z_coords, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.float64, copy=False)

    rel_world_coords = world_coords - np.asarray(seg.start_coord, dtype=np.float64)
    if seg.theta != 0 or seg.psi != 0:
        local_coords = rotate_by_theta_psi_fast(
            rel_world_coords,
            seg.theta,
            seg.psi,
            None,
            inverse=True,
        )
    else:
        local_coords = rel_world_coords

    inside = (
        (local_coords[:, 2] >= 0.0)
        & (local_coords[:, 2] <= z_max)
        & ((local_coords[:, 0] / radius_x) ** 2 + (local_coords[:, 1] / radius_y) ** 2 <= 1.0)
    )
    if not np.any(inside):
        return np.empty((0, 3), dtype=np.float64)

    return world_coords[inside]

def _write_label_coords(
    coords_int: np.ndarray,
    trace_mask: np.ndarray,
):
    if coords_int.size == 0:
        return trace_mask, None

    xs = coords_int[:, 0]
    ys = coords_int[:, 1]
    zs = coords_int[:, 2]

    sz, sy, sx = trace_mask.shape
    valid = (xs >= 0) & (xs < sx) & (ys >= 0) & (ys < sy) & (zs >= 0) & (zs < sz)
    if not np.any(valid):
        return trace_mask, None

    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]

    trace_mask[zs, ys, xs] = 1

    bbox = np.array(
        [xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()],
        dtype=np.intp,
    )
    return trace_mask, bbox


def label_tracing_mask(
    seg,
    trace_mask: np.ndarray,
    dilate: bool,
    *,
    start: int | None = None,
    end: int | None = None,
):
    """
    label traced voxels by dilated segment on a tracing mask
    """
    if hasattr(seg, "_segments") and hasattr(seg, "label_bboxes"):
        if start is None:
            start = 0
        if end is None:
            end = len(seg) - 1

        if seg.label_bboxes is None or seg.label_bboxes.shape != (len(seg), 6):
            seg.label_bboxes = np.full((len(seg), 6), -1, dtype=np.intp)
        if start > end or len(seg) == 0:
            return trace_mask

        for idx in range(start, end + 1):
            seg_dilated = seg[idx].copy()
            if dilate:
                seg_dilated._dilate_segment()
            coords_3d = _dense_segment_coords(seg_dilated)
            if coords_3d.size == 0:
                seg.label_bboxes[idx] = -1
                continue

            coords_int = np.rint(coords_3d).astype(np.intp)
            trace_mask, bbox = _write_label_coords(coords_int, trace_mask)
            if bbox is None:
                seg.label_bboxes[idx] = -1
            else:
                seg.label_bboxes[idx] = bbox

        return trace_mask

    if start is not None or end is not None:
        raise ValueError("Segment subrange is only supported for SegmentChain inputs.")

    seg_dilated = seg.copy()
    if dilate:
        seg_dilated._dilate_segment()

    coords_3d = _dense_segment_coords(seg_dilated)
    if coords_3d.size == 0:
        return trace_mask

    coords_int = np.rint(coords_3d).astype(np.intp)
    trace_mask, _ = _write_label_coords(coords_int, trace_mask)
    return trace_mask
