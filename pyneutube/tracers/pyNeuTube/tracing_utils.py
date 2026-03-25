

import numpy as np

from .tracing_base import BaseTracingSegment
from .filters import MexicanHatFilter


_TRACE_MASK_FILTER = MexicanHatFilter(max_dist2=1.0)

def label_tracing_mask(seg: BaseTracingSegment, trace_mask: np.ndarray, dilate: bool) -> np.ndarray:
    """
    label traced voxels by dilated segment on a tracing mask
    """
    seg_dilated = seg.copy()
    if dilate:
        seg_dilated._dilate_segment()

    coords_3d, _, _ = _TRACE_MASK_FILTER(seg_dilated)
 
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

    # convert to linear indices and deduplicate
    flat_idx = np.ravel_multi_index((zs, ys, xs), dims=trace_mask.shape)
    uniq_idx = np.unique(flat_idx)

    trace_mask.flat[uniq_idx] = 1

    return trace_mask
