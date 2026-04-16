from typing import Callable, Union

import numpy as np
from scipy.optimize import minimize

from pyneutube.core.processing.sampling import sample_voxels

from .config import Defaults, Optimization
from .filters import MexicanHatFilter, correlation_score
from .seg_utils import set_orientation
from .tracing_base import BaseTracingSegment

_OPTIMIZATION_SEG_FILTER = MexicanHatFilter()


def optimize_segment(
    seg: BaseTracingSegment,
    image: np.ndarray,
    score_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]] = correlation_score,
    var_init=None,
):
    """Fit radius, orientation, and scale for a tracing segment."""
    if var_init is None:
        var_init = [seg.radius, seg.theta, seg.psi, seg.scale]

    var_bound = np.array(
        [
            (Defaults.MIN_SEG_RADIUS, 50.0),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (0.2, 20.0),
        ]
    )

    seg_filter = _OPTIMIZATION_SEG_FILTER
    tmpseg = seg.copy()
    init_dir = set_orientation(seg.theta, seg.psi)

    def _score_func(x: np.ndarray) -> float:
        tmpseg.radius, tmpseg.theta, tmpseg.psi, tmpseg.scale = x
        new_dir = set_orientation(tmpseg.theta, tmpseg.psi)
        dot_product = np.dot(init_dir, new_dir)
        # penalty = 1000.0 * abs(dot_product) if dot_product < 0 else 0.0  
        penalty = 0 # temporarily disable the penalty

        coords_3d, _, weights_3d = seg_filter(tmpseg)
        intensities = sample_voxels(image, coords_3d)
        score = score_func(intensities, weights_3d)

        return -score + penalty

    return minimize(
        fun=_score_func,
        x0=var_init,
        method="L-BFGS-B",
        jac="3-point",
        bounds=var_bound,
        tol=1e-3,
        options={"maxiter": Optimization.MAX_ITER},
    )
