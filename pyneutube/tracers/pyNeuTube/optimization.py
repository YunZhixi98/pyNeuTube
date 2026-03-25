# tracers/pyNeuTube/optimization.py

"""
optimization.py

Definition of optimization functions for the pyNeuTube tracing framework.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from pyneutube.core.processing.sampling import sample_voxels

from .config import Defaults, Optimization
from .filters import MexicanHatFilter, correlation_score, mean_intensity_score
from .seg_utils import set_orientation
from .tracing_base import BaseTracingSegment

_OPTIMIZATION_SEG_FILTER = MexicanHatFilter()


def optimize_segment(seg: BaseTracingSegment, image: np.ndarray,
                    score_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]] = correlation_score,
                    var_init = None):
    """
        
    Variable bounds in C-script source code
    NAME = {"r1", "cone coefficient", "theta", "psi", "height", "curvature", "alpha", "ellipse scale", "x", "y", "z", "?"}
    MIN = {0.5, -4.0, -Infinity, -Infinity, 2.0, 0.0, -Infinity, 0.2,	-Infinity, -Infinity, -Infinity, 0.5}
    MAX = {50.0, 4.0, Infinity, Infinity, 30.0, NEUROSEG_MAX_CURVATURE, Infinity, 20.0,	Infinity, Infinity, Infinity, 6.0}
    """
    if var_init is None:
        var_init = [seg.radius, seg.theta, seg.psi, seg.scale]
    
    var_bound = np.array([(Defaults.MIN_SEG_RADIUS, 50.0), (-np.inf, np.inf), (-np.inf, np.inf), (0.2, 20)])
    delta = np.array([Optimization.DELTA_RADIUS, Optimization.DELTA_THETA, 
                      Optimization.DELTA_PSI, Optimization.DELTA_SCALE])

    seg_filter = _OPTIMIZATION_SEG_FILTER
    tmpseg = seg.copy()

    init_dir = set_orientation(seg.theta, seg.psi)

    def _score_func(x: np.ndarray,) -> float:
        
        # Hard clipping
        # x = np.clip(x, var_bound[:, 0], var_bound[:, 1])

        tmpseg.radius, tmpseg.theta, tmpseg.psi, tmpseg.scale = x
        new_dir = set_orientation(tmpseg.theta, tmpseg.psi)
        dot_product = np.dot(init_dir, new_dir)
        if dot_product < 0:
            # 添加一个大的惩罚项，确保优化器避免这个方向
            penalty = 1000.0 * abs(dot_product)
        else:
            penalty = 0.0

        coords_3d, _, weights_3d = seg_filter(tmpseg)
        # tmpseg._set_orientation()
        # tmpseg._set_coordinate(tmpseg.center_coord, 'center')
        intensities = sample_voxels(image, coords_3d)
        score = score_func(intensities, weights_3d)

        return -score + penalty
        
    def grad_fn(x):
        """Finite-difference gradient with sign-correction like perceptor_gradient_partial"""
        grad = np.zeros_like(x, dtype=np.float64)
        score0 = _score_func(x)

        var = x.copy()

        for i in range(len(x)):
            step = delta[i]
            # forward
            var[i] += step
            rscore = _score_func(var)
            var[i] -= step  # restore
            g = (rscore - score0) / step
            if g < 0:
                # try backward
                var[i] -= step
                lscore = _score_func(var)
                if lscore < score0:
                    g = 0.0
                else:
                    g = (score0 - lscore) / step
                var[i] += step  # restore
            elif g > 0:
                var[i] -= step
                lscore = _score_func(var)
                if lscore > score0:
                    g = (score0 - lscore) / step
                var[i] += step
            else:
                var[i] -= step
                lscore = _score_func(var)
                g = (score0 - lscore) / step
                var[i] += step
            grad[i] = g

        return grad

    opt_res = minimize(fun=_score_func, x0=var_init, 
                        method='L-BFGS-B', 
                        jac='3-point',
                        # jac = grad_fn, 
                        bounds=var_bound,
                        tol=1e-3, options={'maxiter':Optimization.MAX_ITER})
    
    return opt_res


class SegmentOptimizer:
    """
    Tidy, class-based wrapper reproducing the C-style optimization code as provided.
    This class does not change the algorithmic logic — it only organizes the functions
    into methods and keeps the same behavior/flow.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        score_func: Callable[[np.ndarray, np.ndarray], float] = correlation_score,
        maxiter: int = Optimization.MAX_ITER,
        min_gradient: float = 1e-3,
        min_direction: float = 1e-3,
        alpha0: float = 0.2,
        ro: float = 0.8,
        c1: float = 0.01,
        weight: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        self.image = np.asarray(image)

        self.score_func = score_func
        self.maxiter = int(maxiter)
        self.min_gradient = min_gradient
        self.min_direction = min_direction
        self.alpha0 = alpha0
        self.ro = ro
        self.c1 = c1
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        self.verbose = verbose

        # variable defaults used by the original script
        self.var_bound = np.array(
            [
                (Defaults.MIN_SEG_RADIUS, 50.0),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (0.2, 20.0),
            ],
            dtype=np.float64,
        )
        self.delta = np.array(
            [
                Optimization.DELTA_RADIUS,
                Optimization.DELTA_THETA,
                Optimization.DELTA_PSI,
                Optimization.DELTA_SCALE,
            ],
            dtype=np.float64,
        )

        # Default segment filter instance (same as original)
        self.seg_filter = _OPTIMIZATION_SEG_FILTER

        # stop gradient constant used in line search
        self.stop_grad = Optimization.LINE_SEARCH_STOP_GRADIENT

    @staticmethod
    def _apply_x_to_seg(seg, x: np.ndarray) -> None:
        """Map optimization vector x -> seg attributes (order: radius, theta, psi, scale)."""
        seg.radius = x[0]
        seg.theta = x[1]
        seg.psi = x[2]
        seg.scale = x[3]

    def _score_at(self, x_local: np.ndarray, tmpseg) -> float:
        """Evaluate score for a given parameter vector x_local using the provided tmpseg."""
        self._apply_x_to_seg(tmpseg, x_local)
        coords3d, _, weights = self.seg_filter(tmpseg)
        intensities = sample_voxels(self.image, coords3d)
        return self.score_func(intensities, weights)

    def compute_fd_gradient(
        self, x: np.ndarray, tmpseg: BaseTracingSegment,
        score_at: Optional[Callable[[np.ndarray, BaseTracingSegment], float]] = None
    ) -> np.ndarray:
        """
        Finite-difference gradient emulating perceptor_gradient_partial.
        Forward step -> if sign unexpected, try backward and adjust exactly as in C code.
        `x` is expected to be mutated in-place inside but restored on exit (caller passes a copy).
        """
        if score_at is None:
            score_at = self._score_at

        n = int(x.size)
        g = np.zeros_like(x, dtype=np.float64)

        delta = self.delta
        base_score = score_at(x, tmpseg)  # score at x

        # Loop over variables, performing forward/backward checks as in C.
        for i in range(n):
            step = delta[i]

            # forward step
            x[i] += step
            rscore = score_at(x, tmpseg)
            x[i] -= step  # restore

            grad_i = (rscore - base_score) / step

            if grad_i < 0.0:
                # try left (backward)
                x[i] -= step
                lscore = score_at(x, tmpseg)
                if lscore < base_score:
                    grad_i = 0.0
                else:
                    grad_i = (base_score - lscore) / step
                x[i] += step  # restore
            elif grad_i > 0.0:
                x[i] -= step
                lscore = score_at(x, tmpseg)
                if lscore > base_score:
                    grad_i = (base_score - lscore) / step
                x[i] += step  # restore
            else:
                x[i] -= step
                lscore = score_at(x, tmpseg)
                grad_i = (base_score - lscore) / step
                x[i] += step  # restore

            g[i] = grad_i

        return g

    def line_search_var_backtrack(
        self,
        x_org: np.ndarray,
        direction: np.ndarray,
        start_score: float,
        start_grad: np.ndarray,
        tmpseg,
        *,
        var_bound: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
        alpha0: Optional[float] = None,
        ro: Optional[float] = None,
        c1: Optional[float] = None,
        min_direction: Optional[float] = None,
        stop_grad: Optional[float] = None,
        score_at: Optional[Callable[[np.ndarray, BaseTracingSegment], float]] = None,
    ) -> Tuple[bool, np.ndarray, float]:
        """
        Emulates Line_Search_Var_Backtrack from the C implementation.

        Returns: (improved, x_new, score_new)
        """

        if score_at is None:
            score_at = self._score_at

        # use provided values or fall back to self
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        ro = self.ro if ro is None else ro
        c1 = self.c1 if c1 is None else c1
        min_direction = self.min_direction if min_direction is None else min_direction
        stop_grad = self.stop_grad if stop_grad is None else stop_grad

        dir_vec = direction * weight if weight is not None else direction.copy()

        dir_len = np.linalg.norm(dir_vec)
        if dir_len <= min_direction:
            return False, x_org.copy(), start_score

        alpha = alpha0 / dir_len
        gd_dot = np.dot(start_grad, dir_vec)
        gd_dot_c1 = gd_dot * c1

        if var_bound is not None:
            low = var_bound[:, 0]
            high = var_bound[:, 1]

        score_at = self._score_at
        org_x = x_org.copy()

        x_trial = np.empty_like(org_x)
        scaled_dir = np.empty_like(dir_vec)

        while True:
            np.multiply(dir_vec, alpha, out=scaled_dir)
            np.add(org_x, scaled_dir, out=x_trial)

            # enforce variable bounds if provided (clip)
            if var_bound is not None:
                np.clip(x_trial, low, high, out=x_trial)

            # evaluate the trial point
            score_trial = score_at(x_trial, tmpseg)

            # stop if step becomes too small
            if alpha * dir_len < stop_grad:
                # revert changes to tmpseg and bail out
                self._apply_x_to_seg(tmpseg, org_x)
                return False, org_x.copy(), start_score

            wolfe1 = max(alpha / ro * gd_dot_c1, 0.0)
            # Accept if score_trial >= start_score + wolfe1 (Armijo-like)
            if score_trial >= start_score + wolfe1:
                return True, x_trial.copy(), score_trial

            # shrink and continue
            alpha *= ro

    def fit(self, seg: BaseTracingSegment) -> None:
        """
        Run the C-like conjugate gradient fitting loop and update `self.seg` with the
        resulting parameters. This method preserves the original logic and stopping
        conditions.
        """
        var_bound = self.var_bound
        delta = self.delta
        score_at = self._score_at

        x = np.array([seg.radius, seg.theta, seg.psi, seg.scale], dtype=np.float64)

        if self.weight is None:
            # fallback weight as in original script: weight = delta / ||delta||
            self.weight = delta / np.linalg.norm(delta)

        tmpseg = seg.copy()  # working copy mutated during searches

        # initial gradient and direction
        g0 = self.compute_fd_gradient(x, tmpseg, score_at)
        update_direction = g0.copy()
        final_score = score_at(x, tmpseg)

        iter_count = 0

        # main optimization loop
        while True:
            improved = False

            dir_len = np.linalg.norm(update_direction)
            if dir_len < self.min_direction:
                # too small to make progress via line-search
                improved = False
            else:
                improved, x_new, new_score = self.line_search_var_backtrack(
                    x,
                    update_direction,
                    final_score,
                    g0,
                    tmpseg,
                    var_bound=var_bound,
                    weight=self.weight,
                    alpha0=self.alpha0,
                    ro=self.ro,
                    c1=self.c1,
                    min_direction=self.min_direction,
                    stop_grad=self.stop_grad,
                    score_at=score_at,
                )
                if improved:
                    x = x_new
                    final_score = new_score
                    if self.verbose:
                        print(f"iter {iter_count + 1}: score {final_score:.6g}")

            if not improved:
                # fallback: try using gradient (g0) as direction
                gnorm = np.linalg.norm(g0)
                if gnorm > self.min_gradient:
                    update_direction = g0.copy()
                    improved, x_new, new_score = self.line_search_var_backtrack(
                        x,
                        update_direction,
                        final_score,
                        g0,
                        tmpseg,
                        var_bound=var_bound,
                        weight=self.weight,
                        alpha0=self.alpha0,
                        ro=self.ro,
                        c1=self.c1,
                        min_direction=self.min_direction,
                        stop_grad=self.stop_grad,
                        score_at=score_at,
                    )
                    if improved:
                        x = x_new
                        final_score = new_score
                        if self.verbose:
                            print(f"iter {iter_count + 1} (fallback): score {final_score:.6g}")

                if not improved:
                    break

            # successful step
            iter_count += 1
            if iter_count >= self.maxiter:
                break

            # compute new gradient at current x
            g1 = self.compute_fd_gradient(x, tmpseg, score_at)

            # Polak-Ribiere style beta (ascent variant)
            denom = np.dot(g0, g0)
            if denom <= 1e-20:
                beta = 0.0
            else:
                beta = np.dot(g1, (g1 - g0)) / denom
                if beta < 0.0:
                    beta = 0.0

            # update direction (C code uses gradient + beta * old_direction)
            update_direction = g1 + beta * update_direction

            # move start_grad
            g0 = g1.copy()

        # write back final parameters to the real segment and update derived values
        self._apply_x_to_seg(seg, x)
        # seg.theta, seg.psi = normalize_euler_zx(seg.theta, seg.psi)
        seg.theta, seg.psi = seg.theta%(2*np.pi), seg.psi%(2*np.pi)
        seg.score = seg.score_segment(self.image, correlation_score)
        seg.mean_intensity = seg.score_segment(self.image, mean_intensity_score)
