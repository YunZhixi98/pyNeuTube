# tracers/pyNeuTube/tracing_base.py

"""
tracing_base.py

Definition of tracing segments and utilities of tracing.
"""

from abc import ABC

import numpy as np

from .config import Defaults

class BaseTracingSegment(ABC):
    def __init__(self, radius: float, coord: np.ndarray, length: float = Defaults.SEG_LENGTH, 
                 theta: float = 0.0, psi: float = 0.0, 
                 scale: float = 1.0, ):
        """
        Initialize a base tracing segment.
        A segment is a cylinder or cone-like shape in 3D space with a specified radius, length, and orientation.

        Parameters
        ----------
        radius : float
            Radius of the segment.
        coord : np.ndarray
            Position of the segment in 3D space as a numpy array of shape (3,).
        length : float
            Length of the segment. Defaults to 11.
        theta : float
            Rotation angle along x-axis (counter-clockwise) of the segment in radians. Defaults to 0.0. Range: [0, π].
        psi : float
            Rotation angle along z-axis (counter-clockwise) of the segment in radians. Defaults to 0.0. Range: [0, 2π].
        scale : float
            Scale factor to control the cross-sectional shape of the segment. Defaults to 1.0.
            If scale=1.0, the cross-section is a circle. If scale≠1.0, the cross-section is an ellipse,
            where x = y * scale.
        """
        self.radius = radius
        self.length = length
        self.theta = theta % np.pi  # Ensure theta is in [0, π]
        self.psi = psi % (2 * np.pi)  # Ensure psi is in [0, 2π]
        self.scale = scale
