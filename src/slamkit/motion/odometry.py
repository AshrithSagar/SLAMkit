"""
Odometry motion model
=====================
"""
# src/slamkit/motion/odometry.py

import math

from slamkit.core.arrays import Mat3
from slamkit.core.frames import BodyFrame
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Odometry
from slamkit.motion.model import MotionModel
from slamkit.probability.covariance import Covariance

__all__ = [
    "OdometryMotionModel",
]


class OdometryMotionModel(MotionModel[Odometry]):
    """The odometry reading *is* the increment; its noise grows with the motion.

    Standard deviations per step::

        σ_xy = translation_noise · |d| + floor
        σ_θ  = rotation_noise · |Δθ| + drift_per_meter · |d| + floor
    """

    def __init__(
        self,
        translation_noise: float = 0.05,
        rotation_noise: float = 0.05,
        drift_per_meter: float = 0.01,
        floor: float = 1e-3,
    ) -> None:
        self._translation_noise: float = translation_noise
        self._rotation_noise: float = rotation_noise
        self._drift_per_meter: float = drift_per_meter
        self._floor: float = floor

    def increment(self, control: Odometry) -> SE2[BodyFrame, BodyFrame]:
        return SE2[BodyFrame, BodyFrame](control.dx, control.dy, control.dtheta)

    def increment_covariance(self, control: Odometry) -> Mat3:
        distance = math.hypot(control.dx, control.dy)
        sigma_xy = self._translation_noise * distance + self._floor
        sigma_theta = (
            self._rotation_noise * abs(control.dtheta)
            + self._drift_per_meter * distance
            + self._floor
        )
        return Mat3(Covariance.diagonal(sigma_xy, sigma_xy, sigma_theta))
