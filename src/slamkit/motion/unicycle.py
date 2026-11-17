"""
Unicycle (velocity) motion model
================================
"""
# src/slamkit/motion/unicycle.py

import numpy as np

from slamkit.core.arrays import Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import VelocityCommand
from slamkit.motion.model import MotionModel
from slamkit.probability.covariance import Covariance

__all__ = [
    "UnicycleMotionModel",
]


class UnicycleMotionModel(MotionModel[VelocityCommand]):
    """Constant ``(v, ω)`` for ``dt``: the increment is ``Exp([v·dt, 0, ω·dt])``,
    an exact arc (no small-step Euler approximation).

    Noise is on the velocities, ``σ_v = linear_noise·|v| + floor`` and
    ``σ_ω = angular_noise·|ω| + floor``, mapped into the pose tangent through
    ``∂τ/∂(v, ω)`` and the SE(2) right Jacobian. A ``floor`` on every axis keeps
    ``Q`` positive-definite (a unicycle cannot slide sideways, but real robots do).
    """

    def __init__(
        self,
        linear_noise: float = 0.05,
        angular_noise: float = 0.05,
        floor: float = 1e-3,
    ) -> None:
        self._linear_noise = linear_noise
        self._angular_noise = angular_noise
        self._floor = floor

    def increment(self, control: VelocityCommand) -> SE2[BodyFrame, BodyFrame]:
        return SE2[BodyFrame, BodyFrame].exp(self._tangent(control))

    def increment_covariance(self, control: VelocityCommand) -> Mat3:
        velocity_covariance = Covariance.diagonal(
            self._linear_noise * abs(control.linear) + self._floor,
            self._angular_noise * abs(control.angular) + self._floor,
        )
        dtau_dvelocity = Matrix([[control.dt, 0.0], [0.0, 0.0], [0.0, control.dt]])
        tangent_covariance = Covariance.propagate(dtau_dvelocity, velocity_covariance)
        right_jacobian = SE2.right_jacobian(self._tangent(control))
        q = (
            Covariance.propagate(Matrix(right_jacobian), tangent_covariance)
            + np.eye(3) * self._floor**2
        )
        return Mat3(Covariance.symmetrize(Matrix(q)))

    @staticmethod
    def _tangent(control: VelocityCommand) -> Vector:
        return Vector([control.linear * control.dt, 0.0, control.angular * control.dt])
