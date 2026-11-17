"""
Motion model
============
How the robot moves: ``p(x_t | x_{t-1}, u_t)``.

Every planar motion model in slamkit is a *body-frame increment*:

    x_t = x_{t-1} ∘ Δ(u_t) ∘ Exp(w),     w ~ N(0, Q(u_t))

A concrete model supplies only ``increment`` (Δ) and ``increment_covariance``
(Q). Everything an estimator needs is derived here, once:

* ``predict``   : the mean                          (all backends)
* ``linearize`` : ``F = ∂x_t/∂x_{t-1} = Ad(Δ⁻¹)``    (EKF)
* ``sample``    : a draw from the distribution      (particle filters)
* ``increment`` : the relative-pose measurement     (pose-graph between-factors)

Because the noise enters on the right, ``Q`` is expressed in the new body
frame -- the same tangent space as the pose covariance everywhere else.
"""
# src/slamkit/motion/model.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Mat3
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.probability.gaussian import Gaussian

__all__ = [
    "MotionLinearization",
    "MotionModel",
]


@dataclass(frozen=True, slots=True)
class MotionLinearization:
    """``x_t ⊕ δ_t ≈ (x_{t-1} ⊕ δ_{t-1}) ∘ Δ ∘ Exp(w)`` with ``δ_t ≈ F δ_{t-1} + w``."""

    predicted: SE2[WorldFrame, BodyFrame]
    wrt_pose: Mat3
    """``F``."""
    noise_covariance: Mat3
    """``Q``."""


class MotionModel[C: Control](ABC):
    """A motion model driven by controls of type ``C``."""

    @abstractmethod
    def increment(self, control: C) -> SE2[BodyFrame, BodyFrame]:
        """``Δ(u)``: the noise-free motion, expressed in the previous body frame."""

    @abstractmethod
    def increment_covariance(self, control: C) -> Mat3:
        """``Q(u)``: covariance of ``w`` in the new body frame's tangent space."""

    def predict(
        self, pose: SE2[WorldFrame, BodyFrame], control: C
    ) -> SE2[WorldFrame, BodyFrame]:
        return pose @ self.increment(control)

    def linearize(
        self, pose: SE2[WorldFrame, BodyFrame], control: C
    ) -> MotionLinearization:
        increment = self.increment(control)
        return MotionLinearization(
            predicted=pose @ increment,
            wrt_pose=increment.inverse().adjoint(),
            noise_covariance=self.increment_covariance(control),
        )

    def sample(
        self, pose: SE2[WorldFrame, BodyFrame], control: C, rng: np.random.Generator
    ) -> SE2[WorldFrame, BodyFrame]:
        noise = Gaussian.zero_mean(self.increment_covariance(control)).sample(rng)
        return self.predict(pose, control).oplus(noise)
