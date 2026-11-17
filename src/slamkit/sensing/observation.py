"""
Observations
============
The estimator-facing view of a sensor reading.

Estimators receive observations from *many* sensors at once, each with its own
measurement type ``M``. ``Observation`` is the existential wrapper that makes
that possible without ``Any``: it hides ``M`` behind the only two operations an
estimator ever needs, both of which return plain vectors and matrices --

* ``linearize``           : innovation ``z ⊖ h(x, l)`` and Jacobians at ``(x, l)``;
* ``initialize_landmark`` : the inverse model ``l = h⁻¹(x, z)`` and its Jacobians.

So EKF-SLAM, FastSLAM and GraphSLAM consume range-bearing, cartesian, or any
future point sensor through this one interface.

Note that the *identity* of the observed landmark (``tag``) is deliberately not
part of the measurement: deciding which landmark was seen is data
association's job; a tag, when present, is just evidence it may use.
"""
# src/slamkit/sensing/observation.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Mat2, Mat2x3, Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.probability.covariance import Covariance
from slamkit.probability.gaussian import Innovation

__all__ = [
    "LandmarkInitialization",
    "MeasurementLinearization",
    "Observation",
]


@dataclass(frozen=True, slots=True)
class MeasurementLinearization:
    """First-order model of one observation about ``(pose, landmark)``:
    ``z ⊖ h(x ⊕ δx, l ⊕ δl) ≈ innovation - wrt_pose δx - wrt_landmark δl``."""

    innovation: Vector
    """``z ⊖ h(x, l)``."""
    wrt_pose: Matrix
    """``∂h/∂x``, shape ``(k, 3)``."""
    wrt_landmark: Matrix
    """``∂h/∂l``, shape ``(k, 2)``."""
    noise_covariance: Matrix
    """``R``, shape ``(k, k)``."""

    @property
    def jacobian(self) -> Matrix:
        """``[∂h/∂x | ∂h/∂l]``, shape ``(k, 5)``."""
        return Matrix(np.hstack((self.wrt_pose, self.wrt_landmark)))

    def innovation_given(self, pose_landmark_covariance: Matrix) -> Innovation:
        """The innovation and ``S = H Σ Hᵀ + R`` for a joint ``(pose, landmark)``
        covariance ``Σ`` (5×5; pass zeros in the pose block to treat it as known)."""
        s = (
            Covariance.propagate(self.jacobian, pose_landmark_covariance)
            + self.noise_covariance
        )
        return Innovation(self.innovation, Covariance.symmetrize(Matrix(s)))


@dataclass(frozen=True, slots=True)
class LandmarkInitialization:
    """A new landmark's position from one observation, with its Jacobians."""

    position: Point2[WorldFrame]
    wrt_pose: Mat2x3
    """``∂l/∂x``."""
    wrt_measurement: Matrix
    """``∂l/∂z``, shape ``(2, k)``."""
    noise_covariance: Matrix
    """``R`` of the generating measurement."""

    def covariance(self, pose_covariance: Mat3 | None = None) -> Mat2:
        """``Σ_l = J_z R J_zᵀ (+ J_x Σ_x J_xᵀ)``."""
        sigma = Covariance.propagate(self.wrt_measurement, self.noise_covariance)
        if pose_covariance is not None:
            sigma = Matrix(
                sigma
                + Covariance.propagate(Matrix(self.wrt_pose), Matrix(pose_covariance))
            )
        return Mat2(Covariance.symmetrize(sigma))


class Observation(ABC):
    """One reading about one (as yet unidentified) point landmark."""

    __slots__ = ()

    @property
    @abstractmethod
    def tag(self) -> LandmarkId | None:
        """An externally-known landmark identity (fiducial id, ground truth), if any."""

    @property
    @abstractmethod
    def sensor_name(self) -> str: ...

    @property
    @abstractmethod
    def noise_covariance(self) -> Matrix: ...

    @abstractmethod
    def linearize(
        self, pose: SE2[WorldFrame, BodyFrame], landmark: Point2[WorldFrame]
    ) -> MeasurementLinearization: ...

    @abstractmethod
    def initialize_landmark(
        self, pose: SE2[WorldFrame, BodyFrame]
    ) -> LandmarkInitialization: ...
