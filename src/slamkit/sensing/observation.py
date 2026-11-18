"""
Observations
============
The estimator-facing view of a sensor reading.

Estimators receive observations from *many* sensors at once, each with its own
measurement type ``M``. ``Observation`` is the existential wrapper that makes
that possible without ``Any``: it hides ``M`` behind the operations estimators
need, all of which return plain vectors, matrices and geometry --

* ``residual``            : ``z ⊖ h(x, l)``, nothing else (sigma-point filters);
* ``linearize``           : the residual plus its Jacobians (EKF, graphs);
* ``initialize_landmark`` : the inverse model ``l = h⁻¹(x, z)``, if one reading
                            fixes the landmark (``None`` for bearing-only);
* ``ray``                 : the half-line the reading confines the landmark to
                            (delayed initialisation by triangulation).

So every estimator consumes range-bearing, cartesian, bearing-only, or any
future point sensor through this one interface.

The *identity* of the observed landmark (``tag``) is deliberately not part of
the measurement: deciding which landmark was seen is data association's job;
a tag, when present, is just evidence it may use.
"""
# src/slamkit/sensing/observation.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

from slamkit.core.arrays import Mat2, Mat2x3, Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.ray import Ray
from slamkit.geometry.se2 import SE2
from slamkit.probability.covariance import Covariance

__all__ = [
    "LandmarkInitialization",
    "MeasurementLinearization",
    "Observation",
]


@dataclass(frozen=True, slots=True)
class MeasurementLinearization:
    """First-order model of one observation about ``(pose, landmark)``:
    ``z ⊖ h(x ⊕ δx, l ⊕ δl) ≈ innovation - wrt_pose δx - wrt_landmark δl``,
    with the remaining error distributed as ``N(0, noise_covariance)``."""

    innovation: Vector
    """``z ⊖ h(x, l)``."""
    wrt_pose: Matrix
    """``∂h/∂x``, shape ``(k, 3)``."""
    wrt_landmark: Matrix
    """``∂h/∂l``, shape ``(k, 2)``."""
    noise_covariance: Matrix
    """``R`` (plus any linearisation-error covariance), shape ``(k, k)``."""

    @property
    def dim(self) -> int:
        return self.innovation.shape[0]


@dataclass(frozen=True, slots=True)
class LandmarkInitialization:
    """A new landmark's position and the information to insert it into a belief.

    The landmark covariance splits into a part correlated with the *current*
    pose estimate (through ``wrt_pose``) and a part independent of it
    (``measurement_covariance``). A single invertible reading gives both; a
    triangulation over past poses gives only the latter.
    """

    position: Point2[WorldFrame]
    wrt_pose: Mat2x3
    """``∂l/∂x`` for the current pose ``x``."""
    measurement_covariance: Mat2
    """The covariance ``l`` would have if the current pose were known exactly."""

    def covariance(self, pose_covariance: Mat3 | None = None) -> Mat2:
        """``Σ_l = J_x Σ_x J_xᵀ + Σ_meas`` (just ``Σ_meas`` for a known pose)."""
        sigma = Matrix(self.measurement_covariance)
        if pose_covariance is not None:
            jx, px = Matrix(self.wrt_pose), Matrix(pose_covariance)
            sigma = Matrix(sigma + Covariance.propagate(jx, px))
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
    def residual(
        self, pose: SE2[WorldFrame, BodyFrame], landmark: Point2[WorldFrame]
    ) -> Vector:
        """``z ⊖ h(x, l)``, without Jacobians."""

    @abstractmethod
    def linearize(
        self, pose: SE2[WorldFrame, BodyFrame], landmark: Point2[WorldFrame]
    ) -> MeasurementLinearization: ...

    @abstractmethod
    def initialize_landmark(
        self, pose: SE2[WorldFrame, BodyFrame]
    ) -> LandmarkInitialization | None:
        """The landmark this reading implies, or ``None`` if one reading cannot fix it."""

    @abstractmethod
    def ray(self, pose: SE2[WorldFrame, BodyFrame]) -> Ray[WorldFrame]:
        """The world-frame half-line from the sensor along which the landmark lies."""
