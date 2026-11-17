"""
Point measurement models
========================
The sensor-specific part of observing a point landmark, and nothing else.

A ``PointMeasurementModel[M]`` only knows how a point *already expressed in the
sensor's frame* maps to a reading ``M`` and back:

    project   : Point2[SensorFrame] → M          (with ∂M/∂p)
    unproject : M → Point2[SensorFrame]          (with ∂p/∂M)

Robot pose, extrinsic calibration and the chain rule through them are handled
once, generically, by ``PointSensor``. Adding a new point sensor therefore means
writing these four small functions -- no SE(2) Jacobians, no estimator changes.
"""
# src/slamkit/sensing/models.py

import math
from abc import ABC, abstractmethod
from typing import Any

from slamkit.core.arrays import Matrix
from slamkit.core.frames import SensorFrame
from slamkit.geometry.point import Point2
from slamkit.probability.covariance import Covariance
from slamkit.sensing.measurement import Measurement, RangeBearing, RelativePosition

__all__ = [
    "CartesianModel",
    "PointMeasurementModel",
    "RangeBearingModel",
]


class PointMeasurementModel[M: Measurement[Any]](ABC):
    """``h: Point2[SensorFrame] → M`` with additive Gaussian noise of covariance ``R``."""

    def __init__(self, covariance: Matrix) -> None:
        Covariance.validate(covariance, covariance.shape[0])
        self._covariance = covariance

    @property
    def covariance(self) -> Matrix:
        """``R``: measurement noise covariance, in ``M``'s tangent space."""
        return self._covariance

    @property
    def dim(self) -> int:
        return self._covariance.shape[0]

    @abstractmethod
    def project(self, point: Point2[SensorFrame]) -> M:
        """``z = h(p)``."""

    @abstractmethod
    def project_jacobian(self, point: Point2[SensorFrame]) -> Matrix:
        """``∂h/∂p``, shape ``(dim, 2)``."""

    @abstractmethod
    def unproject(self, measurement: M) -> Point2[SensorFrame]:
        """``p = h⁻¹(z)``. Requires the measurement to fully determine the point."""

    @abstractmethod
    def unproject_jacobian(self, measurement: M) -> Matrix:
        """``∂h⁻¹/∂z``, shape ``(2, dim)``."""


class RangeBearingModel(PointMeasurementModel[RangeBearing]):
    """Polar coordinates of the point: the classic EKF-SLAM landmark sensor."""

    def __init__(self, range_std: float, bearing_std: float) -> None:
        super().__init__(Covariance.diagonal(range_std, bearing_std))

    def project(self, point: Point2[SensorFrame]) -> RangeBearing:
        return RangeBearing(math.hypot(point.x, point.y), math.atan2(point.y, point.x))

    def project_jacobian(self, point: Point2[SensorFrame]) -> Matrix:
        q = point.x**2 + point.y**2
        r = math.sqrt(q)
        return Matrix([[point.x / r, point.y / r], [-point.y / q, point.x / q]])

    def unproject(self, measurement: RangeBearing) -> Point2[SensorFrame]:
        r, b = measurement.range, measurement.bearing
        return Point2[SensorFrame](r * math.cos(b), r * math.sin(b))

    def unproject_jacobian(self, measurement: RangeBearing) -> Matrix:
        r, b = measurement.range, measurement.bearing
        c, s = math.cos(b), math.sin(b)
        return Matrix([[c, -r * s], [s, r * c]])


class CartesianModel(PointMeasurementModel[RelativePosition]):
    """The point's coordinates themselves, with independent per-axis noise."""

    def __init__(self, x_std: float, y_std: float) -> None:
        super().__init__(Covariance.diagonal(x_std, y_std))

    def project(self, point: Point2[SensorFrame]) -> RelativePosition:
        return RelativePosition(point.x, point.y)

    def project_jacobian(self, point: Point2[SensorFrame]) -> Matrix:
        return Matrix([[1.0, 0.0], [0.0, 1.0]])

    def unproject(self, measurement: RelativePosition) -> Point2[SensorFrame]:
        return Point2[SensorFrame](measurement.x, measurement.y)

    def unproject_jacobian(self, measurement: RelativePosition) -> Matrix:
        return Matrix([[1.0, 0.0], [0.0, 1.0]])
