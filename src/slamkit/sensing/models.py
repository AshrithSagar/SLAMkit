"""
Point measurement models
========================
The sensor-specific part of observing a point landmark, and nothing else.

A ``PointMeasurementModel[M]`` only knows how a point *already expressed in the
sensor's frame* maps to a reading ``M``:

    project   : Point2[SensorFrame] → M          (with ∂M/∂p)
    direction : M → unit vector in SensorFrame   (the ray the reading allows)

An ``InvertiblePointMeasurementModel[M]`` additionally recovers the point from
a single reading -- range-bearing and cartesian sensors can, a bearing-only
sensor cannot:

    unproject : M → Point2[SensorFrame]          (with ∂p/∂M)

Robot pose, extrinsic calibration and the chain rule through them are handled
once, generically, by ``PointSensor``. Adding a new point sensor therefore means
writing these few small functions -- no SE(2) Jacobians, no estimator changes.
"""
# src/slamkit/sensing/models.py

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from slamkit.core.arrays import Matrix, Vec2
from slamkit.core.frames import SensorFrame
from slamkit.geometry.point import Point2
from slamkit.probability.covariance import Covariance
from slamkit.sensing.measurement import (
    Bearing,
    Measurement,
    RangeBearing,
    RelativePosition,
)

__all__ = [
    "BearingModel",
    "CartesianModel",
    "InvertiblePointMeasurementModel",
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
    def direction(self, measurement: M) -> Vec2:
        """The unit direction, in the sensor frame, along which the point lies."""


class InvertiblePointMeasurementModel[M: Measurement[Any]](PointMeasurementModel[M]):
    """A point model whose single reading determines the point."""

    @abstractmethod
    def unproject(self, measurement: M) -> Point2[SensorFrame]:
        """``p = h⁻¹(z)``."""

    @abstractmethod
    def unproject_jacobian(self, measurement: M) -> Matrix:
        """``∂h⁻¹/∂z``, shape ``(2, dim)``."""

    def direction(self, measurement: M) -> Vec2:
        point = self.unproject(measurement).to_vector()
        return Vec2(point / np.linalg.norm(point))


class RangeBearingModel(InvertiblePointMeasurementModel[RangeBearing]):
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


class CartesianModel(InvertiblePointMeasurementModel[RelativePosition]):
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


class BearingModel(PointMeasurementModel[Bearing]):
    """Direction only. Not invertible: landmarks seen by this sensor alone must
    be triangulated from several viewpoints (see ``slamkit.initialization``)."""

    def __init__(self, bearing_std: float) -> None:
        super().__init__(Covariance.diagonal(bearing_std))

    def project(self, point: Point2[SensorFrame]) -> Bearing:
        return Bearing(math.atan2(point.y, point.x))

    def project_jacobian(self, point: Point2[SensorFrame]) -> Matrix:
        q = point.x**2 + point.y**2
        return Matrix([[-point.y / q, point.x / q]])

    def direction(self, measurement: Bearing) -> Vec2:
        return Vec2([math.cos(measurement.angle), math.sin(measurement.angle)])
