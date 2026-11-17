"""
Sensors
=======
A sensor = a measurement model + where it is mounted + what it can see.

``PointSensor[M]`` performs the frame bookkeeping for *any* point measurement
model: it moves the landmark from world to body to sensor frame, applies the
model, and chains the Jacobians through both rigid transforms. Mounting a
sensor off-centre is just a non-identity ``extrinsic``; nothing else changes.
"""
# src/slamkit/sensing/sensor.py

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from slamkit.core.arrays import Mat2x3, Matrix
from slamkit.core.frames import BodyFrame, SensorFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.probability.gaussian import Gaussian
from slamkit.sensing.measurement import Measurement
from slamkit.sensing.models import PointMeasurementModel
from slamkit.sensing.observation import (
    LandmarkInitialization,
    MeasurementLinearization,
    Observation,
)

__all__ = [
    "MeasurementPrediction",
    "PointObservation",
    "PointSensor",
    "SensingRange",
    "Sensor",
]


@dataclass(frozen=True, slots=True)
class SensingRange:
    """The region of the sensor frame in which landmarks are detectable."""

    max_range: float = math.inf
    min_range: float = 0.0
    field_of_view: float = 2.0 * math.pi
    """Total angular width, centred on the sensor's +x axis."""

    def contains(self, point: Point2[SensorFrame]) -> bool:
        distance = math.hypot(point.x, point.y)
        bearing = math.atan2(point.y, point.x)
        return (
            self.min_range <= distance <= self.max_range
            and abs(bearing) <= self.field_of_view / 2.0
        )


@dataclass(frozen=True, slots=True)
class MeasurementPrediction[M: Measurement[Any]]:
    """``ẑ = h(x, l)`` with its Jacobians."""

    measurement: M
    wrt_pose: Matrix
    wrt_landmark: Matrix


class Sensor(ABC):
    """Anything that can produce landmark observations from a (true) robot pose.

    This is the measurement-type-erased face of a sensor used by simulators
    and drivers, so heterogeneous sensors share one ``Sequence[Sensor]``.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def observe(
        self,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
        tag: LandmarkId | None,
        rng: np.random.Generator,
    ) -> Observation | None:
        """A noisy observation of ``landmark``, or ``None`` if it is not visible."""


class PointSensor[M: Measurement[Any]](Sensor):
    """A sensor that observes point landmarks through a ``PointMeasurementModel[M]``."""

    def __init__(
        self,
        name: str,
        model: PointMeasurementModel[M],
        extrinsic: SE2[BodyFrame, SensorFrame] | None = None,
        sensing_range: SensingRange | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._extrinsic = (
            extrinsic
            if extrinsic is not None
            else SE2[BodyFrame, SensorFrame].identity()
        )
        self._sensing_range = (
            sensing_range if sensing_range is not None else SensingRange()
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> PointMeasurementModel[M]:
        return self._model

    @property
    def extrinsic(self) -> SE2[BodyFrame, SensorFrame]:
        return self._extrinsic

    def predict(
        self, pose: SE2[WorldFrame, BodyFrame], landmark: Point2[WorldFrame]
    ) -> MeasurementPrediction[M]:
        """``ẑ = h(T_bs⁻¹ T_wb⁻¹ l)``, chaining Jacobians through both transforms."""
        in_body = pose.inverse_transform_with_jacobians(landmark)
        in_sensor = self._extrinsic.inverse_transform_with_jacobians(in_body.point)
        dz_dbody = self._model.project_jacobian(in_sensor.point) @ in_sensor.wrt_point
        return MeasurementPrediction[M](
            measurement=self._model.project(in_sensor.point),
            wrt_pose=Matrix(dz_dbody @ in_body.wrt_transform),
            wrt_landmark=Matrix(dz_dbody @ in_body.wrt_point),
        )

    def linearize(
        self,
        measurement: M,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
    ) -> MeasurementLinearization:
        predicted = self.predict(pose, landmark)
        return MeasurementLinearization(
            innovation=measurement.ominus(predicted.measurement),
            wrt_pose=predicted.wrt_pose,
            wrt_landmark=predicted.wrt_landmark,
            noise_covariance=self._model.covariance,
        )

    def initialize_landmark(
        self, measurement: M, pose: SE2[WorldFrame, BodyFrame]
    ) -> LandmarkInitialization:
        """``l = T_wb T_bs h⁻¹(z)``, chaining Jacobians through both transforms."""
        in_sensor = self._model.unproject(measurement)
        in_body = self._extrinsic.transform_with_jacobians(in_sensor)
        in_world = pose.transform_with_jacobians(in_body.point)
        dworld_dsensor = in_world.wrt_point @ in_body.wrt_point
        return LandmarkInitialization(
            position=in_world.point,
            wrt_pose=Mat2x3(in_world.wrt_transform),
            wrt_measurement=Matrix(
                dworld_dsensor @ self._model.unproject_jacobian(measurement)
            ),
            noise_covariance=self._model.covariance,
        )

    def measure(
        self,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
        rng: np.random.Generator,
    ) -> M | None:
        """Sample a noisy reading of ``landmark``, or ``None`` if out of range."""
        in_sensor = (pose @ self._extrinsic).inverse_transform(landmark)
        if not self._sensing_range.contains(in_sensor):
            return None
        noise = Gaussian.zero_mean(self._model.covariance).sample(rng)
        return self._model.project(in_sensor).oplus(noise)

    def observe(
        self,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
        tag: LandmarkId | None,
        rng: np.random.Generator,
    ) -> PointObservation[M] | None:
        measurement = self.measure(pose, landmark, rng)
        return None if measurement is None else PointObservation(self, measurement, tag)


class PointObservation[M: Measurement[Any]](Observation):
    """A reading ``M`` from a ``PointSensor[M]``; the two stay paired, so the
    measurement type can never be confused with another sensor's."""

    __slots__ = ("_measurement", "_sensor", "_tag")

    def __init__(
        self, sensor: PointSensor[M], measurement: M, tag: LandmarkId | None = None
    ) -> None:
        self._sensor = sensor
        self._measurement = measurement
        self._tag = tag

    @property
    def sensor(self) -> PointSensor[M]:
        return self._sensor

    @property
    def measurement(self) -> M:
        return self._measurement

    @property
    def tag(self) -> LandmarkId | None:
        return self._tag

    @property
    def sensor_name(self) -> str:
        return self._sensor.name

    @property
    def noise_covariance(self) -> Matrix:
        return self._sensor.model.covariance

    def linearize(
        self, pose: SE2[WorldFrame, BodyFrame], landmark: Point2[WorldFrame]
    ) -> MeasurementLinearization:
        return self._sensor.linearize(self._measurement, pose, landmark)

    def initialize_landmark(
        self, pose: SE2[WorldFrame, BodyFrame]
    ) -> LandmarkInitialization:
        return self._sensor.initialize_landmark(self._measurement, pose)

    def __repr__(self) -> str:
        return f"PointObservation(sensor={self.sensor_name!r}, measurement={self._measurement!r}, tag={self._tag})"
