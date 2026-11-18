import math
from typing import Any

import numpy as np
import pytest

from slamkit.core import BodyFrame, LandmarkId, SensorFrame, WorldFrame
from slamkit.geometry import SE2, NumericalJacobian, Point2
from slamkit.probability import Covariance
from slamkit.sensing import (
    Bearing,
    BearingModel,
    CartesianModel,
    LandmarkInitialization,
    Measurement,
    PointSensor,
    RangeBearing,
    RangeBearingModel,
    SensingRange,
)

POSE = SE2[WorldFrame, BodyFrame](1.0, -2.0, 0.7)
LANDMARK = Point2[WorldFrame](4.0, 1.5)
EXTRINSIC = SE2[BodyFrame, SensorFrame](0.3, -0.1, -0.5)

BEARING = PointSensor("bearing", BearingModel(0.01), extrinsic=EXTRINSIC)

INVERTIBLE: list[PointSensor[Any]] = [
    PointSensor("range-bearing", RangeBearingModel(0.1, 0.02)),
    PointSensor(
        "range-bearing-offset", RangeBearingModel(0.1, 0.02), extrinsic=EXTRINSIC
    ),
    PointSensor("cartesian-offset", CartesianModel(0.1, 0.1), extrinsic=EXTRINSIC),
]


class TestMeasurement:
    def test_range_bearing_innovation_wraps(self) -> None:
        a, b = RangeBearing(1.0, math.pi - 0.01), RangeBearing(1.0, -math.pi + 0.01)
        assert float(a.ominus(b)[1]) == pytest.approx(-0.02)


@pytest.mark.parametrize("sensor", [*INVERTIBLE, BEARING], ids=lambda s: s.name)
class TestPointSensor:
    jacobian = NumericalJacobian()

    def test_prediction_jacobians(self, sensor: PointSensor[Measurement[Any]]) -> None:
        prediction = sensor.predict(POSE, LANDMARK)
        assert np.allclose(
            self.jacobian(lambda x: sensor.predict(x, LANDMARK).measurement, POSE),
            prediction.wrt_pose,
            atol=1e-6,
        )
        assert np.allclose(
            self.jacobian(lambda lm: sensor.predict(POSE, lm).measurement, LANDMARK),
            prediction.wrt_landmark,
            atol=1e-6,
        )

    def test_linearization_of_exact_measurement_has_zero_innovation(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, LANDMARK).measurement
        assert np.allclose(
            sensor.linearize(measurement, POSE, LANDMARK).innovation, 0.0
        )

    def test_residual_matches_linearization(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, Point2[WorldFrame](4.2, 1.3)).measurement
        assert np.allclose(
            sensor.residual(measurement, POSE, LANDMARK),
            sensor.linearize(measurement, POSE, LANDMARK).innovation,
        )

    def test_ray_passes_through_landmark(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        ray = sensor.ray(sensor.predict(POSE, LANDMARK).measurement, POSE)
        depth = ray.depth_of(LANDMARK)
        assert depth > 0.0
        assert np.allclose(ray.point_at(depth).to_vector(), LANDMARK.to_vector())


@pytest.mark.parametrize("sensor", INVERTIBLE, ids=lambda s: s.name)
class TestInvertibleSensor:
    jacobian = NumericalJacobian()

    def initialize(
        self,
        sensor: PointSensor[Measurement[Any]],
        measurement: Measurement[Any],
        pose: SE2[WorldFrame, BodyFrame],
    ) -> LandmarkInitialization:
        initialization = sensor.initialize_landmark(measurement, pose)
        assert initialization is not None
        return initialization

    def test_initialization_inverts_prediction(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, LANDMARK).measurement
        assert np.allclose(
            self.initialize(sensor, measurement, POSE).position.to_vector(),
            LANDMARK.to_vector(),
        )

    def test_initialization_jacobians(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, LANDMARK).measurement
        init = self.initialize(sensor, measurement, POSE)
        assert np.allclose(
            self.jacobian(
                lambda x: self.initialize(sensor, measurement, x).position, POSE
            ),
            init.wrt_pose,
            atol=1e-6,
        )
        wrt_measurement = self.jacobian(
            lambda z: self.initialize(sensor, z, POSE).position, measurement
        )
        assert np.allclose(
            Covariance.propagate(wrt_measurement, sensor.model.covariance),
            init.measurement_covariance,
            atol=1e-9,
        )


class TestBearingSensor:
    def test_cannot_initialize_from_one_reading(self) -> None:
        measurement = BEARING.predict(POSE, LANDMARK).measurement
        assert BEARING.initialize_landmark(measurement, POSE) is None

    def test_bearing_innovation_wraps(self) -> None:
        assert float(
            Bearing(math.pi - 0.01).ominus(Bearing(-math.pi + 0.01))[0]
        ) == pytest.approx(-0.02)


class TestSensingRange:
    def test_limits(self) -> None:
        sensing = SensingRange(max_range=5.0, field_of_view=math.pi / 2)
        assert sensing.contains(Point2[SensorFrame](3.0, 0.5))
        assert not sensing.contains(Point2[SensorFrame](6.0, 0.0))
        assert not sensing.contains(Point2[SensorFrame](0.0, 3.0))

    def test_out_of_range_landmark_is_not_observed(self) -> None:
        sensor = PointSensor(
            "short",
            RangeBearingModel(0.1, 0.01),
            sensing_range=SensingRange(max_range=1.0),
        )
        rng = np.random.default_rng(0)
        assert sensor.observe(POSE, LANDMARK, LandmarkId(0), rng) is None
