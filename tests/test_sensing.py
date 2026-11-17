import math
from typing import Any

import numpy as np
import pytest

from slamkit.core import BodyFrame, LandmarkId, SensorFrame, WorldFrame
from slamkit.geometry import SE2, NumericalJacobian, Point2
from slamkit.sensing import (
    CartesianModel,
    Measurement,
    PointSensor,
    RangeBearing,
    RangeBearingModel,
    SensingRange,
)

POSE = SE2[WorldFrame, BodyFrame](1.0, -2.0, 0.7)
LANDMARK = Point2[WorldFrame](4.0, 1.5)
EXTRINSIC = SE2[BodyFrame, SensorFrame](0.3, -0.1, -0.5)

SENSORS: list[PointSensor[Any]] = [
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


@pytest.mark.parametrize("sensor", SENSORS, ids=lambda s: s.name)
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

    def test_initialization_inverts_prediction(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, LANDMARK).measurement
        assert np.allclose(
            sensor.initialize_landmark(measurement, POSE).position.to_vector(),
            LANDMARK.to_vector(),
        )

    def test_initialization_jacobians(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, LANDMARK).measurement
        init = sensor.initialize_landmark(measurement, POSE)
        assert np.allclose(
            self.jacobian(
                lambda x: sensor.initialize_landmark(measurement, x).position, POSE
            ),
            init.wrt_pose,
            atol=1e-6,
        )
        assert np.allclose(
            self.jacobian(
                lambda z: sensor.initialize_landmark(z, POSE).position, measurement
            ),
            init.wrt_measurement,
            atol=1e-6,
        )

    def test_linearization_of_exact_measurement_has_zero_innovation(
        self, sensor: PointSensor[Measurement[Any]]
    ) -> None:
        measurement = sensor.predict(POSE, LANDMARK).measurement
        assert np.allclose(
            sensor.linearize(measurement, POSE, LANDMARK).innovation, 0.0
        )


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
