import math
from typing import Any

import numpy as np

from slamkit.core import BodyFrame, Mat3, WorldFrame
from slamkit.geometry import SE2, Point2
from slamkit.initialization import (
    FallbackInitializer,
    Sighting,
    SingleShotInitializer,
    TriangulatingInitializer,
)
from slamkit.sensing import (
    BearingModel,
    Measurement,
    PointObservation,
    PointSensor,
    RangeBearingModel,
)

LANDMARK = Point2[WorldFrame](5.0, 3.0)
BEARING = PointSensor("cam", BearingModel(math.radians(0.5)))
RANGE_BEARING = PointSensor("rb", RangeBearingModel(0.1, 0.01))


def look_from[M: Measurement[Any]](
    sensor: PointSensor[M], *xs: float
) -> list[Sighting[None]]:
    """Noise-free sightings of ``LANDMARK`` from poses along the x-axis."""
    result: list[Sighting[None]] = []
    for x in xs:
        pose = SE2[WorldFrame, BodyFrame](x, 0.0, 0.3)
        measurement = sensor.predict(pose, LANDMARK).measurement
        result.append(
            Sighting(
                pose,
                Mat3(np.zeros((3, 3))),
                PointObservation(sensor, measurement),
                None,
            )
        )
    return result


class TestSingleShot:
    def test_inverts_an_invertible_reading(self) -> None:
        initialization = SingleShotInitializer().initialize(
            look_from(RANGE_BEARING, 0.0)
        )
        assert initialization is not None
        assert np.allclose(initialization.position.to_vector(), LANDMARK.to_vector())

    def test_waits_on_a_bearing(self) -> None:
        assert SingleShotInitializer().initialize(look_from(BEARING, 0.0, 4.0)) is None


class TestTriangulation:
    def test_recovers_the_point_from_bearings(self) -> None:
        initialization = TriangulatingInitializer().initialize(
            look_from(BEARING, 0.0, 2.0, 4.0)
        )
        assert initialization is not None
        assert np.allclose(
            initialization.position.to_vector(), LANDMARK.to_vector(), atol=1e-6
        )
        assert np.all(np.linalg.eigvalsh(initialization.measurement_covariance) > 0.0)

    def test_waits_for_parallax(self) -> None:
        nearby = look_from(BEARING, 0.0, 0.05)
        assert (
            TriangulatingInitializer(min_parallax=math.radians(5.0)).initialize(nearby)
            is None
        )

    def test_waits_for_enough_sightings(self) -> None:
        assert TriangulatingInitializer().initialize(look_from(BEARING, 0.0)) is None

    def test_more_baseline_means_less_uncertainty(self) -> None:
        short = TriangulatingInitializer(min_parallax=0.0).initialize(
            look_from(BEARING, 0.0, 1.0)
        )
        long = TriangulatingInitializer(min_parallax=0.0).initialize(
            look_from(BEARING, 0.0, 4.0)
        )
        assert short is not None and long is not None
        assert np.trace(long.measurement_covariance) < np.trace(
            short.measurement_covariance
        )


class TestFallback:
    def test_prefers_single_shot_then_triangulates(self) -> None:
        standard = FallbackInitializer.standard()
        assert standard.initialize(look_from(RANGE_BEARING, 0.0)) is not None
        assert standard.initialize(look_from(BEARING, 0.0)) is None
        assert standard.initialize(look_from(BEARING, 0.0, 4.0)) is not None
