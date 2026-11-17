from collections.abc import Collection

import numpy as np
import pytest

from slamkit.association import (
    AssociationContext,
    KnownCorrespondence,
    Matched,
    NearestNeighbor,
    NewLandmark,
    Unassociated,
)
from slamkit.core import (
    BodyFrame,
    LandmarkId,
    LandmarkIdAllocator,
    Matrix,
    Vector,
    WorldFrame,
)
from slamkit.geometry import SE2, Point2
from slamkit.probability import Innovation
from slamkit.sensing import (
    CartesianModel,
    Observation,
    PointObservation,
    PointSensor,
    RelativePosition,
)

SENSOR = PointSensor("cart", CartesianModel(0.1, 0.1))
ORIGIN = SE2[WorldFrame, BodyFrame].identity()


class FixedMap(AssociationContext):
    """A map with known landmark positions and an isotropic innovation covariance."""

    def __init__(
        self, landmarks: dict[LandmarkId, Point2[WorldFrame]], sigma: float = 0.1
    ) -> None:
        self._landmarks = landmarks
        self._covariance = Matrix(np.eye(2) * sigma**2)

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._landmarks.keys()

    def innovation(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> Innovation:
        residual = observation.linearize(
            ORIGIN, self._landmarks[landmark_id]
        ).innovation
        return Innovation(Vector(residual), self._covariance)


def observe(
    x: float, y: float, tag: int | None = None
) -> PointObservation[RelativePosition]:
    return PointObservation(
        SENSOR, RelativePosition(x, y), None if tag is None else LandmarkId(tag)
    )


MAP = FixedMap(
    {
        LandmarkId(0): Point2[WorldFrame](0.0, 0.0),
        LandmarkId(1): Point2[WorldFrame](5.0, 0.0),
    }
)


class TestKnownCorrespondence:
    def test_uses_tags(self) -> None:
        result = KnownCorrespondence().associate(
            [observe(0, 0, 1), observe(9, 9, 7), observe(9, 9, 7), observe(0, 0)], MAP
        )
        assert result[0] == Matched(LandmarkId(1))
        assert result[1] == NewLandmark(LandmarkId(7))
        assert result[2] == Matched(LandmarkId(7))
        assert isinstance(result[3], Unassociated)


class TestNearestNeighbor:
    def test_matches_new_and_ambiguous(self) -> None:
        associator = NearestNeighbor(
            gate=9.21, new_landmark=100.0, allocator=LandmarkIdAllocator(10)
        )
        result = associator.associate(
            [
                observe(5.05, 0.0),
                observe(0.02, 0.01),
                observe(20.0, 20.0),
                observe(0.6, 0.0),
            ],
            MAP,
        )
        assert result[0] == Matched(LandmarkId(1))
        assert result[1] == Matched(LandmarkId(0))
        assert result[2] == NewLandmark(LandmarkId(10))
        assert isinstance(
            result[3], Unassociated
        )  # d² = 36: beyond the gate, below new_landmark

    def test_each_landmark_matched_at_most_once(self) -> None:
        result = NearestNeighbor().associate(
            [observe(0.01, 0.0), observe(0.02, 0.0)], MAP
        )
        assert result.count(Matched(LandmarkId(0))) == 1

    def test_rejects_inconsistent_thresholds(self) -> None:
        with pytest.raises(ValueError):
            NearestNeighbor(gate=10.0, new_landmark=5.0)
