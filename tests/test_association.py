from collections.abc import Collection, Sequence

import numpy as np
import pytest

from slamkit.association import (
    AssociationContext,
    JointCompatibility,
    KnownCorrespondence,
    Matched,
    NearestNeighbor,
    NewLandmark,
    Pairing,
    Unassociated,
)
from slamkit.core import (
    BodyFrame,
    LandmarkId,
    LandmarkIdAllocator,
    Mat3,
    Matrix,
    WorldFrame,
)
from slamkit.geometry import SE2, Point2
from slamkit.probability import ChiSquare, Covariance
from slamkit.sensing import (
    CartesianModel,
    MeasurementLinearization,
    Observation,
    PointObservation,
    PointSensor,
    RelativePosition,
)

SENSOR = PointSensor("cart", CartesianModel(0.1, 0.1))
ORIGIN = SE2[WorldFrame, BodyFrame].identity()


class FixedMap(AssociationContext):
    """Landmarks at known, exact positions, seen from the origin with a given
    pose uncertainty (zero by default, so that ``S = R``)."""

    def __init__(
        self,
        landmarks: dict[LandmarkId, Point2[WorldFrame]],
        pose_covariance: Mat3 | None = None,
    ) -> None:
        self._landmarks = landmarks
        self._pose_covariance = (
            pose_covariance if pose_covariance is not None else Mat3(np.zeros((3, 3)))
        )

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._landmarks.keys()

    def linearize(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> MeasurementLinearization:
        return observation.linearize(ORIGIN, self._landmarks[landmark_id])

    def covariance(self, landmark_ids: Sequence[LandmarkId]) -> Matrix:
        n = 3 + 2 * len(landmark_ids)
        joint = np.zeros((n, n))
        joint[:3, :3] = self._pose_covariance
        return Matrix(joint)


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
MAP_LANDMARKS = {LandmarkId(i): Point2[WorldFrame](float(i), 5.0) for i in range(4)}


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


class TestJointInnovation:
    def test_pose_uncertainty_correlates_observations(self) -> None:
        context = FixedMap(MAP_LANDMARKS, Mat3(Covariance.diagonal(0.5, 0.01, 0.001)))
        joint = context.joint_innovation(
            [
                Pairing(observe(0.0, 5.0), LandmarkId(0)),
                Pairing(observe(1.0, 5.0), LandmarkId(1)),
            ]
        )
        assert joint.dim == 4
        assert joint.covariance[0, 2] == pytest.approx(0.25, rel=1e-3)  # shared x
        assert joint.covariance[0, 0] == pytest.approx(0.25 + 0.01, rel=1e-3)


class TestJointCompatibility:
    """A robot unsure of its x-position sees a row of evenly spaced landmarks,
    all shifted by the same unknown offset. Each observation individually sits
    closer to the *wrong* neighbour; only jointly is the shift recognisable."""

    POSE_COVARIANCE = Covariance.diagonal(0.5, 0.01, 0.001)

    def scene(self) -> tuple[FixedMap, list[PointObservation[RelativePosition]]]:
        context = FixedMap(MAP_LANDMARKS, Mat3(self.POSE_COVARIANCE))
        observations = [observe(x + 0.6, 5.0) for x in (0.0, 1.0, 2.0, 3.0)]
        return context, observations

    def test_nearest_neighbour_is_fooled(self) -> None:
        context, observations = self.scene()
        result = NearestNeighbor().associate(observations, context)
        assert result[0] == Matched(LandmarkId(1))

    def test_joint_compatibility_recovers_the_shift(self) -> None:
        context, observations = self.scene()
        result = JointCompatibility().associate(observations, context)
        assert result == [Matched(LandmarkId(i)) for i in range(4)]

    def test_unexplained_observation_becomes_new_landmark(self) -> None:
        context, observations = self.scene()
        far = observe(50.0, 50.0)
        result = JointCompatibility(allocator=LandmarkIdAllocator(100)).associate(
            [*observations, far], context
        )
        assert result[-1] == NewLandmark(LandmarkId(100))
        assert result[:4] == [Matched(LandmarkId(i)) for i in range(4)]


class TestChiSquare:
    @pytest.mark.parametrize(
        ("probability", "dof", "exact"),
        [
            (0.95, 1, 3.8415),
            (0.99, 1, 6.6349),
            (0.95, 2, 5.9915),
            (0.99, 2, 9.2103),
            (0.99, 3, 11.3449),
            (0.99, 4, 13.2767),
        ],
    )
    def test_quantile_close_to_tables(
        self, probability: float, dof: int, exact: float
    ) -> None:
        tolerance = 1e-4 if dof <= 2 else 5e-3
        assert ChiSquare.quantile(probability, dof) == pytest.approx(
            exact, rel=tolerance
        )
