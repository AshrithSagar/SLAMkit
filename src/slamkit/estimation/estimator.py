"""
Estimators
==========
The SLAM estimator interface, and the map-hypothesis abstraction shared by
every backend.

What *differs* between EKF-SLAM, FastSLAM and GraphSLAM is only how the belief
over ``(trajectory, map)`` is represented and updated. Motion models, sensors,
observations and data association are shared, unchanged, by all of them; an
estimator is configured by composing those parts.
"""
# src/slamkit/estimation/estimator.py

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from slamkit.association.base import (
    Association,
    AssociationContext,
    DataAssociator,
    Matched,
    NewLandmark,
    Unassociated,
)
from slamkit.core.arrays import Mat3
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.sensing.observation import Observation

__all__ = [
    "MapHypothesis",
    "SlamEstimate",
    "SlamEstimator",
    "SmoothingSlamEstimator",
]


@dataclass(frozen=True, slots=True)
class SlamEstimate:
    """A point estimate of the current pose and the map."""

    pose: SE2[WorldFrame, BodyFrame]
    landmarks: Mapping[LandmarkId, Point2[WorldFrame]] = field(
        default_factory=dict[LandmarkId, Point2[WorldFrame]]
    )
    pose_covariance: Mat3 | None = None


class SlamEstimator[C: Control](ABC):
    """Online SLAM: alternately predict with a control and update with observations."""

    @abstractmethod
    def predict(self, control: C) -> None:
        """Propagate the belief through the motion model."""

    @abstractmethod
    def update(self, observations: Sequence[Observation]) -> None:
        """Condition the belief on a batch of simultaneous observations."""

    @abstractmethod
    def estimate(self) -> SlamEstimate: ...

    def step(self, control: C, observations: Sequence[Observation]) -> SlamEstimate:
        self.predict(control)
        self.update(observations)
        return self.estimate()


class SmoothingSlamEstimator[C: Control](SlamEstimator[C]):
    """An estimator that keeps (and re-estimates) the whole trajectory, not just
    the latest pose -- full SLAM rather than online SLAM."""

    @abstractmethod
    def trajectory(self) -> Sequence[SE2[WorldFrame, BodyFrame]]:
        """Current estimate of every pose so far, starting with the initial one."""


class MapHypothesis(AssociationContext):
    """One self-consistent hypothesis about the map, into which associated
    observations are folded.

    An EKF holds one such hypothesis; a factor graph holds one; a particle
    filter holds one *per particle*. Each implements only how a matched or a
    new landmark changes it -- the association round-trip is shared.
    """

    @abstractmethod
    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """Fold in ``observation`` as a re-observation of a mapped landmark."""

    @abstractmethod
    def add_landmark(self, observation: Observation, landmark_id: LandmarkId) -> None:
        """Extend the map with the landmark first seen in ``observation``."""

    def assimilate(
        self, observations: Sequence[Observation], associator: DataAssociator
    ) -> list[Association]:
        """Associate a batch against this hypothesis, then apply it in order."""
        associations = associator.associate(observations, self)
        for observation, association in zip(observations, associations, strict=True):
            match association:
                case Matched(landmark_id):
                    self.update_landmark(observation, landmark_id)
                case NewLandmark(landmark_id):
                    self.add_landmark(observation, landmark_id)
                case Unassociated():
                    pass
        return associations
