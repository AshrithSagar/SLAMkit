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
from collections.abc import Collection, Mapping, Sequence
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
from slamkit.initialization.base import LandmarkInitializer, Sighting
from slamkit.initialization.fallback import FallbackInitializer
from slamkit.motion.controls import Control
from slamkit.sensing.observation import LandmarkInitialization, Observation

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


class MapHypothesis[A](AssociationContext):
    """One self-consistent hypothesis about the map, into which associated
    observations are folded.

    An EKF holds one such hypothesis; a factor graph holds one; a particle
    filter holds one *per particle*. Each implements only how a matched or a
    new landmark changes it -- the association round-trip, and the buffering
    of sightings of landmarks that cannot be initialised yet, are shared.

    ``A`` is the *anchor* each sighting is stamped with (see ``anchor``).
    """

    def __init__(
        self, initializer: LandmarkInitializer | None = None, max_sightings: int = 20
    ) -> None:
        self._initializer = (
            initializer if initializer is not None else FallbackInitializer.standard()
        )
        self._max_sightings = max_sightings
        self._pending: dict[LandmarkId, list[Sighting[A]]] = {}

    @property
    @abstractmethod
    def current_pose(self) -> SE2[WorldFrame, BodyFrame]:
        """The pose estimate observations are currently made from."""

    @property
    @abstractmethod
    def current_pose_covariance(self) -> Mat3:
        """The uncertainty of ``current_pose``, in its right tangent space."""

    @property
    @abstractmethod
    def anchor(self) -> A:
        """What this hypothesis needs to remember about *now*, to use a sighting
        made now once its landmark is finally initialised."""

    @property
    def pending_landmarks(self) -> Collection[LandmarkId]:
        """Landmarks seen but not yet initialised (e.g. awaiting parallax)."""
        return self._pending.keys()

    @abstractmethod
    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """Fold in ``observation`` as a re-observation of a mapped landmark."""

    @abstractmethod
    def add_landmark(
        self,
        landmark_id: LandmarkId,
        initialization: LandmarkInitialization,
        sightings: Sequence[Sighting[A]],
    ) -> None:
        """Extend the map with an initialised landmark, given every sighting that
        led to it (oldest first; the last was made from the current pose)."""

    def introduce(self, observation: Observation, landmark_id: LandmarkId) -> bool:
        """Record a sighting of an unmapped landmark, and add it to the map as
        soon as the initialiser can place it. Returns whether it was added."""
        sightings = self._pending.setdefault(landmark_id, [])
        sightings.append(
            Sighting(
                self.current_pose,
                self.current_pose_covariance,
                observation,
                self.anchor,
            )
        )
        del sightings[: -self._max_sightings]
        initialization = self._initializer.initialize(sightings)
        if initialization is None:
            return False
        del self._pending[landmark_id]
        self.add_landmark(landmark_id, initialization, sightings)
        return True

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
                    self.introduce(observation, landmark_id)
                case Unassociated():
                    pass
        return associations

    def _inherit_pending(self, other: MapHypothesis[A]) -> None:
        """Copy ``other``'s pending sightings (for hypotheses that fork)."""
        self._pending = {lid: list(s) for lid, s in other._pending.items()}
