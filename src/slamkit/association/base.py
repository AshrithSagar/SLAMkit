"""
Data association
================
Deciding *which* landmark each observation is of -- the correspondence problem.

The associator never sees an estimator's internals. It talks to an
``AssociationContext``: a narrow view, implemented by each estimator, that
lists the mapped landmarks and scores an observation against one of them as an
``Innovation`` (residual + covariance). That is all a gating / nearest-neighbour
/ joint-compatibility test needs, and it is equally answerable by an EKF (from
its joint covariance), a particle (from its per-landmark EKFs) or a factor
graph (from its marginals). One associator therefore serves every backend.
"""
# src/slamkit/association/base.py

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from slamkit.core.ids import LandmarkId
from slamkit.probability.gaussian import Innovation
from slamkit.sensing.observation import Observation

__all__ = [
    "Association",
    "AssociationContext",
    "DataAssociator",
    "Matched",
    "NewLandmark",
    "Unassociated",
]


@dataclass(frozen=True, slots=True)
class Matched:
    """The observation is of an already-mapped landmark."""

    landmark_id: LandmarkId


@dataclass(frozen=True, slots=True)
class NewLandmark:
    """The observation is of a landmark not yet in the map; add it under this id."""

    landmark_id: LandmarkId


@dataclass(frozen=True, slots=True)
class Unassociated:
    """The observation should be discarded (ambiguous, outlier, untagged...)."""

    reason: str


type Association = Matched | NewLandmark | Unassociated


class AssociationContext(ABC):
    """An estimator's current map, as seen by a data associator."""

    @abstractmethod
    def landmark_ids(self) -> Collection[LandmarkId]: ...

    @abstractmethod
    def innovation(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> Innovation:
        """``z ⊖ ẑ`` and its covariance ``S`` if ``observation`` were of ``landmark_id``."""


class DataAssociator(ABC):
    """Maps a batch of simultaneous observations to associations, one per observation."""

    @abstractmethod
    def associate(
        self, observations: Sequence[Observation], context: AssociationContext
    ) -> list[Association]: ...
