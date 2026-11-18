"""
Data association
================
Deciding *which* landmark each observation is of -- the correspondence problem.

The associator never sees an estimator's internals. It talks to an
``AssociationContext``: a narrow view, implemented by each estimator, with
just two primitives --

* ``linearize(observation, landmark)``: the observation's innovation and
  Jacobians if it were of that landmark;
* ``covariance(landmarks)``: the joint covariance of the pose and those
  landmarks.

From these the context derives, once and for all estimators, both the
single-pair ``Innovation`` that gating and nearest-neighbour tests use, and
the *joint* innovation of several pairings at once -- including the
correlations between them through the shared pose -- which joint-compatibility
tests need. An EKF answers the primitives from its joint covariance, a
particle from its per-landmark EKFs, a factor graph from its marginals. One
associator therefore serves every backend.
"""
# src/slamkit/association/base.py

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Matrix, Vector
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.probability.covariance import Covariance
from slamkit.probability.gaussian import Innovation
from slamkit.sensing.observation import MeasurementLinearization, Observation

__all__ = [
    "Association",
    "AssociationContext",
    "DataAssociator",
    "Matched",
    "NewLandmark",
    "Pairing",
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


@dataclass(frozen=True, slots=True)
class Pairing:
    """A hypothesised correspondence: ``observation`` is of ``landmark_id``."""

    observation: Observation
    landmark_id: LandmarkId


class AssociationContext(ABC):
    """An estimator's current map, as seen by a data associator."""

    POSE_DOF = SE2.DOF
    LANDMARK_DOF = Point2.DOF

    @abstractmethod
    def landmark_ids(self) -> Collection[LandmarkId]: ...

    @abstractmethod
    def linearize(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> MeasurementLinearization:
        """Linearise ``observation`` as if it were of ``landmark_id``, at the
        current estimate."""

    @abstractmethod
    def covariance(self, landmark_ids: Sequence[LandmarkId]) -> Matrix:
        """Joint covariance of ``[pose, landmark₁, …, landmarkₖ]``, in that order."""

    def innovation(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> Innovation:
        """``z ⊖ ẑ`` and its covariance ``S`` if ``observation`` were of ``landmark_id``."""
        return self.joint_innovation([Pairing(observation, landmark_id)])

    def joint_innovation(self, pairings: Sequence[Pairing]) -> Innovation:
        """The stacked innovation of several pairings and its full covariance
        ``S = H Σ Hᵀ + R``, whose off-diagonal blocks carry the correlations
        induced by the shared pose and by shared landmarks."""
        if not pairings:
            raise ValueError("at least one pairing is required")
        landmarks = list(dict.fromkeys(p.landmark_id for p in pairings))
        column = {
            lid: self.POSE_DOF + i * self.LANDMARK_DOF
            for i, lid in enumerate(landmarks)
        }
        linearizations = [
            self.linearize(p.observation, p.landmark_id) for p in pairings
        ]
        rows = sum(lin.dim for lin in linearizations)
        jacobian = np.zeros((rows, self.POSE_DOF + self.LANDMARK_DOF * len(landmarks)))
        noise = np.zeros((rows, rows))
        row = 0
        for pairing, lin in zip(pairings, linearizations, strict=True):
            block = slice(row, row + lin.dim)
            start = column[pairing.landmark_id]
            jacobian[block, : self.POSE_DOF] = lin.wrt_pose
            jacobian[block, start : start + self.LANDMARK_DOF] = lin.wrt_landmark
            noise[block, block] = lin.noise_covariance
            row += lin.dim
        s = Covariance.propagate(Matrix(jacobian), self.covariance(landmarks)) + noise
        residual = Vector(np.concatenate([lin.innovation for lin in linearizations]))
        return Innovation(residual, Covariance.symmetrize(Matrix(s)))


class DataAssociator(ABC):
    """Maps a batch of simultaneous observations to associations, one per observation."""

    @abstractmethod
    def associate(
        self, observations: Sequence[Observation], context: AssociationContext
    ) -> list[Association]: ...
