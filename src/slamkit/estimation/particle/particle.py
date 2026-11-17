"""
Particles
=========
A FastSLAM particle: one sampled trajectory hypothesis, carrying its own map.

Conditioned on a known trajectory, landmarks are independent (the
Rao-Blackwellisation behind FastSLAM), so each particle keeps a small,
separate 2×2 EKF per landmark instead of one joint covariance.
"""
# src/slamkit/estimation/particle/particle.py

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Mat2, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.estimation.estimator import MapHypothesis
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.probability.covariance import Covariance
from slamkit.probability.gaussian import Innovation
from slamkit.sensing.observation import MeasurementLinearization, Observation

__all__ = [
    "LandmarkBelief",
    "Particle",
]


@dataclass(frozen=True, slots=True)
class LandmarkBelief:
    """A landmark's Gaussian, conditioned on the owning particle's trajectory."""

    mean: Point2[WorldFrame]
    covariance: Mat2


class Particle(MapHypothesis):
    """A pose sample, its log-weight, and its own landmark map.

    Landmark beliefs are immutable, so copying a particle on resampling is a
    shallow dictionary copy.
    """

    _POSE_DOF = SE2.DOF

    def __init__(
        self,
        pose: SE2[WorldFrame, BodyFrame],
        log_weight: float = 0.0,
        landmarks: Mapping[LandmarkId, LandmarkBelief] | None = None,
    ) -> None:
        self._pose = pose
        self._log_weight = log_weight
        self._landmarks: dict[LandmarkId, LandmarkBelief] = (
            dict(landmarks) if landmarks else {}
        )

    @property
    def pose(self) -> SE2[WorldFrame, BodyFrame]:
        return self._pose

    @property
    def log_weight(self) -> float:
        return self._log_weight

    @property
    def landmarks(self) -> Mapping[LandmarkId, LandmarkBelief]:
        return self._landmarks

    def move_to(self, pose: SE2[WorldFrame, BodyFrame]) -> None:
        self._pose = pose

    def reweight(self, log_weight: float) -> None:
        self._log_weight = log_weight

    def copy(self) -> Particle:
        return Particle(self._pose, self._log_weight, self._landmarks)

    # ── MapHypothesis ─────────────────────────────────────────────────────────

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._landmarks.keys()

    def innovation(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> Innovation:
        linearization, belief = self._linearize(observation, landmark_id)
        return self._innovation(linearization, belief)

    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """Per-landmark EKF update; the particle is reweighted by the innovation likelihood."""
        linearization, belief = self._linearize(observation, landmark_id)
        innovation = self._innovation(linearization, belief)
        h, r, p = (
            linearization.wrt_landmark,
            linearization.noise_covariance,
            belief.covariance,
        )
        gain = np.linalg.solve(innovation.covariance, h @ p).T
        i_kh = np.eye(Point2.DOF) - gain @ h
        covariance = Covariance.symmetrize(
            Matrix(i_kh @ p @ i_kh.T + gain @ r @ gain.T)
        )
        mean = belief.mean.oplus(Vector(gain @ innovation.residual))
        self._landmarks[landmark_id] = LandmarkBelief(mean, Mat2(covariance))
        self._log_weight += innovation.log_likelihood()

    def add_landmark(self, observation: Observation, landmark_id: LandmarkId) -> None:
        initialization = observation.initialize_landmark(self._pose)
        self._landmarks[landmark_id] = LandmarkBelief(
            initialization.position, initialization.covariance()
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _linearize(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> tuple[MeasurementLinearization, LandmarkBelief]:
        belief = self._landmarks[landmark_id]
        return observation.linearize(self._pose, belief.mean), belief

    def _innovation(
        self, linearization: MeasurementLinearization, belief: LandmarkBelief
    ) -> Innovation:
        # The particle's pose is a sample, i.e. known: only the landmark block is uncertain.
        joint = np.zeros((self._POSE_DOF + Point2.DOF,) * 2)
        joint[self._POSE_DOF :, self._POSE_DOF :] = belief.covariance
        return linearization.innovation_given(Matrix(joint))
