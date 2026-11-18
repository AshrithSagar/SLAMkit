"""
Particles
=========
A FastSLAM particle: one trajectory hypothesis, carrying its own map.

Conditioned on a known trajectory, landmarks are independent (the
Rao-Blackwellisation behind FastSLAM), so each particle keeps a small,
separate 2×2 EKF per landmark instead of one joint covariance.

A particle's pose is normally a *sample* (known exactly). Between a motion
prediction and its sampling it may instead carry a Gaussian pose prior, which
observations refine before the pose is drawn -- the FastSLAM 2.0 proposal.
"""
# src/slamkit/estimation/particle/particle.py

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from slamkit.core.arrays import Mat2, Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.estimation.estimator import MapHypothesis
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.initialization.base import LandmarkInitializer, Sighting
from slamkit.motion.model import MotionLinearization
from slamkit.probability.covariance import Covariance
from slamkit.probability.gaussian import Gaussian, Innovation
from slamkit.sensing.observation import (
    LandmarkInitialization,
    MeasurementLinearization,
    Observation,
)

__all__ = [
    "LandmarkBelief",
    "Particle",
]


@dataclass(frozen=True, slots=True)
class LandmarkBelief:
    """A landmark's Gaussian, conditioned on the owning particle's trajectory."""

    mean: Point2[WorldFrame]
    covariance: Mat2


class Particle(MapHypothesis[None]):
    """A pose (sample or Gaussian prior), its log-weight, and its own landmark map.

    Landmark beliefs are immutable, so copying a particle on resampling is a
    shallow dictionary copy.
    """

    def __init__(
        self,
        pose: SE2[WorldFrame, BodyFrame],
        log_weight: float = 0.0,
        landmarks: Mapping[LandmarkId, LandmarkBelief] | None = None,
        pose_covariance: Mat3 | None = None,
        initializer: LandmarkInitializer | None = None,
    ) -> None:
        super().__init__(initializer)
        self._pose = pose
        self._pose_covariance = (
            pose_covariance if pose_covariance is not None else Mat3(np.zeros((3, 3)))
        )
        self._log_weight = log_weight
        self._landmarks: dict[LandmarkId, LandmarkBelief] = (
            dict(landmarks) if landmarks else {}
        )

    @property
    def pose(self) -> SE2[WorldFrame, BodyFrame]:
        return self._pose

    @property
    def pose_covariance(self) -> Mat3:
        """Zero once the pose has been sampled."""
        return self._pose_covariance

    @property
    def log_weight(self) -> float:
        return self._log_weight

    @property
    def landmarks(self) -> Mapping[LandmarkId, LandmarkBelief]:
        return self._landmarks

    def move_to(self, pose: SE2[WorldFrame, BodyFrame]) -> None:
        """Set the pose to a known sample."""
        self._pose = pose
        self._pose_covariance = Mat3(np.zeros((3, 3)))

    def reweight(self, log_weight: float) -> None:
        self._log_weight = log_weight

    def copy(self) -> Particle:
        twin = Particle(
            self._pose,
            self._log_weight,
            self._landmarks,
            self._pose_covariance,
            self._initializer,
        )
        twin._inherit_pending(self)
        return twin

    # ── Pose proposal (FastSLAM 2.0) ──────────────────────────────────────────

    def predict_pose(self, motion: MotionLinearization) -> None:
        """Replace the pose by a Gaussian prior ``N(g(x, u), F Σ Fᵀ + Q)``."""
        f = Matrix(motion.wrt_pose)
        self._pose = motion.predicted
        self._pose_covariance = Mat3(
            Covariance.propagate(f, Matrix(self._pose_covariance))
            + motion.noise_covariance
        )

    def condition_pose(self, observation: Observation, landmark_id: LandmarkId) -> None:
        """Kalman-update the pose prior with one observation of a mapped landmark,
        the landmark's uncertainty folded into the measurement noise; the
        particle is reweighted by the observation's predictive likelihood."""
        linearization, belief = self._linearize(observation, landmark_id)
        hx, sigma = linearization.wrt_pose, Matrix(self._pose_covariance)
        noise = Matrix(
            linearization.noise_covariance
            + Covariance.propagate(
                linearization.wrt_landmark, Matrix(belief.covariance)
            )
        )
        s = Covariance.symmetrize(Matrix(Covariance.propagate(hx, sigma) + noise))
        gain = np.linalg.solve(s, hx @ sigma).T
        i_kh = np.eye(self.POSE_DOF) - gain @ hx
        self._pose = self._pose.oplus(Vector(gain @ linearization.innovation))
        self._pose_covariance = Mat3(
            Covariance.symmetrize(Matrix(i_kh @ sigma @ i_kh.T + gain @ noise @ gain.T))
        )
        self._log_weight += Innovation(linearization.innovation, s).log_likelihood()

    def sample_pose(self, rng: np.random.Generator) -> None:
        """Draw the pose from its Gaussian prior, making it known."""
        if np.any(self._pose_covariance):
            self.move_to(
                self._pose.oplus(
                    Gaussian.zero_mean(Matrix(self._pose_covariance)).sample(rng)
                )
            )

    # ── MapHypothesis ─────────────────────────────────────────────────────────

    @property
    def anchor(self) -> None:
        return None

    @property
    def current_pose_covariance(self) -> Mat3:
        return self._pose_covariance

    @property
    def current_pose(self) -> SE2[WorldFrame, BodyFrame]:
        return self._pose

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._landmarks.keys()

    def linearize(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> MeasurementLinearization:
        return self._linearize(observation, landmark_id)[0]

    def covariance(self, landmark_ids: Sequence[LandmarkId]) -> Matrix:
        """Block-diagonal: given the trajectory, the pose and landmarks are independent."""
        n = self.POSE_DOF + self.LANDMARK_DOF * len(landmark_ids)
        joint = np.zeros((n, n))
        joint[: self.POSE_DOF, : self.POSE_DOF] = self._pose_covariance
        for i, landmark_id in enumerate(landmark_ids):
            block = slice(
                self.POSE_DOF + i * self.LANDMARK_DOF,
                self.POSE_DOF + (i + 1) * self.LANDMARK_DOF,
            )
            joint[block, block] = self._landmarks[landmark_id].covariance
        return Matrix(joint)

    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """FastSLAM 1.0: reweight by the innovation likelihood at the (known)
        pose, then update the landmark."""
        self._log_weight += self.innovation(observation, landmark_id).log_likelihood()
        self.correct_landmark(observation, landmark_id)

    def correct_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """Per-landmark EKF update at the current (known) pose; no reweighting."""
        linearization, belief = self._linearize(observation, landmark_id)
        h, r, p = (
            linearization.wrt_landmark,
            linearization.noise_covariance,
            belief.covariance,
        )
        s = Covariance.propagate(h, Matrix(p)) + r
        gain = np.linalg.solve(s, h @ p).T
        i_kh = np.eye(self.LANDMARK_DOF) - gain @ h
        covariance = Covariance.symmetrize(
            Matrix(i_kh @ p @ i_kh.T + gain @ r @ gain.T)
        )
        mean = belief.mean.oplus(Vector(gain @ linearization.innovation))
        self._landmarks[landmark_id] = LandmarkBelief(mean, Mat2(covariance))

    def add_landmark(
        self,
        landmark_id: LandmarkId,
        initialization: LandmarkInitialization,
        sightings: Sequence[Sighting[None]],
    ) -> None:
        self._landmarks[landmark_id] = LandmarkBelief(
            initialization.position, initialization.covariance()
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _linearize(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> tuple[MeasurementLinearization, LandmarkBelief]:
        belief = self._landmarks[landmark_id]
        return observation.linearize(self._pose, belief.mean), belief
