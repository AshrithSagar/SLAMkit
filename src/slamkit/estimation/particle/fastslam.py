"""
FastSLAM
========
FastSLAM 1.0: a particle filter over trajectories, an EKF per landmark per
particle. Handles multi-modal pose beliefs and per-particle data association
(each particle associates against its own map), at ``O(M·n)`` per update.
"""
# src/slamkit/estimation/particle/fastslam.py

from collections.abc import Sequence

import numpy as np

from slamkit.association.base import DataAssociator
from slamkit.core.arrays import Mat3, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.estimation.estimator import SlamEstimate, SlamEstimator
from slamkit.estimation.particle.particle import Particle
from slamkit.estimation.particle.resampling import Resampler, SystematicResampler
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionModel
from slamkit.sensing.observation import Observation

__all__ = [
    "FastSlam",
]


class FastSlam[C: Control](SlamEstimator[C]):
    """FastSLAM 1.0, composed from a motion model, a data associator and a resampler.

    Resampling happens only when the effective sample size drops below
    ``resample_threshold · num_particles``, which limits particle depletion.
    """

    def __init__(
        self,
        motion_model: MotionModel[C],
        associator: DataAssociator,
        rng: np.random.Generator,
        num_particles: int = 100,
        resampler: Resampler | None = None,
        resample_threshold: float = 0.5,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
    ) -> None:
        if num_particles < 1:
            raise ValueError("num_particles must be at least 1")
        if not 0.0 <= resample_threshold <= 1.0:
            raise ValueError("resample_threshold must lie in [0, 1]")
        self._motion_model = motion_model
        self._associator = associator
        self._rng = rng
        self._resampler = resampler if resampler is not None else SystematicResampler()
        self._resample_threshold = resample_threshold
        pose = (
            initial_pose
            if initial_pose is not None
            else SE2[WorldFrame, BodyFrame].identity()
        )
        self._particles = [Particle(pose) for _ in range(num_particles)]

    @property
    def particles(self) -> Sequence[Particle]:
        return tuple(self._particles)

    def weights(self) -> Vector:
        """Normalised particle weights."""
        return Vector(np.exp(self._normalized_log_weights()))

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights())))

    # ── SlamEstimator ─────────────────────────────────────────────────────────

    def predict(self, control: C) -> None:
        for particle in self._particles:
            particle.move_to(
                self._motion_model.sample(particle.pose, control, self._rng)
            )

    def update(self, observations: Sequence[Observation]) -> None:
        if not observations:
            return
        for particle in self._particles:
            particle.assimilate(observations, self._associator)
        self._normalize()
        if self.effective_sample_size() < self._resample_threshold * len(
            self._particles
        ):
            self._resample()

    def estimate(self) -> SlamEstimate:
        """The highest-weight particle's pose and map, with the pose covariance of
        the whole (weighted) particle cloud around it."""
        weights = self.weights()
        best = self._particles[int(np.argmax(weights))]
        spread = np.array([p.pose.ominus(best.pose) for p in self._particles])
        covariance = (spread * weights[:, None]).T @ spread
        return SlamEstimate(
            pose=best.pose,
            landmarks={lid: belief.mean for lid, belief in best.landmarks.items()},
            pose_covariance=Mat3(covariance),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalized_log_weights(self) -> Vector:
        """``log wᵢ - log Σ w``, via log-sum-exp so tiny weights never underflow to ``log 0``."""
        log_weights = np.array([p.log_weight for p in self._particles])
        peak = float(log_weights.max())
        return Vector(log_weights - (peak + np.log(np.sum(np.exp(log_weights - peak)))))

    def _normalize(self) -> None:
        log_weights = self._normalized_log_weights()
        for particle, log_weight in zip(self._particles, log_weights, strict=True):
            particle.reweight(float(log_weight))

    def _resample(self) -> None:
        indices = self._resampler.indices(self.weights(), self._rng)
        uniform = -float(np.log(len(self._particles)))
        self._particles = [self._particles[i].copy() for i in indices]
        for particle in self._particles:
            particle.reweight(uniform)
