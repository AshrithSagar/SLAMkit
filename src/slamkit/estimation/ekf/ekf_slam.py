"""
EKF-SLAM
========
One joint Gaussian over the pose and every landmark, updated by an
(error-state) extended Kalman filter. Exact up to linearisation, ``O(n²)`` per
update in the number of landmarks.
"""
# src/slamkit/estimation/ekf/ekf_slam.py

from collections.abc import Collection, Sequence

import numpy as np

from slamkit.association.base import DataAssociator
from slamkit.core.arrays import Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.estimation.ekf.state import EkfSlamState
from slamkit.estimation.estimator import MapHypothesis, SlamEstimate, SlamEstimator
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionModel
from slamkit.probability.covariance import Covariance
from slamkit.probability.gaussian import Innovation
from slamkit.sensing.observation import Observation

__all__ = [
    "EkfSlam",
]


class EkfSlam[C: Control](SlamEstimator[C], MapHypothesis):
    """EKF-SLAM, composed from a motion model and a data associator."""

    def __init__(
        self,
        motion_model: MotionModel[C],
        associator: DataAssociator,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
        initial_covariance: Mat3 | None = None,
    ) -> None:
        self._motion_model = motion_model
        self._associator = associator
        self._state = EkfSlamState(
            initial_pose
            if initial_pose is not None
            else SE2[WorldFrame, BodyFrame].identity(),
            initial_covariance
            if initial_covariance is not None
            else Mat3(np.eye(3) * 1e-9),
        )

    @property
    def state(self) -> EkfSlamState:
        return self._state

    # ── SlamEstimator ─────────────────────────────────────────────────────────

    def predict(self, control: C) -> None:
        self._state.propagate(self._motion_model.linearize(self._state.pose, control))

    def update(self, observations: Sequence[Observation]) -> None:
        self.assimilate(observations, self._associator)

    def estimate(self) -> SlamEstimate:
        return SlamEstimate(
            self._state.pose, self._state.landmarks, self._state.pose_covariance
        )

    # ── MapHypothesis ─────────────────────────────────────────────────────────

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._state.landmark_ids()

    def innovation(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> Innovation:
        linearization = observation.linearize(
            self._state.pose, self._state.landmark(landmark_id)
        )
        return linearization.innovation_given(
            self._state.pose_landmark_covariance(landmark_id)
        )

    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """The Kalman update, with the Joseph-form covariance for numerical robustness."""
        state = self._state
        linearization = observation.linearize(state.pose, state.landmark(landmark_id))
        h = state.embed(linearization, landmark_id)
        r = linearization.noise_covariance
        p = state.covariance
        s = Covariance.propagate(h, p) + r
        gain = np.linalg.solve(s, h @ p).T
        i_kh = np.eye(state.dim) - gain @ h
        corrected = i_kh @ p @ i_kh.T + gain @ r @ gain.T
        state.correct(Vector(gain @ linearization.innovation), Matrix(corrected))

    def add_landmark(self, observation: Observation, landmark_id: LandmarkId) -> None:
        self._state.augment(
            landmark_id, observation.initialize_landmark(self._state.pose)
        )
