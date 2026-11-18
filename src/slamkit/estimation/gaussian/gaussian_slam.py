"""
Gaussian SLAM
=============
One joint Gaussian over the pose and every landmark, updated by an
(error-state) Kalman filter. Exact up to linearisation, ``O(n²)`` per update
in the number of landmarks.

The filter equations are written once here; *how* the motion and measurement
models are linearised is a pluggable ``Linearizer``. ``EkfSlam`` and
``UkfSlam`` are this class with the analytic and the unscented linearizer.
"""
# src/slamkit/estimation/gaussian/gaussian_slam.py

from collections.abc import Collection, Sequence

import numpy as np

from slamkit.association.base import DataAssociator
from slamkit.core.arrays import Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.estimation.estimator import MapHypothesis, SlamEstimate, SlamEstimator
from slamkit.estimation.gaussian.linearizer import (
    AnalyticLinearizer,
    Linearizer,
    UnscentedLinearizer,
)
from slamkit.estimation.gaussian.state import GaussianSlamState
from slamkit.geometry.se2 import SE2
from slamkit.initialization.base import LandmarkInitializer, Sighting
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionModel
from slamkit.probability.covariance import Covariance
from slamkit.probability.unscented import UnscentedTransform
from slamkit.sensing.observation import (
    LandmarkInitialization,
    MeasurementLinearization,
    Observation,
)

__all__ = [
    "EkfSlam",
    "GaussianSlam",
    "UkfSlam",
]


class GaussianSlam[C: Control](SlamEstimator[C], MapHypothesis[None]):
    """Kalman-filter SLAM, composed from a motion model, a data associator, a
    linearizer and a landmark initializer."""

    def __init__(
        self,
        motion_model: MotionModel[C],
        associator: DataAssociator,
        linearizer: Linearizer,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
        initial_covariance: Mat3 | None = None,
        initializer: LandmarkInitializer | None = None,
    ) -> None:
        super().__init__(initializer)
        self._motion_model = motion_model
        self._associator = associator
        self._linearizer = linearizer
        self._state = GaussianSlamState(
            initial_pose
            if initial_pose is not None
            else SE2[WorldFrame, BodyFrame].identity(),
            initial_covariance
            if initial_covariance is not None
            else Mat3(np.eye(3) * 1e-9),
        )

    @property
    def state(self) -> GaussianSlamState:
        return self._state

    # ── SlamEstimator ─────────────────────────────────────────────────────────

    def predict(self, control: C) -> None:
        state = self._state
        self._state.propagate(
            self._linearizer.linearize_motion(
                self._motion_model, state.pose, state.pose_covariance, control
            )
        )

    def update(self, observations: Sequence[Observation]) -> None:
        self.assimilate(observations, self._associator)

    def estimate(self) -> SlamEstimate:
        return SlamEstimate(
            self._state.pose, self._state.landmarks, self._state.pose_covariance
        )

    # ── MapHypothesis ─────────────────────────────────────────────────────────

    @property
    def anchor(self) -> None:
        return None

    @property
    def current_pose_covariance(self) -> Mat3:
        return self._state.pose_covariance

    @property
    def current_pose(self) -> SE2[WorldFrame, BodyFrame]:
        return self._state.pose

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._state.landmark_ids()

    def linearize(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> MeasurementLinearization:
        return self._linearizer.linearize_measurement(
            observation,
            self._state.pose,
            self._state.landmark(landmark_id),
            self._state.joint_covariance([landmark_id]),
        )

    def covariance(self, landmark_ids: Sequence[LandmarkId]) -> Matrix:
        return self._state.joint_covariance(landmark_ids)

    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        """The Kalman update, with the Joseph-form covariance for numerical robustness."""
        state = self._state
        linearization = self.linearize(observation, landmark_id)
        h = state.embed(linearization, landmark_id)
        r = linearization.noise_covariance
        p = state.covariance
        s = Covariance.propagate(h, p) + r
        gain = np.linalg.solve(s, h @ p).T
        i_kh = np.eye(state.dim) - gain @ h
        corrected = i_kh @ p @ i_kh.T + gain @ r @ gain.T
        state.correct(Vector(gain @ linearization.innovation), Matrix(corrected))

    def add_landmark(
        self,
        landmark_id: LandmarkId,
        initialization: LandmarkInitialization,
        sightings: Sequence[Sighting[None]],
    ) -> None:
        self._state.augment(landmark_id, initialization)


class EkfSlam[C: Control](GaussianSlam[C]):
    """EKF-SLAM: Gaussian SLAM linearised by analytic Jacobians at the mean."""

    def __init__(
        self,
        motion_model: MotionModel[C],
        associator: DataAssociator,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
        initial_covariance: Mat3 | None = None,
        initializer: LandmarkInitializer | None = None,
    ) -> None:
        super().__init__(
            motion_model,
            associator,
            AnalyticLinearizer(),
            initial_pose,
            initial_covariance,
            initializer,
        )


class UkfSlam[C: Control](GaussianSlam[C]):
    """UKF-SLAM: Gaussian SLAM linearised by unscented statistical regression.

    Needs no Jacobians, and accounts for model curvature over the current
    uncertainty -- most visible when headings are uncertain.
    """

    def __init__(
        self,
        motion_model: MotionModel[C],
        associator: DataAssociator,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
        initial_covariance: Mat3 | None = None,
        initializer: LandmarkInitializer | None = None,
        transform: UnscentedTransform | None = None,
    ) -> None:
        super().__init__(
            motion_model,
            associator,
            UnscentedLinearizer(transform),
            initial_pose,
            initial_covariance,
            initializer,
        )
