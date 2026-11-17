"""
EKF-SLAM state
==============
The joint Gaussian over ``[pose, landmark₁, landmark₂, ...]`` and its layout.

This class owns *bookkeeping only*: where each block lives in the covariance,
how the state grows, and how a tangent-space correction is applied. The
filter equations themselves live in ``EkfSlam``.

The pose is kept on SE(2) and the covariance in its (right) tangent space, so
this is an error-state EKF: corrections are retracted with ``⊕`` rather than
added to an ``[x, y, θ]`` vector.
"""
# src/slamkit/estimation/ekf/state.py

from collections.abc import Collection, Mapping

import numpy as np

from slamkit.core.arrays import Mat2, Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.motion.model import MotionLinearization
from slamkit.probability.covariance import Covariance
from slamkit.sensing.observation import LandmarkInitialization, MeasurementLinearization

__all__ = [
    "EkfSlamState",
]


class EkfSlamState:
    """Mean (pose on SE(2), landmarks in the plane) and joint covariance."""

    POSE_DOF = SE2.DOF
    LANDMARK_DOF = Point2.DOF
    POSE = slice(0, POSE_DOF)

    def __init__(self, pose: SE2[WorldFrame, BodyFrame], pose_covariance: Mat3) -> None:
        Covariance.validate(pose_covariance, self.POSE_DOF)
        self._pose: SE2[WorldFrame, BodyFrame] = pose
        self._landmarks: dict[LandmarkId, Point2[WorldFrame]] = {}
        self._offsets: dict[LandmarkId, int] = {}
        self._covariance: Matrix = Matrix(pose_covariance, dtype=np.float64)

    # ── Read access ───────────────────────────────────────────────────────────

    @property
    def pose(self) -> SE2[WorldFrame, BodyFrame]:
        return self._pose

    @property
    def pose_covariance(self) -> Mat3:
        return Mat3(self._covariance[self.POSE, self.POSE])

    @property
    def covariance(self) -> Matrix:
        return self._covariance

    @property
    def dim(self) -> int:
        return self._covariance.shape[0]

    @property
    def landmarks(self) -> Mapping[LandmarkId, Point2[WorldFrame]]:
        return dict(self._landmarks)

    def landmark_ids(self) -> Collection[LandmarkId]:
        return self._landmarks.keys()

    def landmark(self, landmark_id: LandmarkId) -> Point2[WorldFrame]:
        return self._landmarks[landmark_id]

    def landmark_covariance(self, landmark_id: LandmarkId) -> Mat2:
        block = self.block(landmark_id)
        return Mat2(self._covariance[block, block])

    def block(self, landmark_id: LandmarkId) -> slice:
        offset = self._offsets[landmark_id]
        return slice(offset, offset + self.LANDMARK_DOF)

    def pose_landmark_covariance(self, landmark_id: LandmarkId) -> Matrix:
        """The 5×5 marginal covariance of ``(pose, landmark)``."""
        offset = self._offsets[landmark_id]
        indices = np.concatenate(
            (np.arange(self.POSE_DOF), np.arange(offset, offset + self.LANDMARK_DOF))
        )
        return Matrix(self._covariance[np.ix_(indices, indices)])

    def embed(
        self, linearization: MeasurementLinearization, landmark_id: LandmarkId
    ) -> Matrix:
        """Scatter an observation's pose/landmark Jacobians into a full-width ``H``."""
        jacobian = np.zeros((linearization.innovation.shape[0], self.dim))
        jacobian[:, self.POSE] = linearization.wrt_pose
        jacobian[:, self.block(landmark_id)] = linearization.wrt_landmark
        return Matrix(jacobian)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def propagate(self, motion: MotionLinearization) -> None:
        """``x ← g(x, u)``; ``P_xx ← F P_xx Fᵀ + Q``, ``P_xm ← F P_xm``. Landmarks are static."""
        f, p = motion.wrt_pose, self._covariance
        p[self.POSE, :] = f @ p[self.POSE, :]
        p[:, self.POSE] = p[:, self.POSE] @ f.T
        p[self.POSE, self.POSE] += motion.noise_covariance
        self._covariance = Covariance.symmetrize(p)
        self._pose = motion.predicted

    def augment(
        self, landmark_id: LandmarkId, initialization: LandmarkInitialization
    ) -> None:
        """Append a landmark, correlated with everything through the current pose."""
        if landmark_id in self._landmarks:
            raise KeyError(f"landmark {landmark_id} is already in the state")
        n = self.dim
        jx = initialization.wrt_pose
        cross = jx @ self._covariance[self.POSE, :]
        own = initialization.covariance(self.pose_covariance)
        grown = np.zeros((n + self.LANDMARK_DOF, n + self.LANDMARK_DOF))
        grown[:n, :n] = self._covariance
        grown[n:, :n] = cross
        grown[:n, n:] = cross.T
        grown[n:, n:] = own
        self._covariance = Covariance.symmetrize(Matrix(grown))
        self._landmarks[landmark_id] = initialization.position
        self._offsets[landmark_id] = n

    def correct(self, delta: Vector, covariance: Matrix) -> None:
        """Retract a tangent-space correction and replace the covariance."""
        if delta.shape != (self.dim,) or covariance.shape != (self.dim, self.dim):
            raise ValueError("correction does not match the state dimension")
        self._pose = self._pose.oplus(Vector(delta[self.POSE]))
        for landmark_id, landmark in self._landmarks.items():
            self._landmarks[landmark_id] = landmark.oplus(
                Vector(delta[self.block(landmark_id)])
            )
        self._covariance = Covariance.symmetrize(covariance)
