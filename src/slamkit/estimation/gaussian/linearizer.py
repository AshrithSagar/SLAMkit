"""
Linearizers
===========
What actually distinguishes the EKF from the UKF.

A Gaussian filter needs, at each step, a *linear* model of the motion and of
each measurement: ``F, Q`` and ``H, R``. Everything after that -- covariance
propagation, Kalman gain, state augmentation -- is identical. So the two
filters differ in exactly one decision, captured here:

* ``AnalyticLinearizer``  : first-order Taylor expansion at the mean, using
                            the models' analytic Jacobians (the EKF);
* ``UnscentedLinearizer`` : statistical linear regression through sigma
                            points spread over the current uncertainty (the
                            UKF). The regression residual -- how non-linear
                            the model is over that spread -- is added to the
                            noise, so the filter is honest about it.

Feeding regression-derived ``(H, R + Ω)`` to the standard Kalman equations
reproduces the UKF update exactly, while only ever drawing sigma points over
the few variables a model actually touches (the pose for motion; the pose and
one landmark for a measurement) rather than the whole growing map.
"""
# src/slamkit/estimation/gaussian/linearizer.py

from abc import ABC, abstractmethod

import numpy as np

from slamkit.core.arrays import Mat3, Matrix, Vector
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.geometry.mean import ManifoldMean
from slamkit.geometry.point import Point2
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionLinearization, MotionModel
from slamkit.probability.covariance import Covariance
from slamkit.probability.unscented import SigmaPoints, UnscentedTransform
from slamkit.sensing.observation import MeasurementLinearization, Observation

__all__ = [
    "AnalyticLinearizer",
    "Linearizer",
    "UnscentedLinearizer",
]

_POSE = slice(0, SE2.DOF)
_LANDMARK = slice(SE2.DOF, SE2.DOF + Point2.DOF)


class Linearizer(ABC):
    """Produces the linear models a Gaussian filter consumes."""

    @abstractmethod
    def linearize_motion[C: Control](
        self,
        model: MotionModel[C],
        pose: SE2[WorldFrame, BodyFrame],
        pose_covariance: Mat3,
        control: C,
    ) -> MotionLinearization:
        """``(g(x, u), F, Q)`` for a pose belief ``N(pose, pose_covariance)``."""

    @abstractmethod
    def linearize_measurement(
        self,
        observation: Observation,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
        covariance: Matrix,
    ) -> MeasurementLinearization:
        """``(z ⊖ ẑ, H, R)`` for a joint ``(pose, landmark)`` belief with 5×5 ``covariance``."""


class AnalyticLinearizer(Linearizer):
    """Taylor expansion at the mean; the uncertainty is not consulted."""

    def linearize_motion[C: Control](
        self,
        model: MotionModel[C],
        pose: SE2[WorldFrame, BodyFrame],
        pose_covariance: Mat3,
        control: C,
    ) -> MotionLinearization:
        return model.linearize(pose, control)

    def linearize_measurement(
        self,
        observation: Observation,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
        covariance: Matrix,
    ) -> MeasurementLinearization:
        return observation.linearize(pose, landmark)


class UnscentedLinearizer(Linearizer):
    """Statistical linear regression through unscented sigma points.

    Only function *values* are evaluated (``predict`` and ``residual``), never
    Jacobians, so models need not be differentiable.
    """

    def __init__(
        self,
        transform: UnscentedTransform | None = None,
        mean: ManifoldMean | None = None,
    ) -> None:
        self._transform = transform if transform is not None else UnscentedTransform()
        self._mean = mean if mean is not None else ManifoldMean()

    def linearize_motion[C: Control](
        self,
        model: MotionModel[C],
        pose: SE2[WorldFrame, BodyFrame],
        pose_covariance: Mat3,
        control: C,
    ) -> MotionLinearization:
        points = self._transform.sigma_points(Matrix(pose_covariance))
        outputs = [
            model.predict(pose.oplus(offset), control) for offset in points.offsets
        ]
        predicted = self._mean(outputs, points.mean_weights)
        deviations = Matrix(np.array([y.ominus(predicted) for y in outputs]))
        jacobian, nonlinearity = self._regress(
            points, Matrix(pose_covariance), deviations
        )
        noise = Covariance.symmetrize(
            Matrix(model.increment_covariance(control) + nonlinearity)
        )
        return MotionLinearization(predicted, Mat3(jacobian), Mat3(noise))

    def linearize_measurement(
        self,
        observation: Observation,
        pose: SE2[WorldFrame, BodyFrame],
        landmark: Point2[WorldFrame],
        covariance: Matrix,
    ) -> MeasurementLinearization:
        points = self._transform.sigma_points(covariance)
        residuals = Matrix(
            np.array(
                [
                    observation.residual(
                        pose.oplus(Vector(offset[_POSE])),
                        landmark.oplus(Vector(offset[_LANDMARK])),
                    )
                    for offset in points.offsets
                ]
            )
        )
        innovation = points.weighted_mean(residuals)
        # z ⊖ h(xᵢ) - mean = -(h(xᵢ) - mean h): negate to regress h itself.
        deviations = Matrix(innovation - residuals)
        jacobian, nonlinearity = self._regress(points, covariance, deviations)
        return MeasurementLinearization(
            innovation=innovation,
            wrt_pose=Matrix(jacobian[:, _POSE]),
            wrt_landmark=Matrix(jacobian[:, _LANDMARK]),
            noise_covariance=Covariance.symmetrize(
                Matrix(observation.noise_covariance + nonlinearity)
            ),
        )

    @staticmethod
    def _regress(
        points: SigmaPoints, input_covariance: Matrix, output_deviations: Matrix
    ) -> tuple[Matrix, Matrix]:
        """Least-squares linear fit ``δy ≈ A δx`` over the sigma points.

        Returns ``A = P_yx P_xx⁻¹`` and the residual covariance
        ``Ω = P_yy - A P_xx Aᵀ`` (zero for an exactly linear model).
        """
        inputs = Matrix(np.array(points.offsets))
        cross = points.cross_covariance(inputs, output_deviations)
        output = points.cross_covariance(output_deviations, output_deviations)
        jacobian = Matrix((np.linalg.pinv(input_covariance, hermitian=True) @ cross).T)
        nonlinearity = Covariance.project_psd(
            Matrix(output - Covariance.propagate(jacobian, input_covariance))
        )
        return jacobian, nonlinearity
