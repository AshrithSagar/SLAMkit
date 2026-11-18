import numpy as np
import pytest

from slamkit.core import BodyFrame, LandmarkId, Mat3, Matrix, Vector, WorldFrame
from slamkit.estimation.gaussian import AnalyticLinearizer, UnscentedLinearizer
from slamkit.geometry import SE2, ManifoldMean, Point2
from slamkit.motion import Odometry, OdometryMotionModel
from slamkit.probability import Covariance, UnscentedTransform
from slamkit.sensing import (
    CartesianModel,
    PointObservation,
    PointSensor,
    RangeBearingModel,
)

POSE = SE2[WorldFrame, BodyFrame](1.0, -2.0, 0.7)
LANDMARK = Point2[WorldFrame](4.0, 1.5)


class TestUnscentedTransform:
    @pytest.mark.parametrize(
        "transform", [UnscentedTransform(), UnscentedTransform(0.5, 2.0, 1.0)]
    )
    def test_sigma_points_reproduce_the_moments(
        self, transform: UnscentedTransform
    ) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 4))
        covariance = Matrix(a @ a.T + np.eye(4))
        points = transform.sigma_points(covariance)
        offsets = Matrix(np.array(points.offsets))
        assert len(points) == 9
        assert np.allclose(points.weighted_mean(offsets), 0.0)
        assert np.allclose(points.cross_covariance(offsets, offsets), covariance)

    def test_semidefinite_covariance_is_accepted(self) -> None:
        points = UnscentedTransform().sigma_points(Matrix(np.diag([1.0, 0.0])))
        assert np.allclose(np.array(points.offsets)[:, 1], 0.0)


class TestManifoldMean:
    def test_symmetric_poses_average_to_the_centre(self) -> None:
        deltas = [Vector([0.3, 0.0, 0.5]), Vector([-0.3, 0.0, -0.5])]
        points = [POSE.oplus(d) for d in deltas]
        mean = ManifoldMean()(points, Vector([0.5, 0.5]))
        assert np.allclose(mean.ominus(POSE), 0.0, atol=1e-9)

    def test_headings_average_across_the_wrap(self) -> None:
        a = SE2[WorldFrame, BodyFrame](0.0, 0.0, np.pi - 0.1)
        b = SE2[WorldFrame, BodyFrame](0.0, 0.0, -np.pi + 0.1)
        mean = ManifoldMean()([a, b], Vector([0.5, 0.5]))
        assert abs(abs(mean.theta) - np.pi) < 1e-9


class TestUnscentedLinearizer:
    """Statistical linearisation tends to the analytic Jacobian as the spread
    shrinks, and is exact (no extra noise) for a linear model."""

    analytic = AnalyticLinearizer()
    unscented = UnscentedLinearizer()

    def test_motion_matches_analytic_for_small_uncertainty(self) -> None:
        model, control = OdometryMotionModel(), Odometry(0.8, 0.1, 0.3)
        small = Mat3(np.eye(3) * 1e-8)
        expected = self.analytic.linearize_motion(model, POSE, small, control)
        actual = self.unscented.linearize_motion(model, POSE, small, control)
        assert np.allclose(actual.predicted.ominus(expected.predicted), 0.0, atol=1e-7)
        assert np.allclose(actual.wrt_pose, expected.wrt_pose, atol=1e-5)
        assert np.allclose(
            actual.noise_covariance, expected.noise_covariance, atol=1e-9
        )

    def test_measurement_matches_analytic_for_small_uncertainty(self) -> None:
        sensor = PointSensor("rb", RangeBearingModel(0.1, 0.02))
        observation = PointObservation(
            sensor, sensor.predict(POSE, Point2[WorldFrame](4.1, 1.4)).measurement
        )
        small = Matrix(np.eye(5) * 1e-8)
        expected = self.analytic.linearize_measurement(
            observation, POSE, LANDMARK, small
        )
        actual = self.unscented.linearize_measurement(
            observation, POSE, LANDMARK, small
        )
        assert np.allclose(actual.innovation, expected.innovation, atol=1e-6)
        assert np.allclose(actual.wrt_pose, expected.wrt_pose, atol=1e-4)
        assert np.allclose(actual.wrt_landmark, expected.wrt_landmark, atol=1e-4)

    def test_linear_measurement_adds_no_noise(self) -> None:
        sensor = PointSensor("cart", CartesianModel(0.1, 0.1))
        observation = PointObservation(
            sensor, sensor.predict(POSE, LANDMARK).measurement, LandmarkId(0)
        )
        landmark_only = Matrix(np.zeros((5, 5)))
        landmark_only[3:, 3:] = Covariance.diagonal(0.5, 0.5)
        actual = self.unscented.linearize_measurement(
            observation, POSE, LANDMARK, landmark_only
        )
        expected = self.analytic.linearize_measurement(
            observation, POSE, LANDMARK, landmark_only
        )
        assert np.allclose(actual.wrt_landmark, expected.wrt_landmark, atol=1e-9)
        assert np.allclose(actual.noise_covariance, sensor.model.covariance, atol=1e-12)

    def test_curvature_inflates_noise(self) -> None:
        sensor = PointSensor("rb", RangeBearingModel(0.1, 0.02))
        observation = PointObservation(
            sensor, sensor.predict(POSE, LANDMARK).measurement
        )
        wide = Matrix(np.diag([1.0, 1.0, 0.3, 1.0, 1.0]))
        actual = self.unscented.linearize_measurement(observation, POSE, LANDMARK, wide)
        extra = actual.noise_covariance - sensor.model.covariance
        assert np.all(np.linalg.eigvalsh(extra) >= -1e-12)
        assert np.trace(extra) > 1e-4
