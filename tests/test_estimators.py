"""End-to-end: every estimator, assembled from the same parts, beats dead reckoning."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from slamkit.association import DataAssociator, KnownCorrespondence, NearestNeighbor
from slamkit.core import Vector
from slamkit.estimation import EkfSlam, FastSlam, GraphSlam, SlamEstimator
from slamkit.estimation.graph import LandmarkKey, PoseKey, Values
from slamkit.estimation.particle import (
    MultinomialResampler,
    Resampler,
    SystematicResampler,
)
from slamkit.evaluation import Evaluator
from slamkit.motion import MotionModel, Odometry, OdometryMotionModel
from slamkit.pipeline import Dataset, SlamRunner
from slamkit.sensing import PointSensor, RangeBearingModel, SensingRange
from slamkit.simulation import CircularPath, LandmarkWorld, Simulator

MOTION = OdometryMotionModel(
    translation_noise=0.05, rotation_noise=0.05, drift_per_meter=0.02
)
SENSOR = PointSensor(
    "rb",
    RangeBearingModel(0.1, math.radians(1.0)),
    sensing_range=SensingRange(max_range=10.0),
)
WORLD = LandmarkWorld.ring(10, radius=9.0)
PATH = CircularPath(radius=6.0, steps_per_loop=24, loops=2)

type Factory = Callable[
    [MotionModel[Odometry], DataAssociator], SlamEstimator[Odometry]
]

ESTIMATORS: dict[str, Factory] = {
    "ekf": lambda m, a: EkfSlam(m, a),
    "fastslam": lambda m, a: FastSlam(m, a, np.random.default_rng(1), num_particles=30),
    "graph": lambda m, a: GraphSlam(m, a, optimize_every=8),
}


# FastSLAM 1.0 samples poses from the motion model alone, so with accurate sensors
# its particles deplete quickly; it is noticeably noisier than the Gaussian backends.
ATE_BOUND = {"ekf": 0.5, "fastslam": 1.0, "graph": 0.5}


def simulate(tagged: bool) -> Dataset[Odometry]:
    return Simulator(
        WORLD, MOTION, [SENSOR], np.random.default_rng(3), tag_observations=tagged
    ).run(PATH.odometry())


def dead_reckoning_error(dataset: Dataset[Odometry]) -> float:
    assert dataset.ground_truth is not None
    pose = dataset.initial_pose
    for step in dataset:
        pose = MOTION.predict(pose, step.control)
    true = dataset.ground_truth.trajectory[-1]
    return math.hypot(pose.x - true.x, pose.y - true.y)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_known_correspondence(name: str) -> None:
    dataset = simulate(tagged=True)
    assert dataset.ground_truth is not None
    result = SlamRunner(ESTIMATORS[name](MOTION, KnownCorrespondence())).run(dataset)
    report = Evaluator().evaluate(result, dataset.ground_truth)
    assert report.landmarks_matched == len(WORLD)
    assert report.final_position_error < dead_reckoning_error(dataset)
    assert report.position_rmse < ATE_BOUND[name]
    assert report.landmark_rmse < ATE_BOUND[name]


@pytest.mark.parametrize("name", ["ekf", "graph"])
def test_nearest_neighbor(name: str) -> None:
    dataset = simulate(tagged=False)
    assert dataset.ground_truth is not None
    result = SlamRunner(
        ESTIMATORS[name](MOTION, NearestNeighbor()), runtime_checks=False
    ).run(dataset)
    report = Evaluator(match_radius=1.0).evaluate(result, dataset.ground_truth)
    assert report.landmarks_matched >= len(WORLD) - 1
    assert report.position_rmse < 0.5


def test_graph_trajectory_has_one_pose_per_step() -> None:
    dataset = simulate(tagged=True)
    result = SlamRunner(
        GraphSlam(MOTION, KnownCorrespondence()), runtime_checks=False
    ).run(dataset)
    assert len(result.trajectory) == len(dataset) + 1


def test_graph_noise_models_are_statistically_consistent() -> None:
    """At ground truth, ½‖r‖² over all factors should be ≈ ½·(residual dimensions),
    and the optimum must not be worse than ground truth."""
    dataset = simulate(tagged=True)
    assert dataset.ground_truth is not None
    slam = GraphSlam(MOTION, KnownCorrespondence(), optimize_every=0)
    SlamRunner(slam, runtime_checks=False).run(dataset)
    truth = Values()
    for index, pose in enumerate(dataset.ground_truth.trajectory):
        truth.set_pose(PoseKey(index), pose)
    for landmark_id, point in dataset.ground_truth.landmarks.items():
        truth.set_landmark(LandmarkKey(landmark_id), point)
    dimensions = sum(factor.noise.dim for factor in slam.graph)
    error_at_truth = slam.graph.error(truth)
    assert 0.3 * dimensions < error_at_truth < 0.7 * dimensions
    assert slam.optimize().final_error <= error_at_truth


@pytest.mark.parametrize(
    "resampler",
    [SystematicResampler(), MultinomialResampler()],
    ids=["systematic", "multinomial"],
)
def test_resampler_draws_proportionally(resampler: Resampler) -> None:
    weights = Vector([0.1, 0.0, 0.6, 0.3])
    rng = np.random.default_rng(0)
    draws = [i for _ in range(5000) for i in resampler.indices(weights, rng)]
    frequencies = np.bincount(draws, minlength=4) / len(draws)
    assert frequencies[1] == 0.0
    assert np.allclose(frequencies, weights, atol=0.02)


def test_systematic_resampling_is_low_variance() -> None:
    """Each particle is copied ⌊N·w⌋ or ⌈N·w⌉ times -- never more, never fewer."""
    rng = np.random.default_rng(0)
    weights = Vector(rng.dirichlet(np.ones(200)))
    counts = np.bincount(SystematicResampler().indices(weights, rng), minlength=200)
    assert np.all(np.abs(counts - 200 * weights) < 1.0)
