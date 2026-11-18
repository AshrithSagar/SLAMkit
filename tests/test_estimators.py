"""End-to-end: every estimator, assembled from the same parts, beats dead reckoning."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from slamkit.association import (
    DataAssociator,
    JointCompatibility,
    KnownCorrespondence,
    NearestNeighbor,
)
from slamkit.core import Vector
from slamkit.estimation import EkfSlam, FastSlam, GraphSlam, SlamEstimator, UkfSlam
from slamkit.estimation.graph import LandmarkKey, PoseKey, Values
from slamkit.estimation.particle import (
    MotionProposal,
    MultinomialResampler,
    ObservationProposal,
    PoseProposal,
    Resampler,
    SystematicResampler,
)
from slamkit.evaluation import Evaluator
from slamkit.motion import MotionModel, Odometry, OdometryMotionModel
from slamkit.pipeline import Dataset, SlamRunner
from slamkit.sensing import BearingModel, PointSensor, RangeBearingModel, SensingRange
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
    "ukf": lambda m, a: UkfSlam(m, a),
    "fastslam": lambda m, a: FastSlam(m, a, np.random.default_rng(1), num_particles=30),
    "fastslam2": lambda m, a: FastSlam(
        m, a, np.random.default_rng(1), num_particles=30, proposal=ObservationProposal()
    ),
    "graph": lambda m, a: GraphSlam(m, a, optimize_every=8),
}


# FastSLAM's particles deplete when sensors are much more accurate than odometry;
# it is noticeably noisier than the Gaussian backends on any single run.
ATE_BOUND = {"ekf": 0.5, "ukf": 0.5, "fastslam": 1.0, "fastslam2": 1.0, "graph": 0.5}


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


@pytest.mark.parametrize("name", ["ekf", "ukf", "graph"])
@pytest.mark.parametrize("associator", [NearestNeighbor, JointCompatibility])
def test_unknown_correspondence(
    name: str, associator: type[NearestNeighbor | JointCompatibility]
) -> None:
    dataset = simulate(tagged=False)
    assert dataset.ground_truth is not None
    result = SlamRunner(
        ESTIMATORS[name](MOTION, associator()), runtime_checks=False
    ).run(dataset)
    report = Evaluator(match_radius=1.0).evaluate(result, dataset.ground_truth)
    assert report.landmarks_matched >= len(WORLD) - 1
    assert report.position_rmse < 0.5


@pytest.mark.parametrize("name", ["ekf", "ukf", "graph", "fastslam2"])
def test_bearing_only(name: str) -> None:
    """A monocular-style sensor: landmarks enter the map only once triangulated."""
    camera = PointSensor(
        "camera",
        BearingModel(math.radians(1.0)),
        sensing_range=SensingRange(max_range=10.0),
    )
    dataset = Simulator(WORLD, MOTION, [camera], np.random.default_rng(3)).run(
        PATH.odometry()
    )
    assert dataset.ground_truth is not None
    result = SlamRunner(
        ESTIMATORS[name](MOTION, KnownCorrespondence()), runtime_checks=False
    ).run(dataset)
    report = Evaluator().evaluate(result, dataset.ground_truth)
    # Landmarks only glimpsed over a short arc never gain enough parallax.
    assert report.landmarks_matched >= 0.7 * len(WORLD)
    assert report.final_position_error < dead_reckoning_error(dataset)
    assert report.position_rmse < 1.0


def test_fastslam2_survives_with_a_single_particle() -> None:
    """With one particle FastSLAM 1.0 is little better than dead reckoning; the
    observation-conditioned proposal keeps 2.0 close to a Gaussian filter."""
    dataset = simulate(tagged=True)
    truth = dataset.ground_truth
    assert truth is not None
    median_error: dict[type[PoseProposal], float] = {}
    for proposal in (MotionProposal, ObservationProposal):
        errors = [
            Evaluator()
            .evaluate(
                SlamRunner(
                    FastSlam(
                        MOTION,
                        KnownCorrespondence(),
                        np.random.default_rng(seed),
                        num_particles=1,
                        proposal=proposal(),
                    ),
                    runtime_checks=False,
                ).run(dataset),
                truth,
            )
            .position_rmse
            for seed in range(3)
        ]
        median_error[proposal] = float(np.median(errors))
    assert median_error[ObservationProposal] < 0.5 * median_error[MotionProposal]


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
