"""
Compare estimators
==================
Runs EKF-SLAM, FastSLAM and GraphSLAM on one simulated dataset, first with
known correspondences and then with nearest-neighbour data association, and
prints their accuracy against ground truth.

Every run is assembled from the *same* motion model,
sensors and associators; only the estimator differs.
"""
# examples/compare_estimators.py

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from slamkit.association import DataAssociator, KnownCorrespondence, NearestNeighbor
from slamkit.core import BodyFrame, SensorFrame
from slamkit.estimation import EkfSlam, FastSlam, GraphSlam, SlamEstimator
from slamkit.evaluation import Evaluator
from slamkit.geometry import SE2
from slamkit.motion import MotionModel, Odometry, OdometryMotionModel
from slamkit.pipeline import Dataset, GroundTruth, SlamRunner
from slamkit.sensing import CartesianModel, PointSensor, RangeBearingModel, SensingRange
from slamkit.simulation import CircularPath, LandmarkWorld, Simulator

type EstimatorFactory = Callable[
    [MotionModel[Odometry], DataAssociator], SlamEstimator[Odometry]
]


@dataclass(frozen=True, slots=True)
class Experiment:
    """One estimator × one association strategy on one dataset."""

    name: str
    factory: EstimatorFactory

    def run(
        self,
        dataset: Dataset[Odometry],
        truth: GroundTruth,
        motion_model: MotionModel[Odometry],
        associator: DataAssociator,
        evaluator: Evaluator,
    ) -> str:
        result = SlamRunner(
            self.factory(motion_model, associator), runtime_checks=False
        ).run(dataset)
        report = evaluator.evaluate(result, truth)
        return f"{self.name:<10} {result.elapsed_seconds:6.2f}s | {report}"


class Demo:
    SEED = 7

    def __init__(self) -> None:
        self._rng = np.random.default_rng(self.SEED)
        self._motion_model = OdometryMotionModel(
            translation_noise=0.05, rotation_noise=0.05, drift_per_meter=0.02
        )
        lidar = PointSensor(
            "lidar",
            RangeBearingModel(range_std=0.15, bearing_std=math.radians(2.0)),
            sensing_range=SensingRange(
                max_range=8.0, field_of_view=math.radians(180.0)
            ),
        )
        camera = PointSensor(
            "stereo",
            CartesianModel(x_std=0.2, y_std=0.2),
            extrinsic=SE2[BodyFrame, SensorFrame](
                0.3, 0.0, math.radians(-90.0)
            ),  # side-facing
            sensing_range=SensingRange(max_range=5.0, field_of_view=math.radians(90.0)),
        )
        self._sensors = (lidar, camera)
        self._world = LandmarkWorld.uniform(
            15, self._rng, x_range=(-12.0, 12.0), y_range=(-4.0, 20.0)
        )
        self._path = CircularPath(radius=8.0, steps_per_loop=40, loops=2)
        self._experiments = (
            Experiment("EKF", lambda m, a: EkfSlam(m, a)),
            Experiment(
                "FastSLAM",
                lambda m, a: FastSlam(
                    m, a, np.random.default_rng(self.SEED), num_particles=25
                ),
            ),
            Experiment("GraphSLAM", lambda m, a: GraphSlam(m, a, optimize_every=20)),
        )

    def run(self) -> None:
        for title, tagged, associator_factory, evaluator in (
            ("known correspondence", True, KnownCorrespondence, Evaluator()),
            (
                "nearest-neighbour association",
                False,
                NearestNeighbor,
                Evaluator(match_radius=1.5),
            ),
        ):
            dataset = Simulator(
                self._world,
                self._motion_model,
                self._sensors,
                np.random.default_rng(self.SEED),
                tag_observations=tagged,
            ).run(self._path.odometry())
            assert dataset.ground_truth is not None
            dead_reckoning = self._dead_reckoning(dataset)
            print(
                f"\n── {title} ── ({len(dataset)} steps, {len(self._world)} landmarks)"
            )
            print(f"{'odometry':<10} {'':>7} | final drift {dead_reckoning:6.3f} m")
            for experiment in self._experiments:
                print(
                    experiment.run(
                        dataset,
                        dataset.ground_truth,
                        self._motion_model,
                        associator_factory(),
                        evaluator,
                    )
                )

    def _dead_reckoning(self, dataset: Dataset[Odometry]) -> float:
        assert dataset.ground_truth is not None
        pose = dataset.initial_pose
        for step in dataset:
            pose = self._motion_model.predict(pose, step.control)
        true = dataset.ground_truth.trajectory[-1]
        return math.hypot(pose.x - true.x, pose.y - true.y)


if __name__ == "__main__":
    Demo().run()
