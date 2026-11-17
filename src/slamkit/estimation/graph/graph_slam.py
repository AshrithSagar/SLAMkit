"""
GraphSLAM
=========
Full SLAM as nonlinear least squares over the whole trajectory and map.

Filtering marginalises past poses away as it goes, freezing their
linearisation errors in; smoothing keeps every pose and re-linearises
everything on each optimisation, which is why it is the most accurate of the
three backends (at the highest cost).
"""
# src/slamkit/estimation/graph/graph_slam.py

from collections.abc import Collection, Sequence

import numpy as np

from slamkit.association.base import DataAssociator
from slamkit.core.arrays import Mat3, Matrix
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.core.ids import LandmarkId
from slamkit.estimation.estimator import (
    MapHypothesis,
    SlamEstimate,
    SmoothingSlamEstimator,
)
from slamkit.estimation.graph.factors import (
    BetweenFactor,
    LandmarkObservationFactor,
    PosePriorFactor,
)
from slamkit.estimation.graph.graph import FactorGraph
from slamkit.estimation.graph.linear import NormalEquations
from slamkit.estimation.graph.optimizers import (
    LevenbergMarquardt,
    OptimizationResult,
    Optimizer,
)
from slamkit.estimation.graph.values import Key, LandmarkKey, PoseKey, Values
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.motion.model import MotionModel
from slamkit.probability.gaussian import GaussianNoise, Innovation
from slamkit.sensing.observation import Observation

__all__ = [
    "GraphSlam",
]


class GraphSlam[C: Control](SmoothingSlamEstimator[C], MapHypothesis):
    """Incremental-batch GraphSLAM, composed from a motion model, a data
    associator and a nonlinear optimiser.

    ``predict`` adds a pose and a between-factor; ``update`` adds landmark
    factors; every ``optimize_every`` updates the whole graph is re-solved
    (``0`` disables automatic optimisation -- call ``optimize()`` yourself).
    """

    def __init__(
        self,
        motion_model: MotionModel[C],
        associator: DataAssociator,
        optimizer: Optimizer | None = None,
        initial_pose: SE2[WorldFrame, BodyFrame] | None = None,
        prior_covariance: Mat3 | None = None,
        optimize_every: int = 1,
    ) -> None:
        if optimize_every < 0:
            raise ValueError("optimize_every must be non-negative")
        self._motion_model = motion_model
        self._associator = associator
        self._optimizer = optimizer if optimizer is not None else LevenbergMarquardt()
        self._optimize_every = optimize_every
        self._updates = 0
        self._graph = FactorGraph()
        self._values = Values()
        self._current = PoseKey(0)
        self._marginals: NormalEquations | None = None

        pose = (
            initial_pose
            if initial_pose is not None
            else SE2[WorldFrame, BodyFrame].identity()
        )
        prior = (
            prior_covariance if prior_covariance is not None else Mat3(np.eye(3) * 1e-6)
        )
        self._values.set_pose(self._current, pose)
        self._add_factor(
            PosePriorFactor(self._current, pose, GaussianNoise(Matrix(prior)))
        )

    @property
    def graph(self) -> FactorGraph:
        return self._graph

    @property
    def values(self) -> Values:
        return self._values

    def optimize(self) -> OptimizationResult:
        result = self._optimizer.optimize(self._graph, self._values)
        self._values = result.values
        self._marginals = None
        return result

    # ── SlamEstimator ─────────────────────────────────────────────────────────

    def predict(self, control: C) -> None:
        previous, self._current = self._current, PoseKey(self._current.index + 1)
        increment = self._motion_model.increment(control)
        self._values.set_pose(self._current, self._values[previous] @ increment)
        noise = GaussianNoise(Matrix(self._motion_model.increment_covariance(control)))
        self._add_factor(BetweenFactor(previous, self._current, increment, noise))

    def update(self, observations: Sequence[Observation]) -> None:
        self.assimilate(observations, self._associator)
        self._updates += 1
        if self._optimize_every and self._updates % self._optimize_every == 0:
            self.optimize()

    def estimate(self) -> SlamEstimate:
        return SlamEstimate(
            pose=self._values[self._current],
            landmarks={
                key.landmark_id: point for key, point in self._values.landmarks.items()
            },
        )

    def trajectory(self) -> Sequence[SE2[WorldFrame, BodyFrame]]:
        return [self._values[PoseKey(i)] for i in range(self._current.index + 1)]

    # ── MapHypothesis ─────────────────────────────────────────────────────────

    def landmark_ids(self) -> Collection[LandmarkId]:
        return [key.landmark_id for key in self._values.landmarks]

    def innovation(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> Innovation:
        landmark = LandmarkKey(landmark_id)
        linearization = observation.linearize(
            self._values[self._current], self._values[landmark]
        )
        return linearization.innovation_given(
            self._joint_covariance(self._current, landmark)
        )

    def update_landmark(
        self, observation: Observation, landmark_id: LandmarkId
    ) -> None:
        self._add_factor(
            LandmarkObservationFactor(
                self._current, LandmarkKey(landmark_id), observation
            )
        )

    def add_landmark(self, observation: Observation, landmark_id: LandmarkId) -> None:
        key = LandmarkKey(landmark_id)
        initialization = observation.initialize_landmark(self._values[self._current])
        self._values.set_landmark(key, initialization.position)
        self._add_factor(LandmarkObservationFactor(self._current, key, observation))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_factor(
        self, factor: PosePriorFactor | BetweenFactor | LandmarkObservationFactor
    ) -> None:
        self._graph.add(factor)
        self._marginals = None

    def _joint_covariance(self, *keys: Key) -> Matrix:
        """Marginal covariance of ``keys`` from ``H⁻¹`` at the current values (cached
        until the graph or the values change)."""
        if self._marginals is None:
            self._marginals = NormalEquations(self._graph, self._values)
        indices = self._marginals.ordering.indices(*keys)
        return Matrix(self._marginals.covariance()[np.ix_(indices, indices)])
