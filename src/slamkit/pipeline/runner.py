"""
Runner
======
Drives any estimator through any dataset and records what it produced.
"""
# src/slamkit/pipeline/runner.py

import time
from contextlib import nullcontext
from dataclasses import dataclass

from typingkit.core import RuntimeOptions

from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.estimation.estimator import (
    SlamEstimate,
    SlamEstimator,
    SmoothingSlamEstimator,
)
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Control
from slamkit.pipeline.dataset import Dataset

__all__ = [
    "RunResult",
    "SlamRunner",
]


@dataclass(frozen=True, slots=True)
class RunResult:
    estimates: tuple[SlamEstimate, ...]
    """The estimate after each step (online / filtered)."""
    trajectory: tuple[SE2[WorldFrame, BodyFrame], ...]
    """Initial pose plus one pose per step: the smoothed trajectory for smoothing
    estimators, otherwise the sequence of filtered poses."""
    elapsed_seconds: float

    @property
    def final(self) -> SlamEstimate:
        return self.estimates[-1]


class SlamRunner[C: Control]:
    """Feeds a dataset step by step into an estimator.

    ``runtime_checks=False`` switches off typingkit's runtime shape/dtype
    validation for the duration of the run (static typing is unaffected).
    Validation is what catches a mis-shaped Jacobian the moment it is built,
    so leave it on while developing a new model; turn it off for speed once
    the model is trusted -- it dominates the cost of particle filters.
    """

    def __init__(
        self, estimator: SlamEstimator[C], runtime_checks: bool = True
    ) -> None:
        self._estimator = estimator
        self._runtime_checks = runtime_checks

    @property
    def estimator(self) -> SlamEstimator[C]:
        return self._estimator

    def run(self, dataset: Dataset[C]) -> RunResult:
        with (
            nullcontext()
            if self._runtime_checks
            else RuntimeOptions.scoped(validate=False)
        ):
            return self._run(dataset)

    def _run(self, dataset: Dataset[C]) -> RunResult:
        start = time.perf_counter()
        initial = self._estimator.estimate()
        estimates = tuple(
            self._estimator.step(step.control, step.observations) for step in dataset
        )
        elapsed = time.perf_counter() - start
        if isinstance(self._estimator, SmoothingSlamEstimator):
            trajectory = tuple(self._estimator.trajectory())
        else:
            trajectory = (initial.pose, *(estimate.pose for estimate in estimates))
        return RunResult(estimates or (initial,), trajectory, elapsed)
