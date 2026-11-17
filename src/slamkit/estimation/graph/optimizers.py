"""
Optimisers
==========
Nonlinear least squares on manifolds: iterate "linearise, solve, retract".
"""
# src/slamkit/estimation/graph/optimizers.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

from slamkit.estimation.graph.graph import FactorGraph
from slamkit.estimation.graph.linear import NormalEquations
from slamkit.estimation.graph.values import Values

__all__ = [
    "GaussNewton",
    "LevenbergMarquardt",
    "OptimizationResult",
    "Optimizer",
]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    values: Values
    iterations: int
    initial_error: float
    final_error: float
    converged: bool


@dataclass(frozen=True, slots=True)
class _Step:
    values: Values
    error: float


class Optimizer(ABC):
    """The shared outer loop; subclasses decide how one iteration moves."""

    def __init__(
        self,
        max_iterations: int = 50,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-12,
    ) -> None:
        self._max_iterations = max_iterations
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def optimize(self, graph: FactorGraph, values: Values) -> OptimizationResult:
        self._reset()
        initial_error = error = graph.error(values)
        for iteration in range(1, self._max_iterations + 1):
            step = self._iterate(graph, values, error)
            if step is None:
                return OptimizationResult(
                    values, iteration, initial_error, error, converged=True
                )
            improvement = error - step.error
            values, error = step.values, step.error
            if (
                improvement
                <= self._relative_tolerance * error + self._absolute_tolerance
            ):
                return OptimizationResult(
                    values, iteration, initial_error, error, converged=True
                )
        return OptimizationResult(
            values, self._max_iterations, initial_error, error, converged=False
        )

    def _reset(self) -> None:
        """Hook to clear per-run state (e.g. a damping factor)."""

    @abstractmethod
    def _iterate(
        self, graph: FactorGraph, values: Values, error: float
    ) -> _Step | None:
        """One iteration from ``values``; ``None`` if no improving step exists."""


class GaussNewton(Optimizer):
    """Full Gauss-Newton steps. Fast near the optimum; can overshoot far from it."""

    def _iterate(
        self, graph: FactorGraph, values: Values, error: float
    ) -> _Step | None:
        candidate = values.retract(NormalEquations(graph, values).solve())
        return _Step(candidate, graph.error(candidate))


class LevenbergMarquardt(Optimizer):
    """Damped Gauss-Newton: only accepts error-decreasing steps, increasing the
    damping ``λ`` after a rejection and relaxing it after an acceptance."""

    def __init__(
        self,
        max_iterations: int = 50,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-12,
        initial_damping: float = 1e-4,
        damping_factor: float = 10.0,
        max_damping: float = 1e10,
    ) -> None:
        super().__init__(max_iterations, relative_tolerance, absolute_tolerance)
        self._initial_damping = initial_damping
        self._damping_factor = damping_factor
        self._max_damping = max_damping
        self._damping = initial_damping

    def _reset(self) -> None:
        self._damping = self._initial_damping

    def _iterate(
        self, graph: FactorGraph, values: Values, error: float
    ) -> _Step | None:
        system = NormalEquations(graph, values)
        while self._damping <= self._max_damping:
            candidate = values.retract(system.solve(self._damping))
            candidate_error = graph.error(candidate)
            if candidate_error < error:
                self._damping /= self._damping_factor
                return _Step(candidate, candidate_error)
            self._damping *= self._damping_factor
        return None
