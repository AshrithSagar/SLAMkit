"""
Factor graph
============
"""
# src/slamkit/estimation/graph/graph.py

from collections.abc import Iterator, Sequence

from slamkit.estimation.graph.factors import Factor
from slamkit.estimation.graph.values import Values

__all__ = [
    "FactorGraph",
]


class FactorGraph:
    """A collection of factors defining ``½ Σ_f ‖r_f(x)‖²_Σf``."""

    def __init__(self, factors: Sequence[Factor] = ()) -> None:
        self._factors: list[Factor] = list(factors)

    def add(self, factor: Factor) -> None:
        self._factors.append(factor)

    @property
    def factors(self) -> Sequence[Factor]:
        return tuple(self._factors)

    def __iter__(self) -> Iterator[Factor]:
        return iter(self._factors)

    def __len__(self) -> int:
        return len(self._factors)

    def error(self, values: Values) -> float:
        return sum(factor.error(values) for factor in self._factors)
