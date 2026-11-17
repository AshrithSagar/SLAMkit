"""
Resampling
==========
Strategies for drawing a new particle set in proportion to the weights.
"""
# src/slamkit/estimation/particle/resampling.py

from abc import ABC, abstractmethod

import numpy as np

from slamkit.core.arrays import Vector

__all__ = [
    "MultinomialResampler",
    "Resampler",
    "SystematicResampler",
]


class Resampler(ABC):
    """Draws ``len(weights)`` particle indices, each with probability ∝ its weight."""

    def indices(self, weights: Vector, rng: np.random.Generator) -> list[int]:
        if weights.ndim != 1 or weights.shape[0] == 0:
            raise ValueError("weights must be a non-empty 1-D array")
        total = float(np.sum(weights))
        if not total > 0.0:
            raise ValueError("weights must have a positive sum")
        cumulative = np.cumsum(weights / total)
        cumulative[-1] = 1.0  # guard against round-off leaving the last bin short
        positions = self._positions(weights.shape[0], rng)
        return [int(i) for i in np.searchsorted(cumulative, positions, side="right")]

    @abstractmethod
    def _positions(self, count: int, rng: np.random.Generator) -> Vector:
        """``count`` sorted-or-unsorted sample points in ``[0, 1)``."""


class MultinomialResampler(Resampler):
    """Independent draws. Simple, but the highest-variance choice."""

    def _positions(self, count: int, rng: np.random.Generator) -> Vector:
        return Vector(rng.random(count))


class SystematicResampler(Resampler):
    """One random offset, then evenly spaced points: ``(u + k) / N``.
    Low variance and ``O(N)``; the usual default."""

    def _positions(self, count: int, rng: np.random.Generator) -> Vector:
        return Vector((rng.random() + np.arange(count)) / count)
