"""
Fallback initialisation
=======================
"""
# src/slamkit/initialization/fallback.py

from collections.abc import Sequence

from slamkit.initialization.base import LandmarkInitializer, Sighting
from slamkit.initialization.single_shot import SingleShotInitializer
from slamkit.initialization.triangulation import TriangulatingInitializer
from slamkit.sensing.observation import LandmarkInitialization

__all__ = [
    "FallbackInitializer",
]


class FallbackInitializer(LandmarkInitializer):
    """Tries each initialiser in turn and takes the first answer (composite)."""

    def __init__(self, *initializers: LandmarkInitializer) -> None:
        if not initializers:
            raise ValueError("at least one initializer is required")
        self._initializers = initializers

    @classmethod
    def standard(cls) -> FallbackInitializer:
        """Undelayed when the sensor allows it, triangulation otherwise -- suits
        any mix of point sensors, and is what estimators use by default."""
        return cls(SingleShotInitializer(), TriangulatingInitializer())

    def initialize[A](
        self, sightings: Sequence[Sighting[A]]
    ) -> LandmarkInitialization | None:
        for initializer in self._initializers:
            result = initializer.initialize(sightings)
            if result is not None:
                return result
        return None
