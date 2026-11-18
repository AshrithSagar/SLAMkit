"""
Single-shot initialisation
==========================
"""
# src/slamkit/initialization/single_shot.py

from collections.abc import Sequence

from slamkit.initialization.base import LandmarkInitializer, Sighting
from slamkit.sensing.observation import LandmarkInitialization

__all__ = [
    "SingleShotInitializer",
]


class SingleShotInitializer(LandmarkInitializer):
    """Inverts the newest reading through its sensor model (undelayed
    initialisation). Waits forever for sensors that cannot invert one reading;
    combine with a triangulating initialiser for those."""

    def initialize[A](
        self, sightings: Sequence[Sighting[A]]
    ) -> LandmarkInitialization | None:
        if not sightings:
            return None
        latest = sightings[-1]
        return latest.observation.initialize_landmark(latest.pose)
