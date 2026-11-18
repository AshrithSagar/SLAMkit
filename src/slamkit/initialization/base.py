"""
Landmark initialisation
=======================
Turning sightings of a not-yet-mapped landmark into a position estimate.

When to insert a new landmark, and where, is its own concern -- separate from
sensing (what a reading means), association (which landmark it is of) and
estimation (how the belief absorbs it):

* a range-bearing reading fixes the landmark immediately (*undelayed*);
* a bearing-only reading only fixes a ray, so the landmark must wait for
  enough parallax across several viewpoints, then be triangulated (*delayed*).

A ``LandmarkInitializer`` sees every sighting of one pending landmark so far
and either produces a ``LandmarkInitialization`` or asks to wait (``None``).
The estimator-side buffering of sightings lives once, in ``MapHypothesis``.
"""
# src/slamkit/initialization/base.py

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from slamkit.core.arrays import Mat3
from slamkit.core.frames import BodyFrame, WorldFrame
from slamkit.geometry.se2 import SE2
from slamkit.sensing.observation import LandmarkInitialization, Observation

__all__ = [
    "LandmarkInitializer",
    "Sighting",
]


@dataclass(frozen=True, slots=True)
class Sighting[A]:
    """An observation of a pending landmark, with the pose belief it was made
    from and an ``anchor``: whatever the recording estimator needs to refer back
    to that moment later (a factor graph's pose key; ``None`` for filters)."""

    pose: SE2[WorldFrame, BodyFrame]
    pose_covariance: Mat3
    observation: Observation
    anchor: A


class LandmarkInitializer(ABC):
    """Decides when, and where, a new landmark enters the map."""

    @abstractmethod
    def initialize[A](
        self, sightings: Sequence[Sighting[A]]
    ) -> LandmarkInitialization | None:
        """A position for the landmark seen in ``sightings`` (oldest first; the
        last is the newest and the current pose), or ``None`` to keep waiting."""
