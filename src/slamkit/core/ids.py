"""
Identifiers
===========
Distinct nominal types for the integers that index SLAM entities, plus the
allocator that hands out fresh landmark ids.
"""
# src/slamkit/core/ids.py

from typing import NewType

__all__ = [
    "LandmarkId",
    "LandmarkIdAllocator",
]

LandmarkId = NewType("LandmarkId", int)
"""Identity of a map landmark. Distinct from a plain ``int`` statically."""


class LandmarkIdAllocator:
    """Hands out monotonically increasing, never-reused ``LandmarkId``s.

    Shared by anything that has to *invent* landmark identities (e.g. a data
    associator deciding an observation is a new landmark) so ids stay unique
    across particles, sensors and estimators that share the allocator.
    """

    def __init__(self, start: int = 0) -> None:
        self._next: int = start

    def allocate(self) -> LandmarkId:
        lid = LandmarkId(self._next)
        self._next += 1
        return lid

    def reserve(self, landmark_id: LandmarkId) -> None:
        """Ensure ``landmark_id`` (e.g. an externally supplied tag) is never reissued."""
        self._next = max(self._next, landmark_id + 1)
