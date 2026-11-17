"""
Frames
======
Phantom-typed coordinate frames.

These classes carry no data and are never instantiated. They exist purely as
type-level tags on geometric values (``Point2[WorldFrame]``,
``SE2[WorldFrame, BodyFrame]``) so that mixing frames is a *static* error.

Each concrete tag is ``@final`` and they are siblings under ``Frame``: with
PEP 695 variance inference an otherwise-unused frame parameter is inferred
covariant, so keeping the tags unrelated is what makes
``Point2[WorldFrame]`` and ``Point2[BodyFrame]`` mutually unassignable.
"""
# src/slamkit/core/frames.py

from typing import final

from slamkit.core.namespace import Namespace

__all__ = [
    "BodyFrame",
    "Frame",
    "SensorFrame",
    "WorldFrame",
]


class Frame(Namespace):
    """Base of all frame tags. ``Point2[Frame]`` means "a point in *some* frame"."""


@final
class WorldFrame(Frame):
    """The fixed global/map frame. Landmarks and trajectories live here."""


@final
class BodyFrame(Frame):
    """The robot's frame: origin at the robot, moving and rotating with it."""


@final
class SensorFrame(Frame):
    """A sensor's own frame, offset from ``BodyFrame`` by its extrinsic calibration."""
