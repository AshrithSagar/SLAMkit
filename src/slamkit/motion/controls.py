"""
Controls
========
Inputs that drive the robot between two timesteps.
"""
# src/slamkit/motion/controls.py

from abc import ABC
from dataclasses import dataclass

__all__ = [
    "Control",
    "Odometry",
    "VelocityCommand",
]


class Control(ABC):
    """Marker base for control inputs; ``MotionModel[C]`` is bound to it."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Odometry(Control):
    """Measured relative motion, expressed in the robot's *previous* body frame
    ("forward / left / turn"), as reported by wheel encoders or scan matching."""

    dx: float
    dy: float
    dtheta: float


@dataclass(frozen=True, slots=True)
class VelocityCommand(Control):
    """Commanded forward speed and turn rate, held constant for ``dt`` seconds."""

    linear: float
    angular: float
    dt: float
