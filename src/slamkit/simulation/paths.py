"""
Paths
=====
Control sequences for canonical test trajectories.
"""
# src/slamkit/simulation/paths.py

import math

from slamkit.core.arrays import Vector
from slamkit.core.frames import BodyFrame
from slamkit.geometry.se2 import SE2
from slamkit.motion.controls import Odometry, VelocityCommand

__all__ = [
    "CircularPath",
]


class CircularPath:
    """Counter-clockwise laps of a circle, starting at the origin facing +x
    (so the circle is centred at ``(0, radius)``)."""

    def __init__(self, radius: float, steps_per_loop: int, loops: int = 1) -> None:
        if radius <= 0.0 or steps_per_loop < 3 or loops < 1:
            raise ValueError("need radius > 0, steps_per_loop >= 3, loops >= 1")
        self._radius = radius
        self._steps = steps_per_loop * loops
        self._turn = 2.0 * math.pi / steps_per_loop

    @property
    def arc_length(self) -> float:
        """Distance travelled per step."""
        return self._radius * self._turn

    def odometry(self) -> list[Odometry]:
        step = SE2[BodyFrame, BodyFrame].exp(Vector([self.arc_length, 0.0, self._turn]))
        return [Odometry(step.x, step.y, step.theta)] * self._steps

    def velocity_commands(self, dt: float = 1.0) -> list[VelocityCommand]:
        return [
            VelocityCommand(self.arc_length / dt, self._turn / dt, dt)
        ] * self._steps
