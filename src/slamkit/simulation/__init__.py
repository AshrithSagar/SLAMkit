"""
Simulation
==========
Synthetic worlds, paths and datasets with ground truth.
"""
# src/slamkit/simulation/__init__.py

from slamkit.simulation.paths import CircularPath
from slamkit.simulation.simulator import Simulator
from slamkit.simulation.world import LandmarkWorld

__all__ = [
    "CircularPath",
    "LandmarkWorld",
    "Simulator",
]
