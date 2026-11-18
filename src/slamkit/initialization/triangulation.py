"""
Triangulation
=============
Delayed initialisation from several direction-constraining sightings.
"""
# src/slamkit/initialization/triangulation.py

import itertools
import math
from collections.abc import Sequence

import numpy as np

from slamkit.core.arrays import Mat2, Mat2x3, Matrix, Vector
from slamkit.core.frames import WorldFrame
from slamkit.geometry.point import Point2
from slamkit.geometry.ray import Ray
from slamkit.initialization.base import LandmarkInitializer, Sighting
from slamkit.probability.covariance import Covariance
from slamkit.sensing.observation import LandmarkInitialization, MeasurementLinearization

__all__ = [
    "TriangulatingInitializer",
]


class TriangulatingInitializer(LandmarkInitializer):
    """Intersects the sightings' rays, then refines the point by Gauss-Newton on
    the measurements themselves.

    Works for *any* point sensor (each reading at least fixes a ray), which is
    what makes bearing-only SLAM possible. It waits until the rays subtend at
    least ``min_parallax`` -- below that, depth is too poorly determined to be
    worth inserting.

    Each reading is weighted by ``R + Hₓ Σₓ Hₓᵀ``: its own noise plus the
    uncertainty of the pose it was made from. The result is therefore
    returned as *independent* of the current pose (``wrt_pose = 0``), with
    that pose uncertainty already inside its covariance -- the conservative
    choice, since a filter keeps no past poses with which to express the true,
    partial correlation. (A factor graph does keep them, and re-adds every
    sighting as a factor, so for it this is only an initial value.)
    ``covariance_inflation`` optionally widens the result further.
    """

    def __init__(
        self,
        min_parallax: float = math.radians(20.0),
        min_sightings: int = 2,
        iterations: int = 10,
        covariance_inflation: float = 1.0,
    ) -> None:
        if min_sightings < 2:
            raise ValueError("triangulation needs at least two sightings")
        self._min_parallax = min_parallax
        self._min_sightings = min_sightings
        self._iterations = iterations
        self._covariance_inflation = covariance_inflation

    def initialize[A](
        self, sightings: Sequence[Sighting[A]]
    ) -> LandmarkInitialization | None:
        if len(sightings) < self._min_sightings:
            return None
        rays = [s.observation.ray(s.pose) for s in sightings]
        if self._parallax(rays) < self._min_parallax:
            return None
        guess = Ray.closest_point(rays)
        if guess is None or any(ray.depth_of(guess) <= 0.0 for ray in rays):
            return None
        point = self._refine(guess, sightings)
        information = self._information(point, sightings)
        covariance = Matrix(self._covariance_inflation * np.linalg.inv(information))
        return LandmarkInitialization(
            position=point,
            wrt_pose=Mat2x3(np.zeros((2, 3))),
            measurement_covariance=Mat2(Covariance.symmetrize(covariance)),
        )

    @staticmethod
    def _parallax(rays: Sequence[Ray[WorldFrame]]) -> float:
        return max(
            (a.angle_to(b) for a, b in itertools.combinations(rays, 2)), default=0.0
        )

    def _refine[A](
        self, point: Point2[WorldFrame], sightings: Sequence[Sighting[A]]
    ) -> Point2[WorldFrame]:
        """Gauss-Newton on ``Σ ‖zᵢ ⊖ h(xᵢ, l)‖²_Rᵢ`` over the landmark ``l``."""
        for _ in range(self._iterations):
            information = np.zeros((2, 2))
            gradient = np.zeros(2)
            for sighting in sightings:
                lin = sighting.observation.linearize(sighting.pose, point)
                weighted = lin.wrt_landmark.T @ np.linalg.inv(
                    self._noise(sighting, lin)
                )
                information += weighted @ lin.wrt_landmark
                gradient += weighted @ lin.innovation
            step = Vector(np.linalg.solve(information, gradient))
            point = point.oplus(step)
            if float(np.linalg.norm(step)) < 1e-9:
                break
        return point

    def _information[A](
        self, point: Point2[WorldFrame], sightings: Sequence[Sighting[A]]
    ) -> Matrix:
        """``Σ Hᵢᵀ (Rᵢ + Hₓ Σₓ Hₓᵀ)⁻¹ Hᵢ`` over the sightings, at ``point``."""
        information = np.zeros((2, 2))
        for sighting in sightings:
            lin = sighting.observation.linearize(sighting.pose, point)
            information += lin.wrt_landmark.T @ np.linalg.solve(
                self._noise(sighting, lin), lin.wrt_landmark
            )
        return Matrix(information)

    @staticmethod
    def _noise[A](sighting: Sighting[A], lin: MeasurementLinearization) -> Matrix:
        """Reading noise plus the uncertainty of the pose it was read from."""
        pose_part = Covariance.propagate(lin.wrt_pose, Matrix(sighting.pose_covariance))
        return Matrix(lin.noise_covariance + pose_part)
