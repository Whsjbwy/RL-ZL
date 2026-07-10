from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config  # noqa: F401 - initializes local src path
from rl_zl.obstacles import CylinderObstacle, EllipsoidObstacle, SphereObstacle, ray_distances


class ObstacleTests(unittest.TestCase):
    def test_sphere_collision_and_ray(self):
        sphere = SphereObstacle(np.array([10.0, 0.0, 0.0]), 2.0)
        self.assertTrue(sphere.contains(np.array([10.0, 0.0, 0.0])))
        self.assertFalse(sphere.contains(np.array([7.0, 0.0, 0.0])))
        distance = sphere.ray_distance(np.zeros(3), np.array([1.0, 0.0, 0.0]), 20.0)
        self.assertAlmostEqual(distance, 8.0, places=8)

    def test_ellipsoid_ray(self):
        obstacle = EllipsoidObstacle(np.array([10.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))
        distance = obstacle.ray_distance(np.zeros(3), np.array([1.0, 0.0, 0.0]), 20.0)
        self.assertAlmostEqual(distance, 8.0, places=8)

    def test_finite_cylinder_side_and_cap(self):
        cylinder = CylinderObstacle(np.array([10.0, 0.0, 5.0]), radius_m=2.0, height_m=4.0)
        side = cylinder.ray_distance(np.array([0.0, 0.0, 5.0]), np.array([1.0, 0.0, 0.0]), 20.0)
        cap = cylinder.ray_distance(np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 20.0)
        self.assertAlmostEqual(side, 8.0, places=8)
        self.assertAlmostEqual(cap, 3.0, places=8)

    def test_ray_batch_uses_nearest_obstacle(self):
        obstacles = [
            SphereObstacle(np.array([10.0, 0.0, 0.0]), 1.0),
            SphereObstacle(np.array([5.0, 0.0, 0.0]), 1.0),
        ]
        distances = ray_distances(np.zeros(3), np.array([[1.0, 0.0, 0.0]]), obstacles, 50.0)
        self.assertAlmostEqual(float(distances[0]), 4.0, places=8)


if __name__ == "__main__":
    unittest.main()

