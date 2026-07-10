"""Obstacle geometry, collision checks and analytic ray distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


EPS = 1e-10


def _positive_roots(a: float, b: float, c: float) -> list[float]:
    if abs(a) < EPS:
        if abs(b) < EPS:
            return []
        root = -c / b
        return [root] if root >= 0.0 else []
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    sqrt_d = float(np.sqrt(max(0.0, discriminant)))
    roots = [(-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a)]
    return sorted(root for root in roots if root >= 0.0)


class Obstacle(Protocol):
    center: np.ndarray

    def contains(self, point_m: np.ndarray, inflation_m: float = 0.0) -> bool: ...

    def signed_distance(self, point_m: np.ndarray, inflation_m: float = 0.0) -> float: ...

    def ray_distance(
        self,
        origin_m: np.ndarray,
        direction_unit: np.ndarray,
        max_distance_m: float,
        inflation_m: float = 0.0,
    ) -> float: ...

    def bounding_radius(self, inflation_m: float = 0.0) -> float: ...

    def to_dict(self) -> dict: ...


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius_m: float

    def contains(self, point_m: np.ndarray, inflation_m: float = 0.0) -> bool:
        return bool(np.linalg.norm(np.asarray(point_m) - self.center) <= self.radius_m + inflation_m)

    def signed_distance(self, point_m: np.ndarray, inflation_m: float = 0.0) -> float:
        return float(np.linalg.norm(np.asarray(point_m) - self.center) - self.radius_m - inflation_m)

    def ray_distance(self, origin_m, direction_unit, max_distance_m, inflation_m=0.0) -> float:
        if self.contains(origin_m, inflation_m):
            return 0.0
        offset = np.asarray(origin_m, dtype=float) - self.center
        direction = np.asarray(direction_unit, dtype=float)
        radius = self.radius_m + inflation_m
        roots = _positive_roots(
            float(np.dot(direction, direction)),
            float(2.0 * np.dot(offset, direction)),
            float(np.dot(offset, offset) - radius * radius),
        )
        return float(roots[0]) if roots and roots[0] <= max_distance_m else float(max_distance_m)

    def bounding_radius(self, inflation_m: float = 0.0) -> float:
        return float(self.radius_m + inflation_m)

    def to_dict(self) -> dict:
        return {"type": "sphere", "center": self.center.tolist(), "radius_m": self.radius_m}


@dataclass
class EllipsoidObstacle:
    center: np.ndarray
    axes_m: np.ndarray

    def contains(self, point_m: np.ndarray, inflation_m: float = 0.0) -> bool:
        axes = self.axes_m + inflation_m
        scaled = (np.asarray(point_m, dtype=float) - self.center) / axes
        return bool(np.dot(scaled, scaled) <= 1.0)

    def signed_distance(self, point_m: np.ndarray, inflation_m: float = 0.0) -> float:
        axes = self.axes_m + inflation_m
        scaled_norm = np.linalg.norm((np.asarray(point_m, dtype=float) - self.center) / axes)
        return float((scaled_norm - 1.0) * np.min(axes))

    def ray_distance(self, origin_m, direction_unit, max_distance_m, inflation_m=0.0) -> float:
        if self.contains(origin_m, inflation_m):
            return 0.0
        axes = self.axes_m + inflation_m
        offset = (np.asarray(origin_m, dtype=float) - self.center) / axes
        direction = np.asarray(direction_unit, dtype=float) / axes
        roots = _positive_roots(
            float(np.dot(direction, direction)),
            float(2.0 * np.dot(offset, direction)),
            float(np.dot(offset, offset) - 1.0),
        )
        return float(roots[0]) if roots and roots[0] <= max_distance_m else float(max_distance_m)

    def bounding_radius(self, inflation_m: float = 0.0) -> float:
        return float(np.linalg.norm(self.axes_m + inflation_m))

    def to_dict(self) -> dict:
        return {"type": "ellipsoid", "center": self.center.tolist(), "axes_m": self.axes_m.tolist()}


@dataclass
class CylinderObstacle:
    center: np.ndarray
    radius_m: float
    height_m: float

    @property
    def z_min(self) -> float:
        return float(self.center[2] - 0.5 * self.height_m)

    @property
    def z_max(self) -> float:
        return float(self.center[2] + 0.5 * self.height_m)

    def contains(self, point_m: np.ndarray, inflation_m: float = 0.0) -> bool:
        point = np.asarray(point_m, dtype=float)
        radial = np.linalg.norm(point[:2] - self.center[:2])
        return bool(
            radial <= self.radius_m + inflation_m
            and self.z_min - inflation_m <= point[2] <= self.z_max + inflation_m
        )

    def signed_distance(self, point_m: np.ndarray, inflation_m: float = 0.0) -> float:
        point = np.asarray(point_m, dtype=float)
        radius = self.radius_m + inflation_m
        half_height = 0.5 * self.height_m + inflation_m
        q = np.array(
            [np.linalg.norm(point[:2] - self.center[:2]) - radius, abs(point[2] - self.center[2]) - half_height]
        )
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(q[0], q[1]), 0.0)
        return float(outside + inside)

    def ray_distance(self, origin_m, direction_unit, max_distance_m, inflation_m=0.0) -> float:
        if self.contains(origin_m, inflation_m):
            return 0.0
        origin = np.asarray(origin_m, dtype=float)
        direction = np.asarray(direction_unit, dtype=float)
        radius = self.radius_m + inflation_m
        z_min = self.z_min - inflation_m
        z_max = self.z_max + inflation_m
        offset_xy = origin[:2] - self.center[:2]
        candidates: list[float] = []

        side_roots = _positive_roots(
            float(np.dot(direction[:2], direction[:2])),
            float(2.0 * np.dot(offset_xy, direction[:2])),
            float(np.dot(offset_xy, offset_xy) - radius * radius),
        )
        for root in side_roots:
            z_hit = origin[2] + root * direction[2]
            if z_min - EPS <= z_hit <= z_max + EPS:
                candidates.append(root)

        if abs(direction[2]) > EPS:
            for z_cap in (z_min, z_max):
                root = (z_cap - origin[2]) / direction[2]
                if root >= 0.0:
                    xy_hit = origin[:2] + root * direction[:2]
                    if np.linalg.norm(xy_hit - self.center[:2]) <= radius + EPS:
                        candidates.append(float(root))

        if not candidates:
            return float(max_distance_m)
        distance = min(candidates)
        return float(distance) if distance <= max_distance_m else float(max_distance_m)

    def bounding_radius(self, inflation_m: float = 0.0) -> float:
        radius = self.radius_m + inflation_m
        half_height = 0.5 * self.height_m + inflation_m
        return float(np.hypot(radius, half_height))

    def to_dict(self) -> dict:
        return {
            "type": "cylinder",
            "center": self.center.tolist(),
            "radius_m": self.radius_m,
            "height_m": self.height_m,
        }


def minimum_signed_distance(
    point_m: np.ndarray,
    obstacles: Iterable[Obstacle],
    inflation_m: float = 0.0,
    default: float = np.inf,
) -> float:
    distances = [obstacle.signed_distance(point_m, inflation_m) for obstacle in obstacles]
    return float(min(distances)) if distances else float(default)


def collides(point_m: np.ndarray, obstacles: Iterable[Obstacle], inflation_m: float = 0.0) -> bool:
    return any(obstacle.contains(point_m, inflation_m) for obstacle in obstacles)


def ray_distances(
    origin_m: np.ndarray,
    directions_world: np.ndarray,
    obstacles: Iterable[Obstacle],
    max_distance_m: float,
    inflation_m: float = 0.0,
) -> np.ndarray:
    obstacle_list = list(obstacles)
    output = np.full(len(directions_world), max_distance_m, dtype=np.float64)
    for index, direction in enumerate(directions_world):
        for obstacle in obstacle_list:
            output[index] = min(
                output[index],
                obstacle.ray_distance(origin_m, direction, output[index], inflation_m),
            )
    return output


def make_26_ray_directions() -> np.ndarray:
    directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                vector = np.array([x, y, z], dtype=np.float64)
                norm = np.linalg.norm(vector)
                if norm > 0.0:
                    directions.append(vector / norm)
    return np.asarray(directions, dtype=np.float64)


RAY_DIRECTIONS_BODY = make_26_ray_directions()


__all__ = [
    "CylinderObstacle",
    "EllipsoidObstacle",
    "Obstacle",
    "RAY_DIRECTIONS_BODY",
    "SphereObstacle",
    "collides",
    "minimum_signed_distance",
    "ray_distances",
]

