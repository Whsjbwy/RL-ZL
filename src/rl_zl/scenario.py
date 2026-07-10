"""Seeded scenario generation and coarse-grid feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product

import numpy as np

from .config import Stage0Config
from .current import AnalyticCurrentField
from .obstacles import (
    CylinderObstacle,
    EllipsoidObstacle,
    Obstacle,
    SphereObstacle,
    collides,
)


@dataclass
class Scenario:
    start_m: np.ndarray
    goal_m: np.ndarray
    obstacles: list[Obstacle]
    current: AnalyticCurrentField
    coarse_path_length_m: float | None
    seed: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "start_m": self.start_m.tolist(),
            "goal_m": self.goal_m.tolist(),
            "coarse_path_length_m": self.coarse_path_length_m,
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "current": self.current.to_dict(),
        }


class ScenarioGenerationError(RuntimeError):
    pass


class ScenarioGenerator:
    def __init__(self, config: Stage0Config):
        self.config = config
        self.world = np.asarray(config.environment.world_size_m, dtype=np.float64)
        self.inflation_m = config.vehicle.radius_m + config.obstacles.inflation_margin_m

    def sample(self, seed: int) -> Scenario:
        rng = np.random.default_rng(seed)
        for _ in range(self.config.feasibility.max_scenario_attempts):
            start, goal = self._sample_start_goal(rng)
            obstacles = self._sample_obstacles(rng, start, goal)
            current = AnalyticCurrentField.sample(
                self.config.current,
                rng,
                self.config.environment.world_size_m,
            )
            path_length = None
            if self.config.feasibility.enabled:
                path_length = coarse_astar_path_length(
                    start,
                    goal,
                    obstacles,
                    world_size_m=self.config.environment.world_size_m,
                    legal_depth_m=self.config.environment.legal_depth_m,
                    resolution_m=self.config.feasibility.grid_resolution_m,
                    inflation_m=self.inflation_m,
                )
                if path_length is None or not self._within_path_budget(path_length):
                    continue
            return Scenario(
                start_m=start,
                goal_m=goal,
                obstacles=obstacles,
                current=current,
                coarse_path_length_m=path_length,
                seed=seed,
            )
        raise ScenarioGenerationError(
            f"Unable to generate a feasible scenario after "
            f"{self.config.feasibility.max_scenario_attempts} attempts (seed={seed})."
        )

    def _sample_start_goal(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        env = self.config.environment
        low = np.array([env.xy_margin_m, env.xy_margin_m, env.start_goal_depth_m[0]])
        high = np.array(
            [self.world[0] - env.xy_margin_m, self.world[1] - env.xy_margin_m, env.start_goal_depth_m[1]]
        )
        for _ in range(500):
            start = rng.uniform(low, high)
            goal = rng.uniform(low, high)
            distance = float(np.linalg.norm(goal - start))
            if env.start_goal_min_distance_m <= distance <= env.start_goal_max_distance_m:
                return start, goal
        raise ScenarioGenerationError("Unable to sample a start-goal pair in the configured distance range")

    def _sample_obstacles(
        self,
        rng: np.random.Generator,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[Obstacle]:
        cfg = self.config.obstacles
        target_count = int(rng.integers(cfg.count_range[0], cfg.count_range[1] + 1))
        obstacles: list[Obstacle] = []
        attempts = 0
        while len(obstacles) < target_count and attempts < cfg.max_generation_attempts:
            attempts += 1
            candidate = self._sample_one_obstacle(rng)
            if candidate.signed_distance(start, self.inflation_m) < cfg.start_goal_clearance_m:
                continue
            if candidate.signed_distance(goal, self.inflation_m) < cfg.start_goal_clearance_m:
                continue
            if any(
                np.linalg.norm(candidate.center - existing.center)
                < candidate.bounding_radius(self.inflation_m)
                + existing.bounding_radius(self.inflation_m)
                + cfg.obstacle_separation_m
                for existing in obstacles
            ):
                continue
            obstacles.append(candidate)
        if len(obstacles) != target_count:
            raise ScenarioGenerationError(
                f"Generated {len(obstacles)}/{target_count} obstacles after {attempts} attempts"
            )
        return obstacles

    def _sample_one_obstacle(self, rng: np.random.Generator) -> Obstacle:
        cfg = self.config.obstacles
        kind = str(rng.choice(cfg.types))
        xy_low = cfg.boundary_clearance_m
        x = rng.uniform(xy_low, self.world[0] - xy_low)
        y = rng.uniform(xy_low, self.world[1] - xy_low)

        if kind == "sphere":
            radius = float(rng.uniform(*cfg.sphere_radius_m))
            z_low = radius + cfg.inflation_margin_m
            z = rng.uniform(z_low, self.world[2] - z_low)
            return SphereObstacle(center=np.array([x, y, z]), radius_m=radius)

        if kind == "cylinder":
            radius = float(rng.uniform(*cfg.cylinder_radius_m))
            height = float(rng.uniform(*cfg.cylinder_height_m))
            z_low = 0.5 * height + cfg.inflation_margin_m
            if 2.0 * z_low >= self.world[2]:
                height = max(10.0, self.world[2] - 2.0 * cfg.inflation_margin_m - 1.0)
                z_low = 0.5 * height + cfg.inflation_margin_m
            z = rng.uniform(z_low, self.world[2] - z_low)
            return CylinderObstacle(center=np.array([x, y, z]), radius_m=radius, height_m=height)

        axes = rng.uniform(cfg.ellipsoid_axes_m[0], cfg.ellipsoid_axes_m[1], size=3)
        z_low = axes[2] + cfg.inflation_margin_m
        z = rng.uniform(z_low, self.world[2] - z_low)
        return EllipsoidObstacle(center=np.array([x, y, z]), axes_m=axes)

    def _within_path_budget(self, path_length_m: float) -> bool:
        max_command = self.config.vehicle.speed_command_range_mps[1]
        conservative_ground_speed = max(0.25, max_command - self.config.current.max_speed_mps)
        distance_budget = (
            conservative_ground_speed
            * self.config.environment.max_steps
            * self.config.vehicle.dt_s
            / self.config.feasibility.path_budget_factor
        )
        return bool(path_length_m <= distance_budget)


def _grid_axis(start: float, stop: float, resolution: float) -> np.ndarray:
    values = np.arange(start, stop + 0.5 * resolution, resolution, dtype=np.float64)
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    else:
        values[-1] = min(values[-1], stop)
    return np.unique(values)


def coarse_astar_path_length(
    start_m: np.ndarray,
    goal_m: np.ndarray,
    obstacles: list[Obstacle],
    world_size_m: tuple[float, float, float],
    legal_depth_m: tuple[float, float],
    resolution_m: float,
    inflation_m: float,
) -> float | None:
    """Return a conservative coarse-grid path length or ``None`` if blocked."""
    axes = (
        _grid_axis(0.0, world_size_m[0], resolution_m),
        _grid_axis(0.0, world_size_m[1], resolution_m),
        _grid_axis(legal_depth_m[0], legal_depth_m[1], resolution_m),
    )
    shape = tuple(len(axis) for axis in axes)
    blocked = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        point = np.array([axes[0][index[0]], axes[1][index[1]], axes[2][index[2]]])
        blocked[index] = collides(point, obstacles, inflation_m)

    def nearest_index(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(int(np.argmin(np.abs(axis - point[dim]))) for dim, axis in enumerate(axes))

    start_idx = nearest_index(np.asarray(start_m))
    goal_idx = nearest_index(np.asarray(goal_m))
    if blocked[start_idx] or blocked[goal_idx]:
        return None

    neighbor_offsets = [offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0)]

    def point_at(index: tuple[int, int, int]) -> np.ndarray:
        return np.array([axes[0][index[0]], axes[1][index[1]], axes[2][index[2]]])

    def heuristic(index: tuple[int, int, int]) -> float:
        return float(np.linalg.norm(point_at(index) - point_at(goal_idx)))

    queue: list[tuple[float, float, tuple[int, int, int]]] = [(heuristic(start_idx), 0.0, start_idx)]
    best_cost = {start_idx: 0.0}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > best_cost.get(current, np.inf) + 1e-9:
            continue
        if current == goal_idx:
            endpoint_correction = np.linalg.norm(start_m - point_at(start_idx)) + np.linalg.norm(
                goal_m - point_at(goal_idx)
            )
            return float(cost + endpoint_correction)
        current_point = point_at(current)
        for offset in neighbor_offsets:
            neighbor = tuple(current[dim] + offset[dim] for dim in range(3))
            if any(neighbor[dim] < 0 or neighbor[dim] >= shape[dim] for dim in range(3)):
                continue
            if blocked[neighbor]:
                continue
            step_cost = float(np.linalg.norm(point_at(neighbor) - current_point))
            candidate = cost + step_cost
            if candidate + 1e-9 < best_cost.get(neighbor, np.inf):
                best_cost[neighbor] = candidate
                heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
    return None


__all__ = [
    "Scenario",
    "ScenarioGenerationError",
    "ScenarioGenerator",
    "coarse_astar_path_length",
]
