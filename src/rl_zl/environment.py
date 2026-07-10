"""Gymnasium-compatible REMUS-100-like 3-D AUV environment."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .compat import gym, spaces
from .config import Stage0Config, load_config
from .dynamics import DynamicsDiagnostics, REMUSPlanningDynamics, VehicleState, body_to_world_matrix
from .obstacles import (
    CylinderObstacle,
    EllipsoidObstacle,
    RAY_DIRECTIONS_BODY,
    SphereObstacle,
    collides,
    minimum_signed_distance,
    ray_distances,
)
from .scenario import Scenario, ScenarioGenerator


class REMUS100Env(gym.Env):
    """Planning-level AUV simulator with local range observations.

    The policy action is normalized to ``[-1, 1]^3`` and mapped to desired
    surge speed, pitch rate and yaw rate.  The 53-D observation contains only
    the documented local/task state; no obstacle map is exposed to the policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    observation_dim = 53

    def __init__(
        self,
        config: Stage0Config | str | Path,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        self.render_mode = render_mode
        self.world = np.asarray(self.config.environment.world_size_m, dtype=np.float64)
        self.legal_depth = self.config.environment.legal_depth_m
        self.inflation_m = self.config.vehicle.radius_m + self.config.obstacles.inflation_margin_m
        self.dynamics = REMUSPlanningDynamics(self.config.vehicle)
        self.scenario_generator = ScenarioGenerator(self.config)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self._master_rng = np.random.default_rng(self.config.seed)
        self.scenario: Scenario | None = None
        self.state: VehicleState | None = None
        self.previous_action = np.zeros(3, dtype=np.float64)
        self.previous_command = np.array(
            [self.config.vehicle.initial_speed_mps, 0.0, 0.0], dtype=np.float64
        )
        self._last_dynamics_diagnostics: DynamicsDiagnostics | None = None
        self.step_count = 0
        self.initial_goal_distance_m = 1.0
        self.trajectory: list[np.ndarray] = []
        window = self.config.environment.oscillation_window + 1
        self._recent_positions: deque[np.ndarray] = deque(maxlen=window)
        self._recent_goal_distances: deque[float] = deque(maxlen=window)
        self._last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if seed is None:
            scenario_seed = int(options.get("scenario_seed", self._master_rng.integers(0, 2**31 - 1)))
        else:
            scenario_seed = int(options.get("scenario_seed", seed))
            self._master_rng = np.random.default_rng(seed)
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(scenario_seed)

        scenario = self.scenario_generator.sample(scenario_seed)
        return self.reset_with_scenario(scenario)

    def reset_with_scenario(
        self,
        scenario: Scenario,
        *,
        initial_yaw_rad: float | None = None,
        initial_pitch_rad: float | None = None,
    ):
        """Initialize a caller-supplied scenario for deterministic Stage-0 cases."""
        self.scenario = scenario
        direction = scenario.goal_m - scenario.start_m
        horizontal = float(np.linalg.norm(direction[:2]))
        yaw = (
            float(np.arctan2(direction[1], direction[0]))
            if initial_yaw_rad is None
            else float(initial_yaw_rad)
        )
        pitch = (
            float(np.arctan2(direction[2], max(horizontal, 1e-9)))
            if initial_pitch_rad is None
            else float(initial_pitch_rad)
        )
        pitch = float(np.clip(pitch, -0.5 * self.dynamics.pitch_limit_rad, 0.5 * self.dynamics.pitch_limit_rad))
        self.state = VehicleState(
            position_m=scenario.start_m.copy(),
            pitch_rad=pitch,
            yaw_rad=yaw,
            speed_mps=self.config.vehicle.initial_speed_mps,
            pitch_rate_rad_s=0.0,
            yaw_rate_rad_s=0.0,
        )
        self.previous_action = np.zeros(3, dtype=np.float64)
        self.previous_command = np.array(
            [self.config.vehicle.initial_speed_mps, 0.0, 0.0], dtype=np.float64
        )
        self._last_dynamics_diagnostics = None
        self.step_count = 0
        self.initial_goal_distance_m = self._goal_distance()
        self.trajectory = [self.state.position_m.copy()]
        self._recent_positions.clear()
        self._recent_goal_distances.clear()
        self._recent_positions.append(self.state.position_m.copy())
        self._recent_goal_distances.append(self.initial_goal_distance_m)
        observation = self._observation()
        info = {
            "scenario_seed": scenario.seed,
            "scenario": scenario.to_dict(),
            "goal_distance_m": self.initial_goal_distance_m,
            "observation_dim": self.observation_dim,
        }
        self._last_info = info
        return observation, info

    def step(self, action):
        if self.state is None or self.scenario is None:
            raise RuntimeError("Call reset() before step().")
        normalized_action = np.asarray(action, dtype=np.float64)
        if normalized_action.shape != (3,) or not np.all(np.isfinite(normalized_action)):
            raise ValueError("Action must be a finite vector with shape (3,)")
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        old_goal_distance = self._goal_distance()
        current = self.scenario.current.velocity(
            self.state.position_m,
            self.step_count * self.config.vehicle.dt_s,
        )
        old_action = self.previous_action.copy()
        self.state, physical_command, dynamics_diagnostics = self.dynamics.step(
            self.state,
            normalized_action,
            current,
            previous_command=self.previous_command,
        )
        self.previous_action = normalized_action.copy()
        self.previous_command = physical_command.copy()
        self._last_dynamics_diagnostics = dynamics_diagnostics
        self.step_count += 1
        self.trajectory.append(self.state.position_m.copy())

        new_goal_distance = self._goal_distance()
        self._recent_positions.append(self.state.position_m.copy())
        self._recent_goal_distances.append(new_goal_distance)
        events = self._terminal_events(new_goal_distance)
        reward_components = self._reward_components(
            old_goal_distance,
            new_goal_distance,
            current,
            normalized_action,
            old_action,
            events,
        )
        reward = float(sum(reward_components.values()))

        failure_type = self._failure_type(events)
        success = bool(events["goal"] and failure_type is None)
        terminated = bool(success or (failure_type is not None and failure_type != "timeout"))
        truncated = bool(failure_type == "timeout" and self.config.environment.timeout_as_truncation)
        if failure_type == "timeout" and not self.config.environment.timeout_as_truncation:
            terminated = True

        safety_cost, safety_cost_normalized, safety_components = self._safety_cost(events)
        observation = self._observation()
        info = {
            "success": success,
            "failure_type": failure_type,
            "goal_distance_m": new_goal_distance,
            "minimum_obstacle_distance_m": self._minimum_obstacle_distance(),
            "current_velocity_mps": current.astype(np.float32),
            "physical_command": physical_command.astype(np.float32),
            "dynamics_diagnostics": dynamics_diagnostics.to_dict(),
            "reward_components": reward_components,
            "safety_cost": safety_cost,
            "safety_cost_normalized": safety_cost_normalized,
            "safety_components": safety_components,
            "events": events,
            "step_count": self.step_count,
        }
        self._last_info = info
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _terminal_events(self, goal_distance_m: float) -> dict[str, bool]:
        assert self.state is not None and self.scenario is not None
        position = self.state.position_m
        xy_out = bool(
            position[0] < 0.0
            or position[0] > self.world[0]
            or position[1] < 0.0
            or position[1] > self.world[1]
        )
        depth_out = bool(position[2] < self.legal_depth[0] or position[2] > self.legal_depth[1])
        dynamics_out = bool(
            self.state.speed_mps < self.config.vehicle.speed_state_range_mps[0] - 1e-9
            or self.state.speed_mps > self.config.vehicle.speed_state_range_mps[1] + 1e-9
            or abs(self.state.pitch_rate_rad_s) > self.dynamics.pitch_rate_limit_rad_s + 1e-9
            or abs(self.state.yaw_rate_rad_s) > self.dynamics.yaw_rate_limit_rad_s + 1e-9
            or (
                self._last_dynamics_diagnostics is not None
                and self._last_dynamics_diagnostics.hard_violation
            )
            or not self.state.finite()
        )
        collision = collides(self.state.position_m, self.scenario.obstacles, self.inflation_m)
        oscillation = self._is_oscillating()
        timeout = bool(
            self.step_count >= self.config.environment.max_steps
            and goal_distance_m > self.config.environment.goal_radius_m
        )
        return {
            "goal": bool(goal_distance_m <= self.config.environment.goal_radius_m),
            "collision": bool(collision),
            "boundary": xy_out,
            "depth": depth_out,
            "dynamics": dynamics_out,
            "oscillation": oscillation,
            "timeout": timeout,
        }

    @staticmethod
    def _failure_type(events: dict[str, bool]) -> str | None:
        for name in ("collision", "depth", "boundary", "dynamics", "oscillation", "timeout"):
            if events[name]:
                return name
        return None

    def _is_oscillating(self) -> bool:
        required = self.config.environment.oscillation_window + 1
        if len(self._recent_positions) < required:
            return False
        progress = self._recent_goal_distances[0] - self._recent_goal_distances[-1]
        points = np.asarray(self._recent_positions)
        extent = float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)))
        return bool(
            progress < self.config.environment.oscillation_progress_m
            and extent < self.config.environment.oscillation_radius_m
            and self._recent_goal_distances[-1] > self.config.environment.goal_radius_m
        )

    def _reward_components(
        self,
        old_distance_m: float,
        new_distance_m: float,
        current_mps: np.ndarray,
        action: np.ndarray,
        old_action: np.ndarray,
        events: dict[str, bool],
    ) -> dict[str, float]:
        assert self.state is not None and self.scenario is not None
        cfg = self.config.reward
        epsilon = 1e-9
        progress_m = old_distance_m - new_distance_m
        goal_direction = (self.scenario.goal_m - self.state.position_m) / (new_distance_m + epsilon)
        current_scale = max(self.config.current.max_speed_mps, epsilon)
        useful_current = float(
            np.clip(np.dot(current_mps, goal_direction) / current_scale, 0.0, 1.0)
        )
        positive_progress = max(0.0, progress_m)
        progress_fraction = float(
            np.clip(
                positive_progress
                / (self.config.vehicle.speed_state_range_mps[1] * self.config.vehicle.dt_s + epsilon),
                0.0,
                1.0,
            )
        )
        clearance = self._minimum_obstacle_distance()
        clearance_risk = max(0.0, (cfg.clearance_warning_m - clearance) / cfg.clearance_warning_m)
        clearance_risk = min(clearance_risk, 1.0)
        energy_proxy = (
            self.state.speed_mps**3
            + cfg.energy_pitch_rate * self.state.pitch_rate_rad_s**2
            + cfg.energy_yaw_rate * self.state.yaw_rate_rad_s**2
        )
        dense_components = {
            "progress": float(cfg.progress * progress_m / (self.initial_goal_distance_m + epsilon)),
            "clearance": float(-cfg.clearance * clearance_risk**2),
            "current": float(cfg.current * useful_current * progress_fraction),
            "energy": float(-cfg.energy * energy_proxy),
            "smooth": float(-cfg.smooth * np.dot(action - old_action, action - old_action)),
            "step": float(-cfg.step),
        }
        dense_raw = float(sum(dense_components.values()))
        dense_clipped = float(np.clip(dense_raw, -cfg.dense_clip, cfg.dense_clip))
        dense_components["dense_clip_adjustment"] = dense_clipped - dense_raw
        terminal_components = {
            "goal": float(cfg.goal if events["goal"] else 0.0),
            "collision": float(-cfg.collision if events["collision"] else 0.0),
            "boundary": float(-cfg.boundary if events["boundary"] else 0.0),
            "depth": float(-cfg.depth if events["depth"] else 0.0),
            "dynamics": float(-cfg.dynamics if events["dynamics"] else 0.0),
            "oscillation": float(-cfg.oscillation if events["oscillation"] else 0.0),
        }
        return {**dense_components, **terminal_components}

    def _safety_cost(self, events: dict[str, bool]) -> tuple[float, float, dict[str, float]]:
        assert self.state is not None
        cfg = self.config.safety
        obstacle_distance = self._minimum_obstacle_distance()
        obstacle_risk = np.clip(
            (self.config.reward.clearance_warning_m - obstacle_distance)
            / self.config.reward.clearance_warning_m,
            0.0,
            1.0,
        )
        x, y, z = self.state.position_m
        boundary_distance = min(x, self.world[0] - x, y, self.world[1] - y)
        boundary_risk = np.clip(
            (cfg.boundary_warning_m - boundary_distance) / cfg.boundary_warning_m,
            0.0,
            1.0,
        )
        depth_clearance = min(z - self.legal_depth[0], self.legal_depth[1] - z)
        depth_risk = np.clip(
            (cfg.depth_warning_m - depth_clearance) / cfg.depth_warning_m,
            0.0,
            1.0,
        )
        pitch_safe = cfg.safe_pitch_fraction * self.dynamics.pitch_limit_rad
        q_safe = cfg.safe_rate_fraction * self.dynamics.pitch_rate_limit_rad_s
        r_safe = cfg.safe_rate_fraction * self.dynamics.yaw_rate_limit_rad_s
        pitch_risk = max(
            0.0,
            (abs(self.state.pitch_rad) - pitch_safe) / (self.dynamics.pitch_limit_rad - pitch_safe + 1e-9),
        )
        q_risk = max(
            0.0,
            (abs(self.state.pitch_rate_rad_s) - q_safe)
            / (self.dynamics.pitch_rate_limit_rad_s - q_safe + 1e-9),
        )
        r_risk = max(
            0.0,
            (abs(self.state.yaw_rate_rad_s) - r_safe)
            / (self.dynamics.yaw_rate_limit_rad_s - r_safe + 1e-9),
        )
        dynamics_risk = min(3.0, pitch_risk + q_risk + r_risk)
        base_components = {
            "terminal_collision": float(events["collision"]),
            "terminal_boundary": float(events["boundary"]),
            "terminal_depth": float(events["depth"]),
            "obstacle_risk": float(cfg.obstacle_weight * obstacle_risk),
            "boundary_risk": float(cfg.boundary_weight * boundary_risk),
            "depth_risk": float(cfg.depth_weight * depth_risk),
            "dynamics_risk": float(cfg.dynamics_weight * dynamics_risk),
        }
        raw_cost = float(sum(base_components.values()))
        clipped_cost = float(np.clip(raw_cost, 0.0, cfg.cost_max))
        normalized_cost = clipped_cost / cfg.cost_max
        components = {
            **base_components,
            "raw_total": raw_cost,
            "clipped_total": clipped_cost,
            "normalized_total": normalized_cost,
        }
        return clipped_cost, normalized_cost, components

    def _observation(self) -> np.ndarray:
        assert self.state is not None and self.scenario is not None
        state = self.state
        rotation = body_to_world_matrix(state.pitch_rad, state.yaw_rad)
        goal_delta = self.scenario.goal_m - state.position_m
        goal_distance = float(np.linalg.norm(goal_delta))
        goal_direction_world = goal_delta / max(goal_distance, 1e-9)
        goal_direction_body = rotation.T @ goal_direction_world
        current = self.scenario.current.velocity(
            state.position_m,
            self.step_count * self.config.vehicle.dt_s,
        )
        ray_directions_world = (rotation @ RAY_DIRECTIONS_BODY.T).T
        rays = ray_distances(
            state.position_m,
            ray_directions_world,
            self.scenario.obstacles,
            self.config.environment.sensor_range_m,
            self.inflation_m,
        )
        _, speed_high = self.config.vehicle.speed_state_range_mps
        normalized_speed = state.speed_mps / speed_high
        current_scale = max(self.config.current.max_speed_mps, 1e-9)
        x, y, z = state.position_m
        depth_span = self.legal_depth[1] - self.legal_depth[0]
        boundary = np.array(
            [
                x / self.world[0],
                (self.world[0] - x) / self.world[0],
                y / self.world[1],
                (self.world[1] - y) / self.world[1],
                (z - self.legal_depth[0]) / depth_span,
                (self.legal_depth[1] - z) / depth_span,
            ]
        )
        observation = np.concatenate(
            [
                state.position_m / self.world,
                np.array([np.sin(state.pitch_rad), np.cos(state.pitch_rad)]),
                np.array([np.sin(state.yaw_rad), np.cos(state.yaw_rad)]),
                np.array(
                    [
                        normalized_speed,
                        state.pitch_rate_rad_s / self.dynamics.pitch_rate_limit_rad_s,
                        state.yaw_rate_rad_s / self.dynamics.yaw_rate_limit_rad_s,
                    ]
                ),
                goal_direction_body,
                np.array([goal_distance / np.linalg.norm(self.world)]),
                current / current_scale if self.config.current.max_speed_mps > 0.0 else np.zeros(3),
                rays / self.config.environment.sensor_range_m,
                boundary,
                self.previous_action,
                np.array([self.step_count / self.config.environment.max_steps]),
            ]
        )
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(f"Observation shape drifted to {observation.shape}")
        observation = np.clip(observation, -1.0, 1.0).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Observation contains NaN or Inf")
        return observation

    def _goal_distance(self) -> float:
        assert self.state is not None and self.scenario is not None
        return float(np.linalg.norm(self.scenario.goal_m - self.state.position_m))

    def _minimum_obstacle_distance(self) -> float:
        assert self.state is not None and self.scenario is not None
        return minimum_signed_distance(
            self.state.position_m,
            self.scenario.obstacles,
            self.inflation_m,
            default=self.config.environment.sensor_range_m,
        )

    def render(self, output_path: str | Path | None = None):
        if self.state is None or self.scenario is None:
            return None
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0.0, self.world[0])
        ax.set_ylim(0.0, self.world[1])
        ax.set_zlim(self.world[2], 0.0)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        ax.set_title("REMUS-100-like Stage-0 trajectory")

        for obstacle in self.scenario.obstacles:
            self._draw_obstacle(ax, obstacle)
        trajectory = np.asarray(self.trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#1f77b4", linewidth=2, label="AUV")
        ax.scatter(*self.scenario.start_m, color="#2ca02c", s=55, marker="o", label="Start")
        ax.scatter(*self.scenario.goal_m, color="#d62728", s=75, marker="*", label="Goal")
        ax.legend(loc="upper right")

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
            plt.close(fig)
            return image
        if self.render_mode != "human":
            plt.close(fig)
        return None

    def render_ray_diagnostic(self, output_path: str | Path) -> None:
        """Save a geometry diagnostic showing all 26 body-aligned rays."""
        if self.state is None or self.scenario is None:
            raise RuntimeError("Call reset() before rendering rays.")
        import matplotlib.pyplot as plt

        rotation = body_to_world_matrix(self.state.pitch_rad, self.state.yaw_rad)
        directions_world = (rotation @ RAY_DIRECTIONS_BODY.T).T
        distances = ray_distances(
            self.state.position_m,
            directions_world,
            self.scenario.obstacles,
            self.config.environment.sensor_range_m,
            self.inflation_m,
        )
        endpoints = self.state.position_m[None, :] + directions_world * distances[:, None]

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0.0, self.world[0])
        ax.set_ylim(0.0, self.world[1])
        ax.set_zlim(self.world[2], 0.0)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        ax.set_title("Heading-aligned 26-ray geometry diagnostic")
        for obstacle in self.scenario.obstacles:
            self._draw_obstacle(ax, obstacle)
        origin = self.state.position_m
        for endpoint, distance in zip(endpoints, distances, strict=True):
            color = "#d62728" if distance < self.config.environment.sensor_range_m else "#9ecae1"
            ax.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=color,
                linewidth=0.8,
                alpha=0.85,
            )
        ax.scatter(*origin, color="#111111", s=40, label="AUV")
        ax.legend(loc="upper right")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _draw_obstacle(self, ax, obstacle) -> None:
        color = "#7f8c8d"
        alpha = 0.20
        if isinstance(obstacle, SphereObstacle):
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 12j]
            x = obstacle.center[0] + obstacle.radius_m * np.cos(u) * np.sin(v)
            y = obstacle.center[1] + obstacle.radius_m * np.sin(u) * np.sin(v)
            z = obstacle.center[2] + obstacle.radius_m * np.cos(v)
        elif isinstance(obstacle, EllipsoidObstacle):
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 12j]
            x = obstacle.center[0] + obstacle.axes_m[0] * np.cos(u) * np.sin(v)
            y = obstacle.center[1] + obstacle.axes_m[1] * np.sin(u) * np.sin(v)
            z = obstacle.center[2] + obstacle.axes_m[2] * np.cos(v)
        elif isinstance(obstacle, CylinderObstacle):
            theta, z_grid = np.mgrid[0 : 2 * np.pi : 24j, obstacle.z_min : obstacle.z_max : 2j]
            x = obstacle.center[0] + obstacle.radius_m * np.cos(theta)
            y = obstacle.center[1] + obstacle.radius_m * np.sin(theta)
            z = z_grid
        else:
            return
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    def close(self):
        return None


__all__ = ["REMUS100Env"]
