"""Typed configuration shared by the staged REMUS simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _pair(value, cast=float):
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got {value!r}")
    return cast(value[0]), cast(value[1])


def _triple(value, cast=float):
    if len(value) != 3:
        raise ValueError(f"Expected three values, got {value!r}")
    return cast(value[0]), cast(value[1]), cast(value[2])


@dataclass(frozen=True)
class EnvironmentConfig:
    world_size_m: tuple[float, float, float]
    legal_depth_m: tuple[float, float]
    start_goal_depth_m: tuple[float, float]
    xy_margin_m: float
    start_goal_min_distance_m: float
    start_goal_max_distance_m: float
    goal_radius_m: float
    max_steps: int
    sensor_range_m: float
    oscillation_window: int
    oscillation_progress_m: float
    oscillation_radius_m: float
    timeout_as_truncation: bool


@dataclass(frozen=True)
class VehicleConfig:
    radius_m: float
    dt_s: float
    initial_speed_mps: float
    speed_state_range_mps: tuple[float, float]
    speed_command_range_mps: tuple[float, float]
    pitch_limit_deg: float
    pitch_rate_limit_deg_s: float
    yaw_rate_limit_deg_s: float
    pitch_rate_command_limit_deg_s: float
    yaw_rate_command_limit_deg_s: float
    pitch_command_delta_limit_deg_s: float
    yaw_command_delta_limit_deg_s: float
    tau_u_s: float
    tau_q_s: float
    tau_r_s: float


@dataclass(frozen=True)
class ObstacleConfig:
    count_range: tuple[int, int]
    inflation_margin_m: float
    start_goal_clearance_m: float
    boundary_clearance_m: float
    obstacle_separation_m: float
    sphere_radius_m: tuple[float, float]
    cylinder_radius_m: tuple[float, float]
    cylinder_height_m: tuple[float, float]
    ellipsoid_axes_m: tuple[float, float]
    max_generation_attempts: int
    types: tuple[str, ...] = ("sphere", "cylinder", "ellipsoid")


@dataclass(frozen=True)
class CurrentConfig:
    mode: str
    max_speed_mps: float
    background_speed_mps: tuple[float, float]
    vortex_count: tuple[int, int]
    vortex_strength_mps: tuple[float, float]
    vortex_core_radius_m: tuple[float, float]
    time_amplitude_mps: tuple[float, float]
    time_period_s: tuple[float, float]
    vertical_fraction: float


@dataclass(frozen=True)
class FeasibilityConfig:
    enabled: bool
    grid_resolution_m: float
    path_budget_factor: float
    max_scenario_attempts: int


@dataclass(frozen=True)
class RewardConfig:
    progress: float
    goal: float
    collision: float
    boundary: float
    depth: float
    dynamics: float
    oscillation: float
    clearance: float
    clearance_warning_m: float
    current: float
    energy: float
    energy_pitch_rate: float
    energy_yaw_rate: float
    smooth: float
    step: float
    dense_clip: float


@dataclass(frozen=True)
class SafetyConfig:
    cost_max: float
    boundary_warning_m: float
    depth_warning_m: float
    safe_pitch_fraction: float
    safe_rate_fraction: float
    obstacle_weight: float
    boundary_weight: float
    depth_weight: float
    dynamics_weight: float


@dataclass(frozen=True)
class Stage0Config:
    seed: int
    environment: EnvironmentConfig
    vehicle: VehicleConfig
    obstacles: ObstacleConfig
    current: CurrentConfig
    feasibility: FeasibilityConfig
    reward: RewardConfig
    safety: SafetyConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Stage0Config":
        env = dict(raw["environment"])
        env["world_size_m"] = _triple(env["world_size_m"])
        env["legal_depth_m"] = _pair(env["legal_depth_m"])
        env["start_goal_depth_m"] = _pair(env["start_goal_depth_m"])

        vehicle = dict(raw["vehicle"])
        vehicle["speed_state_range_mps"] = _pair(vehicle["speed_state_range_mps"])
        vehicle["speed_command_range_mps"] = _pair(vehicle["speed_command_range_mps"])

        obstacles = dict(raw["obstacles"])
        obstacles["types"] = tuple(
            str(value).lower()
            for value in obstacles.get("types", ("sphere", "cylinder", "ellipsoid"))
        )
        obstacles["count_range"] = _pair(obstacles["count_range"], int)
        for key in (
            "sphere_radius_m",
            "cylinder_radius_m",
            "cylinder_height_m",
            "ellipsoid_axes_m",
        ):
            obstacles[key] = _pair(obstacles[key])

        current = dict(raw["current"])
        current["vortex_count"] = _pair(current["vortex_count"], int)
        for key in (
            "background_speed_mps",
            "vortex_strength_mps",
            "vortex_core_radius_m",
            "time_amplitude_mps",
            "time_period_s",
        ):
            current[key] = _pair(current[key])

        config = cls(
            seed=int(raw["seed"]),
            environment=EnvironmentConfig(**env),
            vehicle=VehicleConfig(**vehicle),
            obstacles=ObstacleConfig(**obstacles),
            current=CurrentConfig(**current),
            feasibility=FeasibilityConfig(**raw["feasibility"]),
            reward=RewardConfig(**raw["reward"]),
            safety=SafetyConfig(**raw["safety"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.vehicle.dt_s <= 0:
            raise ValueError("vehicle.dt_s must be positive")
        if min(self.vehicle.tau_u_s, self.vehicle.tau_q_s, self.vehicle.tau_r_s) <= 0:
            raise ValueError("Actuator time constants must be positive")
        if self.vehicle.pitch_rate_command_limit_deg_s > self.vehicle.pitch_rate_limit_deg_s:
            raise ValueError("Pitch-rate command limit cannot exceed the physical state limit")
        if self.vehicle.yaw_rate_command_limit_deg_s > self.vehicle.yaw_rate_limit_deg_s:
            raise ValueError("Yaw-rate command limit cannot exceed the physical state limit")
        if self.environment.max_steps <= 0:
            raise ValueError("environment.max_steps must be positive")
        if self.environment.sensor_range_m <= 0:
            raise ValueError("environment.sensor_range_m must be positive")
        z_min, z_max = self.environment.legal_depth_m
        if not 0 <= z_min < z_max <= self.environment.world_size_m[2]:
            raise ValueError("Invalid legal depth range")
        if self.current.max_speed_mps < 0:
            raise ValueError("current.max_speed_mps cannot be negative")
        allowed_obstacles = {"sphere", "cylinder", "ellipsoid"}
        if not self.obstacles.types:
            raise ValueError("obstacles.types cannot be empty")
        if not set(self.obstacles.types).issubset(allowed_obstacles):
            raise ValueError(
                f"Unsupported obstacle type(s): {set(self.obstacles.types) - allowed_obstacles}"
            )
        if self.reward.dense_clip <= 0:
            raise ValueError("reward.dense_clip must be positive")
        if self.safety.cost_max <= 0:
            raise ValueError("safety.cost_max must be positive")


def load_config(path: str | Path) -> Stage0Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return Stage0Config.from_mapping(raw)


__all__ = ["Stage0Config", "load_config"]
