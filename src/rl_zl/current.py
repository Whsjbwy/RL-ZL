"""Deterministic analytic current field for Stage-0 validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CurrentConfig


@dataclass(frozen=True)
class Vortex:
    center_xy_m: np.ndarray
    strength_mps: float
    core_radius_m: float
    rotation_sign: float
    phase_rad: float

    def velocity(self, position_m: np.ndarray, time_s: float, depth_m: float, world_depth_m: float) -> np.ndarray:
        delta = np.asarray(position_m, dtype=np.float64)[:2] - self.center_xy_m
        radius = float(np.linalg.norm(delta))
        if radius < 1e-9:
            tangent = np.zeros(2, dtype=np.float64)
        else:
            radial_ratio = radius / self.core_radius_m
            speed = self.strength_mps * radial_ratio * np.exp(0.5 * (1.0 - radial_ratio**2))
            tangent = self.rotation_sign * speed * np.array([-delta[1], delta[0]]) / radius

        depth_factor = float(np.sin(np.pi * np.clip(depth_m / world_depth_m, 0.0, 1.0)) ** 2)
        horizontal = tangent * (0.35 + 0.65 * depth_factor)
        vertical = (
            0.08
            * self.strength_mps
            * np.exp(-0.5 * (radius / self.core_radius_m) ** 2)
            * np.sin(2.0 * np.pi * time_s / 300.0 + self.phase_rad)
        )
        return np.array([horizontal[0], horizontal[1], vertical], dtype=np.float64)


class AnalyticCurrentField:
    def __init__(
        self,
        background_mps: np.ndarray,
        vortices: list[Vortex],
        time_direction: np.ndarray,
        time_amplitude_mps: float,
        time_period_s: float,
        time_phase_rad: float,
        max_speed_mps: float,
        vertical_fraction: float,
        world_depth_m: float,
    ):
        self.background_mps = np.asarray(background_mps, dtype=np.float64)
        self.vortices = list(vortices)
        self.time_direction = np.asarray(time_direction, dtype=np.float64)
        self.time_amplitude_mps = float(time_amplitude_mps)
        self.time_period_s = float(time_period_s)
        self.time_phase_rad = float(time_phase_rad)
        self.max_speed_mps = float(max_speed_mps)
        self.vertical_fraction = float(vertical_fraction)
        self.world_depth_m = float(world_depth_m)

    @classmethod
    def sample(
        cls,
        config: CurrentConfig,
        rng: np.random.Generator,
        world_size_m: tuple[float, float, float],
    ) -> "AnalyticCurrentField":
        if config.mode.lower() == "none" or config.max_speed_mps <= 0.0:
            return cls(
                background_mps=np.zeros(3),
                vortices=[],
                time_direction=np.zeros(3),
                time_amplitude_mps=0.0,
                time_period_s=1.0,
                time_phase_rad=0.0,
                max_speed_mps=0.0,
                vertical_fraction=config.vertical_fraction,
                world_depth_m=world_size_m[2],
            )

        bg_speed = rng.uniform(*config.background_speed_mps)
        bg_angle = rng.uniform(-np.pi, np.pi)
        background = np.array([bg_speed * np.cos(bg_angle), bg_speed * np.sin(bg_angle), 0.0])

        n_vortices = int(rng.integers(config.vortex_count[0], config.vortex_count[1] + 1))
        vortices = []
        for _ in range(n_vortices):
            vortices.append(
                Vortex(
                    center_xy_m=np.array(
                        [rng.uniform(0.1, 0.9) * world_size_m[0], rng.uniform(0.1, 0.9) * world_size_m[1]]
                    ),
                    strength_mps=float(rng.uniform(*config.vortex_strength_mps)),
                    core_radius_m=float(rng.uniform(*config.vortex_core_radius_m)),
                    rotation_sign=float(rng.choice([-1.0, 1.0])),
                    phase_rad=float(rng.uniform(-np.pi, np.pi)),
                )
            )

        time_angle = rng.uniform(-np.pi, np.pi)
        vertical_sign = rng.choice([-1.0, 1.0])
        time_direction = np.array(
            [
                np.cos(time_angle),
                np.sin(time_angle),
                vertical_sign * config.vertical_fraction,
            ],
            dtype=np.float64,
        )
        time_direction /= max(np.linalg.norm(time_direction), 1e-12)
        return cls(
            background_mps=background,
            vortices=vortices,
            time_direction=time_direction,
            time_amplitude_mps=float(rng.uniform(*config.time_amplitude_mps)),
            time_period_s=float(rng.uniform(*config.time_period_s)),
            time_phase_rad=float(rng.uniform(-np.pi, np.pi)),
            max_speed_mps=config.max_speed_mps,
            vertical_fraction=config.vertical_fraction,
            world_depth_m=world_size_m[2],
        )

    def velocity(self, position_m: np.ndarray, time_s: float) -> np.ndarray:
        if self.max_speed_mps <= 0.0:
            return np.zeros(3, dtype=np.float64)
        position = np.asarray(position_m, dtype=np.float64)
        velocity = self.background_mps.copy()
        for vortex in self.vortices:
            velocity += vortex.velocity(position, time_s, position[2], self.world_depth_m)
        velocity += (
            self.time_amplitude_mps
            * np.sin(2.0 * np.pi * time_s / self.time_period_s + self.time_phase_rad)
            * self.time_direction
        )
        horizontal_norm = float(np.linalg.norm(velocity[:2]))
        vertical_cap = self.max_speed_mps * self.vertical_fraction
        velocity[2] = float(np.clip(velocity[2], -vertical_cap, vertical_cap))
        norm = float(np.linalg.norm(velocity))
        if norm > self.max_speed_mps:
            velocity *= self.max_speed_mps / norm
        if horizontal_norm == 0.0 and abs(velocity[2]) < 1e-15:
            return np.zeros(3, dtype=np.float64)
        return velocity

    def to_dict(self) -> dict:
        return {
            "background_mps": self.background_mps.tolist(),
            "vortices": [
                {
                    "center_xy_m": vortex.center_xy_m.tolist(),
                    "strength_mps": vortex.strength_mps,
                    "core_radius_m": vortex.core_radius_m,
                    "rotation_sign": vortex.rotation_sign,
                    "phase_rad": vortex.phase_rad,
                }
                for vortex in self.vortices
            ],
            "time_direction": self.time_direction.tolist(),
            "time_amplitude_mps": self.time_amplitude_mps,
            "time_period_s": self.time_period_s,
            "time_phase_rad": self.time_phase_rad,
            "max_speed_mps": self.max_speed_mps,
        }


__all__ = ["AnalyticCurrentField", "Vortex"]

