"""REMUS-100-like planning-level dynamics.

All internal angles use radians.  Positive depth and positive pitch point
downward, matching the experiment document's z-down convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VehicleConfig


def wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def body_to_world_matrix(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Rotation matrix with zero roll and a z-down pitch convention."""
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array(
        [
            [cy * cp, -sy, -cy * sp],
            [sy * cp, cy, -sy * sp],
            [sp, 0.0, cp],
        ],
        dtype=np.float64,
    )


@dataclass
class VehicleState:
    position_m: np.ndarray
    pitch_rad: float
    yaw_rad: float
    speed_mps: float
    pitch_rate_rad_s: float
    yaw_rate_rad_s: float

    def copy(self) -> "VehicleState":
        return VehicleState(
            position_m=self.position_m.copy(),
            pitch_rad=float(self.pitch_rad),
            yaw_rad=float(self.yaw_rad),
            speed_mps=float(self.speed_mps),
            pitch_rate_rad_s=float(self.pitch_rate_rad_s),
            yaw_rate_rad_s=float(self.yaw_rate_rad_s),
        )

    def finite(self) -> bool:
        values = np.concatenate(
            [
                np.asarray(self.position_m, dtype=np.float64),
                np.array(
                    [
                        self.pitch_rad,
                        self.yaw_rad,
                        self.speed_mps,
                        self.pitch_rate_rad_s,
                        self.yaw_rate_rad_s,
                    ]
                ),
            ]
        )
        return bool(np.all(np.isfinite(values)))


@dataclass(frozen=True)
class DynamicsDiagnostics:
    raw_command: np.ndarray
    limited_command: np.ndarray
    preclip_speed_mps: float
    preclip_pitch_rate_rad_s: float
    preclip_yaw_rate_rad_s: float
    preclip_pitch_rad: float
    speed_clipped: bool
    pitch_rate_clipped: bool
    yaw_rate_clipped: bool
    pitch_clipped: bool
    hard_violation: bool

    def to_dict(self) -> dict:
        return {
            "raw_command": self.raw_command.tolist(),
            "limited_command": self.limited_command.tolist(),
            "preclip_speed_mps": self.preclip_speed_mps,
            "preclip_pitch_rate_rad_s": self.preclip_pitch_rate_rad_s,
            "preclip_yaw_rate_rad_s": self.preclip_yaw_rate_rad_s,
            "preclip_pitch_rad": self.preclip_pitch_rad,
            "speed_clipped": self.speed_clipped,
            "pitch_rate_clipped": self.pitch_rate_clipped,
            "yaw_rate_clipped": self.yaw_rate_clipped,
            "pitch_clipped": self.pitch_clipped,
            "hard_violation": self.hard_violation,
        }


class REMUSPlanningDynamics:
    def __init__(self, config: VehicleConfig):
        self.config = config
        self.pitch_limit_rad = np.deg2rad(config.pitch_limit_deg)
        self.pitch_rate_limit_rad_s = np.deg2rad(config.pitch_rate_limit_deg_s)
        self.yaw_rate_limit_rad_s = np.deg2rad(config.yaw_rate_limit_deg_s)
        self.pitch_rate_command_limit_rad_s = np.deg2rad(config.pitch_rate_command_limit_deg_s)
        self.yaw_rate_command_limit_rad_s = np.deg2rad(config.yaw_rate_command_limit_deg_s)
        self.pitch_command_delta_limit_rad_s = np.deg2rad(config.pitch_command_delta_limit_deg_s)
        self.yaw_command_delta_limit_rad_s = np.deg2rad(config.yaw_command_delta_limit_deg_s)

    def action_to_command(self, normalized_action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (3,):
            raise ValueError(f"Action must have shape (3,), got {action.shape}")
        speed_low, speed_high = self.config.speed_command_range_mps
        speed_command = speed_low + 0.5 * (action[0] + 1.0) * (speed_high - speed_low)
        return np.array(
            [
                speed_command,
                action[1] * self.pitch_rate_command_limit_rad_s,
                action[2] * self.yaw_rate_command_limit_rad_s,
            ],
            dtype=np.float64,
        )

    def step(
        self,
        state: VehicleState,
        normalized_action: np.ndarray,
        current_velocity_mps: np.ndarray,
        previous_command: np.ndarray | None = None,
    ) -> tuple[VehicleState, np.ndarray, DynamicsDiagnostics]:
        """Advance one step using first-order actuators and semi-implicit Euler."""
        raw_command = self.action_to_command(normalized_action)
        command = raw_command.copy()
        if previous_command is not None:
            previous = np.asarray(previous_command, dtype=np.float64)
            if previous.shape != (3,):
                raise ValueError(f"previous_command must have shape (3,), got {previous.shape}")
            command[1] = np.clip(
                command[1],
                previous[1] - self.pitch_command_delta_limit_rad_s,
                previous[1] + self.pitch_command_delta_limit_rad_s,
            )
            command[2] = np.clip(
                command[2],
                previous[2] - self.yaw_command_delta_limit_rad_s,
                previous[2] + self.yaw_command_delta_limit_rad_s,
            )
        dt = self.config.dt_s

        preclip_speed = state.speed_mps + dt * (command[0] - state.speed_mps) / self.config.tau_u_s
        preclip_pitch_rate = state.pitch_rate_rad_s + dt * (
            command[1] - state.pitch_rate_rad_s
        ) / self.config.tau_q_s
        preclip_yaw_rate = state.yaw_rate_rad_s + dt * (
            command[2] - state.yaw_rate_rad_s
        ) / self.config.tau_r_s

        speed = float(np.clip(preclip_speed, *self.config.speed_state_range_mps))
        pitch_rate = float(
            np.clip(preclip_pitch_rate, -self.pitch_rate_limit_rad_s, self.pitch_rate_limit_rad_s)
        )
        yaw_rate = float(np.clip(preclip_yaw_rate, -self.yaw_rate_limit_rad_s, self.yaw_rate_limit_rad_s))

        preclip_pitch = float(state.pitch_rad + pitch_rate * dt)
        pitch = float(np.clip(preclip_pitch, -self.pitch_limit_rad, self.pitch_limit_rad))
        yaw = wrap_angle(state.yaw_rad + yaw_rate * dt)
        relative_velocity = np.array(
            [
                speed * np.cos(pitch) * np.cos(yaw),
                speed * np.cos(pitch) * np.sin(yaw),
                speed * np.sin(pitch),
            ],
            dtype=np.float64,
        )
        current = np.asarray(current_velocity_mps, dtype=np.float64)
        if current.shape != (3,):
            raise ValueError(f"Current must have shape (3,), got {current.shape}")
        position = state.position_m + (relative_velocity + current) * dt

        next_state = VehicleState(
            position_m=position,
            pitch_rad=pitch,
            yaw_rad=yaw,
            speed_mps=speed,
            pitch_rate_rad_s=pitch_rate,
            yaw_rate_rad_s=yaw_rate,
        )
        speed_clipped = not np.isclose(speed, preclip_speed)
        pitch_rate_clipped = not np.isclose(pitch_rate, preclip_pitch_rate)
        yaw_rate_clipped = not np.isclose(yaw_rate, preclip_yaw_rate)
        pitch_clipped = not np.isclose(pitch, preclip_pitch)
        diagnostics = DynamicsDiagnostics(
            raw_command=raw_command,
            limited_command=command.copy(),
            preclip_speed_mps=float(preclip_speed),
            preclip_pitch_rate_rad_s=float(preclip_pitch_rate),
            preclip_yaw_rate_rad_s=float(preclip_yaw_rate),
            preclip_pitch_rad=float(preclip_pitch),
            speed_clipped=bool(speed_clipped),
            pitch_rate_clipped=bool(pitch_rate_clipped),
            yaw_rate_clipped=bool(yaw_rate_clipped),
            pitch_clipped=bool(pitch_clipped),
            hard_violation=bool(
                speed_clipped or pitch_rate_clipped or yaw_rate_clipped or pitch_clipped
            ),
        )
        return next_state, command, diagnostics


__all__ = [
    "REMUSPlanningDynamics",
    "DynamicsDiagnostics",
    "VehicleState",
    "body_to_world_matrix",
    "wrap_angle",
]
