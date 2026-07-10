"""Failure-retaining evaluation for Stage-1 teacher checkpoints."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .config import Stage0Config
from .environment import REMUS100Env


class Policy(Protocol):
    def select_action(self, observation, deterministic: bool = False) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeMetrics:
    seed: int
    success: bool
    failure_type: str | None
    episode_return: float
    steps: int
    path_length_m: float
    travel_time_s: float
    energy_proxy: float
    minimum_clearance_m: float


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    safety_failure_rate: float
    mean_return: float
    mean_path_length_success_m: float | None
    mean_path_length_all_m: float
    mean_travel_time_success_s: float | None
    mean_energy_success: float | None
    mean_minimum_clearance_m: float
    failure_counts: dict[str, int]
    records: tuple[EpisodeMetrics, ...]

    def stage1_gate_passes(
        self,
        success_rate_threshold: float = 0.90,
        collision_rate_limit: float = 0.10,
    ) -> bool:
        # V4 specifies strictly greater than 90% for Stage 1.
        return bool(
            self.success_rate > success_rate_threshold
            and self.collision_rate <= collision_rate_limit
        )

    def to_dict(self, include_records: bool = True) -> dict:
        result = {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "safety_failure_rate": self.safety_failure_rate,
            "mean_return": self.mean_return,
            "mean_path_length_success_m": self.mean_path_length_success_m,
            "mean_path_length_all_m": self.mean_path_length_all_m,
            "mean_travel_time_success_s": self.mean_travel_time_success_s,
            "mean_energy_success": self.mean_energy_success,
            "mean_minimum_clearance_m": self.mean_minimum_clearance_m,
            "failure_counts": self.failure_counts,
        }
        if include_records:
            result["records"] = [asdict(record) for record in self.records]
        return result


def evaluate_policy(
    policy: Policy,
    environment_config: Stage0Config,
    episodes: int,
    base_seed: int,
    deterministic: bool = True,
) -> EvaluationSummary:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    environment = REMUS100Env(environment_config)
    records: list[EpisodeMetrics] = []
    failure_counts: Counter[str] = Counter()

    for episode_index in range(episodes):
        episode_seed = int(base_seed + episode_index)
        observation, _ = environment.reset(seed=episode_seed)
        terminated = truncated = False
        episode_return = 0.0
        energy_proxy = 0.0
        minimum_clearance = float("inf")
        info: dict = {"success": False, "failure_type": None}

        while not (terminated or truncated):
            action = policy.select_action(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_return += float(reward)
            minimum_clearance = min(
                minimum_clearance,
                float(info["minimum_obstacle_distance_m"]),
            )
            assert environment.state is not None
            reward_cfg = environment_config.reward
            energy_proxy += (
                environment.state.speed_mps**3
                + reward_cfg.energy_pitch_rate * environment.state.pitch_rate_rad_s**2
                + reward_cfg.energy_yaw_rate * environment.state.yaw_rate_rad_s**2
            ) * environment_config.vehicle.dt_s

        trajectory = np.asarray(environment.trajectory, dtype=np.float64)
        path_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
        success = bool(info.get("success", False))
        failure_type = info.get("failure_type")
        failure_counts["success" if success else str(failure_type or "unknown")] += 1
        records.append(
            EpisodeMetrics(
                seed=episode_seed,
                success=success,
                failure_type=failure_type,
                episode_return=episode_return,
                steps=int(info.get("step_count", len(trajectory) - 1)),
                path_length_m=path_length,
                travel_time_s=(len(trajectory) - 1) * environment_config.vehicle.dt_s,
                energy_proxy=float(energy_proxy),
                minimum_clearance_m=float(minimum_clearance),
            )
        )
    environment.close()

    successful = [record for record in records if record.success]
    safety_failures = sum(
        failure_counts.get(name, 0) for name in ("collision", "boundary", "depth", "dynamics")
    )

    def success_mean(attribute: str) -> float | None:
        if not successful:
            return None
        return float(np.mean([getattr(record, attribute) for record in successful]))

    return EvaluationSummary(
        episodes=episodes,
        success_rate=len(successful) / episodes,
        collision_rate=failure_counts.get("collision", 0) / episodes,
        timeout_rate=failure_counts.get("timeout", 0) / episodes,
        safety_failure_rate=safety_failures / episodes,
        mean_return=float(np.mean([record.episode_return for record in records])),
        mean_path_length_success_m=success_mean("path_length_m"),
        mean_path_length_all_m=float(np.mean([record.path_length_m for record in records])),
        mean_travel_time_success_s=success_mean("travel_time_s"),
        mean_energy_success=success_mean("energy_proxy"),
        mean_minimum_clearance_m=float(np.mean([record.minimum_clearance_m for record in records])),
        failure_counts=dict(failure_counts),
        records=tuple(records),
    )


__all__ = ["EpisodeMetrics", "EvaluationSummary", "Policy", "evaluate_policy"]
