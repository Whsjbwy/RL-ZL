"""Validated Stage-1 SAC and curriculum configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import Stage0Config, load_config


def _int_tuple(values) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _float_pair(values) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"Expected two values, got {values!r}")
    return float(values[0]), float(values[1])


def _int_pair(values) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"Expected two values, got {values!r}")
    return int(values[0]), int(values[1])


@dataclass(frozen=True)
class SACAlgorithmConfig:
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    learning_rate: float
    gamma: float
    tau: float
    initial_alpha: float
    automatic_entropy_tuning: bool
    target_entropy: float | None
    log_std_bounds: tuple[float, float]
    gradient_clip_norm: float
    target_update_interval: int


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int
    batch_size: int
    learning_starts: int
    update_after: int
    updates_per_step: int


@dataclass(frozen=True)
class TrainingLoopConfig:
    evaluation_interval_steps: int
    evaluation_episodes: int
    final_evaluation_episodes: int
    evaluation_seed: int
    checkpoint_interval_steps: int
    log_interval_steps: int


@dataclass(frozen=True)
class CurriculumStageConfig:
    name: str
    obstacle_types: tuple[str, ...]
    obstacle_count_range: tuple[int, int]
    obstacle_radius_m: tuple[float, float]
    goal_radius_m: float
    current_mode: str
    current_max_speed_mps: float
    background_speed_mps: tuple[float, float]
    vortex_count: tuple[int, int]
    vortex_strength_mps: tuple[float, float]
    time_amplitude_mps: tuple[float, float]
    minimum_training_steps: int
    maximum_training_steps: int
    promotion_success_rate: float
    promotion_collision_rate: float


@dataclass(frozen=True)
class Stage1TrainingConfig:
    seed: int
    base_environment_path: Path
    output_dir: Path
    device: str
    deterministic_torch: bool
    sac: SACAlgorithmConfig
    replay: ReplayConfig
    training: TrainingLoopConfig
    curriculum: tuple[CurriculumStageConfig, ...]

    def load_base_environment(self) -> Stage0Config:
        return load_config(self.base_environment_path)

    def validate(self) -> None:
        if not self.curriculum:
            raise ValueError("Stage 1 requires at least one curriculum stage")
        if len(set(stage.name for stage in self.curriculum)) != len(self.curriculum):
            raise ValueError("Curriculum stage names must be unique")
        if min(self.sac.actor_hidden_dims + self.sac.critic_hidden_dims) <= 0:
            raise ValueError("All SAC hidden dimensions must be positive")
        if not 0.0 < self.sac.gamma <= 1.0:
            raise ValueError("sac.gamma must be in (0, 1]")
        if not 0.0 < self.sac.tau <= 1.0:
            raise ValueError("sac.tau must be in (0, 1]")
        if self.sac.learning_rate <= 0.0 or self.sac.initial_alpha <= 0.0:
            raise ValueError("SAC learning rate and alpha must be positive")
        if self.replay.batch_size <= 1 or self.replay.capacity < self.replay.batch_size:
            raise ValueError("Replay capacity must be at least one batch")
        if self.replay.update_after < self.replay.batch_size:
            raise ValueError("replay.update_after must be at least replay.batch_size")
        if self.training.final_evaluation_episodes < 100:
            raise ValueError("V4 Stage-1 final evaluation requires at least 100 episodes")
        for stage in self.curriculum:
            if stage.minimum_training_steps < 0:
                raise ValueError(f"{stage.name}: minimum_training_steps cannot be negative")
            if stage.maximum_training_steps < stage.minimum_training_steps:
                raise ValueError(f"{stage.name}: maximum_training_steps is too small")
            if not 0.0 <= stage.promotion_success_rate <= 1.0:
                raise ValueError(f"{stage.name}: invalid promotion success rate")
            if not 0.0 <= stage.promotion_collision_rate <= 1.0:
                raise ValueError(f"{stage.name}: invalid promotion collision rate")
            if stage.current_max_speed_mps < 0.0:
                raise ValueError(f"{stage.name}: current speed cannot be negative")


def _parse_curriculum_stage(raw: Mapping[str, Any]) -> CurriculumStageConfig:
    return CurriculumStageConfig(
        name=str(raw["name"]),
        obstacle_types=tuple(str(value).lower() for value in raw["obstacle_types"]),
        obstacle_count_range=_int_pair(raw["obstacle_count_range"]),
        obstacle_radius_m=_float_pair(raw["obstacle_radius_m"]),
        goal_radius_m=float(raw["goal_radius_m"]),
        current_mode=str(raw["current_mode"]),
        current_max_speed_mps=float(raw["current_max_speed_mps"]),
        background_speed_mps=_float_pair(raw["background_speed_mps"]),
        vortex_count=_int_pair(raw["vortex_count"]),
        vortex_strength_mps=_float_pair(raw["vortex_strength_mps"]),
        time_amplitude_mps=_float_pair(raw["time_amplitude_mps"]),
        minimum_training_steps=int(raw["minimum_training_steps"]),
        maximum_training_steps=int(raw["maximum_training_steps"]),
        promotion_success_rate=float(raw["promotion_success_rate"]),
        promotion_collision_rate=float(raw["promotion_collision_rate"]),
    )


def load_stage1_config(path: str | Path) -> Stage1TrainingConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    sac_raw = dict(raw["sac"])
    sac_raw["actor_hidden_dims"] = _int_tuple(sac_raw["actor_hidden_dims"])
    sac_raw["critic_hidden_dims"] = _int_tuple(sac_raw["critic_hidden_dims"])
    sac_raw["log_std_bounds"] = _float_pair(sac_raw["log_std_bounds"])
    if sac_raw.get("target_entropy") is not None:
        sac_raw["target_entropy"] = float(sac_raw["target_entropy"])

    base_path = Path(raw["base_environment"])
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    output_path = Path(raw["output_dir"])
    if not output_path.is_absolute():
        output_path = config_path.parent.parent / output_path

    config = Stage1TrainingConfig(
        seed=int(raw["seed"]),
        base_environment_path=base_path.resolve(),
        output_dir=output_path.resolve(),
        device=str(raw.get("device", "auto")),
        deterministic_torch=bool(raw.get("deterministic_torch", True)),
        sac=SACAlgorithmConfig(**sac_raw),
        replay=ReplayConfig(**raw["replay"]),
        training=TrainingLoopConfig(**raw["training"]),
        curriculum=tuple(_parse_curriculum_stage(item) for item in raw["curriculum"]),
    )
    config.validate()
    return config


def apply_curriculum_stage(
    base: Stage0Config,
    stage: CurriculumStageConfig,
) -> Stage0Config:
    """Return an immutable environment config with one V4 curriculum applied."""
    environment = replace(base.environment, goal_radius_m=stage.goal_radius_m)
    obstacles = replace(
        base.obstacles,
        types=stage.obstacle_types,
        count_range=stage.obstacle_count_range,
        sphere_radius_m=stage.obstacle_radius_m,
        cylinder_radius_m=stage.obstacle_radius_m,
    )
    current = replace(
        base.current,
        mode=stage.current_mode,
        max_speed_mps=stage.current_max_speed_mps,
        background_speed_mps=stage.background_speed_mps,
        vortex_count=stage.vortex_count,
        vortex_strength_mps=stage.vortex_strength_mps,
        time_amplitude_mps=stage.time_amplitude_mps,
    )
    configured = replace(base, environment=environment, obstacles=obstacles, current=current)
    configured.validate()
    return configured


__all__ = [
    "CurriculumStageConfig",
    "ReplayConfig",
    "SACAlgorithmConfig",
    "Stage1TrainingConfig",
    "TrainingLoopConfig",
    "apply_curriculum_stage",
    "load_stage1_config",
]
