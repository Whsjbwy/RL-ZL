"""Stage-1 SAC teacher training loop with curriculum gates and audit logs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .environment import REMUS100Env
from .evaluation import EvaluationSummary, evaluate_policy
from .replay import ReplayBuffer
from .sac import SACAgent, SACUpdateMetrics, set_global_seeds
from .training_config import (
    Stage1TrainingConfig,
    apply_curriculum_stage,
    confirmation_due,
    curriculum_gate_episode_count,
    derive_resume_seed,
    next_evaluation_step,
    updates_allowed,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _episode_diagnostics(environment: REMUS100Env, info: dict[str, Any]) -> dict[str, Any]:
    """Retain scenario and terminal dynamics fields needed for failure analysis."""
    assert environment.scenario is not None and environment.state is not None
    scenario = environment.scenario
    state = environment.state
    goal_delta = scenario.goal_m - scenario.start_m
    horizontal_distance = float(np.linalg.norm(goal_delta[:2]))
    return {
        "start_m": [float(value) for value in scenario.start_m],
        "goal_m": [float(value) for value in scenario.goal_m],
        "vertical_delta_m": float(goal_delta[2]),
        "initial_goal_pitch_deg": float(
            np.rad2deg(np.arctan2(goal_delta[2], max(horizontal_distance, 1e-9)))
        ),
        "obstacle_count": len(scenario.obstacles),
        "terminal_position_m": [float(value) for value in state.position_m],
        "terminal_goal_distance_m": float(
            info.get("goal_distance_m", np.linalg.norm(scenario.goal_m - state.position_m))
        ),
        "terminal_pitch_deg": float(np.rad2deg(state.pitch_rad)),
        "terminal_pitch_rate_deg_s": float(np.rad2deg(state.pitch_rate_rad_s)),
        "terminal_yaw_rate_deg_s": float(np.rad2deg(state.yaw_rate_rad_s)),
        "terminal_current_speed_mps": float(
            np.linalg.norm(info.get("current_velocity_mps", np.zeros(3)))
        ),
        "dynamics_diagnostics": dict(info.get("dynamics_diagnostics", {})),
    }


class Stage1Trainer:
    def __init__(self, config: Stage1TrainingConfig):
        self.config = config
        set_global_seeds(config.seed, config.deterministic_torch)
        self.rng = np.random.default_rng(config.seed)
        self.base_environment = config.load_base_environment()
        observation_dim = REMUS100Env.observation_dim
        action_dim = 3
        self.agent = SACAgent(
            observation_dim,
            action_dim,
            config.sac,
            device=config.device,
        )
        self.replay = ReplayBuffer(
            observation_dim,
            action_dim,
            config.replay.capacity,
            seed=config.seed + 1,
        )
        self.total_steps = 0
        self.total_episodes = 0
        self.resume_stage_index = 0
        self.resume_stage_steps = 0
        self.resume_update_after_step = 0
        self.resumed = False
        self.resume_seed: int | None = None
        self.confirmed_stage_indices: set[int] = set()
        self.current_stage_steps = 0
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_log_path = self.output_dir / "training_episodes.jsonl"
        self.update_log_path = self.output_dir / "training_updates.jsonl"
        self.evaluation_dir = self.output_dir / "evaluations"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.last_update_metrics: SACUpdateMetrics | None = None

    def resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        metadata = self.agent.load(checkpoint_path)
        self.total_steps = int(metadata.get("total_steps", 0))
        self.total_episodes = int(metadata.get("total_episodes", 0))
        self.resume_stage_index = int(metadata.get("stage_index", 0))
        self.resume_stage_steps = int(metadata.get("stage_training_steps", 0))
        self.confirmed_stage_indices = {
            int(value) for value in metadata.get("confirmed_stage_indices", [])
        }
        if not 0 <= self.resume_stage_index < len(self.config.curriculum):
            raise ValueError("Checkpoint curriculum stage is not present in the current config")
        self.resumed = True
        self.resume_seed = derive_resume_seed(
            self.config.seed, self.total_steps, self.total_episodes
        )
        set_global_seeds(self.resume_seed, self.config.deterministic_torch)
        self.rng = np.random.default_rng(self.resume_seed)
        self.resume_update_after_step = self.total_steps + self.config.replay.resume_warmup_steps
        return metadata

    def _checkpoint(self, name: str, stage_index: int, stage_name: str) -> Path:
        path = self.checkpoint_dir / name
        self.agent.save(
            path,
            extra={
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "stage_index": stage_index,
                "stage_name": stage_name,
                "stage_training_steps": self.current_stage_steps,
                "seed": self.config.seed,
                "replay_size": len(self.replay),
                "resume_update_after_step": self.resume_update_after_step,
                "resume_seed": self.resume_seed,
                "confirmed_stage_indices": sorted(self.confirmed_stage_indices),
                "evaluation_protocol_version": 2,
                "note": "Replay contents are not embedded; resume refills replay before updates.",
            },
        )
        return path

    def _evaluate(
        self,
        stage_index: int,
        stage_name: str,
        environment_config,
        episodes: int,
        suffix: str,
        base_seed: int,
        evaluation_split: str,
    ) -> EvaluationSummary:
        seed = int(base_seed) + stage_index * 100_000
        print(
            json.dumps(
                {
                    "event": "evaluation_started",
                    "split": evaluation_split,
                    "stage": stage_name,
                    "total_steps": self.total_steps,
                    "episodes": episodes,
                    "base_seed": seed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        summary = evaluate_policy(
            self.agent,
            environment_config,
            episodes=episodes,
            base_seed=seed,
            deterministic=True,
        )
        payload = {
            "stage_index": stage_index,
            "stage_name": stage_name,
            "total_steps": self.total_steps,
            "checkpoint_update_count": self.agent.update_count,
            "evaluation_split": evaluation_split,
            "base_seed": seed,
            **summary.to_dict(include_records=True),
        }
        _write_json(self.evaluation_dir / f"{stage_name}_{suffix}.json", payload)
        print(
            json.dumps(
                {
                    "event": "evaluation_finished",
                    "split": evaluation_split,
                    "stage": stage_name,
                    "total_steps": self.total_steps,
                    "episodes": episodes,
                    "success_rate": summary.success_rate,
                    "collision_rate": summary.collision_rate,
                    "failure_counts": summary.failure_counts,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return summary

    def run(
        self,
        max_steps_override: int | None = None,
        final_evaluation_episodes_override: int | None = None,
    ) -> dict[str, Any]:
        started_at = time.time()
        global_step_limit = (
            int(max_steps_override)
            if max_steps_override is not None
            else sum(stage.maximum_training_steps for stage in self.config.curriculum)
        )
        if global_step_limit <= 0:
            raise ValueError("Training step limit must be positive")
        if self.resumed:
            print(
                json.dumps(
                    {
                        "event": "resume_ready",
                        "total_steps": self.total_steps,
                        "stage_index": self.resume_stage_index,
                        "stage_training_steps": self.resume_stage_steps,
                        "resume_seed": self.resume_seed,
                        "gradient_updates_resume_at_step": self.resume_update_after_step,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        stage_results: list[dict[str, Any]] = []
        all_stages_passed = True
        last_environment_config = None
        last_stage_index = 0
        last_stage_name = self.config.curriculum[0].name

        # Checkpoints created before the 100-episode confirmation protocol do
        # not contain proof that earlier curriculum stages met the formal gate.
        for prior_stage_index in range(self.resume_stage_index):
            if prior_stage_index in self.confirmed_stage_indices:
                continue
            prior_stage = self.config.curriculum[prior_stage_index]
            prior_environment_config = apply_curriculum_stage(
                self.base_environment, prior_stage
            )
            prior_summary = self._evaluate(
                prior_stage_index,
                prior_stage.name,
                prior_environment_config,
                self.config.training.confirmation_evaluation_episodes,
                suffix=f"resume_confirmation_step_{self.total_steps}",
                base_seed=self.config.training.confirmation_seed,
                evaluation_split="resume_protocol_confirmation",
            )
            prior_passed = bool(
                prior_summary.success_rate > prior_stage.promotion_success_rate
                and prior_summary.collision_rate <= prior_stage.promotion_collision_rate
            )
            stage_results.append(
                {
                    "name": prior_stage.name,
                    "training_steps": 0,
                    "total_steps": self.total_steps,
                    "resume_protocol_confirmation": True,
                    "gate_passed": prior_passed,
                    "gate": prior_summary.to_dict(include_records=False),
                }
            )
            if not prior_passed:
                raise RuntimeError(
                    f"Resume checkpoint failed the required 100-episode confirmation "
                    f"for {prior_stage.name}; do not continue to a harder curriculum stage."
                )
            self.confirmed_stage_indices.add(prior_stage_index)

        for stage_index, stage in enumerate(self.config.curriculum):
            if stage_index < self.resume_stage_index:
                continue
            if self.total_steps >= global_step_limit:
                all_stages_passed = False
                break
            environment_config = apply_curriculum_stage(self.base_environment, stage)
            last_environment_config = environment_config
            last_stage_index = stage_index
            last_stage_name = stage.name
            environment = REMUS100Env(environment_config)
            episode_seed = int(self.rng.integers(0, 2**31 - 1))
            observation, _ = environment.reset(seed=episode_seed)
            episode_return = 0.0
            episode_steps = 0
            carried_stage_steps = (
                self.resume_stage_steps if stage_index == self.resume_stage_index else 0
            )
            stage_start_step = self.total_steps - carried_stage_steps
            self.current_stage_steps = carried_stage_steps
            stage_passed = False
            best_success_rate = -1.0
            gate_summary: EvaluationSummary | None = None
            last_confirmation_step: int | None = None
            scheduled_evaluation_step = next_evaluation_step(
                stage_start_step,
                self.total_steps,
                stage.minimum_training_steps,
                self.config.training.evaluation_interval_steps,
            )
            next_checkpoint_step = self.total_steps + self.config.training.checkpoint_interval_steps
            next_log_step = self.total_steps + self.config.training.log_interval_steps

            while (
                self.total_steps < global_step_limit
                and self.total_steps - stage_start_step < stage.maximum_training_steps
            ):
                if self.total_steps < self.config.replay.learning_starts:
                    action = environment.action_space.sample()
                else:
                    action = self.agent.select_action(observation, deterministic=False)

                next_observation, reward, terminated, truncated, info = environment.step(action)
                self.replay.add(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )
                observation = next_observation
                episode_return += float(reward)
                episode_steps += 1
                self.total_steps += 1
                self.current_stage_steps = self.total_steps - stage_start_step

                if updates_allowed(
                    self.total_steps,
                    len(self.replay),
                    self.config.replay,
                    self.resume_update_after_step,
                ):
                    for _ in range(self.config.replay.updates_per_step):
                        self.last_update_metrics = self.agent.update(
                            self.replay.sample(self.config.replay.batch_size)
                        )

                if terminated or truncated:
                    self.total_episodes += 1
                    _append_json_line(
                        self.episode_log_path,
                        {
                            "total_steps": self.total_steps,
                            "episode": self.total_episodes,
                            "stage": stage.name,
                            "scenario_seed": episode_seed,
                            "return": episode_return,
                            "steps": episode_steps,
                            "success": bool(info.get("success", False)),
                            "failure_type": info.get("failure_type"),
                            "minimum_obstacle_distance_m": float(
                                info.get("minimum_obstacle_distance_m", environment_config.environment.sensor_range_m)
                            ),
                            **_episode_diagnostics(environment, info),
                        },
                    )
                    episode_seed = int(self.rng.integers(0, 2**31 - 1))
                    observation, _ = environment.reset(seed=episode_seed)
                    episode_return = 0.0
                    episode_steps = 0

                if self.total_steps >= next_log_step:
                    if self.last_update_metrics is not None:
                        _append_json_line(
                            self.update_log_path,
                            {
                                "total_steps": self.total_steps,
                                "stage": stage.name,
                                **asdict(self.last_update_metrics),
                            },
                        )
                    next_log_step += self.config.training.log_interval_steps

                if self.total_steps >= next_checkpoint_step:
                    self._checkpoint("latest.pt", stage_index, stage.name)
                    next_checkpoint_step += self.config.training.checkpoint_interval_steps

                stage_steps = self.total_steps - stage_start_step
                if (
                    stage_steps >= stage.minimum_training_steps
                    and self.total_steps >= scheduled_evaluation_step
                ):
                    summary = self._evaluate(
                        stage_index,
                        stage.name,
                        environment_config,
                        self.config.training.evaluation_episodes,
                        suffix=f"step_{self.total_steps}",
                        base_seed=self.config.training.validation_seed,
                        evaluation_split="validation_trend",
                    )
                    if summary.success_rate > best_success_rate:
                        best_success_rate = summary.success_rate
                        self._checkpoint(f"best_{stage.name}.pt", stage_index, stage.name)
                    trend_candidate = bool(
                        summary.success_rate > stage.promotion_success_rate
                        and summary.collision_rate <= stage.promotion_collision_rate
                    )
                    should_confirm = confirmation_due(
                        trend_candidate,
                        self.total_steps,
                        last_confirmation_step,
                        self.config.training.confirmation_interval_steps,
                    )
                    if should_confirm:
                        last_confirmation_step = self.total_steps
                        gate_summary = self._evaluate(
                            stage_index,
                            stage.name,
                            environment_config,
                            self.config.training.confirmation_evaluation_episodes,
                            suffix=f"confirmation_step_{self.total_steps}",
                            base_seed=self.config.training.confirmation_seed,
                            evaluation_split="curriculum_confirmation",
                        )
                        stage_passed = bool(
                            gate_summary.success_rate > stage.promotion_success_rate
                            and gate_summary.collision_rate <= stage.promotion_collision_rate
                        )
                    scheduled_evaluation_step = next_evaluation_step(
                        stage_start_step,
                        self.total_steps,
                        stage.minimum_training_steps,
                        self.config.training.evaluation_interval_steps,
                    )
                    if stage_passed:
                        break

            environment.close()

            # Always produce a gate record, including short smoke runs and exhausted budgets.
            gate_episodes = curriculum_gate_episode_count(
                self.config.training,
                final_evaluation_episodes_override,
            )
            if gate_summary is None or not stage_passed:
                gate_summary = self._evaluate(
                    stage_index,
                    stage.name,
                    environment_config,
                    gate_episodes,
                    suffix=f"gate_step_{self.total_steps}",
                    base_seed=self.config.training.confirmation_seed,
                    evaluation_split=(
                        "smoke_gate"
                        if final_evaluation_episodes_override is not None
                        else "curriculum_confirmation"
                    ),
                )
                if max_steps_override is None:
                    stage_passed = bool(
                        gate_summary.success_rate > stage.promotion_success_rate
                        and gate_summary.collision_rate <= stage.promotion_collision_rate
                    )
            stage_result = {
                "name": stage.name,
                "training_steps": self.current_stage_steps,
                "total_steps": self.total_steps,
                "gate_passed": stage_passed,
                "gate": gate_summary.to_dict(include_records=False),
            }
            stage_results.append(stage_result)
            if stage_passed and max_steps_override is None:
                self.confirmed_stage_indices.add(stage_index)
            self._checkpoint("latest.pt", stage_index, stage.name)
            if not stage_passed:
                all_stages_passed = False
            if not stage_passed and max_steps_override is None:
                break
            self.resume_stage_index = min(stage_index + 1, len(self.config.curriculum) - 1)
            self.resume_stage_steps = 0

        if last_environment_config is None:
            last_environment_config = apply_curriculum_stage(
                self.base_environment, self.config.curriculum[0]
            )

        final_episodes = (
            int(final_evaluation_episodes_override)
            if final_evaluation_episodes_override is not None
            else self.config.training.final_evaluation_episodes
        )
        final_summary = self._evaluate(
            last_stage_index,
            last_stage_name,
            last_environment_config,
            final_episodes,
            suffix="final",
            base_seed=self.config.training.final_test_seed,
            evaluation_split=(
                "smoke_final"
                if final_evaluation_episodes_override is not None
                else "independent_final_test"
            ),
        )
        final_checkpoint = self._checkpoint("final.pt", last_stage_index, last_stage_name)
        result = {
            "stage": 1,
            "seed": self.config.seed,
            "device": str(self.agent.device),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "updates": self.agent.update_count,
            "replay_size": len(self.replay),
            "resumed": self.resumed,
            "resume_update_after_step": self.resume_update_after_step,
            "resume_seed": self.resume_seed,
            "evaluation_protocol_version": 2,
            "confirmed_stage_indices": sorted(self.confirmed_stage_indices),
            "curriculum_results": stage_results,
            "all_curriculum_stages_passed": all_stages_passed,
            "final_evaluation": final_summary.to_dict(include_records=False),
            "stage1_gate_passed": bool(
                max_steps_override is None and all_stages_passed and final_summary.stage1_gate_passes()
            ),
            "final_checkpoint": str(final_checkpoint),
            "elapsed_seconds": time.time() - started_at,
            "note": (
                "A smoke run validates code execution only."
                if max_steps_override is not None
                else "Curriculum promotion and the independent final test each use at least "
                "100 episodes; the V4 gate requires success rate strictly above 90%."
            ),
        }
        _write_json(self.output_dir / "summary.json", result)
        return result


__all__ = ["Stage1Trainer"]
