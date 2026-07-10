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
from .training_config import Stage1TrainingConfig, apply_curriculum_stage


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
                "seed": self.config.seed,
                "replay_size": len(self.replay),
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
    ) -> EvaluationSummary:
        seed = self.config.training.evaluation_seed + stage_index * 100_000
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
            **summary.to_dict(include_records=True),
        }
        _write_json(self.evaluation_dir / f"{stage_name}_{suffix}.json", payload)
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

        stage_results: list[dict[str, Any]] = []
        all_stages_passed = True
        last_environment_config = None
        last_stage_index = 0
        last_stage_name = self.config.curriculum[0].name

        for stage_index, stage in enumerate(self.config.curriculum):
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
            stage_start_step = self.total_steps
            stage_passed = False
            best_success_rate = -1.0
            next_evaluation_step = self.total_steps + self.config.training.evaluation_interval_steps
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

                if (
                    self.total_steps >= self.config.replay.update_after
                    and len(self.replay) >= self.config.replay.batch_size
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
                    and self.total_steps >= next_evaluation_step
                ):
                    summary = self._evaluate(
                        stage_index,
                        stage.name,
                        environment_config,
                        self.config.training.evaluation_episodes,
                        suffix=f"step_{self.total_steps}",
                    )
                    if summary.success_rate > best_success_rate:
                        best_success_rate = summary.success_rate
                        self._checkpoint(f"best_{stage.name}.pt", stage_index, stage.name)
                    stage_passed = bool(
                        summary.success_rate > stage.promotion_success_rate
                        and summary.collision_rate <= stage.promotion_collision_rate
                    )
                    next_evaluation_step += self.config.training.evaluation_interval_steps
                    if stage_passed:
                        break

            environment.close()

            # Always produce a gate record, including short smoke runs and exhausted budgets.
            gate_episodes = (
                int(final_evaluation_episodes_override)
                if final_evaluation_episodes_override is not None
                else self.config.training.evaluation_episodes
            )
            gate_summary = self._evaluate(
                stage_index,
                stage.name,
                environment_config,
                gate_episodes,
                suffix=f"gate_step_{self.total_steps}",
            )
            if max_steps_override is None:
                stage_passed = bool(
                    gate_summary.success_rate > stage.promotion_success_rate
                    and gate_summary.collision_rate <= stage.promotion_collision_rate
                )
            stage_result = {
                "name": stage.name,
                "training_steps": self.total_steps - stage_start_step,
                "total_steps": self.total_steps,
                "gate_passed": stage_passed,
                "gate": gate_summary.to_dict(include_records=False),
            }
            stage_results.append(stage_result)
            self._checkpoint("latest.pt", stage_index, stage.name)
            if not stage_passed:
                all_stages_passed = False
            if not stage_passed and max_steps_override is None:
                break

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
            "curriculum_results": stage_results,
            "all_curriculum_stages_passed": all_stages_passed,
            "final_evaluation": final_summary.to_dict(include_records=False),
            "stage1_gate_passed": bool(
                max_steps_override is None and all_stages_passed and final_summary.stage1_gate_passes()
            ),
            "final_checkpoint": str(final_checkpoint),
            "elapsed_seconds": time.time() - started_at,
            "note": (
                "A smoke run validates code execution only. The V4 gate requires a full "
                "100-episode evaluation with success rate strictly above 90%."
            ),
        }
        _write_json(self.output_dir / "summary.json", result)
        return result


__all__ = ["Stage1Trainer"]
