"""Run deterministic Stage-0 rollouts and write a compact audit artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/rl_zl_matplotlib")))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rl_zl import REMUS100Env, load_config
from rl_zl.obstacles import RAY_DIRECTIONS_BODY


def heuristic_action(observation: np.ndarray) -> np.ndarray:
    """Non-learning controller used only to exercise the environment."""
    goal_body = observation[10:13].astype(np.float64)
    ray_fraction = observation[17:43].astype(np.float64)
    desired = goal_body.copy()
    warning_fraction = 0.35
    for direction, distance_fraction in zip(RAY_DIRECTIONS_BODY, ray_fraction, strict=True):
        if distance_fraction < warning_fraction:
            strength = ((warning_fraction - distance_fraction) / warning_fraction) ** 2
            desired -= 1.8 * strength * direction
    desired_norm = np.linalg.norm(desired)
    if desired_norm < 1e-9:
        desired = np.array([1.0, 0.0, 0.0])
    else:
        desired /= desired_norm
    yaw_error = np.arctan2(desired[1], max(1e-6, desired[0]))
    pitch_error = np.arctan2(desired[2], max(1e-6, np.linalg.norm(desired[:2])))
    nearest_ray = float(np.min(ray_fraction))
    speed_action = np.clip(2.0 * nearest_ray - 0.2, -0.6, 0.8)
    return np.array(
        [
            speed_action,
            np.clip(pitch_error / np.deg2rad(8.0), -1.0, 1.0),
            np.clip(yaw_error / np.deg2rad(12.0), -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage0.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts/stage0_validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(ROOT / args.config)
    base_seed = config.seed if args.seed is None else args.seed
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = REMUS100Env(config)
    outcomes: Counter[str] = Counter()
    reward_component_abs_max: dict[str, float] = defaultdict(float)
    rewards: list[float] = []
    episode_steps: list[int] = []
    observation_min = np.inf
    observation_max = -np.inf
    numerical_failures = 0
    feasible_scenarios = 0

    for episode in range(args.episodes):
        observation, reset_info = env.reset(seed=base_seed + episode)
        if reset_info["scenario"]["coarse_path_length_m"] is not None:
            feasible_scenarios += 1
        episode_reward = 0.0
        terminated = truncated = False
        info = {"failure_type": None, "success": False}
        while not (terminated or truncated):
            observation_min = min(observation_min, float(np.min(observation)))
            observation_max = max(observation_max, float(np.max(observation)))
            if not np.all(np.isfinite(observation)) or not env.observation_space.contains(observation):
                numerical_failures += 1
                break
            action = heuristic_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(reward):
                numerical_failures += 1
                break
            if not 0.0 <= float(info["safety_cost_normalized"]) <= 1.0:
                numerical_failures += 1
                break
            episode_reward += reward
            for key, value in info["reward_components"].items():
                reward_component_abs_max[key] = max(reward_component_abs_max[key], abs(float(value)))

        rewards.append(float(episode_reward))
        episode_steps.append(int(info.get("step_count", 0)))
        outcome = "success" if info.get("success") else str(info.get("failure_type") or "validation_error")
        outcomes[outcome] += 1

    env.render(output_dir / "trajectory.png")
    env.render_ray_diagnostic(output_dir / "ray_geometry.png")
    summary = {
        "stage": 0,
        "config": args.config,
        "base_seed": base_seed,
        "episodes": args.episodes,
        "outcomes": dict(outcomes),
        "mean_return": float(np.mean(rewards)) if rewards else None,
        "return_std": float(np.std(rewards)) if rewards else None,
        "mean_episode_steps": float(np.mean(episode_steps)) if episode_steps else None,
        "observation_min": float(observation_min),
        "observation_max": float(observation_max),
        "reward_component_abs_max": dict(reward_component_abs_max),
        "numerical_failures": numerical_failures,
        "feasible_scenarios": feasible_scenarios,
        "stage0_environment_gate_pass": bool(
            numerical_failures == 0
            and feasible_scenarios == args.episodes
            and observation_min >= -1.000001
            and observation_max <= 1.000001
        ),
        "note": "Heuristic rollouts validate environment execution only; they are not paper results.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["stage0_environment_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
