"""Execute the six deterministic Stage-0 cases required by the V4 plan."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/rl_zl_matplotlib")))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rl_zl import REMUS100Env, load_config
from rl_zl.current import AnalyticCurrentField
from rl_zl.dynamics import REMUSPlanningDynamics, VehicleState, body_to_world_matrix
from rl_zl.obstacles import RAY_DIRECTIONS_BODY, SphereObstacle, ray_distances
from rl_zl.scenario import Scenario


def zero_current(config, seed: int = 0) -> AnalyticCurrentField:
    current_config = replace(config.current, mode="none", max_speed_mps=0.0)
    return AnalyticCurrentField.sample(
        current_config,
        np.random.default_rng(seed),
        config.environment.world_size_m,
    )


def manual_scenario(config, start, goal, obstacles=None, seed=0) -> Scenario:
    return Scenario(
        start_m=np.asarray(start, dtype=np.float64),
        goal_m=np.asarray(goal, dtype=np.float64),
        obstacles=list(obstacles or []),
        current=zero_current(config, seed),
        coarse_path_length_m=float(np.linalg.norm(np.asarray(goal) - np.asarray(start))),
        seed=seed,
    )


def target_action(env: REMUS100Env, target_m: np.ndarray, speed_action: float = 0.0) -> np.ndarray:
    assert env.state is not None
    delta_world = np.asarray(target_m, dtype=np.float64) - env.state.position_m
    direction_world = delta_world / max(np.linalg.norm(delta_world), 1e-9)
    direction_body = body_to_world_matrix(env.state.pitch_rad, env.state.yaw_rad).T @ direction_world
    yaw_error = np.arctan2(direction_body[1], max(1e-6, direction_body[0]))
    pitch_error = np.arctan2(direction_body[2], max(1e-6, np.linalg.norm(direction_body[:2])))
    return np.array(
        [
            speed_action,
            np.clip(pitch_error / env.dynamics.pitch_rate_command_limit_rad_s, -1.0, 1.0),
            np.clip(yaw_error / env.dynamics.yaw_rate_command_limit_rad_s, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def run_random_action(config) -> dict:
    env = REMUS100Env(config)
    observation, _ = env.reset(seed=9001)
    steps = 0
    finite = True
    terminated = truncated = False
    info = {"failure_type": None}
    while steps < 120 and not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        finite = finite and bool(np.all(np.isfinite(observation)) and np.isfinite(reward))
        steps += 1
    return {"passed": finite, "steps": steps, "outcome": info.get("failure_type") or "running"}


def run_straight_and_turn(config) -> tuple[dict, dict]:
    dynamics = REMUSPlanningDynamics(config.vehicle)
    straight = VehicleState(np.array([100.0, 100.0, 50.0]), 0.0, 0.0, 1.0, 0.0, 0.0)
    previous_command = np.array([1.0, 0.0, 0.0])
    for _ in range(20):
        straight, previous_command, _ = dynamics.step(
            straight, np.zeros(3), np.zeros(3), previous_command=previous_command
        )
    straight_result = {
        "passed": bool(straight.position_m[0] > 120.0 and np.allclose(straight.position_m[1:], [100.0, 50.0])),
        "final_position_m": straight.position_m.tolist(),
    }

    turning = VehicleState(np.array([100.0, 100.0, 50.0]), 0.0, 0.0, 1.0, 0.0, 0.0)
    previous_command = np.array([1.0, 0.0, 0.0])
    for _ in range(10):
        turning, previous_command, diagnostics = dynamics.step(
            turning,
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            previous_command=previous_command,
        )
        if diagnostics.hard_violation:
            break
    turn_result = {
        "passed": bool(turning.yaw_rad > 0.5 and turning.position_m[1] > 100.0),
        "final_position_m": turning.position_m.tolist(),
        "final_yaw_deg": float(np.rad2deg(turning.yaw_rad)),
    }
    return straight_result, turn_result


def run_no_obstacle_arrival(config) -> dict:
    env = REMUS100Env(config)
    scenario = manual_scenario(config, [100.0, 100.0, 50.0], [135.0, 100.0, 50.0], seed=9002)
    observation, _ = env.reset_with_scenario(scenario, initial_yaw_rad=0.0, initial_pitch_rad=0.0)
    terminated = truncated = False
    info = {"success": False, "failure_type": None}
    steps = 0
    while steps < 80 and not (terminated or truncated):
        action = target_action(env, scenario.goal_m, speed_action=0.0)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
    return {"passed": bool(info.get("success")), "steps": steps, "outcome": info.get("failure_type")}


def run_single_obstacle_avoidance(config, output_dir: Path) -> dict:
    env = REMUS100Env(config)
    obstacle = SphereObstacle(np.array([125.0, 100.0, 50.0]), radius_m=6.0)
    scenario = manual_scenario(
        config,
        [100.0, 100.0, 50.0],
        [155.0, 100.0, 50.0],
        obstacles=[obstacle],
        seed=9003,
    )
    env.reset_with_scenario(scenario, initial_yaw_rad=0.0, initial_pitch_rad=0.0)
    waypoints = [
        np.array([112.0, 82.0, 50.0]),
        np.array([138.0, 82.0, 50.0]),
        scenario.goal_m,
    ]
    waypoint_index = 0
    terminated = truncated = False
    info = {"success": False, "failure_type": None, "minimum_obstacle_distance_m": np.inf}
    min_clearance = np.inf
    steps = 0
    while steps < 180 and not (terminated or truncated):
        assert env.state is not None
        target = waypoints[waypoint_index]
        if np.linalg.norm(target - env.state.position_m) < 5.0 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            target = waypoints[waypoint_index]
        action = target_action(env, target, speed_action=-0.1)
        _, _, terminated, truncated, info = env.step(action)
        min_clearance = min(min_clearance, float(info["minimum_obstacle_distance_m"]))
        steps += 1
    env.render(output_dir / "single_obstacle_avoidance.png")
    env.render_ray_diagnostic(output_dir / "single_obstacle_rays.png")
    return {
        "passed": bool(info.get("success") and min_clearance > 0.0),
        "steps": steps,
        "outcome": info.get("failure_type"),
        "minimum_inflated_clearance_m": min_clearance,
    }


def run_current_drift(config) -> dict:
    dynamics = REMUSPlanningDynamics(config.vehicle)
    state_no_current = VehicleState(np.array([100.0, 100.0, 50.0]), 0.0, 0.0, 1.0, 0.0, 0.0)
    state_with_current = state_no_current.copy()
    previous_no_current = previous_with_current = np.array([1.0, 0.0, 0.0])
    for _ in range(20):
        state_no_current, previous_no_current, _ = dynamics.step(
            state_no_current, np.zeros(3), np.zeros(3), previous_command=previous_no_current
        )
        state_with_current, previous_with_current, _ = dynamics.step(
            state_with_current,
            np.zeros(3),
            np.array([0.0, 0.2, 0.0]),
            previous_command=previous_with_current,
        )
    drift = state_with_current.position_m - state_no_current.position_m
    return {"passed": bool(np.allclose(drift, [0.0, 4.0, 0.0], atol=1e-8)), "drift_m": drift.tolist()}


def run_ray_geometry(config) -> dict:
    obstacle = SphereObstacle(np.array([20.0, 0.0, 0.0]), radius_m=6.0)
    distances = ray_distances(
        np.zeros(3),
        RAY_DIRECTIONS_BODY,
        [obstacle],
        config.environment.sensor_range_m,
        config.vehicle.radius_m + config.obstacles.inflation_margin_m,
    )
    forward_index = int(np.argmax(RAY_DIRECTIONS_BODY[:, 0]))
    expected = 20.0 - 6.0 - config.vehicle.radius_m - config.obstacles.inflation_margin_m
    return {
        "passed": bool(np.isclose(distances[forward_index], expected, atol=1e-8)),
        "forward_distance_m": float(distances[forward_index]),
        "expected_m": expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage0.yaml")
    parser.add_argument("--output-dir", default="artifacts/stage0_cases")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    straight, turn = run_straight_and_turn(config)
    cases = {
        "random_action": run_random_action(config),
        "fixed_straight": straight,
        "fixed_turn": turn,
        "no_obstacle_arrival": run_no_obstacle_arrival(config),
        "single_obstacle_avoidance": run_single_obstacle_avoidance(config, output_dir),
        "current_drift": run_current_drift(config),
        "ray_geometry": run_ray_geometry(config),
    }
    result = {
        "v4_stage0_cases": cases,
        "all_passed": all(case["passed"] for case in cases.values()),
        "note": "These are deterministic engineering validation cases, not RL paper results.",
    }
    with (output_dir / "stage0_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

