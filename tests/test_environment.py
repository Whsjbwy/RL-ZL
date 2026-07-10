from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from _support import stage0_config
from rl_zl.environment import REMUS100Env


class EnvironmentTests(unittest.TestCase):
    def test_reset_and_random_rollout_are_finite(self):
        env = REMUS100Env(stage0_config())
        observation, info = env.reset(seed=2026)
        self.assertEqual(observation.shape, (53,))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIsNotNone(info["scenario"]["coarse_path_length_m"])
        for _ in range(120):
            observation, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.isfinite(step_info["safety_cost"]))
            if terminated or truncated:
                break

    def test_same_seed_produces_same_reset(self):
        env = REMUS100Env(stage0_config())
        obs_a, info_a = env.reset(seed=99)
        obs_b, info_b = env.reset(seed=99)
        np.testing.assert_array_equal(obs_a, obs_b)
        self.assertEqual(info_a["scenario"], info_b["scenario"])

    def test_timeout_is_reported_as_truncation(self):
        config = stage0_config()
        short_env = replace(config.environment, max_steps=1, oscillation_window=50)
        short_feasibility = replace(config.feasibility, enabled=False)
        config = replace(config, environment=short_env, feasibility=short_feasibility)
        env = REMUS100Env(config)
        env.reset(seed=77)
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["failure_type"], "timeout")

    def test_v4_failure_priority(self):
        events = {
            "goal": False,
            "collision": True,
            "depth": True,
            "boundary": True,
            "dynamics": True,
            "oscillation": True,
            "timeout": True,
        }
        self.assertEqual(REMUS100Env._failure_type(events), "collision")
        events["collision"] = False
        self.assertEqual(REMUS100Env._failure_type(events), "depth")

    def test_v4_safety_cost_is_clipped_and_normalized(self):
        env = REMUS100Env(stage0_config())
        env.reset(seed=55)
        assert env.state is not None
        env.state.position_m[2] = 4.0
        events = {
            "goal": False,
            "collision": True,
            "depth": True,
            "boundary": True,
            "dynamics": False,
            "oscillation": False,
            "timeout": False,
        }
        cost, normalized, components = env._safety_cost(events)
        self.assertLessEqual(cost, env.config.safety.cost_max)
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)
        self.assertAlmostEqual(components["normalized_total"], normalized)

    def test_goal_at_final_allowed_step_is_success_not_timeout(self):
        config = stage0_config()
        env = REMUS100Env(config)
        env.reset(seed=12)
        assert env.state is not None and env.scenario is not None
        env.step_count = config.environment.max_steps
        env.state.position_m = env.scenario.goal_m.copy()
        events = env._terminal_events(0.0)
        self.assertTrue(events["goal"])
        self.assertFalse(events["timeout"])


if __name__ == "__main__":
    unittest.main()
