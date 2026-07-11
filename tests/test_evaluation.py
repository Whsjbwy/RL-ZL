from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config
from rl_zl.evaluation import EvaluationSummary, evaluate_policy


class ConstantPolicy:
    def select_action(self, observation, deterministic: bool = False):
        del observation, deterministic
        return np.zeros(3, dtype=np.float32)


class EvaluationTests(unittest.TestCase):
    def test_stage1_gate_is_strictly_greater_than_ninety_percent(self):
        base = dict(
            episodes=100,
            collision_rate=0.05,
            timeout_rate=0.0,
            safety_failure_rate=0.05,
            mean_return=0.0,
            mean_path_length_success_m=1.0,
            mean_path_length_all_m=1.0,
            mean_travel_time_success_s=1.0,
            mean_energy_success=1.0,
            mean_minimum_clearance_m=1.0,
            failure_counts={},
            records=(),
        )
        at_threshold = EvaluationSummary(success_rate=0.90, **base)
        above_threshold = EvaluationSummary(success_rate=0.91, **base)
        self.assertFalse(at_threshold.stage1_gate_passes())
        self.assertTrue(above_threshold.stage1_gate_passes())

    def test_environment_config_remains_finite(self):
        config = stage0_config()
        self.assertTrue(np.isfinite(config.environment.goal_radius_m))

    def test_episode_records_retain_scenario_and_terminal_diagnostics(self):
        summary = evaluate_policy(
            ConstantPolicy(),
            stage0_config(),
            episodes=1,
            base_seed=987_654,
        )
        record = summary.records[0]
        self.assertEqual(record.seed, 987_654)
        self.assertEqual(len(record.start_m), 3)
        self.assertEqual(len(record.goal_m), 3)
        self.assertEqual(len(record.terminal_position_m), 3)
        self.assertGreater(record.obstacle_count, 0)
        self.assertTrue(np.isfinite(record.initial_goal_pitch_deg))
        self.assertTrue(np.isfinite(record.terminal_pitch_rate_deg_s))
        self.assertIsInstance(record.dynamics_diagnostics, dict)


if __name__ == "__main__":
    unittest.main()
