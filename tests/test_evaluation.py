from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config
from rl_zl.evaluation import EvaluationSummary


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


if __name__ == "__main__":
    unittest.main()
