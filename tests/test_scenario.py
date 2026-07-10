from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config
from rl_zl.scenario import ScenarioGenerator


class ScenarioTests(unittest.TestCase):
    def test_seeded_scenario_is_reproducible_and_feasible(self):
        config = stage0_config()
        generator = ScenarioGenerator(config)
        first = generator.sample(12345)
        second = generator.sample(12345)
        np.testing.assert_allclose(first.start_m, second.start_m, atol=0.0)
        np.testing.assert_allclose(first.goal_m, second.goal_m, atol=0.0)
        self.assertIsNotNone(first.coarse_path_length_m)
        self.assertGreaterEqual(len(first.obstacles), config.obstacles.count_range[0])
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()

