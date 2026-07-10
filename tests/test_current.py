from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config
from rl_zl.current import AnalyticCurrentField


class CurrentTests(unittest.TestCase):
    def test_current_is_deterministic_and_capped(self):
        config = stage0_config()
        field_a = AnalyticCurrentField.sample(config.current, np.random.default_rng(7), config.environment.world_size_m)
        field_b = AnalyticCurrentField.sample(config.current, np.random.default_rng(7), config.environment.world_size_m)
        point = np.array([123.0, 234.0, 45.0])
        velocity_a = field_a.velocity(point, 17.0)
        velocity_b = field_b.velocity(point, 17.0)
        np.testing.assert_allclose(velocity_a, velocity_b, atol=0.0)
        self.assertLessEqual(np.linalg.norm(velocity_a), config.current.max_speed_mps + 1e-12)
        self.assertLessEqual(abs(velocity_a[2]), config.current.max_speed_mps * config.current.vertical_fraction + 1e-12)


if __name__ == "__main__":
    unittest.main()

