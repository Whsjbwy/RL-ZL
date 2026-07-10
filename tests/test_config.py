from __future__ import annotations

import unittest

from _support import stage0_config


class ConfigTests(unittest.TestCase):
    def test_configuration_loads_and_is_explicit(self):
        config = stage0_config()
        self.assertEqual(config.environment.world_size_m, (500.0, 500.0, 100.0))
        self.assertEqual(config.environment.legal_depth_m, (5.0, 95.0))
        self.assertGreater(config.environment.sensor_range_m, 0.0)
        self.assertGreater(config.obstacles.inflation_margin_m, 0.0)


if __name__ == "__main__":
    unittest.main()

