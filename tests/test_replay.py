from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config  # noqa: F401 - initializes local src path
from rl_zl.replay import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_ring_buffer_and_timeout_semantics(self):
        replay = ReplayBuffer(observation_dim=2, action_dim=1, capacity=2, seed=7)
        replay.add([1, 2], [0.1], 3.0, [2, 3], terminated=False, truncated=True)
        replay.add([4, 5], [0.2], 6.0, [5, 6], terminated=True, truncated=False)
        self.assertEqual(len(replay), 2)
        np.testing.assert_array_equal(replay.terminated[:2, 0], [0.0, 1.0])
        np.testing.assert_array_equal(replay.truncated[:2, 0], [1.0, 0.0])
        replay.add([7, 8], [0.3], 9.0, [8, 9], terminated=False, truncated=False)
        self.assertEqual(len(replay), 2)
        np.testing.assert_array_equal(replay.observations[0], [7.0, 8.0])

    def test_rejects_non_finite_transition(self):
        replay = ReplayBuffer(observation_dim=2, action_dim=1, capacity=2, seed=7)
        with self.assertRaises(ValueError):
            replay.add([np.nan, 0], [0], 0.0, [0, 0], False, False)


if __name__ == "__main__":
    unittest.main()
