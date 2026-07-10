from __future__ import annotations

import unittest

import numpy as np

from _support import stage0_config
from rl_zl.dynamics import REMUSPlanningDynamics, VehicleState, body_to_world_matrix


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.config = stage0_config()
        self.dynamics = REMUSPlanningDynamics(self.config.vehicle)

    def test_forward_motion_and_current_are_added_once(self):
        state = VehicleState(np.array([10.0, 20.0, 30.0]), 0.0, 0.0, 1.0, 0.0, 0.0)
        next_state, _, _ = self.dynamics.step(
            state, np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.2, 0.0])
        )
        self.assertAlmostEqual(next_state.speed_mps, 1.15, places=8)
        np.testing.assert_allclose(next_state.position_m, [11.25, 20.2, 30.0], atol=1e-9)

    def test_positive_pitch_increases_depth(self):
        state = VehicleState(np.zeros(3), np.deg2rad(10.0), 0.0, 1.0, 0.0, 0.0)
        next_state, _, _ = self.dynamics.step(state, np.array([0.0, 0.0, 0.0]), np.zeros(3))
        self.assertGreater(next_state.position_m[2], 0.0)

    def test_v4_command_rate_limit_is_applied(self):
        state = VehicleState(np.zeros(3), 0.0, 0.0, 1.0, 0.0, 0.0)
        _, command, diagnostics = self.dynamics.step(
            state,
            np.array([0.0, 1.0, 1.0]),
            np.zeros(3),
            previous_command=np.array([1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(command[1], np.deg2rad(2.0), places=10)
        self.assertAlmostEqual(command[2], np.deg2rad(3.0), places=10)
        self.assertGreater(diagnostics.raw_command[1], command[1])

    def test_v4_pitch_is_clipped_and_preclip_violation_is_logged(self):
        state = VehicleState(
            np.zeros(3),
            self.dynamics.pitch_limit_rad - np.deg2rad(1.0),
            0.0,
            1.0,
            0.0,
            0.0,
        )
        next_state, _, diagnostics = self.dynamics.step(
            state,
            np.array([0.0, 1.0, 0.0]),
            np.zeros(3),
            previous_command=np.array([1.0, np.deg2rad(4.0), 0.0]),
        )
        self.assertAlmostEqual(next_state.pitch_rad, self.dynamics.pitch_limit_rad, places=10)
        self.assertTrue(diagnostics.pitch_clipped)
        self.assertTrue(diagnostics.hard_violation)

    def test_body_rotation_maps_forward_axis(self):
        rotation = body_to_world_matrix(np.deg2rad(10.0), np.deg2rad(90.0))
        forward = rotation @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(forward, [0.0, np.cos(np.deg2rad(10.0)), np.sin(np.deg2rad(10.0))], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
