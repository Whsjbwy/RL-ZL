from __future__ import annotations

import unittest

from _support import ROOT
from rl_zl.training_config import apply_curriculum_stage, load_stage1_config


class Stage1TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_stage1_config(ROOT / "configs" / "stage1_teacher.yaml")

    def test_v4_teacher_architecture_and_final_evaluation_are_frozen(self):
        self.assertEqual(self.config.sac.actor_hidden_dims, (256, 256, 128))
        self.assertEqual(self.config.sac.critic_hidden_dims, (256, 256, 128))
        self.assertEqual(self.config.training.final_evaluation_episodes, 100)
        self.assertEqual(self.config.replay.capacity, 1_000_000)

    def test_stage1a_matches_v4_low_difficulty_environment(self):
        stage = self.config.curriculum[0]
        environment = apply_curriculum_stage(self.config.load_base_environment(), stage)
        self.assertEqual(stage.name, "stage1a_no_current")
        self.assertEqual(environment.environment.goal_radius_m, 10.0)
        self.assertEqual(environment.obstacles.count_range, (4, 6))
        self.assertEqual(environment.obstacles.types, ("sphere", "cylinder"))
        self.assertEqual(environment.obstacles.sphere_radius_m, (8.0, 15.0))
        self.assertEqual(environment.current.mode, "none")
        self.assertEqual(environment.current.max_speed_mps, 0.0)

    def test_stage1b_uses_only_weak_background_current(self):
        stage = self.config.curriculum[1]
        environment = apply_curriculum_stage(self.config.load_base_environment(), stage)
        self.assertEqual(environment.obstacles.count_range, (6, 8))
        self.assertEqual(environment.environment.goal_radius_m, 10.0)
        self.assertEqual(environment.current.max_speed_mps, 0.15)
        self.assertEqual(environment.current.vortex_count, (0, 0))
        self.assertEqual(environment.current.time_amplitude_mps, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
