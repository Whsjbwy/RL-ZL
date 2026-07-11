from __future__ import annotations

import unittest

from _support import ROOT
from rl_zl.training_config import (
    apply_curriculum_stage,
    confirmation_due,
    curriculum_gate_episode_count,
    derive_resume_seed,
    load_stage1_config,
    next_evaluation_step,
    updates_allowed,
)


class Stage1TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_stage1_config(ROOT / "configs" / "stage1_teacher.yaml")

    def test_v4_teacher_architecture_and_final_evaluation_are_frozen(self):
        self.assertEqual(self.config.sac.actor_hidden_dims, (256, 256, 128))
        self.assertEqual(self.config.sac.critic_hidden_dims, (256, 256, 128))
        self.assertEqual(self.config.training.final_evaluation_episodes, 100)
        self.assertEqual(self.config.training.confirmation_evaluation_episodes, 100)
        self.assertEqual(self.config.replay.capacity, 1_000_000)
        self.assertEqual(self.config.replay.resume_warmup_steps, 10_000)

    def test_evaluation_seed_blocks_are_independent(self):
        seeds = {
            self.config.training.validation_seed,
            self.config.training.confirmation_seed,
            self.config.training.final_test_seed,
        }
        self.assertEqual(len(seeds), 3)

    def test_stage_relative_evaluation_schedule_has_no_catch_up_burst(self):
        interval = self.config.training.evaluation_interval_steps
        self.assertEqual(next_evaluation_step(0, 0, 100_000, interval), 100_000)
        self.assertEqual(next_evaluation_step(0, 100_000, 100_000, interval), 110_000)
        # Resuming Stage 1B at global step 380k means 100k was already completed.
        self.assertEqual(next_evaluation_step(280_000, 380_000, 100_000, interval), 390_000)

    def test_resume_requires_fresh_replay_before_updates(self):
        replay = self.config.replay
        self.assertTrue(updates_allowed(10_000, replay.batch_size, replay))
        self.assertFalse(
            updates_allowed(
                380_001,
                replay.batch_size,
                replay,
                resume_update_after_step=390_000,
            )
        )
        self.assertTrue(
            updates_allowed(
                390_000,
                replay.batch_size,
                replay,
                resume_update_after_step=390_000,
            )
        )

    def test_formal_confirmation_is_rate_limited_and_requires_candidate(self):
        interval = self.config.training.confirmation_interval_steps
        self.assertFalse(confirmation_due(False, 390_000, None, interval))
        self.assertTrue(confirmation_due(True, 390_000, None, interval))
        self.assertFalse(confirmation_due(True, 400_000, 390_000, interval))
        self.assertTrue(confirmation_due(True, 440_000, 390_000, interval))

    def test_formal_gate_cannot_fall_back_to_twenty_episode_trend_screen(self):
        self.assertEqual(curriculum_gate_episode_count(self.config.training), 100)
        self.assertEqual(curriculum_gate_episode_count(self.config.training, 2), 2)

    def test_resume_seed_is_reproducible_and_not_the_fresh_seed(self):
        first = derive_resume_seed(self.config.seed, 380_000, 1_770)
        second = derive_resume_seed(self.config.seed, 380_000, 1_770)
        changed = derive_resume_seed(self.config.seed, 390_000, 1_800)
        self.assertEqual(first, second)
        self.assertNotEqual(first, self.config.seed)
        self.assertNotEqual(first, changed)

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
