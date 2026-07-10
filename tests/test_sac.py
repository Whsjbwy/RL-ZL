from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from _support import ROOT
from rl_zl.replay import ReplayBatch
from rl_zl.training_config import load_stage1_config


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from rl_zl.sac import SACAgent, set_global_seeds


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class SACTests(unittest.TestCase):
    def setUp(self):
        set_global_seeds(123)
        config = load_stage1_config(ROOT / "configs" / "stage1_teacher.yaml").sac
        self.config = replace(
            config,
            actor_hidden_dims=(32, 32),
            critic_hidden_dims=(32, 32),
        )
        self.agent = SACAgent(5, 3, self.config, device="cpu")

    def random_batch(self, batch_size: int = 16) -> ReplayBatch:
        rng = np.random.default_rng(11)
        return ReplayBatch(
            observations=rng.normal(size=(batch_size, 5)).astype(np.float32),
            actions=rng.uniform(-1, 1, size=(batch_size, 3)).astype(np.float32),
            rewards=rng.normal(size=(batch_size, 1)).astype(np.float32),
            next_observations=rng.normal(size=(batch_size, 5)).astype(np.float32),
            terminated=rng.integers(0, 2, size=(batch_size, 1)).astype(np.float32),
            truncated=rng.integers(0, 2, size=(batch_size, 1)).astype(np.float32),
        )

    def test_actor_action_and_log_probability_are_finite(self):
        observation = torch.full((8, 5), 1000.0)
        action, log_probability, mean, log_std = self.agent.actor.sample(observation)
        self.assertEqual(tuple(action.shape), (8, 3))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(log_std).all())
        self.assertTrue(torch.all(action <= 1.0) and torch.all(action >= -1.0))

    def test_update_changes_parameters_without_nan(self):
        before = [parameter.detach().clone() for parameter in self.agent.actor.parameters()]
        metrics = self.agent.update(self.random_batch())
        after = list(self.agent.actor.parameters())
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(before, after)))
        self.assertTrue(np.all(np.isfinite(list(metrics.__dict__.values()))))

    def test_checkpoint_round_trip_preserves_deterministic_action(self):
        observation = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
        self.agent.update(self.random_batch())
        expected = self.agent.select_action(observation, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent.pt"
            self.agent.save(checkpoint, extra={"step": 7})
            restored = SACAgent(5, 3, self.config, device="cpu")
            metadata = restored.load(checkpoint)
            actual = restored.select_action(observation, deterministic=True)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
        self.assertEqual(metadata["step"], 7)


if __name__ == "__main__":
    unittest.main()
