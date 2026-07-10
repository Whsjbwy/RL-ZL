"""Deterministic uniform replay buffer for the Stage-1 SAC teacher."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    """Fixed-size ring buffer that keeps termination and truncation separate."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
    ):
        if observation_dim <= 0 or action_dim <= 0 or capacity <= 0:
            raise ValueError("Replay dimensions and capacity must be positive")
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.truncated = np.empty((capacity, 1), dtype=np.float32)
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if observation.shape != self.observations.shape[1:]:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        if next_observation.shape != self.next_observations.shape[1:]:
            raise ValueError(f"Unexpected next-observation shape: {next_observation.shape}")
        if action.shape != self.actions.shape[1:]:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        if not (
            np.all(np.isfinite(observation))
            and np.all(np.isfinite(action))
            and np.all(np.isfinite(next_observation))
            and np.isfinite(reward)
        ):
            raise ValueError("Replay transition contains NaN or Inf")

        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index, 0] = float(reward)
        self.next_observations[index] = next_observation
        self.terminated[index, 0] = float(terminated)
        self.truncated[index, 0] = float(truncated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} transitions, need {batch_size}")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            observations=self.observations[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_observations=self.next_observations[indices].copy(),
            terminated=self.terminated[indices].copy(),
            truncated=self.truncated[indices].copy(),
        )


__all__ = ["ReplayBatch", "ReplayBuffer"]
