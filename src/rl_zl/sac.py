"""Numerically stable PyTorch Soft Actor-Critic implementation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised on machines without training extras
    raise ImportError(
        "Stage 1 requires PyTorch. Activate the auv-rl environment and install "
        "a CUDA-compatible PyTorch build before running SAC training."
    ) from exc

from .replay import ReplayBatch
from .training_config import SACAlgorithmConfig


def set_global_seeds(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layer = nn.Linear(previous, width)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.ReLU()])
        previous = width
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        log_std_bounds: tuple[float, float],
    ):
        super().__init__()
        if log_std_bounds[0] >= log_std_bounds[1]:
            raise ValueError("log_std_bounds must be increasing")
        self.backbone = _mlp(observation_dim, hidden_dims, 2 * action_dim)
        self.action_dim = action_dim
        self.log_std_min, self.log_std_max = log_std_bounds

    def distribution_parameters(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.backbone(observation).chunk(2, dim=-1)
        log_std = torch.tanh(raw_log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def sample(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        with_log_probability: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        standard_deviation = log_std.exp()
        distribution = torch.distributions.Normal(mean, standard_deviation)
        pre_tanh = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = None
        if with_log_probability:
            # Stable tanh-Jacobian correction from the SAC reference implementation.
            correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
            log_probability = (distribution.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)
        return action, log_probability, mean, log_std


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
    ):
        super().__init__()
        self.network = _mlp(observation_dim + action_dim, hidden_dims, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


class TwinCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
    ):
        super().__init__()
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dims)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)


@dataclass(frozen=True)
class SACUpdateMetrics:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    q1_mean: float
    q2_mean: float
    target_q_mean: float
    entropy: float


class SACAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: SACAlgorithmConfig,
        device: str | torch.device = "auto",
    ):
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.actor = GaussianActor(
            observation_dim,
            action_dim,
            config.actor_hidden_dims,
            config.log_std_bounds,
        ).to(self.device)
        self.critic = TwinCritic(
            observation_dim,
            action_dim,
            config.critic_hidden_dims,
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=config.automatic_entropy_tuning,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
            if config.automatic_entropy_tuning
            else None
        )
        self.target_entropy = (
            -float(action_dim) if config.target_entropy is None else float(config.target_entropy)
        )
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def select_action(self, observation, deterministic: bool = False) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (self.observation_dim,):
            raise ValueError(f"Expected observation shape {(self.observation_dim,)}, got {observation_array.shape}")
        if not np.all(np.isfinite(observation_array)):
            raise ValueError("Observation contains NaN or Inf")
        observation_tensor = torch.as_tensor(observation_array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.actor.sample(
                observation_tensor,
                deterministic=deterministic,
                with_log_probability=False,
            )
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def update(self, batch: ReplayBatch) -> SACUpdateMetrics:
        observation = self._to_tensor(batch.observations)
        action = self._to_tensor(batch.actions)
        reward = self._to_tensor(batch.rewards)
        next_observation = self._to_tensor(batch.next_observations)
        terminated = self._to_tensor(batch.terminated)

        with torch.no_grad():
            next_action, next_log_probability, _, _ = self.actor.sample(next_observation)
            assert next_log_probability is not None
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target_q = torch.minimum(target_q1, target_q2) - self.alpha * next_log_probability
            bellman_target = reward + self.config.gamma * (1.0 - terminated) * target_q

        q1, q2 = self.critic(observation, action)
        critic_loss = F.mse_loss(q1, bellman_target) + F.mse_loss(q2, bellman_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
        self.critic_optimizer.step()

        policy_action, log_probability, _, _ = self.actor.sample(observation)
        assert log_probability is not None
        policy_q1, policy_q2 = self.critic(observation, policy_action)
        actor_loss = (self.alpha * log_probability - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.actor_optimizer.step()

        if self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.soft_update_target()

        values = (critic_loss, actor_loss, alpha_loss, self.alpha, q1, q2, bellman_target, log_probability)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("SAC update produced NaN or Inf")
        return SACUpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            alpha_loss=float(alpha_loss.detach().cpu()),
            alpha=float(self.alpha.cpu()),
            q1_mean=float(q1.detach().mean().cpu()),
            q2_mean=float(q2.detach().mean().cpu()),
            target_q_mean=float(bellman_target.detach().mean().cpu()),
            entropy=float(-log_probability.detach().mean().cpu()),
        )

    @torch.no_grad()
    def soft_update_target(self) -> None:
        tau = self.config.tau
        for target_parameter, parameter in zip(
            self.target_critic.parameters(), self.critic.parameters(), strict=True
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def checkpoint(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format_version": 1,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "update_count": self.update_count,
            "extra": dict(extra or {}),
        }

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(self.checkpoint(extra), temporary_path)
        temporary_path.replace(checkpoint_path)

    def load(self, path: str | Path, load_optimizers: bool = True) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported SAC checkpoint format")
        if payload["observation_dim"] != self.observation_dim or payload["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint dimensions do not match the current agent")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.update_count = int(payload.get("update_count", 0))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            if self.alpha_optimizer is not None and payload.get("alpha_optimizer") is not None:
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        return dict(payload.get("extra", {}))


__all__ = [
    "GaussianActor",
    "QNetwork",
    "SACAgent",
    "SACUpdateMetrics",
    "TwinCritic",
    "resolve_device",
    "set_global_seeds",
]
