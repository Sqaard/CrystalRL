"""Deprecated off-policy actor-critic utilities for Stage 0.1 Q experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch as th
from torch import nn


def make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, int(hidden)), activation()])
        prev = int(hidden)
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Fixed-size replay buffer for flat continuous observations/actions."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, *, seed: int = 0) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.pos, 0] = float(reward)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[self.pos, 0] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, *, device: th.device | str) -> dict[str, th.Tensor]:
        if len(self) < 1:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = self.rng.integers(0, len(self), size=int(batch_size))
        return {
            "obs": th.as_tensor(self.obs[idx], dtype=th.float32, device=device),
            "actions": th.as_tensor(self.actions[idx], dtype=th.float32, device=device),
            "rewards": th.as_tensor(self.rewards[idx], dtype=th.float32, device=device),
            "next_obs": th.as_tensor(self.next_obs[idx], dtype=th.float32, device=device),
            "dones": th.as_tensor(self.dones[idx], dtype=th.float32, device=device),
        }


class ActionValueCritic(nn.Module):
    """Q(s, a) regressor."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256), *, device: str = "cpu") -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = th.device(device)
        self.net = make_mlp(self.obs_dim + self.action_dim, hidden_dims, 1)
        self.to(self.device)

    def forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        return self.net(th.cat([obs.float(), action.float()], dim=1)).reshape(-1, 1)


class DeterministicPMActor(nn.Module):
    """Continuous PM actor: risky budget q and normalized horizon control."""

    def __init__(self, obs_dim: int, hidden_dims: Sequence[int] = (256, 128), *, q_min: float = 0.0, q_max: float = 1.0, device: str = "cpu") -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = 2
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.device = th.device(device)
        self.net = make_mlp(self.obs_dim, hidden_dims, self.action_dim, activation=nn.Tanh)
        self.to(self.device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        raw = th.sigmoid(self.net(obs.float()))
        q = self.q_min + raw[:, :1] * max(self.q_max - self.q_min, 1e-8)
        horizon_unit = raw[:, 1:2]
        return th.cat([q, horizon_unit], dim=1)


class DeterministicTraderActor(nn.Module):
    """Shared per-stock actor that outputs a full portfolio simplex."""

    def __init__(
        self,
        obs_dim: int,
        *,
        stock_dim: int,
        stock_feature_dim: int,
        task_dim: int,
        stock_hidden_dim: int = 64,
        context_hidden_dims: Sequence[int] = (128,),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.stock_dim = int(stock_dim)
        self.asset_dim = self.stock_dim + 1
        self.stock_feature_dim = int(stock_feature_dim)
        self.task_dim = int(task_dim)
        self.per_stock_dim = self.stock_feature_dim + self.task_dim
        expected = self.stock_dim * self.per_stock_dim
        if self.obs_dim != expected:
            raise ValueError(f"Trader obs_dim={self.obs_dim}, expected={expected}")
        self.device = th.device(device)
        self.stock_encoder = make_mlp(self.per_stock_dim, [stock_hidden_dim, stock_hidden_dim], stock_hidden_dim, activation=nn.Tanh)
        self.stock_score = nn.Linear(stock_hidden_dim * 2, 1)
        self.cash_score = make_mlp(stock_hidden_dim * 2, context_hidden_dims, 1, activation=nn.Tanh)
        self.to(self.device)

    def _context(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        batch = obs.shape[0]
        per_stock = obs.reshape(batch, self.stock_dim, self.per_stock_dim)
        encoded = self.stock_encoder(per_stock.reshape(batch * self.stock_dim, self.per_stock_dim))
        encoded = encoded.reshape(batch, self.stock_dim, -1)
        mean_ctx = encoded.mean(dim=1)
        std_ctx = encoded.std(dim=1, unbiased=False)
        return encoded, th.cat([mean_ctx, std_ctx], dim=1)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        encoded, ctx = self._context(obs.float())
        batch = obs.shape[0]
        score_ctx = ctx[:, : encoded.shape[-1]].unsqueeze(1).expand(-1, self.stock_dim, -1)
        stock_logits = self.stock_score(th.cat([encoded, score_ctx], dim=2).reshape(batch * self.stock_dim, -1)).reshape(batch, self.stock_dim)
        cash_logit = self.cash_score(ctx)
        logits = th.cat([stock_logits, cash_logit], dim=1)
        return th.softmax(logits, dim=1)


@dataclass
class QUpdateStats:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_q_mean: float


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with th.no_grad():
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
