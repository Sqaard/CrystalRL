"""Auxiliary action-value critic for PPO rollout interpretation.

The main Stage 0.1 trainer remains PPO with a state-value baseline.  This
module adds an optional, separate `Q(s, a)` regressor trained on the same
rollout returns.  It is intentionally auxiliary: it gives an action-value lens
for diagnostics without turning PPO into an off-policy actor-critic algorithm.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn


def flatdim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.MultiDiscrete):
        return int(np.sum(space.nvec))
    raise TypeError(f"Unsupported space for auxiliary Q critic: {type(space).__name__}")


def flatten_action(action_space: spaces.Space, actions: th.Tensor) -> th.Tensor:
    if isinstance(action_space, spaces.Box):
        return actions.reshape(actions.shape[0], -1).float()
    if isinstance(action_space, spaces.Discrete):
        return nn.functional.one_hot(actions.long().reshape(-1), num_classes=int(action_space.n)).float()
    if isinstance(action_space, spaces.MultiDiscrete):
        pieces = []
        action_int = actions.long().reshape(actions.shape[0], -1)
        for idx, n in enumerate(action_space.nvec):
            pieces.append(nn.functional.one_hot(action_int[:, idx], num_classes=int(n)).float())
        return th.cat(pieces, dim=1)
    raise TypeError(f"Unsupported action space for auxiliary Q critic: {type(action_space).__name__}")


class AuxiliaryQCritic(nn.Module):
    """Small MLP that predicts rollout return from observation and action."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("AuxiliaryQCritic currently expects a Box observation space.")
        self.observation_dim = int(np.prod(observation_space.shape))
        self.action_dim = flatdim(action_space)
        self.action_space = action_space
        layers: list[nn.Module] = []
        prev = self.observation_dim + self.action_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), activation()])
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
        obs = observations.reshape(observations.shape[0], -1).float()
        act = flatten_action(self.action_space, actions)
        return self.net(th.cat([obs, act], dim=1)).reshape(-1)
