"""M6 — a jointly-trained SOFT DECISION TREE policy head for CRYSTAL-1 (born-legible, feature-gated).

The C-4 pilot distilled a sklearn tree post-hoc; this is the REAL M6: a differentiable soft decision tree
(Frosst & Hinton 2017 style) that IS the PPO actor and is trained end-to-end. Two structural properties give the
controllability win:
  1. LEGIBILITY — the actor is a small tree; every action is a soft path to a leaf with a named action distribution.
  2. FEATURE-GATING — the tree reads ONLY a chosen subset of the observation (for CRYSTAL-1: [belief, inv, time],
     EXCLUDING the raw `burst` observable). So flipping burst at fixed belief cannot change the action distribution
     by construction — this closes the C-1 residual leak IN THE ACTION DISTRIBUTION, not just in return.
The critic sees the full observation (training-only; does not affect the deployed action distribution).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class SoftDecisionTree(nn.Module):
    """Depth-`depth` soft binary tree. Internal nodes are sigmoid gates over the input; leaves hold action
    distributions. Output = log of the path-probability-weighted mixture of leaf distributions (valid logits for a
    Categorical, since softmax(log p) = p for a normalized p)."""

    def __init__(self, in_dim: int, depth: int, n_actions: int, beta: float = 1.0):
        super().__init__()
        self.depth = int(depth)
        self.n_internal = 2 ** self.depth - 1
        self.n_leaves = 2 ** self.depth
        self.gate = nn.Linear(in_dim, self.n_internal)   # column j = node j's split hyperplane
        self.leaf_logits = nn.Parameter(torch.zeros(self.n_leaves, n_actions))
        self.beta = float(beta)
        # precompute (leaf -> list of (node, go_left)) paths
        self._paths = []
        for leaf in range(self.n_leaves):
            node, steps = 0, []
            for d in range(self.depth):
                go_left = ((leaf >> (self.depth - 1 - d)) & 1) == 0
                steps.append((node, go_left))
                node = 2 * node + 1 if go_left else 2 * node + 2
            self._paths.append(steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        g = torch.sigmoid(self.beta * self.gate(x))       # (B, n_internal) = P(go left) at each node
        path = torch.empty(B, self.n_leaves, device=x.device, dtype=g.dtype)
        for leaf, steps in enumerate(self._paths):
            p = torch.ones(B, device=x.device, dtype=g.dtype)
            for node, go_left in steps:
                gp = g[:, node]
                p = p * (gp if go_left else (1.0 - gp))
            path[:, leaf] = p
        leaf_prob = torch.softmax(self.leaf_logits, dim=-1)   # (n_leaves, n_actions)
        mix = path @ leaf_prob                                 # (B, n_actions)
        return torch.log(mix + 1e-9)

    @torch.no_grad()
    def leaf_report(self, n_actions: int):
        """Human-readable leaf table: per leaf the argmax action + its probability + the gate hyperplanes."""
        lp = torch.softmax(self.leaf_logits, dim=-1).cpu().numpy()
        W = self.gate.weight.detach().cpu().numpy(); b = self.gate.bias.detach().cpu().numpy()
        leaves = [{"leaf": i, "argmax_action": int(lp[i].argmax()), "action_probs": [round(float(v), 3) for v in lp[i]]}
                  for i in range(self.n_leaves)]
        gates = [{"node": j, "w": [round(float(v), 3) for v in W[j]], "b": round(float(b[j]), 3)} for j in range(self.n_internal)]
        return {"gates": gates, "leaves": leaves}


class SoftTreeActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose ACTOR is a SoftDecisionTree over a feature subset of the obs; the critic is a plain MLP over
    the full obs. Pass via policy_kwargs: feat_idx (obs indices the tree may read), tree_depth, beta, critic_arch."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 feat_idx=(0, 2, 1), tree_depth: int = 3, beta: float = 1.0, critic_arch=(64, 64), **kwargs):
        self._feat_idx = list(feat_idx)
        self._tree_depth = int(tree_depth)
        self._beta = float(beta)
        self._critic_arch = tuple(critic_arch)
        # strip net_arch/activation from kwargs (unused; our nets are custom) to avoid noisy warnings
        kwargs.pop("net_arch", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        # self.action_dist is already a MultiCategoricalDistribution (set in ActorCriticPolicy.__init__)
        n_actions = int(np.asarray(self.action_space.nvec).reshape(-1)[0])
        self.tree = SoftDecisionTree(len(self._feat_idx), self._tree_depth, n_actions, beta=self._beta)
        obs_dim = int(np.prod(self.observation_space.shape))
        layers, d = [], obs_dim
        for h in self._critic_arch:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers += [nn.Linear(d, 1)]
        self.critic = nn.Sequential(*layers)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    # ---- core overrides: action distribution from the tree over selected features; value from the full-obs critic ----
    def _sel(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.float()[:, self._feat_idx]

    def _distribution(self, obs: torch.Tensor):
        logits = self.tree(self._sel(obs))
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self._distribution(obs)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.critic(obs.float())
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self._distribution(obs)
        return self.critic(obs.float()), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs: torch.Tensor):
        return self._distribution(obs)

    def predict_values(self, obs: torch.Tensor):
        return self.critic(obs.float())

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        return self._distribution(observation).get_actions(deterministic=deterministic)
