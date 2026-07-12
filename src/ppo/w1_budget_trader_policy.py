"""Policy modules for W1 budget-PM / stock-selector Trader.

W1 intentionally keeps the actor contract small:

* PM actor samples a target risky exposure `q_target` and a discrete review
  horizon.
* Trader actor samples the daily full-portfolio simplex `[stocks..., cash]`.

Both modules expose the minimal PPO API used by the existing custom trainers:
`forward(obs) -> action, value, log_prob` and `evaluate_actions(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F


EPS = 1e-8


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], activation: type[nn.Module] = nn.Tanh) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, int(hidden)), activation()])
        prev = int(hidden)
    return nn.Sequential(*layers), prev


class GraphHierarchicalAssetEncoder(nn.Module):
    """Residual-correlation graph + group-pooling encoder for stock latents."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        group_ids: Sequence[int],
        relation_matrix: np.ndarray | None,
        layers: int = 1,
        use_group_context: bool = True,
        init_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers_count = int(max(0, layers))
        self.use_group_context = bool(use_group_context)
        if group_ids is None:
            group_ids = []
        self.group_count = int(max(group_ids) + 1) if len(group_ids) else 1
        self.register_buffer("group_ids", th.as_tensor(group_ids, dtype=th.long), persistent=False)
        if relation_matrix is None:
            relation = np.zeros((len(group_ids), len(group_ids)), dtype=np.float32)
        else:
            relation = np.asarray(relation_matrix, dtype=np.float32)
        self.register_buffer("relation_matrix", th.as_tensor(relation, dtype=th.float32), persistent=False)
        relation_active = bool(relation.size and float(np.abs(relation).sum()) > 0.0)
        self.enabled = bool(self.layers_count > 0 and (relation_active or self.use_group_context))
        message_parts = 2 + int(self.use_group_context)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim * message_parts, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.layers_count)
            ]
        )
        self.residual_scales = nn.ParameterList(
            [nn.Parameter(th.tensor(float(init_scale), dtype=th.float32)) for _ in range(self.layers_count)]
        )

    def _group_context(self, encoded: th.Tensor) -> th.Tensor:
        batch, stock_dim, hidden = encoded.shape
        out = th.zeros_like(encoded)
        if self.group_ids.numel() != stock_dim:
            return out
        for group in range(max(1, self.group_count)):
            mask = self.group_ids == group
            if bool(mask.any()):
                group_mean = encoded[:, mask, :].mean(dim=1, keepdim=True)
                out[:, mask, :] = group_mean.expand(batch, int(mask.sum().item()), hidden)
        return out

    def _relation_context(self, encoded: th.Tensor) -> th.Tensor:
        relation = self.relation_matrix.to(device=encoded.device, dtype=encoded.dtype)
        if relation.numel() == 0 or relation.shape[0] != encoded.shape[1]:
            return th.zeros_like(encoded)
        return th.einsum("ij,bjh->bih", relation, encoded)

    def forward(self, encoded: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if not self.enabled:
            zeros = th.zeros_like(encoded)
            return encoded, zeros, zeros
        out = encoded
        relation_ctx = th.zeros_like(encoded)
        group_ctx = th.zeros_like(encoded)
        for layer, scale in zip(self.layers, self.residual_scales):
            relation_ctx = self._relation_context(out)
            group_ctx = self._group_context(out) if self.use_group_context else th.zeros_like(out)
            message = th.cat([out, relation_ctx, group_ctx], dim=2)
            update = layer(message.reshape(out.shape[0] * out.shape[1], -1)).reshape_as(out)
            out = out + th.tanh(scale) * update
        return out, relation_ctx, group_ctx


def _beta_from_mean_kappa(mean_raw: th.Tensor, kappa_raw: th.Tensor, *, floor: float, kappa_min: float, kappa_max: float):
    mean = th.clamp(th.sigmoid(mean_raw), min=1e-4, max=1.0 - 1e-4)
    kappa = th.clamp(F.softplus(kappa_raw) + kappa_min, min=kappa_min, max=kappa_max)
    alpha = th.clamp(mean * kappa + floor, min=1e-4)
    beta = th.clamp((1.0 - mean) * kappa + floor, min=1e-4)
    return th.distributions.Beta(alpha, beta), mean


def _scale_q(unit_q: th.Tensor, q_min: float, q_max: float) -> th.Tensor:
    return q_min + unit_q * max(q_max - q_min, EPS)


def _unscale_q(q: th.Tensor, q_min: float, q_max: float) -> th.Tensor:
    return th.clamp((q - q_min) / max(q_max - q_min, EPS), min=1e-6, max=1.0 - 1e-6)


@dataclass
class PolicyStats:
    entropy: th.Tensor
    q_mean: th.Tensor


class BudgetPMActorCritic(nn.Module):
    """High-level PM: Beta risk/cash target and review horizon."""

    def __init__(
        self,
        obs_dim: int,
        *,
        horizon_choices: Sequence[int],
        hidden_dims: Sequence[int] = (256, 128),
        learning_rate: float = 1e-4,
        q_min: float = 0.0,
        q_max: float = 1.0,
        beta_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        aux_output_dim: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        if not horizon_choices:
            raise ValueError("BudgetPMActorCritic requires at least one horizon choice.")
        self.obs_dim = int(obs_dim)
        self.horizon_choices = [int(x) for x in horizon_choices]
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.beta_floor = float(beta_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.aux_output_dim = int(max(0, aux_output_dim))
        self.device = th.device(device)

        self.actor, actor_out = _make_mlp(self.obs_dim, hidden_dims)
        self.q_mean_head = nn.Linear(actor_out, 1)
        self.q_kappa_head = nn.Linear(actor_out, 1)
        self.horizon_head = nn.Linear(actor_out, len(self.horizon_choices))
        self.aux_head = nn.Linear(actor_out, self.aux_output_dim) if self.aux_output_dim > 0 else None

        self.critic, critic_out = _make_mlp(self.obs_dim, hidden_dims)
        self.value_head = nn.Linear(critic_out, 1)
        self.optimizer = th.optim.Adam(self.parameters(), lr=float(learning_rate))
        self.to(self.device)

    def _dists(self, obs: th.Tensor):
        latent = self.actor(obs)
        q_dist, q_mean = _beta_from_mean_kappa(
            self.q_mean_head(latent).squeeze(-1),
            self.q_kappa_head(latent).squeeze(-1),
            floor=self.beta_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )
        horizon_dist = th.distributions.Categorical(logits=self.horizon_head(latent))
        return q_dist, horizon_dist, q_mean

    def aux_predictions(self, obs: th.Tensor) -> th.Tensor:
        if self.aux_head is None:
            raise ValueError("BudgetPMActorCritic auxiliary head is disabled")
        obs = obs.to(self.device)
        latent = self.actor(obs)
        return self.aux_head(latent)

    def _value(self, obs: th.Tensor) -> th.Tensor:
        return self.value_head(self.critic(obs)).flatten()

    def forward(self, obs: th.Tensor, deterministic: bool = False, critic_obs: th.Tensor | None = None):
        obs = obs.to(self.device)
        q_dist, horizon_dist, q_mean = self._dists(obs)
        if deterministic:
            unit_q = q_mean
            horizon_idx = th.argmax(horizon_dist.logits, dim=1)
        else:
            unit_q = q_dist.sample()
            horizon_idx = horizon_dist.sample()
        q = _scale_q(unit_q, self.q_min, self.q_max)
        log_prob = q_dist.log_prob(unit_q) - np.log(max(self.q_max - self.q_min, EPS))
        log_prob = log_prob + horizon_dist.log_prob(horizon_idx)
        action = th.stack([q, horizon_idx.to(dtype=th.float32)], dim=1)
        return action, self._value(obs), log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, critic_obs: th.Tensor | None = None):
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        q = actions[:, 0]
        horizon_idx = th.clamp(actions[:, 1].round().long(), min=0, max=len(self.horizon_choices) - 1)
        unit_q = _unscale_q(q, self.q_min, self.q_max)
        q_dist, horizon_dist, _q_mean = self._dists(obs)
        log_prob = q_dist.log_prob(unit_q) - np.log(max(self.q_max - self.q_min, EPS))
        log_prob = log_prob + horizon_dist.log_prob(horizon_idx)
        entropy = q_dist.entropy() + horizon_dist.entropy()
        return self._value(obs), log_prob, entropy


class BudgetTraderActorCritic(nn.Module):
    """Low-level Trader: daily Dirichlet full-portfolio allocation.

    Observation layout:

    ```text
    [stock_0 features + task fields, ..., stock_N features + task fields]
    ```

    The same per-stock encoder is applied to every stock.  A mean pooled
    context lets every stock score see the current cross-sectional backdrop.
    """

    def __init__(
        self,
        obs_dim: int,
        *,
        stock_dim: int,
        stock_feature_dim: int,
        task_dim: int = 2,
        stock_hidden_dim: int = 64,
        critic_hidden_dims: Sequence[int] = (256, 128),
        learning_rate: float = 1e-4,
        alpha_min: float = 0.05,
        alpha_max: float = 100.0,
        ticker_embedding_dim: int = 0,
        group_ids: Sequence[int] | None = None,
        use_group_context: bool = False,
        relation_matrix: np.ndarray | None = None,
        relation_adapter_mode: str = "concat",
        relation_adapter_init_scale: float = 0.0,
        critic_extra_dim: int = 0,
        aux_per_stock_dim: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.stock_dim = int(stock_dim)
        self.stock_feature_dim = int(stock_feature_dim)
        self.task_dim = int(task_dim)
        self.per_stock_dim = self.stock_feature_dim + self.task_dim
        expected = self.stock_dim * self.per_stock_dim
        if self.obs_dim != expected:
            raise ValueError(
                f"BudgetTraderActorCritic obs_dim={self.obs_dim}, expected={expected} "
                f"(stock_dim={self.stock_dim}, per_stock_dim={self.per_stock_dim})"
            )
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.ticker_embedding_dim = int(max(0, ticker_embedding_dim))
        self.use_group_context = bool(use_group_context)
        self.relation_adapter_mode = str(relation_adapter_mode or "concat").lower()
        self.critic_extra_dim = int(max(0, critic_extra_dim))
        self.aux_per_stock_dim = int(max(0, aux_per_stock_dim))
        self.device = th.device(device)

        encoder_input_dim = self.per_stock_dim + self.ticker_embedding_dim
        self.ticker_embedding = nn.Embedding(self.stock_dim, self.ticker_embedding_dim) if self.ticker_embedding_dim > 0 else None
        if group_ids is None:
            group_ids = [0] * self.stock_dim
        if len(group_ids) != self.stock_dim:
            raise ValueError(f"group_ids length {len(group_ids)} != stock_dim {self.stock_dim}")
        self.register_buffer("group_ids", th.as_tensor(group_ids, dtype=th.long), persistent=False)
        self.group_count = int(max(group_ids) + 1) if group_ids else 1
        if relation_matrix is None:
            relation_matrix = np.zeros((self.stock_dim, self.stock_dim), dtype=np.float32)
        relation = th.as_tensor(relation_matrix, dtype=th.float32)
        if tuple(relation.shape) != (self.stock_dim, self.stock_dim):
            raise ValueError(f"relation_matrix shape {tuple(relation.shape)} != {(self.stock_dim, self.stock_dim)}")
        self.register_buffer("relation_matrix", relation, persistent=False)
        self.use_relation_context = bool(float(relation.abs().sum().detach().cpu()) > 0.0)
        if self.relation_adapter_mode not in {"concat", "residual_logit"}:
            raise ValueError(f"Unsupported relation_adapter_mode: {self.relation_adapter_mode}")

        self.stock_encoder, encoder_out = _make_mlp(encoder_input_dim, [stock_hidden_dim, stock_hidden_dim])
        concat_relation = self.use_relation_context and self.relation_adapter_mode == "concat"
        residual_relation = self.use_relation_context and self.relation_adapter_mode == "residual_logit"
        score_parts = 2 + int(self.use_group_context) + int(concat_relation)
        self.stock_score_head = nn.Linear(encoder_out * score_parts, 1)
        self.relation_score_head = nn.Linear(encoder_out, 1) if residual_relation else None
        self.relation_logit_scale = (
            nn.Parameter(th.tensor(float(relation_adapter_init_scale), dtype=th.float32)) if residual_relation else None
        )
        self.aux_stock_head = nn.Linear(encoder_out, self.aux_per_stock_dim) if self.aux_per_stock_dim > 0 else None
        self.cash_score_head = nn.Linear(encoder_out * 2, 1)
        critic_input_dim = encoder_out * 2
        if self.critic_extra_dim > 0:
            critic_input_dim += self.critic_extra_dim
        self.critic, critic_out = _make_mlp(critic_input_dim, critic_hidden_dims)
        self.value_head = nn.Linear(critic_out, 1)
        self.optimizer = th.optim.Adam(self.parameters(), lr=float(learning_rate))
        self.to(self.device)

    def _encode(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        batch = obs.shape[0]
        per_stock = obs.reshape(batch, self.stock_dim, self.per_stock_dim)
        if self.ticker_embedding is not None:
            ticker_ids = th.arange(self.stock_dim, device=obs.device).reshape(1, self.stock_dim).expand(batch, -1)
            tickers = self.ticker_embedding(ticker_ids)
            per_stock = th.cat([per_stock, tickers], dim=2)
        encoded = self.stock_encoder(per_stock.reshape(batch * self.stock_dim, -1))
        encoded = encoded.reshape(batch, self.stock_dim, -1)
        mean_ctx = encoded.mean(dim=1)
        std_ctx = encoded.std(dim=1, unbiased=False)
        return encoded, mean_ctx, std_ctx

    def _group_context(self, encoded: th.Tensor) -> th.Tensor:
        batch, _stock_dim, hidden = encoded.shape
        out = th.zeros_like(encoded)
        for group in range(max(1, self.group_count)):
            mask = self.group_ids == group
            if bool(mask.any()):
                group_mean = encoded[:, mask, :].mean(dim=1, keepdim=True)
                out[:, mask, :] = group_mean.expand(batch, int(mask.sum().item()), hidden)
        return out

    def _relation_context(self, encoded: th.Tensor) -> th.Tensor:
        relation = self.relation_matrix.to(device=encoded.device, dtype=encoded.dtype)
        return th.einsum("ij,bjh->bih", relation, encoded)

    def _dists(self, obs: th.Tensor):
        encoded, mean_ctx, std_ctx = self._encode(obs)
        batch = obs.shape[0]
        actor_ctx = mean_ctx.unsqueeze(1).expand(-1, self.stock_dim, -1)
        score_parts = [encoded, actor_ctx]
        if self.use_group_context:
            score_parts.append(self._group_context(encoded))
        relation_ctx = self._relation_context(encoded) if self.use_relation_context else None
        if relation_ctx is not None and self.relation_adapter_mode == "concat":
            score_parts.append(relation_ctx)
        score_input = th.cat(score_parts, dim=2)
        raw_alpha = self.stock_score_head(score_input.reshape(batch * self.stock_dim, -1)).reshape(batch, self.stock_dim)
        if relation_ctx is not None and self.relation_adapter_mode == "residual_logit":
            if self.relation_score_head is None or self.relation_logit_scale is None:
                raise RuntimeError("residual_logit relation adapter was not initialized")
            relation_delta = self.relation_score_head(relation_ctx.reshape(batch * self.stock_dim, -1)).reshape(batch, self.stock_dim)
            raw_alpha = raw_alpha + self.relation_logit_scale * relation_delta
        alpha = th.clamp(F.softplus(raw_alpha) + self.alpha_min, min=1e-4, max=self.alpha_max)
        q_ctx = th.cat([mean_ctx, std_ctx], dim=1)
        cash_alpha = th.clamp(F.softplus(self.cash_score_head(q_ctx)) + self.alpha_min, min=1e-4, max=self.alpha_max)
        portfolio_alpha = th.cat([alpha, cash_alpha], dim=1)
        portfolio_dist = th.distributions.Dirichlet(portfolio_alpha)

        return portfolio_dist, q_ctx

    def aux_predictions(self, obs: th.Tensor) -> th.Tensor:
        if self.aux_stock_head is None:
            raise ValueError("BudgetTraderActorCritic auxiliary head is disabled")
        obs = obs.to(self.device)
        encoded, _mean_ctx, _std_ctx = self._encode(obs)
        aux = self.aux_stock_head(encoded)
        return aux.reshape(obs.shape[0], self.stock_dim * self.aux_per_stock_dim)

    def _value_from_context(self, q_ctx: th.Tensor, critic_obs: th.Tensor | None = None) -> th.Tensor:
        if self.critic_extra_dim > 0:
            if critic_obs is None:
                critic_obs = th.zeros((q_ctx.shape[0], self.critic_extra_dim), dtype=q_ctx.dtype, device=q_ctx.device)
            else:
                critic_obs = critic_obs.to(device=q_ctx.device, dtype=q_ctx.dtype)
            if critic_obs.shape[1] != self.critic_extra_dim:
                raise ValueError(f"critic_obs dim {critic_obs.shape[1]} != expected {self.critic_extra_dim}")
            q_ctx = th.cat([q_ctx, critic_obs], dim=1)
        return self.value_head(self.critic(q_ctx)).flatten()

    def forward(self, obs: th.Tensor, deterministic: bool = False, critic_obs: th.Tensor | None = None):
        obs = obs.to(self.device)
        portfolio_dist, q_ctx = self._dists(obs)
        if deterministic:
            action = portfolio_dist.concentration / th.clamp(portfolio_dist.concentration.sum(dim=1, keepdim=True), min=EPS)
        else:
            action = portfolio_dist.sample()
        action = th.clamp(action, min=EPS)
        action = action / th.clamp(action.sum(dim=1, keepdim=True), min=EPS)
        log_prob = portfolio_dist.log_prob(action)
        return action, self._value_from_context(q_ctx, critic_obs), log_prob

    def decode_actions(self, obs: th.Tensor | np.ndarray, actions: th.Tensor | np.ndarray) -> th.Tensor:
        """Map raw policy actions to executable portfolio weights.

        The vanilla Trader already samples executable full-portfolio weights, so
        decoding is the identity.  Latent-action Trader variants override this.
        """

        if isinstance(actions, np.ndarray):
            return th.as_tensor(actions, dtype=th.float32, device=self.device)
        return actions.to(self.device)

    def action_diagnostics(self, obs: th.Tensor | np.ndarray, actions: th.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, critic_obs: th.Tensor | None = None):
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        action = th.clamp(actions, min=EPS)
        action = action / th.clamp(action.sum(dim=1, keepdim=True), min=EPS)
        portfolio_dist, q_ctx = self._dists(obs)
        log_prob = portfolio_dist.log_prob(action)
        entropy = portfolio_dist.entropy()
        return self._value_from_context(q_ctx, critic_obs), log_prob, entropy


class LatentActionTraderActorCritic(nn.Module):
    """Trader whose primary action is a latent action token.

    PPO stores and evaluates raw actions

    ```text
    [latent_code_index, residual_portfolio_simplex...]
    ```

    The environment receives decoded portfolio weights.  The decoded action is
    a prototype portfolio for the selected latent code, re-budgeted to the PM's
    current q_target, optionally blended with a residual Dirichlet allocation.
    """

    def __init__(
        self,
        obs_dim: int,
        *,
        stock_dim: int,
        stock_feature_dim: int,
        prototype_weights: np.ndarray,
        prototype_code_values: np.ndarray | Sequence[int] | None = None,
        task_dim: int = 2,
        stock_hidden_dim: int = 64,
        critic_hidden_dims: Sequence[int] = (256, 128),
        learning_rate: float = 1e-4,
        alpha_min: float = 0.05,
        alpha_max: float = 100.0,
        residual_mix: float = 0.10,
        ticker_embedding_dim: int = 0,
        group_ids: Sequence[int] | None = None,
        relation_matrix: np.ndarray | None = None,
        graph_layers: int = 0,
        graph_use_group_context: bool = True,
        graph_residual_init_scale: float = 0.10,
        two_channel: bool = False,
        two_channel_cash_threshold: float = 0.30,
        two_channel_risk_threshold: float = 0.15,
        critic_extra_dim: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.stock_dim = int(stock_dim)
        self.asset_dim = self.stock_dim + 1
        self.cash_index = self.stock_dim
        self.stock_feature_dim = int(stock_feature_dim)
        self.task_dim = int(task_dim)
        self.per_stock_dim = self.stock_feature_dim + self.task_dim
        expected = self.stock_dim * self.per_stock_dim
        if self.obs_dim != expected:
            raise ValueError(
                f"LatentActionTraderActorCritic obs_dim={self.obs_dim}, expected={expected} "
                f"(stock_dim={self.stock_dim}, per_stock_dim={self.per_stock_dim})"
            )
        prototypes = np.asarray(prototype_weights, dtype=np.float32)
        if prototypes.ndim != 2 or prototypes.shape[1] != self.asset_dim:
            raise ValueError(f"prototype_weights shape {prototypes.shape} != (*, {self.asset_dim})")
        prototypes = np.maximum(prototypes, 0.0)
        prototypes = prototypes / np.maximum(prototypes.sum(axis=1, keepdims=True), EPS)
        self.num_codes = int(prototypes.shape[0])
        if prototype_code_values is None:
            prototype_code_values = np.arange(self.num_codes, dtype=np.int64)
        code_values = np.asarray(prototype_code_values, dtype=np.int64).reshape(-1)
        if code_values.shape[0] != self.num_codes:
            raise ValueError(f"prototype_code_values length {code_values.shape[0]} != num_codes {self.num_codes}")
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.residual_mix = float(np.clip(residual_mix, 0.0, 1.0))
        self.two_channel = bool(two_channel)
        self.ticker_embedding_dim = int(max(0, ticker_embedding_dim))
        self.critic_extra_dim = int(max(0, critic_extra_dim))
        self.device = th.device(device)
        if group_ids is None:
            group_ids = [0] * self.stock_dim
        if len(group_ids) != self.stock_dim:
            raise ValueError(f"group_ids length {len(group_ids)} != stock_dim {self.stock_dim}")
        self.register_buffer("prototype_weights", th.as_tensor(prototypes, dtype=th.float32), persistent=True)
        self.register_buffer("prototype_code_values", th.as_tensor(code_values, dtype=th.long), persistent=True)

        encoder_input_dim = self.per_stock_dim + self.ticker_embedding_dim
        self.ticker_embedding = nn.Embedding(self.stock_dim, self.ticker_embedding_dim) if self.ticker_embedding_dim > 0 else None
        self.stock_encoder, encoder_out = _make_mlp(encoder_input_dim, [stock_hidden_dim, stock_hidden_dim])
        self.graph_encoder = GraphHierarchicalAssetEncoder(
            encoder_out,
            group_ids=group_ids,
            relation_matrix=relation_matrix,
            layers=int(graph_layers),
            use_group_context=bool(graph_use_group_context),
            init_scale=float(graph_residual_init_scale),
        )
        residual_score_parts = 4 if self.graph_encoder.enabled else 2
        proto_cash = prototypes[:, self.cash_index]
        macro_ids = np.full(self.num_codes, 1, dtype=np.int64)
        macro_ids[proto_cash >= float(two_channel_cash_threshold)] = 0
        macro_ids[proto_cash <= float(two_channel_risk_threshold)] = 2
        used_macros = sorted(np.unique(macro_ids).astype(int).tolist())
        macro_remap = {macro: idx for idx, macro in enumerate(used_macros)}
        macro_ids = np.asarray([macro_remap[int(macro)] for macro in macro_ids], dtype=np.int64)
        self.num_macro_codes = int(max(macro_ids.max() + 1, 1))
        self.register_buffer("prototype_macro_ids", th.as_tensor(macro_ids, dtype=th.long), persistent=False)
        if self.two_channel:
            self.macro_head = nn.Linear(encoder_out * 2, self.num_macro_codes)
            self.conditional_code_head = nn.Linear(encoder_out * 2, self.num_codes)
            self.code_head = None
        else:
            self.code_head = nn.Linear(encoder_out * 2, self.num_codes)
            self.macro_head = None
            self.conditional_code_head = None
        self.residual_stock_head = nn.Linear(encoder_out * residual_score_parts, 1)
        self.residual_cash_head = nn.Linear(encoder_out * 2, 1)

        critic_input_dim = encoder_out * 2
        if self.critic_extra_dim > 0:
            critic_input_dim += self.critic_extra_dim
        self.critic, critic_out = _make_mlp(critic_input_dim, critic_hidden_dims)
        self.value_head = nn.Linear(critic_out, 1)
        self.optimizer = th.optim.Adam(self.parameters(), lr=float(learning_rate))
        self.to(self.device)

    def _code_distribution(self, q_ctx: th.Tensor) -> th.distributions.Categorical:
        if not self.two_channel:
            if self.code_head is None:
                raise RuntimeError("code_head is disabled but two_channel is false")
            return th.distributions.Categorical(logits=self.code_head(q_ctx))
        if self.macro_head is None or self.conditional_code_head is None:
            raise RuntimeError("two_channel is enabled but macro/conditional heads are missing")
        macro_logp = F.log_softmax(self.macro_head(q_ctx), dim=1)
        code_logits = self.conditional_code_head(q_ctx)
        macro_ids = self.prototype_macro_ids.to(device=q_ctx.device)
        code_logp_parts: list[th.Tensor] = []
        for macro in range(self.num_macro_codes):
            mask = macro_ids == int(macro)
            masked_logits = code_logits.masked_fill(~mask.reshape(1, -1), -1e9)
            conditional = F.log_softmax(masked_logits, dim=1)
            code_logp_parts.append(macro_logp[:, macro : macro + 1] + conditional)
        full_logp = th.logsumexp(th.stack(code_logp_parts, dim=0), dim=0)
        return th.distributions.Categorical(logits=full_logp)

    def _encode(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        batch = obs.shape[0]
        per_stock = obs.reshape(batch, self.stock_dim, self.per_stock_dim)
        if self.ticker_embedding is not None:
            ticker_ids = th.arange(self.stock_dim, device=obs.device).reshape(1, self.stock_dim).expand(batch, -1)
            tickers = self.ticker_embedding(ticker_ids)
            per_stock = th.cat([per_stock, tickers], dim=2)
        encoded = self.stock_encoder(per_stock.reshape(batch * self.stock_dim, -1))
        encoded = encoded.reshape(batch, self.stock_dim, -1)
        encoded, relation_ctx, group_ctx = self.graph_encoder(encoded)
        mean_ctx = encoded.mean(dim=1)
        std_ctx = encoded.std(dim=1, unbiased=False)
        return encoded, mean_ctx, std_ctx, relation_ctx, group_ctx

    def _dists(self, obs: th.Tensor):
        encoded, mean_ctx, std_ctx, relation_ctx, group_ctx = self._encode(obs)
        batch = obs.shape[0]
        q_ctx = th.cat([mean_ctx, std_ctx], dim=1)
        code_dist = self._code_distribution(q_ctx)
        actor_ctx = mean_ctx.unsqueeze(1).expand(-1, self.stock_dim, -1)
        score_parts = [encoded, actor_ctx]
        if self.graph_encoder.enabled:
            score_parts.extend([relation_ctx, group_ctx])
        score_input = th.cat(score_parts, dim=2)
        raw_stock = self.residual_stock_head(score_input.reshape(batch * self.stock_dim, -1)).reshape(batch, self.stock_dim)
        stock_alpha = th.clamp(F.softplus(raw_stock) + self.alpha_min, min=1e-4, max=self.alpha_max)
        cash_alpha = th.clamp(F.softplus(self.residual_cash_head(q_ctx)) + self.alpha_min, min=1e-4, max=self.alpha_max)
        residual_dist = th.distributions.Dirichlet(th.cat([stock_alpha, cash_alpha], dim=1))
        return code_dist, residual_dist, q_ctx

    def _value_from_context(self, q_ctx: th.Tensor, critic_obs: th.Tensor | None = None) -> th.Tensor:
        if self.critic_extra_dim > 0:
            if critic_obs is None:
                critic_obs = th.zeros((q_ctx.shape[0], self.critic_extra_dim), dtype=q_ctx.dtype, device=q_ctx.device)
            else:
                critic_obs = critic_obs.to(device=q_ctx.device, dtype=q_ctx.dtype)
            if critic_obs.shape[1] != self.critic_extra_dim:
                raise ValueError(f"critic_obs dim {critic_obs.shape[1]} != expected {self.critic_extra_dim}")
            q_ctx = th.cat([q_ctx, critic_obs], dim=1)
        return self.value_head(self.critic(q_ctx)).flatten()

    def forward(self, obs: th.Tensor, deterministic: bool = False, critic_obs: th.Tensor | None = None):
        obs = obs.to(self.device)
        code_dist, residual_dist, q_ctx = self._dists(obs)
        if deterministic:
            code_idx = th.argmax(code_dist.logits, dim=1)
            residual = residual_dist.concentration / th.clamp(residual_dist.concentration.sum(dim=1, keepdim=True), min=EPS)
        else:
            code_idx = code_dist.sample()
            residual = residual_dist.sample()
        residual = th.clamp(residual, min=EPS)
        residual = residual / th.clamp(residual.sum(dim=1, keepdim=True), min=EPS)
        log_prob = code_dist.log_prob(code_idx) + residual_dist.log_prob(residual)
        action = th.cat([code_idx.to(dtype=th.float32).unsqueeze(1), residual], dim=1)
        return action, self._value_from_context(q_ctx, critic_obs), log_prob

    def _task_q_target(self, obs: th.Tensor) -> th.Tensor:
        per_stock = obs.reshape(obs.shape[0], self.stock_dim, self.per_stock_dim)
        prev_stock = per_stock[:, :, self.stock_feature_dim]
        signed_budget = per_stock[:, 0, self.stock_feature_dim + 1]
        q_prev = prev_stock.sum(dim=1)
        return th.clamp(q_prev + signed_budget, min=0.0, max=1.0)

    def decode_actions(self, obs: th.Tensor | np.ndarray, actions: th.Tensor | np.ndarray) -> th.Tensor:
        obs_t = th.as_tensor(obs, dtype=th.float32, device=self.device) if isinstance(obs, np.ndarray) else obs.to(self.device)
        actions_t = th.as_tensor(actions, dtype=th.float32, device=self.device) if isinstance(actions, np.ndarray) else actions.to(self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.reshape(1, -1)
        if actions_t.ndim == 1:
            actions_t = actions_t.reshape(1, -1)
        code_idx = th.clamp(actions_t[:, 0].round().long(), min=0, max=self.num_codes - 1)
        residual = th.clamp(actions_t[:, 1 : 1 + self.asset_dim], min=EPS)
        residual = residual / th.clamp(residual.sum(dim=1, keepdim=True), min=EPS)
        proto = self.prototype_weights[code_idx].clone()
        proto_stock = th.clamp(proto[:, : self.stock_dim], min=EPS)
        proto_stock = proto_stock / th.clamp(proto_stock.sum(dim=1, keepdim=True), min=EPS)
        q_target = self._task_q_target(obs_t).unsqueeze(1)
        proto_target = th.cat([q_target * proto_stock, 1.0 - q_target], dim=1)
        decoded = (1.0 - self.residual_mix) * proto_target + self.residual_mix * residual
        decoded = th.clamp(decoded, min=EPS)
        return decoded / th.clamp(decoded.sum(dim=1, keepdim=True), min=EPS)

    def action_diagnostics(self, obs: th.Tensor | np.ndarray, actions: th.Tensor | np.ndarray) -> dict[str, np.ndarray]:
        obs_t = th.as_tensor(obs, dtype=th.float32, device=self.device) if isinstance(obs, np.ndarray) else obs.to(self.device)
        actions_t = th.as_tensor(actions, dtype=th.float32, device=self.device) if isinstance(actions, np.ndarray) else actions.to(self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.reshape(1, -1)
        if actions_t.ndim == 1:
            actions_t = actions_t.reshape(1, -1)
        decoded = self.decode_actions(obs_t, actions_t)
        code_idx = th.clamp(actions_t[:, 0].round().long(), min=0, max=self.num_codes - 1)
        proto = self.prototype_weights[code_idx]
        actual_code = self.prototype_code_values[code_idx]
        code_dist, _residual_dist, _q_ctx = self._dists(obs_t)
        probs = th.softmax(code_dist.logits, dim=1)
        selected_prob = probs.gather(1, code_idx.reshape(-1, 1)).reshape(-1)
        top2 = th.topk(probs, k=min(2, self.num_codes), dim=1).values
        top1_prob = top2[:, 0]
        top2_prob = top2[:, 1] if top2.shape[1] > 1 else th.zeros_like(top1_prob)
        uniform_prob = 1.0 / max(self.num_codes, 1)
        selected_prob_norm = th.clamp((selected_prob - uniform_prob) / max(1.0 - uniform_prob, EPS), min=0.0, max=1.0)
        margin_strength = th.clamp((top1_prob - top2_prob) / max(1.0 - uniform_prob, EPS), min=0.0, max=1.0)
        entropy = code_dist.entropy()
        max_entropy = float(np.log(max(self.num_codes, 2)))
        entropy_strength = th.clamp(1.0 - entropy / max(max_entropy, EPS), min=0.0, max=1.0)
        probability_strength = th.sqrt(th.clamp(0.50 * selected_prob + 0.50 * top1_prob, min=0.0, max=1.0))
        primitive_confidence_strength = th.clamp(
            0.50 * selected_prob_norm + 0.30 * entropy_strength + 0.20 * margin_strength,
            min=0.0,
            max=1.0,
        )
        primitive_strength = th.clamp(
            0.70 * probability_strength + 0.30 * primitive_confidence_strength,
            min=0.0,
            max=1.0,
        )
        return {
            "latent_action_code": actual_code.detach().cpu().numpy().astype(np.int32),
            "latent_action_code_index": code_idx.detach().cpu().numpy().astype(np.int32),
            "latent_action_macro_code": self.prototype_macro_ids[code_idx].detach().cpu().numpy().astype(np.int32),
            "latent_action_two_channel": np.full(actions_t.shape[0], float(self.two_channel), dtype=np.float32),
            "latent_action_residual_mix": np.full(actions_t.shape[0], self.residual_mix, dtype=np.float32),
            "latent_action_proto_cash": proto[:, self.cash_index].detach().cpu().numpy().astype(np.float32),
            "latent_action_decoded_cash": decoded[:, self.cash_index].detach().cpu().numpy().astype(np.float32),
            "latent_action_selected_prob": selected_prob.detach().cpu().numpy().astype(np.float32),
            "latent_action_top1_prob": top1_prob.detach().cpu().numpy().astype(np.float32),
            "latent_action_top2_prob": top2_prob.detach().cpu().numpy().astype(np.float32),
            "latent_action_probability_strength": probability_strength.detach().cpu().numpy().astype(np.float32),
            "latent_action_confidence_strength": primitive_confidence_strength.detach().cpu().numpy().astype(np.float32),
            "latent_action_entropy_strength": entropy_strength.detach().cpu().numpy().astype(np.float32),
            "latent_action_margin_strength": margin_strength.detach().cpu().numpy().astype(np.float32),
            "latent_action_primitive_strength": primitive_strength.detach().cpu().numpy().astype(np.float32),
        }

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, critic_obs: th.Tensor | None = None):
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        code_idx = th.clamp(actions[:, 0].round().long(), min=0, max=self.num_codes - 1)
        residual = th.clamp(actions[:, 1 : 1 + self.asset_dim], min=EPS)
        residual = residual / th.clamp(residual.sum(dim=1, keepdim=True), min=EPS)
        code_dist, residual_dist, q_ctx = self._dists(obs)
        log_prob = code_dist.log_prob(code_idx) + residual_dist.log_prob(residual)
        entropy = code_dist.entropy() + residual_dist.entropy()
        return self._value_from_context(q_ctx, critic_obs), log_prob, entropy
