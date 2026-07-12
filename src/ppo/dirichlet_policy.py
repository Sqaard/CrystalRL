"""Dirichlet actor-critic policy for simplex portfolio weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.nn import functional as F


class DirichletDistribution(Distribution):
    """Torch Dirichlet distribution wrapper compatible with SB3 policies."""

    def __init__(self, action_dim: int, alpha_min: float = 1.0, alpha_max: float = 80.0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.distribution: th.distributions.Dirichlet | None = None
        self.alpha: th.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, raw_alpha: th.Tensor) -> "DirichletDistribution":
        alpha = F.softplus(raw_alpha) + self.alpha_min
        alpha = th.clamp(alpha, min=1e-4, max=self.alpha_max)
        self.alpha = alpha
        self.distribution = th.distributions.Dirichlet(alpha)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        actions = th.clamp(actions, min=1e-8)
        actions = actions / th.clamp(actions.sum(dim=1, keepdim=True), min=1e-8)
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        if self.alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        # The Dirichlet mode is undefined at boundaries when any alpha <= 1.
        # The mean is deterministic, stable, and valid for evaluation.
        return self.alpha / th.clamp(self.alpha.sum(dim=1, keepdim=True), min=1e-8)

    def actions_from_params(self, raw_alpha: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_alpha)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_alpha: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_alpha)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class DirichletActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy whose action distribution is Dirichlet on a simplex."""

    def __init__(
        self,
        *args: Any,
        alpha_min: float = 1.0,
        alpha_max: float = 80.0,
        **kwargs: Any,
    ):
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("DirichletActorCriticPolicy requires a 1-D Box action space.")

        action_dim = int(np.prod(self.action_space.shape))
        self.action_dist = DirichletDistribution(
            action_dim,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        raw_alpha = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_alpha)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(alpha_min=self.alpha_min, alpha_max=self.alpha_max)
        return data


class SharedStockMlpExtractor(nn.Module):
    """Shared per-stock actor scorer for stock-only Dirichlet policies.

    Observation layout expected by `TwoAgentBaseEnv.stock_obs()`:

    ```text
    per-stock selected features
    optional per-stock order-book proxy features
    portfolio state
    root private state
    ```

    The actor applies one scorer to each stock, which is the intended
    generalization structure for the low-level stock policy. The critic may
    still use the full flattened observation as a value baseline.
    """

    def __init__(
        self,
        *,
        features_dim: int,
        stock_dim: int,
        stock_feature_dim: int,
        order_book_proxy_dim: int = 0,
        global_context_dim: int = 12,
        activation_fn: type[nn.Module],
        stock_hidden_dim: int = 64,
        stock_group_ids: list[int] | None = None,
        ticker_embedding_dim: int = 0,
        asset_relation_mode: str = "none",
        vf_arch: list[int] | None = None,
    ):
        super().__init__()
        self.features_dim = int(features_dim)
        self.stock_dim = int(stock_dim)
        self.stock_feature_dim = int(stock_feature_dim)
        self.order_book_proxy_dim = max(0, int(order_book_proxy_dim))
        self.global_context_dim = max(0, int(global_context_dim))
        self.per_stock_dim = self.stock_feature_dim + self.order_book_proxy_dim
        self.stock_flat_dim = self.stock_dim * self.stock_feature_dim
        self.proxy_flat_dim = self.stock_dim * self.order_book_proxy_dim
        expected_dim = self.stock_flat_dim + self.proxy_flat_dim + self.global_context_dim
        if self.features_dim != expected_dim:
            raise ValueError(
                "SharedStockMlpExtractor observation mismatch: "
                f"features_dim={self.features_dim}, expected={expected_dim} "
                f"(stock_dim={self.stock_dim}, stock_feature_dim={self.stock_feature_dim}, "
                f"order_book_proxy_dim={self.order_book_proxy_dim}, global_context_dim={self.global_context_dim})"
            )

        if stock_group_ids is None:
            stock_group_ids = [0] * self.stock_dim
        if len(stock_group_ids) != self.stock_dim:
            raise ValueError("stock_group_ids length must match stock_dim.")
        group_ids = [int(x) for x in stock_group_ids]
        group_count = max(group_ids) + 1 if group_ids else 1
        self.group_count = group_count
        self.asset_relation_mode = str(asset_relation_mode).lower()
        if self.asset_relation_mode not in {"none", "group_one_hot"}:
            raise ValueError("asset_relation_mode must be 'none' or 'group_one_hot' for SharedStockDirichletPolicy.")
        group_one_hot = th.zeros(self.stock_dim, group_count, dtype=th.float32)
        for stock_idx, group_idx in enumerate(group_ids):
            group_one_hot[stock_idx, group_idx] = 1.0
        self.register_buffer("group_one_hot", group_one_hot)

        self.ticker_embedding_dim = max(0, int(ticker_embedding_dim))
        self.ticker_embedding = nn.Embedding(self.stock_dim, self.ticker_embedding_dim) if self.ticker_embedding_dim > 0 else None

        relation_dim = group_count if self.asset_relation_mode == "group_one_hot" else 0
        stock_input_dim = self.per_stock_dim + self.global_context_dim + relation_dim + self.ticker_embedding_dim
        self.stock_scorer, stock_out_dim = _make_mlp(
            stock_input_dim,
            [stock_hidden_dim, stock_hidden_dim],
            activation_fn,
        )
        self.stock_alpha_head = nn.Linear(stock_out_dim, 1)
        self.value_net, vf_out_dim = _make_mlp(
            self.features_dim,
            vf_arch or [256, 128],
            activation_fn,
        )
        # Actor latent is already raw per-stock Dirichlet scores.
        self.latent_dim_pi = self.stock_dim
        self.latent_dim_vf = vf_out_dim

    def _split_obs(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        batch_size = features.shape[0]
        stock_features = features[:, : self.stock_flat_dim].reshape(batch_size, self.stock_dim, self.stock_feature_dim)
        offset = self.stock_flat_dim
        if self.order_book_proxy_dim > 0:
            proxy = features[:, offset : offset + self.proxy_flat_dim].reshape(
                batch_size,
                self.stock_dim,
                self.order_book_proxy_dim,
            )
            offset += self.proxy_flat_dim
            per_stock = th.cat([stock_features, proxy], dim=2)
        else:
            per_stock = stock_features
        global_context = features[:, offset : offset + self.global_context_dim]
        return per_stock, global_context

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.shape[0]
        per_stock, global_context = self._split_obs(features)
        parts = [per_stock, global_context.unsqueeze(1).expand(-1, self.stock_dim, -1)]
        if self.asset_relation_mode == "group_one_hot":
            parts.append(self.group_one_hot.unsqueeze(0).expand(batch_size, -1, -1).to(features.device))
        if self.ticker_embedding is not None:
            ticker_ids = th.arange(self.stock_dim, dtype=th.long, device=features.device)
            parts.append(self.ticker_embedding(ticker_ids).unsqueeze(0).expand(batch_size, -1, -1))
        stock_input = th.cat(parts, dim=2)
        hidden = self.stock_scorer(stock_input.reshape(batch_size * self.stock_dim, -1))
        return self.stock_alpha_head(hidden).reshape(batch_size, self.stock_dim)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SharedStockDirichletActorCriticPolicy(DirichletActorCriticPolicy):
    """Dirichlet stock policy with one shared scorer applied to every stock."""

    def __init__(
        self,
        *args: Any,
        stock_dim: int,
        stock_feature_dim: int,
        order_book_proxy_dim: int = 0,
        global_context_dim: int = 12,
        stock_hidden_dim: int = 64,
        stock_group_ids: list[int] | None = None,
        ticker_embedding_dim: int = 0,
        asset_relation_mode: str = "none",
        **kwargs: Any,
    ):
        self.stock_dim = int(stock_dim)
        self.stock_feature_dim = int(stock_feature_dim)
        self.order_book_proxy_dim = max(0, int(order_book_proxy_dim))
        self.global_context_dim = max(0, int(global_context_dim))
        self.stock_hidden_dim = int(stock_hidden_dim)
        self.stock_group_ids = list(stock_group_ids) if stock_group_ids is not None else None
        self.ticker_embedding_dim = max(0, int(ticker_embedding_dim))
        self.asset_relation_mode = str(asset_relation_mode).lower()
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        vf_arch = [256, 128]
        if isinstance(self.net_arch, dict):
            vf_arch = list(self.net_arch.get("vf", vf_arch))
        self.mlp_extractor = SharedStockMlpExtractor(
            features_dim=self.features_dim,
            stock_dim=self.stock_dim,
            stock_feature_dim=self.stock_feature_dim,
            order_book_proxy_dim=self.order_book_proxy_dim,
            global_context_dim=self.global_context_dim,
            activation_fn=self.activation_fn,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_group_ids=self.stock_group_ids,
            ticker_embedding_dim=self.ticker_embedding_dim,
            asset_relation_mode=self.asset_relation_mode,
            vf_arch=vf_arch,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("SharedStockDirichletActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim={self.stock_dim}")
        self.action_dist = DirichletDistribution(
            action_dim,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            stock_dim=self.stock_dim,
            stock_feature_dim=self.stock_feature_dim,
            order_book_proxy_dim=self.order_book_proxy_dim,
            global_context_dim=self.global_context_dim,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_group_ids=self.stock_group_ids,
            ticker_embedding_dim=self.ticker_embedding_dim,
            asset_relation_mode=self.asset_relation_mode,
        )
        return data


class SharedStockBranchingExecutionMlpExtractor(nn.Module):
    """Shared per-stock actor for synthetic price/quantity execution branches."""

    def __init__(
        self,
        *,
        features_dim: int,
        stock_dim: int,
        stock_feature_dim: int,
        order_book_proxy_dim: int,
        execution_task_dim: int = 4,
        global_context_dim: int = 12,
        price_levels: int = 5,
        quantity_levels: int = 5,
        activation_fn: type[nn.Module],
        stock_hidden_dim: int = 64,
        stock_group_ids: list[int] | None = None,
        ticker_embedding_dim: int = 0,
        asset_relation_mode: str = "group_one_hot",
        vf_arch: list[int] | None = None,
    ):
        super().__init__()
        self.features_dim = int(features_dim)
        self.stock_dim = int(stock_dim)
        self.stock_feature_dim = int(stock_feature_dim)
        self.order_book_proxy_dim = int(order_book_proxy_dim)
        self.execution_task_dim = int(execution_task_dim)
        self.global_context_dim = int(global_context_dim)
        self.price_levels = int(price_levels)
        self.quantity_levels = int(quantity_levels)
        self.stock_flat_dim = self.stock_dim * self.stock_feature_dim
        self.proxy_flat_dim = self.stock_dim * self.order_book_proxy_dim
        self.task_flat_dim = self.stock_dim * self.execution_task_dim
        expected_dim = self.stock_flat_dim + self.proxy_flat_dim + self.task_flat_dim + self.global_context_dim
        if self.features_dim != expected_dim:
            raise ValueError(
                "SharedStockBranchingExecution observation mismatch: "
                f"features_dim={self.features_dim}, expected={expected_dim} "
                f"(stock_dim={self.stock_dim}, stock_feature_dim={self.stock_feature_dim}, "
                f"order_book_proxy_dim={self.order_book_proxy_dim}, execution_task_dim={self.execution_task_dim}, "
                f"global_context_dim={self.global_context_dim})"
            )
        if stock_group_ids is None:
            stock_group_ids = [0] * self.stock_dim
        if len(stock_group_ids) != self.stock_dim:
            raise ValueError("stock_group_ids length must match stock_dim.")
        group_ids = [int(x) for x in stock_group_ids]
        group_count = max(group_ids) + 1 if group_ids else 1
        self.group_count = group_count
        self.asset_relation_mode = str(asset_relation_mode).lower()
        if self.asset_relation_mode not in {"none", "group_one_hot"}:
            raise ValueError("asset_relation_mode must be 'none' or 'group_one_hot'.")
        group_one_hot = th.zeros(self.stock_dim, group_count, dtype=th.float32)
        for stock_idx, group_idx in enumerate(group_ids):
            group_one_hot[stock_idx, group_idx] = 1.0
        self.register_buffer("group_one_hot", group_one_hot)

        self.ticker_embedding_dim = max(0, int(ticker_embedding_dim))
        self.ticker_embedding = nn.Embedding(self.stock_dim, self.ticker_embedding_dim) if self.ticker_embedding_dim > 0 else None

        relation_dim = group_count if self.asset_relation_mode == "group_one_hot" else 0
        per_stock_input_dim = (
            self.stock_feature_dim
            + self.order_book_proxy_dim
            + self.execution_task_dim
            + self.global_context_dim
            + relation_dim
            + self.ticker_embedding_dim
        )
        self.stock_scorer, stock_out_dim = _make_mlp(
            per_stock_input_dim,
            [stock_hidden_dim, stock_hidden_dim],
            activation_fn,
        )
        self.price_head = nn.Linear(stock_out_dim, self.price_levels)
        self.quantity_head = nn.Linear(stock_out_dim, self.quantity_levels)
        self.value_net, vf_out_dim = _make_mlp(self.features_dim, vf_arch or [256, 128], activation_fn)
        self.latent_dim_pi = self.stock_dim * (self.price_levels + self.quantity_levels)
        self.latent_dim_vf = vf_out_dim

    def _split_obs(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        batch_size = features.shape[0]
        offset = 0
        stock_features = features[:, offset : offset + self.stock_flat_dim].reshape(
            batch_size, self.stock_dim, self.stock_feature_dim
        )
        offset += self.stock_flat_dim
        proxy = features[:, offset : offset + self.proxy_flat_dim].reshape(
            batch_size, self.stock_dim, self.order_book_proxy_dim
        )
        offset += self.proxy_flat_dim
        task = features[:, offset : offset + self.task_flat_dim].reshape(
            batch_size, self.stock_dim, self.execution_task_dim
        )
        offset += self.task_flat_dim
        global_context = features[:, offset : offset + self.global_context_dim]
        per_stock = th.cat([stock_features, proxy, task], dim=2)
        return per_stock, global_context

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.shape[0]
        per_stock, global_context = self._split_obs(features)
        parts = [per_stock, global_context.unsqueeze(1).expand(-1, self.stock_dim, -1)]
        if self.asset_relation_mode == "group_one_hot":
            parts.append(self.group_one_hot.unsqueeze(0).expand(batch_size, -1, -1).to(features.device))
        if self.ticker_embedding is not None:
            ticker_ids = th.arange(self.stock_dim, dtype=th.long, device=features.device)
            parts.append(self.ticker_embedding(ticker_ids).unsqueeze(0).expand(batch_size, -1, -1))
        stock_input = th.cat(parts, dim=2)
        hidden = self.stock_scorer(stock_input.reshape(batch_size * self.stock_dim, -1))
        price_logits = self.price_head(hidden).reshape(batch_size, self.stock_dim * self.price_levels)
        quantity_logits = self.quantity_head(hidden).reshape(batch_size, self.stock_dim * self.quantity_levels)
        return th.cat([price_logits, quantity_logits], dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SharedStockBranchingExecutionPolicy(ActorCriticPolicy):
    """MultiDiscrete price/quantity execution policy with shared per-stock scorer."""

    def __init__(
        self,
        *args: Any,
        stock_dim: int,
        stock_feature_dim: int,
        order_book_proxy_dim: int,
        execution_task_dim: int = 4,
        global_context_dim: int = 12,
        price_levels: int = 5,
        quantity_levels: int = 5,
        stock_hidden_dim: int = 64,
        stock_group_ids: list[int] | None = None,
        ticker_embedding_dim: int = 0,
        asset_relation_mode: str = "group_one_hot",
        **kwargs: Any,
    ):
        self.stock_dim = int(stock_dim)
        self.stock_feature_dim = int(stock_feature_dim)
        self.order_book_proxy_dim = int(order_book_proxy_dim)
        self.execution_task_dim = int(execution_task_dim)
        self.global_context_dim = int(global_context_dim)
        self.price_levels = int(price_levels)
        self.quantity_levels = int(quantity_levels)
        self.stock_hidden_dim = int(stock_hidden_dim)
        self.stock_group_ids = list(stock_group_ids) if stock_group_ids is not None else None
        self.ticker_embedding_dim = max(0, int(ticker_embedding_dim))
        self.asset_relation_mode = str(asset_relation_mode).lower()
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        vf_arch = [256, 128]
        if isinstance(self.net_arch, dict):
            vf_arch = list(self.net_arch.get("vf", vf_arch))
        self.mlp_extractor = SharedStockBranchingExecutionMlpExtractor(
            features_dim=self.features_dim,
            stock_dim=self.stock_dim,
            stock_feature_dim=self.stock_feature_dim,
            order_book_proxy_dim=self.order_book_proxy_dim,
            execution_task_dim=self.execution_task_dim,
            global_context_dim=self.global_context_dim,
            price_levels=self.price_levels,
            quantity_levels=self.quantity_levels,
            activation_fn=self.activation_fn,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_group_ids=self.stock_group_ids,
            ticker_embedding_dim=self.ticker_embedding_dim,
            asset_relation_mode=self.asset_relation_mode,
            vf_arch=vf_arch,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            raise ValueError("SharedStockBranchingExecutionPolicy requires a MultiDiscrete action space.")
        expected_nvec = [self.price_levels] * self.stock_dim + [self.quantity_levels] * self.stock_dim
        if list(map(int, self.action_space.nvec.tolist())) != expected_nvec:
            raise ValueError("Execution action_space.nvec does not match price/quantity branch dimensions.")
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            stock_dim=self.stock_dim,
            stock_feature_dim=self.stock_feature_dim,
            order_book_proxy_dim=self.order_book_proxy_dim,
            execution_task_dim=self.execution_task_dim,
            global_context_dim=self.global_context_dim,
            price_levels=self.price_levels,
            quantity_levels=self.quantity_levels,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_group_ids=self.stock_group_ids,
            ticker_embedding_dim=self.ticker_embedding_dim,
            asset_relation_mode=self.asset_relation_mode,
        )
        return data


class HierarchicalDirichletDistribution(Distribution):
    """Dirichlet-tree distribution over final portfolio leaf weights."""

    def __init__(
        self,
        action_dim: int,
        group_indices: list[list[int]],
        alpha_min: float = 1.0,
        alpha_max: float = 80.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.group_indices = [list(group) for group in group_indices]
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.root_dim = len(self.group_indices)
        self.inner_group_indices = [i for i, group in enumerate(self.group_indices) if len(group) > 1]
        self.param_dim = self.root_dim + sum(len(self.group_indices[i]) for i in self.inner_group_indices)
        self.root_dist: th.distributions.Dirichlet | None = None
        self.inner_dists: list[tuple[int, th.distributions.Dirichlet]] = []
        self.root_alpha: th.Tensor | None = None
        self.inner_alphas: list[tuple[int, th.Tensor]] = []

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.param_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "HierarchicalDirichletDistribution":
        root_raw = raw_params[:, : self.root_dim]
        root_alpha = th.clamp(F.softplus(root_raw) + self.alpha_min, min=1e-4, max=self.alpha_max)
        self.root_alpha = root_alpha
        self.root_dist = th.distributions.Dirichlet(root_alpha)

        self.inner_dists = []
        self.inner_alphas = []
        offset = self.root_dim
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner_raw = raw_params[:, offset : offset + group_size]
            offset += group_size
            inner_alpha = th.clamp(F.softplus(inner_raw) + self.alpha_min, min=1e-4, max=self.alpha_max)
            self.inner_alphas.append((group_idx, inner_alpha))
            self.inner_dists.append((group_idx, th.distributions.Dirichlet(inner_alpha)))
        return self

    def _compose_leaf_weights(self, root_weights: th.Tensor, inner_weights: list[tuple[int, th.Tensor]]) -> th.Tensor:
        batch_size = root_weights.shape[0]
        actions = th.zeros((batch_size, self.action_dim), dtype=root_weights.dtype, device=root_weights.device)
        inner_by_group = {group_idx: weights for group_idx, weights in inner_weights}
        for group_idx, leaf_indices in enumerate(self.group_indices):
            budget = root_weights[:, group_idx : group_idx + 1]
            idx = th.as_tensor(leaf_indices, dtype=th.long, device=root_weights.device)
            if len(leaf_indices) == 1:
                actions[:, idx] = budget
            else:
                actions[:, idx] = budget * inner_by_group[group_idx]
        return actions

    def sample(self) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        root_sample = self.root_dist.sample()
        inner_samples = [(group_idx, dist.sample()) for group_idx, dist in self.inner_dists]
        return self._compose_leaf_weights(root_sample, inner_samples)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        root_mean = self.root_alpha / th.clamp(self.root_alpha.sum(dim=1, keepdim=True), min=1e-8)
        inner_means = [
            (group_idx, alpha / th.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8))
            for group_idx, alpha in self.inner_alphas
        ]
        return self._compose_leaf_weights(root_mean, inner_means)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        actions = th.clamp(actions, min=1e-8)
        actions = actions / th.clamp(actions.sum(dim=1, keepdim=True), min=1e-8)

        root_parts = []
        for leaf_indices in self.group_indices:
            idx = th.as_tensor(leaf_indices, dtype=th.long, device=actions.device)
            root_parts.append(actions[:, idx].sum(dim=1, keepdim=True))
        root_weights = th.cat(root_parts, dim=1)
        root_weights = th.clamp(root_weights, min=1e-8)
        root_weights = root_weights / th.clamp(root_weights.sum(dim=1, keepdim=True), min=1e-8)

        log_prob = self.root_dist.log_prob(root_weights)
        jacobian_log = th.zeros_like(log_prob)

        for group_idx, inner_dist in self.inner_dists:
            leaf_indices = self.group_indices[group_idx]
            idx = th.as_tensor(leaf_indices, dtype=th.long, device=actions.device)
            budget = th.clamp(root_weights[:, group_idx : group_idx + 1], min=1e-8)
            inner_weights = actions[:, idx] / budget
            inner_weights = th.clamp(inner_weights, min=1e-8)
            inner_weights = inner_weights / th.clamp(inner_weights.sum(dim=1, keepdim=True), min=1e-8)
            log_prob = log_prob + inner_dist.log_prob(inner_weights)
            jacobian_log = jacobian_log + (len(leaf_indices) - 1) * th.log(budget.squeeze(1))

        # Dirichlet-tree density over leaf weights: p(y) / |J|, where
        # |J| = product_g sector_weight_g ** (n_g - 1).
        return log_prob - jacobian_log

    def entropy(self) -> None:
        # Closed-form entropy for the transformed leaf distribution is not
        # implemented here. SB3 falls back to a log-prob approximation.
        return None

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class HierarchicalDirichletActorCriticPolicy(DirichletActorCriticPolicy):
    """Actor-critic policy with a Dirichlet-tree distribution over leaf weights."""

    def __init__(
        self,
        *args: Any,
        group_indices: list[list[int]],
        alpha_min: float = 1.0,
        alpha_max: float = 80.0,
        **kwargs: Any,
    ):
        self.group_indices = [list(group) for group in group_indices]
        super().__init__(*args, alpha_min=alpha_min, alpha_max=alpha_max, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("HierarchicalDirichletActorCriticPolicy requires a 1-D Box action space.")

        action_dim = int(np.prod(self.action_space.shape))
        flat_indices = sorted(idx for group in self.group_indices for idx in group)
        if flat_indices != list(range(action_dim)):
            raise ValueError(
                "group_indices must partition all action dimensions exactly once. "
                f"got={flat_indices}, expected={list(range(action_dim))}"
            )

        self.action_dist = HierarchicalDirichletDistribution(
            action_dim,
            group_indices=self.group_indices,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(group_indices=self.group_indices)
        return data


class RootSplitBetaDirichletDistribution(Distribution):
    """Beta root split plus Dirichlet risky allocation.

    The sampled action is a factor vector, not final portfolio weights:

    action[:, 0]  = q, invested fraction
    action[:, 1:] = u, conditional risky allocation over stocks

    The environment maps this deterministically to final target weights:

    cash = 1 - q
    stock_i = q * u_i
    """

    def __init__(
        self,
        stock_dim: int,
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        risky_alpha_max: float = 100.0,
    ):
        super().__init__()
        self.stock_dim = int(stock_dim)
        self.action_dim = self.stock_dim + 1
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if not 0.0 <= self.q_min < self.q_max <= 1.0:
            raise ValueError(f"Invalid q bounds: q_min={q_min}, q_max={q_max}")
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.risky_alpha_max = float(risky_alpha_max)
        self.root_dist: th.distributions.Beta | None = None
        self.risky_dist: th.distributions.Dirichlet | None = None
        self.root_alpha: th.Tensor | None = None
        self.root_beta: th.Tensor | None = None
        self.risky_alpha: th.Tensor | None = None
        self.q_mean_unit: th.Tensor | None = None

    @property
    def q_range(self) -> float:
        return self.q_max - self.q_min

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 + self.stock_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "RootSplitBetaDirichletDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        risky_raw = raw_params[:, 2:]

        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)

        root_alpha = self.alpha_floor + q_mean_unit * kappa
        root_beta = self.alpha_floor + (1.0 - q_mean_unit) * kappa
        root_alpha = th.clamp(root_alpha, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(root_beta, min=1e-4, max=self.kappa_max + self.alpha_floor)

        risky_alpha = F.softplus(risky_raw) + self.alpha_floor
        risky_alpha = th.clamp(risky_alpha, min=1e-4, max=self.risky_alpha_max)

        self.q_mean_unit = q_mean_unit
        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.risky_alpha = risky_alpha
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        self.risky_dist = th.distributions.Dirichlet(risky_alpha)
        return self

    def _compose_action(self, q_unit: th.Tensor, risky_weights: th.Tensor) -> th.Tensor:
        q = self.q_min + self.q_range * q_unit
        return th.cat([q.unsqueeze(-1), risky_weights], dim=1)

    def sample(self) -> th.Tensor:
        if self.root_dist is None or self.risky_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = self.root_dist.sample()
        risky = self.risky_dist.sample()
        return self._compose_action(q_unit, risky)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None or self.risky_alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        risky = self.risky_alpha / th.clamp(self.risky_alpha.sum(dim=1, keepdim=True), min=1e-8)
        return self._compose_action(q_unit, risky)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None or self.risky_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q = th.clamp(actions[:, 0], min=self.q_min + 1e-6, max=self.q_max - 1e-6)
        q_unit = (q - self.q_min) / self.q_range
        q_unit = th.clamp(q_unit, min=1e-6, max=1.0 - 1e-6)
        risky = th.clamp(actions[:, 1:], min=1e-8)
        risky = risky / th.clamp(risky.sum(dim=1, keepdim=True), min=1e-8)

        root_log_prob = self.root_dist.log_prob(q_unit) - np.log(self.q_range)
        risky_log_prob = self.risky_dist.log_prob(risky)
        return root_log_prob + risky_log_prob

    def entropy(self) -> th.Tensor:
        if self.root_dist is None or self.risky_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.root_dist.entropy() + self.risky_dist.entropy()

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class RootBetaDistribution(Distribution):
    """Scalar Beta distribution over invested fraction q in [q_min, q_max]."""

    def __init__(
        self,
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
    ):
        super().__init__()
        self.action_dim = 1
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if not 0.0 <= self.q_min < self.q_max <= 1.0:
            raise ValueError(f"Invalid q bounds: q_min={q_min}, q_max={q_max}")
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.root_dist: th.distributions.Beta | None = None
        self.root_alpha: th.Tensor | None = None
        self.root_beta: th.Tensor | None = None

    @property
    def q_range(self) -> float:
        return self.q_max - self.q_min

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2)

    def proba_distribution(self, raw_params: th.Tensor) -> "RootBetaDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)
        root_alpha = self.alpha_floor + q_mean_unit * kappa
        root_beta = self.alpha_floor + (1.0 - q_mean_unit) * kappa
        root_alpha = th.clamp(root_alpha, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(root_beta, min=1e-4, max=self.kappa_max + self.alpha_floor)
        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        return self

    def _compose_action(self, q_unit: th.Tensor) -> th.Tensor:
        q = self.q_min + self.q_range * q_unit
        return q.unsqueeze(-1)

    def sample(self) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self._compose_action(self.root_dist.sample())

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        return self._compose_action(q_unit)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q = th.clamp(actions[:, 0], min=self.q_min + 1e-6, max=self.q_max - 1e-6)
        q_unit = (q - self.q_min) / self.q_range
        q_unit = th.clamp(q_unit, min=1e-6, max=1.0 - 1e-6)
        return self.root_dist.log_prob(q_unit) - np.log(self.q_range)

    def entropy(self) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.root_dist.entropy()

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class RootBetaActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with a scalar Beta action for root risk/cash."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        **kwargs,
    ):
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_space, spaces.Box) or tuple(self.action_space.shape) != (1,):
            raise ValueError("RootBetaActorCriticPolicy requires a 1-D Box action space with shape=(1,).")
        self.action_dist = RootBetaDistribution(
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        raw_params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_params)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )
        return data


class RootSplitBetaDirichletKpDistribution(RootSplitBetaDirichletDistribution):
    """Root split plus risky allocation plus stochastic bounded Kp gate factors.

    The sampled action is:

    action[:, 0]                 = q, invested fraction
    action[:, 1:1 + stock_dim]   = u, conditional risky allocation
    action[:, 1 + stock_dim]     = z_root_gate in [0, 1]
    action[:, 2 + stock_dim]     = z_inner_gate in [0, 1]
    """

    def __init__(
        self,
        stock_dim: int,
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        risky_alpha_max: float = 100.0,
        gate_kappa_min: float = 8.0,
        gate_kappa_max: float = 80.0,
    ):
        super().__init__(
            stock_dim,
            q_min=q_min,
            q_max=q_max,
            alpha_floor=alpha_floor,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            risky_alpha_max=risky_alpha_max,
        )
        self.action_dim = self.stock_dim + 3
        self.gate_kappa_min = float(gate_kappa_min)
        self.gate_kappa_max = float(gate_kappa_max)
        self.root_gate_dist: th.distributions.Beta | None = None
        self.inner_gate_dist: th.distributions.Beta | None = None
        self.root_gate_alpha: th.Tensor | None = None
        self.root_gate_beta: th.Tensor | None = None
        self.inner_gate_alpha: th.Tensor | None = None
        self.inner_gate_beta: th.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 + self.stock_dim + 4)

    def _gate_alpha_beta(self, mean_logit: th.Tensor, kappa_raw: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean = th.sigmoid(mean_logit)
        kappa = self.gate_kappa_min + F.softplus(kappa_raw)
        kappa = th.clamp(kappa, min=self.gate_kappa_min, max=self.gate_kappa_max)
        alpha = th.clamp(self.alpha_floor + mean * kappa, min=1e-4, max=self.gate_kappa_max + self.alpha_floor)
        beta = th.clamp(self.alpha_floor + (1.0 - mean) * kappa, min=1e-4, max=self.gate_kappa_max + self.alpha_floor)
        return alpha, beta

    def proba_distribution(self, raw_params: th.Tensor) -> "RootSplitBetaDirichletKpDistribution":
        super().proba_distribution(raw_params[:, : 2 + self.stock_dim])
        gate_raw = raw_params[:, 2 + self.stock_dim :]
        root_gate_alpha, root_gate_beta = self._gate_alpha_beta(gate_raw[:, 0:1], gate_raw[:, 1:2])
        inner_gate_alpha, inner_gate_beta = self._gate_alpha_beta(gate_raw[:, 2:3], gate_raw[:, 3:4])

        self.root_gate_alpha = root_gate_alpha
        self.root_gate_beta = root_gate_beta
        self.inner_gate_alpha = inner_gate_alpha
        self.inner_gate_beta = inner_gate_beta
        self.root_gate_dist = th.distributions.Beta(root_gate_alpha.squeeze(-1), root_gate_beta.squeeze(-1))
        self.inner_gate_dist = th.distributions.Beta(inner_gate_alpha.squeeze(-1), inner_gate_beta.squeeze(-1))
        return self

    def sample(self) -> th.Tensor:
        base = super().sample()
        if self.root_gate_dist is None or self.inner_gate_dist is None:
            raise RuntimeError("Gate distribution parameters are not initialized.")
        root_gate = self.root_gate_dist.sample().unsqueeze(-1)
        inner_gate = self.inner_gate_dist.sample().unsqueeze(-1)
        return th.cat([base, root_gate, inner_gate], dim=1)

    def mode(self) -> th.Tensor:
        base = super().mode()
        if (
            self.root_gate_alpha is None
            or self.root_gate_beta is None
            or self.inner_gate_alpha is None
            or self.inner_gate_beta is None
        ):
            raise RuntimeError("Gate distribution parameters are not initialized.")
        root_gate = self.root_gate_alpha / th.clamp(self.root_gate_alpha + self.root_gate_beta, min=1e-8)
        inner_gate = self.inner_gate_alpha / th.clamp(self.inner_gate_alpha + self.inner_gate_beta, min=1e-8)
        return th.cat([base, root_gate, inner_gate], dim=1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_gate_dist is None or self.inner_gate_dist is None:
            raise RuntimeError("Gate distribution parameters are not initialized.")
        base_log_prob = super().log_prob(actions[:, : 1 + self.stock_dim])
        root_gate = th.clamp(actions[:, 1 + self.stock_dim], min=1e-6, max=1.0 - 1e-6)
        inner_gate = th.clamp(actions[:, 2 + self.stock_dim], min=1e-6, max=1.0 - 1e-6)
        return (
            base_log_prob
            + self.root_gate_dist.log_prob(root_gate)
            + self.inner_gate_dist.log_prob(inner_gate)
        )

    def entropy(self) -> th.Tensor:
        if self.root_gate_dist is None or self.inner_gate_dist is None:
            raise RuntimeError("Gate distribution parameters are not initialized.")
        return super().entropy() + self.root_gate_dist.entropy() + self.inner_gate_dist.entropy()


class RiskCashSectorDirichletTreeDistribution(Distribution):
    """Bounded Beta cash/invested root plus sector and within-sector Dirichlet factors."""

    def __init__(
        self,
        group_indices: list[list[int]],
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        group_alpha_max: float = 100.0,
        leaf_alpha_max: float = 120.0,
    ):
        super().__init__()
        self.group_indices = [list(group) for group in group_indices]
        self.stock_dim = sum(len(group) for group in self.group_indices)
        self.group_dim = len(self.group_indices)
        self.inner_group_indices = [i for i, group in enumerate(self.group_indices) if len(group) > 1]
        self.action_dim = 1 + self.group_dim + sum(len(self.group_indices[i]) for i in self.inner_group_indices)
        self.param_dim = 2 + self.group_dim + sum(len(self.group_indices[i]) for i in self.inner_group_indices)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if not 0.0 <= self.q_min < self.q_max <= 1.0:
            raise ValueError(f"Invalid q bounds: q_min={q_min}, q_max={q_max}")
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.group_alpha_max = float(group_alpha_max)
        self.leaf_alpha_max = float(leaf_alpha_max)
        self.root_dist: th.distributions.Beta | None = None
        self.group_dist: th.distributions.Dirichlet | None = None
        self.inner_dists: list[tuple[int, th.distributions.Dirichlet]] = []
        self.root_alpha: th.Tensor | None = None
        self.root_beta: th.Tensor | None = None
        self.group_alpha: th.Tensor | None = None
        self.inner_alphas: list[tuple[int, th.Tensor]] = []

    @property
    def q_range(self) -> float:
        return self.q_max - self.q_min

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.param_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "RiskCashSectorDirichletTreeDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)
        root_alpha = th.clamp(self.alpha_floor + q_mean_unit * kappa, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(
            self.alpha_floor + (1.0 - q_mean_unit) * kappa,
            min=1e-4,
            max=self.kappa_max + self.alpha_floor,
        )

        offset = 2
        group_raw = raw_params[:, offset : offset + self.group_dim]
        offset += self.group_dim
        group_alpha = th.clamp(
            F.softplus(group_raw) + self.alpha_floor,
            min=1e-4,
            max=self.group_alpha_max,
        )

        self.inner_dists = []
        self.inner_alphas = []
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner_raw = raw_params[:, offset : offset + group_size]
            offset += group_size
            inner_alpha = th.clamp(
                F.softplus(inner_raw) + self.alpha_floor,
                min=1e-4,
                max=self.leaf_alpha_max,
            )
            self.inner_alphas.append((group_idx, inner_alpha))
            self.inner_dists.append((group_idx, th.distributions.Dirichlet(inner_alpha)))

        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.group_alpha = group_alpha
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        self.group_dist = th.distributions.Dirichlet(group_alpha)
        return self

    def _compose_action(self, q_unit: th.Tensor, group: th.Tensor, inners: list[tuple[int, th.Tensor]]) -> th.Tensor:
        q = self.q_min + self.q_range * q_unit
        parts = [q.unsqueeze(-1), group]
        inner_by_group = {group_idx: weights for group_idx, weights in inners}
        for group_idx in self.inner_group_indices:
            parts.append(inner_by_group[group_idx])
        return th.cat(parts, dim=1)

    def sample(self) -> th.Tensor:
        if self.root_dist is None or self.group_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = self.root_dist.sample()
        group = self.group_dist.sample()
        inners = [(group_idx, dist.sample()) for group_idx, dist in self.inner_dists]
        return self._compose_action(q_unit, group, inners)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None or self.group_alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        group = self.group_alpha / th.clamp(self.group_alpha.sum(dim=1, keepdim=True), min=1e-8)
        inners = [
            (group_idx, alpha / th.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8))
            for group_idx, alpha in self.inner_alphas
        ]
        return self._compose_action(q_unit, group, inners)

    def _split_actions(
        self, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, list[tuple[int, th.Tensor]]]:
        q = th.clamp(actions[:, 0], min=self.q_min + 1e-6, max=self.q_max - 1e-6)
        q_unit = th.clamp((q - self.q_min) / self.q_range, min=1e-6, max=1.0 - 1e-6)
        offset = 1
        group = th.clamp(actions[:, offset : offset + self.group_dim], min=1e-8)
        group = group / th.clamp(group.sum(dim=1, keepdim=True), min=1e-8)
        offset += self.group_dim
        inners: list[tuple[int, th.Tensor]] = []
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner = th.clamp(actions[:, offset : offset + group_size], min=1e-8)
            offset += group_size
            inner = inner / th.clamp(inner.sum(dim=1, keepdim=True), min=1e-8)
            inners.append((group_idx, inner))
        return q_unit, group, inners

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None or self.group_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit, group, inners = self._split_actions(actions)
        log_prob = self.root_dist.log_prob(q_unit) - np.log(self.q_range)
        log_prob = log_prob + self.group_dist.log_prob(group)
        inner_by_group = {group_idx: weights for group_idx, weights in inners}
        for group_idx, dist in self.inner_dists:
            log_prob = log_prob + dist.log_prob(inner_by_group[group_idx])
        return log_prob

    def entropy(self) -> th.Tensor:
        if self.root_dist is None or self.group_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        entropy = self.root_dist.entropy() + self.group_dist.entropy()
        for _, dist in self.inner_dists:
            entropy = entropy + dist.entropy()
        return entropy

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class RiskCashGroupLogisticNormalTreeDistribution(RiskCashSectorDirichletTreeDistribution):
    """Cash root Beta + logistic-normal group simplex + within-group Dirichlets.

    Group weights use additive-log-ratio coordinates with the last group as
    reference:

    z_j = log(g_j / g_K), j=1..K-1
    g = softmax([z, 0])
    """

    def __init__(
        self,
        group_indices: list[list[int]],
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        leaf_alpha_max: float = 120.0,
        group_log_std_min: float = -2.5,
        group_log_std_max: float = 0.3,
    ):
        super().__init__(
            group_indices,
            q_min=q_min,
            q_max=q_max,
            alpha_floor=alpha_floor,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            group_alpha_max=1.0,
            leaf_alpha_max=leaf_alpha_max,
        )
        self.group_latent_dim = self.group_dim - 1
        if self.group_latent_dim < 1:
            raise ValueError("Logistic-normal group layer requires at least two groups.")
        self.param_dim = 2 + 2 * self.group_latent_dim + sum(
            len(self.group_indices[i]) for i in self.inner_group_indices
        )
        self.group_log_std_min = float(group_log_std_min)
        self.group_log_std_max = float(group_log_std_max)
        self.group_base_dist: th.distributions.Independent | None = None
        self.group_mu: th.Tensor | None = None
        self.group_log_std: th.Tensor | None = None
        self.group_std: th.Tensor | None = None

    def _alr_inverse(self, z: th.Tensor) -> th.Tensor:
        zeros = th.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return F.softmax(th.cat([z, zeros], dim=1), dim=1)

    def _alr_forward(self, group: th.Tensor) -> th.Tensor:
        group = th.clamp(group, min=1e-8)
        group = group / th.clamp(group.sum(dim=1, keepdim=True), min=1e-8)
        return th.log(group[:, :-1]) - th.log(group[:, -1:])

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.param_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "RiskCashGroupLogisticNormalTreeDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)
        root_alpha = th.clamp(self.alpha_floor + q_mean_unit * kappa, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(
            self.alpha_floor + (1.0 - q_mean_unit) * kappa,
            min=1e-4,
            max=self.kappa_max + self.alpha_floor,
        )

        offset = 2
        group_mu = raw_params[:, offset : offset + self.group_latent_dim]
        offset += self.group_latent_dim
        group_log_std = raw_params[:, offset : offset + self.group_latent_dim]
        offset += self.group_latent_dim
        group_log_std = th.clamp(group_log_std, min=self.group_log_std_min, max=self.group_log_std_max)
        group_std = th.exp(group_log_std)

        self.inner_dists = []
        self.inner_alphas = []
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner_raw = raw_params[:, offset : offset + group_size]
            offset += group_size
            inner_alpha = th.clamp(
                F.softplus(inner_raw) + self.alpha_floor,
                min=1e-4,
                max=self.leaf_alpha_max,
            )
            self.inner_alphas.append((group_idx, inner_alpha))
            self.inner_dists.append((group_idx, th.distributions.Dirichlet(inner_alpha)))

        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.group_mu = group_mu
        self.group_log_std = group_log_std
        self.group_std = group_std
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        self.group_base_dist = th.distributions.Independent(th.distributions.Normal(group_mu, group_std), 1)
        return self

    def sample(self) -> th.Tensor:
        if self.root_dist is None or self.group_base_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = self.root_dist.sample()
        group_z = self.group_base_dist.base_dist.sample()
        group = self._alr_inverse(group_z)
        inners = [(group_idx, dist.sample()) for group_idx, dist in self.inner_dists]
        return self._compose_action(q_unit, group, inners)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None or self.group_mu is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        group = self._alr_inverse(self.group_mu)
        inners = [
            (group_idx, alpha / th.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8))
            for group_idx, alpha in self.inner_alphas
        ]
        return self._compose_action(q_unit, group, inners)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None or self.group_base_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit, group, inners = self._split_actions(actions)
        group_z = self._alr_forward(group)
        log_jacobian = th.sum(th.log(th.clamp(group, min=1e-8)), dim=1)
        log_prob = self.root_dist.log_prob(q_unit) - np.log(self.q_range)
        log_prob = log_prob + self.group_base_dist.log_prob(group_z) - log_jacobian
        inner_by_group = {group_idx: weights for group_idx, weights in inners}
        for group_idx, dist in self.inner_dists:
            log_prob = log_prob + dist.log_prob(inner_by_group[group_idx])
        return log_prob

    def entropy(self) -> th.Tensor:
        if self.root_dist is None or self.group_base_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        entropy = self.root_dist.entropy() + self.group_base_dist.entropy()
        for _, dist in self.inner_dists:
            entropy = entropy + dist.entropy()
        return entropy


class RootSplitBetaDirichletActorCriticPolicy(DirichletActorCriticPolicy):
    """Shared-encoder actor-critic for root cash/invested split."""

    def __init__(
        self,
        *args: Any,
        stock_dim: int,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        risky_alpha_max: float = 100.0,
        **kwargs: Any,
    ):
        self.stock_dim = int(stock_dim)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.risky_alpha_max = float(risky_alpha_max)
        super().__init__(*args, alpha_min=alpha_floor, alpha_max=risky_alpha_max, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RootSplitBetaDirichletActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 1:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+1={self.stock_dim + 1}")

        self.action_dist = RootSplitBetaDirichletDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            stock_dim=self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        return data


class RootSplitBetaDirichletKpActorCriticPolicy(RootSplitBetaDirichletActorCriticPolicy):
    """Root-split policy whose action includes stochastic Kp gate factors."""

    def __init__(
        self,
        *args: Any,
        gate_kappa_min: float = 8.0,
        gate_kappa_max: float = 80.0,
        **kwargs: Any,
    ):
        self.gate_kappa_min = float(gate_kappa_min)
        self.gate_kappa_max = float(gate_kappa_max)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RootSplitBetaDirichletKpActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 3:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+3={self.stock_dim + 3}")

        self.action_dist = RootSplitBetaDirichletKpDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
            gate_kappa_min=self.gate_kappa_min,
            gate_kappa_max=self.gate_kappa_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(gate_kappa_min=self.gate_kappa_min, gate_kappa_max=self.gate_kappa_max)
        return data


class RiskCashSectorDirichletTreeActorCriticPolicy(DirichletActorCriticPolicy):
    """Policy over local cash/invested, sector, and within-sector tree factors."""

    def __init__(
        self,
        *args: Any,
        group_indices: list[list[int]],
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        group_alpha_max: float = 100.0,
        leaf_alpha_max: float = 120.0,
        **kwargs: Any,
    ):
        self.group_indices = [list(group) for group in group_indices]
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.group_alpha_max = float(group_alpha_max)
        self.leaf_alpha_max = float(leaf_alpha_max)
        super().__init__(*args, alpha_min=alpha_floor, alpha_max=max(group_alpha_max, leaf_alpha_max), **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RiskCashSectorDirichletTreeActorCriticPolicy requires a 1-D Box action space.")

        self.action_dist = RiskCashSectorDirichletTreeDistribution(
            self.group_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            group_alpha_max=self.group_alpha_max,
            leaf_alpha_max=self.leaf_alpha_max,
        )
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.action_dist.action_dim:
            raise ValueError(f"action_dim={action_dim} must equal tree factor dim={self.action_dist.action_dim}")

        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        raw_params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_params)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            group_indices=self.group_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            group_alpha_max=self.group_alpha_max,
            leaf_alpha_max=self.leaf_alpha_max,
        )
        return data


class RiskCashGroupLogisticNormalTreeActorCriticPolicy(RiskCashSectorDirichletTreeActorCriticPolicy):
    """Policy with logistic-normal group allocation and Dirichlet leaves."""

    def __init__(
        self,
        *args: Any,
        group_log_std_min: float = -2.5,
        group_log_std_max: float = 0.3,
        **kwargs: Any,
    ):
        self.group_log_std_min = float(group_log_std_min)
        self.group_log_std_max = float(group_log_std_max)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RiskCashGroupLogisticNormalTreeActorCriticPolicy requires a 1-D Box action space.")

        self.action_dist = RiskCashGroupLogisticNormalTreeDistribution(
            self.group_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            leaf_alpha_max=self.leaf_alpha_max,
            group_log_std_min=self.group_log_std_min,
            group_log_std_max=self.group_log_std_max,
        )
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.action_dist.action_dim:
            raise ValueError(f"action_dim={action_dim} must equal tree factor dim={self.action_dist.action_dim}")

        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(group_log_std_min=self.group_log_std_min, group_log_std_max=self.group_log_std_max)
        return data


def _make_mlp(input_dim: int, layer_dims: list[int], activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    modules: list[nn.Module] = []
    last_dim = int(input_dim)
    for dim in layer_dims:
        modules.append(nn.Linear(last_dim, int(dim)))
        modules.append(activation_fn())
        last_dim = int(dim)
    return nn.Sequential(*modules), last_dim


class RootSplitRoutedMlpExtractor(nn.Module):
    """Actor routing extractor for root-risk and risky-allocation branches."""

    def __init__(
        self,
        *,
        features_dim: int,
        stock_dim: int,
        feature_columns: list[str],
        root_feature_names: list[str],
        activation_fn: type[nn.Module],
        root_latent_dim: int = 32,
        risky_latent_dim: int = 32,
        hidden_dim: int = 128,
        vf_arch: list[int] | None = None,
    ):
        super().__init__()
        self.features_dim = int(features_dim)
        self.stock_dim = int(stock_dim)
        self.feature_columns = list(feature_columns)
        self.feature_dim = len(self.feature_columns)
        self.root_feature_names = [name for name in root_feature_names if name in self.feature_columns]
        self.root_feature_indices = [self.feature_columns.index(name) for name in self.root_feature_names]
        self.asset_flat_dim = self.stock_dim * self.feature_dim
        self.prev_weights_dim = self.stock_dim + 1
        self.portfolio_state_dim = 6
        expected_min_dim = self.asset_flat_dim + self.prev_weights_dim + self.portfolio_state_dim
        if self.features_dim < expected_min_dim:
            raise ValueError(f"features_dim={features_dim} is smaller than expected minimum {expected_min_dim}.")
        if not self.root_feature_indices:
            raise ValueError("Routed root split policy requires at least one root feature.")

        root_input_dim = len(self.root_feature_indices) + self.portfolio_state_dim
        risky_input_dim = self.asset_flat_dim + self.stock_dim + root_latent_dim

        self.root_net, root_out_dim = _make_mlp(root_input_dim, [hidden_dim, root_latent_dim], activation_fn)
        self.risky_net, risky_out_dim = _make_mlp(risky_input_dim, [hidden_dim, risky_latent_dim], activation_fn)
        self.value_net, vf_out_dim = _make_mlp(
            self.features_dim,
            vf_arch or [256, 128, 64],
            activation_fn,
        )
        self.latent_dim_pi = root_out_dim + risky_out_dim
        self.latent_dim_vf = vf_out_dim

    def _slices(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        asset_flat = features[:, : self.asset_flat_dim]
        prev_start = self.asset_flat_dim
        prev_end = prev_start + self.prev_weights_dim
        previous_weights = features[:, prev_start:prev_end]
        state_start = prev_end
        state_end = state_start + self.portfolio_state_dim
        portfolio_state = features[:, state_start:state_end]
        return asset_flat, previous_weights, portfolio_state

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        asset_flat, previous_weights, portfolio_state = self._slices(features)
        batch_size = features.shape[0]
        asset_matrix = asset_flat.reshape(batch_size, self.stock_dim, self.feature_dim)
        root_market = asset_matrix[:, 0, self.root_feature_indices]
        root_input = th.cat([root_market, portfolio_state], dim=1)
        z_root = self.root_net(root_input)

        risky_input = th.cat(
            [
                asset_flat,
                previous_weights[:, : self.stock_dim],
                z_root.detach(),
            ],
            dim=1,
        )
        z_risky = self.risky_net(risky_input)
        return th.cat([z_root, z_risky], dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class RootSplitRoutedActionNet(nn.Module):
    """Map routed 64-d actor latent to root/risky distribution parameters."""

    def __init__(self, root_dim: int, risky_dim: int, stock_dim: int):
        super().__init__()
        self.root_dim = int(root_dim)
        self.risky_dim = int(risky_dim)
        self.stock_dim = int(stock_dim)
        self.root_mean = nn.Linear(self.root_dim, 1)
        self.root_kappa = nn.Linear(self.root_dim, 1)
        self.risky_alpha = nn.Linear(self.risky_dim + self.root_dim, self.stock_dim)

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        z_root = latent_pi[:, : self.root_dim]
        z_risky = latent_pi[:, self.root_dim : self.root_dim + self.risky_dim]
        risky_input = th.cat([z_risky, z_root.detach()], dim=1)
        return th.cat([self.root_mean(z_root), self.root_kappa(z_root), self.risky_alpha(risky_input)], dim=1)


class RoutedRootSplitBetaDirichletActorCriticPolicy(RootSplitBetaDirichletActorCriticPolicy):
    """Root-split policy with routed actor encoders.

    The root branch sees only selected risk/market features plus portfolio
    state. The risky branch sees the stock feature panel, previous stock
    weights, and a detached root risk context.
    """

    def __init__(
        self,
        *args: Any,
        feature_columns: list[str],
        root_feature_names: list[str],
        root_latent_dim: int = 32,
        risky_latent_dim: int = 32,
        routed_hidden_dim: int = 128,
        **kwargs: Any,
    ):
        self.feature_columns = list(feature_columns)
        self.root_feature_names = list(root_feature_names)
        self.root_latent_dim = int(root_latent_dim)
        self.risky_latent_dim = int(risky_latent_dim)
        self.routed_hidden_dim = int(routed_hidden_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        vf_arch = [256, 128, 64]
        if isinstance(self.net_arch, dict):
            vf_arch = list(self.net_arch.get("vf", vf_arch))
        self.mlp_extractor = RootSplitRoutedMlpExtractor(
            features_dim=self.features_dim,
            stock_dim=self.stock_dim,
            feature_columns=self.feature_columns,
            root_feature_names=self.root_feature_names,
            activation_fn=self.activation_fn,
            root_latent_dim=self.root_latent_dim,
            risky_latent_dim=self.risky_latent_dim,
            hidden_dim=self.routed_hidden_dim,
            vf_arch=vf_arch,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RoutedRootSplitBetaDirichletActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 1:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+1={self.stock_dim + 1}")

        self.action_dist = RootSplitBetaDirichletDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        self.action_net = RootSplitRoutedActionNet(
            self.root_latent_dim,
            self.risky_latent_dim,
            self.stock_dim,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            feature_columns=self.feature_columns,
            root_feature_names=self.root_feature_names,
            root_latent_dim=self.root_latent_dim,
            risky_latent_dim=self.risky_latent_dim,
            routed_hidden_dim=self.routed_hidden_dim,
        )
        return data


class TwoLevelRootStockMlpExtractor(nn.Module):
    """Two-level actor extractor: compact root branch + shared per-stock scorer.

    The root branch is the portfolio-manager layer. It sees selected market
    features plus portfolio state and outputs a latent used only for the Beta
    cash/risk split.

    The stock branch is the professional-trader layer. The same small scorer is
    applied to each stock with selected stock features, previous stock weight,
    detached root context, and an optional group one-hot context. This keeps the
    stock allocator from becoming a brittle ticker-position memorizer.
    """

    def __init__(
        self,
        *,
        features_dim: int,
        stock_dim: int,
        feature_columns: list[str],
        root_feature_names: list[str],
        stock_feature_names: list[str],
        activation_fn: type[nn.Module],
        root_latent_dim: int = 32,
        root_hidden_dim: int = 64,
        stock_hidden_dim: int = 64,
        stock_group_ids: list[int] | None = None,
        ticker_embedding_dim: int = 0,
        asset_relation_mode: str = "group_one_hot",
        root_raw_window_feature_names: list[str] | None = None,
        root_raw_window_days: int = 0,
        vf_arch: list[int] | None = None,
    ):
        super().__init__()
        self.features_dim = int(features_dim)
        self.stock_dim = int(stock_dim)
        self.feature_columns = list(feature_columns)
        self.feature_dim = len(self.feature_columns)
        self.root_feature_names = [name for name in root_feature_names if name in self.feature_columns]
        self.stock_feature_names = [name for name in stock_feature_names if name in self.feature_columns]
        self.root_feature_indices = [self.feature_columns.index(name) for name in self.root_feature_names]
        self.stock_feature_indices = [self.feature_columns.index(name) for name in self.stock_feature_names]
        self.asset_flat_dim = self.stock_dim * self.feature_dim
        self.prev_weights_dim = self.stock_dim + 1
        self.portfolio_state_dim = 6
        raw_window_names = list(root_raw_window_feature_names or [])
        self.root_raw_window_days = max(0, int(root_raw_window_days))
        if self.root_raw_window_days > 0:
            if not raw_window_names:
                raise ValueError("root_raw_window_feature_names must be non-empty when root_raw_window_days > 0.")
            missing_raw_window = sorted(set(raw_window_names).difference(self.feature_columns))
            if missing_raw_window:
                raise ValueError(f"root_raw_window_feature_names missing from feature_columns: {missing_raw_window}")
        else:
            raw_window_names = []
        self.root_raw_window_feature_names = raw_window_names
        self.root_raw_window_feature_indices = [self.feature_columns.index(name) for name in raw_window_names]
        self.root_raw_window_dim = self.root_raw_window_days * len(self.root_raw_window_feature_indices)

        expected_min_dim = (
            self.asset_flat_dim
            + self.prev_weights_dim
            + self.portfolio_state_dim
            + self.root_raw_window_dim
        )
        if self.features_dim < expected_min_dim:
            raise ValueError(f"features_dim={features_dim} is smaller than expected minimum {expected_min_dim}.")
        if not self.root_feature_indices:
            raise ValueError("Two-level root policy requires at least one valid root feature.")
        if not self.stock_feature_indices:
            raise ValueError("Two-level stock policy requires at least one valid stock feature.")

        if stock_group_ids is None:
            stock_group_ids = [0] * self.stock_dim
        if len(stock_group_ids) != self.stock_dim:
            raise ValueError("stock_group_ids length must match stock_dim.")
        group_ids = [int(x) for x in stock_group_ids]
        if any(group_id < 0 for group_id in group_ids):
            raise ValueError("stock_group_ids must be non-negative.")
        group_count = max(group_ids) + 1 if group_ids else 1
        self.group_count = group_count
        self.asset_relation_mode = str(asset_relation_mode).lower()
        valid_relation_modes = {"none", "group_one_hot", "group_mean_context"}
        if self.asset_relation_mode not in valid_relation_modes:
            raise ValueError(f"asset_relation_mode must be one of {sorted(valid_relation_modes)}.")
        self.ticker_embedding_dim = max(0, int(ticker_embedding_dim))

        stock_group_index = th.as_tensor(group_ids, dtype=th.long)
        self.register_buffer("stock_group_index", stock_group_index)
        group_one_hot = th.zeros(self.stock_dim, group_count, dtype=th.float32)
        for stock_idx, group_idx in enumerate(group_ids):
            group_one_hot[stock_idx, group_idx] = 1.0
        self.register_buffer("group_one_hot", group_one_hot)
        if self.ticker_embedding_dim > 0:
            self.ticker_embedding = nn.Embedding(self.stock_dim, self.ticker_embedding_dim)
        else:
            self.ticker_embedding = None

        root_input_dim = len(self.root_feature_indices) + self.portfolio_state_dim + self.root_raw_window_dim
        relation_dim = 0
        if self.asset_relation_mode in {"group_one_hot", "group_mean_context"}:
            relation_dim += group_count
        if self.asset_relation_mode == "group_mean_context":
            relation_dim += len(self.stock_feature_indices) + 1
        stock_input_dim = (
            len(self.stock_feature_indices)
            + 1
            + root_latent_dim
            + relation_dim
            + self.ticker_embedding_dim
        )

        self.root_net, root_out_dim = _make_mlp(
            root_input_dim,
            [root_hidden_dim, root_latent_dim],
            activation_fn,
        )
        self.stock_scorer, stock_out_dim = _make_mlp(
            stock_input_dim,
            [stock_hidden_dim, stock_hidden_dim],
            activation_fn,
        )
        self.stock_alpha_head = nn.Linear(stock_out_dim, 1)
        self.value_net, vf_out_dim = _make_mlp(
            self.features_dim,
            vf_arch or [256, 128, 64],
            activation_fn,
        )
        self.latent_dim_pi = root_out_dim + self.stock_dim
        self.latent_dim_vf = vf_out_dim
        self.root_latent_dim = root_out_dim

    def _slices(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        asset_flat = features[:, : self.asset_flat_dim]
        prev_start = self.asset_flat_dim
        prev_end = prev_start + self.prev_weights_dim
        previous_weights = features[:, prev_start:prev_end]
        state_start = prev_end
        state_end = state_start + self.portfolio_state_dim
        portfolio_state = features[:, state_start:state_end]
        raw_start = state_end
        raw_end = raw_start + self.root_raw_window_dim
        if self.root_raw_window_dim > 0:
            root_raw_window = features[:, raw_start:raw_end]
        else:
            root_raw_window = features.new_zeros((features.shape[0], 0))
        return asset_flat, previous_weights, portfolio_state, root_raw_window

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        asset_flat, previous_weights, portfolio_state, root_raw_window = self._slices(features)
        batch_size = features.shape[0]
        asset_matrix = asset_flat.reshape(batch_size, self.stock_dim, self.feature_dim)

        root_market = asset_matrix[:, 0, self.root_feature_indices]
        root_input = th.cat([root_market, portfolio_state, root_raw_window], dim=1)
        z_root = self.root_net(root_input)

        stock_features = asset_matrix[:, :, self.stock_feature_indices]
        prev_stock_weights = previous_weights[:, : self.stock_dim].unsqueeze(-1)
        root_context = z_root.detach().unsqueeze(1).expand(-1, self.stock_dim, -1)
        stock_input_parts = [stock_features, prev_stock_weights, root_context]
        if self.asset_relation_mode in {"group_one_hot", "group_mean_context"}:
            group_context = self.group_one_hot.unsqueeze(0).expand(batch_size, -1, -1).to(features.device)
            stock_input_parts.append(group_context)
        if self.asset_relation_mode == "group_mean_context":
            denom = th.clamp(self.group_one_hot.sum(dim=0), min=1.0).to(features.device)
            group_feature_sum = th.einsum("bif,ig->bgf", stock_features, self.group_one_hot.to(features.device))
            group_feature_mean = group_feature_sum / denom.view(1, self.group_count, 1)
            group_prev_sum = th.einsum("bi,ig->bg", previous_weights[:, : self.stock_dim], self.group_one_hot.to(features.device))
            group_prev_mean = (group_prev_sum / denom.view(1, self.group_count)).unsqueeze(-1)
            stock_group_index = self.stock_group_index.to(features.device)
            stock_input_parts.append(group_feature_mean[:, stock_group_index, :])
            stock_input_parts.append(group_prev_mean[:, stock_group_index, :])
        if self.ticker_embedding is not None:
            ticker_ids = th.arange(self.stock_dim, dtype=th.long, device=features.device)
            ticker_context = self.ticker_embedding(ticker_ids).unsqueeze(0).expand(batch_size, -1, -1)
            stock_input_parts.append(ticker_context)
        stock_input = th.cat(stock_input_parts, dim=2)
        stock_hidden = self.stock_scorer(stock_input.reshape(batch_size * self.stock_dim, -1))
        stock_raw = self.stock_alpha_head(stock_hidden).reshape(batch_size, self.stock_dim)
        return th.cat([z_root, stock_raw], dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class TwoLevelRootStockActionNet(nn.Module):
    """Root Beta head plus passthrough per-stock Dirichlet raw scores."""

    def __init__(self, root_dim: int, stock_dim: int):
        super().__init__()
        self.root_dim = int(root_dim)
        self.stock_dim = int(stock_dim)
        self.root_mean = nn.Linear(self.root_dim, 1)
        self.root_kappa = nn.Linear(self.root_dim, 1)

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        z_root = latent_pi[:, : self.root_dim]
        stock_raw = latent_pi[:, self.root_dim : self.root_dim + self.stock_dim]
        return th.cat([self.root_mean(z_root), self.root_kappa(z_root), stock_raw], dim=1)


class TwoLevelRootStockBetaDirichletActorCriticPolicy(RootSplitBetaDirichletActorCriticPolicy):
    """Root cash/risk branch plus shared per-stock stock-allocation branch."""

    def __init__(
        self,
        *args: Any,
        feature_columns: list[str],
        root_feature_names: list[str],
        stock_feature_names: list[str],
        root_latent_dim: int = 32,
        root_hidden_dim: int = 64,
        stock_hidden_dim: int = 64,
        stock_group_ids: list[int] | None = None,
        ticker_embedding_dim: int = 0,
        asset_relation_mode: str = "group_one_hot",
        root_raw_window_feature_names: list[str] | None = None,
        root_raw_window_days: int = 0,
        **kwargs: Any,
    ):
        self.feature_columns = list(feature_columns)
        self.root_feature_names = list(root_feature_names)
        self.stock_feature_names = list(stock_feature_names)
        self.root_latent_dim = int(root_latent_dim)
        self.root_hidden_dim = int(root_hidden_dim)
        self.stock_hidden_dim = int(stock_hidden_dim)
        self.stock_group_ids = list(stock_group_ids) if stock_group_ids is not None else None
        self.ticker_embedding_dim = max(0, int(ticker_embedding_dim))
        self.asset_relation_mode = str(asset_relation_mode).lower()
        self.root_raw_window_feature_names = list(root_raw_window_feature_names or [])
        self.root_raw_window_days = max(0, int(root_raw_window_days))
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        vf_arch = [256, 128, 64]
        if isinstance(self.net_arch, dict):
            vf_arch = list(self.net_arch.get("vf", vf_arch))
        self.mlp_extractor = TwoLevelRootStockMlpExtractor(
            features_dim=self.features_dim,
            stock_dim=self.stock_dim,
            feature_columns=self.feature_columns,
            root_feature_names=self.root_feature_names,
            stock_feature_names=self.stock_feature_names,
            activation_fn=self.activation_fn,
            root_latent_dim=self.root_latent_dim,
            root_hidden_dim=self.root_hidden_dim,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_group_ids=self.stock_group_ids,
            ticker_embedding_dim=self.ticker_embedding_dim,
            asset_relation_mode=self.asset_relation_mode,
            root_raw_window_feature_names=self.root_raw_window_feature_names,
            root_raw_window_days=self.root_raw_window_days,
            vf_arch=vf_arch,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("TwoLevelRootStockBetaDirichletActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 1:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+1={self.stock_dim + 1}")

        self.action_dist = RootSplitBetaDirichletDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        self.action_net = TwoLevelRootStockActionNet(self.root_latent_dim, self.stock_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            feature_columns=self.feature_columns,
            root_feature_names=self.root_feature_names,
            stock_feature_names=self.stock_feature_names,
            root_latent_dim=self.root_latent_dim,
            root_hidden_dim=self.root_hidden_dim,
            stock_hidden_dim=self.stock_hidden_dim,
            stock_group_ids=self.stock_group_ids,
            ticker_embedding_dim=self.ticker_embedding_dim,
            asset_relation_mode=self.asset_relation_mode,
            root_raw_window_feature_names=self.root_raw_window_feature_names,
            root_raw_window_days=self.root_raw_window_days,
        )
        return data
