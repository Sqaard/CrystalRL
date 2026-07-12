"""Train H1: R6c-style portfolio manager plus T5-style trader.

H1 is the first explicit PM/trader candidate:

- PM actor samples full root-split portfolio targets `(q, u_29)`.
- Trader actor executes scheduled PM targets with a synthetic LOB action
  branch `(price_level_i, quantity_level_i)`.
- PM is warm-started from R6c-like teacher traces.
- Trader is pretrained on Wang-style quantity/time subtasks, then on teacher
  targets, before iterative PM/trader PPO fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from torch import nn

from src.ppo.dirichlet_policy import RootSplitBetaDirichletActorCriticPolicy, SharedStockBranchingExecutionPolicy
from src.ppo.execution.helpers import EPS
from src.ppo.pm_trader_env import PMTraderHierarchicalEnv
from src.ppo.stage0_1_train import feature_csv_for_fold
from src.ppo.stage0_1_two_agent_joint_train import as_tensor, compute_gae, resolve, sample_policy
from src.ppo.stage0_1_two_agent_train import load_folds, make_provider, selected_variants
from src.ppo.weight_panel import WeightPanel, load_weight_panel
from src.ppo.synthetic_lob_execution_env import SyntheticLobExecutionCurriculumEnv, SyntheticLobExecutionEnv
from src.ppo.two_agent_env import stock_order_book_proxy_dim


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class H1Batch:
    actor_obs: np.ndarray
    central_obs: np.ndarray
    actions: np.ndarray
    old_log_prob: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def as_policy_batch(self):
        from src.ppo.stage0_1_two_agent_joint_train import PolicyBatch

        return PolicyBatch(
            obs=self.actor_obs,
            actions=self.actions.astype(np.float32),
            old_log_prob=self.old_log_prob,
            values=self.values,
            rewards=self.rewards,
            dones=self.dones,
        )


class CentralizedValueCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], *, learning_rate: float, device: str = "cpu"):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.Tanh()])
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.device = th.device(device)
        self.to(self.device)
        self.optimizer = th.optim.Adam(self.parameters(), lr=float(learning_rate))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs).flatten()

    def value_np(self, obs: np.ndarray) -> float:
        with th.no_grad():
            value = self(as_tensor(obs.reshape(1, -1), device=self.device))
        return float(value.detach().cpu().numpy().reshape(-1)[0])


def central_obs(pm_obs: np.ndarray, trader_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([pm_obs.astype(np.float32), trader_obs.astype(np.float32)]).astype(np.float32)


def critic_update(
    *,
    critic: CentralizedValueCritic,
    central_obs_batch: np.ndarray,
    returns: np.ndarray,
    n_epochs: int,
    batch_size: int,
    max_grad_norm: float,
) -> dict[str, float]:
    if len(central_obs_batch) == 0:
        return {"updates": 0.0}
    obs_t = as_tensor(central_obs_batch, device=critic.device)
    returns_t = as_tensor(returns.astype(np.float32), device=critic.device)
    indices = np.arange(len(central_obs_batch))
    losses: list[float] = []
    critic.train()
    for _ in range(max(1, int(n_epochs))):
        np.random.shuffle(indices)
        for start in range(0, len(indices), max(1, int(batch_size))):
            mb = indices[start : start + max(1, int(batch_size))]
            pred = critic(obs_t[mb])
            loss = th.nn.functional.mse_loss(pred, returns_t[mb])
            critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic.optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"value_loss": float(np.mean(losses)) if losses else 0.0, "updates": float(len(losses))}


def actor_update(
    *,
    policy: Any,
    batch: H1Batch,
    advantages: np.ndarray,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    max_grad_norm: float,
    discrete_actions: bool,
) -> dict[str, float]:
    if len(batch.actor_obs) == 0:
        return {"updates": 0.0}
    device = policy.device
    obs_t = as_tensor(batch.actor_obs, device=device)
    action_dtype = th.long if discrete_actions else th.float32
    actions_t = th.as_tensor(batch.actions, dtype=action_dtype, device=device)
    old_log_t = as_tensor(batch.old_log_prob, device=device)
    adv = advantages.astype(np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
    adv_t = as_tensor(adv, device=device)
    indices = np.arange(len(batch.actor_obs))
    stats: dict[str, list[float]] = {"policy_loss": [], "entropy_loss": [], "approx_kl": [], "clip_fraction": [], "loss": []}
    policy.train()
    for _ in range(max(1, int(n_epochs))):
        np.random.shuffle(indices)
        for start in range(0, len(indices), max(1, int(batch_size))):
            mb = indices[start : start + max(1, int(batch_size))]
            _values, log_prob, entropy = policy.evaluate_actions(obs_t[mb], actions_t[mb])
            ratio = th.exp(log_prob - old_log_t[mb])
            pg_loss_1 = adv_t[mb] * ratio
            pg_loss_2 = adv_t[mb] * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = -th.min(pg_loss_1, pg_loss_2).mean()
            entropy_loss = -log_prob.mean() if entropy is None else -entropy.mean()
            loss = policy_loss + ent_coef * entropy_loss
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy.optimizer.step()
            with th.no_grad():
                log_ratio = log_prob - old_log_t[mb]
                approx_kl = th.mean((th.exp(log_ratio) - 1.0) - log_ratio).detach().cpu().item()
                clipped = th.mean((th.abs(ratio - 1.0) > clip_range).float()).detach().cpu().item()
            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["entropy_loss"].append(float(entropy_loss.detach().cpu()))
            stats["approx_kl"].append(float(approx_kl))
            stats["clip_fraction"].append(float(clipped))
            stats["loss"].append(float(loss.detach().cpu()))
    return {key: float(np.mean(values)) for key, values in stats.items() if values} | {"updates": float(len(stats["loss"]))}


def h1_env_kwargs(config: dict[str, Any], variant: dict[str, Any], panel: WeightPanel, provider: Any) -> dict[str, Any]:
    h1 = variant.get("h1", {})
    pm_state = h1.get("pm_state", {})
    trader_state = h1.get("trader_state", {})
    trader = h1.get("trader", {})
    env_cfg = config.get("environment", {})
    return {
        "pm_feature_names": list(pm_state.get("feature_names", [])),
        "pm_raw_window_feature_names": list(pm_state.get("raw_window_feature_names", [])),
        "pm_raw_window_days": int(pm_state.get("raw_window_days", 0)),
        "trader_stock_feature_names": list(trader_state.get("stock_feature_names", [])),
        "stock_order_book_proxy": dict(trader.get("order_book_proxy", {})),
        "execution_mode": str(trader.get("execution_mode", "synthetic_lob")),
        "root_window_days": int(h1.get("root_window_days", 20)),
        "q_min": float(h1.get("q_min", 0.0)),
        "q_max": float(h1.get("q_max", 0.995)),
        "transaction_cost_pct": float(env_cfg.get("transaction_cost_pct", 0.001)),
        "initial_amount": float(env_cfg.get("initial_amount", 1_000_000.0)),
        "reward_scale": float(env_cfg.get("reward_scale", 100.0)),
        "teacher_provider": provider,
        "sector_map_name": config.get("universe", {}).get("sector_map", "dow30_static"),
        "pm_reward_config": dict(variant.get("h1_reward", {}).get("pm", {})),
        "trader_reward_config": dict(variant.get("h1_reward", {}).get("trader", {})),
        "price_levels": int(trader.get("price_levels", 5)),
        "quantity_levels": int(trader.get("quantity_levels", 5)),
        "forced_cleanup": bool(trader.get("forced_cleanup", True)),
        "low_level_diag_alpha": float(h1.get("low_level_diag_alpha", 0.30)),
        "initial_weights_source": str(h1.get("initial_weights_source", "cash")),
    }


def make_pm_policy(env: PMTraderHierarchicalEnv, variant: dict[str, Any]) -> Any:
    h1 = variant.get("h1", {})
    cfg = h1.get("pm_ppo", {})
    lr = float(cfg.get("learning_rate", 1e-4))
    return RootSplitBetaDirichletActorCriticPolicy(
        env.pm_observation_space,
        env.pm_action_space,
        lambda _: lr,
        stock_dim=env.stock_dim,
        q_min=env.q_min,
        q_max=env.q_max,
        alpha_floor=float(h1.get("alpha_floor", 0.05)),
        kappa_min=float(h1.get("root_kappa_min", 2.0)),
        kappa_max=float(h1.get("root_kappa_max", 80.0)),
        risky_alpha_max=float(h1.get("risky_alpha_max", 100.0)),
        net_arch=h1.get("pm_net_arch", {"pi": [256, 128], "vf": [256, 128]}),
    )


def trader_policy_kwargs(env: PMTraderHierarchicalEnv, variant: dict[str, Any]) -> dict[str, Any]:
    h1 = variant.get("h1", {})
    trader = h1.get("trader", {})
    trader_state = h1.get("trader_state", {})
    order_book_cfg = trader.get("order_book_proxy", {})
    return {
        "stock_dim": env.stock_dim,
        "stock_feature_dim": len(trader_state.get("stock_feature_names", [])),
        "order_book_proxy_dim": stock_order_book_proxy_dim(order_book_cfg),
        "execution_task_dim": env.execution_task_dim,
        "global_context_dim": env.global_context_dim,
        "price_levels": env.price_levels,
        "quantity_levels": env.quantity_levels,
        "stock_hidden_dim": int(trader.get("stock_shared_hidden_dim", 64)),
        "stock_group_ids": env.group_ids,
        "ticker_embedding_dim": int(trader.get("stock_ticker_embedding_dim", 0)),
        "asset_relation_mode": str(trader.get("stock_asset_relation_mode", "group_one_hot")),
        "net_arch": trader.get("net_arch", {"vf": [256, 128]}),
    }


def make_trader_policy(env: PMTraderHierarchicalEnv, variant: dict[str, Any]) -> Any:
    h1 = variant.get("h1", {})
    lr = float(h1.get("trader_ppo", {}).get("learning_rate", 1e-4))
    return SharedStockBranchingExecutionPolicy(
        env.trader_observation_space,
        env.trader_action_space,
        lambda _: lr,
        **trader_policy_kwargs(env, variant),
    )


def make_critics(env: PMTraderHierarchicalEnv, variant: dict[str, Any]) -> tuple[CentralizedValueCritic, CentralizedValueCritic]:
    cfg = variant.get("h1", {}).get("centralized_critic", {})
    input_dim = env.pm_observation_space.shape[0] + env.trader_observation_space.shape[0]
    lr = float(cfg.get("learning_rate", 1e-4))
    pm_critic = CentralizedValueCritic(input_dim, list(cfg.get("pm_net_arch", [512, 256])), learning_rate=lr)
    trader_critic = CentralizedValueCritic(input_dim, list(cfg.get("trader_net_arch", [512, 256])), learning_rate=lr)
    return pm_critic, trader_critic


def pm_bc_dataset(env: PMTraderHierarchicalEnv, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    saved_state = env.state
    saved_diag = env.low_level_diag.copy()
    step = max(1, env.root_window_days)
    for day in range(0, len(env.panel.dates) - 1, step):
        if len(obs_rows) >= max_rows:
            break
        env.state = type(saved_state)(
            day=day,
            previous_weights=env.teacher_provider.weights_for_date(env.panel.dates[day]) if env.teacher_provider else saved_state.previous_weights.copy(),
            portfolio_value=saved_state.portfolio_value,
            peak_value=saved_state.peak_value,
            previous_drawdown=0.0,
            last_turnover=0.0,
        )
        env.low_level_diag = np.zeros_like(saved_diag)
        obs_rows.append(env.pm_obs())
        action_rows.append(env.teacher_pm_action_for_date(env.panel.dates[day]))
    env.state = saved_state
    env.low_level_diag = saved_diag
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32)


def pm_bc_pretrain(
    policy: Any,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    out_path: Path,
    nll_weight: float,
    root_mse_weight: float,
    risky_mse_weight: float,
) -> dict[str, Any]:
    """Behavior-clone the PM with an explicit root-risk objective.

    A plain `-log_prob(q, u)` loss is too weak for the scalar root split when
    the action also contains a 29-dim risky Dirichlet factor. The root decision
    is the whole point of the PM warm start, so H1 uses an additional
    supervised loss on deterministic q plus a small risky-allocation MSE.
    """
    if epochs <= 0:
        return {"enabled": False, "rows": len(obs)}
    device = getattr(policy, "device", "cpu")
    dataset = th.utils.data.TensorDataset(
        th.as_tensor(obs, dtype=th.float32),
        th.as_tensor(actions, dtype=th.float32),
    )
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)
    final: dict[str, float] = {}
    policy.train()
    for _ in range(epochs):
        losses: list[float] = []
        nll_losses: list[float] = []
        q_losses: list[float] = []
        risky_losses: list[float] = []
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            _values, log_prob, _entropy = policy.evaluate_actions(batch_obs, batch_actions)
            pred_actions, _pred_values, _pred_log_prob = policy(batch_obs, deterministic=True)
            q_loss = th.nn.functional.mse_loss(pred_actions[:, 0], batch_actions[:, 0])
            risky_loss = th.nn.functional.mse_loss(pred_actions[:, 1:], batch_actions[:, 1:])
            nll_loss = -log_prob.mean()
            loss = nll_weight * nll_loss + root_mse_weight * q_loss + risky_mse_weight * risky_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            nll_losses.append(float(nll_loss.detach().cpu()))
            q_losses.append(float(q_loss.detach().cpu()))
            risky_losses.append(float(risky_loss.detach().cpu()))
        final = {
            "final_loss": float(np.mean(losses)) if losses else 0.0,
            "final_nll": float(np.mean(nll_losses)) if nll_losses else 0.0,
            "final_q_mse": float(np.mean(q_losses)) if q_losses else 0.0,
            "final_risky_mse": float(np.mean(risky_losses)) if risky_losses else 0.0,
        }
    summary = {
        "enabled": True,
        "rows": int(len(obs)),
        "epochs": int(epochs),
        "nll_weight": float(nll_weight),
        "root_mse_weight": float(root_mse_weight),
        "risky_mse_weight": float(risky_mse_weight),
        **final,
    }
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return summary


def synthetic_env_kwargs(config: dict[str, Any], variant: dict[str, Any], provider: Any) -> dict[str, Any]:
    h1 = variant.get("h1", {})
    trader = h1.get("trader", {})
    trader_state = h1.get("trader_state", {})
    env_cfg = config.get("environment", {})
    return {
        "teacher_provider": provider,
        "stock_feature_names": list(trader_state.get("stock_feature_names", [])),
        "stock_order_book_proxy": dict(trader.get("order_book_proxy", {})),
        "execution_mode": str(trader.get("execution_mode", "synthetic_lob")),
        "sector_map_name": config.get("universe", {}).get("sector_map", "dow30_static"),
        "price_levels": int(trader.get("price_levels", 5)),
        "quantity_levels": int(trader.get("quantity_levels", 5)),
        "transaction_cost_pct": float(env_cfg.get("transaction_cost_pct", 0.001)),
        "reward_scale": float(env_cfg.get("reward_scale", 100.0)),
        "forced_cleanup": bool(trader.get("forced_cleanup", True)),
        "tracking_penalty": float(variant.get("h1_reward", {}).get("trader", {}).get("tracking_penalty", 0.05)),
        "slippage_weight": float(trader.get("slippage_weight", 1.0)),
        "impact_weight": float(trader.get("impact_weight", 1.0)),
    }


def pretrain_trader_policy(
    *,
    train_panel: WeightPanel,
    provider: Any,
    config: dict[str, Any],
    variant: dict[str, Any],
    run_dir: Path,
    smoke_test: bool,
) -> Any | None:
    pre_cfg = variant.get("h1", {}).get("pretrain", {}).get("trader", {})
    trader_cfg = variant.get("h1", {}).get("trader_ppo", {})
    if not bool(pre_cfg.get("enabled", True)):
        return None
    curriculum_steps = int(pre_cfg.get("curriculum_timesteps", 10_000))
    teacher_steps = int(pre_cfg.get("teacher_timesteps", 20_000))
    if smoke_test:
        curriculum_steps = min(curriculum_steps, 512)
        teacher_steps = min(teacher_steps, 512)
    base_kwargs = synthetic_env_kwargs(config, variant, provider)
    curriculum_env = SyntheticLobExecutionCurriculumEnv(
        train_panel,
        **base_kwargs,
        target_quantities=list(pre_cfg.get("target_quantities", [0.01, 0.025, 0.05, 0.10])),
        remaining_time_levels=list(pre_cfg.get("remaining_time_levels", [1, 2, 5, 10])),
        initial_risky_fraction=float(pre_cfg.get("initial_risky_fraction", 0.60)),
        subtask_assets=int(pre_cfg.get("subtask_assets", 3)),
    )
    total_steps = max(curriculum_steps, 1)
    model = PPO(
        SharedStockBranchingExecutionPolicy,
        curriculum_env,
        policy_kwargs={
            "stock_dim": curriculum_env.stock_dim,
            "stock_feature_dim": len(base_kwargs["stock_feature_names"]),
            "order_book_proxy_dim": curriculum_env.proxy_dim,
            "execution_task_dim": curriculum_env.execution_task_dim,
            "global_context_dim": curriculum_env.global_context_dim,
            "price_levels": curriculum_env.price_levels,
            "quantity_levels": curriculum_env.quantity_levels,
            "stock_hidden_dim": int(variant.get("h1", {}).get("trader", {}).get("stock_shared_hidden_dim", 64)),
            "stock_group_ids": curriculum_env.group_ids,
            "ticker_embedding_dim": int(variant.get("h1", {}).get("trader", {}).get("stock_ticker_embedding_dim", 0)),
            "asset_relation_mode": str(variant.get("h1", {}).get("trader", {}).get("stock_asset_relation_mode", "group_one_hot")),
            "net_arch": variant.get("h1", {}).get("trader", {}).get("net_arch", {"vf": [256, 128]}),
        },
        learning_rate=float(trader_cfg.get("learning_rate", 1e-4)),
        n_steps=min(int(trader_cfg.get("n_steps", 1024)), total_steps),
        batch_size=int(trader_cfg.get("batch_size", 256)),
        n_epochs=int(trader_cfg.get("n_epochs", 4)),
        gamma=float(trader_cfg.get("gamma", 0.99)),
        gae_lambda=float(trader_cfg.get("gae_lambda", 0.95)),
        clip_range=float(trader_cfg.get("clip_range", 0.1)),
        ent_coef=float(trader_cfg.get("ent_coef", 0.001)),
        vf_coef=float(trader_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(trader_cfg.get("max_grad_norm", 0.5)),
        verbose=1,
        device="cpu",
    )
    model.set_logger(configure(str(run_dir / "trader_curriculum_pretrain_logs"), ["stdout", "csv"]))
    if curriculum_steps > 0:
        model.learn(total_timesteps=curriculum_steps, progress_bar=False)
    if teacher_steps > 0:
        teacher_env = SyntheticLobExecutionEnv(train_panel, **base_kwargs)
        model.set_env(teacher_env)
        model.set_logger(configure(str(run_dir / "trader_teacher_pretrain_logs"), ["stdout", "csv"]))
        model.learn(total_timesteps=teacher_steps, reset_num_timesteps=False, progress_bar=False)
    model.save(run_dir / "trader_pretrained_model.zip")
    pd.DataFrame(
        [
            {
                "enabled": True,
                "curriculum_timesteps": curriculum_steps,
                "teacher_timesteps": teacher_steps,
                "target_quantities": ";".join(map(str, pre_cfg.get("target_quantities", [0.01, 0.025, 0.05, 0.10]))),
                "remaining_time_levels": ";".join(map(str, pre_cfg.get("remaining_time_levels", [1, 2, 5, 10]))),
            }
        ]
    ).to_csv(run_dir / "trader_pretrain_summary.csv", index=False)
    return model.policy


def collect_episode(
    env: PMTraderHierarchicalEnv,
    pm_policy: Any,
    trader_policy: Any,
    pm_critic: CentralizedValueCritic,
    trader_critic: CentralizedValueCritic,
    *,
    deterministic: bool = False,
) -> tuple[H1Batch, H1Batch, pd.DataFrame]:
    env.reset_hierarchical()
    pm_actor_obs_rows: list[np.ndarray] = []
    pm_central_obs_rows: list[np.ndarray] = []
    pm_actions: list[np.ndarray] = []
    pm_log_probs: list[float] = []
    pm_values: list[float] = []
    pm_rewards: list[float] = []
    pm_dones: list[float] = []

    trader_actor_obs_rows: list[np.ndarray] = []
    trader_central_obs_rows: list[np.ndarray] = []
    trader_actions: list[np.ndarray] = []
    trader_log_probs: list[float] = []
    trader_values: list[float] = []
    trader_rewards: list[float] = []
    trader_dones: list[float] = []

    rows: list[dict[str, Any]] = []
    pm_anchor = env.previous_weights.copy()
    pm_anchor_start_day = 0
    pm_open = False
    open_pm_reward = 0.0

    while not env.done():
        if (not pm_open) or (env.day - pm_anchor_start_day) >= env.root_window_days:
            if pm_open:
                pm_rewards.append(float(open_pm_reward))
                pm_dones.append(0.0)
            pm_obs = env.pm_obs()
            provisional_trader_obs = env.trader_obs(env.previous_weights, remaining_days=env.root_window_days)
            c_obs = central_obs(pm_obs, provisional_trader_obs)
            pm_action, _value, pm_log_prob = sample_policy(pm_policy, pm_obs, deterministic=deterministic)
            pm_anchor = env._weights_from_pm_action(pm_action)
            pm_anchor_start_day = int(env.day)
            pm_open = True
            open_pm_reward = 0.0
            pm_actor_obs_rows.append(pm_obs)
            pm_central_obs_rows.append(c_obs)
            pm_actions.append(pm_action.astype(np.float32))
            pm_log_probs.append(float(pm_log_prob))
            pm_values.append(pm_critic.value_np(c_obs))

        remaining = max(1, env.root_window_days - max(0, env.day - pm_anchor_start_day))
        scheduled = env.previous_weights + (pm_anchor - env.previous_weights) / float(remaining)
        scheduled = scheduled / max(float(np.sum(scheduled)), EPS)
        trader_obs = env.trader_obs(scheduled, remaining_days=remaining)
        t_c_obs = central_obs(env.pm_obs(), trader_obs)
        trader_action, _value, trader_log_prob = sample_policy(trader_policy, trader_obs, deterministic=deterministic)
        info = env.step_hierarchical(pm_anchor_weights=pm_anchor, pm_anchor_start_day=pm_anchor_start_day, trader_action=trader_action)
        done = bool(info.get("terminated", False))
        open_pm_reward += float(info.get("pm_reward", 0.0))

        trader_actor_obs_rows.append(trader_obs)
        trader_central_obs_rows.append(t_c_obs)
        trader_actions.append(trader_action.astype(np.int64))
        trader_log_probs.append(float(trader_log_prob))
        trader_values.append(trader_critic.value_np(t_c_obs))
        trader_rewards.append(float(info.get("trader_reward", 0.0)))
        trader_dones.append(1.0 if done else 0.0)

        row = {
            "date": info.get("date", ""),
            "gross_return": info.get("gross_return", 0.0),
            "net_return": info.get("net_return", 0.0),
            "benchmark_return": info.get("benchmark_return", 0.0),
            "pm_reward": info.get("pm_reward", 0.0),
            "trader_reward": info.get("trader_reward", 0.0),
            "q_anchor": info.get("q_anchor", np.nan),
            "q_scheduled": info.get("q_scheduled", np.nan),
            "q_executed": info.get("q_executed", np.nan),
            "target_cash": float(np.asarray(info["pm_anchor_weights"])[env.cash_index]),
            "scheduled_cash": float(np.asarray(info["scheduled_target_weights"])[env.cash_index]),
            "executed_cash": float(np.asarray(info["executed_weights"])[env.cash_index]),
            "turnover_l1": info.get("turnover_l1", 0.0),
            "stock_turnover_l1": info.get("stock_turnover_l1", 0.0),
            "transaction_cost": info.get("transaction_cost", 0.0),
            "commission_cost": info.get("commission_cost", 0.0),
            "slippage_cost": info.get("slippage_cost", 0.0),
            "tracking_l1": info.get("tracking_l1", 0.0),
            "limit_fill_l1": info.get("limit_fill_l1", 0.0),
            "cleanup_l1": info.get("cleanup_l1", 0.0),
            "fill_prob_mean": info.get("fill_prob_mean", 0.0),
            "price_aggr_mean": info.get("price_aggr_mean", 0.0),
            "qty_frac_mean": info.get("qty_frac_mean", 0.0),
            "drawdown": info.get("drawdown", 0.0),
            "root_remaining_days": info.get("root_remaining_days", np.nan),
            "risky_entropy": info.get("risky_entropy", np.nan),
            "low_diag_tracking": float(env.low_level_diag[0]),
            "low_diag_slippage": float(env.low_level_diag[1]),
            "low_diag_cleanup": float(env.low_level_diag[2]),
            "low_diag_fill_prob": float(env.low_level_diag[3]),
            "low_diag_trader_reward": float(env.low_level_diag[4]),
            "low_diag_active_return": float(env.low_level_diag[5]),
        }
        for ticker, value in zip(env.panel.tickers, np.asarray(info["pm_anchor_weights"])[: env.stock_dim]):
            row[f"pm_anchor_weight_{ticker}"] = float(value)
        row["pm_anchor_weight_CASH"] = float(np.asarray(info["pm_anchor_weights"])[env.cash_index])
        for ticker, value in zip(env.panel.tickers, np.asarray(info["executed_weights"])[: env.stock_dim]):
            row[f"executed_weight_{ticker}"] = float(value)
        row["executed_weight_CASH"] = float(np.asarray(info["executed_weights"])[env.cash_index])
        rows.append(row)

    if pm_open:
        pm_rewards.append(float(open_pm_reward))
        pm_dones.append(1.0)

    pm_batch = H1Batch(
        actor_obs=np.asarray(pm_actor_obs_rows, dtype=np.float32),
        central_obs=np.asarray(pm_central_obs_rows, dtype=np.float32),
        actions=np.asarray(pm_actions, dtype=np.float32),
        old_log_prob=np.asarray(pm_log_probs, dtype=np.float32),
        values=np.asarray(pm_values, dtype=np.float32),
        rewards=np.asarray(pm_rewards, dtype=np.float32),
        dones=np.asarray(pm_dones, dtype=np.float32),
    )
    trader_batch = H1Batch(
        actor_obs=np.asarray(trader_actor_obs_rows, dtype=np.float32),
        central_obs=np.asarray(trader_central_obs_rows, dtype=np.float32),
        actions=np.asarray(trader_actions, dtype=np.int64),
        old_log_prob=np.asarray(trader_log_probs, dtype=np.float32),
        values=np.asarray(trader_values, dtype=np.float32),
        rewards=np.asarray(trader_rewards, dtype=np.float32),
        dones=np.asarray(trader_dones, dtype=np.float32),
    )
    return pm_batch, trader_batch, pd.DataFrame(rows)


def evaluate(
    *,
    env: PMTraderHierarchicalEnv,
    pm_policy: Any,
    trader_policy: Any,
    pm_critic: CentralizedValueCritic,
    trader_critic: CentralizedValueCritic,
    out_dir: Path,
    split: str,
) -> dict[str, Any]:
    _pm_batch, _trader_batch, daily = collect_episode(env, pm_policy, trader_policy, pm_critic, trader_critic, deterministic=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / f"{split}_daily.csv", index=False)
    returns = daily["net_return"].to_numpy(dtype=np.float64) if len(daily) else np.zeros(0)
    cumulative = float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0
    sharpe = float(np.sqrt(252) * returns.mean() / (returns.std(ddof=0) + 1e-12)) if len(returns) else 0.0
    summary = {
        "split": split,
        "days": int(len(daily)),
        "return_pct": cumulative,
        "sharpe": sharpe,
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else 0.0,
        "turnover_l1_mean": float(daily["turnover_l1"].mean()) if len(daily) else 0.0,
        "stock_turnover_l1_mean": float(daily["stock_turnover_l1"].mean()) if len(daily) else 0.0,
        "cash_weight_mean": float(daily["executed_cash"].mean()) if len(daily) else 0.0,
        "tracking_l1_mean": float(daily["tracking_l1"].mean()) if len(daily) else 0.0,
        "slippage_cost_mean": float(daily["slippage_cost"].mean()) if len(daily) else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{split}_summary.csv", index=False)
    return summary


def train_one(config: dict[str, Any], variant: dict[str, Any], fold: pd.Series, *, out_root: Path, smoke_test: bool, force: bool) -> dict[str, Any]:
    fold_id = str(fold["fold"])
    variant_name = str(variant["name"])
    run_dir = out_root / variant_name / fold_id
    summary_path = run_dir / "validation_summary.csv"
    if summary_path.exists() and not force:
        out = pd.read_csv(summary_path).iloc[0].to_dict()
        out.update({"variant": variant_name, "fold": fold_id, "status": "skipped_existing"})
        return out
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_info = feature_csv_for_fold(config, variant, fold, out_root, force=force)
    feature_csv = feature_info["model_ready_csv"]
    train_panel = load_weight_panel(feature_csv, str(fold["train_start"]), str(fold["train_end_inclusive"]))
    validation_panel = load_weight_panel(feature_csv, str(fold["validation_start"]), str(fold["validation_end_inclusive"]))
    provider = make_provider(config, variant, train_panel, fold_id)

    train_env = PMTraderHierarchicalEnv(train_panel, **h1_env_kwargs(config, variant, train_panel, provider))
    validation_provider = make_provider(config, variant, validation_panel, fold_id)
    validation_env = PMTraderHierarchicalEnv(validation_panel, **h1_env_kwargs(config, variant, validation_panel, validation_provider))

    pm_policy = make_pm_policy(train_env, variant)
    pm_critic, trader_critic = make_critics(train_env, variant)

    pm_pre_cfg = variant.get("h1", {}).get("pretrain", {}).get("pm", {})
    pm_bc_summary: dict[str, Any] = {"enabled": False}
    if bool(pm_pre_cfg.get("enabled", True)):
        obs, actions = pm_bc_dataset(train_env, int(pm_pre_cfg.get("max_rows", 8192)))
        pm_bc_summary = pm_bc_pretrain(
            pm_policy,
            obs,
            actions,
            epochs=int(pm_pre_cfg.get("epochs", 100 if not smoke_test else 1)),
            batch_size=int(pm_pre_cfg.get("batch_size", 128)),
            lr=float(pm_pre_cfg.get("learning_rate", 3e-4)),
            out_path=run_dir / "pm_bc_summary.csv",
            nll_weight=float(pm_pre_cfg.get("nll_weight", 0.5)),
            root_mse_weight=float(pm_pre_cfg.get("root_mse_weight", 500.0)),
            risky_mse_weight=float(pm_pre_cfg.get("risky_mse_weight", 0.2)),
        )

    trader_policy = pretrain_trader_policy(
        train_panel=train_panel,
        provider=provider,
        config=config,
        variant=variant,
        run_dir=run_dir,
        smoke_test=smoke_test,
    )
    if trader_policy is None:
        trader_policy = make_trader_policy(train_env, variant)

    it_cfg = variant.get("h1", {}).get("iterative_training", {})
    target_days = int(it_cfg.get("total_internal_trading_days", 70_000))
    if smoke_test:
        target_days = min(target_days, 2048)
    phase_order = list(it_cfg.get("phase_order", ["trader", "pm", "joint"]))
    n_epochs = int(it_cfg.get("n_epochs", 4 if not smoke_test else 1))
    clip_range = float(it_cfg.get("clip_range", 0.1))
    gae_lambda = float(it_cfg.get("gae_lambda", 0.95))
    pm_gamma = float(it_cfg.get("pm_gamma", 0.99))
    trader_gamma = float(it_cfg.get("trader_gamma", 0.99))
    pm_batch_size = int(it_cfg.get("pm_batch_size", 64))
    trader_batch_size = int(it_cfg.get("trader_batch_size", 512))
    max_grad_norm = float(it_cfg.get("max_grad_norm", 0.5))
    pm_ent_coef = float(it_cfg.get("pm_ent_coef", 0.0))
    trader_ent_coef = float(it_cfg.get("trader_ent_coef", 0.001))

    processed_days = 0
    iteration_rows: list[dict[str, Any]] = []
    train_trace_frames: list[pd.DataFrame] = []
    iteration = 0
    while processed_days < target_days:
        for phase in phase_order:
            if processed_days >= target_days:
                break
            iteration += 1
            pm_batch, trader_batch, trace = collect_episode(train_env, pm_policy, trader_policy, pm_critic, trader_critic, deterministic=False)
            pm_adv, pm_returns = compute_gae(pm_batch.as_policy_batch(), gamma=pm_gamma, gae_lambda=gae_lambda)
            trader_adv, trader_returns = compute_gae(trader_batch.as_policy_batch(), gamma=trader_gamma, gae_lambda=gae_lambda)
            update_pm = phase in {"pm", "joint", "both"}
            update_trader = phase in {"trader", "joint", "both"}
            pm_actor_stats = {"updates": 0.0}
            pm_critic_stats = {"updates": 0.0}
            trader_actor_stats = {"updates": 0.0}
            trader_critic_stats = {"updates": 0.0}
            if update_pm:
                pm_actor_stats = actor_update(
                    policy=pm_policy,
                    batch=pm_batch,
                    advantages=pm_adv,
                    n_epochs=n_epochs,
                    batch_size=pm_batch_size,
                    clip_range=clip_range,
                    ent_coef=pm_ent_coef,
                    max_grad_norm=max_grad_norm,
                    discrete_actions=False,
                )
                pm_critic_stats = critic_update(
                    critic=pm_critic,
                    central_obs_batch=pm_batch.central_obs,
                    returns=pm_returns,
                    n_epochs=n_epochs,
                    batch_size=pm_batch_size,
                    max_grad_norm=max_grad_norm,
                )
            if update_trader:
                trader_actor_stats = actor_update(
                    policy=trader_policy,
                    batch=trader_batch,
                    advantages=trader_adv,
                    n_epochs=n_epochs,
                    batch_size=trader_batch_size,
                    clip_range=clip_range,
                    ent_coef=trader_ent_coef,
                    max_grad_norm=max_grad_norm,
                    discrete_actions=True,
                )
                trader_critic_stats = critic_update(
                    critic=trader_critic,
                    central_obs_batch=trader_batch.central_obs,
                    returns=trader_returns,
                    n_epochs=n_epochs,
                    batch_size=trader_batch_size,
                    max_grad_norm=max_grad_norm,
                )
            processed_days += len(trader_batch.rewards)
            returns = trace["net_return"].to_numpy(dtype=np.float64) if len(trace) else np.zeros(0)
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "phase": phase,
                    "processed_internal_trading_days": processed_days,
                    "episode_days": int(len(trace)),
                    "episode_return_pct": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
                    "episode_cash_mean": float(trace["executed_cash"].mean()) if len(trace) else 0.0,
                    "episode_tracking_l1_mean": float(trace["tracking_l1"].mean()) if len(trace) else 0.0,
                    "episode_slippage_cost_mean": float(trace["slippage_cost"].mean()) if len(trace) else 0.0,
                    "pm_actor_updates": pm_actor_stats.get("updates", 0.0),
                    "trader_actor_updates": trader_actor_stats.get("updates", 0.0),
                    "pm_actor_kl": pm_actor_stats.get("approx_kl", np.nan),
                    "trader_actor_kl": trader_actor_stats.get("approx_kl", np.nan),
                    "pm_critic_loss": pm_critic_stats.get("value_loss", np.nan),
                    "trader_critic_loss": trader_critic_stats.get("value_loss", np.nan),
                }
            )
            print(
                "[H1-ITER] "
                f"iteration={iteration} phase={phase} "
                f"processed_internal_days={processed_days}/{target_days} "
                f"episode_days={len(trace)} "
                f"episode_return={float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0:.6f} "
                f"cash_mean={float(trace['executed_cash'].mean()) if len(trace) else 0.0:.4f} "
                f"tracking_mean={float(trace['tracking_l1'].mean()) if len(trace) else 0.0:.6f} "
                f"slippage_mean={float(trace['slippage_cost'].mean()) if len(trace) else 0.0:.6f}",
                flush=True,
            )
            save_trace_episodes = int(it_cfg.get("save_train_trace_episodes", 1))
            save_trace_mode = str(it_cfg.get("save_train_trace_mode", "tail")).lower()
            if save_trace_episodes != 0:
                frame = trace.copy()
                frame["iteration"] = iteration
                frame["phase"] = phase
                if save_trace_mode == "all" or save_trace_episodes < 0:
                    train_trace_frames.append(frame)
                elif save_trace_mode == "head":
                    if len(train_trace_frames) < save_trace_episodes:
                        train_trace_frames.append(frame)
                else:
                    train_trace_frames.append(frame)
                    if len(train_trace_frames) > save_trace_episodes:
                        train_trace_frames.pop(0)

    pd.DataFrame(iteration_rows).to_csv(run_dir / "h1_iteration_summary.csv", index=False)
    if train_trace_frames:
        pd.concat(train_trace_frames, ignore_index=True).to_csv(run_dir / "train_trace_daily.csv", index=False)
    th.save(pm_policy.state_dict(), run_dir / "pm_policy.pt")
    th.save(trader_policy.state_dict(), run_dir / "trader_policy.pt")
    th.save(pm_critic.state_dict(), run_dir / "pm_central_critic.pt")
    th.save(trader_critic.state_dict(), run_dir / "trader_central_critic.pt")

    validation_summary = evaluate(
        env=validation_env,
        pm_policy=pm_policy,
        trader_policy=trader_policy,
        pm_critic=pm_critic,
        trader_critic=trader_critic,
        out_dir=run_dir,
        split="validation",
    )
    metadata = {
        "variant": variant,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "pm_bc_summary": pm_bc_summary,
        "target_internal_trading_days": target_days,
        "observed_internal_trading_days": processed_days,
        "validation_summary": validation_summary,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"variant": variant_name, "fold": fold_id, "status": "trained", **validation_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated/stage0_1_h1_pm_trader.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or config.get("output", {}).get("run_name", "weight_based_h1_pm_trader")
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== H1 PM/trader: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nH1 run written to {out_root}")


if __name__ == "__main__":
    main()
