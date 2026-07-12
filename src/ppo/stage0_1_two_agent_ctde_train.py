"""Train Stage 0.1 T4 with CTDE: centralized critics, decentralized actors.

Actors:

- root actor sees only root observation and samples macro `q_anchor`;
- stock actor sees only stock/private observation and samples daily risky simplex.

Critics:

- root centralized critic sees root + stock context at macro decision points;
- stock centralized critic sees root + stock context at daily decision points.

The critics provide the values used for GAE/PPO advantages. Actor local value
heads are intentionally ignored in this trainer.
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
from torch import nn

from src.ppo.stage0_1_train import feature_csv_for_fold
from src.ppo.stage0_1_two_agent_joint_train import (
    PolicyBatch,
    as_tensor,
    compute_gae,
    make_policies,
    resolve,
    sample_policy,
)
from src.ppo.stage0_1_two_agent_train import (
    bc_dataset_root,
    bc_dataset_stock,
    bc_pretrain,
    env_kwargs,
    load_folds,
    make_provider,
    selected_variants,
)
from src.ppo.weight_panel import WeightPanel, load_weight_panel
from src.ppo.two_agent_env import JointTwoAgentEnv, StockAllocationEnv


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


@dataclass
class CTDEBatch:
    actor_obs: np.ndarray
    central_obs: np.ndarray
    actions: np.ndarray
    old_log_prob: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def as_policy_batch(self) -> PolicyBatch:
        return PolicyBatch(
            obs=self.actor_obs,
            actions=self.actions,
            old_log_prob=self.old_log_prob,
            values=self.values,
            rewards=self.rewards,
            dones=self.dones,
        )


def central_obs(root_obs: np.ndarray, stock_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([root_obs.astype(np.float32), stock_obs.astype(np.float32)]).astype(np.float32)


def actor_update(
    *,
    policy: Any,
    batch: CTDEBatch,
    advantages: np.ndarray,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    if len(batch.actor_obs) == 0:
        return {"updates": 0.0}
    device = policy.device
    obs_t = as_tensor(batch.actor_obs, device=device)
    actions_t = as_tensor(batch.actions, device=device)
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
            _local_values, log_prob, entropy = policy.evaluate_actions(obs_t[mb], actions_t[mb])
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


def make_critics(env: JointTwoAgentEnv, variant: dict[str, Any]) -> tuple[CentralizedValueCritic, CentralizedValueCritic]:
    cfg = variant.get("two_agent", {}).get("centralized_critic", {})
    root_hidden = list(cfg.get("root_net_arch", [256, 128]))
    stock_hidden = list(cfg.get("stock_net_arch", [512, 256]))
    lr = float(cfg.get("learning_rate", 1e-4))
    input_dim = env.root_observation_space.shape[0] + env.stock_observation_space.shape[0]
    root_critic = CentralizedValueCritic(input_dim, root_hidden, learning_rate=lr)
    stock_critic = CentralizedValueCritic(input_dim, stock_hidden, learning_rate=lr)
    return root_critic, stock_critic


def collect_episode_ctde(
    env: JointTwoAgentEnv,
    root_policy: Any,
    stock_policy: Any,
    root_critic: CentralizedValueCritic,
    stock_critic: CentralizedValueCritic,
) -> tuple[CTDEBatch, CTDEBatch, pd.DataFrame]:
    env.reset_joint()
    root_actor_obs_rows: list[np.ndarray] = []
    root_central_obs_rows: list[np.ndarray] = []
    root_actions: list[np.ndarray] = []
    root_log_probs: list[float] = []
    root_values: list[float] = []
    root_rewards: list[float] = []
    root_dones: list[float] = []

    stock_actor_obs_rows: list[np.ndarray] = []
    stock_central_obs_rows: list[np.ndarray] = []
    stock_actions: list[np.ndarray] = []
    stock_log_probs: list[float] = []
    stock_values: list[float] = []
    stock_rewards: list[float] = []
    stock_dones: list[float] = []

    q_anchor = 0.0
    root_anchor_start_day = 0
    open_root_reward = 0.0
    root_open = False
    daily_rows: list[dict[str, Any]] = []

    while not env._done():
        if (not root_open) or (env.day - root_anchor_start_day) >= env.root_window_days:
            if root_open:
                root_rewards.append(float(open_root_reward))
                root_dones.append(0.0)
            root_obs = env.root_obs()
            stock_context_obs = env.stock_obs(q_anchor=None, remaining_days=env.root_window_days)
            c_obs = central_obs(root_obs, stock_context_obs)
            root_action, _local_value, root_log_prob = sample_policy(root_policy, root_obs, deterministic=False)
            q_anchor = float(np.clip(root_action[0], env.q_min, env.q_max))
            root_anchor_start_day = int(env.day)
            open_root_reward = 0.0
            root_open = True
            root_actor_obs_rows.append(root_obs)
            root_central_obs_rows.append(c_obs)
            root_actions.append(np.array([q_anchor], dtype=np.float32))
            root_log_probs.append(root_log_prob)
            root_values.append(root_critic.value_np(c_obs))

        remaining = env.root_remaining_days(root_anchor_start_day)
        root_context_obs = env.root_obs()
        stock_obs = env.stock_obs(q_anchor=q_anchor, remaining_days=remaining)
        s_c_obs = central_obs(root_context_obs, stock_obs)
        stock_action, _local_value, stock_log_prob = sample_policy(stock_policy, stock_obs, deterministic=False)
        info = env.step_joint(q_anchor=q_anchor, root_anchor_start_day=root_anchor_start_day, stock_action=stock_action)
        done = env._done()
        open_root_reward += float(info["root_reward"])

        stock_actor_obs_rows.append(stock_obs)
        stock_central_obs_rows.append(s_c_obs)
        stock_actions.append(stock_action.astype(np.float32))
        stock_log_probs.append(stock_log_prob)
        stock_values.append(stock_critic.value_np(s_c_obs))
        stock_rewards.append(float(info["stock_reward"]))
        stock_dones.append(1.0 if done else 0.0)

        target = np.asarray(info.get("target_weights", np.zeros(env.asset_dim)), dtype=np.float64)
        row = {
            "date": info.get("date", ""),
            "net_return": info.get("net_return", 0.0),
            "benchmark_return": info.get("benchmark_return", 0.0),
            "root_reward": info.get("root_reward", 0.0),
            "stock_reward": info.get("stock_reward", 0.0),
            "q_anchor": q_anchor,
            "q_scheduled": info.get("q_scheduled", np.nan),
            "cash_weight": float(target[env.cash_index]),
            "turnover_l1": info.get("turnover_l1", 0.0),
            "stock_turnover_l1": info.get("stock_turnover_l1", 0.0),
            "transaction_cost": info.get("transaction_cost", 0.0),
            "drawdown": info.get("drawdown", 0.0),
            "risky_entropy": info.get("risky_entropy", np.nan),
            "root_remaining_days": info.get("root_remaining_days", np.nan),
            "low_diag_stock_turnover": float(env.low_level_diag[0]),
            "low_diag_stock_entropy": float(env.low_level_diag[3]),
            "low_diag_stock_alpha": float(env.low_level_diag[5]),
        }
        for ticker, value in zip(env.panel.tickers, target[: env.stock_dim]):
            row[f"target_weight_{ticker}"] = float(value)
        row["target_weight_CASH"] = float(target[env.cash_index])
        daily_rows.append(row)

    if root_open:
        root_rewards.append(float(open_root_reward))
        root_dones.append(1.0)

    root_batch = CTDEBatch(
        actor_obs=np.asarray(root_actor_obs_rows, dtype=np.float32),
        central_obs=np.asarray(root_central_obs_rows, dtype=np.float32),
        actions=np.asarray(root_actions, dtype=np.float32),
        old_log_prob=np.asarray(root_log_probs, dtype=np.float32),
        values=np.asarray(root_values, dtype=np.float32),
        rewards=np.asarray(root_rewards, dtype=np.float32),
        dones=np.asarray(root_dones, dtype=np.float32),
    )
    stock_batch = CTDEBatch(
        actor_obs=np.asarray(stock_actor_obs_rows, dtype=np.float32),
        central_obs=np.asarray(stock_central_obs_rows, dtype=np.float32),
        actions=np.asarray(stock_actions, dtype=np.float32),
        old_log_prob=np.asarray(stock_log_probs, dtype=np.float32),
        values=np.asarray(stock_values, dtype=np.float32),
        rewards=np.asarray(stock_rewards, dtype=np.float32),
        dones=np.asarray(stock_dones, dtype=np.float32),
    )
    return root_batch, stock_batch, pd.DataFrame(daily_rows)


def evaluate_policies(
    *,
    root_policy: Any,
    stock_policy: Any,
    panel: WeightPanel,
    config: dict[str, Any],
    variant: dict[str, Any],
    provider: Any,
    out_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    env = StockAllocationEnv(panel, frozen_root_model=root_policy, **env_kwargs(config, variant, panel, provider))
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action, _ = stock_policy.predict(obs, deterministic=True)
        obs, reward, done, _truncated, info = env.step(action)
        target = np.asarray(info.get("target_weights", np.zeros(env.asset_dim)), dtype=np.float64)
        row = {
            "date": info.get("date", ""),
            "net_return": info.get("net_return", 0.0),
            "gross_return": info.get("gross_return", 0.0),
            "reward": reward,
            "stock_reward": info.get("stock_reward", 0.0),
            "q_anchor": info.get("q_anchor", np.nan),
            "q_scheduled": info.get("q_scheduled", np.nan),
            "cash_weight": float(target[env.cash_index]),
            "turnover_l1": info.get("turnover_l1", 0.0),
            "stock_turnover_l1": info.get("stock_turnover_l1", 0.0),
            "transaction_cost": info.get("transaction_cost", 0.0),
            "drawdown": info.get("drawdown", 0.0),
            "risky_entropy": info.get("risky_entropy", np.nan),
        }
        for ticker, value in zip(panel.tickers, target[: env.stock_dim]):
            row[f"target_weight_{ticker}"] = float(value)
        row["target_weight_CASH"] = float(target[env.cash_index])
        rows.append(row)
    daily = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / f"{split_name}_daily.csv", index=False)
    returns = daily["net_return"].to_numpy(dtype=np.float64) if len(daily) else np.zeros(0)
    cumulative = float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0
    sharpe = float(np.sqrt(252) * returns.mean() / (returns.std(ddof=0) + 1e-12)) if len(returns) else 0.0
    summary = {
        "split": split_name,
        "days": int(len(daily)),
        "return_pct": cumulative,
        "sharpe": sharpe,
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else 0.0,
        "turnover_l1_mean": float(daily["turnover_l1"].mean()) if len(daily) else 0.0,
        "cash_weight_mean": float(daily["cash_weight"].mean()) if len(daily) else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{split_name}_summary.csv", index=False)
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
    joint_env = JointTwoAgentEnv(train_panel, **env_kwargs(config, variant, train_panel, provider))
    root_policy, stock_policy = make_policies(joint_env, variant)
    root_critic, stock_critic = make_critics(joint_env, variant)

    bc_cfg = variant.get("two_agent", {}).get("bc_pretrain", {})
    if bool(bc_cfg.get("enabled", True)):
        root_obs, root_actions = bc_dataset_root(joint_env, int(bc_cfg.get("max_rows", 8192)))
        stock_obs, stock_actions = bc_dataset_stock(joint_env, int(bc_cfg.get("max_rows", 8192)))
        bc_pretrain(
            root_policy,
            root_obs,
            root_actions,
            epochs=int(bc_cfg.get("root_epochs", 2 if not smoke_test else 1)),
            batch_size=int(bc_cfg.get("batch_size", 256)),
            lr=float(bc_cfg.get("learning_rate", 3e-4)),
            out_path=run_dir / "root_bc_summary.csv",
        )
        bc_pretrain(
            stock_policy,
            stock_obs,
            stock_actions,
            epochs=int(bc_cfg.get("stock_epochs", 2 if not smoke_test else 1)),
            batch_size=int(bc_cfg.get("batch_size", 256)),
            lr=float(bc_cfg.get("learning_rate", 3e-4)),
            out_path=run_dir / "stock_bc_summary.csv",
        )

    joint_cfg = variant.get("two_agent", {}).get("joint_ppo", {})
    target_days = int(joint_cfg.get("total_internal_trading_days", 70_000))
    if smoke_test:
        target_days = min(target_days, 2048)
    n_epochs = int(joint_cfg.get("n_epochs", 4 if not smoke_test else 1))
    clip_range = float(joint_cfg.get("clip_range", config.get("ppo", {}).get("clip_range", 0.1)))
    gamma_stock = float(joint_cfg.get("stock_gamma", config.get("ppo", {}).get("gamma", 0.99)))
    gamma_root = float(joint_cfg.get("root_gamma", 0.99))
    gae_lambda = float(joint_cfg.get("gae_lambda", config.get("ppo", {}).get("gae_lambda", 0.95)))
    max_grad_norm = float(joint_cfg.get("max_grad_norm", config.get("ppo", {}).get("max_grad_norm", 0.5)))
    root_batch_size = int(joint_cfg.get("root_batch_size", 64))
    stock_batch_size = int(joint_cfg.get("stock_batch_size", 512))
    root_ent_coef = float(joint_cfg.get("root_ent_coef", 0.0))
    stock_ent_coef = float(joint_cfg.get("stock_ent_coef", 0.0))

    processed_days = 0
    iteration = 0
    episode_rows: list[dict[str, Any]] = []
    train_trace_frames: list[pd.DataFrame] = []
    while processed_days < target_days:
        iteration += 1
        root_batch, stock_batch, trace = collect_episode_ctde(joint_env, root_policy, stock_policy, root_critic, stock_critic)
        root_adv, root_returns = compute_gae(root_batch.as_policy_batch(), gamma=gamma_root, gae_lambda=gae_lambda)
        stock_adv, stock_returns = compute_gae(stock_batch.as_policy_batch(), gamma=gamma_stock, gae_lambda=gae_lambda)
        root_actor_stats = actor_update(
            policy=root_policy,
            batch=root_batch,
            advantages=root_adv,
            n_epochs=n_epochs,
            batch_size=root_batch_size,
            clip_range=clip_range,
            ent_coef=root_ent_coef,
            max_grad_norm=max_grad_norm,
        )
        stock_actor_stats = actor_update(
            policy=stock_policy,
            batch=stock_batch,
            advantages=stock_adv,
            n_epochs=n_epochs,
            batch_size=stock_batch_size,
            clip_range=clip_range,
            ent_coef=stock_ent_coef,
            max_grad_norm=max_grad_norm,
        )
        root_critic_stats = critic_update(
            critic=root_critic,
            central_obs_batch=root_batch.central_obs,
            returns=root_returns,
            n_epochs=n_epochs,
            batch_size=root_batch_size,
            max_grad_norm=max_grad_norm,
        )
        stock_critic_stats = critic_update(
            critic=stock_critic,
            central_obs_batch=stock_batch.central_obs,
            returns=stock_returns,
            n_epochs=n_epochs,
            batch_size=stock_batch_size,
            max_grad_norm=max_grad_norm,
        )
        processed_days += len(stock_batch.rewards)
        returns = trace["net_return"].to_numpy(dtype=np.float64) if len(trace) else np.zeros(0)
        episode_rows.append(
            {
                "iteration": iteration,
                "processed_internal_trading_days": processed_days,
                "episode_days": int(len(trace)),
                "episode_return_pct": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
                "episode_cash_mean": float(trace["cash_weight"].mean()) if len(trace) else 0.0,
                "episode_turnover_l1_mean": float(trace["turnover_l1"].mean()) if len(trace) else 0.0,
                "root_transitions": int(len(root_batch.rewards)),
                "stock_transitions": int(len(stock_batch.rewards)),
                **{f"root_actor_{k}": v for k, v in root_actor_stats.items()},
                **{f"stock_actor_{k}": v for k, v in stock_actor_stats.items()},
                **{f"root_central_critic_{k}": v for k, v in root_critic_stats.items()},
                **{f"stock_central_critic_{k}": v for k, v in stock_critic_stats.items()},
            }
        )
        if iteration <= int(joint_cfg.get("save_train_trace_episodes", 1)):
            trace = trace.copy()
            trace["iteration"] = iteration
            train_trace_frames.append(trace)
        print(
            f"[T4-CTDE] iter={iteration} days={processed_days}/{target_days} "
            f"root_kl={root_actor_stats.get('approx_kl', np.nan):.5f} "
            f"stock_kl={stock_actor_stats.get('approx_kl', np.nan):.5f} "
            f"root_v={root_critic_stats.get('value_loss', np.nan):.4g} "
            f"stock_v={stock_critic_stats.get('value_loss', np.nan):.4g}",
            flush=True,
        )

    pd.DataFrame(episode_rows).to_csv(run_dir / "train_ctde_iterations.csv", index=False)
    if train_trace_frames:
        pd.concat(train_trace_frames, ignore_index=True).to_csv(run_dir / "train_trace_daily.csv", index=False)
    th.save({"state_dict": root_policy.state_dict(), "variant": variant}, run_dir / "root_actor_policy.pt")
    th.save({"state_dict": stock_policy.state_dict(), "variant": variant}, run_dir / "stock_actor_policy.pt")
    th.save({"state_dict": root_critic.state_dict(), "variant": variant}, run_dir / "root_central_critic.pt")
    th.save({"state_dict": stock_critic.state_dict(), "variant": variant}, run_dir / "stock_central_critic.pt")

    validation_provider = make_provider(config, variant, validation_panel, fold_id)
    validation_summary = evaluate_policies(
        root_policy=root_policy,
        stock_policy=stock_policy,
        panel=validation_panel,
        config=config,
        variant=variant,
        provider=validation_provider,
        out_dir=run_dir,
        split_name="validation",
    )
    metadata = {
        "variant": variant,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "target_internal_trading_days": target_days,
        "observed_internal_trading_days": processed_days,
        "validation_summary": validation_summary,
        "ctde": {
            "root_actor_obs_dim": int(joint_env.root_observation_space.shape[0]),
            "stock_actor_obs_dim": int(joint_env.stock_observation_space.shape[0]),
            "central_obs_dim": int(joint_env.root_observation_space.shape[0] + joint_env.stock_observation_space.shape[0]),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    validation_summary.update(
        {
            "variant": variant_name,
            "fold": fold_id,
            "status": "trained",
            "observed_internal_trading_days": processed_days,
            "target_internal_trading_days": target_days,
        }
    )
    return validation_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated/stage0_1_t4_ctde_two_agent.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or config.get("output", {}).get("run_name", "weight_based_t4_ctde_two_agent")
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== T4 CTDE two-agent: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nT4 CTDE run written to {out_root}")


if __name__ == "__main__":
    main()
