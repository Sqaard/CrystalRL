"""Deprecated train runner for W2 off-policy Q-actor-critic PM/Trader candidate.

W2 is deliberately separate from PPO-based W1/T/H variants.  Each level has
its own deterministic actor, replay buffer, action-value critic, target actor,
and target critic.  The TD target is:

```text
r + gamma * Q_target(s_next, actor_target(s_next))
```

For continuous actions this is the standard practical approximation of
`r + gamma * max_a Q(s_next, a)`: the actor is trained to maximize Q.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml
from torch.nn import functional as F

from src.ppo.deprecated.offpolicy_q import (
    ActionValueCritic,
    DeterministicPMActor,
    DeterministicTraderActor,
    ReplayBuffer,
    QUpdateStats,
    soft_update,
)
from src.ppo.stage0_1_train import feature_csv_for_fold, resolve_variant_inheritance
from src.ppo.weight_panel import load_weight_panel
from src.ppo.stage0_1_w1_budget_trader_train import _daily_row, env_kwargs, load_folds, selected_variants
from src.ppo.w1_budget_trader_env import W1BudgetTraderEnv


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def as_tensor(array: np.ndarray, *, device: th.device | str) -> th.Tensor:
    return th.as_tensor(array, dtype=th.float32, device=device)


def horizon_from_unit(env: W1BudgetTraderEnv, horizon_unit: float) -> int:
    idx = int(np.clip(round(float(horizon_unit) * (len(env.horizon_choices) - 1)), 0, len(env.horizon_choices) - 1))
    return int(env.horizon_choices[idx])


def noisy_pm_action(actor: DeterministicPMActor, obs: np.ndarray, *, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    with th.no_grad():
        action = actor(as_tensor(obs.reshape(1, -1), device=actor.device)).cpu().numpy().reshape(-1)
    if noise_std > 0:
        action += rng.normal(0.0, noise_std, size=action.shape)
    action[0] = np.clip(action[0], actor.q_min, actor.q_max)
    action[1] = np.clip(action[1], 0.0, 1.0)
    return action.astype(np.float32)


def noisy_trader_action(actor: DeterministicTraderActor, obs: np.ndarray, *, rng: np.random.Generator, mix_prob: float) -> np.ndarray:
    with th.no_grad():
        action = actor(as_tensor(obs.reshape(1, -1), device=actor.device)).cpu().numpy().reshape(-1)
    if mix_prob > 0:
        random_simplex = rng.dirichlet(np.ones(actor.asset_dim, dtype=np.float64))
        action = (1.0 - mix_prob) * action + mix_prob * random_simplex
    action = np.maximum(action, 1e-8)
    action = action / np.sum(action)
    return action.astype(np.float32)


def q_update(
    *,
    actor: th.nn.Module,
    actor_target: th.nn.Module,
    critic: ActionValueCritic,
    critic_target: ActionValueCritic,
    actor_optimizer: th.optim.Optimizer,
    critic_optimizer: th.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    max_grad_norm: float,
) -> QUpdateStats | None:
    if len(replay) < batch_size:
        return None
    batch = replay.sample(batch_size, device=critic.device)
    with th.no_grad():
        next_action = actor_target(batch["next_obs"])
        target_q = critic_target(batch["next_obs"], next_action)
        td_target = batch["rewards"] + float(gamma) * (1.0 - batch["dones"]) * target_q

    q_pred = critic(batch["obs"], batch["actions"])
    critic_loss = F.mse_loss(q_pred, td_target)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    th.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_optimizer.step()

    actor_action = actor(batch["obs"])
    actor_loss = -critic(batch["obs"], actor_action).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    th.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_optimizer.step()

    soft_update(actor, actor_target, tau)
    soft_update(critic, critic_target, tau)
    return QUpdateStats(
        critic_loss=float(critic_loss.detach().cpu()),
        actor_loss=float(actor_loss.detach().cpu()),
        q_mean=float(q_pred.detach().mean().cpu()),
        target_q_mean=float(td_target.detach().mean().cpu()),
    )


def make_w2_models(env: W1BudgetTraderEnv, variant: dict[str, Any]):
    cfg = variant.get("w2_qcritic", {})
    pm_cfg = cfg.get("pm_actor", {})
    trader_cfg = cfg.get("trader_actor", {})
    critic_cfg = cfg.get("critic", {})
    device = str(cfg.get("device", "cpu"))
    pm_actor = DeterministicPMActor(
        env.pm_obs_dim,
        hidden_dims=list(pm_cfg.get("hidden_dims", [256, 128])),
        q_min=env.q_min,
        q_max=env.q_max,
        device=device,
    )
    trader_actor = DeterministicTraderActor(
        env.trader_obs_dim,
        stock_dim=env.stock_dim,
        stock_feature_dim=env.stock_feature_dim,
        task_dim=env.trader_task_dim,
        stock_hidden_dim=int(trader_cfg.get("stock_hidden_dim", 64)),
        context_hidden_dims=list(trader_cfg.get("context_hidden_dims", [128])),
        device=device,
    )
    pm_critic = ActionValueCritic(
        env.pm_obs_dim,
        2,
        hidden_dims=list(critic_cfg.get("pm_hidden_dims", [256, 256])),
        device=device,
    )
    trader_critic = ActionValueCritic(
        env.trader_obs_dim,
        env.asset_dim,
        hidden_dims=list(critic_cfg.get("trader_hidden_dims", [256, 256])),
        device=device,
    )
    pm_actor_target = copy.deepcopy(pm_actor).to(pm_actor.device)
    trader_actor_target = copy.deepcopy(trader_actor).to(trader_actor.device)
    pm_critic_target = copy.deepcopy(pm_critic).to(pm_critic.device)
    trader_critic_target = copy.deepcopy(trader_critic).to(trader_critic.device)
    return pm_actor, trader_actor, pm_critic, trader_critic, pm_actor_target, trader_actor_target, pm_critic_target, trader_critic_target


def deterministic_pm_action(actor: DeterministicPMActor, obs: np.ndarray) -> np.ndarray:
    with th.no_grad():
        return actor(as_tensor(obs.reshape(1, -1), device=actor.device)).cpu().numpy().reshape(-1).astype(np.float32)


def deterministic_trader_action(actor: DeterministicTraderActor, obs: np.ndarray) -> np.ndarray:
    with th.no_grad():
        return actor(as_tensor(obs.reshape(1, -1), device=actor.device)).cpu().numpy().reshape(-1).astype(np.float32)


def evaluate(env: W1BudgetTraderEnv, pm_actor: DeterministicPMActor, trader_actor: DeterministicTraderActor, *, out_dir: Path, split_name: str) -> dict[str, Any]:
    env.reset()
    rows: list[dict[str, Any]] = []
    pm_open = False
    pm_start_day = 0
    pm_horizon_days = 1
    q_target = 0.0
    pm_action = np.zeros(2, dtype=np.float32)
    while not env.done():
        elapsed = int(env.day) - int(pm_start_day)
        if (not pm_open) or elapsed >= pm_horizon_days:
            pm_action = deterministic_pm_action(pm_actor, env.pm_obs())
            q_target = float(pm_action[0])
            pm_horizon_days = horizon_from_unit(env, float(pm_action[1]))
            pm_start_day = int(env.day)
            pm_open = True
        remaining = max(1, pm_horizon_days - (int(env.day) - int(pm_start_day)))
        trader_obs = env.trader_obs(q_target=q_target, remaining_days=remaining)
        trader_action = deterministic_trader_action(trader_actor, trader_obs)
        info = env.step_trader(q_target=q_target, remaining_days=remaining, trader_action=trader_action)
        row = _daily_row(env, info, pm_horizon_days=pm_horizon_days, pm_start_day=pm_start_day, trader_action=trader_action)
        row["pm_horizon_unit"] = float(pm_action[1])
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
        "stock_turnover_l1_mean": float(daily["stock_turnover_l1"].mean()) if len(daily) else 0.0,
        "cash_exec_mean": float(daily["cash_exec"].mean()) if len(daily) else 0.0,
        "q_target_mean": float(daily["q_target"].mean()) if len(daily) else 0.0,
        "q_exec_mean": float(daily["q_exec"].mean()) if len(daily) else 0.0,
        "tracking_l1_mean": float(daily["tracking_l1"].mean()) if len(daily) else 0.0,
        "pm_horizon_days_mean": float(daily["pm_horizon_days"].mean()) if len(daily) else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{split_name}_summary.csv", index=False)
    return summary


def train_one(config: dict[str, Any], variant: dict[str, Any], fold: pd.Series, *, out_root: Path, smoke_test: bool, force: bool) -> dict[str, Any]:
    fold_id = str(fold["fold"])
    variant_name = str(variant["name"])
    run_dir = out_root / variant_name / fold_id
    summary_path = run_dir / "validation_summary.csv"
    if summary_path.exists() and not force:
        row = pd.read_csv(summary_path).iloc[0].to_dict()
        row.update({"variant": variant_name, "fold": fold_id, "status": "skipped_existing"})
        return row
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_info = feature_csv_for_fold(config, variant, fold, out_root, force=force)
    feature_csv = feature_info["model_ready_csv"]
    train_panel = load_weight_panel(feature_csv, str(fold["train_start"]), str(fold["train_end_inclusive"]))
    validation_panel = load_weight_panel(feature_csv, str(fold["validation_start"]), str(fold["validation_end_inclusive"]))
    train_env = W1BudgetTraderEnv(train_panel, **env_kwargs(config, variant, train_panel))
    val_env = W1BudgetTraderEnv(validation_panel, **env_kwargs(config, variant, validation_panel))

    (
        pm_actor,
        trader_actor,
        pm_critic,
        trader_critic,
        pm_actor_target,
        trader_actor_target,
        pm_critic_target,
        trader_critic_target,
    ) = make_w2_models(train_env, variant)
    q_cfg = variant.get("w2_qcritic", {})
    learning_rate = float(q_cfg.get("learning_rate", 1e-4))
    pm_actor_opt = th.optim.Adam(pm_actor.parameters(), lr=learning_rate)
    trader_actor_opt = th.optim.Adam(trader_actor.parameters(), lr=learning_rate)
    pm_critic_opt = th.optim.Adam(pm_critic.parameters(), lr=learning_rate)
    trader_critic_opt = th.optim.Adam(trader_critic.parameters(), lr=learning_rate)
    rng = np.random.default_rng(int(q_cfg.get("seed", 0)))

    pm_replay = ReplayBuffer(train_env.pm_obs_dim, 2, int(q_cfg.get("pm_replay_size", 50_000)), seed=int(q_cfg.get("seed", 0)))
    trader_replay = ReplayBuffer(train_env.trader_obs_dim, train_env.asset_dim, int(q_cfg.get("trader_replay_size", 100_000)), seed=int(q_cfg.get("seed", 0)) + 1)
    target_days = int(q_cfg.get("total_internal_trading_days", 70_000))
    if smoke_test:
        target_days = min(target_days, 2048)
    batch_size = int(q_cfg.get("batch_size", 256))
    pm_batch_size = int(q_cfg.get("pm_batch_size", 32))
    warmup_days = int(q_cfg.get("warmup_days", 1024 if not smoke_test else 128))
    updates_per_step = int(q_cfg.get("updates_per_step", 1))
    gamma_pm = float(q_cfg.get("pm_gamma", 0.99))
    gamma_trader = float(q_cfg.get("trader_gamma", 0.99))
    tau = float(q_cfg.get("tau", 0.005))
    max_grad_norm = float(q_cfg.get("max_grad_norm", 0.5))
    pm_noise = float(q_cfg.get("pm_noise_std", 0.10))
    trader_mix = float(q_cfg.get("trader_dirichlet_mix", 0.10))
    noise_decay = float(q_cfg.get("exploration_decay", 0.9995))
    min_noise_scale = float(q_cfg.get("min_exploration_scale", 0.10))

    processed_days = 0
    iteration_rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    episode = 0
    while processed_days < target_days:
        episode += 1
        train_env.reset()
        rows: list[dict[str, Any]] = []
        pm_open = False
        pm_start_day = 0
        pm_horizon_days = 1
        q_target = 0.0
        pm_obs_open: np.ndarray | None = None
        pm_action_open: np.ndarray | None = None
        pm_infos: list[dict[str, Any]] = []
        pm_losses: list[QUpdateStats] = []
        trader_losses: list[QUpdateStats] = []

        while not train_env.done() and processed_days < target_days:
            elapsed = int(train_env.day) - int(pm_start_day)
            if (not pm_open) or elapsed >= pm_horizon_days:
                if pm_open and pm_obs_open is not None and pm_action_open is not None:
                    reward = train_env.pm_window_reward_from_infos(pm_infos, q_target=q_target)
                    pm_replay.add(pm_obs_open, pm_action_open, reward, train_env.pm_obs(), False)
                pm_obs_open = train_env.pm_obs()
                scale = max(min_noise_scale, noise_decay ** max(processed_days - warmup_days, 0))
                pm_action_open = noisy_pm_action(pm_actor, pm_obs_open, rng=rng, noise_std=pm_noise * scale)
                q_target = float(pm_action_open[0])
                pm_horizon_days = horizon_from_unit(train_env, float(pm_action_open[1]))
                pm_start_day = int(train_env.day)
                pm_infos = []
                pm_open = True

            remaining = max(1, pm_horizon_days - (int(train_env.day) - int(pm_start_day)))
            trader_obs = train_env.trader_obs(q_target=q_target, remaining_days=remaining)
            scale = max(min_noise_scale, noise_decay ** max(processed_days - warmup_days, 0))
            trader_action = noisy_trader_action(trader_actor, trader_obs, rng=rng, mix_prob=trader_mix * scale)
            info = train_env.step_trader(q_target=q_target, remaining_days=remaining, trader_action=trader_action)
            done = train_env.done()
            next_remaining = max(1, remaining - 1)
            next_trader_obs = np.zeros_like(trader_obs) if done else train_env.trader_obs(q_target=q_target, remaining_days=next_remaining)
            trader_replay.add(trader_obs, trader_action, float(info["trader_reward"]), next_trader_obs, done)
            pm_infos.append(info)
            row = _daily_row(train_env, info, pm_horizon_days=pm_horizon_days, pm_start_day=pm_start_day, trader_action=trader_action)
            row["pm_horizon_unit"] = float(pm_action_open[1]) if pm_action_open is not None else np.nan
            rows.append(row)
            processed_days += 1

            if processed_days >= warmup_days:
                for _ in range(updates_per_step):
                    stat = q_update(
                        actor=trader_actor,
                        actor_target=trader_actor_target,
                        critic=trader_critic,
                        critic_target=trader_critic_target,
                        actor_optimizer=trader_actor_opt,
                        critic_optimizer=trader_critic_opt,
                        replay=trader_replay,
                        batch_size=batch_size,
                        gamma=gamma_trader,
                        tau=tau,
                        max_grad_norm=max_grad_norm,
                    )
                    if stat is not None:
                        trader_losses.append(stat)
                    stat = q_update(
                        actor=pm_actor,
                        actor_target=pm_actor_target,
                        critic=pm_critic,
                        critic_target=pm_critic_target,
                        actor_optimizer=pm_actor_opt,
                        critic_optimizer=pm_critic_opt,
                        replay=pm_replay,
                        batch_size=max(8, int(pm_batch_size)),
                        gamma=gamma_pm,
                        tau=tau,
                        max_grad_norm=max_grad_norm,
                    )
                    if stat is not None:
                        pm_losses.append(stat)

        if pm_open and pm_obs_open is not None and pm_action_open is not None and pm_infos:
            reward = train_env.pm_window_reward_from_infos(pm_infos, q_target=q_target)
            pm_replay.add(pm_obs_open, pm_action_open, reward, np.zeros_like(pm_obs_open), True)

        trace = pd.DataFrame(rows)
        returns = trace["net_return"].to_numpy(dtype=np.float64) if len(trace) else np.zeros(0)
        iteration_rows.append(
            {
                "episode": episode,
                "processed_internal_trading_days": processed_days,
                "episode_days": int(len(trace)),
                "episode_return_pct": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
                "episode_cash_exec_mean": float(trace["cash_exec"].mean()) if len(trace) else 0.0,
                "episode_q_target_mean": float(trace["q_target"].mean()) if len(trace) else 0.0,
                "episode_q_exec_mean": float(trace["q_exec"].mean()) if len(trace) else 0.0,
                "episode_tracking_l1_mean": float(trace["tracking_l1"].mean()) if len(trace) else 0.0,
                "episode_horizon_mean": float(trace["pm_horizon_days"].mean()) if len(trace) else 0.0,
                "pm_replay_size": int(len(pm_replay)),
                "trader_replay_size": int(len(trader_replay)),
                "pm_critic_loss": float(np.mean([x.critic_loss for x in pm_losses])) if pm_losses else np.nan,
                "pm_actor_loss": float(np.mean([x.actor_loss for x in pm_losses])) if pm_losses else np.nan,
                "trader_critic_loss": float(np.mean([x.critic_loss for x in trader_losses])) if trader_losses else np.nan,
                "trader_actor_loss": float(np.mean([x.actor_loss for x in trader_losses])) if trader_losses else np.nan,
            }
        )
        if episode <= int(q_cfg.get("save_train_trace_episodes", 1)):
            trace = trace.copy()
            trace["episode"] = episode
            trace_frames.append(trace)
        print(f"[W2-QAC] ep={episode} days={processed_days}/{target_days} pm_buf={len(pm_replay)} trader_buf={len(trader_replay)}", flush=True)

    pd.DataFrame(iteration_rows).to_csv(run_dir / "train_qcritic_iterations.csv", index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(run_dir / "train_trace_daily.csv", index=False)
    th.save({"state_dict": pm_actor.state_dict(), "variant": variant}, run_dir / "pm_actor.pt")
    th.save({"state_dict": trader_actor.state_dict(), "variant": variant}, run_dir / "trader_actor.pt")
    th.save({"state_dict": pm_critic.state_dict(), "variant": variant}, run_dir / "pm_q_critic.pt")
    th.save({"state_dict": trader_critic.state_dict(), "variant": variant}, run_dir / "trader_q_critic.pt")

    validation_summary = evaluate(val_env, pm_actor, trader_actor, out_dir=run_dir, split_name="validation")
    metadata = {
        "variant": variant,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "target_internal_trading_days": target_days,
        "observed_internal_trading_days": processed_days,
        "pm_obs_dim": train_env.pm_obs_dim,
        "trader_obs_dim": train_env.trader_obs_dim,
        "stock_feature_dim": train_env.stock_feature_dim,
        "trader_task_dim": train_env.trader_task_dim,
        "qcritic_semantics": "DDPG-style deterministic actor maximizes learned Q; TD target is r + gamma * Q_target(s_next, actor_target(s_next)).",
        "validation_summary": validation_summary,
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
    parser.add_argument("--config", default="configs/deprecated/stage0_1_w2_qcritic_budget_trader.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or config.get("output", {}).get("run_name", "weight_based_w2_qcritic_budget_trader")
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== W2 Q-critic PM/trader: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nW2 Q-critic run written to {out_root}")


if __name__ == "__main__":
    main()
