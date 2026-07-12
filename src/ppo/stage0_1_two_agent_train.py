"""Train Stage 0.1 T2 with two separate PPO policies.

T2 is intentionally not a single joint actor. It trains:

- stock PPO: Dirichlet policy over risky simplex with stock/private state
- root PPO: Beta policy over invested fraction q with root-only state

Training is iterative and pragmatic:

1. optional teacher BC for stock
2. stock PPO with teacher root anchors
3. optional teacher BC for root
4. root PPO with frozen stock model
5. optional stock refinement with frozen root model
6. joint validation rollout
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
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from torch.utils.data import DataLoader, TensorDataset

from src.ppo.dirichlet_policy import RootBetaActorCriticPolicy, SharedStockDirichletActorCriticPolicy
from src.ppo.stage0_1_train import deep_merge_dicts, feature_csv_for_fold, resolve_variant_inheritance
from src.ppo.weight_panel import WeightPanel, load_weight_panel
from src.ppo.two_agent_env import (
    PortfolioState,
    RootAllocationEnv,
    StockAllocationEnv,
    TeacherTraceProvider,
    stock_order_book_proxy_dim,
)


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def selected_variants(config: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    variants = resolve_variant_inheritance(config["variants"])
    if not names:
        return [v for v in variants if v.get("enabled", True)]
    wanted = set(names)
    out = [v for v in variants if v["name"] in wanted]
    missing = wanted.difference(v["name"] for v in out)
    if missing:
        raise ValueError(f"Unknown variants: {sorted(missing)}")
    return out


def load_folds(config: dict[str, Any], names: list[str] | None) -> pd.DataFrame:
    folds = pd.read_csv(resolve(config["walk_forward"]["folds_csv"]))
    if names:
        folds = folds[folds["fold"].isin(names)].copy()
    if folds.empty:
        raise ValueError(f"No folds selected: {names}")
    return folds


def teacher_ids(profile: str) -> list[str]:
    if profile == "r6c_raw_intent":
        return ["teacher_r6c_current_best_interpretable"]
    if profile == "r6c_r3_r8_r7_mix":
        return [
            "teacher_r6c_current_best_interpretable",
            "teacher_r3_confidence_slice_baseline",
            "teacher_r8_sellside_priority_ablation",
            "teacher_r7_rescorr_groupquality_contrast",
        ]
    if profile == "flat_compact_after_export":
        return [
            "teacher_flat_pd_stock",
            "teacher_compact22_stock_features",
        ]
    if profile == "e5_e6_e12_aux":
        return [
            "teacher_e5_root_execution",
            "teacher_e6_learned_kp",
            "teacher_e12_style_bank",
        ]
    if profile == "flat_compact_e_aux_after_export":
        return [
            "teacher_flat_pd_stock",
            "teacher_compact22_stock_features",
            "teacher_e5_root_execution",
            "teacher_e6_learned_kp",
            "teacher_e12_style_bank",
        ]
    if profile == "flat_compact_plus_rline_aux_after_export":
        return [
            "teacher_flat_pd_stock",
            "teacher_compact22_stock_features",
            "teacher_r6c_current_best_interpretable",
            "teacher_r3_confidence_slice_baseline",
            "teacher_r8_sellside_priority_ablation",
            "teacher_r7_rescorr_groupquality_contrast",
        ]
    if profile == "all_teacher_mix_after_export":
        return [
            "teacher_flat_pd_stock",
            "teacher_compact22_stock_features",
            "teacher_e5_root_execution",
            "teacher_e6_learned_kp",
            "teacher_e12_style_bank",
            "teacher_r6c_current_best_interpretable",
            "teacher_r3_confidence_slice_baseline",
            "teacher_r8_sellside_priority_ablation",
            "teacher_r7_rescorr_groupquality_contrast",
        ]
    raise ValueError(f"Unknown teacher profile: {profile}")


def make_provider(config: dict[str, Any], variant: dict[str, Any], panel: WeightPanel, fold_id: str) -> TeacherTraceProvider:
    cfg = variant.get("teacher_provider", {})
    profile = str(cfg.get("profile", "r6c_r3_r8_r7_mix"))
    ids = list(cfg.get("teacher_ids") or teacher_ids(profile))
    return TeacherTraceProvider(
        manifest_csv=resolve(cfg.get("manifest_csv", "HS/data/current_stage0_1/pretrain_teacher_manifest_portable.csv")),
        fold_id=fold_id,
        teacher_ids=ids,
        tickers=panel.tickers,
        target_source=str(cfg.get("target_source", "anchor")),
    )


def env_kwargs(config: dict[str, Any], variant: dict[str, Any], panel: WeightPanel, provider: TeacherTraceProvider) -> dict[str, Any]:
    env_cfg = config.get("environment", {})
    t2 = variant.get("two_agent", {})
    root_state = t2.get("root_state", {})
    stock_state = t2.get("stock_state", {})
    return {
        "root_feature_names": list(root_state.get("feature_names", [])),
        "stock_feature_names": list(stock_state.get("feature_names", [])),
        "root_raw_window_feature_names": list(root_state.get("raw_window_feature_names", [])),
        "root_raw_window_days": int(root_state.get("raw_window_days", 0)),
        "stock_order_book_proxy": dict(stock_state.get("order_book_proxy", {})),
        "root_window_days": int(t2.get("root_window_days", 20)),
        "q_min": float(t2.get("q_min", 0.0)),
        "q_max": float(t2.get("q_max", 0.995)),
        "transaction_cost_pct": float(env_cfg.get("transaction_cost_pct", 0.001)),
        "initial_amount": float(env_cfg.get("initial_amount", 1_000_000.0)),
        "reward_scale": float(env_cfg.get("reward_scale", 100.0)),
        "teacher_provider": provider,
        "sector_map_name": config.get("universe", {}).get("sector_map", "dow30_static"),
        "reward_config": copy.deepcopy(variant.get("two_agent_reward", {})),
        "root_low_level_diagnostics": dict(root_state.get("low_level_diagnostics", {})),
    }


def ppo_params(config: dict[str, Any], variant: dict[str, Any], level: str, smoke_test: bool) -> tuple[dict[str, Any], int]:
    base = dict(config.get("ppo", {}))
    cfg = deep_merge_dicts(base, variant.get("two_agent", {}).get(f"{level}_ppo", {}))
    total_timesteps = int(cfg.pop("total_timesteps", 50_000))
    if smoke_test:
        total_timesteps = min(total_timesteps, 1024)
    out = {
        "learning_rate": float(cfg.get("learning_rate", 1e-4)),
        "n_steps": min(int(cfg.get("n_steps", 1024)), total_timesteps),
        "batch_size": int(cfg.get("batch_size", 256)),
        "n_epochs": int(cfg.get("n_epochs", 4)),
        "gamma": float(cfg.get("gamma", 0.99)),
        "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
        "clip_range": float(cfg.get("clip_range", 0.1)),
        "ent_coef": float(cfg.get("ent_coef", 0.0)),
        "vf_coef": float(cfg.get("vf_coef", 0.5)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 0.5)),
        "verbose": int(cfg.get("verbose", 1)),
        "device": "cpu",
    }
    return out, total_timesteps


def stock_policy_kwargs(variant: dict[str, Any], env: StockAllocationEnv) -> dict[str, Any]:
    """Build low-level stock policy kwargs from the structured stock observation.

    The stock observation is intentionally not a flat feature bag:
    [stock features per ticker] + [optional per-ticker execution proxy] +
    [portfolio/root private context].  The shared stock policy needs these
    dimensions explicitly so it can apply the same scorer to each stock.
    """

    two_agent = variant.get("two_agent", {})
    stock_state = two_agent.get("stock_state", {})
    order_book_cfg = stock_state.get("order_book_proxy", {})
    order_book_proxy_dim = stock_order_book_proxy_dim(order_book_cfg)
    return {
        "alpha_min": 0.05,
        "alpha_max": 100.0,
        "stock_dim": env.stock_dim,
        "stock_feature_dim": len(stock_state.get("feature_names", [])),
        "order_book_proxy_dim": order_book_proxy_dim,
        "global_context_dim": 12,
        "stock_hidden_dim": int(two_agent.get("stock_shared_hidden_dim", 64)),
        "stock_group_ids": env.group_ids,
        "ticker_embedding_dim": int(two_agent.get("stock_ticker_embedding_dim", 0)),
        "asset_relation_mode": str(two_agent.get("stock_asset_relation_mode", "group_one_hot")),
        "net_arch": two_agent.get("stock_net_arch", {"vf": [256, 128]}),
    }


def set_env_state_from_teacher(env: RootAllocationEnv | StockAllocationEnv, weights: np.ndarray, day: int) -> None:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    weights = weights / max(float(np.sum(weights)), 1e-8)
    env.state = PortfolioState(
        day=int(day),
        previous_weights=weights,
        portfolio_value=env.initial_amount,
        peak_value=env.initial_amount,
        previous_drawdown=0.0,
        last_turnover=0.0,
    )


def bc_dataset_root(env: RootAllocationEnv, max_rows: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    # Root actions are macro decisions: one q anchor is held for a full
    # root_window.  BC should therefore imitate teacher q on the same macro
    # update cadence instead of silently teaching a daily root policy.
    for day in range(0, len(env.panel.dates) - 1, env.root_window_days):
        date = env.panel.dates[day]
        weights = env.teacher_provider.weights_for_date(date) if env.teacher_provider else env.state.previous_weights
        set_env_state_from_teacher(env, weights, day)
        obs_rows.append(env.root_obs())
        q = env.teacher_provider.q_for_date(date) if env.teacher_provider else float(np.sum(weights[: env.stock_dim]))
        action_rows.append(np.array([np.clip(q, env.q_min + 1e-5, env.q_max - 1e-5)], dtype=np.float32))
    return _thin(np.asarray(obs_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32), max_rows)


def bc_dataset_stock(env: StockAllocationEnv, max_rows: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    root_anchor_q = env.teacher_provider.q_for_date(env.panel.dates[0]) if env.teacher_provider else 0.5
    root_anchor_start_day = 0
    for day, date in enumerate(env.panel.dates[:-1]):
        if day == 0 or (day - root_anchor_start_day) >= env.root_window_days:
            root_anchor_q = env.teacher_provider.q_for_date(date) if env.teacher_provider else float(np.sum(env.state.previous_weights[: env.stock_dim]))
            root_anchor_start_day = int(day)
        remaining = max(1, env.root_window_days - max(0, int(day) - int(root_anchor_start_day)))
        weights = env.teacher_provider.weights_for_date(date) if env.teacher_provider else env.state.previous_weights
        set_env_state_from_teacher(env, weights, day)
        env.root_anchor_q = float(root_anchor_q)
        env.root_anchor_start_day = int(root_anchor_start_day)
        obs_rows.append(env.stock_obs(q_anchor=env.root_anchor_q, remaining_days=remaining))
        u = env.teacher_provider.u_for_date(date) if env.teacher_provider else np.full(env.stock_dim, 1.0 / env.stock_dim)
        action_rows.append(u.astype(np.float32))
    return _thin(np.asarray(obs_rows, dtype=np.float32), np.asarray(action_rows, dtype=np.float32), max_rows)


def _thin(obs: np.ndarray, actions: np.ndarray, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows > 0 and len(obs) > max_rows:
        idx = np.linspace(0, len(obs) - 1, num=max_rows, dtype=int)
        return obs[idx], actions[idx]
    return obs, actions


def bc_pretrain(model: Any, obs: np.ndarray, actions: np.ndarray, *, epochs: int, batch_size: int, lr: float, out_path: Path) -> dict[str, Any]:
    if epochs <= 0:
        return {"enabled": False, "rows": len(obs)}
    policy = getattr(model, "policy", model)
    device = getattr(model, "device", getattr(policy, "device", "cpu"))
    dataset = TensorDataset(th.as_tensor(obs, dtype=th.float32), th.as_tensor(actions, dtype=th.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)
    final_loss = None
    policy.train()
    for _ in range(epochs):
        losses = []
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            _values, log_prob, _entropy = policy.evaluate_actions(batch_obs, batch_actions)
            loss = -log_prob.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else None
    summary = {"enabled": True, "rows": len(obs), "epochs": epochs, "final_nll": final_loss}
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return summary


def evaluate_joint(
    *,
    root_model: PPO,
    stock_model: PPO,
    panel: WeightPanel,
    config: dict[str, Any],
    variant: dict[str, Any],
    provider: TeacherTraceProvider,
    out_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    env = StockAllocationEnv(panel, frozen_root_model=root_model, **env_kwargs(config, variant, panel, provider))
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action, _ = stock_model.predict(obs, deterministic=True)
        obs, reward, done, _truncated, info = env.step(action)
        row = {
            "date": info.get("date", ""),
            "net_return": info.get("net_return", 0.0),
            "gross_return": info.get("gross_return", 0.0),
            "reward": reward,
            "stock_reward": info.get("stock_reward", 0.0),
            "q_anchor": info.get("q_anchor", np.nan),
            "q_scheduled": info.get("q_scheduled", np.nan),
            "cash_weight": float(info.get("target_weights", np.zeros(env.asset_dim))[env.cash_index]),
            "turnover_l1": info.get("turnover_l1", 0.0),
            "stock_turnover_l1": info.get("stock_turnover_l1", 0.0),
            "transaction_cost": info.get("transaction_cost", 0.0),
            "drawdown": info.get("drawdown", 0.0),
            "risky_entropy": info.get("risky_entropy", np.nan),
        }
        target = np.asarray(info.get("target_weights", np.zeros(env.asset_dim)), dtype=np.float64)
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
    kwargs = env_kwargs(config, variant, train_panel, provider)

    stock_env = StockAllocationEnv(train_panel, **kwargs)
    stock_params, stock_steps = ppo_params(config, variant, "stock", smoke_test)
    stock_model = PPO(
        SharedStockDirichletActorCriticPolicy,
        stock_env,
        policy_kwargs=stock_policy_kwargs(variant, stock_env),
        **stock_params,
    )
    stock_model.set_logger(configure(str(run_dir / "stock_sb3_logs"), ["stdout", "csv"]))
    bc_cfg = variant.get("two_agent", {}).get("bc_pretrain", {})
    stock_bc_summary = {}
    if bool(bc_cfg.get("enabled", True)):
        obs, actions = bc_dataset_stock(stock_env, int(bc_cfg.get("max_rows", 8192)))
        stock_bc_summary = bc_pretrain(
            stock_model,
            obs,
            actions,
            epochs=int(bc_cfg.get("stock_epochs", 2 if not smoke_test else 1)),
            batch_size=int(bc_cfg.get("batch_size", 256)),
            lr=float(bc_cfg.get("learning_rate", 3e-4)),
            out_path=run_dir / "stock_bc_summary.csv",
        )
    stock_model.learn(total_timesteps=stock_steps, progress_bar=False)

    root_env = RootAllocationEnv(train_panel, frozen_stock_model=stock_model, **kwargs)
    root_params, root_steps = ppo_params(config, variant, "root", smoke_test)
    root_model = PPO(
        RootBetaActorCriticPolicy,
        root_env,
        policy_kwargs={
            "q_min": float(variant.get("two_agent", {}).get("q_min", 0.0)),
            "q_max": float(variant.get("two_agent", {}).get("q_max", 0.995)),
            "alpha_floor": 0.05,
            "kappa_min": 2.0,
            "kappa_max": 80.0,
            "net_arch": variant.get("two_agent", {}).get("root_net_arch", {"pi": [128, 64], "vf": [128, 64]}),
        },
        **root_params,
    )
    root_model.set_logger(configure(str(run_dir / "root_sb3_logs"), ["stdout", "csv"]))
    root_bc_summary = {}
    if bool(bc_cfg.get("enabled", True)):
        obs, actions = bc_dataset_root(root_env, int(bc_cfg.get("max_rows", 8192)))
        root_bc_summary = bc_pretrain(
            root_model,
            obs,
            actions,
            epochs=int(bc_cfg.get("root_epochs", 2 if not smoke_test else 1)),
            batch_size=int(bc_cfg.get("batch_size", 256)),
            lr=float(bc_cfg.get("learning_rate", 3e-4)),
            out_path=run_dir / "root_bc_summary.csv",
        )
    root_model.learn(total_timesteps=root_steps, progress_bar=False)

    refine_steps = int(variant.get("two_agent", {}).get("stock_refine_timesteps", 0))
    if smoke_test:
        refine_steps = min(refine_steps, 512)
    if refine_steps > 0:
        refine_env = StockAllocationEnv(train_panel, frozen_root_model=root_model, **kwargs)
        stock_model.set_env(refine_env)
        stock_model.learn(total_timesteps=refine_steps, reset_num_timesteps=False, progress_bar=False)

    root_model.save(run_dir / "root_model.zip")
    stock_model.save(run_dir / "stock_model.zip")
    validation_provider = make_provider(config, variant, validation_panel, fold_id)
    validation_summary = evaluate_joint(
        root_model=root_model,
        stock_model=stock_model,
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
        "root_steps": root_steps,
        "stock_steps": stock_steps,
        "stock_bc_summary": stock_bc_summary,
        "root_bc_summary": root_bc_summary,
        "validation_summary": validation_summary,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    validation_summary.update({"variant": variant_name, "fold": fold_id, "status": "trained"})
    return validation_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated/stage0_1_t2_two_agent.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or config.get("output", {}).get("run_name", "weight_based_t2_two_agent")
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== T2 two-agent: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nT2 run written to {out_root}")


if __name__ == "__main__":
    main()
