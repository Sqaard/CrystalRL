"""Train T5 synthetic LOB low-level execution policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from src.ppo.dirichlet_policy import SharedStockBranchingExecutionPolicy
from src.ppo.stage0_1_train import feature_csv_for_fold, resolve_variant_inheritance
from src.ppo.stage0_1_two_agent_train import load_folds, make_provider, selected_variants
from src.ppo.weight_panel import load_weight_panel
from src.ppo.synthetic_lob_execution_env import SyntheticLobExecutionEnv
from src.ppo.two_agent_env import stock_order_book_proxy_dim


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def env_kwargs(config: dict[str, Any], variant: dict[str, Any], provider: Any) -> dict[str, Any]:
    env_cfg = config.get("environment", {})
    t5 = variant.get("synthetic_lob_execution", {})
    return {
        "teacher_provider": provider,
        "stock_feature_names": list(t5.get("stock_feature_names", [])),
        "stock_order_book_proxy": dict(t5.get("order_book_proxy", {})),
        "sector_map_name": config.get("universe", {}).get("sector_map", "dow30_static"),
        "price_levels": int(t5.get("price_levels", 5)),
        "quantity_levels": int(t5.get("quantity_levels", 5)),
        "transaction_cost_pct": float(env_cfg.get("transaction_cost_pct", 0.001)),
        "reward_scale": float(env_cfg.get("reward_scale", 100.0)),
        "forced_cleanup": bool(t5.get("forced_cleanup", True)),
        "tracking_penalty": float(t5.get("tracking_penalty", 0.05)),
        "slippage_weight": float(t5.get("slippage_weight", 1.0)),
        "impact_weight": float(t5.get("impact_weight", 1.0)),
    }


def ppo_params(config: dict[str, Any], variant: dict[str, Any], smoke_test: bool) -> tuple[dict[str, Any], int]:
    cfg = {**config.get("ppo", {}), **variant.get("ppo", {})}
    total_timesteps = int(cfg.pop("total_timesteps", 70_000))
    if smoke_test:
        total_timesteps = min(total_timesteps, 2048)
    return {
        "learning_rate": float(cfg.get("learning_rate", 1e-4)),
        "n_steps": min(int(cfg.get("n_steps", 1024)), total_timesteps),
        "batch_size": int(cfg.get("batch_size", 256)),
        "n_epochs": int(cfg.get("n_epochs", 4)),
        "gamma": float(cfg.get("gamma", 0.99)),
        "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
        "clip_range": float(cfg.get("clip_range", 0.1)),
        "ent_coef": float(cfg.get("ent_coef", 0.001)),
        "vf_coef": float(cfg.get("vf_coef", 0.5)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 0.5)),
        "verbose": int(cfg.get("verbose", 1)),
        "device": "cpu",
    }, total_timesteps


def policy_kwargs(variant: dict[str, Any], env: SyntheticLobExecutionEnv) -> dict[str, Any]:
    t5 = variant.get("synthetic_lob_execution", {})
    order_book_cfg = t5.get("order_book_proxy", {})
    return {
        "stock_dim": env.stock_dim,
        "stock_feature_dim": len(t5.get("stock_feature_names", [])),
        "order_book_proxy_dim": stock_order_book_proxy_dim(order_book_cfg),
        "execution_task_dim": env.execution_task_dim,
        "global_context_dim": env.global_context_dim,
        "price_levels": env.price_levels,
        "quantity_levels": env.quantity_levels,
        "stock_hidden_dim": int(t5.get("stock_shared_hidden_dim", 64)),
        "stock_group_ids": env.group_ids,
        "ticker_embedding_dim": int(t5.get("stock_ticker_embedding_dim", 0)),
        "asset_relation_mode": str(t5.get("stock_asset_relation_mode", "group_one_hot")),
        "net_arch": t5.get("net_arch", {"vf": [256, 128]}),
    }


def evaluate(model: PPO, env: SyntheticLobExecutionEnv, out_dir: Path, split: str) -> dict[str, Any]:
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _truncated, info = env.step(action)
        rows.append(dict(info, reward=float(reward)))
    daily = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / f"{split}_daily.csv", index=False)
    returns = daily["net_return"].to_numpy(dtype=np.float64) if len(daily) else np.zeros(0)
    summary = {
        "split": split,
        "days": int(len(daily)),
        "return_pct": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
        "sharpe": float(np.sqrt(252) * returns.mean() / (returns.std(ddof=0) + 1e-12)) if len(returns) else 0.0,
        "turnover_l1_mean": float(daily["turnover_l1"].mean()) if len(daily) else 0.0,
        "slippage_cost_mean": float(daily["slippage_cost"].mean()) if len(daily) else 0.0,
        "commission_cost_mean": float(daily["commission_cost"].mean()) if len(daily) else 0.0,
        "tracking_l1_mean": float(daily["tracking_l1"].mean()) if len(daily) else 0.0,
        "cleanup_l1_mean": float(daily["cleanup_l1"].mean()) if len(daily) else 0.0,
        "fill_prob_mean": float(daily["fill_prob_mean"].mean()) if len(daily) else 0.0,
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
    train_panel = load_weight_panel(feature_info["model_ready_csv"], str(fold["train_start"]), str(fold["train_end_inclusive"]))
    validation_panel = load_weight_panel(
        feature_info["model_ready_csv"], str(fold["validation_start"]), str(fold["validation_end_inclusive"])
    )
    provider = make_provider(config, variant, train_panel, fold_id)
    train_env = SyntheticLobExecutionEnv(train_panel, **env_kwargs(config, variant, provider))
    params, total_timesteps = ppo_params(config, variant, smoke_test)
    model = PPO(
        SharedStockBranchingExecutionPolicy,
        train_env,
        policy_kwargs=policy_kwargs(variant, train_env),
        **params,
    )
    model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(run_dir / "model.zip")

    validation_provider = make_provider(config, variant, validation_panel, fold_id)
    validation_env = SyntheticLobExecutionEnv(validation_panel, **env_kwargs(config, variant, validation_provider))
    summary = evaluate(model, validation_env, run_dir, "validation")
    metadata = {
        "variant": variant,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "total_timesteps": total_timesteps,
        "validation_summary": summary,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    summary.update({"variant": variant_name, "fold": fold_id, "status": "trained"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated/stage0_1_t5_synthetic_lob_execution.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or config.get("output", {}).get("run_name", "weight_based_t5_synthetic_lob")
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== T5 synthetic LOB execution: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nT5 run written to {out_root}")


if __name__ == "__main__":
    main()
