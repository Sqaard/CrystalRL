"""Train-only teacher warm-start helpers for Stage 0.1 PPO policies.

The pretrain implemented here is deliberately narrow: it warm-starts policies
whose action is `(q, risky simplex)` by maximizing the log-probability of
teacher action factors reconstructed from completed train traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, TensorDataset

from src.ppo.stage0_1_weight_env import Stage01WeightPortfolioEnv, WeightPanel, make_env_from_config


EPS = 1e-8


@dataclass(frozen=True)
class PretrainSummary:
    enabled: bool
    status: str
    rows: int = 0
    epochs: int = 0
    final_loss: float | None = None
    final_nll: float | None = None
    final_q_mse: float | None = None
    profile: str = ""
    teacher_ids: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "rows": self.rows,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "final_nll": self.final_nll,
            "final_q_mse": self.final_q_mse,
            "profile": self.profile,
            "teacher_ids": self.teacher_ids,
        }


def read_csv_source(source_path: str) -> pd.DataFrame:
    """Read a plain CSV or a zip-member source encoded as `zip_path::member`."""
    if "::" not in source_path:
        return pd.read_csv(source_path)
    zip_part, member = source_path.split("::", 1)
    with zipfile.ZipFile(zip_part, "r") as zf:
        with zf.open(member) as f:
            return pd.read_csv(f)


def profile_teacher_ids(profile: str) -> list[str]:
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
    raise ValueError(f"Unknown pretrain profile: {profile}")


def weight_prefix_candidates(target_source: str) -> list[str]:
    source = str(target_source).lower()
    if source == "raw":
        return ["raw_weight_", "anchor_weight_", "target_weight_", "executed_weight_"]
    if source == "anchor":
        return ["anchor_weight_", "raw_weight_", "target_weight_", "executed_weight_"]
    if source == "target":
        return ["target_weight_", "anchor_weight_", "raw_weight_", "executed_weight_"]
    if source == "executed":
        return ["executed_weight_", "target_weight_", "anchor_weight_", "raw_weight_"]
    raise ValueError(f"Unknown teacher target_source: {target_source}")


def choose_weight_prefix(frame: pd.DataFrame, tickers: list[str], target_source: str) -> str:
    required = set(tickers + ["CASH"])
    for prefix in weight_prefix_candidates(target_source):
        available = {col.removeprefix(prefix) for col in frame.columns if col.startswith(prefix)}
        if required.issubset(available):
            return prefix
    raise ValueError(
        f"No complete teacher weight columns for target_source={target_source}. "
        f"Tried {weight_prefix_candidates(target_source)}"
    )


def weights_to_action(
    weights: np.ndarray,
    *,
    stock_dim: int,
    q_min: float,
    q_max: float,
    action_dim: int,
) -> np.ndarray:
    stock_weights = np.asarray(weights[:stock_dim], dtype=np.float64)
    stock_weights = np.maximum(stock_weights, 0.0)
    q = float(np.sum(stock_weights))
    q = float(np.clip(q, q_min + 1e-5, q_max - 1e-5))
    if q > 1e-7 and np.sum(stock_weights) > 1e-12:
        risky = stock_weights / np.sum(stock_weights)
    else:
        risky = np.full(stock_dim, 1.0 / stock_dim, dtype=np.float64)
    risky = np.maximum(risky, 1e-6)
    risky = risky / np.sum(risky)
    action = np.concatenate([[q], risky]).astype(np.float32)
    if action_dim == stock_dim + 1:
        return action
    if action_dim == stock_dim + 3:
        # Learned-Kp policies add two gate factors. Use neutral gates for BC.
        return np.concatenate([action, np.array([0.5, 0.5], dtype=np.float32)])
    raise ValueError(f"Unsupported action_dim={action_dim}; expected {stock_dim + 1} or {stock_dim + 3}.")


def set_env_state_from_teacher_row(env: Stage01WeightPortfolioEnv, row: pd.Series, tickers: list[str]) -> None:
    pre_cols = [f"pre_trade_weight_{ticker}" for ticker in tickers] + ["pre_trade_weight_CASH"]
    if all(col in row.index for col in pre_cols):
        weights = np.array([float(row[col]) for col in pre_cols], dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        denom = float(np.sum(weights))
        if denom > EPS:
            env.previous_weights = weights / denom
    elif all(f"executed_weight_{ticker}" in row.index for ticker in tickers + ["CASH"]):
        weights = np.array([float(row[f"executed_weight_{ticker}"]) for ticker in tickers + ["CASH"]], dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        denom = float(np.sum(weights))
        if denom > EPS:
            env.previous_weights = weights / denom

    if "portfolio_value_before" in row.index and np.isfinite(float(row["portfolio_value_before"])):
        env.portfolio_value = float(row["portfolio_value_before"])
    elif "portfolio_value" in row.index and np.isfinite(float(row["portfolio_value"])):
        env.portfolio_value = float(row["portfolio_value"])
    if "drawdown" in row.index and np.isfinite(float(row["drawdown"])):
        env.previous_drawdown = float(row["drawdown"])
    if "turnover_l1" in row.index and np.isfinite(float(row["turnover_l1"])):
        env.last_turnover = float(row["turnover_l1"])


def build_teacher_dataset(
    *,
    config: dict[str, Any],
    variant: dict[str, Any],
    panel: WeightPanel,
    fold_id: str,
    teacher_ids: list[str],
    manifest_csv: str | Path,
    target_source: str,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing teacher manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    selected = manifest[
        manifest["teacher_id"].astype(str).isin(teacher_ids)
        & manifest["fold"].astype(str).eq(str(fold_id))
        & manifest["has_train_trace_daily"].astype(bool)
    ].copy()
    if selected.empty:
        raise ValueError(f"No train teacher traces for fold={fold_id}, teacher_ids={teacher_ids}")

    env = make_env_from_config(panel, config, variant)
    date_to_day = {str(pd.Timestamp(date).date()): idx for idx, date in enumerate(panel.dates)}
    q_min = float(variant.get("root_split", {}).get("q_min", config.get("policy", {}).get("root_split", {}).get("q_min", 0.0)))
    q_max = float(variant.get("root_split", {}).get("q_max", config.get("policy", {}).get("root_split", {}).get("q_max", 0.995)))

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    for _, rec in selected.iterrows():
        teacher_id = str(rec["teacher_id"])
        frame = read_csv_source(str(rec["source_path"]))
        if "date" not in frame.columns:
            continue
        prefix = choose_weight_prefix(frame, panel.tickers, target_source)
        source_counts[f"{teacher_id}:{prefix}"] = source_counts.get(f"{teacher_id}:{prefix}", 0) + len(frame)
        for _, row in frame.iterrows():
            date_key = str(pd.Timestamp(row["date"]).date())
            day = date_to_day.get(date_key)
            if day is None or day >= len(panel.dates) - 1:
                continue
            weights = np.array(
                [float(row[f"{prefix}{ticker}"]) for ticker in panel.tickers] + [float(row[f"{prefix}CASH"])],
                dtype=np.float64,
            )
            if not np.isfinite(weights).all():
                continue
            env.day = int(day)
            set_env_state_from_teacher_row(env, row, panel.tickers)
            obs_rows.append(env._get_obs())
            action_rows.append(
                weights_to_action(
                    weights,
                    stock_dim=len(panel.tickers),
                    q_min=q_min,
                    q_max=q_max,
                    action_dim=int(env.action_space.shape[0]),
                )
            )

    if not obs_rows:
        raise ValueError(f"No usable teacher rows for fold={fold_id}, teacher_ids={teacher_ids}")

    obs = np.asarray(obs_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    if max_rows > 0 and len(obs) > max_rows:
        # Deterministic thinning preserves the full train period instead of
        # taking only the beginning.
        indices = np.linspace(0, len(obs) - 1, num=max_rows, dtype=int)
        obs = obs[indices]
        actions = actions[indices]

    meta = {
        "source_counts": source_counts,
        "rows_before_cap": len(obs_rows),
        "rows_after_cap": len(obs),
        "target_source": target_source,
        "teacher_ids": teacher_ids,
    }
    return obs, actions, meta


def pretrain_policy_from_teachers(
    *,
    model: Any,
    config: dict[str, Any],
    variant: dict[str, Any],
    panel: WeightPanel,
    fold_id: str,
    run_dir: Path,
) -> PretrainSummary:
    cfg = dict(variant.get("pretrain_teachers", {}))
    if not bool(cfg.get("enabled", False)):
        return PretrainSummary(enabled=False, status="disabled")

    profile = str(cfg.get("profile", ""))
    teacher_ids = list(cfg.get("teacher_ids") or profile_teacher_ids(profile))
    target_source = str(cfg.get("target_source", "anchor"))
    max_rows = int(cfg.get("max_rows", 8192))
    epochs = int(cfg.get("epochs", 3))
    batch_size = int(cfg.get("batch_size", 256))
    learning_rate = float(cfg.get("learning_rate", 3e-4))
    q_mse_weight = float(cfg.get("q_mse_weight", 0.05))
    manifest_csv = cfg.get("manifest_csv", "HS/data/current_stage0_1/teacher_bank_manifest.csv")

    obs, actions, meta = build_teacher_dataset(
        config=config,
        variant=variant,
        panel=panel,
        fold_id=fold_id,
        teacher_ids=teacher_ids,
        manifest_csv=manifest_csv,
        target_source=target_source,
        max_rows=max_rows,
    )
    device = model.device
    dataset = TensorDataset(
        th.as_tensor(obs, dtype=th.float32),
        th.as_tensor(actions, dtype=th.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)

    final_loss = final_nll = final_q_mse = None
    model.policy.train()
    for _epoch in range(epochs):
        losses: list[float] = []
        nlls: list[float] = []
        q_mses: list[float] = []
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            _values, log_prob, _entropy = model.policy.evaluate_actions(batch_obs, batch_actions)
            nll = -log_prob.mean()
            mode_actions = model.policy._predict(batch_obs, deterministic=True)
            q_mse = th.mean((mode_actions[:, 0] - batch_actions[:, 0]) ** 2)
            loss = nll + q_mse_weight * q_mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            nlls.append(float(nll.detach().cpu()))
            q_mses.append(float(q_mse.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else None
        final_nll = float(np.mean(nlls)) if nlls else None
        final_q_mse = float(np.mean(q_mses)) if q_mses else None

    summary = PretrainSummary(
        enabled=True,
        status="trained",
        rows=int(len(obs)),
        epochs=epochs,
        final_loss=final_loss,
        final_nll=final_nll,
        final_q_mse=final_q_mse,
        profile=profile,
        teacher_ids=";".join(teacher_ids),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary.as_dict() | {"meta": str(meta)}]).to_csv(run_dir / "pretrain_teacher_summary.csv", index=False)
    return summary
