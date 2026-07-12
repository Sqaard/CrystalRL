"""Train W1 budget-PM / stock-selector Trader.

W1 is a clean two-policy baseline:

* PM policy: macro/risk state -> Beta `q_target` + categorical horizon.
* Trader policy: per-stock state + PM private task -> Dirichlet full-portfolio weights.

There is no CTDE, no controller, no threshold/trigger/slice layer, no Top-K,
no group/ticker/graph actor relation, and no BC/iterative pretrain.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml

from src.ppo.w1_config_utils import feature_csv_for_fold, resolve_variant_inheritance
from src.ppo.weight_panel import load_weight_panel
from src.ppo.w1_budget_trader_env import W1BudgetTraderEnv
from src.ppo.w1_budget_trader_policy import BudgetPMActorCritic, BudgetTraderActorCritic, LatentActionTraderActorCritic


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def filter_state_dict_by_regex_patterns(
    state_dict: dict[str, th.Tensor],
    patterns: list[str] | tuple[str, ...],
) -> tuple[dict[str, th.Tensor], list[str]]:
    """Drop selected checkpoint keys before loading a warm-start checkpoint.

    This is used when we want actor weights from one teacher but must preserve
    buffers created from another teacher, such as latent-action prototypes.
    """

    if not patterns:
        return state_dict, []
    compiled = [re.compile(str(pattern)) for pattern in patterns]
    kept: dict[str, th.Tensor] = {}
    ignored: list[str] = []
    for key, value in state_dict.items():
        if any(pattern.search(key) for pattern in compiled):
            ignored.append(key)
            continue
        kept[key] = value
    return kept, ignored


@dataclass
class PolicyBatch:
    obs: np.ndarray
    actions: np.ndarray
    old_log_prob: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    critic_obs: np.ndarray | None = None
    aux_targets: np.ndarray | None = None
    sample_weight: np.ndarray | None = None


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
        folds = folds[folds["fold"].astype(str).isin(names)].copy()
    if folds.empty:
        raise ValueError(f"No folds selected: {names}")
    return folds


def as_tensor(array: np.ndarray, *, device: th.device | str) -> th.Tensor:
    return th.as_tensor(array, dtype=th.float32, device=device)


def sample_policy(policy: Any, obs: np.ndarray, *, deterministic: bool = False, critic_obs: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    obs_t = as_tensor(obs.reshape(1, -1), device=policy.device)
    critic_t = as_tensor(critic_obs.reshape(1, -1), device=policy.device) if critic_obs is not None else None
    with th.no_grad():
        action_t, value_t, log_prob_t = policy(obs_t, deterministic=deterministic, critic_obs=critic_t)
    return (
        action_t.detach().cpu().numpy().reshape(-1),
        float(value_t.detach().cpu().numpy().reshape(-1)[0]),
        float(log_prob_t.detach().cpu().numpy().reshape(-1)[0]),
    )


def decode_policy_actions(policy: Any, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if not hasattr(policy, "decode_actions"):
        return np.asarray(actions, dtype=np.float32)
    obs_t = as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1) if obs.ndim == 1 else obs, device=policy.device)
    actions_arr = np.asarray(actions, dtype=np.float32)
    actions_t = as_tensor(actions_arr.reshape(1, -1) if actions_arr.ndim == 1 else actions_arr, device=policy.device)
    with th.no_grad():
        decoded = policy.decode_actions(obs_t, actions_t)
    out = decoded.detach().cpu().numpy().astype(np.float32)
    return out.reshape(-1) if actions_arr.ndim == 1 else out


def policy_action_diagnostics(policy: Any, obs: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    if not hasattr(policy, "action_diagnostics"):
        return {}
    with th.no_grad():
        return policy.action_diagnostics(obs, actions)


def compute_gae(batch: PolicyBatch, *, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    rewards = batch.rewards.astype(np.float64)
    values = batch.values.astype(np.float64)
    dones = batch.dones.astype(np.float64)
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value = values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def ppo_update(
    *,
    policy: Any,
    batch: PolicyBatch,
    advantages: np.ndarray,
    returns: np.ndarray,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    reference_policy: Any | None = None,
    reference_kl_coef: float = 0.0,
    aux_loss_coef: float = 0.0,
    aux_loss_kind: str = "mse",
) -> dict[str, float]:
    if len(batch.obs) == 0:
        return {"updates": 0.0}
    device = policy.device
    obs_t = as_tensor(batch.obs, device=device)
    actions_t = as_tensor(batch.actions, device=device)
    old_log_t = as_tensor(batch.old_log_prob, device=device)
    if batch.sample_weight is None:
        sample_weight = np.ones(len(batch.obs), dtype=np.float32)
    else:
        sample_weight = np.asarray(batch.sample_weight, dtype=np.float32).reshape(-1)
        if len(sample_weight) != len(batch.obs):
            raise ValueError(f"sample_weight length mismatch: {len(sample_weight)} != {len(batch.obs)}")
        sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
        sample_weight = np.maximum(sample_weight, 1e-6)
    sample_weight = sample_weight / max(float(sample_weight.mean()), 1e-6)
    sample_weight_t = as_tensor(sample_weight, device=device)
    adv = advantages.astype(np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
    adv_t = as_tensor(adv, device=device)
    returns_t = as_tensor(returns.astype(np.float32), device=device)
    critic_obs_t = as_tensor(batch.critic_obs, device=device) if batch.critic_obs is not None else None
    aux_targets_t = as_tensor(batch.aux_targets, device=device) if batch.aux_targets is not None else None
    indices = np.arange(len(batch.obs))
    stats: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
        "loss": [],
        "reference_kl": [],
        "aux_loss": [],
        "aux_accuracy": [],
    }
    policy.train()
    if reference_policy is not None:
        reference_policy.eval()
    def weighted_mean(values: th.Tensor, weights: th.Tensor) -> th.Tensor:
        return th.sum(values * weights) / th.clamp(th.sum(weights), min=1e-8)

    def reference_kl_divergence(reference_outputs: tuple[Any, ...], current_outputs: tuple[Any, ...]) -> th.Tensor:
        if len(reference_outputs) >= 3 and len(current_outputs) >= 3:
            return th.distributions.kl_divergence(reference_outputs[0], current_outputs[0]) + th.distributions.kl_divergence(
                reference_outputs[1], current_outputs[1]
            )
        return th.distributions.kl_divergence(reference_outputs[0], current_outputs[0])

    for _ in range(max(1, int(n_epochs))):
        np.random.shuffle(indices)
        for start in range(0, len(indices), max(1, int(batch_size))):
            mb = indices[start : start + max(1, int(batch_size))]
            mb_weight = sample_weight_t[mb]
            critic_mb = critic_obs_t[mb] if critic_obs_t is not None else None
            values, log_prob, entropy = policy.evaluate_actions(obs_t[mb], actions_t[mb], critic_obs=critic_mb)
            values = values.flatten()
            ratio = th.exp(log_prob - old_log_t[mb])
            pg_loss_1 = adv_t[mb] * ratio
            pg_loss_2 = adv_t[mb] * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = -weighted_mean(th.min(pg_loss_1, pg_loss_2), mb_weight)
            value_loss = weighted_mean(th.nn.functional.mse_loss(values, returns_t[mb], reduction="none"), mb_weight)
            entropy_loss = -weighted_mean(log_prob, mb_weight) if entropy is None else -weighted_mean(entropy, mb_weight)
            reference_kl = th.zeros((), dtype=values.dtype, device=values.device)
            if reference_policy is not None and float(reference_kl_coef) > 0.0:
                if not hasattr(policy, "_dists") or not hasattr(reference_policy, "_dists"):
                    raise ValueError("reference_policy KL regularization requires policies with a _dists(obs) method")
                current_outputs = policy._dists(obs_t[mb])
                with th.no_grad():
                    reference_outputs = reference_policy._dists(obs_t[mb])
                reference_kl = weighted_mean(reference_kl_divergence(reference_outputs, current_outputs), mb_weight)
            aux_loss = th.zeros((), dtype=values.dtype, device=values.device)
            aux_accuracy = th.zeros((), dtype=values.dtype, device=values.device)
            if aux_targets_t is not None and float(aux_loss_coef) > 0.0:
                if not hasattr(policy, "aux_predictions"):
                    raise ValueError("aux_loss requested, but policy has no aux_predictions method")
                aux_pred = policy.aux_predictions(obs_t[mb])
                aux_target = aux_targets_t[mb]
                kind = str(aux_loss_kind).lower()
                if kind in {"bce", "binary", "binary_cross_entropy"}:
                    per_item = th.nn.functional.binary_cross_entropy_with_logits(aux_pred, aux_target, reduction="none").mean(dim=1)
                    aux_loss = weighted_mean(per_item, mb_weight)
                    with th.no_grad():
                        per_item_accuracy = ((th.sigmoid(aux_pred) >= 0.5) == (aux_target >= 0.5)).float().mean(dim=1)
                        aux_accuracy = weighted_mean(per_item_accuracy, mb_weight)
                elif kind in {"mse", "regression"}:
                    per_item = th.nn.functional.mse_loss(aux_pred, aux_target, reduction="none").mean(dim=1)
                    aux_loss = weighted_mean(per_item, mb_weight)
                else:
                    raise ValueError(f"Unsupported aux_loss_kind: {aux_loss_kind}")
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            if float(reference_kl_coef) > 0.0:
                loss = loss + float(reference_kl_coef) * reference_kl
            if float(aux_loss_coef) > 0.0:
                loss = loss + float(aux_loss_coef) * aux_loss
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy.optimizer.step()
            with th.no_grad():
                log_ratio = log_prob - old_log_t[mb]
                approx_kl = weighted_mean((th.exp(log_ratio) - 1.0) - log_ratio, mb_weight).detach().cpu().item()
                clipped = weighted_mean((th.abs(ratio - 1.0) > clip_range).float(), mb_weight).detach().cpu().item()
            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy_loss"].append(float(entropy_loss.detach().cpu()))
            stats["approx_kl"].append(float(approx_kl))
            stats["clip_fraction"].append(float(clipped))
            stats["reference_kl"].append(float(reference_kl.detach().cpu()))
            stats["aux_loss"].append(float(aux_loss.detach().cpu()))
            stats["aux_accuracy"].append(float(aux_accuracy.detach().cpu()))
            stats["loss"].append(float(loss.detach().cpu()))
    return {key: float(np.mean(values)) for key, values in stats.items() if values} | {
        "updates": float(len(stats["loss"])),
        "sample_weight_mean": float(sample_weight.mean()),
        "sample_weight_max": float(sample_weight.max()),
        "sample_weight_min": float(sample_weight.min()),
    }


def env_kwargs(config: dict[str, Any], variant: dict[str, Any], panel: Any) -> dict[str, Any]:
    env_cfg = config.get("environment", {})
    w1 = variant.get("w1", {})
    pm_state = w1.get("pm_state", {})
    trader_state = w1.get("trader_state", {})
    return {
        "pm_feature_names": list(pm_state.get("feature_names", [])),
        "pm_raw_window_feature_names": list(pm_state.get("raw_window_feature_names", [])),
        "stock_feature_names": list(trader_state.get("stock_feature_names", [])),
        "horizon_choices": list(w1.get("horizon_choices", [1, 5, 10, 20])),
        "transaction_cost_pct": float(env_cfg.get("transaction_cost_pct", 0.001)),
        "initial_amount": float(env_cfg.get("initial_amount", 1_000_000.0)),
        "reward_scale": float(env_cfg.get("reward_scale", 100.0)),
        "q_min": float(w1.get("q_min", 0.0)),
        "q_max": float(w1.get("q_max", 0.995)),
        "max_horizon_days": int(w1.get("max_horizon_days", max(w1.get("horizon_choices", [20])))),
        "sector_map_name": config.get("universe", {}).get("sector_map", "dow30_static"),
        "pm_reward_config": dict(variant.get("w1_reward", {}).get("pm", {})),
        "trader_reward_config": dict(variant.get("w1_reward", {}).get("trader", {})),
        "execution_config": dict(w1.get("execution", {})),
        "risk_stop_config": dict(w1.get("risk_stop", {})),
        "trader_feedback_alpha": float(w1.get("trader_feedback_alpha", 0.30)),
        "initial_weights_source": str(w1.get("initial_weights_source", "cash")),
    }


def build_stock_relation_matrix(env: W1BudgetTraderEnv, relation_cfg: dict[str, Any]) -> np.ndarray | None:
    if not bool(relation_cfg.get("enabled", False)):
        return None
    mode = str(relation_cfg.get("mode", "sector")).lower()
    n = env.stock_dim
    relation = np.zeros((n, n), dtype=np.float32)
    if mode == "sector":
        groups = np.asarray(env.group_ids, dtype=int)
        relation = (groups.reshape(-1, 1) == groups.reshape(1, -1)).astype(np.float32)
        np.fill_diagonal(relation, 0.0)
    elif mode in {"rescorr", "graph"}:
        returns = np.asarray(env.panel.returns_next, dtype=np.float64)
        if returns.ndim != 2 or returns.shape[1] != n:
            return None
        residual = returns - returns.mean(axis=1, keepdims=True)
        corr = np.corrcoef(residual.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np.fill_diagonal(corr, 0.0)
        if mode == "graph" or bool(relation_cfg.get("absolute", False)):
            corr = np.abs(corr)
        else:
            corr = np.maximum(corr, 0.0)
        top_k = int(relation_cfg.get("top_k", 0))
        if top_k > 0 and top_k < n:
            keep = np.zeros_like(corr)
            for i in range(n):
                idx = np.argsort(corr[i])[-top_k:]
                keep[i, idx] = corr[i, idx]
            corr = keep
        relation = corr
    else:
        raise ValueError(f"Unsupported W1 relation_layer mode: {mode}")

    row_sum = relation.sum(axis=1, keepdims=True)
    relation = np.divide(relation, np.maximum(row_sum, 1e-8), out=np.zeros_like(relation), where=row_sum > 1e-8)
    return relation.astype(np.float32)


def build_latent_action_prototypes(
    env: W1BudgetTraderEnv,
    latent_cfg: dict[str, Any],
    *,
    fold_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train-only latent action prototype portfolios for the Trader.

    The prototype file is expected to be a date x asset target table such as
    P13's `p13_asset_corrected_teacher_targets_long.csv`.  Codes with no usable
    prototype fall back to an equal-risky portfolio at `fallback_q`.
    """

    path = resolve(latent_cfg.get("prototype_path", "artifacts/pretrain/P13_corrected_teacher_trajectories_v1/p13_asset_corrected_teacher_targets_long.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Latent-action prototype file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    code_col = str(latent_cfg.get("code_col", "corrected_action_code"))
    target_col = str(latent_cfg.get("target_col", "corrected_teacher_target_weight"))
    weight_col = str(latent_cfg.get("weight_col", "corrected_teacher_sample_weight"))
    required = {code_col, "tic", target_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Latent-action prototype file missing columns {missing}: {path}")
    if "split" in df.columns:
        df = df.loc[df["split"].astype(str).eq("train")].copy()
    df[code_col] = df[code_col].map(lambda value: str(int(float(value))) if pd.notna(value) and str(value).replace(".", "", 1).isdigit() else str(value))
    df = df.loc[df[code_col].ne("") & df[code_col].ne("nan")].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if weight_col in df.columns:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df[weight_col] = 1.0
    positive = df[weight_col].gt(0.0)
    # Keep the code vocabulary stable for strict checkpoint loading; estimate
    # prototype weights below from the current fold only.
    code_source = df.loc[positive].copy() if positive.any() else df
    codes = sorted(pd.to_numeric(code_source[code_col], errors="coerce").dropna().astype(int).unique().tolist())
    if not codes:
        codes = [0]
    configured_codes = latent_cfg.get("code_values", None)
    if configured_codes:
        codes = sorted({int(x) for x in configured_codes})
    code_to_index = {int(code): idx for idx, code in enumerate(codes)}
    num_codes = len(codes)
    if "fold" in df.columns and fold_id:
        df = df.loc[df["fold"].astype(str).eq(str(fold_id))].copy()
    positive = df[weight_col].gt(0.0)
    if positive.any():
        df = df.loc[positive].copy()
    fallback_q = float(latent_cfg.get("fallback_q", 0.85))
    fallback = np.zeros(env.asset_dim, dtype=np.float32)
    fallback[: env.stock_dim] = fallback_q / max(env.stock_dim, 1)
    fallback[env.cash_index] = 1.0 - fallback_q
    prototypes = np.repeat(fallback.reshape(1, -1), num_codes, axis=0)

    asset_names = list(env.panel.tickers) + ["CASH"]
    asset_index = {str(tic): idx for idx, tic in enumerate(asset_names)}
    grouped_rows = []
    for (code_raw, tic), group in df.groupby([code_col, "tic"], sort=True):
        code = int(float(code_raw))
        tic = str(tic)
        if code not in code_to_index or tic not in asset_index:
            continue
        weights = group[weight_col].to_numpy(dtype=np.float64)
        values = group[target_col].to_numpy(dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            value = float(np.nanmean(values))
        else:
            value = float(np.average(values, weights=weights))
        grouped_rows.append((code_to_index[code], asset_index[tic], max(value, 0.0)))
    for code_idx in range(num_codes):
        rows = [(idx, value) for row_code, idx, value in grouped_rows if row_code == code_idx]
        if not rows:
            continue
        vector = np.zeros(env.asset_dim, dtype=np.float32)
        for idx, value in rows:
            vector[idx] = float(value)
        total = float(vector.sum())
        if total > 0.0:
            prototypes[code_idx] = vector / total
    return prototypes.astype(np.float32), np.asarray(codes, dtype=np.int64)


def trader_central_critic_obs(env: W1BudgetTraderEnv, pm_obs: np.ndarray, *, q_target: float, horizon_days: int, remaining_days: int) -> np.ndarray:
    max_horizon = float(max(env.max_horizon_days, 1))
    scalars = np.asarray(
        [
            float(q_target),
            float(horizon_days) / max_horizon,
            float(max(1, remaining_days)) / max_horizon,
        ],
        dtype=np.float32,
    )
    return np.concatenate([pm_obs.astype(np.float32).reshape(-1), scalars], axis=0).astype(np.float32)


def make_policies(
    env: W1BudgetTraderEnv,
    variant: dict[str, Any],
    *,
    fold_id: str | None = None,
) -> tuple[BudgetPMActorCritic, BudgetTraderActorCritic]:
    w1 = variant.get("w1", {})
    pm_cfg = w1.get("pm_policy", {})
    trader_cfg = w1.get("trader_policy", {})
    relation_cfg = trader_cfg.get("relation_layer", {})
    pm = BudgetPMActorCritic(
        env.pm_obs_dim,
        horizon_choices=env.horizon_choices,
        hidden_dims=list(pm_cfg.get("hidden_dims", [256, 128])),
        learning_rate=float(pm_cfg.get("learning_rate", 1e-4)),
        q_min=env.q_min,
        q_max=env.q_max,
        beta_floor=float(pm_cfg.get("beta_floor", 0.05)),
        kappa_min=float(pm_cfg.get("kappa_min", 2.0)),
        kappa_max=float(pm_cfg.get("kappa_max", 80.0)),
        aux_output_dim=int(pm_cfg.get("aux_output_dim", 0)),
    )
    relation_matrix = build_stock_relation_matrix(env, relation_cfg)
    group_cfg = trader_cfg.get("group_layer", {})
    ctde_cfg = w1.get("ctde", {})
    trader_critic_extra_dim = env.pm_obs_dim + 3 if bool(ctde_cfg.get("trader_central_critic", False)) else 0
    trader_type = str(trader_cfg.get("type", "dirichlet")).lower()
    if trader_type in {"latent_action", "latent_action_code", "latent_code"}:
        latent_cfg = dict(trader_cfg.get("latent_action", {}))
        prototypes, prototype_code_values = build_latent_action_prototypes(env, latent_cfg, fold_id=fold_id)
        trader = LatentActionTraderActorCritic(
            env.trader_obs_dim,
            stock_dim=env.stock_dim,
            stock_feature_dim=env.stock_feature_dim,
            prototype_weights=prototypes,
            prototype_code_values=prototype_code_values,
            task_dim=env.trader_task_dim,
            stock_hidden_dim=int(trader_cfg.get("stock_hidden_dim", 64)),
            critic_hidden_dims=list(trader_cfg.get("critic_hidden_dims", [256, 128])),
            learning_rate=float(trader_cfg.get("learning_rate", 1e-4)),
            alpha_min=float(trader_cfg.get("alpha_min", 0.05)),
            alpha_max=float(trader_cfg.get("alpha_max", 100.0)),
            residual_mix=float(latent_cfg.get("residual_mix", 0.10)),
            ticker_embedding_dim=int(trader_cfg.get("ticker_embedding_dim", 0)),
            group_ids=env.group_ids,
            relation_matrix=relation_matrix,
            graph_layers=int(relation_cfg.get("graph_layers", relation_cfg.get("layers", 0))),
            graph_use_group_context=bool(relation_cfg.get("use_group_context", True)),
            graph_residual_init_scale=float(relation_cfg.get("graph_residual_init_scale", relation_cfg.get("adapter_init_scale", 0.10))),
            two_channel=bool(latent_cfg.get("two_channel", False)),
            two_channel_cash_threshold=float(latent_cfg.get("two_channel_cash_threshold", 0.30)),
            two_channel_risk_threshold=float(latent_cfg.get("two_channel_risk_threshold", 0.15)),
            critic_extra_dim=trader_critic_extra_dim,
        )
    else:
        trader = BudgetTraderActorCritic(
            env.trader_obs_dim,
            stock_dim=env.stock_dim,
            stock_feature_dim=env.stock_feature_dim,
            task_dim=env.trader_task_dim,
            stock_hidden_dim=int(trader_cfg.get("stock_hidden_dim", 64)),
            critic_hidden_dims=list(trader_cfg.get("critic_hidden_dims", [256, 128])),
            learning_rate=float(trader_cfg.get("learning_rate", 1e-4)),
            alpha_min=float(trader_cfg.get("alpha_min", 0.05)),
            alpha_max=float(trader_cfg.get("alpha_max", 100.0)),
            ticker_embedding_dim=int(trader_cfg.get("ticker_embedding_dim", 0)),
            group_ids=env.group_ids,
            use_group_context=bool(group_cfg.get("enabled", False)),
            relation_matrix=relation_matrix,
            relation_adapter_mode=str(relation_cfg.get("adapter_mode", "concat")),
            relation_adapter_init_scale=float(relation_cfg.get("adapter_init_scale", 0.0)),
            critic_extra_dim=trader_critic_extra_dim,
            aux_per_stock_dim=int(trader_cfg.get("aux_per_stock_dim", 0)),
        )
    return pm, trader


def maybe_load_trader_warm_start(
    trader_policy: BudgetTraderActorCritic,
    variant: dict[str, Any],
    run_dir: Path | None = None,
    *,
    fold_id: str | None = None,
) -> dict[str, Any]:
    """Load a supervised Trader checkpoint before PPO, when requested.

    The checkpoint is actor-compatible with BudgetTraderActorCritic.  Optimizer
    state is intentionally not loaded: PPO starts with a fresh optimizer around
    pretrained actor parameters.
    """

    cfg = dict(variant.get("w1", {}).get("trader_warm_start", {}))
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False}
    raw_checkpoint_path = str(cfg.get("checkpoint_path", ""))
    checkpoint_path = resolve(raw_checkpoint_path.format(fold=fold_id or ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trader warm-start checkpoint not found: {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=trader_policy.device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    expected = {
        "obs_dim": trader_policy.obs_dim,
        "stock_dim": trader_policy.stock_dim,
        "stock_feature_dim": trader_policy.stock_feature_dim,
        "task_dim": trader_policy.task_dim,
    }
    checkpoint_dims = {key: checkpoint.get(key) for key in expected}
    mismatches = {
        key: {"expected": expected[key], "checkpoint": checkpoint_dims.get(key)}
        for key in expected
        if checkpoint_dims.get(key) is not None and int(checkpoint_dims[key]) != int(expected[key])
    }
    if mismatches and bool(cfg.get("strict_dims", True)):
        raise ValueError(f"Trader warm-start dimension mismatch: {mismatches}")
    strict = bool(cfg.get("strict_state_dict", True))
    state_dict, ignored_state_keys = filter_state_dict_by_regex_patterns(
        state_dict,
        [str(pattern) for pattern in cfg.get("ignore_state_keys", [])],
    )
    if ignored_state_keys:
        strict = False
    ignored_shape_mismatches: dict[str, dict[str, Any]] = {}
    if bool(cfg.get("ignore_mismatched_shapes", False)):
        current_state = trader_policy.state_dict()
        filtered_state = {}
        for key, value in state_dict.items():
            if key in current_state and tuple(current_state[key].shape) != tuple(value.shape):
                ignored_shape_mismatches[key] = {
                    "checkpoint": list(value.shape),
                    "current": list(current_state[key].shape),
                }
                continue
            filtered_state[key] = value
        state_dict = filtered_state
        strict = False
    load_result = trader_policy.load_state_dict(state_dict, strict=strict)
    cash_score_bias_shift = float(cfg.get("cash_score_bias_shift", 0.0))
    stock_score_bias_shift = float(cfg.get("stock_score_bias_shift", 0.0))
    with th.no_grad():
        if cash_score_bias_shift != 0.0:
            if not hasattr(trader_policy, "cash_score_head") or trader_policy.cash_score_head.bias is None:
                raise ValueError("cash_score_bias_shift requested, but trader policy has no cash_score_head.bias")
            trader_policy.cash_score_head.bias.add_(cash_score_bias_shift)
        if stock_score_bias_shift != 0.0:
            if not hasattr(trader_policy, "stock_score_head") or trader_policy.stock_score_head.bias is None:
                raise ValueError("stock_score_bias_shift requested, but trader policy has no stock_score_head.bias")
            trader_policy.stock_score_head.bias.add_(stock_score_bias_shift)
    info = {
        "enabled": True,
        "checkpoint_path": str(checkpoint_path),
        "strict_state_dict": strict,
        "strict_dims": bool(cfg.get("strict_dims", True)),
        "cash_score_bias_shift": cash_score_bias_shift,
        "stock_score_bias_shift": stock_score_bias_shift,
        "checkpoint_source": checkpoint.get("source", ""),
        "checkpoint_variant": checkpoint.get("variant", ""),
        "checkpoint_best_epoch": checkpoint.get("best_epoch", None),
        "expected_dims": expected,
        "checkpoint_dims": checkpoint_dims,
        "dimension_mismatches": mismatches,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "ignored_state_keys": ignored_state_keys,
        "ignored_shape_mismatches": ignored_shape_mismatches,
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trader_warm_start_loaded.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


def maybe_make_trader_behavior_prior(
    env: W1BudgetTraderEnv,
    variant: dict[str, Any],
    *,
    fold_id: str | None = None,
    run_dir: Path | None = None,
) -> tuple[BudgetTraderActorCritic | None, dict[str, Any]]:
    """Create a frozen Trader reference policy for PPO KL regularization."""

    cfg = dict(variant.get("w1", {}).get("trader_behavior_prior", {}))
    if not bool(cfg.get("enabled", False)):
        return None, {"enabled": False}
    coef = float(cfg.get("kl_coef", 0.0))
    if coef <= 0.0:
        return None, {"enabled": False, "reason": "kl_coef <= 0"}

    prior_variant = copy.deepcopy(variant)
    warm_cfg = dict(prior_variant.get("w1", {}).get("trader_warm_start", {}))
    if cfg.get("checkpoint_path"):
        warm_cfg["checkpoint_path"] = cfg["checkpoint_path"]
    warm_cfg["enabled"] = True
    warm_cfg["strict_dims"] = bool(cfg.get("strict_dims", warm_cfg.get("strict_dims", True)))
    warm_cfg["strict_state_dict"] = bool(cfg.get("strict_state_dict", warm_cfg.get("strict_state_dict", True)))
    warm_cfg["cash_score_bias_shift"] = float(cfg.get("cash_score_bias_shift", 0.0))
    warm_cfg["stock_score_bias_shift"] = float(cfg.get("stock_score_bias_shift", 0.0))
    prior_variant.setdefault("w1", {})["trader_warm_start"] = warm_cfg

    _pm_unused, prior = make_policies(env, prior_variant, fold_id=fold_id)
    load_info = maybe_load_trader_warm_start(prior, prior_variant, run_dir=None)
    prior.eval()
    for param in prior.parameters():
        param.requires_grad_(False)
    info = {
        "enabled": True,
        "kl_coef": coef,
        "checkpoint_path": load_info.get("checkpoint_path", ""),
        "checkpoint_source": load_info.get("checkpoint_source", ""),
        "checkpoint_variant": load_info.get("checkpoint_variant", ""),
        "cash_score_bias_shift": warm_cfg.get("cash_score_bias_shift", 0.0),
        "stock_score_bias_shift": warm_cfg.get("stock_score_bias_shift", 0.0),
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trader_behavior_prior_loaded.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return prior, info


def maybe_load_pm_warm_start(
    pm_policy: BudgetPMActorCritic,
    variant: dict[str, Any],
    *,
    fold_id: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Load a supervised PM checkpoint before PPO, when requested."""

    cfg = dict(variant.get("w1", {}).get("pm_warm_start", {}))
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False}
    raw_path = str(cfg.get("checkpoint_path", ""))
    checkpoint_path = resolve(raw_path.format(fold=fold_id))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PM warm-start checkpoint not found: {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=pm_policy.device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    expected_obs_dim = int(pm_policy.obs_dim)
    checkpoint_obs_dim = checkpoint.get("obs_dim")
    expected_horizons = [int(x) for x in pm_policy.horizon_choices]
    checkpoint_horizons = checkpoint.get("horizon_choices")
    mismatches: dict[str, Any] = {}
    if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != expected_obs_dim:
        mismatches["obs_dim"] = {"expected": expected_obs_dim, "checkpoint": int(checkpoint_obs_dim)}
    if checkpoint_horizons is not None and [int(x) for x in checkpoint_horizons] != expected_horizons:
        mismatches["horizon_choices"] = {"expected": expected_horizons, "checkpoint": checkpoint_horizons}
    if mismatches and bool(cfg.get("strict_dims", True)):
        raise ValueError(f"PM warm-start dimension mismatch: {mismatches}")
    strict = bool(cfg.get("strict_state_dict", True))
    load_result = pm_policy.load_state_dict(state_dict, strict=strict)
    info = {
        "enabled": True,
        "checkpoint_path": str(checkpoint_path),
        "strict_state_dict": strict,
        "strict_dims": bool(cfg.get("strict_dims", True)),
        "checkpoint_source": checkpoint.get("source", ""),
        "checkpoint_variant": checkpoint.get("variant", ""),
        "checkpoint_fold": checkpoint.get("fold", ""),
        "checkpoint_best_epoch": checkpoint.get("best_epoch", None),
        "expected_obs_dim": expected_obs_dim,
        "checkpoint_obs_dim": checkpoint_obs_dim,
        "expected_horizon_choices": expected_horizons,
        "checkpoint_horizon_choices": checkpoint_horizons,
        "dimension_mismatches": mismatches,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "pm_warm_start_loaded.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


def collect_episode(env: W1BudgetTraderEnv, pm_policy: Any, trader_policy: Any) -> tuple[PolicyBatch, PolicyBatch, pd.DataFrame]:
    env.reset()
    pm_obs_rows: list[np.ndarray] = []
    pm_actions: list[np.ndarray] = []
    pm_log_probs: list[float] = []
    pm_values: list[float] = []
    pm_rewards: list[float] = []
    pm_dones: list[float] = []

    trader_obs_rows: list[np.ndarray] = []
    trader_actions: list[np.ndarray] = []
    trader_log_probs: list[float] = []
    trader_values: list[float] = []
    trader_rewards: list[float] = []
    trader_dones: list[float] = []
    trader_critic_obs_rows: list[np.ndarray] = []

    rows: list[dict[str, Any]] = []
    pm_open = False
    pm_start_day = 0
    pm_horizon_days = 1
    q_target = 0.0
    open_pm_infos: list[dict[str, Any]] = []

    while not env.done():
        elapsed = int(env.day) - int(pm_start_day)
        if (not pm_open) or elapsed >= pm_horizon_days:
            if pm_open:
                pm_rewards.append(float(env.pm_window_reward_from_infos(open_pm_infos, q_target=q_raw_target)))
                pm_dones.append(0.0)
            obs = env.pm_obs()
            action, value, log_prob = sample_policy(pm_policy, obs, deterministic=False)
            q_raw_target = float(np.clip(action[0], env.q_min, env.q_max))
            pm_horizon_days = env.horizon_from_action(action[1])
            pm_start_day = int(env.day)
            open_pm_infos = []
            pm_open = True
            pm_obs_rows.append(obs)
            pm_actions.append(np.asarray([q_raw_target, action[1]], dtype=np.float32))
            pm_values.append(value)
            pm_log_probs.append(log_prob)

        remaining = max(1, pm_horizon_days - (int(env.day) - int(pm_start_day)))
        q_target, risk_stop_info = env.apply_risk_stop(q_raw_target)
        trader_obs = env.trader_obs(q_target=q_target, remaining_days=remaining)
        critic_obs = None
        if getattr(trader_policy, "critic_extra_dim", 0) > 0:
            critic_obs = trader_central_critic_obs(
                env,
                env.pm_obs(),
                q_target=q_target,
                horizon_days=pm_horizon_days,
                remaining_days=remaining,
            )
        trader_action, trader_value, trader_log_prob = sample_policy(
            trader_policy,
            trader_obs,
            deterministic=False,
            critic_obs=critic_obs,
        )
        executable_trader_action = decode_policy_actions(trader_policy, trader_obs, trader_action)
        trader_diag = policy_action_diagnostics(trader_policy, trader_obs, trader_action)
        info = env.step_trader(
            q_target=q_target,
            remaining_days=remaining,
            trader_action=executable_trader_action,
            q_raw_target=q_raw_target,
            risk_stop_info=risk_stop_info,
            execution_context=trader_diag,
        )
        done = env.done()
        open_pm_infos.append(info)
        trader_obs_rows.append(trader_obs)
        trader_actions.append(trader_action.astype(np.float32))
        trader_values.append(trader_value)
        trader_log_probs.append(trader_log_prob)
        trader_rewards.append(float(info["trader_reward"]))
        trader_dones.append(1.0 if done else 0.0)
        if critic_obs is not None:
            trader_critic_obs_rows.append(critic_obs.astype(np.float32))
        rows.append(
            _daily_row(
                env,
                info,
                pm_horizon_days=pm_horizon_days,
                pm_start_day=pm_start_day,
                trader_action=executable_trader_action,
                trader_action_diagnostics=trader_diag,
            )
        )

    if pm_open:
        pm_rewards.append(float(env.pm_window_reward_from_infos(open_pm_infos, q_target=q_raw_target)))
        pm_dones.append(1.0)

    pm_batch = PolicyBatch(
        obs=np.asarray(pm_obs_rows, dtype=np.float32),
        actions=np.asarray(pm_actions, dtype=np.float32),
        old_log_prob=np.asarray(pm_log_probs, dtype=np.float32),
        values=np.asarray(pm_values, dtype=np.float32),
        rewards=np.asarray(pm_rewards, dtype=np.float32),
        dones=np.asarray(pm_dones, dtype=np.float32),
    )
    trader_batch = PolicyBatch(
        obs=np.asarray(trader_obs_rows, dtype=np.float32),
        actions=np.asarray(trader_actions, dtype=np.float32),
        old_log_prob=np.asarray(trader_log_probs, dtype=np.float32),
        values=np.asarray(trader_values, dtype=np.float32),
        rewards=np.asarray(trader_rewards, dtype=np.float32),
        dones=np.asarray(trader_dones, dtype=np.float32),
        critic_obs=np.asarray(trader_critic_obs_rows, dtype=np.float32) if trader_critic_obs_rows else None,
    )
    return pm_batch, trader_batch, pd.DataFrame(rows)


def _daily_row(
    env: W1BudgetTraderEnv,
    info: dict[str, Any],
    *,
    pm_horizon_days: int,
    pm_start_day: int,
    trader_action: np.ndarray,
    trader_action_diagnostics: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    target = np.asarray(info.get("target_weights", np.zeros(env.asset_dim)), dtype=np.float64)
    desired = np.asarray(info.get("desired_weights", target), dtype=np.float64)
    pre = np.asarray(info.get("pre_trade_weights", np.zeros(env.asset_dim)), dtype=np.float64)
    row: dict[str, Any] = {
        "date": info.get("date", ""),
        "next_date": info.get("next_date", ""),
        "net_return": info.get("net_return", 0.0),
        "gross_return": info.get("gross_return", 0.0),
        "benchmark_return": info.get("benchmark_return", 0.0),
        "pm_reward": info.get("pm_reward", 0.0),
        "trader_reward": info.get("trader_reward", 0.0),
        "q_raw_target": info.get("q_raw_target", info.get("q_target", np.nan)),
        "q_target": info.get("q_target", np.nan),
        "q_exec": info.get("q_exec", np.nan),
        "desired_q": info.get("desired_q", info.get("execution_desired_q", np.nan)),
        "cash_target_pm": info.get("cash_target_pm", np.nan),
        "cash_exec": info.get("cash_exec", np.nan),
        "desired_cash": info.get("desired_cash", info.get("execution_desired_cash", np.nan)),
        "tracking_l1": info.get("tracking_l1", np.nan),
        "remaining_days": info.get("remaining_days", np.nan),
        "remaining_days_normalized": info.get("remaining_days_normalized", np.nan),
        "pm_horizon_days": int(pm_horizon_days),
        "pm_start_day": int(pm_start_day),
        "turnover_l1": info.get("turnover_l1", 0.0),
        "stock_turnover_l1": info.get("stock_turnover_l1", 0.0),
        "transaction_cost": info.get("transaction_cost", 0.0),
        "drawdown": info.get("drawdown", 0.0),
        "drawdown_increment": info.get("drawdown_increment", 0.0),
        "risky_entropy": info.get("risky_entropy", np.nan),
        "portfolio_entropy": info.get("portfolio_entropy", np.nan),
        "trader_flow_select_5d": info.get("trader_flow_select_5d", np.nan),
        "trader_group_relative_5d": info.get("trader_group_relative_5d", np.nan),
        "trader_entropy_bonus": info.get("trader_entropy_bonus", np.nan),
        "trader_entropy_reward_term": info.get("trader_entropy_reward_term", np.nan),
        "trader_vol_adjusted_position_change": info.get("trader_vol_adjusted_position_change", np.nan),
        "trader_tracking_multiplier": info.get("trader_tracking_multiplier", np.nan),
        "trader_tracking_penalty_term": info.get("trader_tracking_penalty_term", np.nan),
        "pm_opportunity_gate": info.get("pm_opportunity_gate", np.nan),
        "pm_cash_opportunity_cost": info.get("pm_cash_opportunity_cost", np.nan),
        "pm_cash_opportunity_prior_cost": info.get("pm_cash_opportunity_prior_cost", np.nan),
        "pm_active_return": info.get("pm_active_return", np.nan),
        "trader_action_cash_weight": float(np.asarray(trader_action).reshape(-1)[env.cash_index]),
        "trader_feedback_change_after_step": float(env.trader_reward_change),
    }
    for key, values in (trader_action_diagnostics or {}).items():
        arr = np.asarray(values).reshape(-1)
        if len(arr):
            row[key] = float(arr[0]) if np.issubdtype(arr.dtype, np.floating) else int(arr[0])
    for key, value in info.items():
        if str(key).startswith("risk_stop_") or str(key).startswith("supervisor_") or str(key).startswith("execution_"):
            row[key] = float(value)
    for ticker, value in zip(env.panel.tickers, desired[: env.stock_dim]):
        row[f"desired_weight_{ticker}"] = float(value)
    row["desired_weight_CASH"] = float(desired[env.cash_index])
    for ticker, value in zip(env.panel.tickers, target[: env.stock_dim]):
        row[f"target_weight_{ticker}"] = float(value)
    row["target_weight_CASH"] = float(target[env.cash_index])
    for ticker, value in zip(env.panel.tickers, pre[: env.stock_dim]):
        row[f"pre_weight_{ticker}"] = float(value)
    row["pre_weight_CASH"] = float(pre[env.cash_index])
    return row


def evaluate(
    env: W1BudgetTraderEnv,
    pm_policy: Any,
    trader_policy: Any,
    *,
    out_dir: Path,
    split_name: str,
    action_supervisor: Any | None = None,
) -> dict[str, Any]:
    env.reset()
    rows: list[dict[str, Any]] = []
    pm_open = False
    pm_start_day = 0
    pm_horizon_days = 1
    q_target = 0.0
    while not env.done():
        elapsed = int(env.day) - int(pm_start_day)
        if (not pm_open) or elapsed >= pm_horizon_days:
            action, _value, _log_prob = sample_policy(pm_policy, env.pm_obs(), deterministic=True)
            q_raw_target = float(np.clip(action[0], env.q_min, env.q_max))
            pm_horizon_days = env.horizon_from_action(action[1])
            pm_start_day = int(env.day)
            pm_open = True
        remaining = max(1, pm_horizon_days - (int(env.day) - int(pm_start_day)))
        q_target, risk_stop_info = env.apply_risk_stop(q_raw_target)
        obs = env.trader_obs(q_target=q_target, remaining_days=remaining)
        trader_action, _value, _log_prob = sample_policy(trader_policy, obs, deterministic=True)
        supervisor_info: dict[str, Any] = {}
        if action_supervisor is not None and getattr(action_supervisor, "enabled", False):
            action_batch, supervisor_batch_info = action_supervisor.apply(
                panel=env.panel,
                day=int(env.day),
                actions=trader_action.reshape(1, -1),
            )
            trader_action = np.asarray(action_batch, dtype=np.float32).reshape(-1)
            supervisor_info = {
                key: float(np.asarray(value).reshape(-1)[0])
                for key, value in supervisor_batch_info.items()
                if np.asarray(value).size
            }
        executable_trader_action = decode_policy_actions(trader_policy, obs, trader_action)
        trader_diag = policy_action_diagnostics(trader_policy, obs, trader_action)
        info = env.step_trader(
            q_target=q_target,
            remaining_days=remaining,
            trader_action=executable_trader_action,
            q_raw_target=q_raw_target,
            risk_stop_info=risk_stop_info,
            execution_context=trader_diag,
        )
        info.update(supervisor_info)
        rows.append(
            _daily_row(
                env,
                info,
                pm_horizon_days=pm_horizon_days,
                pm_start_day=pm_start_day,
                trader_action=executable_trader_action,
                trader_action_diagnostics=trader_diag,
            )
        )
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
    pm_policy, trader_policy = make_policies(train_env, variant, fold_id=fold_id)
    pm_warm_start_info = maybe_load_pm_warm_start(pm_policy, variant, fold_id=fold_id, run_dir=run_dir)
    warm_start_info = maybe_load_trader_warm_start(trader_policy, variant, run_dir=run_dir, fold_id=fold_id)

    joint_cfg = variant.get("w1", {}).get("joint_ppo", {})
    target_days = int(joint_cfg.get("total_internal_trading_days", 70_000))
    if smoke_test:
        target_days = min(target_days, 2048)
    n_epochs = int(joint_cfg.get("n_epochs", 4 if not smoke_test else 1))
    clip_range = float(joint_cfg.get("clip_range", config.get("ppo", {}).get("clip_range", 0.1)))
    gamma_pm = float(joint_cfg.get("pm_gamma", config.get("ppo", {}).get("gamma", 0.99)))
    gamma_trader = float(joint_cfg.get("trader_gamma", config.get("ppo", {}).get("gamma", 0.99)))
    gae_lambda = float(joint_cfg.get("gae_lambda", config.get("ppo", {}).get("gae_lambda", 0.95)))
    max_grad_norm = float(joint_cfg.get("max_grad_norm", config.get("ppo", {}).get("max_grad_norm", 0.5)))
    pm_batch_size = int(joint_cfg.get("pm_batch_size", 64))
    trader_batch_size = int(joint_cfg.get("trader_batch_size", 512))
    pm_ent_coef = float(joint_cfg.get("pm_ent_coef", 0.0))
    trader_ent_coef = float(joint_cfg.get("trader_ent_coef", 0.0))
    vf_coef = float(joint_cfg.get("vf_coef", config.get("ppo", {}).get("vf_coef", 0.5)))

    processed_days = 0
    iteration_rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    iteration = 0
    while processed_days < target_days:
        iteration += 1
        pm_batch, trader_batch, trace = collect_episode(train_env, pm_policy, trader_policy)
        pm_adv, pm_returns = compute_gae(pm_batch, gamma=gamma_pm, gae_lambda=gae_lambda)
        trader_adv, trader_returns = compute_gae(trader_batch, gamma=gamma_trader, gae_lambda=gae_lambda)
        pm_stats = ppo_update(
            policy=pm_policy,
            batch=pm_batch,
            advantages=pm_adv,
            returns=pm_returns,
            n_epochs=n_epochs,
            batch_size=pm_batch_size,
            clip_range=clip_range,
            ent_coef=pm_ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )
        trader_stats = ppo_update(
            policy=trader_policy,
            batch=trader_batch,
            advantages=trader_adv,
            returns=trader_returns,
            n_epochs=n_epochs,
            batch_size=trader_batch_size,
            clip_range=clip_range,
            ent_coef=trader_ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )
        processed_days += len(trader_batch.rewards)
        returns = trace["net_return"].to_numpy(dtype=np.float64) if len(trace) else np.zeros(0)
        iteration_rows.append(
            {
                "iteration": iteration,
                "processed_internal_trading_days": processed_days,
                "episode_days": int(len(trace)),
                "episode_return_pct": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
                "episode_cash_exec_mean": float(trace["cash_exec"].mean()) if len(trace) else 0.0,
                "episode_q_target_mean": float(trace["q_target"].mean()) if len(trace) else 0.0,
                "episode_q_exec_mean": float(trace["q_exec"].mean()) if len(trace) else 0.0,
                "episode_tracking_l1_mean": float(trace["tracking_l1"].mean()) if len(trace) else 0.0,
                "episode_horizon_mean": float(trace["pm_horizon_days"].mean()) if len(trace) else 0.0,
                "pm_transitions": int(len(pm_batch.rewards)),
                "trader_transitions": int(len(trader_batch.rewards)),
                **{f"pm_{k}": v for k, v in pm_stats.items()},
                **{f"trader_{k}": v for k, v in trader_stats.items()},
            }
        )
        if iteration <= int(joint_cfg.get("save_train_trace_episodes", 1)):
            trace = trace.copy()
            trace["iteration"] = iteration
            trace_frames.append(trace)
        print(
            f"[W1] iter={iteration} days={processed_days}/{target_days} "
            f"pm_kl={pm_stats.get('approx_kl', np.nan):.5f} "
            f"trader_kl={trader_stats.get('approx_kl', np.nan):.5f}",
            flush=True,
        )

    pd.DataFrame(iteration_rows).to_csv(run_dir / "train_joint_iterations.csv", index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(run_dir / "train_trace_daily.csv", index=False)
    th.save({"state_dict": pm_policy.state_dict(), "variant": variant}, run_dir / "pm_policy.pt")
    th.save({"state_dict": trader_policy.state_dict(), "variant": variant}, run_dir / "trader_policy.pt")

    val_env = W1BudgetTraderEnv(validation_panel, **env_kwargs(config, variant, validation_panel))
    validation_summary = evaluate(val_env, pm_policy, trader_policy, out_dir=run_dir, split_name="validation")
    metadata = {
        "variant": variant,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "target_internal_trading_days": target_days,
        "observed_internal_trading_days": processed_days,
        "pm_obs_dim": train_env.pm_obs_dim,
        "trader_obs_dim": train_env.trader_obs_dim,
        "trader_action_semantics": "full_portfolio_dirichlet_stocks_plus_cash",
        "stock_feature_dim": train_env.stock_feature_dim,
        "trader_task_dim": train_env.trader_task_dim,
        "horizon_choices": train_env.horizon_choices,
        "pm_warm_start": pm_warm_start_info,
        "trader_warm_start": warm_start_info,
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
    parser.add_argument("--config", default="configs/generated/stage0_1_w1_budget_trader.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or config.get("output", {}).get("run_name", "weight_based_w1_budget_trader")
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== W1 budget trader: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nW1 run written to {out_root}")


if __name__ == "__main__":
    main()
