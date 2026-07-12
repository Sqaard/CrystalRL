"""Vectorized W1 budget-PM / Trader trainer.

This runner keeps the W1 policy/reward contract intact while collecting several
independent train rollouts in one synchronous batch.  It is the practical
foundation for a future Sample Factory backend: the expensive market/env math is
batched, and the actor calls already operate on batch tensors.

It deliberately reuses:

* W1BudgetTraderEnv for dimensions/configuration and deterministic validation.
* BudgetPMActorCritic / BudgetTraderActorCritic for the policy contract.
* The same PPO update code as the sequential W1 runner.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml

from src.ppo.execution.budgeted_flow import apply_budgeted_flow_execution_batch
from src.ppo.stage0_1_w1_budget_trader_train import (
    PolicyBatch,
    compute_gae,
    decode_policy_actions,
    env_kwargs,
    evaluate,
    load_folds,
    make_policies,
    maybe_load_pm_warm_start,
    maybe_make_trader_behavior_prior,
    maybe_load_trader_warm_start,
    policy_action_diagnostics,
    ppo_update,
    selected_variants,
)
from src.ppo.w1_config_utils import feature_csv_for_fold
from src.ppo.weight_panel import load_weight_panel
from src.ppo.w1_budget_trader_env import W1BudgetTraderEnv, entropy_simplex
from src.ppo.execution.helpers import sigmoid_scalar
from src.ppo.w1_risk_stop import compute_risk_stop, features_from_panel_row


ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-8


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_teacher_csv_with_optional_columns(
    path: Path,
    requested_columns: list[str],
    required_columns: set[str],
) -> pd.DataFrame:
    """Read a teacher trajectory CSV while tolerating optional metadata fields."""

    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in requested_columns if col in available_columns]
    missing_required = sorted(required_columns.difference(usecols))
    if missing_required:
        raise ValueError(f"Teacher file is missing required columns {missing_required}: {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def nonnegative_optional_series(df: pd.DataFrame, column: str, *, absolute: bool = False) -> pd.Series:
    """Return a non-negative numeric Series, or zeros when an optional column is absent."""

    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if absolute:
        return values.abs()
    return values.clip(lower=0.0)


def as_tensor(array: np.ndarray, *, device: th.device | str) -> th.Tensor:
    return th.as_tensor(array, dtype=th.float32, device=device)


def normalize_simplex_batch(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    denom = np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    out = weights / denom
    bad = ~np.isfinite(out).all(axis=1)
    if np.any(bad):
        out[bad] = 1.0 / weights.shape[1]
    return out


def entropy_simplex_batch(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=np.float64), EPS)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), EPS)
    return -np.sum(w * np.log(w), axis=1)


def sample_policy_batch(
    policy: Any,
    obs: np.ndarray,
    *,
    deterministic: bool = False,
    critic_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_t = as_tensor(obs, device=policy.device)
    critic_t = as_tensor(critic_obs, device=policy.device) if critic_obs is not None else None
    with th.no_grad():
        action_t, value_t, log_prob_t = policy(obs_t, deterministic=deterministic, critic_obs=critic_t)
    return (
        action_t.detach().cpu().numpy().astype(np.float32),
        value_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
        log_prob_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
    )


def evaluate_policy_action_batch(
    policy: Any,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    critic_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    obs_t = as_tensor(obs, device=policy.device)
    actions_t = as_tensor(actions, device=policy.device)
    critic_t = as_tensor(critic_obs, device=policy.device) if critic_obs is not None else None
    with th.no_grad():
        value_t, log_prob_t, _entropy_t = policy.evaluate_actions(obs_t, actions_t, critic_obs=critic_t)
    return (
        value_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
        log_prob_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
    )


@dataclass
class OpenPMState:
    obs: np.ndarray
    action: np.ndarray
    value: np.ndarray
    log_prob: np.ndarray
    active: np.ndarray
    q_raw_target: np.ndarray
    q_target: np.ndarray
    horizon_days: np.ndarray
    start_day: np.ndarray
    return_factor: np.ndarray
    benchmark_factor: np.ndarray
    drawdown_sum: np.ndarray
    opportunity_sum: np.ndarray
    window_len: np.ndarray
    aux_target: np.ndarray
    rare_weight_sum: np.ndarray
    rare_weight_max: np.ndarray
    rare_weight_len: np.ndarray


class ClosedLoopActionSupervisor:
    """Online train-time supervisor for latent-action Trader primitives.

    It uses only current feature-derived market context at rollout time.  The
    context -> target action-code table is estimated from train-split corrected
    teacher trajectories before PPO starts.
    """

    CONTEXTS = ("stress", "recovery", "bull_trend", "calm_hold", "choppy_rotation")

    def __init__(
        self,
        *,
        config: dict[str, Any],
        fold_id: str,
        panel: Any,
        trader_policy: Any,
        seed: int,
    ) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.fold_id = str(fold_id)
        self.rng = np.random.default_rng(int(seed))
        self.code_values = np.asarray(
            getattr(trader_policy, "prototype_code_values", np.arange(getattr(trader_policy, "num_codes", 0)))
            .detach()
            .cpu()
            .numpy()
            if hasattr(getattr(trader_policy, "prototype_code_values", None), "detach")
            else getattr(trader_policy, "prototype_code_values", np.arange(getattr(trader_policy, "num_codes", 0))),
            dtype=np.int64,
        ).reshape(-1)
        self.code_to_index = {int(code): int(idx) for idx, code in enumerate(self.code_values)}
        self.context_to_code: dict[str, int] = {}
        self.context_to_index: dict[str, int] = {}
        self.context_confidence: dict[str, float] = {}
        self.context_score_sum: dict[str, float] = {}
        self.global_code: int | None = None
        self.global_index: int | None = None
        self.last_intervention_day: np.ndarray | None = None
        self.summary: dict[str, Any] = {"enabled": self.enabled}
        if self.enabled:
            if not hasattr(trader_policy, "prototype_code_values"):
                raise ValueError("closed_loop_supervisor requires a latent-action Trader with prototype_code_values.")
            self._build_context_code_table()
        self.feature_index = {name: idx for idx, name in enumerate(panel.feature_columns)}

    def _build_context_code_table(self) -> None:
        teacher_path = resolve(
            self.config.get(
                "teacher_path",
                "artifacts/pretrain/P14_p12_world_model_augmented_trajectories_v1/p13_asset_corrected_teacher_targets_long.csv",
            )
        )
        if not teacher_path.exists():
            raise FileNotFoundError(f"closed_loop_supervisor teacher_path not found: {teacher_path}")
        requested_usecols = [
            "fold",
            "split",
            "valid",
            "market_context",
            "corrected_action_code",
            "correction_weight",
            "corrected_teacher_sample_weight",
            "corrected_teacher_trade_abs_weight",
            "trajectory_source",
        ]
        df = read_teacher_csv_with_optional_columns(
            teacher_path,
            requested_usecols,
            {"fold", "split", "market_context", "corrected_action_code"},
        )
        df = df[df["fold"].astype(str).eq(self.fold_id) & df["split"].astype(str).eq("train")].copy()
        if "valid" in df.columns:
            df = df[df["valid"].fillna(True).astype(bool)]
        if not bool(self.config.get("include_synthetic", False)) and "trajectory_source" in df.columns:
            df = df[df["trajectory_source"].astype(str).eq("real_chrl_p13")]
        df["corrected_action_code"] = pd.to_numeric(df["corrected_action_code"], errors="coerce")
        df = df[df["corrected_action_code"].notna()].copy()
        if df.empty:
            raise ValueError(f"closed_loop_supervisor has no train teacher rows for {self.fold_id}")
        base = float(self.config.get("base_weight", 0.25))
        sample_scale = float(self.config.get("sample_weight_scale", 1.0))
        correction_scale = float(self.config.get("correction_weight_scale", 1.0))
        trade_scale = float(self.config.get("trade_abs_weight_scale", 0.5))
        sample_w = nonnegative_optional_series(df, "corrected_teacher_sample_weight")
        correction_w = nonnegative_optional_series(df, "correction_weight")
        trade_w = nonnegative_optional_series(df, "corrected_teacher_trade_abs_weight", absolute=True)
        df["_supervisor_score"] = base + sample_scale * sample_w + correction_scale * correction_w + trade_scale * trade_w
        scores = (
            df.groupby(["market_context", "corrected_action_code"], dropna=True)["_supervisor_score"]
            .sum()
            .reset_index()
        )
        mapped_rows: list[dict[str, Any]] = []
        for context, group in scores.groupby("market_context"):
            total = float(group["_supervisor_score"].sum())
            top = group.sort_values("_supervisor_score", ascending=False).iloc[0]
            code = int(top["corrected_action_code"])
            if code not in self.code_to_index:
                continue
            self.context_to_code[str(context)] = code
            self.context_to_index[str(context)] = self.code_to_index[code]
            self.context_confidence[str(context)] = float(top["_supervisor_score"] / max(total, EPS))
            self.context_score_sum[str(context)] = total
            mapped_rows.append(
                {
                    "market_context": str(context),
                    "target_code": code,
                    "target_code_index": self.code_to_index[code],
                    "confidence": self.context_confidence[str(context)],
                    "score_sum": total,
                }
            )
        global_scores = df.groupby("corrected_action_code")["_supervisor_score"].sum().sort_values(ascending=False)
        for code_f in global_scores.index:
            code = int(code_f)
            if code in self.code_to_index:
                self.global_code = code
                self.global_index = self.code_to_index[code]
                break
        self.summary = {
            "enabled": True,
            "teacher_path": str(teacher_path),
            "fold": self.fold_id,
            "rows": int(len(df)),
            "include_synthetic": bool(self.config.get("include_synthetic", False)),
            "target_table": mapped_rows,
            "global_code": self.global_code,
            "global_index": self.global_index,
            "replace_probability": float(self.config.get("replace_probability", 0.35)),
            "max_intervention_rate": float(self.config.get("max_intervention_rate", 0.40)),
        }

    def _feature(self, panel: Any, day: int, name: str, default: float = 0.0) -> float:
        idx = self.feature_index.get(str(name))
        if idx is None:
            return float(default)
        value = float(panel.features[int(day), 0, idx])
        return value if np.isfinite(value) else float(default)

    def infer_context(self, panel: Any, day: int) -> tuple[str, dict[str, float]]:
        trend = self._feature(panel, day, str(self.config.get("trend_feature", "SP500_Trend")))
        trend_delta = self._feature(panel, day, str(self.config.get("trend_delta_feature", "SP500_Trend_delta_5d")))
        vix_5d = self._feature(panel, day, str(self.config.get("vix_change_feature", "VIX_change_5d")))
        turbulence = self._feature(panel, day, str(self.config.get("turbulence_feature", "turbulence")))
        residual = self._feature(panel, day, str(self.config.get("residual_feature", "residual_universe_return_20d")))
        breadth = self._feature(panel, day, str(self.config.get("breadth_feature", "residual_breadth_20d")), default=0.5)
        stress_score = max(
            -trend / max(abs(float(self.config.get("stress_trend_threshold", -0.65))), EPS),
            vix_5d / max(float(self.config.get("stress_vix_change_threshold", 0.75)), EPS),
            turbulence / max(float(self.config.get("stress_turbulence_threshold", 0.75)), EPS),
            0.0,
        )
        recovery_score = max(
            trend_delta / max(float(self.config.get("recovery_trend_delta_threshold", 0.40)), EPS),
            residual / max(float(self.config.get("recovery_residual_threshold", 0.35)), EPS),
            (breadth - 0.5) / max(float(self.config.get("recovery_breadth_excess_threshold", 0.12)), EPS),
            0.0,
        )
        bull_score = max(trend / max(float(self.config.get("bull_trend_threshold", 0.55)), EPS), 0.0)
        calm_score = max(
            1.0 - abs(trend) / max(float(self.config.get("calm_abs_trend_threshold", 0.50)), EPS),
            0.0,
        ) * max(1.0 - max(vix_5d, 0.0) / max(float(self.config.get("calm_vix_change_threshold", 0.50)), EPS), 0.0)
        if stress_score >= float(self.config.get("stress_score_gate", 1.0)):
            context = "stress"
        elif recovery_score >= float(self.config.get("recovery_score_gate", 1.0)):
            context = "recovery"
        elif bull_score >= float(self.config.get("bull_score_gate", 1.0)):
            context = "bull_trend"
        elif calm_score >= float(self.config.get("calm_score_gate", 0.50)):
            context = "calm_hold"
        else:
            context = "choppy_rotation"
        return context, {
            "stress_score": float(stress_score),
            "recovery_score": float(recovery_score),
            "bull_score": float(bull_score),
            "calm_score": float(calm_score),
            "trend": float(trend),
            "trend_delta": float(trend_delta),
            "vix_change_5d": float(vix_5d),
            "turbulence": float(turbulence),
            "residual": float(residual),
            "breadth": float(breadth),
        }

    def apply(self, *, panel: Any, day: int, actions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        action = np.asarray(actions, dtype=np.float32).copy()
        n = action.shape[0]
        zeros = np.zeros(n, dtype=np.float64)
        if not self.enabled or self.global_index is None:
            return action, {
                "supervisor_enabled": zeros,
                "supervisor_intervened": zeros,
                "supervisor_context_id": np.full(n, -1.0, dtype=np.float64),
                "supervisor_target_code": np.full(n, np.nan, dtype=np.float64),
                "supervisor_original_code": np.full(n, np.nan, dtype=np.float64),
                "supervisor_confidence": zeros,
            }
        context, scores = self.infer_context(panel, day)
        target_idx = self.context_to_index.get(context, self.global_index)
        target_code = self.context_to_code.get(context, self.global_code)
        confidence = float(self.context_confidence.get(context, 0.0))
        min_conf = float(self.config.get("min_context_confidence", 0.12))
        replace_prob = float(np.clip(self.config.get("replace_probability", 0.35), 0.0, 1.0))
        max_rate = float(np.clip(self.config.get("max_intervention_rate", 0.40), 0.0, 1.0))
        cooldown_days = int(max(0, self.config.get("cooldown_days", 0)))
        confidence_power = float(max(0.0, self.config.get("confidence_power", 0.0)))
        if confidence_power > 0.0:
            replace_prob *= float(np.clip(confidence, 0.0, 1.0) ** confidence_power)
        if self.last_intervention_day is None or len(self.last_intervention_day) != n:
            self.last_intervention_day = np.full(n, -1_000_000, dtype=np.int32)
        original_idx = np.clip(np.rint(action[:, 0]).astype(np.int64), 0, len(self.code_values) - 1)
        original_code = self.code_values[original_idx].astype(np.float64)
        eligible = (original_idx != int(target_idx)) & (confidence >= min_conf)
        if cooldown_days > 0:
            eligible &= (int(day) - self.last_intervention_day) >= cooldown_days
        random_gate = self.rng.random(n) < replace_prob
        intervene = eligible & random_gate
        max_n = int(np.floor(max_rate * n))
        if max_rate > 0.0 and n > 0 and max_n == 0:
            max_n = 1
        if intervene.sum() > max_n:
            selected = self.rng.choice(np.where(intervene)[0], size=max_n, replace=False) if max_n > 0 else np.asarray([], dtype=int)
            capped = np.zeros(n, dtype=bool)
            capped[selected] = True
            intervene = capped
        action[intervene, 0] = float(target_idx)
        self.last_intervention_day[intervene] = int(day)
        context_id = float(self.CONTEXTS.index(context)) if context in self.CONTEXTS else -1.0
        return action, {
            "supervisor_enabled": np.ones(n, dtype=np.float64),
            "supervisor_intervened": intervene.astype(np.float64),
            "supervisor_context_id": np.full(n, context_id, dtype=np.float64),
            "supervisor_target_code": np.full(n, float(target_code if target_code is not None else np.nan), dtype=np.float64),
            "supervisor_target_code_index": np.full(n, float(target_idx), dtype=np.float64),
            "supervisor_original_code": original_code,
            "supervisor_original_code_index": original_idx.astype(np.float64),
            "supervisor_confidence": np.full(n, confidence, dtype=np.float64),
            "supervisor_cooldown_days": np.full(n, float(cooldown_days), dtype=np.float64),
            "supervisor_replace_probability_effective": np.full(n, float(replace_prob), dtype=np.float64),
            **{f"supervisor_{k}": np.full(n, v, dtype=np.float64) for k, v in scores.items()},
        }


@dataclass
class ReplaySample:
    batch: PolicyBatch
    advantages: np.ndarray
    returns: np.ndarray
    indices: np.ndarray
    probabilities: np.ndarray
    is_weights: np.ndarray


class PrioritizedPolicyReplay:
    """Fixed-size prioritized replay for W1 PM/Trader policy transitions.

    Priorities are TD-target errors, initialized from |return - old_value| and
    refreshed after replay updates from the current value head.  This keeps the
    PPO rollout contract but adds a real off-policy replay source instead of
    only weighting the latest on-policy batch.
    """

    def __init__(
        self,
        *,
        capacity: int,
        alpha: float = 0.60,
        priority_eps: float = 1e-4,
        max_priority: float = 100.0,
        seed: int = 0,
    ) -> None:
        self.capacity = int(max(1, capacity))
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.priority_eps = float(max(priority_eps, EPS))
        self.max_priority = float(max(max_priority, self.priority_eps))
        self.rng = np.random.default_rng(int(seed))
        self.obs: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.old_log_prob: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.rewards: np.ndarray | None = None
        self.dones: np.ndarray | None = None
        self.critic_obs: np.ndarray | None = None
        self.aux_targets: np.ndarray | None = None
        self.sample_weight: np.ndarray | None = None
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None
        self.priorities: np.ndarray | None = None

    def __len__(self) -> int:
        return 0 if self.obs is None else int(len(self.obs))

    @staticmethod
    def _append_optional(old: np.ndarray | None, new: np.ndarray | None) -> np.ndarray | None:
        if old is None and new is None:
            return None
        if old is None or new is None:
            raise ValueError("Prioritized replay optional fields must stay consistently present or absent.")
        return np.concatenate([old, new], axis=0)

    def _trim(self) -> None:
        n = len(self)
        if n <= self.capacity:
            return
        keep = slice(n - self.capacity, n)
        self.obs = self.obs[keep] if self.obs is not None else None
        self.actions = self.actions[keep] if self.actions is not None else None
        self.old_log_prob = self.old_log_prob[keep] if self.old_log_prob is not None else None
        self.values = self.values[keep] if self.values is not None else None
        self.rewards = self.rewards[keep] if self.rewards is not None else None
        self.dones = self.dones[keep] if self.dones is not None else None
        self.critic_obs = self.critic_obs[keep] if self.critic_obs is not None else None
        self.aux_targets = self.aux_targets[keep] if self.aux_targets is not None else None
        self.sample_weight = self.sample_weight[keep] if self.sample_weight is not None else None
        self.advantages = self.advantages[keep] if self.advantages is not None else None
        self.returns = self.returns[keep] if self.returns is not None else None
        self.priorities = self.priorities[keep] if self.priorities is not None else None

    def add(self, batch: PolicyBatch, *, advantages: np.ndarray, returns: np.ndarray) -> None:
        if len(batch.obs) == 0:
            return
        obs = np.asarray(batch.obs, dtype=np.float32)
        actions = np.asarray(batch.actions, dtype=np.float32)
        old_log_prob = np.asarray(batch.old_log_prob, dtype=np.float32).reshape(-1)
        values = np.asarray(batch.values, dtype=np.float32).reshape(-1)
        rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(batch.dones, dtype=np.float32).reshape(-1)
        adv = np.asarray(advantages, dtype=np.float32).reshape(-1)
        ret = np.asarray(returns, dtype=np.float32).reshape(-1)
        if not (len(obs) == len(actions) == len(old_log_prob) == len(values) == len(rewards) == len(dones) == len(adv) == len(ret)):
            raise ValueError("Prioritized replay add() received mismatched batch lengths.")
        critic_obs = None if batch.critic_obs is None else np.asarray(batch.critic_obs, dtype=np.float32)
        aux_targets = None if batch.aux_targets is None else np.asarray(batch.aux_targets, dtype=np.float32)
        if batch.sample_weight is None:
            sample_weight = np.ones(len(obs), dtype=np.float32)
        else:
            sample_weight = np.asarray(batch.sample_weight, dtype=np.float32).reshape(-1)
            sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
            sample_weight = np.maximum(sample_weight, 1e-6).astype(np.float32)
        priorities = np.clip(np.abs(ret - values) + self.priority_eps, self.priority_eps, self.max_priority).astype(np.float32)

        self.obs = obs if self.obs is None else np.concatenate([self.obs, obs], axis=0)
        self.actions = actions if self.actions is None else np.concatenate([self.actions, actions], axis=0)
        self.old_log_prob = old_log_prob if self.old_log_prob is None else np.concatenate([self.old_log_prob, old_log_prob], axis=0)
        self.values = values if self.values is None else np.concatenate([self.values, values], axis=0)
        self.rewards = rewards if self.rewards is None else np.concatenate([self.rewards, rewards], axis=0)
        self.dones = dones if self.dones is None else np.concatenate([self.dones, dones], axis=0)
        self.critic_obs = critic_obs if self.critic_obs is None else self._append_optional(self.critic_obs, critic_obs)
        self.aux_targets = aux_targets if self.aux_targets is None else self._append_optional(self.aux_targets, aux_targets)
        self.sample_weight = sample_weight if self.sample_weight is None else np.concatenate([self.sample_weight, sample_weight], axis=0)
        self.advantages = adv if self.advantages is None else np.concatenate([self.advantages, adv], axis=0)
        self.returns = ret if self.returns is None else np.concatenate([self.returns, ret], axis=0)
        self.priorities = priorities if self.priorities is None else np.concatenate([self.priorities, priorities], axis=0)
        self._trim()

    def sample(self, batch_size: int, *, beta: float, is_clip: float) -> ReplaySample:
        n = len(self)
        if n == 0 or self.priorities is None:
            raise RuntimeError("Cannot sample from empty prioritized replay.")
        scaled = np.power(np.clip(self.priorities.astype(np.float64), self.priority_eps, self.max_priority), self.alpha)
        probs = scaled / max(float(scaled.sum()), EPS)
        size = int(max(1, batch_size))
        replace = n < size
        idx = self.rng.choice(np.arange(n), size=size, replace=replace, p=probs).astype(np.int64)
        beta = float(np.clip(beta, 0.0, 1.0))
        is_weights = np.power(n * np.maximum(probs[idx], EPS), -beta)
        is_weights = is_weights / max(float(np.max(is_weights)), EPS)
        is_weights = np.clip(is_weights, 0.0, float(max(is_clip, EPS))).astype(np.float32)
        original_weights = np.ones(size, dtype=np.float32) if self.sample_weight is None else self.sample_weight[idx].astype(np.float32)
        combined_weights = np.maximum(original_weights * is_weights, 1e-6).astype(np.float32)
        sample_batch = PolicyBatch(
            obs=self.obs[idx].astype(np.float32),
            actions=self.actions[idx].astype(np.float32),
            old_log_prob=self.old_log_prob[idx].astype(np.float32),
            values=self.values[idx].astype(np.float32),
            rewards=self.rewards[idx].astype(np.float32),
            dones=self.dones[idx].astype(np.float32),
            critic_obs=None if self.critic_obs is None else self.critic_obs[idx].astype(np.float32),
            aux_targets=None if self.aux_targets is None else self.aux_targets[idx].astype(np.float32),
            sample_weight=combined_weights,
        )
        return ReplaySample(
            batch=sample_batch,
            advantages=self.advantages[idx].astype(np.float32),
            returns=self.returns[idx].astype(np.float32),
            indices=idx,
            probabilities=probs[idx].astype(np.float32),
            is_weights=is_weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        if self.priorities is None:
            return
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if len(idx) != len(errors):
            raise ValueError("Priority update length mismatch.")
        values = np.clip(np.abs(errors) + self.priority_eps, self.priority_eps, self.max_priority).astype(np.float32)
        for i, value in zip(idx, values):
            if 0 <= int(i) < len(self.priorities):
                self.priorities[int(i)] = float(value)

    def diagnostics(self) -> dict[str, float]:
        if self.priorities is None or len(self.priorities) == 0:
            return {"size": 0.0}
        p = self.priorities.astype(np.float64)
        return {
            "size": float(len(p)),
            "priority_mean": float(np.mean(p)),
            "priority_max": float(np.max(p)),
            "priority_min": float(np.min(p)),
        }


class VectorizedW1Collector:
    """Synchronous vectorized train collector for one market panel."""

    def __init__(
        self,
        template_env: W1BudgetTraderEnv,
        *,
        num_envs: int,
        seed: int = 0,
        rare_event_config: dict[str, Any] | None = None,
        aux_config: dict[str, Any] | None = None,
        scenario_config: dict[str, Any] | None = None,
        curriculum_config: dict[str, Any] | None = None,
    ):
        self.template = template_env
        self.panel = template_env.panel
        self.num_envs = int(max(1, num_envs))
        self.rng = np.random.default_rng(int(seed))
        self.stock_dim = template_env.stock_dim
        self.asset_dim = template_env.asset_dim
        self.cash_index = template_env.cash_index
        self.state_day = 0
        self.episode_start_day = 0
        self.episode_end_day = len(self.panel.dates) - 1
        self.previous_weights = np.zeros((self.num_envs, self.asset_dim), dtype=np.float64)
        self.portfolio_value = np.full(self.num_envs, template_env.initial_amount, dtype=np.float64)
        self.peak_value = self.portfolio_value.copy()
        self.previous_drawdown = np.zeros(self.num_envs, dtype=np.float64)
        self.trader_reward_ewma = np.zeros(self.num_envs, dtype=np.float64)
        self.trader_reward_ewma_prev = np.zeros(self.num_envs, dtype=np.float64)
        self.trader_reward_change = np.zeros(self.num_envs, dtype=np.float64)
        self.rare_event_config = dict(rare_event_config or {})
        self.aux_config = dict(aux_config or {})
        self.scenario_config = dict(scenario_config or {})
        self.curriculum_config = dict(curriculum_config or {})
        self.pm_aux_targets, self.trader_aux_targets, self.aux_target_info = self._build_auxiliary_targets()
        self.reset()

    def reset(self) -> None:
        self.state_day = self._sample_curriculum_start_day()
        self.episode_start_day = int(self.state_day)
        if bool(self.curriculum_config.get("enabled", False)):
            episode_days = int(self.curriculum_config.get("episode_days", 504))
            self.episode_end_day = min(len(self.panel.dates) - 1, self.episode_start_day + max(1, episode_days))
        else:
            self.episode_end_day = len(self.panel.dates) - 1
        self.previous_weights.fill(0.0)
        if self.template.initial_weights_source == "equal_weight":
            self.previous_weights[:, : self.stock_dim] = 1.0 / self.stock_dim
        else:
            self.previous_weights[:, self.cash_index] = 1.0
        self.portfolio_value.fill(self.template.initial_amount)
        self.peak_value.fill(self.template.initial_amount)
        self.previous_drawdown.fill(0.0)
        self.trader_reward_ewma.fill(0.0)
        self.trader_reward_ewma_prev.fill(0.0)
        self.trader_reward_change.fill(0.0)

    def done(self) -> bool:
        return self.state_day >= self.episode_end_day

    def _sample_curriculum_start_day(self) -> int:
        cfg = self.curriculum_config
        if not bool(cfg.get("enabled", False)) or not hasattr(self, "scenario_context_targets"):
            return 0
        episode_days = int(cfg.get("episode_days", 504))
        max_start = max(0, len(self.panel.dates) - 1 - max(1, episode_days))
        if max_start <= 0:
            return 0
        start_min = int(max(0, cfg.get("start_min_day", 0)))
        candidates = np.arange(start_min, max_start + 1, dtype=np.int32)
        scenario = self.scenario_context_targets[candidates].astype(np.float64)
        weights = (
            float(cfg.get("base_weight", 1.0))
            + float(cfg.get("stress_start_scale", 1.0)) * scenario[:, 0]
            + float(cfg.get("recovery_start_scale", 1.0)) * scenario[:, 1]
            + float(cfg.get("calm_start_scale", 0.2)) * scenario[:, 2]
        )
        weights = np.maximum(weights, EPS)
        weights = weights / weights.sum()
        return int(self.rng.choice(candidates, p=weights))

    def current_q(self) -> np.ndarray:
        return self.previous_weights[:, : self.stock_dim].sum(axis=1)

    def _pm_raw_window_obs(self) -> np.ndarray:
        end = self.state_day + 1
        start = max(0, end - self.template.max_horizon_days)
        window = np.take(self.panel.features[start:end, 0, :], self.template.raw_window_indices, axis=1).astype(np.float32)
        if window.shape[0] == 0:
            window = np.zeros((1, len(self.template.raw_window_indices)), dtype=np.float32)
        pad = self.template.max_horizon_days - window.shape[0]
        if pad > 0:
            window = np.concatenate([np.repeat(window[:1], pad, axis=0), window], axis=0)
        flat = window[-self.template.max_horizon_days :].reshape(-1).astype(np.float32)
        return np.repeat(flat.reshape(1, -1), self.num_envs, axis=0)

    def pm_obs(self) -> np.ndarray:
        pm_features = self.panel.features[self.state_day, 0, self.template.pm_indices].astype(np.float32)
        pm_features = np.repeat(pm_features.reshape(1, -1), self.num_envs, axis=0)
        cash_q = np.stack([self.previous_weights[:, self.cash_index], self.current_q()], axis=1).astype(np.float32)
        feedback = self.trader_reward_change.reshape(-1, 1).astype(np.float32)
        return np.concatenate([pm_features, cash_q, self._pm_raw_window_obs(), feedback], axis=1).astype(np.float32)

    def trader_obs(self, *, q_target: np.ndarray, remaining_days: np.ndarray) -> np.ndarray:
        stock_features = np.take(self.panel.features[self.state_day], self.template.stock_indices, axis=1).astype(np.float32)
        stock_features = np.repeat(stock_features.reshape(1, self.stock_dim, -1), self.num_envs, axis=0)
        prev_stock_weights = self.previous_weights[:, : self.stock_dim].astype(np.float32)
        signed_budget = (q_target.astype(np.float64) - self.current_q()).astype(np.float32)
        remaining_norm = (np.maximum(1, remaining_days.astype(int)) / float(max(self.template.max_horizon_days, 1))).astype(np.float32)
        task = np.stack(
            [
                prev_stock_weights,
                np.repeat(signed_budget.reshape(-1, 1), self.stock_dim, axis=1),
                np.repeat(remaining_norm.reshape(-1, 1), self.stock_dim, axis=1),
            ],
            axis=2,
        )
        return np.concatenate([stock_features, task], axis=2).reshape(self.num_envs, -1).astype(np.float32)

    def horizons_from_actions(self, horizon_actions: np.ndarray) -> np.ndarray:
        idx = np.rint(horizon_actions.astype(np.float64)).astype(int)
        idx = np.clip(idx, 0, len(self.template.horizon_choices) - 1)
        choices = np.asarray(self.template.horizon_choices, dtype=np.int32)
        return choices[idx]

    def _future_stock_return_from(self, start_day: int, horizon: int) -> np.ndarray:
        start = int(start_day)
        end = min(len(self.panel.returns_next), start + max(1, int(horizon)))
        if end <= start:
            return np.zeros(self.stock_dim, dtype=np.float64)
        return np.prod(1.0 + self.panel.returns_next[start:end], axis=0) - 1.0

    @staticmethod
    def _rank01(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        n = values.size
        if n <= 1 or float(np.nanmax(values) - np.nanmin(values)) <= EPS:
            return np.full(n, 0.5, dtype=np.float32)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return (ranks / max(n - 1, 1)).astype(np.float32)

    def _future_universe_path(self, start_day: int, horizon: int) -> np.ndarray:
        start = int(start_day)
        end = min(len(self.panel.returns_next), start + max(1, int(horizon)))
        if end <= start:
            return np.zeros(1, dtype=np.float64)
        daily = self.panel.returns_next[start:end].astype(np.float64).mean(axis=1)
        return np.cumprod(1.0 + daily) - 1.0

    def _build_auxiliary_targets(self) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        n_days = len(self.panel.returns_next)
        pm_future20 = np.zeros(n_days, dtype=np.float64)
        pm_stress20 = np.zeros(n_days, dtype=np.float64)
        trader = np.zeros((n_days, self.stock_dim * 2), dtype=np.float32)
        group_ids = np.asarray(self.template.group_ids, dtype=int)
        for day in range(n_days):
            path20 = self._future_universe_path(day, 20)
            pm_future20[day] = float(path20[-1]) if len(path20) else 0.0
            pm_stress20[day] = float(max(0.0, -np.min(path20))) if len(path20) else 0.0
            future5 = self._future_stock_return_from(day, 5)
            residual = future5 - float(np.mean(future5))
            group_mean = np.zeros_like(future5)
            for gid in range(max(1, self.template.n_groups)):
                idx = np.where(group_ids == gid)[0]
                if idx.size:
                    group_mean[idx] = float(np.mean(future5[idx]))
            group_relative = future5 - group_mean
            trader[day] = np.concatenate([self._rank01(residual), self._rank01(group_relative)], axis=0)
        self.pm_future20_return = pm_future20.astype(np.float32)
        self.pm_future20_stress = pm_stress20.astype(np.float32)
        opportunity_threshold = float(np.quantile(pm_future20, 0.70))
        stress_threshold = float(np.quantile(pm_stress20, 0.70))
        calm_stress_threshold = float(np.quantile(pm_stress20, 0.35))
        calm_return_threshold = float(np.quantile(pm_future20, 0.35))
        opportunity_label = (pm_future20 >= opportunity_threshold).astype(np.float32)
        stress_label = (pm_stress20 >= stress_threshold).astype(np.float32)
        stress_context = stress_label.astype(bool)
        recovery_context = (~stress_context) & (pm_future20 >= opportunity_threshold)
        calm_context = (
            (~stress_context)
            & (~recovery_context)
            & (pm_stress20 <= calm_stress_threshold)
            & (pm_future20 >= calm_return_threshold)
        )
        neutral_context = ~(stress_context | recovery_context | calm_context)
        scenario = np.stack(
            [
                stress_context.astype(np.float32),
                recovery_context.astype(np.float32),
                calm_context.astype(np.float32),
                neutral_context.astype(np.float32),
            ],
            axis=1,
        )
        self.pm_opportunity_label = opportunity_label.astype(np.float32)
        self.pm_stress_label = stress_label.astype(np.float32)
        self.scenario_context_targets = scenario.astype(np.float32)
        self.scenario_context_names = ["stress", "recovery", "calm", "neutral"]
        self.scenario_context_ids = np.argmax(self.scenario_context_targets, axis=1).astype(np.int32)
        pm_target_mode = str(self.aux_config.get("pm_target_mode", "opportunity_stress")).lower()
        if pm_target_mode == "scenario_context":
            pm = self.scenario_context_targets
            self.pm_aux_target_names = [f"context_{name}" for name in self.scenario_context_names]
        elif pm_target_mode == "opportunity_stress":
            pm = np.stack([opportunity_label, stress_label], axis=1)
            self.pm_aux_target_names = ["future20_opportunity_label", "future20_stress_label"]
        else:
            raise ValueError(f"Unsupported pm aux target mode: {pm_target_mode}")
        info = {
            "pm_future20_opportunity_threshold_q70": opportunity_threshold,
            "pm_stress20_threshold_q70": stress_threshold,
            "pm_future20_calm_stress_threshold_q35": calm_stress_threshold,
            "pm_future20_calm_return_threshold_q35": calm_return_threshold,
            "pm_opportunity_positive_rate": float(opportunity_label.mean()),
            "pm_stress_positive_rate": float(stress_label.mean()),
            "scenario_stress_rate": float(scenario[:, 0].mean()),
            "scenario_recovery_rate": float(scenario[:, 1].mean()),
            "scenario_calm_rate": float(scenario[:, 2].mean()),
            "scenario_neutral_rate": float(scenario[:, 3].mean()),
            "pm_aux_target_mode": pm_target_mode,
            "trader_aux_dim": float(trader.shape[1]),
        }
        return pm.astype(np.float32), trader.astype(np.float32), info

    def rare_event_weights(self, info: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        cfg = self.rare_event_config
        if not bool(cfg.get("enabled", False)):
            ones = np.ones(self.num_envs, dtype=np.float64)
            zeros = np.zeros(self.num_envs, dtype=np.float64)
            return {
                "rare_pm_sample_weight": ones,
                "rare_trader_sample_weight": ones,
                "rare_pm_severity": zeros,
                "rare_trader_severity": zeros,
                "rare_missed_rally": zeros,
                "rare_cash_lock": zeros,
                "rare_false_rerisk": zeros,
                "rare_drawdown_start": zeros,
                "rare_bad_buy": zeros,
                "rare_bad_sell": zeros,
                "rare_turnover_without_return": zeros,
            }

        day = int(info["step_day"][0])
        base = float(cfg.get("base_weight", 1.0))
        max_weight = float(cfg.get("max_weight", 3.0))
        opportunity_label = float(self.pm_opportunity_label[day])
        stress_label = float(self.pm_stress_label[day])
        q_exec = np.asarray(info["q_exec"], dtype=np.float64)
        q_target = np.asarray(info["q_target"], dtype=np.float64)
        cash_exec = np.asarray(info["cash_exec"], dtype=np.float64)
        benchmark_return = np.asarray(info["benchmark_return"], dtype=np.float64)
        drawdown_increment = np.asarray(info["drawdown_increment"], dtype=np.float64)
        stock_turnover_l1 = np.asarray(info["stock_turnover_l1"], dtype=np.float64)
        net_return = np.asarray(info["net_return"], dtype=np.float64)

        cash_threshold = float(cfg.get("cash_lock_threshold", 0.35))
        risk_threshold = float(cfg.get("false_rerisk_q_threshold", 0.70))
        drawdown_scale = max(float(cfg.get("drawdown_increment_scale", 0.01)), EPS)
        turnover_scale = max(float(cfg.get("turnover_l1_scale", 0.05)), EPS)

        cash_lock = opportunity_label * np.maximum(cash_exec - cash_threshold, 0.0) / max(1.0 - cash_threshold, EPS)
        missed_rally = opportunity_label * (0.5 * cash_exec + 0.5 * np.maximum(float(benchmark_return[0]), 0.0) / 0.01)
        missed_rally = np.clip(missed_rally, 0.0, 1.0)
        false_rerisk = stress_label * np.maximum(q_exec - risk_threshold, 0.0) / max(1.0 - risk_threshold, EPS)
        drawdown_start = np.clip(drawdown_increment / drawdown_scale, 0.0, 1.0)

        delta = np.asarray(info["trade_delta_weights"], dtype=np.float64)[:, : self.stock_dim]
        trader_aux = self.trader_aux_targets[day].astype(np.float64)
        residual_rank = trader_aux[: self.stock_dim].reshape(1, -1)
        buy = np.maximum(delta, 0.0)
        sell = np.maximum(-delta, 0.0)
        buy_sum = buy.sum(axis=1)
        sell_sum = sell.sum(axis=1)
        bad_buy_quality = np.divide(
            (buy * (1.0 - residual_rank)).sum(axis=1),
            np.maximum(buy_sum, EPS),
            out=np.zeros(self.num_envs, dtype=np.float64),
            where=buy_sum > EPS,
        )
        bad_sell_quality = np.divide(
            (sell * residual_rank).sum(axis=1),
            np.maximum(sell_sum, EPS),
            out=np.zeros(self.num_envs, dtype=np.float64),
            where=sell_sum > EPS,
        )
        trade_scale = np.clip((buy_sum + sell_sum) / turnover_scale, 0.0, 1.0)
        bad_buy = bad_buy_quality * np.clip(buy_sum / turnover_scale, 0.0, 1.0)
        bad_sell = bad_sell_quality * np.clip(sell_sum / turnover_scale, 0.0, 1.0)
        turnover_without_return = trade_scale * np.clip(np.maximum(-net_return, 0.0) / 0.01, 0.0, 1.0)

        pm_severity = (
            float(cfg.get("missed_rally_scale", 0.80)) * missed_rally
            + float(cfg.get("cash_lock_scale", 0.60)) * cash_lock
            + float(cfg.get("false_rerisk_scale", 0.80)) * false_rerisk
            + float(cfg.get("drawdown_start_scale", 0.50)) * drawdown_start
        )
        trader_severity = (
            float(cfg.get("bad_buy_scale", 0.80)) * bad_buy
            + float(cfg.get("bad_sell_scale", 0.80)) * bad_sell
            + float(cfg.get("turnover_without_return_scale", 0.40)) * turnover_without_return
        )
        pm_weight = np.clip(base + pm_severity, base, max_weight)
        trader_weight = np.clip(base + trader_severity, base, max_weight)
        return {
            "rare_pm_sample_weight": pm_weight,
            "rare_trader_sample_weight": trader_weight,
            "rare_pm_severity": pm_severity,
            "rare_trader_severity": trader_severity,
            "rare_missed_rally": missed_rally,
            "rare_cash_lock": cash_lock,
            "rare_false_rerisk": false_rerisk,
            "rare_drawdown_start": drawdown_start,
            "rare_bad_buy": bad_buy,
            "rare_bad_sell": bad_sell,
            "rare_turnover_without_return": turnover_without_return,
            "rare_q_target_minus_exec": q_target - q_exec,
        }

    def scenario_event_weights(self, info: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        cfg = self.scenario_config
        day = int(info["step_day"][0])
        if hasattr(self, "scenario_context_targets"):
            context = self.scenario_context_targets[day].astype(np.float64)
            stress = np.full(self.num_envs, float(context[0]), dtype=np.float64)
            recovery = np.full(self.num_envs, float(context[1]), dtype=np.float64)
            calm = np.full(self.num_envs, float(context[2]), dtype=np.float64)
            neutral = np.full(self.num_envs, float(context[3]), dtype=np.float64)
            context_id = np.full(self.num_envs, float(self.scenario_context_ids[day]), dtype=np.float64)
        else:
            stress = np.zeros(self.num_envs, dtype=np.float64)
            recovery = np.zeros(self.num_envs, dtype=np.float64)
            calm = np.zeros(self.num_envs, dtype=np.float64)
            neutral = np.zeros(self.num_envs, dtype=np.float64)
            context_id = np.full(self.num_envs, -1.0, dtype=np.float64)
        if not bool(cfg.get("enabled", False)):
            ones = np.ones(self.num_envs, dtype=np.float64)
            zeros = np.zeros(self.num_envs, dtype=np.float64)
            return {
                "scenario_pm_sample_weight": ones,
                "scenario_trader_sample_weight": ones,
                "scenario_pm_severity": zeros,
                "scenario_trader_severity": zeros,
                "scenario_context_id": context_id,
                "scenario_stress": stress,
                "scenario_recovery": recovery,
                "scenario_calm": calm,
                "scenario_neutral": neutral,
                "scenario_missed_recovery": zeros,
                "scenario_false_risk_in_stress": zeros,
                "scenario_calm_overtrade": zeros,
                "scenario_recovery_bad_buy": zeros,
                "scenario_stress_bad_buy": zeros,
                "scenario_stress_bad_sell": zeros,
            }

        base = float(cfg.get("base_weight", 1.0))
        max_weight = float(cfg.get("max_weight", 2.5))

        q_exec = np.asarray(info["q_exec"], dtype=np.float64)
        cash_exec = np.asarray(info["cash_exec"], dtype=np.float64)
        benchmark_return = np.asarray(info["benchmark_return"], dtype=np.float64)
        stock_turnover_l1 = np.asarray(info["stock_turnover_l1"], dtype=np.float64)
        drawdown_increment = np.asarray(info["drawdown_increment"], dtype=np.float64)
        delta = np.asarray(info["trade_delta_weights"], dtype=np.float64)[:, : self.stock_dim]

        turnover_scale = max(float(cfg.get("turnover_l1_scale", 0.05)), EPS)
        recovery_cash_threshold = float(cfg.get("recovery_cash_threshold", 0.25))
        stress_q_threshold = float(cfg.get("stress_q_threshold", 0.70))
        missed_recovery = recovery * np.clip(
            0.5 * np.maximum(cash_exec - recovery_cash_threshold, 0.0) / max(1.0 - recovery_cash_threshold, EPS)
            + 0.5 * np.maximum(benchmark_return, 0.0) / 0.01,
            0.0,
            1.0,
        )
        false_risk_in_stress = stress * np.clip(
            np.maximum(q_exec - stress_q_threshold, 0.0) / max(1.0 - stress_q_threshold, EPS)
            + np.maximum(drawdown_increment, 0.0) / max(float(cfg.get("drawdown_increment_scale", 0.01)), EPS),
            0.0,
            1.0,
        )
        calm_overtrade = calm * np.clip(stock_turnover_l1 / turnover_scale, 0.0, 1.0)

        trader_aux = self.trader_aux_targets[day].astype(np.float64)
        residual_rank = trader_aux[: self.stock_dim].reshape(1, -1)
        buy = np.maximum(delta, 0.0)
        sell = np.maximum(-delta, 0.0)
        buy_sum = buy.sum(axis=1)
        sell_sum = sell.sum(axis=1)
        bad_buy_quality = np.divide(
            (buy * (1.0 - residual_rank)).sum(axis=1),
            np.maximum(buy_sum, EPS),
            out=np.zeros(self.num_envs, dtype=np.float64),
            where=buy_sum > EPS,
        )
        bad_sell_quality = np.divide(
            (sell * residual_rank).sum(axis=1),
            np.maximum(sell_sum, EPS),
            out=np.zeros(self.num_envs, dtype=np.float64),
            where=sell_sum > EPS,
        )
        recovery_bad_buy = recovery * bad_buy_quality * np.clip(buy_sum / turnover_scale, 0.0, 1.0)
        stress_bad_buy = stress * np.clip(buy_sum / turnover_scale, 0.0, 1.0)
        stress_bad_sell = stress * bad_sell_quality * np.clip(sell_sum / turnover_scale, 0.0, 1.0)

        pm_severity = (
            float(cfg.get("missed_recovery_scale", 0.60)) * missed_recovery
            + float(cfg.get("false_risk_stress_scale", 0.60)) * false_risk_in_stress
            + float(cfg.get("calm_overtrade_pm_scale", 0.15)) * calm_overtrade
        )
        trader_severity = (
            float(cfg.get("recovery_bad_buy_scale", 0.60)) * recovery_bad_buy
            + float(cfg.get("stress_bad_buy_scale", 0.50)) * stress_bad_buy
            + float(cfg.get("stress_bad_sell_scale", 0.30)) * stress_bad_sell
            + float(cfg.get("calm_overtrade_trader_scale", 0.30)) * calm_overtrade
        )
        pm_weight = np.clip(base + pm_severity, base, max_weight)
        trader_weight = np.clip(base + trader_severity, base, max_weight)
        return {
            "scenario_pm_sample_weight": pm_weight,
            "scenario_trader_sample_weight": trader_weight,
            "scenario_pm_severity": pm_severity,
            "scenario_trader_severity": trader_severity,
            "scenario_context_id": context_id,
            "scenario_stress": stress,
            "scenario_recovery": recovery,
            "scenario_calm": calm,
            "scenario_neutral": neutral,
            "scenario_missed_recovery": missed_recovery,
            "scenario_false_risk_in_stress": false_risk_in_stress,
            "scenario_calm_overtrade": calm_overtrade,
            "scenario_recovery_bad_buy": recovery_bad_buy,
            "scenario_stress_bad_buy": stress_bad_buy,
            "scenario_stress_bad_sell": stress_bad_sell,
        }

    def pm_window_sample_weights(self, pm_state: OpenPMState) -> np.ndarray:
        if not (bool(self.rare_event_config.get("enabled", False)) or bool(self.scenario_config.get("enabled", False))):
            return np.ones(self.num_envs, dtype=np.float32)
        mode = str(
            self.scenario_config.get(
                "pm_weight_aggregation",
                self.rare_event_config.get("pm_weight_aggregation", "max"),
            )
        ).lower()
        if mode == "mean":
            weights = pm_state.rare_weight_sum / np.maximum(pm_state.rare_weight_len.astype(np.float64), 1.0)
        elif mode == "max":
            weights = pm_state.rare_weight_max
        else:
            raise ValueError(f"Unsupported pm_weight_aggregation: {mode}")
        max_weight = max(
            float(self.rare_event_config.get("max_weight", 1.0)),
            float(self.scenario_config.get("max_weight", 1.0)),
            1.0,
        )
        return np.clip(weights, 1e-6, max_weight).astype(np.float32)

    def _group_returns_from(self, stock_returns: np.ndarray) -> np.ndarray:
        out = np.zeros(self.stock_dim, dtype=np.float64)
        group_ids = np.asarray(self.template.group_ids, dtype=int)
        for gid in range(self.template.n_groups):
            idx = np.where(group_ids == gid)[0]
            if idx.size:
                out[idx] = float(np.mean(stock_returns[idx]))
        return out

    def apply_risk_stop_batch(self, q_raw_target: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        features = features_from_panel_row(self.panel.feature_columns, self.panel.features[self.state_day, 0, :])
        q_safe, info = compute_risk_stop(
            q_raw_target,
            features=features,
            current_drawdown=self.previous_drawdown,
            trader_feedback_change=self.trader_reward_change,
            config=self.template.risk_stop_config,
            q_min=self.template.q_min,
            q_max=self.template.q_max,
        )
        return np.asarray(q_safe, dtype=np.float64), info

    def execution_opportunity_score_batch(self) -> np.ndarray:
        cfg = self.template.execution_config
        if not bool(cfg.get("opportunity_score_enabled", False)):
            return np.ones(self.num_envs, dtype=np.float64)

        def feature(name: str, default: float = 0.0) -> float:
            idx = self.template._safe_feature_index(name)
            return float(default if idx is None else self.panel.features[self.state_day, 0, idx])

        trend = feature(str(cfg.get("opportunity_trend_feature", "SP500_Trend")))
        residual = feature(str(cfg.get("opportunity_residual_feature", "universe_return_20d")))
        regime = feature(str(cfg.get("opportunity_regime_feature", "Regime_1_Prob")))
        vix_change = feature(str(cfg.get("opportunity_vix_change_feature", "VIX_change_5d")))
        turbulence = feature(str(cfg.get("opportunity_turbulence_feature", "turbulence")))
        raw = (
            float(cfg.get("opportunity_bias", 0.0))
            + float(cfg.get("opportunity_trend_weight", 0.40)) * trend
            + float(cfg.get("opportunity_residual_weight", 0.35)) * residual
            + float(cfg.get("opportunity_regime_weight", 0.30)) * (2.0 * regime - 1.0)
            - float(cfg.get("opportunity_vix_change_weight", 0.20)) * vix_change
            - float(cfg.get("opportunity_turbulence_weight", 0.15)) * turbulence
        )
        power = float(cfg.get("opportunity_score_power", 1.0))
        score = float(np.clip(sigmoid_scalar(raw) ** max(power, EPS), 0.0, 1.0))
        return np.full(self.num_envs, score, dtype=np.float64)

    def step_batch(
        self,
        *,
        q_target: np.ndarray,
        remaining_days: np.ndarray,
        trader_actions: np.ndarray,
        q_raw_target: np.ndarray | None = None,
        risk_stop_info: dict[str, np.ndarray] | None = None,
        execution_context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        day = self.state_day
        prev_weights = self.previous_weights.copy()
        desired_weights = normalize_simplex_batch(trader_actions)
        clipped_q = np.clip(q_target.astype(np.float64), self.template.q_min, self.template.q_max)
        desired_weights, target_weights, execution_info = apply_budgeted_flow_execution_batch(
            previous_weights=prev_weights,
            desired_weights=desired_weights,
            q_target=clipped_q,
            remaining_days=remaining_days,
            stock_dim=self.stock_dim,
            cash_index=self.cash_index,
            config=self.template.execution_config,
            primitive_strength=(
                None
                if not execution_context or "latent_action_primitive_strength" not in execution_context
                else np.asarray(execution_context["latent_action_primitive_strength"], dtype=np.float64)
            ),
            opportunity_score=(
                self.execution_opportunity_score_batch()
                if bool(self.template.execution_config.get("opportunity_score_enabled", False))
                else None
            ),
            q_min=self.template.q_min,
            q_max=self.template.q_max,
        )
        q_exec = target_weights[:, : self.stock_dim].sum(axis=1)
        trade_delta = target_weights - prev_weights
        stock_turnover_l1 = np.abs(trade_delta[:, : self.stock_dim]).sum(axis=1)
        turnover_l1 = np.abs(trade_delta).sum(axis=1)
        transaction_cost = self.template.transaction_cost_pct * stock_turnover_l1

        asset_returns = self.panel.returns_next[day].astype(np.float64)
        benchmark_return = float(np.mean(asset_returns))
        gross_return = target_weights[:, : self.stock_dim] @ asset_returns
        net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0

        new_value = self.portfolio_value * (1.0 + net_return)
        peak_value = np.maximum(self.peak_value, new_value)
        drawdown = new_value / np.maximum(peak_value, EPS) - 1.0
        drawdown_increment = np.maximum(0.0, self.previous_drawdown - drawdown)

        next_values = np.zeros_like(target_weights)
        next_values[:, : self.stock_dim] = target_weights[:, : self.stock_dim] * (1.0 + asset_returns.reshape(1, -1))
        next_values[:, self.cash_index] = target_weights[:, self.cash_index]
        post_market = normalize_simplex_batch(next_values)

        risky_u = np.divide(
            target_weights[:, : self.stock_dim],
            np.maximum(q_exec.reshape(-1, 1), EPS),
            out=np.full((self.num_envs, self.stock_dim), 1.0 / self.stock_dim, dtype=np.float64),
            where=q_exec.reshape(-1, 1) > EPS,
        )
        portfolio_entropy = entropy_simplex_batch(target_weights)
        risky_entropy = entropy_simplex_batch(risky_u)
        tracking_l1 = np.abs(q_exec - clipped_q)

        pm_reward, pm_parts = self._pm_reward_batch(
            net_return=net_return,
            benchmark_return=benchmark_return,
            drawdown_increment=drawdown_increment,
            cash_target_pm=1.0 - clipped_q,
        )
        trader_reward, trader_parts = self._trader_reward_batch(
            trade_delta=trade_delta,
            u=risky_u,
            target_weights=target_weights,
            portfolio_entropy=portfolio_entropy,
            tracking_l1=tracking_l1,
            remaining_days=remaining_days,
            transaction_cost=transaction_cost,
        )

        self.previous_weights = post_market
        self.portfolio_value = new_value
        self.peak_value = peak_value
        self.previous_drawdown = drawdown
        self.state_day += 1
        self._update_trader_feedback(trader_reward)

        return {
            "date": np.asarray([str(pd.Timestamp(self.panel.dates[day]).date())] * self.num_envs, dtype=object),
            "next_date": np.asarray([str(pd.Timestamp(self.panel.dates[day + 1]).date())] * self.num_envs, dtype=object),
            "step_day": np.full(self.num_envs, day, dtype=np.int32),
            "q_raw_target": clipped_q if q_raw_target is None else np.asarray(q_raw_target, dtype=np.float64),
            "q_target": clipped_q,
            "q_exec": q_exec,
            "cash_target_pm": 1.0 - clipped_q,
            "cash_exec": target_weights[:, self.cash_index],
            "desired_q": execution_info.get("execution_desired_q", np.full(self.num_envs, np.nan)),
            "desired_cash": execution_info.get("execution_desired_cash", np.full(self.num_envs, np.nan)),
            "remaining_days": remaining_days.astype(np.int32),
            "remaining_days_normalized": remaining_days / float(max(self.template.max_horizon_days, 1)),
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": np.full(self.num_envs, benchmark_return, dtype=np.float64),
            "transaction_cost": transaction_cost,
            "stock_turnover_l1": stock_turnover_l1,
            "turnover_l1": turnover_l1,
            "drawdown": drawdown,
            "drawdown_increment": drawdown_increment,
            "target_weights": target_weights,
            "desired_weights": desired_weights,
            "pre_trade_weights": prev_weights,
            "trade_delta_weights": trade_delta,
            "portfolio_value": new_value,
            "risky_entropy": risky_entropy,
            "portfolio_entropy": portfolio_entropy,
            "tracking_l1": tracking_l1,
            "pm_reward": pm_reward,
            "trader_reward": trader_reward,
            **execution_info,
            **pm_parts,
            **trader_parts,
            **(risk_stop_info or {}),
        }

    def _pm_reward_batch(
        self,
        *,
        net_return: np.ndarray,
        benchmark_return: float,
        drawdown_increment: np.ndarray,
        cash_target_pm: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.template.pm_reward_config
        trend_idx = self.template._safe_feature_index(str(cfg.get("opportunity_feature", "SP500_Trend")))
        trend = 0.0 if trend_idx is None else float(self.panel.features[self.state_day, 0, trend_idx])
        opportunity_gate = float(1.0 / (1.0 + np.exp(-trend)))
        cash_opp = opportunity_gate * cash_target_pm * max(float(benchmark_return), 0.0)
        prior_threshold = float(cfg.get("cash_opportunity_prior_threshold", 0.55))
        cash_opp_prior = cash_target_pm * max(0.0, opportunity_gate - prior_threshold)
        active_return = net_return - float(benchmark_return)
        reward = (
            float(cfg.get("return_weight", 1.0)) * net_return
            - float(cfg.get("cash_opportunity_penalty", 0.30)) * cash_opp
            - float(cfg.get("cash_opportunity_prior_penalty", 0.0)) * cash_opp_prior
            - float(cfg.get("drawdown_penalty", 2.0)) * drawdown_increment
            + float(cfg.get("active_return_weight", 0.0)) * opportunity_gate * active_return
        )
        return self.template.reward_scale * reward, {
            "pm_opportunity_gate": np.full(self.num_envs, opportunity_gate, dtype=np.float64),
            "pm_cash_opportunity_cost": cash_opp,
            "pm_cash_opportunity_prior_cost": cash_opp_prior,
            "pm_active_return": active_return,
        }

    def _trader_reward_batch(
        self,
        *,
        trade_delta: np.ndarray,
        u: np.ndarray,
        target_weights: np.ndarray,
        portfolio_entropy: np.ndarray,
        tracking_l1: np.ndarray,
        remaining_days: np.ndarray,
        transaction_cost: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.template.trader_reward_config
        delta = trade_delta[:, : self.stock_dim]
        horizon = int(cfg.get("flow_horizon", 5))
        future = self._future_stock_return_from(self.state_day, horizon)
        residual_future = future - float(np.mean(future))
        flow_select = delta @ residual_future
        group_relative_weight = float(cfg.get("group_relative_weight", 0.0))
        if group_relative_weight != 0.0:
            group_returns = self._group_returns_from(future)
            group_relative = u @ (future - group_returns)
        else:
            group_relative = np.zeros(self.num_envs, dtype=np.float64)
        vol_idx = self.template._safe_feature_index(str(cfg.get("vol_feature", "realized_vol_20d")))
        if vol_idx is None:
            vol = np.zeros(self.stock_dim, dtype=np.float64)
        else:
            vol = self.panel.features[self.state_day, :, vol_idx].astype(np.float64)
        vol_adjusted_position_change = np.sum(np.abs(delta) * np.maximum(vol.reshape(1, -1), 0.0), axis=1)
        tracking_mode = str(cfg.get("tracking_multiplier", "constant")).lower()
        if tracking_mode == "remaining_days":
            tracking_multiplier = np.maximum(1, remaining_days.astype(np.float64))
        elif tracking_mode == "remaining_days_normalized":
            tracking_multiplier = remaining_days.astype(np.float64) / float(max(self.template.max_horizon_days, 1))
        elif tracking_mode == "deadline_inverse":
            tracking_multiplier = float(max(self.template.max_horizon_days, 1)) / np.maximum(1.0, remaining_days.astype(np.float64))
        elif tracking_mode == "deadline_linear":
            remaining_norm = remaining_days.astype(np.float64) / float(max(self.template.max_horizon_days, 1))
            slope = float(cfg.get("deadline_tracking_slope", 2.0))
            tracking_multiplier = 1.0 + slope * np.maximum(0.0, 1.0 - remaining_norm)
        else:
            tracking_multiplier = np.full(self.num_envs, float(cfg.get("tracking_multiplier", 1.0)), dtype=np.float64)
        tracking_penalty_term = float(cfg.get("tracking_penalty", 0.05)) * tracking_multiplier * tracking_l1
        entropy_scope = str(cfg.get("entropy_scope", "portfolio")).lower()
        if entropy_scope == "risky":
            entropy_value = entropy_simplex_batch(u)
        elif entropy_scope == "portfolio":
            entropy_value = portfolio_entropy
        else:
            raise ValueError(f"Unsupported W1 trader entropy_scope: {entropy_scope}")
        entropy_term = float(cfg.get("entropy_bonus", 0.001)) * entropy_value
        reward = (
            float(cfg.get("flow_select_weight", 0.30)) * flow_select
            + group_relative_weight * group_relative
            + entropy_term
            - float(cfg.get("vol_adjusted_cost", 0.02)) * vol_adjusted_position_change
            - float(cfg.get("transaction_cost_weight", 1.0)) * transaction_cost
            - tracking_penalty_term
        )
        return self.template.reward_scale * reward, {
            "trader_flow_select_5d": flow_select,
            "trader_group_relative_5d": group_relative,
            "trader_entropy_bonus": entropy_value,
            "trader_entropy_reward_term": entropy_term,
            "trader_vol_adjusted_position_change": vol_adjusted_position_change,
            "trader_tracking_multiplier": tracking_multiplier,
            "trader_tracking_penalty_term": tracking_penalty_term,
        }

    def _update_trader_feedback(self, trader_reward: np.ndarray) -> None:
        self.trader_reward_ewma_prev = self.trader_reward_ewma.copy()
        scaled = trader_reward.astype(np.float64) / max(self.template.reward_scale, EPS)
        alpha = self.template.trader_feedback_alpha
        self.trader_reward_ewma = (1.0 - alpha) * self.trader_reward_ewma + alpha * scaled
        self.trader_reward_change = self.trader_reward_ewma - self.trader_reward_ewma_prev

    def pm_window_reward(
        self,
        *,
        return_factor: np.ndarray,
        benchmark_factor: np.ndarray,
        drawdown_sum: np.ndarray,
        opportunity_sum: np.ndarray,
        window_len: np.ndarray,
        q_target: np.ndarray,
    ) -> np.ndarray:
        cfg = self.template.pm_reward_config
        window_return = return_factor - 1.0
        window_benchmark = benchmark_factor - 1.0
        opportunity_gate = opportunity_sum / np.maximum(window_len.astype(np.float64), 1.0)
        cash = 1.0 - np.clip(q_target, self.template.q_min, self.template.q_max)
        cash_opp = opportunity_gate * cash * np.maximum(window_benchmark, 0.0)
        prior_threshold = float(cfg.get("cash_opportunity_prior_threshold", 0.55))
        cash_opp_prior = cash * np.maximum(0.0, opportunity_gate - prior_threshold)
        active_return = window_return - window_benchmark
        reward = (
            float(cfg.get("return_weight", 1.0)) * window_return
            - float(cfg.get("cash_opportunity_penalty", 0.30)) * cash_opp
            - float(cfg.get("cash_opportunity_prior_penalty", 0.0)) * cash_opp_prior
            - float(cfg.get("drawdown_penalty", 2.0)) * drawdown_sum
            + float(cfg.get("active_return_weight", 0.0)) * opportunity_gate * active_return
        )
        return self.template.reward_scale * reward


def empty_pm_state(num_envs: int, pm_obs_dim: int, pm_aux_dim: int) -> OpenPMState:
    return OpenPMState(
        obs=np.zeros((num_envs, pm_obs_dim), dtype=np.float32),
        action=np.zeros((num_envs, 2), dtype=np.float32),
        value=np.zeros(num_envs, dtype=np.float32),
        log_prob=np.zeros(num_envs, dtype=np.float32),
        active=np.zeros(num_envs, dtype=bool),
        q_raw_target=np.zeros(num_envs, dtype=np.float64),
        q_target=np.zeros(num_envs, dtype=np.float64),
        horizon_days=np.ones(num_envs, dtype=np.int32),
        start_day=np.zeros(num_envs, dtype=np.int32),
        return_factor=np.ones(num_envs, dtype=np.float64),
        benchmark_factor=np.ones(num_envs, dtype=np.float64),
        drawdown_sum=np.zeros(num_envs, dtype=np.float64),
        opportunity_sum=np.zeros(num_envs, dtype=np.float64),
        window_len=np.zeros(num_envs, dtype=np.int32),
        aux_target=np.zeros((num_envs, pm_aux_dim), dtype=np.float32),
        rare_weight_sum=np.zeros(num_envs, dtype=np.float64),
        rare_weight_max=np.ones(num_envs, dtype=np.float64),
        rare_weight_len=np.zeros(num_envs, dtype=np.int32),
    )


def append_pm_records(
    *,
    pm_state: OpenPMState,
    mask: np.ndarray,
    rewards: np.ndarray,
    done: float,
    obs_rows: list[list[np.ndarray]],
    actions: list[list[np.ndarray]],
    log_probs: list[list[float]],
    values: list[list[float]],
    out_rewards: list[list[float]],
    dones: list[list[float]],
    aux_targets: list[list[np.ndarray]] | None = None,
    sample_weights: list[list[float]] | None = None,
    sample_weight_values: np.ndarray | None = None,
) -> None:
    idxs = np.where(mask)[0]
    for idx in idxs:
        obs_rows[idx].append(pm_state.obs[idx].copy())
        actions[idx].append(pm_state.action[idx].copy())
        log_probs[idx].append(float(pm_state.log_prob[idx]))
        values[idx].append(float(pm_state.value[idx]))
        out_rewards[idx].append(float(rewards[idx]))
        dones[idx].append(float(done))
        if aux_targets is not None:
            aux_targets[idx].append(pm_state.aux_target[idx].astype(np.float32).copy())
        if sample_weights is not None:
            value = 1.0 if sample_weight_values is None else float(sample_weight_values[idx])
            sample_weights[idx].append(value)


def flatten_env_lists(items: list[list[Any]], *, dtype: Any) -> np.ndarray:
    flat: list[Any] = []
    for env_items in items:
        flat.extend(env_items)
    return np.asarray(flat, dtype=dtype)


def collect_vectorized_episode(
    collector: VectorizedW1Collector,
    pm_policy: Any,
    trader_policy: Any,
    *,
    keep_trace_env: int = 0,
    supervisor: ClosedLoopActionSupervisor | None = None,
) -> tuple[PolicyBatch, PolicyBatch, pd.DataFrame]:
    collector.reset()
    num_envs = collector.num_envs
    pm_state = empty_pm_state(num_envs, collector.template.pm_obs_dim, collector.pm_aux_targets.shape[1])

    pm_obs_rows: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    pm_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    pm_log_probs: list[list[float]] = [[] for _ in range(num_envs)]
    pm_values: list[list[float]] = [[] for _ in range(num_envs)]
    pm_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    pm_dones: list[list[float]] = [[] for _ in range(num_envs)]
    pm_aux_targets: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    pm_sample_weights: list[list[float]] = [[] for _ in range(num_envs)]

    trader_obs_rows: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    trader_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    trader_log_probs: list[list[float]] = [[] for _ in range(num_envs)]
    trader_values: list[list[float]] = [[] for _ in range(num_envs)]
    trader_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    trader_dones: list[list[float]] = [[] for _ in range(num_envs)]
    trader_critic_obs_rows: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    trader_aux_targets: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    trader_sample_weights: list[list[float]] = [[] for _ in range(num_envs)]
    trace_rows: list[dict[str, Any]] = []

    while not collector.done():
        elapsed = collector.state_day - pm_state.start_day
        update_mask = (~pm_state.active) | (elapsed >= pm_state.horizon_days)
        close_mask = update_mask & pm_state.active
        if np.any(close_mask):
            close_rewards = collector.pm_window_reward(
                return_factor=pm_state.return_factor,
                benchmark_factor=pm_state.benchmark_factor,
                drawdown_sum=pm_state.drawdown_sum,
                opportunity_sum=pm_state.opportunity_sum,
                window_len=pm_state.window_len,
                q_target=pm_state.q_raw_target,
            )
            append_pm_records(
                pm_state=pm_state,
                mask=close_mask,
                rewards=close_rewards,
                done=0.0,
                obs_rows=pm_obs_rows,
                actions=pm_actions,
                log_probs=pm_log_probs,
                values=pm_values,
                out_rewards=pm_rewards,
                dones=pm_dones,
                aux_targets=pm_aux_targets,
                sample_weights=pm_sample_weights,
                sample_weight_values=collector.pm_window_sample_weights(pm_state),
            )
        if np.any(update_mask):
            full_pm_obs = collector.pm_obs()
            subset_obs = full_pm_obs[update_mask]
            action, value, log_prob = sample_policy_batch(pm_policy, subset_obs, deterministic=False)
            idxs = np.where(update_mask)[0]
            pm_state.obs[idxs] = subset_obs
            pm_state.action[idxs] = action
            pm_state.value[idxs] = value
            pm_state.log_prob[idxs] = log_prob
            pm_state.q_raw_target[idxs] = np.clip(action[:, 0].astype(np.float64), collector.template.q_min, collector.template.q_max)
            pm_state.q_target[idxs] = pm_state.q_raw_target[idxs]
            pm_state.horizon_days[idxs] = collector.horizons_from_actions(action[:, 1])
            pm_state.start_day[idxs] = collector.state_day
            pm_state.return_factor[idxs] = 1.0
            pm_state.benchmark_factor[idxs] = 1.0
            pm_state.drawdown_sum[idxs] = 0.0
            pm_state.opportunity_sum[idxs] = 0.0
            pm_state.window_len[idxs] = 0
            pm_state.aux_target[idxs] = collector.pm_aux_targets[collector.state_day]
            pm_state.rare_weight_sum[idxs] = 0.0
            pm_state.rare_weight_max[idxs] = 1.0
            pm_state.rare_weight_len[idxs] = 0
            pm_state.active[idxs] = True

        remaining = np.maximum(1, pm_state.horizon_days - (collector.state_day - pm_state.start_day)).astype(np.int32)
        pm_state.q_target, risk_stop_info = collector.apply_risk_stop_batch(pm_state.q_raw_target)
        trader_obs = collector.trader_obs(q_target=pm_state.q_target, remaining_days=remaining)
        trader_critic_obs = None
        if getattr(trader_policy, "critic_extra_dim", 0) > 0:
            current_pm_obs = collector.pm_obs()
            scalars = np.stack(
                [
                    pm_state.q_target.astype(np.float32),
                    pm_state.horizon_days.astype(np.float32) / float(max(collector.template.max_horizon_days, 1)),
                    remaining.astype(np.float32) / float(max(collector.template.max_horizon_days, 1)),
                ],
                axis=1,
            )
            trader_critic_obs = np.concatenate([current_pm_obs.astype(np.float32), scalars.astype(np.float32)], axis=1)
        trader_action, trader_value, trader_log_prob = sample_policy_batch(
            trader_policy,
            trader_obs,
            deterministic=False,
            critic_obs=trader_critic_obs,
        )
        supervisor_info: dict[str, np.ndarray] = {}
        if supervisor is not None and supervisor.enabled:
            trader_action, supervisor_info = supervisor.apply(panel=collector.panel, day=collector.state_day, actions=trader_action)
            trader_value, trader_log_prob = evaluate_policy_action_batch(
                trader_policy,
                trader_obs,
                trader_action,
                critic_obs=trader_critic_obs,
            )
        executable_trader_action = decode_policy_actions(trader_policy, trader_obs, trader_action)
        trader_diag = policy_action_diagnostics(trader_policy, trader_obs, trader_action)
        info = collector.step_batch(
            q_target=pm_state.q_target,
            remaining_days=remaining,
            trader_actions=executable_trader_action,
            q_raw_target=pm_state.q_raw_target,
            risk_stop_info=risk_stop_info,
            execution_context=trader_diag,
        )
        if supervisor_info:
            info.update(supervisor_info)
        rare_info = collector.rare_event_weights(info)
        info.update(rare_info)
        scenario_info = collector.scenario_event_weights(info)
        info.update(scenario_info)
        max_train_weight = max(
            float(collector.rare_event_config.get("max_weight", 1.0)),
            float(collector.scenario_config.get("max_weight", 1.0)),
            1.0,
        )
        info["train_pm_sample_weight"] = np.clip(
            info["rare_pm_sample_weight"] * info["scenario_pm_sample_weight"],
            1e-6,
            max_train_weight,
        )
        info["train_trader_sample_weight"] = np.clip(
            info["rare_trader_sample_weight"] * info["scenario_trader_sample_weight"],
            1e-6,
            max_train_weight,
        )
        if supervisor is not None and supervisor.enabled:
            bonus = float(supervisor.config.get("sample_weight_bonus", 0.50))
            supervisor_max_weight = float(supervisor.config.get("max_sample_weight", max_train_weight))
            info["train_trader_sample_weight"] = np.clip(
                info["train_trader_sample_weight"] * (1.0 + bonus * info.get("supervisor_intervened", 0.0)),
                1e-6,
                max(max_train_weight, supervisor_max_weight),
            )
        done = collector.done()

        pm_state.return_factor *= 1.0 + info["net_return"]
        pm_state.benchmark_factor *= 1.0 + info["benchmark_return"]
        pm_state.drawdown_sum += info["drawdown_increment"]
        pm_state.opportunity_sum += info["pm_opportunity_gate"]
        pm_state.window_len += 1
        pm_state.rare_weight_sum += info["train_pm_sample_weight"]
        pm_state.rare_weight_max = np.maximum(pm_state.rare_weight_max, info["train_pm_sample_weight"])
        pm_state.rare_weight_len += 1

        for env_idx in range(num_envs):
            trader_obs_rows[env_idx].append(trader_obs[env_idx].astype(np.float32).copy())
            trader_actions[env_idx].append(trader_action[env_idx].astype(np.float32).copy())
            trader_values[env_idx].append(float(trader_value[env_idx]))
            trader_log_probs[env_idx].append(float(trader_log_prob[env_idx]))
            trader_rewards[env_idx].append(float(info["trader_reward"][env_idx]))
            trader_dones[env_idx].append(1.0 if done else 0.0)
            trader_aux_targets[env_idx].append(collector.trader_aux_targets[int(info["step_day"][env_idx])].astype(np.float32).copy())
            trader_sample_weights[env_idx].append(float(info["train_trader_sample_weight"][env_idx]))
            if trader_critic_obs is not None:
                trader_critic_obs_rows[env_idx].append(trader_critic_obs[env_idx].astype(np.float32).copy())

        if 0 <= keep_trace_env < num_envs:
            idx = int(keep_trace_env)
            trace_rows.append(
                trace_row_from_batch(
                    collector,
                    info,
                    idx=idx,
                    pm_state=pm_state,
                    trader_action=executable_trader_action[idx],
                    trader_action_diagnostics=trader_diag,
                )
            )

    if np.any(pm_state.active):
        final_rewards = collector.pm_window_reward(
            return_factor=pm_state.return_factor,
            benchmark_factor=pm_state.benchmark_factor,
            drawdown_sum=pm_state.drawdown_sum,
            opportunity_sum=pm_state.opportunity_sum,
            window_len=pm_state.window_len,
            q_target=pm_state.q_raw_target,
        )
        append_pm_records(
            pm_state=pm_state,
            mask=pm_state.active,
            rewards=final_rewards,
            done=1.0,
            obs_rows=pm_obs_rows,
            actions=pm_actions,
            log_probs=pm_log_probs,
            values=pm_values,
            out_rewards=pm_rewards,
            dones=pm_dones,
            aux_targets=pm_aux_targets,
            sample_weights=pm_sample_weights,
            sample_weight_values=collector.pm_window_sample_weights(pm_state),
        )

    pm_batch = PolicyBatch(
        obs=flatten_env_lists(pm_obs_rows, dtype=np.float32),
        actions=flatten_env_lists(pm_actions, dtype=np.float32),
        old_log_prob=flatten_env_lists(pm_log_probs, dtype=np.float32),
        values=flatten_env_lists(pm_values, dtype=np.float32),
        rewards=flatten_env_lists(pm_rewards, dtype=np.float32),
        dones=flatten_env_lists(pm_dones, dtype=np.float32),
        aux_targets=flatten_env_lists(pm_aux_targets, dtype=np.float32) if any(pm_aux_targets) else None,
        sample_weight=flatten_env_lists(pm_sample_weights, dtype=np.float32) if any(pm_sample_weights) else None,
    )
    trader_batch = PolicyBatch(
        obs=flatten_env_lists(trader_obs_rows, dtype=np.float32),
        actions=flatten_env_lists(trader_actions, dtype=np.float32),
        old_log_prob=flatten_env_lists(trader_log_probs, dtype=np.float32),
        values=flatten_env_lists(trader_values, dtype=np.float32),
        rewards=flatten_env_lists(trader_rewards, dtype=np.float32),
        dones=flatten_env_lists(trader_dones, dtype=np.float32),
        critic_obs=flatten_env_lists(trader_critic_obs_rows, dtype=np.float32) if any(trader_critic_obs_rows) else None,
        aux_targets=flatten_env_lists(trader_aux_targets, dtype=np.float32) if any(trader_aux_targets) else None,
        sample_weight=flatten_env_lists(trader_sample_weights, dtype=np.float32) if any(trader_sample_weights) else None,
    )
    return pm_batch, trader_batch, pd.DataFrame(trace_rows)


def trace_row_from_batch(
    collector: VectorizedW1Collector,
    info: dict[str, np.ndarray],
    *,
    idx: int,
    pm_state: OpenPMState,
    trader_action: np.ndarray,
    trader_action_diagnostics: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    target = np.asarray(info["target_weights"][idx], dtype=np.float64)
    desired = np.asarray(info.get("desired_weights", info["target_weights"])[idx], dtype=np.float64)
    pre = np.asarray(info["pre_trade_weights"][idx], dtype=np.float64)
    delta = target[: collector.stock_dim] - pre[: collector.stock_dim]
    aux_day = int(info["step_day"][idx])
    trader_aux = collector.trader_aux_targets[aux_day]
    residual_rank = trader_aux[: collector.stock_dim].astype(np.float64)
    group_rank = trader_aux[collector.stock_dim :].astype(np.float64)
    buy = np.maximum(delta, 0.0)
    sell = np.maximum(-delta, 0.0)
    buy_sum = float(buy.sum())
    sell_sum = float(sell.sum())
    buy_residual_rank = float(np.sum(buy * residual_rank) / buy_sum) if buy_sum > EPS else np.nan
    sell_residual_rank = float(np.sum(sell * residual_rank) / sell_sum) if sell_sum > EPS else np.nan
    buy_group_rank = float(np.sum(buy * group_rank) / buy_sum) if buy_sum > EPS else np.nan
    sell_group_rank = float(np.sum(sell * group_rank) / sell_sum) if sell_sum > EPS else np.nan
    row: dict[str, Any] = {
        "date": info["date"][idx],
        "next_date": info["next_date"][idx],
        "net_return": float(info["net_return"][idx]),
        "gross_return": float(info["gross_return"][idx]),
        "benchmark_return": float(info["benchmark_return"][idx]),
        "pm_reward": float(info["pm_reward"][idx]),
        "trader_reward": float(info["trader_reward"][idx]),
        "q_raw_target": float(info["q_raw_target"][idx]),
        "q_target": float(info["q_target"][idx]),
        "q_exec": float(info["q_exec"][idx]),
        "desired_q": float(info.get("desired_q", info.get("execution_desired_q"))[idx]) if ("desired_q" in info or "execution_desired_q" in info) else np.nan,
        "cash_target_pm": float(info["cash_target_pm"][idx]),
        "cash_exec": float(info["cash_exec"][idx]),
        "desired_cash": float(info.get("desired_cash", info.get("execution_desired_cash"))[idx]) if ("desired_cash" in info or "execution_desired_cash" in info) else np.nan,
        "tracking_l1": float(info["tracking_l1"][idx]),
        "remaining_days": int(info["remaining_days"][idx]),
        "remaining_days_normalized": float(info["remaining_days_normalized"][idx]),
        "pm_horizon_days": int(pm_state.horizon_days[idx]),
        "pm_start_day": int(pm_state.start_day[idx]),
        "curriculum_episode_start_day": int(collector.episode_start_day),
        "curriculum_episode_end_day": int(collector.episode_end_day),
        "pm_aux_future20_opportunity_label": float(collector.pm_opportunity_label[aux_day]),
        "pm_aux_future20_stress_label": float(collector.pm_stress_label[aux_day]),
        "trader_aux_buy_residual_rank_weighted": buy_residual_rank,
        "trader_aux_sell_residual_rank_weighted": sell_residual_rank,
        "trader_aux_buy_group_rank_weighted": buy_group_rank,
        "trader_aux_sell_group_rank_weighted": sell_group_rank,
        "turnover_l1": float(info["turnover_l1"][idx]),
        "stock_turnover_l1": float(info["stock_turnover_l1"][idx]),
        "transaction_cost": float(info["transaction_cost"][idx]),
        "drawdown": float(info["drawdown"][idx]),
        "drawdown_increment": float(info["drawdown_increment"][idx]),
        "risky_entropy": float(info["risky_entropy"][idx]),
        "portfolio_entropy": float(info["portfolio_entropy"][idx]),
        "trader_flow_select_5d": float(info["trader_flow_select_5d"][idx]),
        "trader_group_relative_5d": float(info["trader_group_relative_5d"][idx]),
        "trader_entropy_bonus": float(info["trader_entropy_bonus"][idx]),
        "trader_entropy_reward_term": float(info["trader_entropy_reward_term"][idx]),
        "trader_vol_adjusted_position_change": float(info["trader_vol_adjusted_position_change"][idx]),
        "trader_tracking_multiplier": float(info["trader_tracking_multiplier"][idx]),
        "trader_tracking_penalty_term": float(info["trader_tracking_penalty_term"][idx]),
        "pm_opportunity_gate": float(info["pm_opportunity_gate"][idx]),
        "pm_cash_opportunity_cost": float(info["pm_cash_opportunity_cost"][idx]),
        "pm_cash_opportunity_prior_cost": float(info["pm_cash_opportunity_prior_cost"][idx]),
        "pm_active_return": float(info["pm_active_return"][idx]),
        "trader_action_cash_weight": float(np.asarray(trader_action).reshape(-1)[collector.cash_index]),
        "trader_feedback_change_after_step": float(collector.trader_reward_change[idx]),
    }
    for key, values in (trader_action_diagnostics or {}).items():
        arr = np.asarray(values).reshape(-1)
        if len(arr) > idx:
            row[key] = float(arr[idx]) if np.issubdtype(arr.dtype, np.floating) else int(arr[idx])
    for aux_idx, name in enumerate(getattr(collector, "pm_aux_target_names", [])):
        if aux_idx < pm_state.aux_target.shape[1]:
            row[f"pm_aux_{name}"] = float(pm_state.aux_target[idx, aux_idx])
    for key, value in info.items():
        if (
            key.startswith("rare_")
            or key.startswith("execution_")
            or key.startswith("risk_stop_")
            or key.startswith("scenario_")
            or key.startswith("supervisor_")
            or key.startswith("train_")
        ):
            arr = np.asarray(value)
            row[key] = float(arr[idx]) if arr.ndim > 0 else float(arr)
    for ticker, value in zip(collector.panel.tickers, desired[: collector.stock_dim]):
        row[f"desired_weight_{ticker}"] = float(value)
    row["desired_weight_CASH"] = float(desired[collector.cash_index])
    for ticker, value in zip(collector.panel.tickers, target[: collector.stock_dim]):
        row[f"target_weight_{ticker}"] = float(value)
    row["target_weight_CASH"] = float(target[collector.cash_index])
    for ticker, value in zip(collector.panel.tickers, pre[: collector.stock_dim]):
        row[f"pre_weight_{ticker}"] = float(value)
    row["pre_weight_CASH"] = float(pre[collector.cash_index])
    return row


def prioritized_replay_update(
    *,
    policy: Any,
    replay: PrioritizedPolicyReplay,
    updates: int,
    batch_size: int,
    beta: float,
    is_clip: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    reference_policy: Any | None = None,
    reference_kl_coef: float = 0.0,
    aux_loss_coef: float = 0.0,
    aux_loss_kind: str = "mse",
) -> dict[str, float]:
    if len(replay) == 0 or int(updates) <= 0:
        return {"updates": 0.0, "replay_size": float(len(replay))}
    device = policy.device
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
        "td_error_abs": [],
        "sample_probability_mean": [],
        "is_weight_mean": [],
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

    for _ in range(int(max(1, updates))):
        sample = replay.sample(int(batch_size), beta=beta, is_clip=is_clip)
        batch = sample.batch
        obs_t = as_tensor(batch.obs, device=device)
        actions_t = as_tensor(batch.actions, device=device)
        old_log_t = as_tensor(batch.old_log_prob, device=device)
        returns_t = as_tensor(sample.returns, device=device)
        critic_obs_t = as_tensor(batch.critic_obs, device=device) if batch.critic_obs is not None else None
        aux_targets_t = as_tensor(batch.aux_targets, device=device) if batch.aux_targets is not None else None
        sample_weight = np.ones(len(batch.obs), dtype=np.float32) if batch.sample_weight is None else batch.sample_weight
        sample_weight = np.nan_to_num(sample_weight.astype(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
        sample_weight = np.maximum(sample_weight, 1e-6)
        sample_weight = sample_weight / max(float(sample_weight.mean()), 1e-6)
        sample_weight_t = as_tensor(sample_weight, device=device)
        adv = sample.advantages.astype(np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        adv_t = as_tensor(adv, device=device)

        values, log_prob, entropy = policy.evaluate_actions(obs_t, actions_t, critic_obs=critic_obs_t)
        values = values.flatten()
        ratio = th.exp(log_prob - old_log_t)
        pg_loss_1 = adv_t * ratio
        pg_loss_2 = adv_t * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -weighted_mean(th.min(pg_loss_1, pg_loss_2), sample_weight_t)
        value_loss = weighted_mean(th.nn.functional.mse_loss(values, returns_t, reduction="none"), sample_weight_t)
        entropy_loss = -weighted_mean(log_prob, sample_weight_t) if entropy is None else -weighted_mean(entropy, sample_weight_t)
        reference_kl = th.zeros((), dtype=values.dtype, device=values.device)
        if reference_policy is not None and float(reference_kl_coef) > 0.0:
            if not hasattr(policy, "_dists") or not hasattr(reference_policy, "_dists"):
                raise ValueError("reference_policy KL regularization requires policies with a _dists(obs) method")
            current_outputs = policy._dists(obs_t)
            with th.no_grad():
                reference_outputs = reference_policy._dists(obs_t)
            reference_kl = weighted_mean(reference_kl_divergence(reference_outputs, current_outputs), sample_weight_t)
        aux_loss = th.zeros((), dtype=values.dtype, device=values.device)
        aux_accuracy = th.zeros((), dtype=values.dtype, device=values.device)
        if aux_targets_t is not None and float(aux_loss_coef) > 0.0:
            if not hasattr(policy, "aux_predictions"):
                raise ValueError("aux_loss requested, but policy has no aux_predictions method")
            aux_pred = policy.aux_predictions(obs_t)
            kind = str(aux_loss_kind).lower()
            if kind in {"bce", "binary", "binary_cross_entropy"}:
                per_item = th.nn.functional.binary_cross_entropy_with_logits(aux_pred, aux_targets_t, reduction="none").mean(dim=1)
                aux_loss = weighted_mean(per_item, sample_weight_t)
                with th.no_grad():
                    per_item_accuracy = ((th.sigmoid(aux_pred) >= 0.5) == (aux_targets_t >= 0.5)).float().mean(dim=1)
                    aux_accuracy = weighted_mean(per_item_accuracy, sample_weight_t)
            elif kind in {"mse", "regression"}:
                per_item = th.nn.functional.mse_loss(aux_pred, aux_targets_t, reduction="none").mean(dim=1)
                aux_loss = weighted_mean(per_item, sample_weight_t)
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
            new_values, new_log_prob, _new_entropy = policy.evaluate_actions(obs_t, actions_t, critic_obs=critic_obs_t)
            new_values = new_values.flatten()
            td_error = (returns_t - new_values).detach().cpu().numpy()
            replay.update_priorities(sample.indices, td_error)
            log_ratio = new_log_prob - old_log_t
            ratio_after = th.exp(log_ratio)
            approx_kl = weighted_mean((ratio_after - 1.0) - log_ratio, sample_weight_t).detach().cpu().item()
            clipped = weighted_mean((th.abs(ratio_after - 1.0) > clip_range).float(), sample_weight_t).detach().cpu().item()

        stats["policy_loss"].append(float(policy_loss.detach().cpu()))
        stats["value_loss"].append(float(value_loss.detach().cpu()))
        stats["entropy_loss"].append(float(entropy_loss.detach().cpu()))
        stats["approx_kl"].append(float(approx_kl))
        stats["clip_fraction"].append(float(clipped))
        stats["reference_kl"].append(float(reference_kl.detach().cpu()))
        stats["aux_loss"].append(float(aux_loss.detach().cpu()))
        stats["aux_accuracy"].append(float(aux_accuracy.detach().cpu()))
        stats["loss"].append(float(loss.detach().cpu()))
        stats["td_error_abs"].append(float(np.mean(np.abs(td_error))))
        stats["sample_probability_mean"].append(float(np.mean(sample.probabilities)))
        stats["is_weight_mean"].append(float(np.mean(sample.is_weights)))

    out = {key: float(np.mean(values)) for key, values in stats.items() if values}
    out.update({f"buffer_{k}": v for k, v in replay.diagnostics().items()})
    return out | {"updates": float(len(stats["loss"])), "replay_size": float(len(replay))}


def train_one(config: dict[str, Any], variant: dict[str, Any], fold: pd.Series, *, out_root: Path, smoke_test: bool, force: bool) -> dict[str, Any]:
    fold_id = str(fold["fold"])
    variant_name = str(variant["name"])
    vector_cfg = dict(variant.get("w1_vectorized", {}))
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
    template_env = W1BudgetTraderEnv(train_panel, **env_kwargs(config, variant, train_panel))
    pm_policy, trader_policy = make_policies(template_env, variant, fold_id=fold_id)
    pm_warm_start_info = maybe_load_pm_warm_start(pm_policy, variant, fold_id=fold_id, run_dir=run_dir)
    warm_start_info = maybe_load_trader_warm_start(trader_policy, variant, run_dir=run_dir, fold_id=fold_id)
    trader_prior_policy, trader_prior_info = maybe_make_trader_behavior_prior(template_env, variant, fold_id=fold_id, run_dir=run_dir)
    trader_prior_coef = float(trader_prior_info.get("kl_coef", 0.0)) if trader_prior_policy is not None else 0.0

    joint_cfg = variant.get("w1", {}).get("joint_ppo", {})
    aux_cfg = variant.get("w1", {}).get("aux_training", {})
    rare_event_cfg = variant.get("w1", {}).get("rare_event_training", {})
    scenario_cfg = variant.get("w1", {}).get("scenario_training", {})
    curriculum_cfg = variant.get("w1", {}).get("curriculum_sampling", {})
    per_cfg = variant.get("w1", {}).get("prioritized_replay", {})
    supervisor_cfg = variant.get("w1", {}).get("closed_loop_supervisor", {})
    num_envs = int(vector_cfg.get("num_envs", 8))
    seed = int(vector_cfg.get("seed", 17))
    target_days = int(joint_cfg.get("total_internal_trading_days", 70_000))
    if smoke_test:
        target_days = min(target_days, 4096)
        num_envs = min(num_envs, 4)
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
    pm_aux_loss_coef = float(aux_cfg.get("pm_loss_coef", 0.0))
    trader_aux_loss_coef = float(aux_cfg.get("trader_loss_coef", 0.0))
    per_enabled = bool(per_cfg.get("enabled", False))
    per_pm_enabled = per_enabled and bool(per_cfg.get("pm_enabled", True))
    per_trader_enabled = per_enabled and bool(per_cfg.get("trader_enabled", True))
    per_alpha = float(per_cfg.get("alpha", 0.60))
    per_beta_start = float(per_cfg.get("beta_start", 0.40))
    per_beta_end = float(per_cfg.get("beta_end", 1.00))
    per_priority_eps = float(per_cfg.get("priority_eps", 1e-4))
    per_max_priority = float(per_cfg.get("max_priority", 100.0))
    per_is_clip = float(per_cfg.get("importance_weight_clip", 5.0))
    per_min_size_pm = int(per_cfg.get("pm_min_size", 256))
    per_min_size_trader = int(per_cfg.get("trader_min_size", 2048))
    per_pm_updates = int(per_cfg.get("pm_updates_per_iteration", 0))
    per_trader_updates = int(per_cfg.get("trader_updates_per_iteration", 0))
    per_pm_batch_size = int(per_cfg.get("pm_batch_size", pm_batch_size))
    per_trader_batch_size = int(per_cfg.get("trader_batch_size", trader_batch_size))

    collector = VectorizedW1Collector(
        template_env,
        num_envs=num_envs,
        seed=seed,
        rare_event_config=rare_event_cfg,
        aux_config=aux_cfg,
        scenario_config=scenario_cfg,
        curriculum_config=curriculum_cfg,
    )
    closed_loop_supervisor = ClosedLoopActionSupervisor(
        config=supervisor_cfg,
        fold_id=fold_id,
        panel=train_panel,
        trader_policy=trader_policy,
        seed=seed + 303,
    )
    pm_replay = (
        PrioritizedPolicyReplay(
            capacity=int(per_cfg.get("pm_capacity", 10_000)),
            alpha=per_alpha,
            priority_eps=per_priority_eps,
            max_priority=per_max_priority,
            seed=seed + 101,
        )
        if per_pm_enabled
        else None
    )
    trader_replay = (
        PrioritizedPolicyReplay(
            capacity=int(per_cfg.get("trader_capacity", 100_000)),
            alpha=per_alpha,
            priority_eps=per_priority_eps,
            max_priority=per_max_priority,
            seed=seed + 202,
        )
        if per_trader_enabled
        else None
    )
    processed_days = 0
    iteration_rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    iteration = 0
    started = time.perf_counter()
    while processed_days < target_days:
        iteration += 1
        pm_batch, trader_batch, trace = collect_vectorized_episode(
            collector,
            pm_policy,
            trader_policy,
            keep_trace_env=0,
            supervisor=closed_loop_supervisor,
        )
        pm_adv, pm_returns = compute_gae(pm_batch, gamma=gamma_pm, gae_lambda=gae_lambda)
        trader_adv, trader_returns = compute_gae(trader_batch, gamma=gamma_trader, gae_lambda=gae_lambda)
        if pm_replay is not None:
            pm_replay.add(pm_batch, advantages=pm_adv, returns=pm_returns)
        if trader_replay is not None:
            trader_replay.add(trader_batch, advantages=trader_adv, returns=trader_returns)
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
            aux_loss_coef=pm_aux_loss_coef,
            aux_loss_kind=str(aux_cfg.get("pm_loss_kind", "bce")),
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
            reference_policy=trader_prior_policy,
            reference_kl_coef=trader_prior_coef,
            aux_loss_coef=trader_aux_loss_coef,
            aux_loss_kind=str(aux_cfg.get("trader_loss_kind", "mse")),
        )
        beta_progress = min(1.0, float(processed_days) / max(float(target_days), 1.0))
        per_beta = per_beta_start + beta_progress * (per_beta_end - per_beta_start)
        pm_replay_stats: dict[str, float] = {"updates": 0.0, "replay_size": float(len(pm_replay)) if pm_replay is not None else 0.0}
        trader_replay_stats: dict[str, float] = {"updates": 0.0, "replay_size": float(len(trader_replay)) if trader_replay is not None else 0.0}
        if pm_replay is not None and len(pm_replay) >= per_min_size_pm and per_pm_updates > 0:
            pm_replay_stats = prioritized_replay_update(
                policy=pm_policy,
                replay=pm_replay,
                updates=per_pm_updates,
                batch_size=per_pm_batch_size,
                beta=per_beta,
                is_clip=per_is_clip,
                clip_range=clip_range,
                ent_coef=pm_ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                aux_loss_coef=pm_aux_loss_coef,
                aux_loss_kind=str(aux_cfg.get("pm_loss_kind", "bce")),
            )
        if trader_replay is not None and len(trader_replay) >= per_min_size_trader and per_trader_updates > 0:
            trader_replay_stats = prioritized_replay_update(
                policy=trader_policy,
                replay=trader_replay,
                updates=per_trader_updates,
                batch_size=per_trader_batch_size,
                beta=per_beta,
                is_clip=per_is_clip,
                clip_range=clip_range,
                ent_coef=trader_ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                reference_policy=trader_prior_policy,
                reference_kl_coef=trader_prior_coef,
                aux_loss_coef=trader_aux_loss_coef,
                aux_loss_kind=str(aux_cfg.get("trader_loss_kind", "mse")),
            )
        processed_days += len(trader_batch.rewards)
        returns = trace["net_return"].to_numpy(dtype=np.float64) if len(trace) else np.zeros(0)
        elapsed = max(time.perf_counter() - started, 1e-9)
        iteration_rows.append(
            {
                "iteration": iteration,
                "processed_internal_trading_days": processed_days,
                "episode_days_per_env": int(len(trace)),
                "num_envs": num_envs,
                "collected_trader_transitions": int(len(trader_batch.rewards)),
                "collection_days_per_second": float(len(trader_batch.rewards) / elapsed),
                "episode_return_pct_env0": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
                "episode_cash_exec_mean_env0": float(trace["cash_exec"].mean()) if len(trace) else 0.0,
                "episode_q_target_mean_env0": float(trace["q_target"].mean()) if len(trace) else 0.0,
                "episode_q_exec_mean_env0": float(trace["q_exec"].mean()) if len(trace) else 0.0,
                "episode_tracking_l1_mean_env0": float(trace["tracking_l1"].mean()) if len(trace) else 0.0,
                "episode_horizon_mean_env0": float(trace["pm_horizon_days"].mean()) if len(trace) else 0.0,
                "episode_rare_pm_sample_weight_mean_env0": float(trace["rare_pm_sample_weight"].mean()) if "rare_pm_sample_weight" in trace else 1.0,
                "episode_rare_trader_sample_weight_mean_env0": float(trace["rare_trader_sample_weight"].mean()) if "rare_trader_sample_weight" in trace else 1.0,
                "episode_scenario_pm_sample_weight_mean_env0": float(trace["scenario_pm_sample_weight"].mean()) if "scenario_pm_sample_weight" in trace else 1.0,
                "episode_scenario_trader_sample_weight_mean_env0": float(trace["scenario_trader_sample_weight"].mean()) if "scenario_trader_sample_weight" in trace else 1.0,
                "episode_train_pm_sample_weight_mean_env0": float(trace["train_pm_sample_weight"].mean()) if "train_pm_sample_weight" in trace else 1.0,
                "episode_train_trader_sample_weight_mean_env0": float(trace["train_trader_sample_weight"].mean()) if "train_trader_sample_weight" in trace else 1.0,
                "episode_supervisor_intervention_rate_env0": float(trace["supervisor_intervened"].mean()) if "supervisor_intervened" in trace else 0.0,
                "episode_supervisor_confidence_mean_env0": float(trace["supervisor_confidence"].mean()) if "supervisor_confidence" in trace else 0.0,
                "episode_scenario_recovery_mean_env0": float(trace["scenario_recovery"].mean()) if "scenario_recovery" in trace else 0.0,
                "episode_scenario_stress_mean_env0": float(trace["scenario_stress"].mean()) if "scenario_stress" in trace else 0.0,
                "episode_scenario_calm_mean_env0": float(trace["scenario_calm"].mean()) if "scenario_calm" in trace else 0.0,
                "episode_rare_bad_buy_mean_env0": float(trace["rare_bad_buy"].mean()) if "rare_bad_buy" in trace else 0.0,
                "episode_rare_bad_sell_mean_env0": float(trace["rare_bad_sell"].mean()) if "rare_bad_sell" in trace else 0.0,
                "episode_rare_cash_lock_mean_env0": float(trace["rare_cash_lock"].mean()) if "rare_cash_lock" in trace else 0.0,
                "episode_risk_stop_active_mean_env0": float(trace["risk_stop_active"].mean()) if "risk_stop_active" in trace else 0.0,
                "episode_risk_stop_gap_mean_env0": float(trace["risk_stop_gap"].mean()) if "risk_stop_gap" in trace else 0.0,
                "episode_q_raw_target_mean_env0": float(trace["q_raw_target"].mean()) if "q_raw_target" in trace else 0.0,
                "trader_reference_kl": float(trader_stats.get("reference_kl", 0.0)),
                "pm_aux_loss": float(pm_stats.get("aux_loss", 0.0)),
                "pm_aux_accuracy": float(pm_stats.get("aux_accuracy", 0.0)),
                "trader_aux_loss": float(trader_stats.get("aux_loss", 0.0)),
                "per_enabled": float(per_enabled),
                "per_beta": float(per_beta),
                **{f"pm_per_{k}": v for k, v in pm_replay_stats.items()},
                **{f"trader_per_{k}": v for k, v in trader_replay_stats.items()},
                "pm_transitions": int(len(pm_batch.rewards)),
                "trader_transitions": int(len(trader_batch.rewards)),
                **{f"pm_{k}": v for k, v in pm_stats.items()},
                **{f"trader_{k}": v for k, v in trader_stats.items()},
            }
        )
        if iteration <= int(joint_cfg.get("save_train_trace_episodes", 1)):
            trace = trace.copy()
            trace["iteration"] = iteration
            trace["vector_env_id"] = 0
            trace_frames.append(trace)
        print(
            f"[W1-vec] iter={iteration} envs={num_envs} days={processed_days}/{target_days} "
            f"pm_kl={pm_stats.get('approx_kl', np.nan):.5f} "
            f"trader_kl={trader_stats.get('approx_kl', np.nan):.5f} "
            f"ref_kl={trader_stats.get('reference_kl', 0.0):.5f} "
            f"aux=({pm_stats.get('aux_loss', 0.0):.4f},{trader_stats.get('aux_loss', 0.0):.4f}) "
            f"sup={iteration_rows[-1].get('episode_supervisor_intervention_rate_env0', 0.0):.3f} "
            f"per=({pm_replay_stats.get('updates', 0.0):.0f},{trader_replay_stats.get('updates', 0.0):.0f})",
            flush=True,
        )

    pd.DataFrame(iteration_rows).to_csv(run_dir / "train_joint_iterations.csv", index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(run_dir / "train_trace_daily.csv", index=False)
    th.save({"state_dict": pm_policy.state_dict(), "variant": variant}, run_dir / "pm_policy.pt")
    th.save({"state_dict": trader_policy.state_dict(), "variant": variant}, run_dir / "trader_policy.pt")

    val_env = W1BudgetTraderEnv(validation_panel, **env_kwargs(config, variant, validation_panel))
    if closed_loop_supervisor is not None:
        closed_loop_supervisor.last_intervention_day = None
    validation_summary = evaluate(
        val_env,
        pm_policy,
        trader_policy,
        out_dir=run_dir,
        split_name="validation",
        action_supervisor=closed_loop_supervisor,
    )
    metadata = {
        "variant": variant,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "target_internal_trading_days": target_days,
        "observed_internal_trading_days": processed_days,
        "vectorized_train_envs": num_envs,
        "pm_obs_dim": template_env.pm_obs_dim,
        "trader_obs_dim": template_env.trader_obs_dim,
        "trader_action_semantics": "full_portfolio_dirichlet_stocks_plus_cash",
        "stock_feature_dim": template_env.stock_feature_dim,
        "trader_task_dim": template_env.trader_task_dim,
        "horizon_choices": template_env.horizon_choices,
        "pm_warm_start": pm_warm_start_info,
        "trader_warm_start": warm_start_info,
        "trader_behavior_prior": trader_prior_info,
        "rare_event_training": {
            "config": rare_event_cfg,
            "enabled": bool(rare_event_cfg.get("enabled", False)),
        },
        "scenario_training": {
            "config": scenario_cfg,
            "enabled": bool(scenario_cfg.get("enabled", False)),
        },
        "curriculum_sampling": {
            "config": curriculum_cfg,
            "enabled": bool(curriculum_cfg.get("enabled", False)),
        },
        "prioritized_replay": {
            "config": per_cfg,
            "enabled": per_enabled,
            "pm_enabled": per_pm_enabled,
            "trader_enabled": per_trader_enabled,
            "pm_buffer": pm_replay.diagnostics() if pm_replay is not None else {"size": 0.0},
            "trader_buffer": trader_replay.diagnostics() if trader_replay is not None else {"size": 0.0},
            "priority_definition": "abs(gae_return_or_td_target - current_value)",
        },
        "closed_loop_supervisor": {
            "config": supervisor_cfg,
            "summary": closed_loop_supervisor.summary,
            "enabled": bool(supervisor_cfg.get("enabled", False)),
            "leakage_rule": "online rollout context uses only current panel features; context-code table is train-split teacher-derived",
        },
        "aux_training": {
            "config": aux_cfg,
            "target_info": collector.aux_target_info,
            "pm_aux_dim": int(collector.pm_aux_targets.shape[1]),
            "trader_aux_dim": int(collector.trader_aux_targets.shape[1]),
        },
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
            "vectorized_train_envs": num_envs,
        }
    )
    return validation_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated/stage0_1_w1_budget_trader.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default="weight_based_w1_budget_trader_vectorized_batch")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or "weight_based_w1_budget_trader_vectorized_batch"
    if args.smoke_test:
        run_name += "_smoke"
    out_root = resolve(config.get("output", {}).get("root_dir", "artifacts/stage0_1")) / run_name
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)
    rows = []
    for variant in variants:
        variant = dict(variant)
        variant.setdefault("w1_vectorized", {})
        for _, fold in folds.iterrows():
            print(f"\n=== W1 vectorized: variant={variant['name']} fold={fold['fold']} ===", flush=True)
            rows.append(train_one(config, variant, fold, out_root=out_root, smoke_test=args.smoke_test, force=args.force))
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "run_summary.csv", index=False)
    print(f"\nW1 vectorized run written to {out_root}")


if __name__ == "__main__":
    main()
