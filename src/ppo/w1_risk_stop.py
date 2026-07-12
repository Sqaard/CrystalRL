"""Deterministic W1 risk-stop layer.

The risk-stop layer is an explicit safety veto placed after the PM raw risk
budget and before the Trader task.  It must not enter the policy observation or
the action log-prob contract: PPO still samples the raw PM action, while the
environment logs the deterministic raw-to-safe correction.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1e-8


def _feature(features: Mapping[str, float], name: str, default: float = 0.0) -> float:
    value = features.get(name, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    return out if np.isfinite(out) else float(default)


def features_from_panel_row(feature_columns: Sequence[str], row: np.ndarray) -> dict[str, float]:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    return {
        str(name): float(values[idx])
        for idx, name in enumerate(feature_columns)
        if idx < values.shape[0] and np.isfinite(values[idx])
    }


def compute_risk_stop(
    q_raw: np.ndarray | float,
    *,
    features: Mapping[str, float],
    current_drawdown: np.ndarray | float,
    trader_feedback_change: np.ndarray | float,
    config: Mapping[str, Any] | None,
    q_min: float,
    q_max: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return safe q and diagnostics for a deterministic risk-stop config."""

    cfg = dict(config or {})
    q = np.asarray(q_raw, dtype=np.float64)
    scalar = q.ndim == 0
    q = q.reshape(-1)
    q_clipped = np.clip(q, float(q_min), float(q_max))
    if not bool(cfg.get("enabled", False)):
        zeros = np.zeros_like(q_clipped, dtype=np.float64)
        info = {
            "risk_stop_q_raw": q_clipped.copy(),
            "risk_stop_q_safe": q_clipped.copy(),
            "risk_stop_active": zeros,
            "risk_stop_score": zeros,
            "risk_stop_cap": np.full_like(q_clipped, float(q_max)),
            "risk_stop_gap": zeros,
            "risk_stop_market_down": zeros,
            "risk_stop_vix_shock": zeros,
            "risk_stop_turbulence": zeros,
            "risk_stop_drawdown_abs": zeros,
            "risk_stop_feedback_bad": zeros,
        }
        return (q_clipped[0] if scalar else q_clipped), info

    drawdown = np.asarray(current_drawdown, dtype=np.float64).reshape(-1)
    feedback = np.asarray(trader_feedback_change, dtype=np.float64).reshape(-1)
    if drawdown.size == 1 and q_clipped.size > 1:
        drawdown = np.repeat(drawdown, q_clipped.size)
    if feedback.size == 1 and q_clipped.size > 1:
        feedback = np.repeat(feedback, q_clipped.size)

    trend = _feature(features, str(cfg.get("trend_feature", "SP500_Trend")))
    vix_change = _feature(features, str(cfg.get("vix_change_feature", "VIX_change_5d")))
    turbulence = _feature(features, str(cfg.get("turbulence_feature", "turbulence")))
    turbulence_delta = _feature(features, str(cfg.get("turbulence_delta_feature", "turbulence_delta_1d")))
    regime_entropy = _feature(features, str(cfg.get("regime_entropy_feature", "regime_entropy")))

    market_down = max(-trend, 0.0)
    vix_shock = max(vix_change, 0.0)
    turbulence_pos = max(turbulence, 0.0) + 0.5 * max(turbulence_delta, 0.0)
    drawdown_abs = np.maximum(-drawdown, 0.0)
    feedback_bad = np.maximum(-feedback, 0.0)

    score_linear = (
        float(cfg.get("market_down_weight", 0.80)) * market_down
        + float(cfg.get("vix_shock_weight", 0.70)) * vix_shock
        + float(cfg.get("turbulence_weight", 0.50)) * turbulence_pos
        + float(cfg.get("regime_entropy_weight", 0.20)) * max(regime_entropy, 0.0)
        + float(cfg.get("drawdown_weight", 2.00)) * drawdown_abs
        + float(cfg.get("feedback_weight", 0.50)) * feedback_bad
        - float(cfg.get("score_bias", 1.25))
    )
    score = 1.0 / (1.0 + np.exp(-np.clip(score_linear, -50.0, 50.0)))
    threshold = float(cfg.get("score_threshold", 0.70))
    max_q_stress = float(cfg.get("max_q_stress", 0.70))
    max_q_extreme = float(cfg.get("max_q_extreme", max_q_stress))
    extreme_threshold = float(cfg.get("extreme_score_threshold", 0.90))
    cap = np.where(score >= extreme_threshold, max_q_extreme, max_q_stress)
    active = (score >= threshold) & (q_clipped > cap)
    safe_q = np.where(active, np.minimum(q_clipped, cap), q_clipped)
    safe_q = np.clip(safe_q, float(q_min), float(q_max))
    gap = q_clipped - safe_q
    info = {
        "risk_stop_q_raw": q_clipped.copy(),
        "risk_stop_q_safe": safe_q.copy(),
        "risk_stop_active": active.astype(np.float64),
        "risk_stop_score": score.astype(np.float64),
        "risk_stop_cap": cap.astype(np.float64),
        "risk_stop_gap": gap.astype(np.float64),
        "risk_stop_market_down": np.full_like(q_clipped, market_down),
        "risk_stop_vix_shock": np.full_like(q_clipped, vix_shock),
        "risk_stop_turbulence": np.full_like(q_clipped, turbulence_pos),
        "risk_stop_drawdown_abs": drawdown_abs.astype(np.float64),
        "risk_stop_feedback_bad": feedback_bad.astype(np.float64),
    }
    return (safe_q[0] if scalar else safe_q), info
