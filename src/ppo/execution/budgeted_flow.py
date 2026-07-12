"""Budgeted execution layer for W1 PM/Trader portfolio weights.

The policy still proposes a desired full-portfolio target.  This module decides
how much of that desired move is executed today by separating net risk/cash flow
from stock-to-stock rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.ppo.execution.helpers import EPS, normalize_simplex, normalize_stock_simplex


@dataclass
class BudgetedExecutionResult:
    desired_weights: np.ndarray
    executed_weights: np.ndarray
    diagnostics: dict[str, float]


def _l1(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(values, dtype=np.float64))))


def _risky_allocation(weights: np.ndarray, stock_dim: int, fallback: np.ndarray | None = None) -> np.ndarray:
    q = float(np.sum(weights[:stock_dim]))
    if q > EPS:
        return normalize_stock_simplex(weights[:stock_dim] / q)
    if fallback is not None:
        return normalize_stock_simplex(fallback)
    return np.full(stock_dim, 1.0 / max(stock_dim, 1), dtype=np.float64)


def _net_budget(
    *,
    pm_gap: float,
    q_request: float,
    remaining_days: int,
    config: dict[str, Any],
) -> tuple[float, float, float]:
    remaining = max(1, int(remaining_days))
    scheduled_gap = float(pm_gap) / float(remaining)
    base = abs(scheduled_gap) * float(config.get("net_budget_multiplier", 1.0))
    if abs(pm_gap) > EPS or bool(config.get("always_apply_min_net_budget", False)):
        base = max(base, float(config.get("min_net_budget_l1", 0.0)))
    same_direction = (
        abs(pm_gap) <= EPS
        or abs(q_request) <= EPS
        or np.sign(pm_gap) == np.sign(q_request)
    )
    if not same_direction:
        base *= float(config.get("counter_direction_net_multiplier", 0.50))
    if bool(config.get("final_day_catchup", True)) and remaining <= 1:
        base = max(base, abs(pm_gap))
    if bool(config.get("prevent_pm_overshoot", True)) and same_direction and abs(pm_gap) > EPS:
        base = min(base, abs(pm_gap))
    return float(max(base, 0.0)), float(scheduled_gap), float(same_direction)


def _net_budget_with_guaranteed_schedule(
    *,
    pm_gap: float,
    q_request: float,
    remaining_days: int,
    config: dict[str, Any],
    primitive_strength: float,
) -> tuple[float, float, float, float, float]:
    """Guarantee PM scheduled flow, scale only extra desired flow by strength."""

    remaining = max(1, int(remaining_days))
    scheduled_gap = float(pm_gap) / float(remaining)
    scheduled_budget = abs(scheduled_gap) * float(config.get("net_budget_multiplier", 1.0))
    if abs(pm_gap) > EPS or bool(config.get("always_apply_min_net_budget", False)):
        scheduled_budget = max(scheduled_budget, float(config.get("min_net_budget_l1", 0.0)))
    same_direction = (
        abs(pm_gap) <= EPS
        or abs(q_request) <= EPS
        or np.sign(pm_gap) == np.sign(q_request)
    )
    if not same_direction:
        scheduled_budget *= float(config.get("counter_direction_net_multiplier", 1.0))
    if bool(config.get("final_day_catchup", True)) and remaining <= 1:
        scheduled_budget = max(scheduled_budget, abs(pm_gap))
    if bool(config.get("prevent_pm_overshoot", True)) and same_direction and abs(pm_gap) > EPS:
        scheduled_budget = min(scheduled_budget, abs(pm_gap))

    requested = abs(float(q_request))
    extra_request = max(0.0, requested - scheduled_budget)
    extra_budget = (
        extra_request
        * float(config.get("extra_net_budget_fraction", 1.0))
        * max(float(config.get("extra_strength_floor", 0.0)) + float(config.get("extra_strength_scale", 1.0)) * primitive_strength, 0.0)
    )
    max_extra = config.get("max_extra_net_budget_l1", None)
    if max_extra is not None:
        extra_budget = min(extra_budget, float(max_extra))
    net_budget = scheduled_budget + max(0.0, extra_budget)
    if bool(config.get("prevent_pm_overshoot", True)) and same_direction and abs(pm_gap) > EPS:
        net_budget = min(net_budget, max(abs(pm_gap), scheduled_budget))
    return (
        float(max(net_budget, 0.0)),
        float(scheduled_gap),
        float(same_direction),
        float(max(scheduled_budget, 0.0)),
        float(max(extra_budget, 0.0)),
    )


def apply_budgeted_flow_execution(
    *,
    previous_weights: np.ndarray,
    desired_weights: np.ndarray,
    q_target: float,
    remaining_days: int,
    stock_dim: int,
    cash_index: int,
    config: dict[str, Any] | None = None,
    primitive_strength: float | None = None,
    opportunity_score: float | None = None,
    q_min: float = 0.0,
    q_max: float = 1.0,
) -> BudgetedExecutionResult:
    """Convert desired portfolio weights into today's executed portfolio weights."""

    cfg = dict(config or {})
    prev = normalize_simplex(previous_weights)
    desired = normalize_simplex(desired_weights)
    mode = str(cfg.get("mode", "direct")).lower()
    if mode in {"", "direct", "none"}:
        executed = desired
        desired_q = float(desired[:stock_dim].sum())
        diagnostics = {
            "execution_mode_budgeted_flow": 0.0,
            "execution_desired_q": desired_q,
            "execution_executed_q": desired_q,
            "execution_desired_cash": float(desired[cash_index]),
            "execution_executed_cash": float(executed[cash_index]),
            "execution_desired_to_prev_l1": _l1(desired - prev),
            "execution_executed_to_prev_l1": _l1(executed - prev),
            "execution_suppressed_net_l1": 0.0,
            "execution_suppressed_rotation_l1": 0.0,
        }
        return BudgetedExecutionResult(desired_weights=desired, executed_weights=executed, diagnostics=diagnostics)
    if mode != "budgeted_flow":
        raise ValueError(f"Unsupported W1 execution mode: {mode}")

    q_prev = float(prev[:stock_dim].sum())
    q_desired = float(desired[:stock_dim].sum())
    q_pm = float(np.clip(q_target, q_min, q_max))
    strength_enabled = bool(cfg.get("primitive_strength_enabled", False))
    strength = 1.0
    if strength_enabled:
        if primitive_strength is None or not np.isfinite(float(primitive_strength)):
            strength = float(cfg.get("primitive_strength_default", 0.50))
        else:
            strength = float(np.clip(primitive_strength, 0.0, 1.0))
    opportunity_enabled = bool(cfg.get("opportunity_score_enabled", False))
    opportunity = 1.0
    if opportunity_enabled:
        if opportunity_score is None or not np.isfinite(float(opportunity_score)):
            opportunity = float(cfg.get("opportunity_score_default", 0.50))
        else:
            opportunity = float(np.clip(opportunity_score, 0.0, 1.0))
    net_strength_multiplier = 1.0
    rotation_strength_multiplier = 1.0
    if strength_enabled:
        net_strength_multiplier = float(cfg.get("net_strength_floor", 0.75)) + float(cfg.get("net_strength_scale", 1.50)) * strength
        rotation_strength_multiplier = float(cfg.get("rotation_strength_floor", 0.75)) + float(cfg.get("rotation_strength_scale", 1.00)) * strength
    pm_gap = q_pm - q_prev
    q_request = q_desired - q_prev
    extra_net_budget = 0.0
    guaranteed_schedule_budget = 0.0
    budget_policy = str(cfg.get("net_budget_policy", "scheduled_gap")).lower()
    if budget_policy in {"guaranteed_schedule_plus_strength_extra", "schedule_plus_strength_extra"}:
        net_budget, scheduled_gap, same_direction, guaranteed_schedule_budget, extra_net_budget = _net_budget_with_guaranteed_schedule(
            pm_gap=pm_gap,
            q_request=q_request,
            remaining_days=remaining_days,
            config=cfg,
            primitive_strength=strength * (opportunity if opportunity_enabled else 1.0),
        )
    else:
        net_budget, scheduled_gap, same_direction = _net_budget(
            pm_gap=pm_gap,
            q_request=q_request,
            remaining_days=remaining_days,
            config=cfg,
        )
        net_budget *= max(net_strength_multiplier, 0.0)
        guaranteed_schedule_budget = net_budget
    q_step = float(np.clip(q_request, -net_budget, net_budget))
    q_exec = float(np.clip(q_prev + q_step, q_min, q_max))
    q_step = q_exec - q_prev

    desired_u = _risky_allocation(desired, stock_dim)
    prev_u = _risky_allocation(prev, stock_dim, fallback=desired_u)
    if q_step >= 0.0:
        base_stock = prev[:stock_dim] + q_step * desired_u
    else:
        if q_prev > EPS:
            base_stock = prev[:stock_dim] * (q_exec / max(q_prev, EPS))
        else:
            base_stock = q_exec * desired_u
    base_stock = np.maximum(base_stock, 0.0)
    base_total = float(base_stock.sum())
    if q_exec > EPS and base_total > EPS:
        base_stock = base_stock * (q_exec / base_total)
    elif q_exec > EPS:
        base_stock = q_exec * desired_u
    else:
        base_stock = np.zeros(stock_dim, dtype=np.float64)

    desired_stock_at_q = q_exec * desired_u
    rotation_request = desired_stock_at_q - base_stock
    rotation_request_l1 = _l1(rotation_request)
    rotation_budget = float(cfg.get("rotation_budget_l1", 1.0)) * max(rotation_strength_multiplier, 0.0)
    if bool(cfg.get("final_day_full_rotation", False)) and int(remaining_days) <= 1:
        rotation_budget = max(rotation_budget, rotation_request_l1)
    if rotation_budget < 0:
        rotation_budget = rotation_request_l1
    rotation_scale = 1.0 if rotation_request_l1 <= EPS else min(1.0, rotation_budget / rotation_request_l1)
    stock_exec = base_stock + rotation_scale * rotation_request
    stock_exec = np.maximum(stock_exec, 0.0)
    stock_total = float(stock_exec.sum())
    if q_exec > EPS and stock_total > EPS:
        stock_exec = stock_exec * (q_exec / stock_total)
    elif q_exec > EPS:
        stock_exec = q_exec * desired_u
    else:
        stock_exec = np.zeros(stock_dim, dtype=np.float64)

    executed = np.zeros_like(prev, dtype=np.float64)
    executed[:stock_dim] = stock_exec
    executed[cash_index] = max(0.0, 1.0 - float(stock_exec.sum()))
    executed = normalize_simplex(executed)
    executed_q = float(executed[:stock_dim].sum())
    net_executed_l1 = abs(executed_q - q_prev)
    net_requested_l1 = abs(q_request)
    diagnostics = {
        "execution_mode_budgeted_flow": 1.0,
        "execution_desired_q": q_desired,
        "execution_executed_q": executed_q,
        "execution_desired_cash": float(desired[cash_index]),
        "execution_executed_cash": float(executed[cash_index]),
        "execution_pm_gap": float(pm_gap),
        "execution_q_request": float(q_request),
        "execution_scheduled_net_step": float(scheduled_gap),
        "execution_net_budget_policy_guaranteed_schedule_extra": float(budget_policy in {"guaranteed_schedule_plus_strength_extra", "schedule_plus_strength_extra"}),
        "execution_guaranteed_schedule_budget_today": float(guaranteed_schedule_budget),
        "execution_extra_net_budget_today": float(extra_net_budget),
        "execution_net_budget_today": float(net_budget),
        "execution_rotation_budget_today": float(rotation_budget),
        "execution_primitive_strength": float(strength),
        "execution_opportunity_score": float(opportunity),
        "execution_primitive_opportunity_strength": float(strength * opportunity),
        "execution_net_strength_multiplier": float(net_strength_multiplier),
        "execution_rotation_strength_multiplier": float(rotation_strength_multiplier),
        "execution_net_execution_scale": float(1.0 if net_requested_l1 <= EPS else min(1.0, net_executed_l1 / net_requested_l1)),
        "execution_rotation_execution_scale": float(rotation_scale),
        "execution_same_direction_as_pm": float(same_direction),
        "execution_desired_to_prev_l1": _l1(desired - prev),
        "execution_executed_to_prev_l1": _l1(executed - prev),
        "execution_rotation_request_l1": float(rotation_request_l1),
        "execution_suppressed_net_l1": float(max(0.0, net_requested_l1 - net_executed_l1)),
        "execution_suppressed_rotation_l1": float(max(0.0, rotation_request_l1 - _l1(rotation_scale * rotation_request))),
    }
    return BudgetedExecutionResult(desired_weights=desired, executed_weights=executed, diagnostics=diagnostics)


def apply_budgeted_flow_execution_batch(
    *,
    previous_weights: np.ndarray,
    desired_weights: np.ndarray,
    q_target: np.ndarray,
    remaining_days: np.ndarray,
    stock_dim: int,
    cash_index: int,
    config: dict[str, Any] | None = None,
    primitive_strength: np.ndarray | None = None,
    opportunity_score: np.ndarray | None = None,
    q_min: float = 0.0,
    q_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    prev = np.asarray(previous_weights, dtype=np.float64)
    desired = np.asarray(desired_weights, dtype=np.float64)
    n = int(prev.shape[0])
    executed_rows: list[np.ndarray] = []
    desired_rows: list[np.ndarray] = []
    diag_rows: list[dict[str, float]] = []
    strength_arr = None if primitive_strength is None else np.asarray(primitive_strength, dtype=np.float64).reshape(-1)
    opportunity_arr = None if opportunity_score is None else np.asarray(opportunity_score, dtype=np.float64).reshape(-1)
    for idx in range(n):
        result = apply_budgeted_flow_execution(
            previous_weights=prev[idx],
            desired_weights=desired[idx],
            q_target=float(np.asarray(q_target).reshape(-1)[idx]),
            remaining_days=int(np.asarray(remaining_days).reshape(-1)[idx]),
            stock_dim=stock_dim,
            cash_index=cash_index,
            config=config,
            primitive_strength=None if strength_arr is None else float(strength_arr[idx]),
            opportunity_score=None if opportunity_arr is None else float(opportunity_arr[idx]),
            q_min=q_min,
            q_max=q_max,
        )
        desired_rows.append(result.desired_weights)
        executed_rows.append(result.executed_weights)
        diag_rows.append(result.diagnostics)
    keys = sorted({key for row in diag_rows for key in row})
    diagnostics = {key: np.asarray([row.get(key, np.nan) for row in diag_rows], dtype=np.float64) for key in keys}
    return np.vstack(desired_rows), np.vstack(executed_rows), diagnostics
