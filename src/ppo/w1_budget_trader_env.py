"""Clean W1 budget-PM / stock-selector Trader simulator.

This module is deliberately separate from the older T/H environments.  W1 is
the first clean analogue of the "portfolio manager assigns a task, trader acts
inside the window" design:

* PM action: target risky exposure `q_target` and review horizon.
* Trader action: daily full-portfolio simplex `[stocks..., cash]`.
* No controller, no Top-K, no thresholds/triggers/slices, no group action layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.dow30_sectors import get_sector_map
from src.ppo.execution.budgeted_flow import apply_budgeted_flow_execution
from src.ppo.execution.helpers import EPS, normalize_simplex, normalize_stock_simplex, sigmoid_scalar
from src.ppo.weight_panel import WeightPanel
from src.ppo.w1_risk_stop import compute_risk_stop, features_from_panel_row


def _date_key(value: object) -> str:
    return str(pd.Timestamp(value).date())


def _feature_indices(feature_columns: list[str], names: Sequence[str]) -> list[int]:
    index = {name: idx for idx, name in enumerate(feature_columns)}
    missing = sorted(set(names).difference(index))
    if missing:
        raise ValueError(f"Features missing from panel: {missing}")
    return [index[name] for name in names]


def entropy_simplex(weights: np.ndarray) -> float:
    w = np.maximum(np.asarray(weights, dtype=np.float64), EPS)
    w = w / np.sum(w)
    return float(-np.sum(w * np.log(w)))


@dataclass
class W1PortfolioState:
    day: int
    previous_weights: np.ndarray
    portfolio_value: float
    peak_value: float
    previous_drawdown: float


class W1BudgetTraderEnv:
    """Daily market simulator with dynamic PM windows."""

    def __init__(
        self,
        panel: WeightPanel,
        *,
        pm_feature_names: list[str],
        pm_raw_window_feature_names: list[str],
        stock_feature_names: list[str],
        horizon_choices: list[int],
        transaction_cost_pct: float = 0.001,
        initial_amount: float = 1_000_000.0,
        reward_scale: float = 100.0,
        q_min: float = 0.0,
        q_max: float = 1.0,
        max_horizon_days: int | None = None,
        sector_map_name: str = "dow30_static",
        pm_reward_config: dict[str, Any] | None = None,
        trader_reward_config: dict[str, Any] | None = None,
        execution_config: dict[str, Any] | None = None,
        risk_stop_config: dict[str, Any] | None = None,
        trader_feedback_alpha: float = 0.30,
        initial_weights_source: str = "cash",
    ):
        self.panel = panel
        self.stock_dim = len(panel.tickers)
        self.asset_dim = self.stock_dim + 1
        self.cash_index = self.stock_dim
        self.pm_indices = _feature_indices(panel.feature_columns, pm_feature_names)
        self.raw_window_indices = _feature_indices(panel.feature_columns, pm_raw_window_feature_names)
        self.stock_indices = _feature_indices(panel.feature_columns, stock_feature_names)
        self.pm_raw_window_days = int(max_horizon_days or 20)
        self.horizon_choices = [int(x) for x in horizon_choices]
        if not self.horizon_choices:
            raise ValueError("W1BudgetTraderEnv requires horizon_choices.")
        self.max_horizon_days = int(max(max(self.horizon_choices), max_horizon_days or 1))
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.initial_amount = float(initial_amount)
        self.reward_scale = float(reward_scale)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.pm_reward_config = dict(pm_reward_config or {})
        self.trader_reward_config = dict(trader_reward_config or {})
        self.execution_config = dict(execution_config or {})
        self.risk_stop_config = dict(risk_stop_config or {})
        self.trader_feedback_alpha = float(np.clip(trader_feedback_alpha, 0.0, 1.0))
        self.initial_weights_source = str(initial_weights_source).lower()

        sector_map = get_sector_map(sector_map_name)
        sectors = [sector_map.get(ticker, "other") for ticker in self.panel.tickers]
        sector_to_id = {sector: idx for idx, sector in enumerate(sorted(set(sectors)))}
        self.group_ids = [sector_to_id[sector] for sector in sectors]
        self.n_groups = max(self.group_ids) + 1 if self.group_ids else 1

        self.stock_feature_dim = len(self.stock_indices)
        self.trader_task_dim = 3
        self.pm_obs_dim = (
            len(self.pm_indices)
            + 2  # current_cash, current_q
            + self.max_horizon_days * len(self.raw_window_indices)
            + 1  # trader_reward EWMA change
        )
        self.trader_obs_dim = self.stock_dim * (self.stock_feature_dim + self.trader_task_dim)
        self.reset()

    @property
    def day(self) -> int:
        return int(self.state.day)

    @property
    def previous_weights(self) -> np.ndarray:
        return self.state.previous_weights

    def reset(self) -> None:
        initial = np.zeros(self.asset_dim, dtype=np.float64)
        if self.initial_weights_source == "equal_weight":
            initial[: self.stock_dim] = 1.0 / self.stock_dim
        else:
            initial[self.cash_index] = 1.0
        self.state = W1PortfolioState(
            day=0,
            previous_weights=normalize_simplex(initial),
            portfolio_value=self.initial_amount,
            peak_value=self.initial_amount,
            previous_drawdown=0.0,
        )
        self.trader_reward_ewma = 0.0
        self.trader_reward_ewma_prev = 0.0
        self.trader_reward_change = 0.0

    def done(self) -> bool:
        return self.day >= len(self.panel.dates) - 1

    def _feature_value(self, name: str, default: float = 0.0) -> float:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return float(default)
        return float(self.panel.features[self.day, 0, idx])

    def _safe_feature_index(self, name: str) -> int | None:
        try:
            return self.panel.feature_columns.index(name)
        except ValueError:
            return None

    def current_q(self) -> float:
        return float(np.sum(self.previous_weights[: self.stock_dim]))

    def pm_obs(self) -> np.ndarray:
        pm_features = self.panel.features[self.day, 0, self.pm_indices].astype(np.float32)
        cash_q = np.array([self.previous_weights[self.cash_index], self.current_q()], dtype=np.float32)
        return np.concatenate([pm_features, cash_q, self._pm_raw_window_obs(), np.array([self.trader_reward_change], dtype=np.float32)]).astype(np.float32)

    def _pm_raw_window_obs(self) -> np.ndarray:
        end = self.day + 1
        start = max(0, end - self.max_horizon_days)
        window = np.take(self.panel.features[start:end, 0, :], self.raw_window_indices, axis=1).astype(np.float32)
        if window.shape[0] == 0:
            window = np.zeros((1, len(self.raw_window_indices)), dtype=np.float32)
        pad = self.max_horizon_days - window.shape[0]
        if pad > 0:
            window = np.concatenate([np.repeat(window[:1], pad, axis=0), window], axis=0)
        return window[-self.max_horizon_days :].reshape(-1).astype(np.float32)

    def trader_obs(self, *, q_target: float, remaining_days: int) -> np.ndarray:
        stock_features = np.take(self.panel.features[self.day], self.stock_indices, axis=1).astype(np.float32)
        prev_stock_weights = self.previous_weights[: self.stock_dim].astype(np.float32)
        q_prev = self.current_q()
        signed_budget = float(q_target) - q_prev
        remaining_norm = float(max(1, int(remaining_days))) / float(max(self.max_horizon_days, 1))
        fields = [prev_stock_weights]
        fields.extend(
            [
                np.full(self.stock_dim, signed_budget, dtype=np.float32),
                np.full(self.stock_dim, remaining_norm, dtype=np.float32),
            ]
        )
        task = np.stack(fields, axis=1)
        return np.concatenate([stock_features, task], axis=1).reshape(-1).astype(np.float32)

    def horizon_from_action(self, horizon_idx: float | int) -> int:
        idx = int(np.clip(int(round(float(horizon_idx))), 0, len(self.horizon_choices) - 1))
        return int(self.horizon_choices[idx])

    def apply_risk_stop(self, q_raw: float) -> tuple[float, dict[str, float]]:
        features = features_from_panel_row(self.panel.feature_columns, self.panel.features[self.day, 0, :])
        q_safe, info = compute_risk_stop(
            q_raw,
            features=features,
            current_drawdown=self.state.previous_drawdown,
            trader_feedback_change=self.trader_reward_change,
            config=self.risk_stop_config,
            q_min=self.q_min,
            q_max=self.q_max,
        )
        scalar_info = {key: float(np.asarray(value).reshape(-1)[0]) for key, value in info.items()}
        return float(q_safe), scalar_info

    def execution_opportunity_score(self) -> float:
        cfg = self.execution_config
        if not bool(cfg.get("opportunity_score_enabled", False)):
            return 1.0
        trend = self._feature_value(str(cfg.get("opportunity_trend_feature", "SP500_Trend")), 0.0)
        residual = self._feature_value(str(cfg.get("opportunity_residual_feature", "universe_return_20d")), 0.0)
        regime = self._feature_value(str(cfg.get("opportunity_regime_feature", "Regime_1_Prob")), 0.0)
        vix_change = self._feature_value(str(cfg.get("opportunity_vix_change_feature", "VIX_change_5d")), 0.0)
        turbulence = self._feature_value(str(cfg.get("opportunity_turbulence_feature", "turbulence")), 0.0)
        raw = (
            float(cfg.get("opportunity_bias", 0.0))
            + float(cfg.get("opportunity_trend_weight", 0.40)) * trend
            + float(cfg.get("opportunity_residual_weight", 0.35)) * residual
            + float(cfg.get("opportunity_regime_weight", 0.30)) * (2.0 * regime - 1.0)
            - float(cfg.get("opportunity_vix_change_weight", 0.20)) * vix_change
            - float(cfg.get("opportunity_turbulence_weight", 0.15)) * turbulence
        )
        power = float(cfg.get("opportunity_score_power", 1.0))
        return float(np.clip(sigmoid_scalar(raw) ** max(power, EPS), 0.0, 1.0))

    def _weights_from_trader_action(self, action: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.asset_dim:
            raise ValueError(f"Trader action must have {self.asset_dim} dims, got {action.shape[0]}")
        weights = normalize_simplex(np.maximum(action, 0.0))
        q_exec = float(np.sum(weights[: self.stock_dim]))
        if q_exec > EPS:
            u = normalize_stock_simplex(weights[: self.stock_dim] / q_exec)
        else:
            u = np.full(self.stock_dim, 1.0 / self.stock_dim, dtype=np.float64)
        return q_exec, u, weights

    def step_trader(
        self,
        *,
        q_target: float,
        remaining_days: int,
        trader_action: np.ndarray,
        q_raw_target: float | None = None,
        risk_stop_info: dict[str, float] | None = None,
        execution_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.done():
            raise RuntimeError("Cannot step W1BudgetTraderEnv after done().")
        step_day = self.day
        prev_weights = self.previous_weights.copy()
        _desired_q, _desired_u, desired_weights = self._weights_from_trader_action(trader_action)
        clipped_q = float(np.clip(q_target, self.q_min, self.q_max))
        execution = apply_budgeted_flow_execution(
            previous_weights=prev_weights,
            desired_weights=desired_weights,
            q_target=clipped_q,
            remaining_days=remaining_days,
            stock_dim=self.stock_dim,
            cash_index=self.cash_index,
            config=self.execution_config,
            primitive_strength=(
                None
                if not execution_context or "latent_action_primitive_strength" not in execution_context
                else float(np.asarray(execution_context["latent_action_primitive_strength"]).reshape(-1)[0])
            ),
            opportunity_score=(
                self.execution_opportunity_score()
                if bool(self.execution_config.get("opportunity_score_enabled", False))
                else None
            ),
            q_min=self.q_min,
            q_max=self.q_max,
        )
        target_weights = execution.executed_weights
        q_exec = float(np.sum(target_weights[: self.stock_dim]))
        if q_exec > EPS:
            u = normalize_stock_simplex(target_weights[: self.stock_dim] / q_exec)
        else:
            u = np.full(self.stock_dim, 1.0 / self.stock_dim, dtype=np.float64)
        trade_delta = target_weights - prev_weights
        stock_turnover_l1 = float(np.sum(np.abs(trade_delta[: self.stock_dim])))
        turnover_l1 = float(np.sum(np.abs(trade_delta)))
        transaction_cost = self.transaction_cost_pct * stock_turnover_l1

        asset_returns = self.panel.returns_next[step_day]
        benchmark_return = float(np.mean(asset_returns))
        gross_return = float(np.dot(target_weights[: self.stock_dim], asset_returns))
        net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0

        old_value = self.state.portfolio_value
        new_value = old_value * (1.0 + net_return)
        peak_value = max(self.state.peak_value, new_value)
        drawdown = new_value / max(peak_value, EPS) - 1.0
        drawdown_increment = max(0.0, self.state.previous_drawdown - drawdown)
        post_market = self._weights_after_market(target_weights, asset_returns)

        info: dict[str, Any] = {
            "date": _date_key(self.panel.dates[step_day]),
            "next_date": _date_key(self.panel.dates[step_day + 1]),
            "step_day": step_day,
            "q_target": clipped_q,
            "q_raw_target": float(clipped_q if q_raw_target is None else q_raw_target),
            "q_exec": q_exec,
            "cash_target_pm": 1.0 - clipped_q,
            "cash_exec": float(target_weights[self.cash_index]),
            "desired_q": float(execution.diagnostics.get("execution_desired_q", np.nan)),
            "desired_cash": float(execution.diagnostics.get("execution_desired_cash", np.nan)),
            "remaining_days": int(max(1, remaining_days)),
            "remaining_days_normalized": float(max(1, remaining_days)) / float(max(self.max_horizon_days, 1)),
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "asset_returns": asset_returns.astype(np.float32),
            "transaction_cost": transaction_cost,
            "stock_turnover_l1": stock_turnover_l1,
            "turnover_l1": turnover_l1,
            "drawdown": drawdown,
            "drawdown_increment": drawdown_increment,
            "target_weights": target_weights,
            "desired_weights": execution.desired_weights,
            "post_market_weights": post_market,
            "pre_trade_weights": prev_weights,
            "trade_delta_weights": trade_delta,
            "portfolio_value": new_value,
            "risky_entropy": entropy_simplex(u),
            "portfolio_entropy": entropy_simplex(target_weights),
            "tracking_l1": abs(q_exec - clipped_q),
        }
        info.update({key: float(value) for key, value in execution.diagnostics.items()})
        if risk_stop_info:
            info.update({key: float(value) for key, value in risk_stop_info.items()})
        pm_reward = self.pm_reward(info)
        trader_reward = self.trader_reward(info, u)
        info["pm_reward"] = pm_reward
        info["trader_reward"] = trader_reward

        self.state = W1PortfolioState(
            day=self.day + 1,
            previous_weights=post_market,
            portfolio_value=new_value,
            peak_value=peak_value,
            previous_drawdown=drawdown,
        )
        self._update_trader_feedback(trader_reward)
        return info

    def _weights_after_market(self, weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        next_values = np.zeros_like(weights)
        next_values[: self.stock_dim] = weights[: self.stock_dim] * (1.0 + asset_returns)
        next_values[self.cash_index] = weights[self.cash_index]
        return normalize_simplex(np.maximum(next_values, 0.0))

    def _update_trader_feedback(self, trader_reward: float) -> None:
        self.trader_reward_ewma_prev = self.trader_reward_ewma
        scaled = float(trader_reward) / max(self.reward_scale, EPS)
        self.trader_reward_ewma = (1.0 - self.trader_feedback_alpha) * self.trader_reward_ewma + self.trader_feedback_alpha * scaled
        self.trader_reward_change = self.trader_reward_ewma - self.trader_reward_ewma_prev

    def _future_stock_return_from(self, start_day: int, horizon: int = 5) -> np.ndarray:
        start = int(start_day)
        end = min(len(self.panel.returns_next), start + max(1, int(horizon)))
        if end <= start:
            return np.zeros(self.stock_dim, dtype=np.float64)
        return np.prod(1.0 + self.panel.returns_next[start:end], axis=0) - 1.0

    def _group_returns_from(self, stock_returns: np.ndarray) -> np.ndarray:
        out = np.zeros(self.stock_dim, dtype=np.float64)
        for gid in range(self.n_groups):
            idx = np.asarray([i for i, g in enumerate(self.group_ids) if g == gid], dtype=int)
            if idx.size:
                out[idx] = float(np.mean(stock_returns[idx]))
        return out

    def pm_reward(self, info: dict[str, Any]) -> float:
        cfg = self.pm_reward_config
        net_return = float(info.get("net_return", 0.0))
        benchmark_return = float(info.get("benchmark_return", 0.0))
        drawdown_inc = float(info.get("drawdown_increment", 0.0))
        # PM should be rewarded for the cash/risk budget it asked for, not for
        # the trader's daily fill. This keeps low-level execution quality from
        # leaking into the PM opportunity term.
        cash = float(info.get("cash_target_pm", info.get("cash_exec", 0.0)))
        trend = float(self._feature_value(str(cfg.get("opportunity_feature", "SP500_Trend")), 0.0))
        opportunity_gate = 1.0 / (1.0 + np.exp(-trend))
        cash_opp = opportunity_gate * cash * max(benchmark_return, 0.0)
        prior_threshold = float(cfg.get("cash_opportunity_prior_threshold", 0.55))
        cash_opp_prior = cash * max(0.0, opportunity_gate - prior_threshold)
        active_return = net_return - benchmark_return
        reward = (
            float(cfg.get("return_weight", 1.0)) * net_return
            - float(cfg.get("cash_opportunity_penalty", 0.30)) * cash_opp
            - float(cfg.get("cash_opportunity_prior_penalty", 0.0)) * cash_opp_prior
            - float(cfg.get("drawdown_penalty", 2.0)) * drawdown_inc
            + float(cfg.get("active_return_weight", 0.0)) * opportunity_gate * active_return
        )
        info["pm_opportunity_gate"] = float(opportunity_gate)
        info["pm_cash_opportunity_cost"] = float(cash_opp)
        info["pm_cash_opportunity_prior_cost"] = float(cash_opp_prior)
        info["pm_active_return"] = float(active_return)
        return float(self.reward_scale * reward)

    def pm_window_reward_from_infos(self, infos: Sequence[dict[str, Any]], *, q_target: float) -> float:
        """Reward one PM decision from the realized outcome of its window.

        The PM acts at variable horizons. Summing daily PM rewards would give
        longer horizons a larger reward scale. Instead, close each high-level
        transition with one window-level result: compounded return, compounded
        benchmark, cumulative drawdown increment, and the PM-requested cash
        budget.
        """

        if not infos:
            return 0.0
        cfg = self.pm_reward_config
        net_returns = np.asarray([float(info.get("net_return", 0.0)) for info in infos], dtype=np.float64)
        bench_returns = np.asarray([float(info.get("benchmark_return", 0.0)) for info in infos], dtype=np.float64)
        window_return = float(np.prod(1.0 + net_returns) - 1.0)
        window_benchmark = float(np.prod(1.0 + bench_returns) - 1.0)
        drawdown_inc = float(np.sum([float(info.get("drawdown_increment", 0.0)) for info in infos]))
        cash = float(1.0 - np.clip(q_target, self.q_min, self.q_max))
        if any("pm_opportunity_gate" in info for info in infos):
            opportunity_gate = float(np.nanmean([float(info.get("pm_opportunity_gate", np.nan)) for info in infos]))
            if not np.isfinite(opportunity_gate):
                opportunity_gate = 0.0
        else:
            trend = float(self._feature_value(str(cfg.get("opportunity_feature", "SP500_Trend")), 0.0))
            opportunity_gate = float(1.0 / (1.0 + np.exp(-trend)))
        cash_opp = opportunity_gate * cash * max(window_benchmark, 0.0)
        prior_threshold = float(cfg.get("cash_opportunity_prior_threshold", 0.55))
        cash_opp_prior = cash * max(0.0, opportunity_gate - prior_threshold)
        active_return = window_return - window_benchmark
        reward = (
            float(cfg.get("return_weight", 1.0)) * window_return
            - float(cfg.get("cash_opportunity_penalty", 0.30)) * cash_opp
            - float(cfg.get("cash_opportunity_prior_penalty", 0.0)) * cash_opp_prior
            - float(cfg.get("drawdown_penalty", 2.0)) * drawdown_inc
            + float(cfg.get("active_return_weight", 0.0)) * opportunity_gate * active_return
        )
        return float(self.reward_scale * reward)

    def trader_reward(self, info: dict[str, Any], u: np.ndarray) -> float:
        cfg = self.trader_reward_config
        delta = np.asarray(info["trade_delta_weights"][: self.stock_dim], dtype=np.float64)
        step_day = int(info.get("step_day", self.day))
        horizon = int(cfg.get("flow_horizon", 5))
        future = self._future_stock_return_from(step_day, horizon)
        residual_future = future - float(np.mean(future))
        flow_select = float(np.dot(delta, residual_future))
        group_relative_weight = float(cfg.get("group_relative_weight", 0.0))
        if group_relative_weight != 0.0:
            group_returns = self._group_returns_from(future)
            group_relative = float(np.dot(u, future - group_returns))
        else:
            group_relative = 0.0
        vol_idx = self._safe_feature_index(str(cfg.get("vol_feature", "realized_vol_20d")))
        if vol_idx is None:
            vol = np.zeros(self.stock_dim, dtype=np.float64)
        else:
            vol = self.panel.features[step_day, :, vol_idx].astype(np.float64)
        vol_adjusted_position_change = float(np.sum(np.abs(delta) * np.maximum(vol, 0.0)))
        entropy_scope = str(cfg.get("entropy_scope", "portfolio")).lower()
        if entropy_scope == "risky":
            entropy_bonus = float(info.get("risky_entropy", entropy_simplex(u)))
        elif entropy_scope == "portfolio":
            entropy_bonus = float(info.get("portfolio_entropy", entropy_simplex(info["target_weights"])))
        else:
            raise ValueError(f"Unsupported W1 trader entropy_scope: {entropy_scope}")
        tracking_l1 = float(info.get("tracking_l1", 0.0))
        tracking_multiplier_mode = str(cfg.get("tracking_multiplier", "constant")).lower()
        if tracking_multiplier_mode == "remaining_days":
            tracking_multiplier = float(max(1, int(info.get("remaining_days", 1))))
        elif tracking_multiplier_mode == "remaining_days_normalized":
            tracking_multiplier = float(info.get("remaining_days_normalized", 1.0))
        elif tracking_multiplier_mode == "deadline_inverse":
            tracking_multiplier = float(self.max_horizon_days) / float(max(1, int(info.get("remaining_days", 1))))
        elif tracking_multiplier_mode == "deadline_linear":
            remaining_norm = float(info.get("remaining_days_normalized", 1.0))
            slope = float(cfg.get("deadline_tracking_slope", 2.0))
            tracking_multiplier = 1.0 + slope * max(0.0, 1.0 - remaining_norm)
        else:
            tracking_multiplier = float(cfg.get("tracking_multiplier", 1.0))
        tracking_penalty_term = float(cfg.get("tracking_penalty", 0.05)) * tracking_multiplier * tracking_l1
        reward = (
            float(cfg.get("flow_select_weight", 0.30)) * flow_select
            + group_relative_weight * group_relative
            + float(cfg.get("entropy_bonus", 0.001)) * entropy_bonus
            - float(cfg.get("vol_adjusted_cost", 0.02)) * vol_adjusted_position_change
            - float(cfg.get("transaction_cost_weight", 1.0)) * float(info.get("transaction_cost", 0.0))
            - tracking_penalty_term
        )
        info["trader_flow_select_5d"] = float(flow_select)
        info["trader_group_relative_5d"] = float(group_relative)
        info["trader_entropy_bonus"] = float(entropy_bonus)
        info["trader_entropy_reward_term"] = float(cfg.get("entropy_bonus", 0.001)) * float(entropy_bonus)
        info["trader_vol_adjusted_position_change"] = float(vol_adjusted_position_change)
        info["trader_tracking_multiplier"] = float(tracking_multiplier)
        info["trader_tracking_penalty_term"] = float(tracking_penalty_term)
        return float(self.reward_scale * reward)
