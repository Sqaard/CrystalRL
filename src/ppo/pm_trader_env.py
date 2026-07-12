"""Hierarchical PM/trader environment for H1.

The portfolio-manager actor samples a full root-split target `(q, u)`.
The trader actor receives the scheduled PM target and chooses synthetic
price/quantity execution actions, T5-style.  Trader diagnostics are fed back
into the PM observation through an EWMA low-level diagnostic vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.dow30_sectors import get_sector_map
from src.ppo.execution.helpers import EPS, normalize_simplex, normalize_stock_simplex
from src.ppo.weight_panel import WeightPanel
from src.ppo.two_agent_env import TeacherTraceProvider, _entropy, stock_order_book_proxy_dim


def _feature_indices(feature_columns: list[str], names: list[str]) -> list[int]:
    index = {name: idx for idx, name in enumerate(feature_columns)}
    missing = sorted(set(names).difference(index))
    if missing:
        raise ValueError(f"Features missing from panel: {missing}")
    return [index[name] for name in names]


def _date_key(value: object) -> str:
    return str(pd.Timestamp(value).date())


@dataclass
class PMTraderState:
    day: int
    previous_weights: np.ndarray
    portfolio_value: float
    peak_value: float
    previous_drawdown: float
    last_turnover: float


class PMTraderHierarchicalEnv(gym.Env):
    """Daily simulator with macro PM decisions and low-level trader actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: WeightPanel,
        *,
        pm_feature_names: list[str],
        trader_stock_feature_names: list[str],
        pm_raw_window_feature_names: list[str] | None = None,
        pm_raw_window_days: int = 0,
        root_window_days: int = 20,
        q_min: float = 0.0,
        q_max: float = 0.995,
        transaction_cost_pct: float = 0.001,
        initial_amount: float = 1_000_000.0,
        reward_scale: float = 100.0,
        teacher_provider: TeacherTraceProvider | None = None,
        sector_map_name: str = "dow30_static",
        pm_reward_config: dict[str, Any] | None = None,
        trader_reward_config: dict[str, Any] | None = None,
        stock_order_book_proxy: dict[str, Any] | None = None,
        execution_mode: str = "synthetic_lob",
        price_levels: int = 5,
        quantity_levels: int = 5,
        forced_cleanup: bool = True,
        low_level_diag_alpha: float = 0.30,
        initial_weights_source: str = "cash",
    ):
        super().__init__()
        self.panel = panel
        self.stock_dim = len(panel.tickers)
        self.asset_dim = self.stock_dim + 1
        self.cash_index = self.stock_dim
        self.pm_indices = _feature_indices(panel.feature_columns, pm_feature_names)
        self.trader_stock_indices = _feature_indices(panel.feature_columns, trader_stock_feature_names)
        self.raw_window_indices = _feature_indices(panel.feature_columns, pm_raw_window_feature_names or [])
        self.pm_raw_window_days = max(0, int(pm_raw_window_days))
        self.root_window_days = max(1, int(root_window_days))
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.initial_amount = float(initial_amount)
        self.reward_scale = float(reward_scale)
        self.teacher_provider = teacher_provider
        self.pm_reward_config = dict(pm_reward_config or {})
        self.trader_reward_config = dict(trader_reward_config or {})
        self.stock_order_book_proxy = dict(stock_order_book_proxy or {})
        self.execution_mode = str(execution_mode).lower()
        self.price_levels = max(2, int(price_levels))
        self.quantity_levels = max(2, int(quantity_levels))
        self.forced_cleanup = bool(forced_cleanup)
        self.low_level_diag_alpha = float(np.clip(low_level_diag_alpha, 0.0, 1.0))
        self.initial_weights_source = str(initial_weights_source).lower()

        sector_map = get_sector_map(sector_map_name)
        sectors = [sector_map.get(ticker, "other") for ticker in self.panel.tickers]
        sector_to_id = {sector: idx for idx, sector in enumerate(sorted(set(sectors)))}
        self.group_ids = [sector_to_id[sector] for sector in sectors]

        self.proxy_dim = stock_order_book_proxy_dim(self.stock_order_book_proxy)
        if self.proxy_dim <= 0 and self.execution_mode not in {"no_lob", "no_lob_quantity", "quantity_only"}:
            raise ValueError("PMTraderHierarchicalEnv requires stock_order_book_proxy.enabled=true.")
        self.execution_task_dim = 4
        self.global_context_dim = 12

        self.pm_action_space = spaces.Box(low=0.0, high=1.0, shape=(self.asset_dim,), dtype=np.float32)
        self.trader_action_space = spaces.MultiDiscrete(
            np.array([self.price_levels] * self.stock_dim + [self.quantity_levels] * self.stock_dim, dtype=np.int64)
        )

        self.reset()
        self.pm_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.pm_obs().shape, dtype=np.float32)
        self.trader_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.trader_obs(self.previous_weights, remaining_days=1).shape,
            dtype=np.float32,
        )

    @property
    def day(self) -> int:
        return int(self.state.day)

    @property
    def previous_weights(self) -> np.ndarray:
        return self.state.previous_weights

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state = PMTraderState(
            day=0,
            previous_weights=np.eye(1, self.asset_dim, self.cash_index, dtype=np.float64).reshape(-1),
            portfolio_value=self.initial_amount,
            peak_value=self.initial_amount,
            previous_drawdown=0.0,
            last_turnover=0.0,
        )
        if self.teacher_provider is not None and self.initial_weights_source == "teacher":
            initial = self.teacher_provider.weights_for_date(self.panel.dates[0])
            self.state.previous_weights = normalize_simplex(np.maximum(initial, 0.0))
        self.low_level_diag = np.zeros(6, dtype=np.float32)
        return self.pm_obs(), {}

    def reset_hierarchical(self) -> None:
        self.reset()

    def done(self) -> bool:
        return self.day >= len(self.panel.dates) - 1

    def _feature_value(self, name: str, default: float = 0.0) -> float:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return float(default)
        return float(self.panel.features[self.day, 0, idx])

    def _feature_vector(self, name: str, default: float = 0.0) -> np.ndarray:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return np.full(self.stock_dim, float(default), dtype=np.float32)
        return self.panel.features[self.day, :, idx].astype(np.float32)

    def _portfolio_state_obs(self) -> np.ndarray:
        weights = self.previous_weights
        q_prev = float(np.sum(weights[: self.stock_dim]))
        hhi = float(np.sum(weights**2))
        ret_since_start = self.state.portfolio_value / self.initial_amount - 1.0
        return np.array(
            [
                weights[self.cash_index],
                q_prev,
                hhi,
                self.state.previous_drawdown,
                self.state.last_turnover,
                ret_since_start,
            ],
            dtype=np.float32,
        )

    def _pm_raw_window_obs(self) -> np.ndarray:
        if self.pm_raw_window_days <= 0:
            return np.zeros(0, dtype=np.float32)
        end = self.day + 1
        start = max(0, end - self.pm_raw_window_days)
        window = self.panel.features[start:end, 0, self.raw_window_indices].astype(np.float32)
        if window.shape[0] == 0:
            window = np.zeros((1, len(self.raw_window_indices)), dtype=np.float32)
        pad = self.pm_raw_window_days - window.shape[0]
        if pad > 0:
            window = np.concatenate([np.repeat(window[:1], pad, axis=0), window], axis=0)
        return window[-self.pm_raw_window_days :].reshape(-1).astype(np.float32)

    def pm_obs(self) -> np.ndarray:
        pm_features = self.panel.features[self.day, 0, self.pm_indices].astype(np.float32)
        return np.concatenate(
            [
                pm_features,
                self._portfolio_state_obs(),
                self._pm_raw_window_obs(),
                self.low_level_diag.astype(np.float32),
            ]
        ).astype(np.float32)

    def _weights_from_pm_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.asset_dim:
            raise ValueError(f"PM action must have {self.asset_dim} factors, got {action.shape[0]}")
        q = float(np.clip(action[0], self.q_min, self.q_max))
        u = normalize_stock_simplex(np.maximum(action[1:], 0.0))
        weights = np.zeros(self.asset_dim, dtype=np.float64)
        weights[: self.stock_dim] = q * u
        weights[self.cash_index] = 1.0 - q
        return normalize_simplex(weights)

    def teacher_pm_action_for_date(self, date: object) -> np.ndarray:
        if self.teacher_provider is None:
            weights = self.previous_weights
        else:
            weights = self.teacher_provider.weights_for_date(date)
        q = float(np.sum(weights[: self.stock_dim]))
        u = normalize_stock_simplex(weights[: self.stock_dim])
        return np.concatenate([[q], u]).astype(np.float32)

    def trader_obs(self, scheduled_target: np.ndarray, *, remaining_days: int) -> np.ndarray:
        target = normalize_simplex(scheduled_target)
        stock_features = self.panel.features[self.day, :, self.trader_stock_indices].reshape(-1).astype(np.float32)
        proxy = self._synthetic_proxy_obs(target, remaining_days=remaining_days)
        delta = target[: self.stock_dim] - self.previous_weights[: self.stock_dim]
        task = np.stack(
            [
                delta.astype(np.float32),
                np.abs(delta).astype(np.float32),
                np.maximum(delta, 0.0).astype(np.float32),
                np.maximum(-delta, 0.0).astype(np.float32),
            ],
            axis=1,
        ).reshape(-1)
        q_target = float(np.sum(target[: self.stock_dim]))
        q_prev = float(np.sum(self.previous_weights[: self.stock_dim]))
        global_context = np.concatenate(
            [
                self._portfolio_state_obs(),
                np.array(
                    [
                        q_target,
                        q_prev,
                        q_target - q_prev,
                        float(max(1, remaining_days)) / float(self.root_window_days),
                        self._feature_value("regime_entropy", 0.0),
                        self._feature_value("VIX_change_5d", 0.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return np.concatenate([stock_features, proxy, task, global_context]).astype(np.float32)

    def _synthetic_proxy_obs(self, target_weights: np.ndarray, *, remaining_days: int) -> np.ndarray:
        expected_dim = stock_order_book_proxy_dim(self.stock_order_book_proxy)
        if expected_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        q_target = float(np.sum(target_weights[: self.stock_dim]))
        q_prev = float(np.sum(self.previous_weights[: self.stock_dim]))
        cfg = self.stock_order_book_proxy
        high_low_range = self._feature_vector(str(cfg.get("range_feature", "high_low_range")))
        atr_rel = self._feature_vector(str(cfg.get("atr_feature", "atr_rel")))
        realized_vol = self._feature_vector(str(cfg.get("vol_feature", "realized_vol_20d")))
        volume_ratio = self._feature_vector(str(cfg.get("liquidity_feature", "volume_ratio")), default=1.0)
        volume_z = self._feature_vector(str(cfg.get("volume_z_feature", "volume_zscore_20d_raw")))
        dollar_volume_z = self._feature_vector(str(cfg.get("dollar_volume_z_feature", "dollar_volume_zscore_20d_raw")))
        liquidity = np.maximum(volume_ratio, 1e-3)
        signed_budget = q_target - q_prev
        abs_budget = abs(signed_budget)
        urgency = abs_budget / max(float(remaining_days), 1.0)
        prev_stock = self.previous_weights[: self.stock_dim].astype(np.float32)
        buy_budget_vec = np.full(self.stock_dim, max(signed_budget, 0.0), dtype=np.float32)
        sell_budget_vec = np.full(self.stock_dim, max(-signed_budget, 0.0), dtype=np.float32)
        signed_budget_vec = np.full(self.stock_dim, signed_budget, dtype=np.float32)
        abs_budget_vec = np.full(self.stock_dim, abs_budget, dtype=np.float32)
        remaining_vec = np.full(self.stock_dim, float(max(1, remaining_days)) / float(self.root_window_days), dtype=np.float32)
        urgency_vec = np.full(self.stock_dim, urgency, dtype=np.float32)
        queue_risk = (urgency_vec * np.maximum(realized_vol, 0.0) / np.maximum(liquidity, 1e-3)).astype(np.float32)

        mode = str(cfg.get("mode", "synthetic_lob_proxy")).lower()
        if mode != "synthetic_lob_proxy":
            parts = [
                prev_stock,
                buy_budget_vec,
                sell_budget_vec,
                remaining_vec,
                urgency_vec,
                high_low_range,
                atr_rel,
                realized_vol,
                volume_ratio,
                volume_z,
                dollar_volume_z,
                queue_risk,
            ]
            out = np.stack(parts, axis=1).reshape(-1).astype(np.float32)
            if out.shape[0] != self.stock_dim * expected_dim:
                raise RuntimeError(f"{mode} produced dim={out.shape[0] // self.stock_dim}, expected={expected_dim}")
            return out

        close_open_return = self._feature_vector(str(cfg.get("close_open_feature", "close_open_return")))
        open_to_prev_close = self._feature_vector(str(cfg.get("open_prev_close_feature", "open_to_prev_close")))
        microtrend = close_open_return + open_to_prev_close
        spread_floor = float(cfg.get("spread_floor", 1e-4))
        spread_scale = float(cfg.get("spread_scale", 0.25))
        liquidity_spread_scale = float(cfg.get("liquidity_spread_scale", 0.01))
        impact_scale = float(cfg.get("impact_scale", 0.10))
        levels = max(1, int(cfg.get("levels", 3)))
        level_step = float(cfg.get("level_step", 1.0))
        spread_proxy = (
            spread_floor
            + spread_scale * np.maximum(high_low_range, 0.0)
            + 0.5 * spread_scale * np.maximum(atr_rel, 0.0)
            + liquidity_spread_scale / np.maximum(liquidity, 1e-3)
        ).astype(np.float32)
        impact_proxy = (
            impact_scale * np.maximum(abs_budget_vec, urgency_vec) * (1.0 + np.maximum(realized_vol, 0.0)) / np.maximum(liquidity, 1e-3)
        ).astype(np.float32)
        imbalance_proxy = np.tanh(5.0 * microtrend + 0.25 * volume_z).astype(np.float32)
        parts = [
            prev_stock,
            signed_budget_vec,
            abs_budget_vec,
            buy_budget_vec,
            sell_budget_vec,
            remaining_vec,
            urgency_vec,
            spread_proxy,
            impact_proxy,
            imbalance_proxy,
        ]
        for level in range(1, levels + 1):
            level_scale = float(level) * level_step
            depth_base = np.maximum(liquidity, 1e-3) / float(level)
            parts.extend(
                [
                    (-level_scale * spread_proxy).astype(np.float32),
                    (level_scale * spread_proxy).astype(np.float32),
                    (depth_base * (1.0 - 0.5 * np.maximum(imbalance_proxy, 0.0))).astype(np.float32),
                    (depth_base * (1.0 + 0.5 * np.minimum(imbalance_proxy, 0.0))).astype(np.float32),
                ]
            )
        out = np.stack(parts, axis=1).reshape(-1).astype(np.float32)
        if out.shape[0] != self.stock_dim * expected_dim:
            raise RuntimeError(f"synthetic_lob_proxy produced dim={out.shape[0] // self.stock_dim}, expected={expected_dim}")
        return out

    def _execution_proxies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.stock_order_book_proxy
        high_low_range = self._feature_vector(str(cfg.get("range_feature", "high_low_range")))
        atr_rel = self._feature_vector(str(cfg.get("atr_feature", "atr_rel")))
        realized_vol = self._feature_vector(str(cfg.get("vol_feature", "realized_vol_20d")))
        volume_ratio = self._feature_vector(str(cfg.get("liquidity_feature", "volume_ratio")), default=1.0)
        close_open_return = self._feature_vector(str(cfg.get("close_open_feature", "close_open_return")))
        open_to_prev_close = self._feature_vector(str(cfg.get("open_prev_close_feature", "open_to_prev_close")))
        volume_z = self._feature_vector(str(cfg.get("volume_z_feature", "volume_zscore_20d_raw")))
        liquidity = np.exp(np.clip(volume_ratio, -3.0, 3.0)).astype(np.float64)
        spread = (
            float(cfg.get("spread_floor", 1e-4))
            + float(cfg.get("spread_scale", 0.25)) * np.maximum(high_low_range, 0.0)
            + 0.5 * float(cfg.get("spread_scale", 0.25)) * np.maximum(atr_rel, 0.0)
            + float(cfg.get("liquidity_spread_scale", 0.01)) / np.maximum(liquidity, 1e-3)
        )
        imbalance = np.tanh(5.0 * (close_open_return + open_to_prev_close) + 0.25 * volume_z)
        impact = float(cfg.get("impact_scale", 0.10)) * (1.0 + np.maximum(realized_vol, 0.0)) / np.maximum(liquidity, 1e-3)
        return spread.astype(np.float64), impact.astype(np.float64), imbalance.astype(np.float64), liquidity.astype(np.float64)

    def _weights_after_market(self, weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        grown = weights.copy()
        grown[: self.stock_dim] *= 1.0 + asset_returns
        return normalize_simplex(np.maximum(grown, 0.0))

    def _pm_reward(self, info: dict[str, Any]) -> float:
        cfg = self.pm_reward_config
        net_return = float(info.get("net_return", 0.0))
        drawdown_inc = float(info.get("drawdown_increment", 0.0))
        cash = float(info["pm_anchor_weights"][self.cash_index])
        opportunity_gate = 1.0 / (1.0 + np.exp(-float(self._feature_value("SP500_Trend", 0.0))))
        benchmark_return = float(info.get("benchmark_return", 0.0))
        cash_opp = cash * max(benchmark_return, 0.0) * opportunity_gate
        reward = (
            float(cfg.get("return_weight", 1.0)) * net_return
            - float(cfg.get("drawdown_penalty", 2.0)) * drawdown_inc
            - float(cfg.get("cash_opportunity_penalty", 0.30)) * cash_opp
            - float(cfg.get("execution_gap_penalty", 0.05)) * float(info.get("tracking_l1", 0.0))
            - float(cfg.get("slippage_penalty", 0.50)) * float(info.get("slippage_cost", 0.0))
        )
        return float(self.reward_scale * reward)

    def _trader_reward(self, info: dict[str, Any]) -> float:
        cfg = self.trader_reward_config
        total_execution_cost = (
            float(info.get("commission_cost", 0.0))
            + float(info.get("slippage_cost", 0.0))
            + float(cfg.get("tracking_penalty", 0.05)) * float(info.get("tracking_l1", 0.0))
        )
        fill_bonus = float(cfg.get("fill_bonus", 0.0)) * float(info.get("limit_fill_l1", 0.0))
        return float(self.reward_scale * (fill_bonus - min(total_execution_cost, 0.50)))

    def step_hierarchical(self, *, pm_anchor_weights: np.ndarray, pm_anchor_start_day: int, trader_action: np.ndarray) -> dict[str, Any]:
        if self.done():
            return {"terminated": True}
        step_day = self.day
        pm_anchor = normalize_simplex(pm_anchor_weights)
        elapsed = max(0, int(self.day) - int(pm_anchor_start_day))
        remaining_days = max(1, self.root_window_days - elapsed)
        prev = self.previous_weights.copy()
        scheduled = normalize_simplex(prev + (pm_anchor - prev) / float(remaining_days))

        action = np.asarray(trader_action, dtype=np.int64).reshape(-1)
        if action.shape[0] != 2 * self.stock_dim:
            raise ValueError(f"Trader action length must be {2 * self.stock_dim}, got {action.shape[0]}")
        price_idx = np.clip(action[: self.stock_dim], 0, self.price_levels - 1)
        qty_idx = np.clip(action[self.stock_dim :], 0, self.quantity_levels - 1)
        price_aggr = price_idx.astype(np.float64) / max(self.price_levels - 1, 1)
        qty_frac = qty_idx.astype(np.float64) / max(self.quantity_levels - 1, 1)

        delta = scheduled[: self.stock_dim] - prev[: self.stock_dim]
        direction = np.sign(delta)
        abs_need = np.abs(delta)
        if self.execution_mode in {"no_lob", "no_lob_quantity", "quantity_only"}:
            limit_fill = abs_need * qty_frac
            cleanup = np.zeros_like(limit_fill)
            executed_delta = direction * limit_fill
            slippage_cost = 0.0
            fill_prob = np.ones(self.stock_dim, dtype=np.float64)
        else:
            requested = abs_need * qty_frac
            spread, impact, imbalance, liquidity = self._execution_proxies()
            adverse_imbalance = np.where(direction >= 0, np.maximum(imbalance, 0.0), np.maximum(-imbalance, 0.0))
            fill_prob = np.clip(
                0.15
                + 0.65 * price_aggr
                + 0.10 * np.tanh(liquidity)
                - 0.15 * adverse_imbalance
                - 0.10 * np.maximum(spread, 0.0),
                0.0,
                1.0,
            )
            limit_fill = np.minimum(requested * fill_prob, abs_need)
            remaining = np.maximum(abs_need - limit_fill, 0.0)
            cleanup = remaining if self.forced_cleanup else np.zeros_like(remaining)
            executed_abs = limit_fill + cleanup
            executed_delta = direction * executed_abs

            limit_slippage = limit_fill * (spread * (price_aggr - 0.35) + impact * requested)
            cleanup_slippage = cleanup * (1.50 * spread + 2.00 * impact * np.maximum(remaining, 0.0))
            slippage_cost = float(max(0.0, np.sum(limit_slippage + cleanup_slippage)))
            slippage_cost = float(min(slippage_cost, 0.25))
        stock_turnover_l1 = float(np.sum(np.abs(executed_delta)))
        commission_cost = self.transaction_cost_pct * stock_turnover_l1

        executed = prev.copy()
        executed[: self.stock_dim] = np.maximum(executed[: self.stock_dim] + executed_delta, 0.0)
        cash_delta = -float(np.sum(executed_delta))
        executed[self.cash_index] = max(executed[self.cash_index] + cash_delta - commission_cost - max(slippage_cost, 0.0), 0.0)
        executed = normalize_simplex(executed)
        tracking_l1 = float(np.sum(np.abs(scheduled - executed)))
        turnover_l1 = float(np.sum(np.abs(executed - prev)))

        asset_returns = self.panel.returns_next[self.day]
        benchmark_return = float(np.mean(asset_returns))
        gross_return = float(np.dot(executed[: self.stock_dim], asset_returns))
        net_return = (1.0 - commission_cost - max(slippage_cost, 0.0)) * (1.0 + gross_return) - 1.0
        net_return = float(np.clip(net_return, -0.95, 0.95))
        old_value = self.state.portfolio_value
        new_value = old_value * (1.0 + net_return)
        peak_value = max(self.state.peak_value, new_value)
        drawdown = new_value / max(peak_value, EPS) - 1.0
        drawdown_increment = max(0.0, self.state.previous_drawdown - drawdown)
        post_market = self._weights_after_market(executed, asset_returns)

        info = {
            "date": _date_key(self.panel.dates[step_day]),
            "step_day": step_day,
            "pm_anchor_weights": pm_anchor.astype(np.float32),
            "scheduled_target_weights": scheduled.astype(np.float32),
            "executed_weights": executed.astype(np.float32),
            "post_market_weights": post_market.astype(np.float32),
            "pre_trade_weights": prev.astype(np.float32),
            "trade_delta_weights": (executed - prev).astype(np.float32),
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "transaction_cost": commission_cost,
            "commission_cost": commission_cost,
            "slippage_cost": slippage_cost,
            "tracking_l1": tracking_l1,
            "stock_turnover_l1": stock_turnover_l1,
            "turnover_l1": turnover_l1,
            "drawdown": drawdown,
            "drawdown_increment": drawdown_increment,
            "limit_fill_l1": float(np.sum(limit_fill)),
            "cleanup_l1": float(np.sum(cleanup)),
            "fill_prob_mean": float(np.mean(fill_prob)),
            "price_aggr_mean": float(np.mean(price_aggr)),
            "qty_frac_mean": float(np.mean(qty_frac)),
            "root_remaining_days": remaining_days,
            "q_anchor": float(np.sum(pm_anchor[: self.stock_dim])),
            "q_scheduled": float(np.sum(scheduled[: self.stock_dim])),
            "q_executed": float(np.sum(executed[: self.stock_dim])),
            "risky_entropy": _entropy(executed[: self.stock_dim]),
        }
        info["pm_reward"] = self._pm_reward(info)
        info["trader_reward"] = self._trader_reward(info)

        self.state = PMTraderState(
            day=self.day + 1,
            previous_weights=post_market,
            portfolio_value=new_value,
            peak_value=peak_value,
            previous_drawdown=drawdown,
            last_turnover=turnover_l1,
        )
        self._update_low_level_diag(info)
        info["terminated"] = self.done()
        return info

    def _update_low_level_diag(self, info: dict[str, Any]) -> None:
        diag = np.array(
            [
                float(info.get("tracking_l1", 0.0)),
                float(info.get("slippage_cost", 0.0)),
                float(info.get("cleanup_l1", 0.0)),
                float(info.get("fill_prob_mean", 0.0)),
                float(info.get("trader_reward", 0.0)) / max(self.reward_scale, EPS),
                float(info.get("net_return", 0.0)) - float(info.get("benchmark_return", 0.0)),
            ],
            dtype=np.float32,
        )
        alpha = self.low_level_diag_alpha
        self.low_level_diag = (1.0 - alpha) * self.low_level_diag + alpha * diag
