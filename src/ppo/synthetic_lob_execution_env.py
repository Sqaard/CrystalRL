"""Synthetic low-level LOB execution environment for Stage 0.1 T5.

This is not a true intraday limit-order-book simulator.  It is a daily
OHLCV/liquidity calibrated execution sandbox that exposes a Wang-like
price/quantity branch action to a low-level policy.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.dow30_sectors import get_sector_map
from src.ppo.execution.helpers import EPS, normalize_simplex
from src.ppo.weight_panel import WeightPanel
from src.ppo.two_agent_env import TeacherTraceProvider, _entropy, stock_order_book_proxy_dim


def _feature_indices(feature_columns: list[str], names: list[str]) -> list[int]:
    index = {name: idx for idx, name in enumerate(feature_columns)}
    missing = sorted(set(names).difference(index))
    if missing:
        raise ValueError(f"Features missing from panel: {missing}")
    return [index[name] for name in names]


class SyntheticLobExecutionEnv(gym.Env):
    """Train a low-level price/quantity execution policy against teacher targets."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: WeightPanel,
        *,
        teacher_provider: TeacherTraceProvider,
        stock_feature_names: list[str],
        stock_order_book_proxy: dict[str, Any],
        execution_mode: str = "synthetic_lob",
        sector_map_name: str = "dow30_static",
        price_levels: int = 5,
        quantity_levels: int = 5,
        transaction_cost_pct: float = 0.001,
        reward_scale: float = 100.0,
        forced_cleanup: bool = True,
        tracking_penalty: float = 0.05,
        slippage_weight: float = 1.0,
        impact_weight: float = 1.0,
    ):
        super().__init__()
        self.panel = panel
        self.teacher_provider = teacher_provider
        self.stock_feature_names = list(stock_feature_names)
        self.stock_indices = _feature_indices(panel.feature_columns, self.stock_feature_names)
        self.stock_order_book_proxy = dict(stock_order_book_proxy or {})
        self.execution_mode = str(execution_mode).lower()
        self.stock_dim = len(panel.tickers)
        self.asset_dim = self.stock_dim + 1
        self.cash_index = self.stock_dim
        self.price_levels = max(2, int(price_levels))
        self.quantity_levels = max(2, int(quantity_levels))
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.reward_scale = float(reward_scale)
        self.forced_cleanup = bool(forced_cleanup)
        self.tracking_penalty = float(tracking_penalty)
        self.slippage_weight = float(slippage_weight)
        self.impact_weight = float(impact_weight)
        sector_map = get_sector_map(sector_map_name)
        sectors = [sector_map.get(ticker, "other") for ticker in panel.tickers]
        sector_to_id = {sector: idx for idx, sector in enumerate(sorted(set(sectors)))}
        self.group_ids = [sector_to_id[sector] for sector in sectors]

        self.proxy_dim = stock_order_book_proxy_dim(self.stock_order_book_proxy)
        if self.proxy_dim <= 0 and self.execution_mode not in {"no_lob", "no_lob_quantity", "quantity_only"}:
            raise ValueError("SyntheticLobExecutionEnv requires stock_order_book_proxy.enabled=true.")
        self.execution_task_dim = 4
        self.global_context_dim = 12
        obs_dim = (
            self.stock_dim * len(self.stock_indices)
            + self.stock_dim * self.proxy_dim
            + self.stock_dim * self.execution_task_dim
            + self.global_context_dim
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(
            np.array([self.price_levels] * self.stock_dim + [self.quantity_levels] * self.stock_dim, dtype=np.int64)
        )
        self.reset()

    @property
    def day(self) -> int:
        return int(self._day)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._day = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.previous_drawdown = 0.0
        self.last_turnover = 0.0
        first_weights = self.teacher_provider.weights_for_date(self.panel.dates[0])
        self.previous_weights = normalize_simplex(np.maximum(first_weights, 0.0))
        return self._obs(), {}

    def _done(self) -> bool:
        return self.day >= len(self.panel.dates) - 1

    def _feature_vector(self, name: str, default: float = 0.0) -> np.ndarray:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return np.full(self.stock_dim, float(default), dtype=np.float32)
        return self.panel.features[self.day, :, idx].astype(np.float32)

    def _feature_value(self, name: str, default: float = 0.0) -> float:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return float(default)
        return float(self.panel.features[self.day, 0, idx])

    def _target_weights(self) -> np.ndarray:
        return self.teacher_provider.weights_for_date(self.panel.dates[self.day])

    def _remaining_context(self) -> float:
        return 1.0

    def _portfolio_state_obs(self) -> np.ndarray:
        weights = self.previous_weights
        gross_exposure = float(np.sum(weights[: self.stock_dim]))
        hhi = float(np.sum(weights**2))
        ret_since_start = float(np.clip(self.portfolio_value - 1.0, -5.0, 5.0))
        return np.array(
            [
                weights[self.cash_index],
                gross_exposure,
                hhi,
                self.previous_drawdown,
                self.last_turnover,
                ret_since_start,
            ],
            dtype=np.float32,
        )

    def _synthetic_proxy_obs(self, target_weights: np.ndarray) -> np.ndarray:
        cfg = self.stock_order_book_proxy
        expected_dim = stock_order_book_proxy_dim(cfg)
        if expected_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        q_target = float(np.sum(target_weights[: self.stock_dim]))
        q_prev = float(np.sum(self.previous_weights[: self.stock_dim]))
        remaining = float(self._remaining_context())
        high_low_range = self._feature_vector(str(cfg.get("range_feature", "high_low_range")))
        atr_rel = self._feature_vector(str(cfg.get("atr_feature", "atr_rel")))
        realized_vol = self._feature_vector(str(cfg.get("vol_feature", "realized_vol_20d")))
        volume_ratio = self._feature_vector(str(cfg.get("liquidity_feature", "volume_ratio")), default=1.0)
        volume_z = self._feature_vector(str(cfg.get("volume_z_feature", "volume_zscore_20d_raw")))
        dollar_volume_z = self._feature_vector(str(cfg.get("dollar_volume_z_feature", "dollar_volume_zscore_20d_raw")))
        range_proxy = 0.01 * np.clip(np.abs(high_low_range), 0.0, 5.0)
        atr_proxy = 0.01 * np.clip(np.abs(atr_rel), 0.0, 5.0)
        vol_proxy = 0.01 * np.clip(np.abs(realized_vol), 0.0, 5.0)
        liquidity = np.exp(np.clip(volume_ratio, -3.0, 3.0)).astype(np.float32)
        signed_budget = q_target - q_prev
        abs_budget = abs(signed_budget)
        urgency = abs_budget
        prev_stock = self.previous_weights[: self.stock_dim].astype(np.float32)
        buy_budget_vec = np.full(self.stock_dim, max(signed_budget, 0.0), dtype=np.float32)
        sell_budget_vec = np.full(self.stock_dim, max(-signed_budget, 0.0), dtype=np.float32)
        remaining_vec = np.full(self.stock_dim, 1.0, dtype=np.float32)
        urgency_vec = np.full(self.stock_dim, urgency, dtype=np.float32)
        queue_risk = (urgency_vec * np.maximum(realized_vol, 0.0) / liquidity).astype(np.float32)

        mode = str(cfg.get("mode", "synthetic_lob_proxy")).lower()
        if mode != "synthetic_lob_proxy":
            parts = [
                prev_stock,
                buy_budget_vec,
                sell_budget_vec,
                remaining_vec,
                urgency_vec,
                range_proxy.astype(np.float32),
                atr_proxy.astype(np.float32),
                vol_proxy.astype(np.float32),
                liquidity.astype(np.float32),
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
            + spread_scale * range_proxy
            + 0.5 * spread_scale * atr_proxy
            + liquidity_spread_scale / np.maximum(liquidity, 1e-3)
        ).astype(np.float32)
        impact_proxy = (
            impact_scale
            * max(abs_budget, urgency)
            * (1.0 + vol_proxy)
            / np.maximum(liquidity, 1e-3)
        ).astype(np.float32)
        imbalance_proxy = np.tanh(5.0 * microtrend + 0.25 * volume_z).astype(np.float32)
        parts = [
            prev_stock,
            np.full(self.stock_dim, signed_budget, dtype=np.float32),
            np.full(self.stock_dim, abs_budget, dtype=np.float32),
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

    def _obs(self) -> np.ndarray:
        target = self._target_weights()
        stock_features = self.panel.features[self.day, :, self.stock_indices].reshape(-1).astype(np.float32)
        proxy = self._synthetic_proxy_obs(target)
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
                        float(self._remaining_context()),
                        self._feature_value("regime_entropy", 0.0),
                        self._feature_value("VIX_change_5d", 0.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return np.concatenate([stock_features, proxy, task, global_context]).astype(np.float32)

    def _weights_after_market(self, weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        grown = weights.copy()
        grown[: self.stock_dim] *= 1.0 + asset_returns
        grown[self.cash_index] *= 1.0
        return normalize_simplex(np.maximum(grown, 0.0))

    def _execution_proxies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.stock_order_book_proxy
        high_low_range = self._feature_vector(str(cfg.get("range_feature", "high_low_range")))
        atr_rel = self._feature_vector(str(cfg.get("atr_feature", "atr_rel")))
        realized_vol = self._feature_vector(str(cfg.get("vol_feature", "realized_vol_20d")))
        volume_ratio = self._feature_vector(str(cfg.get("liquidity_feature", "volume_ratio")), default=1.0)
        close_open_return = self._feature_vector(str(cfg.get("close_open_feature", "close_open_return")))
        open_to_prev_close = self._feature_vector(str(cfg.get("open_prev_close_feature", "open_to_prev_close")))
        volume_z = self._feature_vector(str(cfg.get("volume_z_feature", "volume_zscore_20d_raw")))
        range_proxy = 0.01 * np.clip(np.abs(high_low_range), 0.0, 5.0)
        atr_proxy = 0.01 * np.clip(np.abs(atr_rel), 0.0, 5.0)
        vol_proxy = 0.01 * np.clip(np.abs(realized_vol), 0.0, 5.0)
        liquidity = np.exp(np.clip(volume_ratio, -3.0, 3.0)).astype(np.float64)
        spread_proxy = (
            float(cfg.get("spread_floor", 1e-4))
            + float(cfg.get("spread_scale", 0.25)) * range_proxy
            + 0.5 * float(cfg.get("spread_scale", 0.25)) * atr_proxy
            + float(cfg.get("liquidity_spread_scale", 0.01)) / liquidity
        )
        imbalance = np.tanh(5.0 * (close_open_return + open_to_prev_close) + 0.25 * volume_z)
        impact = float(cfg.get("impact_scale", 0.10)) * (1.0 + vol_proxy) / liquidity
        return spread_proxy.astype(np.float64), impact.astype(np.float64), imbalance.astype(np.float64), liquidity.astype(np.float64)

    def step(self, action: np.ndarray):
        if self._done():
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != 2 * self.stock_dim:
            raise ValueError(f"action length must be {2 * self.stock_dim}, got {action.shape[0]}")
        price_idx = np.clip(action[: self.stock_dim], 0, self.price_levels - 1)
        qty_idx = np.clip(action[self.stock_dim :], 0, self.quantity_levels - 1)
        price_aggr = price_idx.astype(np.float64) / max(self.price_levels - 1, 1)
        qty_frac = qty_idx.astype(np.float64) / max(self.quantity_levels - 1, 1)

        target = self._target_weights()
        prev = self.previous_weights.copy()
        delta = target[: self.stock_dim] - prev[: self.stock_dim]
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

            limit_slippage = limit_fill * (spread * (price_aggr - 0.35) + self.impact_weight * impact * requested)
            cleanup_slippage = cleanup * (1.50 * spread + 2.00 * self.impact_weight * impact * np.maximum(remaining, 0.0))
            slippage_cost = float(max(0.0, self.slippage_weight * np.sum(limit_slippage + cleanup_slippage)))
            slippage_cost = float(min(slippage_cost, 0.25))
        turnover_l1 = float(np.sum(np.abs(executed_delta)))
        commission_cost = self.transaction_cost_pct * turnover_l1

        next_weights = prev.copy()
        next_weights[: self.stock_dim] = np.maximum(next_weights[: self.stock_dim] + executed_delta, 0.0)
        cash_delta = -float(np.sum(executed_delta))
        next_weights[self.cash_index] = max(next_weights[self.cash_index] + cash_delta - commission_cost - max(slippage_cost, 0.0), 0.0)
        next_weights = normalize_simplex(next_weights)
        tracking_l1 = float(np.sum(np.abs(target - next_weights)))
        tracking_cost = self.tracking_penalty * tracking_l1
        total_execution_cost = float(min(commission_cost + slippage_cost + tracking_cost, 0.50))
        reward = -self.reward_scale * total_execution_cost

        asset_returns = self.panel.returns_next[self.day]
        gross_return = float(np.dot(next_weights[: self.stock_dim], asset_returns))
        net_return = (1.0 - commission_cost - max(slippage_cost, 0.0)) * (1.0 + gross_return) - 1.0
        net_return = float(np.clip(net_return, -0.95, 0.95))
        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = self.portfolio_value / max(self.peak_value, EPS) - 1.0
        self.previous_drawdown = float(drawdown)
        self.last_turnover = turnover_l1
        self.previous_weights = self._weights_after_market(next_weights, asset_returns)
        current_date = str(pd.Timestamp(self.panel.dates[self.day]).date())
        self._day += 1
        terminated = self._done()
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "date": current_date,
            "reward": float(reward),
            "net_return": net_return,
            "turnover_l1": turnover_l1,
            "commission_cost": commission_cost,
            "slippage_cost": slippage_cost,
            "tracking_l1": tracking_l1,
            "limit_fill_l1": float(np.sum(limit_fill)),
            "cleanup_l1": float(np.sum(cleanup)),
            "fill_prob_mean": float(np.mean(fill_prob)),
            "price_aggr_mean": float(np.mean(price_aggr)),
            "qty_frac_mean": float(np.mean(qty_frac)),
            "target_cash": float(target[self.cash_index]),
            "executed_cash": float(next_weights[self.cash_index]),
            "risky_entropy": _entropy(next_weights[: self.stock_dim]),
        }
        return obs, float(reward), terminated, False, info


class SyntheticLobExecutionCurriculumEnv(SyntheticLobExecutionEnv):
    """Wang-style low-level execution curriculum over target quantity/time.

    The project does not have true intraday LOB data, so this curriculum uses
    the same synthetic execution sandbox as T5 while explicitly sweeping over
    target quantity, direction, asset, and remaining-time subtasks.  It is used
    only for trader pretraining; validation should still use real PM targets.
    """

    def __init__(
        self,
        *args: Any,
        target_quantities: list[float] | None = None,
        remaining_time_levels: list[int] | None = None,
        initial_risky_fraction: float = 0.60,
        subtask_assets: int = 3,
        **kwargs: Any,
    ):
        self.target_quantities = [float(x) for x in (target_quantities or [0.01, 0.025, 0.05, 0.10])]
        self.remaining_time_levels = [max(1, int(x)) for x in (remaining_time_levels or [1, 2, 5, 10])]
        self.initial_risky_fraction = float(np.clip(initial_risky_fraction, 0.0, 0.995))
        self.subtask_assets = max(1, int(subtask_assets))
        self._curriculum_remaining = 1.0
        super().__init__(*args, **kwargs)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        weights = np.zeros(self.asset_dim, dtype=np.float64)
        weights[: self.stock_dim] = self.initial_risky_fraction / float(self.stock_dim)
        weights[self.cash_index] = 1.0 - self.initial_risky_fraction
        self.previous_weights = normalize_simplex(weights)
        return self._obs(), info

    def _remaining_context(self) -> float:
        return float(self._curriculum_remaining)

    def _target_weights(self) -> np.ndarray:
        prev = self.previous_weights.copy()
        if not self.target_quantities:
            return super()._target_weights()

        q_idx = self.day % len(self.target_quantities)
        t_idx = (self.day // max(1, len(self.target_quantities))) % len(self.remaining_time_levels)
        direction_sign = 1.0 if ((self.day // max(1, len(self.target_quantities) * len(self.remaining_time_levels))) % 2 == 0) else -1.0
        self._curriculum_remaining = float(self.remaining_time_levels[t_idx])
        requested = float(self.target_quantities[q_idx])

        asset_start = self.day % self.stock_dim
        asset_indices = [(asset_start + offset) % self.stock_dim for offset in range(self.subtask_assets)]
        per_asset = requested / float(len(asset_indices))
        target = prev.copy()
        if direction_sign > 0:
            available_cash = float(target[self.cash_index])
            spend = min(requested, available_cash)
            if spend > 0.0:
                per_asset = spend / float(len(asset_indices))
                for idx in asset_indices:
                    target[idx] += per_asset
                target[self.cash_index] -= spend
        else:
            remaining_to_sell = requested
            for idx in asset_indices:
                sell = min(float(target[idx]), per_asset)
                target[idx] -= sell
                target[self.cash_index] += sell
                remaining_to_sell -= sell
            if remaining_to_sell > 1e-12:
                for idx in np.argsort(-target[: self.stock_dim]):
                    if remaining_to_sell <= 1e-12:
                        break
                    sell = min(float(target[idx]), remaining_to_sell)
                    target[idx] -= sell
                    target[self.cash_index] += sell
                    remaining_to_sell -= sell
        return normalize_simplex(np.maximum(target, 0.0))
