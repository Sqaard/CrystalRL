"""Two-agent root/stock training environments for Stage 0.1 T2.

This module intentionally separates:

- root policy: scalar invested fraction q with root-only state/reward
- stock policy: risky simplex u with stock/private state/reward

The two environments can be trained iteratively with SB3 PPO while sharing the
same market simulator and teacher-trace provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.dow30_sectors import get_sector_map
from src.ppo.execution.helpers import EPS, normalize_simplex, normalize_stock_simplex
from src.ppo.pretrain_teachers import read_csv_source
from src.ppo.weight_panel import WeightPanel


def _feature_indices(feature_columns: list[str], names: list[str]) -> list[int]:
    index = {name: idx for idx, name in enumerate(feature_columns)}
    missing = sorted(set(names).difference(index))
    if missing:
        raise ValueError(f"Features missing from panel: {missing}")
    return [index[name] for name in names]


def _date_key(value: object) -> str:
    return str(pd.Timestamp(value).date())


def _entropy(weights: np.ndarray) -> float:
    w = np.maximum(np.asarray(weights, dtype=np.float64), EPS)
    w = w / np.sum(w)
    return float(-np.sum(w * np.log(w)))


def stock_order_book_proxy_dim(config: dict[str, Any] | None) -> int:
    """Return per-stock execution-book proxy dimension.

    The project does not currently have true intraday level-2 order-book data.
    These modes are daily execution-context proxies, not a real LOB simulator.
    `synthetic_lob_proxy` is the closest mode: it constructs pseudo bid/ask
    price offsets and depth levels from OHLCV/liquidity features.
    """

    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return 0
    mode = str(cfg.get("mode", "daily_ohlcv_proxy")).lower()
    if mode in {"daily_ohlcv_proxy", "urgency_liquidity_proxy"}:
        return 12
    if mode == "synthetic_lob_proxy":
        levels = max(1, int(cfg.get("levels", 3)))
        # common context:
        # prev_stock, signed_budget, abs_budget, buy_budget, sell_budget,
        # remaining_frac, urgency, spread_proxy, impact_proxy, imbalance_proxy
        # plus per level: bid_offset, ask_offset, bid_depth, ask_depth.
        return 10 + 4 * levels
    raise ValueError(
        "stock_order_book_proxy.mode must be one of "
        "['daily_ohlcv_proxy', 'urgency_liquidity_proxy', 'synthetic_lob_proxy']"
    )


@dataclass
class PortfolioState:
    day: int
    previous_weights: np.ndarray
    portfolio_value: float
    peak_value: float
    previous_drawdown: float
    last_turnover: float


class TeacherTraceProvider:
    """Averages train-only teacher traces by date into q/u targets."""

    def __init__(
        self,
        *,
        manifest_csv: str | Path,
        fold_id: str,
        teacher_ids: list[str],
        tickers: list[str],
        target_source: str = "anchor",
    ):
        self.tickers = list(tickers)
        self.asset_names = self.tickers + ["CASH"]
        self.fold_id = str(fold_id)
        self.teacher_ids = list(teacher_ids)
        self.target_source = str(target_source)
        manifest_path = Path(manifest_csv)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing teacher manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        selected = manifest[
            manifest["fold"].astype(str).eq(self.fold_id)
            & manifest["teacher_id"].astype(str).isin(self.teacher_ids)
            & manifest["has_train_trace_daily"].astype(bool)
        ].copy()
        if selected.empty:
            raise ValueError(f"No teacher traces for fold={fold_id}, teacher_ids={teacher_ids}")
        frames: list[pd.DataFrame] = []
        for _, row in selected.iterrows():
            frame = read_csv_source(str(row["source_path"]))
            prefix = self._choose_prefix(frame)
            cols = ["date"] + [f"{prefix}{name}" for name in self.asset_names]
            sub = frame[cols].copy()
            sub.columns = ["date"] + self.asset_names
            sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
            frames.append(sub)
        merged = pd.concat(frames, ignore_index=True)
        grouped = merged.groupby("date", as_index=False)[self.asset_names].mean()
        self.by_date = grouped.set_index("date")

    def _choose_prefix(self, frame: pd.DataFrame) -> str:
        candidates = {
            "raw": ["raw_weight_", "anchor_weight_", "target_weight_", "executed_weight_"],
            "anchor": ["anchor_weight_", "raw_weight_", "target_weight_", "executed_weight_"],
            "target": ["target_weight_", "anchor_weight_", "raw_weight_", "executed_weight_"],
            "executed": ["executed_weight_", "target_weight_", "anchor_weight_", "raw_weight_"],
        }.get(self.target_source, ["anchor_weight_", "target_weight_", "executed_weight_"])
        required = set(self.asset_names)
        for prefix in candidates:
            available = {col.removeprefix(prefix) for col in frame.columns if col.startswith(prefix)}
            if required.issubset(available):
                return prefix
        raise ValueError(f"No complete teacher weight prefix for target_source={self.target_source}")

    def weights_for_date(self, date: object) -> np.ndarray:
        key = _date_key(date)
        if key not in self.by_date.index:
            weights = np.zeros(len(self.asset_names), dtype=np.float64)
            weights[:-1] = 1.0 / max(len(self.tickers), 1)
            return weights
        weights = self.by_date.loc[key, self.asset_names].to_numpy(dtype=np.float64)
        return normalize_simplex(np.maximum(weights, 0.0))

    def q_for_date(self, date: object) -> float:
        weights = self.weights_for_date(date)
        return float(np.sum(weights[:-1]))

    def u_for_date(self, date: object) -> np.ndarray:
        weights = self.weights_for_date(date)
        return normalize_stock_simplex(weights[:-1])


class TwoAgentBaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: WeightPanel,
        *,
        root_feature_names: list[str],
        stock_feature_names: list[str],
        root_raw_window_feature_names: list[str] | None = None,
        root_raw_window_days: int = 0,
        root_window_days: int = 20,
        q_min: float = 0.0,
        q_max: float = 0.995,
        transaction_cost_pct: float = 0.001,
        initial_amount: float = 1_000_000.0,
        reward_scale: float = 100.0,
        teacher_provider: TeacherTraceProvider | None = None,
        sector_map_name: str = "dow30_static",
        reward_config: dict[str, Any] | None = None,
        stock_order_book_proxy: dict[str, Any] | None = None,
        root_low_level_diagnostics: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.panel = panel
        self.stock_dim = len(panel.tickers)
        self.asset_dim = self.stock_dim + 1
        self.cash_index = self.stock_dim
        self.root_indices = _feature_indices(panel.feature_columns, root_feature_names)
        self.stock_indices = _feature_indices(panel.feature_columns, stock_feature_names)
        self.raw_window_indices = _feature_indices(panel.feature_columns, root_raw_window_feature_names or [])
        self.root_raw_window_days = max(0, int(root_raw_window_days))
        self.root_window_days = max(1, int(root_window_days))
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.initial_amount = float(initial_amount)
        self.reward_scale = float(reward_scale)
        self.teacher_provider = teacher_provider
        self.reward_config = dict(reward_config or {})
        self.stock_order_book_proxy = dict(stock_order_book_proxy or {})
        self.root_low_level_diagnostics = dict(root_low_level_diagnostics or {})
        sector_map = get_sector_map(sector_map_name)
        self.group_ids = self._group_ids(sector_map)
        self.n_groups = max(self.group_ids) + 1 if self.group_ids else 1
        self.reset()

    def _group_ids(self, sector_map: dict[str, str]) -> list[int]:
        sectors = [sector_map.get(ticker, "other") for ticker in self.panel.tickers]
        sector_to_id = {sector: idx for idx, sector in enumerate(sorted(set(sectors)))}
        return [sector_to_id[sector] for sector in sectors]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state = PortfolioState(
            day=0,
            previous_weights=np.eye(1, self.asset_dim, self.cash_index, dtype=np.float64).reshape(-1),
            portfolio_value=self.initial_amount,
            peak_value=self.initial_amount,
            previous_drawdown=0.0,
            last_turnover=0.0,
        )
        self.root_anchor_q = 0.0
        self.root_anchor_start_day = 0
        self.low_level_diag = np.zeros(6, dtype=np.float32)
        return self._obs(), {}

    @property
    def day(self) -> int:
        return int(self.state.day)

    def _done(self) -> bool:
        return self.day >= len(self.panel.dates) - 1

    def _portfolio_state_obs(self) -> np.ndarray:
        weights = self.state.previous_weights
        gross_exposure = float(np.sum(weights[: self.stock_dim]))
        hhi = float(np.sum(weights**2))
        ret_since_start = self.state.portfolio_value / self.initial_amount - 1.0
        return np.array(
            [
                weights[self.cash_index],
                gross_exposure,
                hhi,
                self.state.previous_drawdown,
                self.state.last_turnover,
                ret_since_start,
            ],
            dtype=np.float32,
        )

    def _root_raw_window_obs(self) -> np.ndarray:
        if self.root_raw_window_days <= 0:
            return np.zeros(0, dtype=np.float32)
        end = self.day + 1
        start = max(0, end - self.root_raw_window_days)
        window = self.panel.features[start:end, 0, self.raw_window_indices].astype(np.float32)
        if window.shape[0] == 0:
            window = np.zeros((1, len(self.raw_window_indices)), dtype=np.float32)
        pad = self.root_raw_window_days - window.shape[0]
        if pad > 0:
            window = np.concatenate([np.repeat(window[:1], pad, axis=0), window], axis=0)
        return window[-self.root_raw_window_days :].reshape(-1).astype(np.float32)

    def root_obs(self) -> np.ndarray:
        root_features = self.panel.features[self.day, 0, self.root_indices].astype(np.float32)
        return np.concatenate(
            [
                root_features,
                self._portfolio_state_obs(),
                self._root_raw_window_obs(),
                self._low_level_diag_obs(),
            ]
        ).astype(np.float32)

    def _low_level_diag_obs(self) -> np.ndarray:
        if not bool(self.root_low_level_diagnostics.get("enabled", False)):
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.low_level_diag, dtype=np.float32)

    def _update_low_level_diag(self, info: dict[str, Any], stock_reward: float, u: np.ndarray) -> None:
        if not bool(self.root_low_level_diagnostics.get("enabled", False)):
            return
        target = np.asarray(info.get("target_weights", np.zeros(self.asset_dim)), dtype=np.float64)
        pre = np.asarray(info.get("pre_trade_weights", np.zeros(self.asset_dim)), dtype=np.float64)
        diag = np.array(
            [
                float(info.get("stock_turnover_l1", 0.0)),
                float(info.get("turnover_l1", 0.0)),
                float(np.sum(np.abs(target[: self.stock_dim] - pre[: self.stock_dim]))),
                _entropy(u),
                float(stock_reward) / max(self.reward_scale, EPS),
                float(info.get("net_return", 0.0)) - float(info.get("benchmark_return", 0.0)),
            ],
            dtype=np.float32,
        )
        alpha = float(self.root_low_level_diagnostics.get("ewma_alpha", 0.30))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.low_level_diag = (1.0 - alpha) * self.low_level_diag + alpha * diag

    def stock_obs(self, *, q_anchor: float | None = None, remaining_days: int | None = None) -> np.ndarray:
        stock_features = self.panel.features[self.day, :, self.stock_indices].reshape(-1).astype(np.float32)
        q_prev = float(np.sum(self.state.previous_weights[: self.stock_dim]))
        q_anchor = q_prev if q_anchor is None else float(q_anchor)
        remaining = float(max(1, remaining_days or 1))
        root_private = np.array(
            [
                q_anchor,
                q_prev,
                q_anchor - q_prev,
                remaining / float(self.root_window_days),
                self._feature_value("regime_entropy", 0.0),
                self._feature_value("VIX_change_5d", 0.0),
            ],
            dtype=np.float32,
        )
        order_book_proxy = self._stock_order_book_proxy_obs(q_anchor=q_anchor, q_prev=q_prev, remaining=remaining)
        return np.concatenate([stock_features, order_book_proxy, self._portfolio_state_obs(), root_private]).astype(np.float32)

    def _feature_vector(self, name: str, default: float = 0.0) -> np.ndarray:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return np.full(self.stock_dim, float(default), dtype=np.float32)
        return self.panel.features[self.day, :, idx].astype(np.float32)

    def _stock_order_book_proxy_obs(self, *, q_anchor: float, q_prev: float, remaining: float) -> np.ndarray:
        cfg = self.stock_order_book_proxy
        if not bool(cfg.get("enabled", False)):
            return np.zeros(0, dtype=np.float32)

        mode = str(cfg.get("mode", "daily_ohlcv_proxy")).lower()
        expected_dim = stock_order_book_proxy_dim(cfg)

        prev_stock = self.state.previous_weights[: self.stock_dim].astype(np.float32)
        buy_budget = max(float(q_anchor - q_prev), 0.0)
        sell_budget = max(float(q_prev - q_anchor), 0.0)
        signed_budget = float(q_anchor - q_prev)
        abs_budget = abs(signed_budget)
        urgency = abs(float(q_anchor - q_prev)) / max(float(remaining), 1.0)
        remaining_frac = float(remaining) / float(max(self.root_window_days, 1))

        high_low_range = self._feature_vector(str(cfg.get("range_feature", "high_low_range")))
        atr_rel = self._feature_vector(str(cfg.get("atr_feature", "atr_rel")))
        realized_vol = self._feature_vector(str(cfg.get("vol_feature", "realized_vol_20d")))
        volume_ratio = self._feature_vector(str(cfg.get("liquidity_feature", "volume_ratio")), default=1.0)
        volume_z = self._feature_vector(str(cfg.get("volume_z_feature", "volume_zscore_20d_raw")))
        dollar_volume_z = self._feature_vector(str(cfg.get("dollar_volume_z_feature", "dollar_volume_zscore_20d_raw")))

        liquidity = np.maximum(volume_ratio, 1e-3)
        urgency_vec = np.full(self.stock_dim, urgency, dtype=np.float32)
        buy_budget_vec = np.full(self.stock_dim, buy_budget, dtype=np.float32)
        sell_budget_vec = np.full(self.stock_dim, sell_budget, dtype=np.float32)
        signed_budget_vec = np.full(self.stock_dim, signed_budget, dtype=np.float32)
        abs_budget_vec = np.full(self.stock_dim, abs_budget, dtype=np.float32)
        remaining_vec = np.full(self.stock_dim, remaining_frac, dtype=np.float32)
        queue_risk = (urgency_vec * np.maximum(realized_vol, 0.0) / liquidity).astype(np.float32)

        if mode == "synthetic_lob_proxy":
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
                impact_scale
                * np.maximum(abs_budget_vec, urgency_vec)
                * (1.0 + np.maximum(realized_vol, 0.0))
                / np.maximum(liquidity, 1e-3)
            ).astype(np.float32)
            # Positive imbalance means buy-side/momentum pressure; negative
            # means sell-side pressure. This is only a daily proxy, not true
            # queue imbalance from level-2 book volumes.
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
                bid_offset = (-level_scale * spread_proxy).astype(np.float32)
                ask_offset = (level_scale * spread_proxy).astype(np.float32)
                depth_base = np.maximum(liquidity, 1e-3) / float(level)
                bid_depth = (depth_base * (1.0 - 0.5 * np.maximum(imbalance_proxy, 0.0))).astype(np.float32)
                ask_depth = (depth_base * (1.0 + 0.5 * np.minimum(imbalance_proxy, 0.0))).astype(np.float32)
                parts.extend([bid_offset, ask_offset, bid_depth, ask_depth])
            out = np.stack(parts, axis=1).reshape(-1).astype(np.float32)
            if out.shape[0] != self.stock_dim * expected_dim:
                raise RuntimeError(
                    f"synthetic_lob_proxy produced dim={out.shape[0] // self.stock_dim}, expected={expected_dim}"
                )
            return out

        if mode not in {"daily_ohlcv_proxy", "urgency_liquidity_proxy"}:
            # Trigger the centralized validation/error message.
            stock_order_book_proxy_dim(cfg)

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

    def _feature_value(self, name: str, default: float = 0.0) -> float:
        try:
            idx = self.panel.feature_columns.index(name)
        except ValueError:
            return float(default)
        return float(self.panel.features[self.day, 0, idx])

    def _obs(self) -> np.ndarray:
        raise NotImplementedError

    def _weights_from_q_u(self, q: float, u: np.ndarray) -> np.ndarray:
        q = float(np.clip(q, self.q_min, self.q_max))
        u = normalize_stock_simplex(u)
        weights = np.zeros(self.asset_dim, dtype=np.float64)
        weights[: self.stock_dim] = q * u
        weights[self.cash_index] = 1.0 - q
        return normalize_simplex(weights)

    def _step_target(self, target_weights: np.ndarray) -> dict[str, Any]:
        if self._done():
            return {"terminated": True, "reward_base": 0.0}
        step_day = self.day
        prev_weights = self.state.previous_weights.copy()
        target_weights = normalize_simplex(target_weights)
        trade_delta = target_weights - prev_weights
        stock_turnover_l1 = float(np.sum(np.abs(trade_delta[: self.stock_dim])))
        turnover_l1 = float(np.sum(np.abs(trade_delta)))
        transaction_cost = self.transaction_cost_pct * stock_turnover_l1
        asset_returns = self.panel.returns_next[self.day]
        benchmark_return = float(np.mean(asset_returns))
        gross_return = float(np.dot(target_weights[: self.stock_dim], asset_returns))
        net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0
        old_value = self.state.portfolio_value
        new_value = old_value * (1.0 + net_return)
        peak_value = max(self.state.peak_value, new_value)
        drawdown = new_value / max(peak_value, EPS) - 1.0
        drawdown_increment = max(0.0, self.state.previous_drawdown - drawdown)
        post_market = self._weights_after_market(target_weights, asset_returns)
        self.state = PortfolioState(
            day=self.day + 1,
            previous_weights=post_market,
            portfolio_value=new_value,
            peak_value=peak_value,
            previous_drawdown=drawdown,
            last_turnover=turnover_l1,
        )
        return {
            "terminated": self._done(),
            "date": _date_key(self.panel.dates[step_day]),
            "step_day": step_day,
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
            "post_market_weights": post_market,
            "pre_trade_weights": prev_weights,
            "trade_delta_weights": trade_delta,
            "portfolio_value": new_value,
        }

    def _weights_after_market(self, weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        next_values = np.zeros_like(weights)
        next_values[: self.stock_dim] = weights[: self.stock_dim] * (1.0 + asset_returns)
        next_values[self.cash_index] = weights[self.cash_index]
        return normalize_simplex(np.maximum(next_values, 0.0))

    def _benchmark_return(self) -> float:
        return float(np.mean(self.panel.returns_next[self.day]))

    def _future_stock_return(self, horizon: int = 5) -> np.ndarray:
        return self._future_stock_return_from(self.day, horizon)

    def _future_stock_return_from(self, start_day: int, horizon: int = 5) -> np.ndarray:
        start = int(start_day)
        end = min(len(self.panel.returns_next), start + max(1, int(horizon)))
        if end <= start:
            return np.zeros(self.stock_dim, dtype=np.float64)
        return np.prod(1.0 + self.panel.returns_next[start:end], axis=0) - 1.0

    def _group_returns_next(self) -> np.ndarray:
        asset_returns = self.panel.returns_next[self.day]
        return self._group_returns_from(asset_returns)

    def _group_returns_from(self, asset_returns: np.ndarray) -> np.ndarray:
        out = np.zeros(self.stock_dim, dtype=np.float64)
        for gid in range(self.n_groups):
            idx = np.asarray([i for i, g in enumerate(self.group_ids) if g == gid], dtype=int)
            if idx.size:
                out[idx] = float(np.mean(asset_returns[idx]))
        return out

    def _safe_feature_index(self, name: str) -> int:
        try:
            return self.panel.feature_columns.index(name)
        except ValueError:
            return 0

    def _root_reward(self, info: dict[str, Any]) -> float:
        cfg = self.reward_config.get("root", {})
        net_return = float(info.get("net_return", 0.0))
        drawdown_inc = float(info.get("drawdown_increment", 0.0))
        cash = float(info["target_weights"][self.cash_index])
        opportunity_gate = 1.0 / (1.0 + np.exp(-float(self._feature_value("SP500_Trend", 0.0))))
        benchmark_return = float(info.get("benchmark_return", 0.0))
        cash_opp = cash * max(benchmark_return, 0.0) * opportunity_gate
        reward = (
            float(cfg.get("return_weight", 1.0)) * net_return
            - float(cfg.get("drawdown_penalty", 2.0)) * drawdown_inc
            - float(cfg.get("cash_opportunity_penalty", 0.25)) * cash_opp
        )
        return float(self.reward_scale * reward)

    def _stock_reward(self, info: dict[str, Any], u: np.ndarray) -> float:
        cfg = self.reward_config.get("stock", {})
        delta = np.asarray(info["trade_delta_weights"][: self.stock_dim], dtype=np.float64)
        step_day = int(info.get("step_day", max(0, self.day - 1)))
        future = self._future_stock_return_from(step_day, int(cfg.get("flow_horizon", 5)))
        residual_future = future - float(np.mean(future))
        flow_select = float(np.dot(delta, residual_future))
        asset_returns = np.asarray(info.get("asset_returns", np.zeros(self.stock_dim)), dtype=np.float64)
        group_returns = self._group_returns_from(asset_returns)
        group_relative = float(np.dot(u, asset_returns - group_returns))
        stock_vol = self.panel.features[step_day, :, self._safe_feature_index("realized_vol_20d")]
        vol_adjusted_cost = float(np.sum(np.abs(delta) * np.maximum(stock_vol, 0.0)))
        reward = (
            float(cfg.get("flow_select_weight", 0.30)) * flow_select
            + float(cfg.get("group_relative_weight", 0.20)) * group_relative
            + float(cfg.get("entropy_bonus", 0.001)) * _entropy(u)
            - float(cfg.get("vol_adjusted_cost", 0.02)) * vol_adjusted_cost
            - float(cfg.get("transaction_cost_weight", 1.0)) * float(info.get("transaction_cost", 0.0))
        )
        return float(self.reward_scale * reward)


class RootAllocationEnv(TwoAgentBaseEnv):
    """Root agent env: action is q, reward is root risk/cash quality."""

    def __init__(self, *args, frozen_stock_model: Any | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_stock_model = frozen_stock_model
        self.action_space = spaces.Box(low=np.array([self.q_min], dtype=np.float32), high=np.array([self.q_max], dtype=np.float32))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.root_obs().shape, dtype=np.float32)

    def _obs(self) -> np.ndarray:
        return self.root_obs()

    def _stock_action(self, q_anchor: float, remaining_days: int) -> np.ndarray:
        if self.frozen_stock_model is not None:
            action, _ = self.frozen_stock_model.predict(self.stock_obs(q_anchor=q_anchor, remaining_days=remaining_days), deterministic=True)
            return normalize_stock_simplex(np.asarray(action, dtype=np.float64).reshape(-1))
        if self.teacher_provider is not None:
            return self.teacher_provider.u_for_date(self.panel.dates[self.day])
        return np.full(self.stock_dim, 1.0 / self.stock_dim, dtype=np.float64)

    def step(self, action):
        q_anchor = float(np.clip(np.asarray(action).reshape(-1)[0], self.q_min, self.q_max))
        total_reward = 0.0
        infos: list[dict[str, Any]] = []
        for substep in range(self.root_window_days):
            if self._done():
                break
            remaining = max(1, self.root_window_days - substep)
            q_prev = float(np.sum(self.state.previous_weights[: self.stock_dim]))
            q_scheduled = q_prev + (q_anchor - q_prev) / float(remaining)
            u = self._stock_action(q_anchor=q_anchor, remaining_days=remaining)
            info = self._step_target(self._weights_from_q_u(q_scheduled, u))
            reward = self._root_reward(info)
            total_reward += reward
            info.update({"root_reward": reward, "q_anchor": q_anchor, "q_scheduled": q_scheduled, "root_substep": substep + 1})
            infos.append(info)
            if info["terminated"]:
                break
        obs = self.root_obs() if not self._done() else np.zeros(self.observation_space.shape, dtype=np.float32)
        macro_info = dict(infos[-1]) if infos else {"terminated": True}
        macro_info["daily_steps"] = infos
        macro_info["macro_root_reward"] = total_reward
        return obs, float(total_reward), self._done(), False, macro_info


class StockAllocationEnv(TwoAgentBaseEnv):
    """Stock agent env: action is risky simplex u, reward is stock selection."""

    def __init__(self, *args, frozen_root_model: Any | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_root_model = frozen_root_model
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.stock_obs().shape, dtype=np.float32)
        self.root_anchor_q = self.teacher_provider.q_for_date(self.panel.dates[0]) if self.teacher_provider else 0.5

    def _obs(self) -> np.ndarray:
        remaining = self._root_remaining_days()
        return self.stock_obs(q_anchor=self.root_anchor_q, remaining_days=remaining)

    def _root_remaining_days(self) -> int:
        elapsed = max(0, self.day - int(self.root_anchor_start_day))
        return max(1, self.root_window_days - elapsed)

    def _refresh_root_anchor_if_due(self) -> None:
        if self.day == 0 or (self.day - int(self.root_anchor_start_day)) >= self.root_window_days:
            if self.frozen_root_model is not None:
                action, _ = self.frozen_root_model.predict(self.root_obs(), deterministic=True)
                self.root_anchor_q = float(np.asarray(action).reshape(-1)[0])
            elif self.teacher_provider is not None:
                self.root_anchor_q = self.teacher_provider.q_for_date(self.panel.dates[self.day])
            self.root_anchor_q = float(np.clip(self.root_anchor_q, self.q_min, self.q_max))
            self.root_anchor_start_day = int(self.day)

    def step(self, action):
        if self._done():
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
        self._refresh_root_anchor_if_due()
        remaining = self._root_remaining_days()
        q_prev = float(np.sum(self.state.previous_weights[: self.stock_dim]))
        q_scheduled = q_prev + (self.root_anchor_q - q_prev) / float(remaining)
        u = normalize_stock_simplex(np.asarray(action, dtype=np.float64).reshape(-1))
        target = self._weights_from_q_u(q_scheduled, u)
        info = self._step_target(target)
        reward = self._stock_reward(info, u)
        info.update(
            {
                "stock_reward": reward,
                "q_anchor": self.root_anchor_q,
                "q_scheduled": q_scheduled,
                "root_remaining_days": remaining,
                "risky_entropy": _entropy(u),
            }
        )
        obs = self.stock_obs(q_anchor=self.root_anchor_q, remaining_days=self._root_remaining_days()) if not self._done() else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), self._done(), False, info


class JointTwoAgentEnv(TwoAgentBaseEnv):
    """Joint simulator for hierarchical root/stock PPO.

    Root actions are macro anchors held for `root_window_days`; stock actions are
    sampled daily and receive root private state.  This class is intentionally
    used by the custom T3 trainer instead of pretending that vanilla SB3 can
    optimize the two temporal levels in one ordinary Gym step contract.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_action_space = spaces.Box(
            low=np.array([self.q_min], dtype=np.float32),
            high=np.array([self.q_max], dtype=np.float32),
        )
        self.stock_action_space = spaces.Box(low=0.0, high=1.0, shape=(self.stock_dim,), dtype=np.float32)
        self.root_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.root_obs().shape, dtype=np.float32)
        self.stock_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.stock_obs().shape, dtype=np.float32)

    def _obs(self) -> np.ndarray:
        return self.root_obs()

    def reset_joint(self) -> None:
        self.reset()

    def root_remaining_days(self, root_anchor_start_day: int) -> int:
        elapsed = max(0, int(self.day) - int(root_anchor_start_day))
        return max(1, self.root_window_days - elapsed)

    def step_joint(self, *, q_anchor: float, root_anchor_start_day: int, stock_action: np.ndarray) -> dict[str, Any]:
        remaining = self.root_remaining_days(root_anchor_start_day)
        q_prev = float(np.sum(self.state.previous_weights[: self.stock_dim]))
        q_scheduled = q_prev + (float(q_anchor) - q_prev) / float(remaining)
        u = normalize_stock_simplex(np.asarray(stock_action, dtype=np.float64).reshape(-1))
        target = self._weights_from_q_u(q_scheduled, u)
        info = self._step_target(target)
        root_reward = self._root_reward(info)
        stock_reward = self._stock_reward(info, u)
        self._update_low_level_diag(info, stock_reward, u)
        info.update(
            {
                "root_reward": root_reward,
                "stock_reward": stock_reward,
                "q_anchor": float(q_anchor),
                "q_scheduled": q_scheduled,
                "root_remaining_days": remaining,
                "risky_entropy": _entropy(u),
            }
        )
        return info
