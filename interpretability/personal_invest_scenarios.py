"""Pure historical-scenario primitives for the W1 personalized-investing interface.

This module deliberately does *not* produce a forward forecast.  It fixes three accounting defects in the
legacy P-4 prototype so that every interface query is at least evaluated on the same empirical measure:

* risky and cash legs are resampled with the same stationary-bootstrap indices;
* drawdown starts from wealth=1 and exposes annual and full-horizon definitions separately;
* probabilities are returned together with conditional shortfall/breach magnitudes.

W3 owns forward-looking scenario ensembles and calibration.  Outputs from this module must therefore keep
the status ``UNCALIBRATED_HISTORICAL_SCENARIOS`` until the W3/W8 gates pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


TRADING_DAYS = 252
HISTORICAL_SCENARIO_STATUS = "UNCALIBRATED_HISTORICAL_SCENARIOS"
DrawdownMetric = Literal["annual_max_drawdown", "full_horizon_max_drawdown"]


@dataclass(frozen=True)
class PathDistribution:
    """Net-return path outcomes. Drawdowns are positive loss magnitudes in ``[0, 1]``."""

    cagr: np.ndarray
    terminal_wealth: np.ndarray
    annual_max_drawdown: np.ndarray
    full_horizon_max_drawdown: np.ndarray
    horizon_years: int
    status: str = HISTORICAL_SCENARIO_STATUS
    minimum_wealth: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.horizon_years, int) or self.horizon_years < 1:
            raise ValueError("horizon_years must be a positive integer")
        arrays = {}
        for name in (
            "cagr",
            "terminal_wealth",
            "annual_max_drawdown",
            "full_horizon_max_drawdown",
        ):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1).copy()
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain at least two finite path outcomes")
            arrays[name] = values
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all PathDistribution outcomes must have the same number of paths")
        if bool((arrays["cagr"] <= -1.0).any()) or bool((arrays["terminal_wealth"] < 0.0).any()):
            raise ValueError("CAGR/terminal wealth contains an impossible loss")
        implied_terminal = np.power(1.0 + arrays["cagr"], self.horizon_years)
        if not np.allclose(arrays["terminal_wealth"], implied_terminal, rtol=1e-8, atol=1e-10):
            raise ValueError("terminal_wealth is inconsistent with CAGR and horizon")
        annual = arrays["annual_max_drawdown"]
        full = arrays["full_horizon_max_drawdown"]
        if bool(((annual < 0.0) | (annual > 1.0) | (full < 0.0) | (full > 1.0)).any()):
            raise ValueError("drawdown magnitudes must be in [0, 1]")
        if bool((full + 1e-12 < annual).any()):
            raise ValueError("full-horizon MDD cannot be below reset-annual MDD on the same path")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.minimum_wealth is not None:
            minimum = np.asarray(self.minimum_wealth, dtype=float).reshape(-1).copy()
            if len(minimum) != next(iter(lengths)) or not np.isfinite(minimum).all():
                raise ValueError("minimum_wealth must be finite and path-aligned")
            if bool((minimum < 0.0).any()):
                raise ValueError("minimum_wealth cannot be negative")
            if bool((minimum > np.minimum(1.0, arrays["terminal_wealth"]) + 1e-12).any()):
                raise ValueError("minimum_wealth cannot exceed the initial or terminal wealth on its path")
            if bool((arrays["full_horizon_max_drawdown"] + 1e-12 < 1.0 - minimum).any()):
                raise ValueError("minimum_wealth is inconsistent with full-horizon drawdown")
            minimum.setflags(write=False)
            object.__setattr__(self, "minimum_wealth", minimum)

    def drawdowns(self, metric: DrawdownMetric) -> np.ndarray:
        if metric == "annual_max_drawdown":
            return self.annual_max_drawdown
        if metric == "full_horizon_max_drawdown":
            return self.full_horizon_max_drawdown
        raise ValueError(f"unknown drawdown metric: {metric}")


def _as_returns(values: Iterable[float], label: str) -> np.ndarray:
    out = np.asarray(list(values), dtype=float).reshape(-1)
    if len(out) < 2:
        raise ValueError(f"{label} needs at least two observations")
    if not np.isfinite(out).all():
        raise ValueError(f"{label} contains non-finite observations")
    if bool((out <= -1.0).any()):
        raise ValueError(f"{label} contains a return <= -100%")
    return out


def stationary_bootstrap_indices(
    sample_length: int,
    path_length: int,
    n_paths: int,
    expected_block_days: float,
    seed: int,
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices with circular continuation."""
    if sample_length < 2:
        raise ValueError("sample_length must be >= 2")
    if path_length < 1 or n_paths < 1:
        raise ValueError("path_length and n_paths must be positive")
    if expected_block_days <= 1:
        raise ValueError("expected_block_days must be > 1")

    rng = np.random.default_rng(seed)
    restart_probability = 1.0 / float(expected_block_days)
    idx = np.empty((n_paths, path_length), dtype=np.int32)
    idx[:, 0] = rng.integers(0, sample_length, size=n_paths, dtype=np.int32)
    for t in range(1, path_length):
        restart = rng.random(n_paths) < restart_probability
        continuation = (idx[:, t - 1] + 1) % sample_length
        fresh = rng.integers(0, sample_length, size=n_paths, dtype=np.int32)
        idx[:, t] = np.where(restart, fresh, continuation)
    return idx


def drawdown_metrics(path_returns: np.ndarray, trading_days: int = TRADING_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(worst annual MDD, full-horizon MDD)`` as positive magnitudes.

    The initial wealth point is explicitly prepended.  Annual MDD resets its high-water mark at each
    ``trading_days`` boundary; full-horizon MDD carries the high-water mark through the whole path.
    """
    returns = np.asarray(path_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] < 1:
        raise ValueError("path_returns must have shape [n_paths, n_days]")
    if not np.isfinite(returns).all() or bool((returns <= -1.0).any()):
        raise ValueError("path_returns contains invalid values")
    if trading_days < 1:
        raise ValueError("trading_days must be positive")

    wealth = np.ones((returns.shape[0], returns.shape[1] + 1), dtype=float)
    wealth[:, 1:] = np.cumprod(1.0 + returns, axis=1)
    full_running_peak = np.maximum.accumulate(wealth, axis=1)
    full_mdd = np.max(1.0 - wealth / full_running_peak, axis=1)

    annual = []
    for start in range(0, returns.shape[1], trading_days):
        stop = min(start + trading_days, returns.shape[1])
        # Include wealth immediately before the first return in the annual slice.
        segment = wealth[:, start : stop + 1]
        running_peak = np.maximum.accumulate(segment, axis=1)
        annual.append(np.max(1.0 - segment / running_peak, axis=1))
    annual_mdd = np.max(np.column_stack(annual), axis=1)
    return annual_mdd, full_mdd


def simulate_historical_blend(
    risky_daily_returns: Iterable[float],
    cash_daily_returns: Iterable[float],
    cash_weight: float,
    horizon_years: int,
    *,
    n_paths: int = 3000,
    expected_block_days: float = 63.0,
    seed: int = 26,
) -> PathDistribution:
    """Jointly resample an aligned risky/cash blend and return its empirical path distribution."""
    risky = _as_returns(risky_daily_returns, "risky_daily_returns")
    cash = _as_returns(cash_daily_returns, "cash_daily_returns")
    if risky.shape != cash.shape:
        raise ValueError("risky and cash paths must be date-aligned and have equal length")
    if not 0.0 <= cash_weight <= 1.0:
        raise ValueError("cash_weight must be in [0, 1]")
    if not isinstance(horizon_years, int) or horizon_years < 1:
        raise ValueError("horizon_years must be a positive integer")

    # This is a daily constant-weight blend. Transaction costs, taxes and FX belong in the supplied net
    # return streams; W3 will model them jointly rather than silently omitting them.
    blended = (1.0 - cash_weight) * risky + cash_weight * cash
    indices = stationary_bootstrap_indices(
        len(blended), TRADING_DAYS * horizon_years, n_paths, expected_block_days, seed
    )
    sampled = blended[indices]
    terminal = np.prod(1.0 + sampled, axis=1)
    cagr = np.power(terminal, 1.0 / horizon_years) - 1.0
    annual_mdd, full_mdd = drawdown_metrics(sampled)
    wealth_path = np.cumprod(1.0 + sampled, axis=1)
    return PathDistribution(
        cagr=cagr,
        terminal_wealth=terminal,
        annual_max_drawdown=annual_mdd,
        full_horizon_max_drawdown=full_mdd,
        horizon_years=horizon_years,
        minimum_wealth=np.minimum(1.0, np.min(wealth_path, axis=1)),
    )


def summarize_distribution(
    distribution: PathDistribution,
    *,
    targets: Iterable[float],
    drawdown_budgets: Iterable[float],
) -> dict:
    """Create structurally paired return/shortfall and drawdown/breach statistics."""
    ret = distribution.cagr
    result = {
        "scenario_status": distribution.status,
        "horizon_years": distribution.horizon_years,
        "expected_cagr": float(np.mean(ret)),
        "cagr_quantiles": {
            "p10": float(np.quantile(ret, 0.10)),
            "p20": float(np.quantile(ret, 0.20)),
            "p50": float(np.quantile(ret, 0.50)),
            "p80": float(np.quantile(ret, 0.80)),
            "p90": float(np.quantile(ret, 0.90)),
        },
        "goals": {},
        "drawdown": {},
    }
    for target in targets:
        target = float(target)
        miss = ret < target
        result["goals"][str(target)] = {
            "probability_reach": float(np.mean(~miss)),
            "expected_shortfall_given_miss": (
                float(np.mean(target - ret[miss])) if bool(miss.any()) else None
            ),
            "cvar_shortfall_90": (
                float(np.mean(np.sort(target - ret[miss])[-max(1, int(np.ceil(miss.sum() * 0.10))):]))
                if bool(miss.any()) else None
            ),
        }
    for metric in ("annual_max_drawdown", "full_horizon_max_drawdown"):
        dd = distribution.drawdowns(metric)
        by_budget = {}
        for budget in drawdown_budgets:
            budget = float(budget)
            breach = dd > budget
            by_budget[str(budget)] = {
                "probability_within": float(np.mean(~breach)),
                "expected_excess_given_breach": (
                    float(np.mean(dd[breach] - budget)) if bool(breach.any()) else None
                ),
                "drawdown_p95": float(np.quantile(dd, 0.95)),
            }
        result["drawdown"][metric] = by_budget
    return result


def conservative_return_floor(distribution: PathDistribution, probability: float) -> float:
    """Return the empirical lower CAGR bound exceeded on approximately ``probability`` of paths.

    This is the POINT estimate of the (1-probability) quantile (e.g. the 20th-percentile CAGR at
    probability=0.8). It carries finite-sample sampling error; ``conservative_return_floor_lcb``
    wraps it in a lower-confidence bound and is what the product object r*(p,H,u) should quote.
    """
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    # Exact empirical inverse of P(R >= q) >= probability, including non-grid probabilities and equality.
    # NumPy interpolation conventions do not provide this coverage guarantee for arbitrary p.
    ordered = np.sort(distribution.cagr)
    required_paths = int(np.ceil(probability * len(ordered)))
    return float(ordered[len(ordered) - required_paths])


def conservative_return_floor_lcb(
    distribution: PathDistribution,
    probability: float,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 12345,
) -> dict:
    """The product-grade floor: a lower-confidence bound (LCB) on the (1-probability) CAGR quantile.

    The point floor (``conservative_return_floor``) is one draw of an estimator with sampling error;
    quoting it as-is over-promises. Here we resample the path population ``n_boot`` times, recompute
    the quantile on each resample, and return the ``1-confidence`` lower percentile of that bootstrap
    distribution. So ``lcb`` is the number we can defend at ``confidence`` against estimation error,
    ``point`` is the raw quantile, and ``se`` is the bootstrap standard error. This realizes the
    r*(p,H,u) = LCB[Q_{1-p}(CAGR)] object: quote ``lcb``, not ``point``.
    """
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    cagr = np.asarray(distribution.cagr, dtype=float)
    n = len(cagr)
    q = 1.0 - probability  # the lower quantile level (0.20 for probability=0.8)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = cagr[rng.integers(0, n, size=n)]
        boot[b] = np.quantile(sample, q, method="lower")
    return {
        "point": float(np.quantile(cagr, q, method="lower")),
        "lcb": float(np.quantile(boot, 1.0 - confidence, method="lower")),
        "se": float(np.std(boot, ddof=1)),
        "confidence": float(confidence),
        "probability": float(probability),
    }
