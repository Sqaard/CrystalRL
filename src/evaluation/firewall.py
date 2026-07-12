"""Finance-safe evaluation firewall for the CHRL self-evolving loop.

Additive module (pure stdlib + numpy). Guards the audit found missing:
  C1 multiple-testing on selection  -> expected_max_sharpe / deflated_sharpe_ratio
  C1 no held-out confirmation       -> held_out_confirmation
  C2 OOD not enforced in rollout    -> OODGate + safe_alpha (call INSIDE the rollout)
  H5 no train->val embargo          -> embargoed_validation_start
  stats for intervention claims     -> block_bootstrap_ci
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

_N = NormalDist()
_EULER = 0.5772156649015329


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """Expected maximum Sharpe achievable by selection alone over n_trials
    independent trials with Sharpes ~ N(0, sr_std^2) (Bailey & Lopez de Prado)."""
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _EULER) * z1 + _EULER * z2)


def prob_sharpe_ratio(observed_sr: float, n_obs: int, sr_benchmark: float = 0.0,
                      skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true SR > sr_benchmark) given sample length and non-normality. observed_sr
    and sr_benchmark in the SAME per-observation units as n_obs."""
    if n_obs < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr ** 2))
    z = (observed_sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom
    return _N.cdf(z)


def deflated_sharpe_ratio(observed_sr: float, sr_std_across_trials: float, n_trials: int,
                          n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR against the selection-inflated benchmark expected_max_sharpe(...). All
    Sharpes per-observation. Returns P(true SR > selection noise floor) in [0,1]."""
    sr0 = expected_max_sharpe(sr_std_across_trials, n_trials)
    return prob_sharpe_ratio(observed_sr, n_obs, sr_benchmark=sr0, skew=skew, kurt=kurt)


def held_out_confirmation(selection_scores: dict, confirmation_scores: dict,
                          min_confirmation: float) -> dict:
    """Pick the winner on selection folds, then confirm on disjoint folds. Report the
    confirmation score, never the selection max."""
    if not selection_scores:
        raise ValueError("no selection scores")
    winner = max(selection_scores, key=selection_scores.get)
    conf = confirmation_scores.get(winner, float("nan"))
    return {"winner": winner, "selection_score": selection_scores[winner],
            "confirmation_score": conf, "confirmed": bool(conf == conf and conf >= min_confirmation)}


def embargoed_validation_start(trading_dates: Sequence, train_end, embargo_trading_days: int):
    """First validation date >= embargo_trading_days trading days after train_end, so
    rolling features in validation never overlap train. trading_dates must be sorted."""
    after = [d for d in trading_dates if d > train_end]
    if len(after) <= embargo_trading_days:
        raise ValueError("not enough trading days after train_end for the embargo")
    return after[embargo_trading_days]


@dataclass
class OODGate:
    """Mahalanobis gate fit on natural hidden states; an edited hidden state is
    admissible only if it stays within the natural manifold."""
    mean: np.ndarray
    cov_inv: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, hidden: np.ndarray, percentile: float = 99.0, ridge: float = 1e-6) -> "OODGate":
        H = np.asarray(hidden, dtype=float)
        mean = H.mean(axis=0)
        cov = np.cov(H, rowvar=False) + ridge * np.eye(H.shape[1])
        cov_inv = np.linalg.inv(cov)
        d = H - mean
        dist = np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv, d))
        return cls(mean=mean, cov_inv=cov_inv, threshold=float(np.percentile(dist, percentile)))

    def distance(self, h: np.ndarray) -> float:
        d = np.asarray(h, dtype=float) - self.mean
        return float(np.sqrt(d @ self.cov_inv @ d))

    def passes(self, h: np.ndarray) -> bool:
        return self.distance(h) <= self.threshold


def safe_alpha(natural_hidden: np.ndarray, direction: np.ndarray, requested_alpha: float,
               gate: OODGate, shrink: float = 0.5, max_steps: int = 12) -> float:
    """Largest alpha in [0, requested_alpha] whose edited hidden state passes the OOD
    gate. 0.0 if even a tiny edit is off-manifold. Use in the rollout in place of raw alpha (C2)."""
    if requested_alpha <= 0:
        return 0.0
    alpha = float(requested_alpha)
    for _ in range(max_steps):
        if gate.passes(natural_hidden + alpha * direction):
            return alpha
        alpha *= shrink
    return 0.0


def block_bootstrap_ci(x, n_boot: int = 10000, block: int = 5, ci: float = 0.95, seed: int = 0):
    """Circular block-bootstrap CI for the mean of a serially-correlated series
    (e.g. paired daily (promote - control) differences). Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nblocks = int(math.ceil(n / block))
    offs = np.arange(block)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=nblocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n] % n
        means[b] = x[idx].mean()
    lo = float(np.percentile(means, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(means, (1.0 + ci) / 2.0 * 100.0))
    return float(x.mean()), lo, hi
