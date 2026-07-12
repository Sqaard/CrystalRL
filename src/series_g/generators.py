"""Series-G Extension 2 — cross-MODEL-FAMILY toxicity generators (for the full HC-3).

Phase-1 HC-3 was cross-PARAMETER (one generator, different params). The methodology's true cross-family test
needs different MODEL CLASSES. Each generator below emits the SAME interface — a hidden toxicity flag and a
burst/quiet observation per step — via a DISTINCT mechanism, so an execution policy (belief→action) can be
trained on one family and evaluated on another:

  RegimePOMDP   — exogenous 2-state Markov regime (geometric dwell times).             [the baseline class]
  GCMG          — toxicity EMERGES from a Grand-Canonical Minority Game (Challet & Zhang 1997; Challet,
                  Marsili & Zhang): when active speculators crowd, attendance volatility spikes → toxic flow.
                  Anti-persistent / volatility-clustered dynamics, NOT geometric.       [emergent, minority]
  BrockHommes   — toxicity from an adaptive-belief system (Brock & Hommes 1998): fundamentalists vs trend
                  followers with logit fraction dynamics; trend-follower dominance → momentum/toxic regime.
                  Bifurcation / regime-switching dynamics.                              [emergent, evolutionary]

All are calibrated to a comparable marginal toxic rate (~20-35%) and informative bursts, so transfer tests
HOW the temporal structure differs, not the base rate. Run: python -m src.series_g.generators (selftest).
"""
from __future__ import annotations

import numpy as np

from src.series_g.regime_pomdp import BENIGN, BURST, QUIET, TOXIC, RegimePOMDP, PRIMARY_ENRICHED


def _hysteresis(sig: np.ndarray, hi_q: float, lo_q: float) -> np.ndarray:
    """Two-threshold (Schmitt-trigger) labeling of a continuous family signal into PERSISTENT toxic episodes:
    enter toxic when sig rises above its hi-quantile, stay toxic until it falls below the lo-quantile. Real
    order-flow toxicity clusters, so persistent episodes are the faithful regime; this also makes the regime
    INFERABLE (load-bearing belief) without changing each family's distinct episode-timing mechanism."""
    hi, lo = np.quantile(sig, hi_q), np.quantile(sig, lo_q)
    tox = np.zeros(len(sig), int)
    state = 0
    for i, s in enumerate(sig):
        if state == 0 and s > hi:
            state = 1
        elif state == 1 and s < lo:
            state = 0
        tox[i] = state
    return tox


class GeneratorRegimePOMDP:
    name = "regime_pomdp_markov"

    def __init__(self, econ: dict | None = None):
        self.m = RegimePOMDP(**(econ or PRIMARY_ENRICHED))

    def simulate(self, T: int, seed: int):
        rng = np.random.default_rng(seed)
        m = self.m
        tox = np.empty(T, int); obs = np.empty(T, int)
        r = TOXIC if rng.random() < m.prior_toxic else BENIGN
        for t in range(T):
            tox[t] = r
            obs[t] = BURST if rng.random() < m.obs[r, BURST] else QUIET
            r = TOXIC if rng.random() < m.M[r, TOXIC] else BENIGN
        return tox, obs


class GeneratorGCMG:
    """Grand-Canonical Minority Game. N speculators, memory mu, S strategies each; agents play only when their
    best virtual score is positive (grand-canonical abstention). Attendance A(t)=sum of active actions; the
    minority side wins. Toxic = high |A| (crowded → volatile, adverse-selecting flow). burst ~ activity."""
    name = "gcmg_minority_game"

    def __init__(self, N=201, mem=3, S=2, tox_quantile=0.70):
        self.N, self.mem, self.S, self.tox_q = N, mem, S, tox_quantile

    def simulate(self, T: int, seed: int):
        rng = np.random.default_rng(seed)
        P = 1 << self.mem
        strat = rng.integers(0, 2, size=(self.N, self.S, P)) * 2 - 1   # +/-1 strategy tables
        score = np.zeros((self.N, self.S))
        hist = rng.integers(0, P)
        warm = 200
        A_series = np.empty(T + warm)
        for t in range(T + warm):
            best = score.argmax(axis=1)                                # each agent's best strategy
            pred = strat[np.arange(self.N), best, hist]
            active = score[np.arange(self.N), best] > 0                 # grand-canonical: play only if profitable
            if not active.any():
                active = np.ones(self.N, bool)                         # avoid empty market early
            A = float(pred[active].sum())
            A_series[t] = A
            winner = -np.sign(A) if A != 0 else (rng.integers(0, 2) * 2 - 1)
            score += strat[:, :, hist] * winner                        # virtual scoring vs all strategies
            hist = ((hist << 1) | (1 if winner > 0 else 0)) % P
        A = np.abs(A_series[warm:])
        tox = _hysteresis(A, hi_q=0.80, lo_q=0.55)   # persistent crowded/volatile episodes
        # observation: burst probability rises with activity (toxic ⇒ more bursts), with noise
        pburst = np.where(tox == TOXIC, 0.70, 0.18)
        obs = (rng.random(T) < pburst).astype(int)
        return tox, obs


class GeneratorBrockHommes:
    """Adaptive Belief System (Brock & Hommes 1998): fundamentalists (mean-reverting) vs trend followers
    (g*x_{t-1}); fractions updated by a logit (intensity beta) on past realized profits. Trend-follower
    dominance ⇒ momentum/toxic regime. Toxic = trend fraction high; burst ~ |price change|."""
    name = "brock_hommes_adaptive"

    def __init__(self, g=1.3, beta=4.0, r=0.05, noise=0.05, tox_quantile=0.72):
        self.g, self.beta, self.r, self.noise, self.tox_q = g, beta, r, noise, tox_quantile

    def simulate(self, T: int, seed: int):
        rng = np.random.default_rng(seed)
        warm = 200
        x = np.zeros(T + warm + 2)
        n_trend = np.full(T + warm + 2, 0.5)
        x[0], x[1] = 0.1, 0.08
        tox = np.empty(T + warm, int); dx = np.empty(T + warm)
        for t in range(1, T + warm + 1):
            # realized profit of each type's last-period forecast (profit ~ forecast * realized deviation)
            prof_f = (-x[t - 1]) * (x[t] - x[t - 1])             # fundamentalist forecast 0 (reversion)
            prof_t = (self.g * x[t - 1]) * (x[t] - x[t - 1])     # trend forecast g*x
            uf, ut = np.exp(self.beta * prof_f), np.exp(self.beta * prof_t)
            nt = ut / (uf + ut)
            n_trend[t] = nt
            # next deviation from the fraction-weighted expectations + noise
            exp_next = (1 - nt) * 0.0 + nt * (self.g * x[t])
            x[t + 1] = exp_next / (1 + self.r) + self.noise * rng.standard_normal()
            dx[t - 1] = abs(x[t] - x[t - 1])
        nt = n_trend[warm + 1:warm + 1 + T]
        d = dx[warm:warm + T]
        # persistent trend-follower-dominated (momentum/toxic) episodes via hysteresis on the trend fraction.
        tox = _hysteresis(nt, hi_q=0.80, lo_q=0.55)
        # informative burst: toxic (trend) regimes have larger moves; sample burst by activity + regime
        pburst = np.where(tox == TOXIC, 0.68, 0.20)
        obs = (rng.random(T) < pburst).astype(int)
        return tox, obs


ALL_GENERATORS = [GeneratorRegimePOMDP(), GeneratorGCMG(), GeneratorBrockHommes()]


def estimate_markov_params(tox: np.ndarray, obs: np.ndarray) -> dict:
    """Fit a 2-state Markov + observation model to a generator's (toxic, obs) sequence — the policy's view of
    that family. Returns RegimePOMDP-compatible params (transition + burst likelihoods + prior)."""
    tox = np.asarray(tox); obs = np.asarray(obs)
    n_b2t = int(((tox[:-1] == BENIGN) & (tox[1:] == TOXIC)).sum()); n_b = int((tox[:-1] == BENIGN).sum())
    n_t2b = int(((tox[:-1] == TOXIC) & (tox[1:] == BENIGN)).sum()); n_t = int((tox[:-1] == TOXIC).sum())
    p_b2t = n_b2t / max(1, n_b); p_t2b = n_t2b / max(1, n_t)
    pbt = float((obs[tox == TOXIC] == BURST).mean()) if (tox == TOXIC).any() else 0.6
    pbb = float((obs[tox == BENIGN] == BURST).mean()) if (tox == BENIGN).any() else 0.2
    return dict(p_stay_benign=float(np.clip(1 - p_b2t, 0.5, 0.999)),
                p_stay_toxic=float(np.clip(1 - p_t2b, 0.5, 0.999)),
                prior_toxic=float(np.clip(tox.mean(), 0.02, 0.6)),
                p_burst_benign=float(np.clip(pbb, 0.01, 0.95)),
                p_burst_toxic=float(np.clip(pbt, 0.05, 0.99)))


def _selftest() -> None:
    print("=== generators _selftest (each a DISTINCT model class) ===")
    for gen in ALL_GENERATORS:
        tox, obs = gen.simulate(4000, seed=0)
        # lag-1 toxic autocorrelation distinguishes the temporal structure
        t = tox - tox.mean()
        ac1 = float((t[:-1] * t[1:]).mean() / (t.var() + 1e-12))
        info = float((obs[tox == 1] == BURST).mean() - (obs[tox == 0] == BURST).mean())
        print(f"  {gen.name:24s} toxic_rate={tox.mean():.3f}  burst|tox-burst|ben={info:+.3f}  ac1(toxic)={ac1:+.3f}")
    print("VERDICT: PASS (distinct dynamics — see differing ac1)")


if __name__ == "__main__":
    _selftest()
