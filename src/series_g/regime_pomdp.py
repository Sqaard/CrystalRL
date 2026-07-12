"""Series-G Phase 0 — the tabular regime-switching POMDP (the "Belief-Driven Toxicity" execution core).

This is the smallest faithful realization of the FINAL_G / SERIES_G_METHODOLOGY_AND_PLAN.md §3 design:
a hidden regime modulates ORDER-FLOW TOXICITY (not the return drift — `regime1.txt` §6: a hidden HMM over
returns collapses to alpha-search, so the regime is attached to toxicity/adverse-selection instead), and
the agent must INFER the regime from a noisy order-flow signal, SELECT an execution mode, and TIME it
against inventory — the three things that make a behaviorally-complex-yet-interpretable optimum.

State (hidden + controllable):
  regime r ∈ {0 BENIGN, 1 TOXIC}      hidden, exogenous Markov chain (sticky; toxic is the minority regime)
  inventory I ∈ {0..I_max}            controllable, observed
  time t ∈ {0..T}                     observed, finite horizon

Observation o ∈ {0 QUIET, 1 BURST}    emitted by the regime: a burst of fills/cancels signals TOXIC flow
                                       (Easley-O'Hara 1992 — inter-trade intensity is informative). The agent
                                       never sees r; it maintains belief b = P(toxic | o-history) (Bayes filter).

Action a ∈ {0 PROVIDE, 1 ABSTAIN, 2 AGGRESS}:
  PROVIDE  passive liquidity: captures spread `s` in BENIGN, suffers adverse selection `-α` in TOXIC; fills
           you with +1 inventory (you accumulate a position).  -> the ride/fade decision the belief drives.
  ABSTAIN  wait: no fill, inventory unchanged.
  AGGRESS  cross the spread to UNWIND: pays impact `-c`, reduces |inventory| by 1.
Per-step carrying cost `-h·I`; terminal liquidation `-L·I_T`. (Long-only inventory keeps Phase 0 minimal;
the §6.4 Interface-A env generalizes to signed inventory + skew later.)

The optimum is belief-DEPENDENT by construction: myopically PROVIDE iff (1-b)·s > b·α  ⇔  b < s/(s+α) (the
Glosten-Milgrom Bayes cutoff, §3.6) — modulated by inventory (high I ⇒ unwind) and horizon (near T ⇒ unwind).
Whether that belief-dependence is *materially* worth tracking is exactly the Phase-0 gate (phase0_gate.py).

Sources: Glosten-Milgrom (1985); Easley-O'Hara (1992); Gode-Sunder (1993, the ZI floor); Hamilton (1989,
regime-switching filter). See reports/SERIES_G_METHODOLOGY_AND_PLAN.md §3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BENIGN, TOXIC = 0, 1
QUIET, BURST = 0, 1
PROVIDE, ABSTAIN, AGGRESS = 0, 1, 2
ACTION_NAMES = {PROVIDE: "PROVIDE", ABSTAIN: "ABSTAIN", AGGRESS: "AGGRESS"}

# The Phase-1 gated/enriched economics (config B + the GM-JSI toxic-unwind cost): the env that cleared all
# four falsifiers. Single source of truth, reused by the multi-asset env + the cross-family generators.
PRIMARY_ENRICHED = dict(spread=2.0, adverse=4.0, aggress_cost=0.3, aggress_cost_toxic=1.5, hold_cost=0.05)


@dataclass
class RegimePOMDP:
    """The tabular 2-state (benign/toxic) regime-POMDP: belief filter, economics, and inventory dynamics."""
    # --- hidden regime process (exogenous; actions do NOT move the regime) ---
    p_stay_benign: float = 0.95          # P(benign->benign); toxic is the sticky minority regime
    p_stay_toxic: float = 0.80           # P(toxic->toxic)
    prior_toxic: float = 0.20            # initial P(toxic)
    # --- observation model P(burst | regime): a burst signals toxic flow ---
    p_burst_benign: float = 0.15
    p_burst_toxic: float = 0.70
    # --- economics ---
    spread: float = 1.0                  # s: PROVIDE capture in BENIGN
    adverse: float = 3.0                 # α: PROVIDE loss in TOXIC  (α>s ⇒ providing into toxicity is costly)
    hold_cost: float = 0.10              # h: per-step carrying cost per unit inventory
    aggress_cost: float = 0.50           # c: AGGRESS impact to unwind one unit (BENIGN-regime resilience)
    aggress_cost_toxic: float = None     # GM-JSI: unwinding in TOXIC is dearer (low resilience, Obizhaeva-Wang)
    liq_penalty: float = 2.0             # L: terminal penalty per unit unliquidated inventory
    I_max: int = 4
    T: int = 20
    # derived
    M: np.ndarray = field(init=False)            # (2,2) regime transition
    obs: np.ndarray = field(init=False)          # (2,2) P(o|r): rows regime, cols {quiet,burst}

    def __post_init__(self):
        self.M = np.array([[self.p_stay_benign, 1 - self.p_stay_benign],
                           [1 - self.p_stay_toxic, self.p_stay_toxic]], dtype=float)
        self.obs = np.array([[1 - self.p_burst_benign, self.p_burst_benign],
                             [1 - self.p_burst_toxic, self.p_burst_toxic]], dtype=float)
        if self.aggress_cost_toxic is None:
            self.aggress_cost_toxic = self.aggress_cost   # backward-compatible: regime-independent unwind

    # ---------------------------------------------------------------- regime / belief
    def stationary_toxic(self) -> float:
        """Analytic stationary P(toxic) of the 2-state chain = M01 / (M01 + M10)."""
        m01, m10 = self.M[BENIGN, TOXIC], self.M[TOXIC, BENIGN]
        return float(m01 / (m01 + m10))

    def gm_threshold(self) -> float:
        """The Glosten-Milgrom myopic Bayes cutoff: PROVIDE iff P(toxic) < s/(s+α). Undefined (nan) when
        s+α ≤ 0 (no adverse selection ⇒ no cutoff; the no-modulation control)."""
        denom = self.spread + self.adverse
        return float(self.spread / denom) if denom > 1e-12 else float("nan")

    def predict(self, b: float) -> float:
        """Time-update the toxic belief through the regime chain (no observation)."""
        return float((1 - b) * self.M[BENIGN, TOXIC] + b * self.M[TOXIC, TOXIC])

    def obs_prob(self, b_pred: float, o: int) -> float:
        """P(o | predicted belief)."""
        return float((1 - b_pred) * self.obs[BENIGN, o] + b_pred * self.obs[TOXIC, o])

    def update(self, b_pred: float, o: int) -> float:
        """Bayes measurement-update of the toxic belief given observation o."""
        num = b_pred * self.obs[TOXIC, o]
        den = self.obs_prob(b_pred, o)
        return float(num / den) if den > 1e-12 else b_pred

    def open_loop_forecast(self) -> np.ndarray:
        """Belief-BLIND regime marginal P(toxic) at each t=0..T (prior iterated through M, NO observations).
        This is exactly the 'belief' available to a policy that ignores the order-flow signal."""
        out = np.empty(self.T + 1)
        b = self.prior_toxic
        out[0] = b
        for t in range(1, self.T + 1):
            b = self.predict(b)
            out[t] = b
        return out

    # ---------------------------------------------------------------- dynamics + reward
    def inventory_next(self, a: int, I: int) -> int:
        """Deterministic inventory transition: PROVIDE fills +1 (capped at I_max), AGGRESS unwinds −1 (floored at 0), ABSTAIN holds."""
        if a == PROVIDE:
            return min(I + 1, self.I_max)
        if a == AGGRESS:
            return max(I - 1, 0)
        return I  # ABSTAIN

    def reward(self, r: int, a: int, I: int) -> float:
        """Immediate reward of action a in regime r at inventory I (carrying cost included)."""
        rew = -self.hold_cost * I                       # per-step carrying cost (always)
        if a == PROVIDE and I < self.I_max:             # at capacity PROVIDE cannot fill -> no spread/adverse
            rew += self.spread if r == BENIGN else -self.adverse
        elif a == AGGRESS and I > 0:                     # unwind: dearer in the toxic (low-resilience) regime
            rew += -(self.aggress_cost_toxic if r == TOXIC else self.aggress_cost)
        return float(rew)

    def expected_reward(self, b: float, a: int, I: int) -> float:
        """E_{r~Bernoulli(b)}[reward(r,a,I)] — the belief-averaged immediate reward."""
        return (1 - b) * self.reward(BENIGN, a, I) + b * self.reward(TOXIC, a, I)

    def terminal(self, I: int) -> float:
        """Terminal liquidation penalty for carrying inventory I past the horizon (−L·I)."""
        return -self.liq_penalty * I


def _selftest() -> None:
    """Ground checks before the solver trusts the model. Run: python -m src.series_g.regime_pomdp"""
    m = RegimePOMDP()
    # 1. stationary toxic = 0.05 / (0.05 + 0.20) = 0.20
    st = m.stationary_toxic()
    assert abs(st - 0.20) < 1e-9, st
    # 2. GM threshold = 1/(1+3) = 0.25
    assert abs(m.gm_threshold() - 0.25) < 1e-9, m.gm_threshold()
    # 3. belief filter: many BURSTs drive belief up; many QUIETs drive it down
    b = m.prior_toxic
    for _ in range(15):
        b = m.update(m.predict(b), BURST)
    assert b > 0.85, f"bursts should push belief high, got {b}"
    b = m.prior_toxic
    for _ in range(15):
        b = m.update(m.predict(b), QUIET)
    assert b < 0.10, f"quiets should push belief low, got {b}"
    # 4. reward signs
    assert m.reward(BENIGN, PROVIDE, 0) == 1.0
    assert m.reward(TOXIC, PROVIDE, 0) == -3.0
    assert m.reward(BENIGN, AGGRESS, 2) < 0  # impact + carrying
    assert m.terminal(3) == -6.0
    # 5. open-loop forecast converges toward stationary (no observations)
    fc = m.open_loop_forecast()
    assert abs(fc[-1] - st) < 0.02, f"open-loop should approach stationary {st}, got {fc[-1]}"
    assert fc[0] == m.prior_toxic
    # 6. inventory dynamics
    assert m.inventory_next(PROVIDE, m.I_max) == m.I_max  # capped
    assert m.inventory_next(AGGRESS, 0) == 0              # floored
    assert m.inventory_next(ABSTAIN, 2) == 2
    print("=== regime_pomdp _selftest ===")
    print(f"  stationary P(toxic)={st:.3f}  GM threshold={m.gm_threshold():.3f}  "
          f"open-loop forecast t=0,5,T = {fc[0]:.3f},{fc[5]:.3f},{fc[-1]:.3f}")
    print("  belief filter, reward signs, inventory dynamics: OK")
    print("VERDICT: PASS")


if __name__ == "__main__":
    _selftest()
