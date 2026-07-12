"""Series-G Phase 0 — the solver + the GATE.

Three knowable-optimum objects on the tabular regime-POMDP (regime_pomdp.py), then the Phase-0 gate.

  1. BELIEF-AWARE optimum  — exact POMDP value via belief-MDP backward induction over a discretized toxic
     belief b ∈ [0,1] × inventory × time. This is the best policy that TRACKS the regime from order flow.
  2. BELIEF-BLIND optimum   — the best policy on (inventory, time) ONLY, using the open-loop regime
     forecast b̄_t (no observation update). The fair "ignores the signal" baseline.
  3. GODE-SUNDER ZI FLOOR    — uniform-random feasible action, exact forward evaluation (unconstrained vs
     budget/feasibility-constrained — the GS honesty diagnostic).

THE GATE (SERIES_G_METHODOLOGY_AND_PLAN.md §7 Phase-0, §8): the belief-aware optimum must STRICTLY DOMINATE
the belief-blind optimum by a MATERIAL margin — i.e. the order-flow signal has real value of information.
If it does not, the regime is not load-bearing and HC-1 would be unbeatable for the WRONG reason: the
generator is mis-specified and must be fixed (strengthen the regime's modulation) BEFORE Phase 1.

GM validation (§3.6): the belief-averaged immediate PROVIDE-vs-ABSTAIN reward must cross zero exactly at the
Glosten-Milgrom cutoff b* = s/(s+α) — the knowable analytic optimum the solver is graded against.

Run: python -m src.series_g.phase0_gate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.series_g.regime_pomdp import (ABSTAIN, AGGRESS, BENIGN, BURST, PROVIDE, QUIET, TOXIC,
                                       ACTION_NAMES, RegimePOMDP)

ACTIONS = (PROVIDE, ABSTAIN, AGGRESS)
OUT = Path(__file__).resolve().parent / "phase0_gate_report.json"


# ---------------------------------------------------------------- belief-aware POMDP value iteration
def solve_belief_aware(m: RegimePOMDP, n_bins: int = 121):
    g = np.linspace(0.0, 1.0, n_bins)
    V = np.zeros((m.T + 1, n_bins, m.I_max + 1))
    pol = np.full((m.T, n_bins, m.I_max + 1), -1, dtype=int)
    for I in range(m.I_max + 1):
        V[m.T, :, I] = m.terminal(I)
    # belief transition on the grid (t/a/I-independent): predict, then per-observation update + weight
    b_pred = (1 - g) * m.M[BENIGN, TOXIC] + g * m.M[TOXIC, TOXIC]
    po, bo = {}, {}
    for o in (QUIET, BURST):
        p_o = (1 - b_pred) * m.obs[BENIGN, o] + b_pred * m.obs[TOXIC, o]
        b_up = np.where(p_o > 1e-12, (b_pred * m.obs[TOXIC, o]) / np.maximum(p_o, 1e-12), b_pred)
        po[o], bo[o] = p_o, b_up
    for t in range(m.T - 1, -1, -1):
        for I in range(m.I_max + 1):
            Q = np.empty((3, n_bins))
            for a in ACTIONS:
                Inext = m.inventory_next(a, I)
                imm = (1 - g) * m.reward(BENIGN, a, I) + g * m.reward(TOXIC, a, I)
                Vn = V[t + 1, :, Inext]
                cont = po[QUIET] * np.interp(bo[QUIET], g, Vn) + po[BURST] * np.interp(bo[BURST], g, Vn)
                Q[a] = imm + cont
            V[t, :, I] = Q.max(axis=0)
            pol[t, :, I] = Q.argmax(axis=0)
    return g, V, pol


# ---------------------------------------------------------------- belief-blind (open-loop) value iteration
def solve_belief_blind(m: RegimePOMDP):
    fc = m.open_loop_forecast()
    V = np.zeros((m.T + 1, m.I_max + 1))
    pol = np.full((m.T, m.I_max + 1), -1, dtype=int)
    for I in range(m.I_max + 1):
        V[m.T, I] = m.terminal(I)
    for t in range(m.T - 1, -1, -1):
        bt = fc[t]
        for I in range(m.I_max + 1):
            Q = [(1 - bt) * m.reward(BENIGN, a, I) + bt * m.reward(TOXIC, a, I) + V[t + 1, m.inventory_next(a, I)]
                 for a in ACTIONS]
            V[t, I] = max(Q)
            pol[t, I] = int(np.argmax(Q))
    return V, pol


# ---------------------------------------------------------------- Gode-Sunder zero-intelligence floor
def _feasible(m: RegimePOMDP, t: int, I: int, constrained: bool):
    """Constrained ZI (budget/feasibility): cannot PROVIDE at capacity, and must AGGRESS-unwind in the last
    I steps if holding inventory (so it doesn't eat the liquidation penalty) — the GS 'structure narrows the
    feasible set' rule. Unconstrained ZI = all actions always."""
    if not constrained:
        return list(ACTIONS)
    if I > 0 and (m.T - t) <= I:        # must liquidate to clear by the horizon
        return [AGGRESS]
    acts = [ABSTAIN, AGGRESS] if I >= m.I_max else list(ACTIONS)  # no PROVIDE at capacity
    return acts if I > 0 else [a for a in acts if a != AGGRESS]   # AGGRESS pointless at I=0


def zi_value(m: RegimePOMDP, constrained: bool) -> float:
    """Exact expected return of the uniform-random feasible policy, by forward-propagating the joint
    distribution over (regime, inventory)."""
    P = np.zeros((2, m.I_max + 1))
    P[BENIGN, 0] = 1 - m.prior_toxic
    P[TOXIC, 0] = m.prior_toxic
    total = 0.0
    for t in range(m.T):
        Pn = np.zeros_like(P)
        for r in (BENIGN, TOXIC):
            for I in range(m.I_max + 1):
                p = P[r, I]
                if p <= 0:
                    continue
                acts = _feasible(m, t, I, constrained)
                pa = 1.0 / len(acts)
                for a in acts:
                    total += p * pa * m.reward(r, a, I)
                    In = m.inventory_next(a, I)
                    for rn in (BENIGN, TOXIC):
                        Pn[rn, In] += p * pa * m.M[r, rn]
        P = Pn
    for r in (BENIGN, TOXIC):
        for I in range(m.I_max + 1):
            total += P[r, I] * m.terminal(I)
    return float(total)


# ---------------------------------------------------------------- belief-aware value at the start state
def aware_start_value(m: RegimePOMDP, g, V) -> float:
    return float(np.interp(m.prior_toxic, g, V[0, :, 0]))


def gm_validation(m: RegimePOMDP, g, pol) -> dict:
    """The belief-averaged immediate PROVIDE-vs-ABSTAIN reward at I=0 must cross 0 at b*=s/(s+α). Also report
    where the FULL belief-aware policy stops PROVIDING at I=0 (shifted by continuation/inventory/horizon)."""
    b = g
    diff = ((1 - b) * m.reward(BENIGN, PROVIDE, 0) + b * m.reward(TOXIC, PROVIDE, 0)) - \
           ((1 - b) * m.reward(BENIGN, ABSTAIN, 0) + b * m.reward(TOXIC, ABSTAIN, 0))
    cross = float(np.interp(0.0, diff[::-1], b[::-1]))  # b where diff=0 (diff decreasing in b)
    # full-policy PROVIDE region at I=0, mid-horizon
    tmid = m.T // 2
    provide_mask = pol[tmid, :, 0] == PROVIDE
    last_provide_b = float(g[provide_mask].max()) if provide_mask.any() else float("nan")
    return {"gm_analytic_threshold": round(m.gm_threshold(), 4),
            "myopic_reward_crossing": round(cross, 4),
            "full_policy_last_PROVIDE_belief_at_I0_midT": round(last_provide_b, 4),
            "matches_analytic": bool(abs(cross - m.gm_threshold()) < 1e-3)}


def policy_summary(m: RegimePOMDP, g, pol) -> dict:
    """Human-readable: for a few (I) at mid-horizon, the action as belief rises — does it ride->fade->unwind?"""
    tmid = m.T // 2
    bands = [0.05, 0.20, 0.35, 0.60, 0.90]
    rows = {}
    for I in (0, 2, m.I_max):
        seq = []
        for bb in bands:
            a = int(pol[tmid, int(np.argmin(np.abs(g - bb))), I])
            seq.append(f"b={bb}:{ACTION_NAMES[a]}")
        rows[f"I={I}"] = seq
    return rows


def run(m: RegimePOMDP) -> dict:
    g, Va, pol_a = solve_belief_aware(m)
    Vb, pol_b = solve_belief_blind(m)
    v_aware = aware_start_value(m, g, Va)
    v_blind = float(Vb[0, 0])
    v_zi_u = zi_value(m, constrained=False)
    v_zi_c = zi_value(m, constrained=True)
    voi = v_aware - v_blind                                  # value of the order-flow signal
    rel = voi / abs(v_blind) if abs(v_blind) > 1e-9 else float("inf")
    # MATERIAL margin: VoI must be a clear fraction of the belief-blind value AND clear vs the ZI spread
    zi_spread = abs(v_zi_c - v_zi_u) + 1e-9
    material = bool(voi > 0.05 * abs(v_blind) and voi > 0.25 * zi_spread and voi > 0.05)
    gm = gm_validation(m, g, pol_a)
    out = {
        "params": {k: getattr(m, k) for k in ("p_stay_benign", "p_stay_toxic", "prior_toxic", "p_burst_benign",
                   "p_burst_toxic", "spread", "adverse", "hold_cost", "aggress_cost", "liq_penalty", "I_max", "T")},
        "values": {"belief_aware_optimum": round(v_aware, 4), "belief_blind_optimum": round(v_blind, 4),
                   "ZI_floor_unconstrained": round(v_zi_u, 4), "ZI_floor_constrained": round(v_zi_c, 4)},
        "GATE": {"value_of_information": round(voi, 4), "voi_relative_to_blind": round(rel, 4),
                 "ZI_honesty_gap_(c_minus_u)": round(v_zi_c - v_zi_u, 4),
                 "ordering_aware>blind>ZI": bool(v_aware > v_blind > v_zi_u),
                 "material_margin": material,
                 "VERDICT": ("PASS — regime belief is load-bearing; optimum requires belief-memory" if material else
                             "FAIL — belief not materially load-bearing; FIX the generator before Phase 1")},
        "gm_validation": gm,
        "policy_ride_fade_unwind_midT": policy_summary(m, g, pol_a),
    }
    return out


if __name__ == "__main__":
    rep = run(RegimePOMDP())
    print(json.dumps(rep, indent=2))
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\n[phase0] wrote {OUT.name}  ->  GATE: {rep['GATE']['VERDICT']}")
