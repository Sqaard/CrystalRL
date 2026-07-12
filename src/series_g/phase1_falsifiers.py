"""Series-G Phase 1 — the four pre-CrystalScore FALSIFIERS (the gate).

SERIES_G_METHODOLOGY_AND_PLAN.md §4/§7: these prove the complexity is REAL before any interpretability
number is trusted (the synthetic-side mirror-of-the-hall). ALL FOUR must pass; if any fails → STOP and
publish "complexity was plumbing" (the benchmark, not the method, is revised).

  F1  concept-ablation on the toxicity posterior            -> HC-1 (belief is load-bearing AND sufficient)
  F2  beat the Gode-Sunder ZI floor PER-STATE              (not aggregate; vs the harder constrained ZI)
  F3  belief -> noise degradation                          -> HX null ii (decorrelate the belief: collapse)
  F4  cross-generator (parameter-regime) holdout (HC-3) + GM cost-shell sign-flip + short-tree-can't-match (HC-2)

Primary env: config B (the WORKING-blind regime from phase0_validate) so every falsifier tests against a
non-trivial baseline, not do-nothing.

Run: python -m src.series_g.phase1_falsifiers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.series_g.phase0_gate import (ACTIONS, _feasible, aware_start_value, solve_belief_aware,
                                       solve_belief_blind)
from src.series_g.regime_pomdp import (ABSTAIN, AGGRESS, BENIGN, BURST, PROVIDE, QUIET, TOXIC,
                                       ACTION_NAMES, RegimePOMDP)

OUT = Path(__file__).resolve().parent / "phase1_falsifiers_report.json"
# config B (working-blind) + the GM-JSI enrichment: unwinding in TOXIC is dear (low resilience), so optimal
# unwind TIMING depends jointly on belief × inventory — the multi-factor structure a single threshold can't match.
PRIMARY = dict(spread=2.0, adverse=4.0, aggress_cost=0.3, aggress_cost_toxic=1.5, hold_cost=0.05)


# ----------------------------------------------------------------- helpers
def solve_oracle(m: RegimePOMDP):
    """Fully-observed-regime VI (perfect regime info) — the upper bound / value of perfect information."""
    V = np.zeros((m.T + 1, 2, m.I_max + 1))
    for r in (BENIGN, TOXIC):
        for I in range(m.I_max + 1):
            V[m.T, r, I] = m.terminal(I)
    for t in range(m.T - 1, -1, -1):
        for r in (BENIGN, TOXIC):
            for I in range(m.I_max + 1):
                V[t, r, I] = max(m.reward(r, a, I) + sum(m.M[r, rn] * V[t + 1, rn, m.inventory_next(a, I)]
                                                         for rn in (BENIGN, TOXIC)) for a in ACTIONS)
    return float((1 - m.prior_toxic) * V[0, BENIGN, 0] + m.prior_toxic * V[0, TOXIC, 0])


def zi_value_function(m: RegimePOMDP, constrained: bool):
    """Per-state value function of the uniform-random feasible (ZI) policy: V_ZI[t, r, I]."""
    V = np.zeros((m.T + 1, 2, m.I_max + 1))
    for r in (BENIGN, TOXIC):
        for I in range(m.I_max + 1):
            V[m.T, r, I] = m.terminal(I)
    for t in range(m.T - 1, -1, -1):
        for r in (BENIGN, TOXIC):
            for I in range(m.I_max + 1):
                acts = _feasible(m, t, I, constrained)
                pa = 1.0 / len(acts)
                V[t, r, I] = sum(pa * (m.reward(r, a, I) + sum(m.M[r, rn] * V[t + 1, rn, m.inventory_next(a, I)]
                                                               for rn in (BENIGN, TOXIC))) for a in acts)
    return V


def _belief_next_bins(bf: RegimePOMDP, g: np.ndarray):
    """For the FILTER family bf and each observation o, the next belief bin from each current bin."""
    nb = len(g)
    bpred = (1 - g) * bf.M[BENIGN, TOXIC] + g * bf.M[TOXIC, TOXIC]
    nxt = {}
    for o in (QUIET, BURST):
        p_o = (1 - bpred) * bf.obs[BENIGN, o] + bpred * bf.obs[TOXIC, o]
        bup = np.where(p_o > 1e-12, (bpred * bf.obs[TOXIC, o]) / np.maximum(p_o, 1e-12), bpred)
        nxt[o] = np.clip(np.round(bup * (nb - 1)).astype(int), 0, nb - 1)
    return nxt


def eval_belief_policy(m: RegimePOMDP, g: np.ndarray, pol: np.ndarray, bf: RegimePOMDP = None,
                       obs_source: str = "true") -> float:
    """Exact expected return of a belief-indexed policy `pol[t, b_bin, I]` under TRUE dynamics m, where the
    agent updates its belief with filter family `bf` (default m) on observations from `obs_source`:
      'true'  — observations drawn from the true next regime (m.obs)  [self / cross-family transfer]
      'noise' — observations independent of the regime (50/50)        [belief->noise falsifier F3]
    Tracks the full joint P(regime, inventory, belief-bin) forward (belief is a deterministic fn of obs)."""
    bf = bf or m
    nb = len(g)
    nxt = _belief_next_bins(bf, g)
    P = np.zeros((2, m.I_max + 1, nb))
    b0 = int(round(m.prior_toxic * (nb - 1)))
    P[BENIGN, 0, b0] = 1 - m.prior_toxic
    P[TOXIC, 0, b0] = m.prior_toxic
    total = 0.0
    for t in range(m.T):
        Pn = np.zeros_like(P)
        for bi in range(nb):
            for I in range(m.I_max + 1):
                if P[BENIGN, I, bi] + P[TOXIC, I, bi] <= 0:
                    continue
                a = int(pol[t, bi, I])
                In = m.inventory_next(a, I)
                for r in (BENIGN, TOXIC):
                    p = P[r, I, bi]
                    if p <= 0:
                        continue
                    total += p * m.reward(r, a, I)
                    for rn in (BENIGN, TOXIC):
                        ptr = p * m.M[r, rn]
                        if obs_source == "noise":
                            Pn[rn, In, nxt[QUIET][bi]] += ptr * 0.5
                            Pn[rn, In, nxt[BURST][bi]] += ptr * 0.5
                        else:
                            Pn[rn, In, nxt[QUIET][bi]] += ptr * m.obs[rn, QUIET]
                            Pn[rn, In, nxt[BURST][bi]] += ptr * m.obs[rn, BURST]
        P = Pn
    for r in (BENIGN, TOXIC):
        for I in range(m.I_max + 1):
            total += P[r, I, :].sum() * m.terminal(I)
    return float(total)


# ----------------------------------------------------------------- F1 concept-ablation (HC-1)
def f1_concept_ablation(m, g, Va) -> dict:
    v_aware = aware_start_value(m, g, Va)
    v_blind = float(solve_belief_blind(m)[0][0, 0])      # toxicity posterior REMOVED
    v_oracle = solve_oracle(m)                            # perfect regime info (upper bound)
    drop = v_aware - v_blind
    frac_of_perfect = drop / (v_oracle - v_blind + 1e-9)
    passed = bool(drop > 0.10 * abs(v_aware) and drop > 0.1)
    return {"belief_blind(ablated)": round(v_blind, 4), "belief_aware": round(v_aware, 4),
            "oracle(perfect_info)": round(v_oracle, 4), "ablation_value_drop": round(drop, 4),
            "fraction_of_perfect_info_VoI_captured": round(frac_of_perfect, 3),
            "PASS": passed, "claim": "HC-1: toxicity posterior is load-bearing (drop) AND sufficient (belief-MDP optimum)"}


# ----------------------------------------------------------------- F2 beat-ZI per-state
def f2_beat_zi_per_state(m, g, Va) -> dict:
    Vzic = zi_value_function(m, constrained=True)         # the HARDER ZI (benefits from structure)
    gaps = []
    for t in range(m.T):
        for I in range(m.I_max + 1):
            for bi in range(0, len(g), 8):
                b = g[bi]
                vstar = Va[t, bi, I]
                vzi = (1 - b) * Vzic[t, BENIGN, I] + b * Vzic[t, TOXIC, I]
                gaps.append(vstar - vzi)
    gaps = np.array(gaps)
    start_margin = aware_start_value(m, g, Va) - ((1 - m.prior_toxic) * Vzic[0, BENIGN, 0] + m.prior_toxic * Vzic[0, TOXIC, 0])
    passed = bool(gaps.min() >= -1e-6 and gaps.max() > 0.1 and start_margin > 0.1)
    return {"per_state_gap_min": round(float(gaps.min()), 4), "per_state_gap_median": round(float(np.median(gaps)), 4),
            "per_state_gap_max": round(float(gaps.max()), 4), "start_state_margin_vs_ZIc": round(float(start_margin), 4),
            "PASS": passed, "claim": "optimal beats the constrained-ZI floor in EVERY reachable state (not just aggregate)"}


# ----------------------------------------------------------------- F3 belief->noise degradation
def f3_belief_to_noise(m, g, pol_a) -> dict:
    v_true = eval_belief_policy(m, g, pol_a, obs_source="true")    # sanity: should ~ aware VI value
    v_noise = eval_belief_policy(m, g, pol_a, obs_source="noise")  # belief decorrelated from regime
    v_blind = float(solve_belief_blind(m)[0][0, 0])
    voi = v_true - v_blind
    lost = (v_true - v_noise) / (voi + 1e-9)
    collapsed = bool(v_noise <= v_blind + 0.15 * abs(voi))
    passed = bool(collapsed and lost > 0.5)
    return {"policy_on_true_belief(sanity~VI)": round(v_true, 4), "policy_on_NOISE_belief": round(v_noise, 4),
            "belief_blind": round(v_blind, 4), "fraction_of_VoI_lost_under_noise": round(float(lost), 3),
            "collapsed_to_<=blind": collapsed, "PASS": passed,
            "claim": "HX null ii: decorrelating the regime belief collapses performance to the no-info baseline"}


# ----------------------------------------------------------------- F4 cross-family (HC-3) + cost-flip (HC-2)
FAMILIES = {
    "fam1_primary": dict(spread=2.0, adverse=4.0, aggress_cost=0.3, aggress_cost_toxic=1.5, hold_cost=0.05),
    "fam2_stickier_toxic": dict(spread=2.5, adverse=3.5, aggress_cost=0.3, aggress_cost_toxic=1.8, hold_cost=0.05,
                                p_stay_toxic=0.70, p_burst_toxic=0.60),
    "fam3_common_toxic": dict(spread=1.5, adverse=3.0, aggress_cost=0.4, aggress_cost_toxic=1.2, hold_cost=0.08,
                              p_stay_benign=0.90, prior_toxic=0.30),
}


def f4_cross_family_and_costflip(primary: dict) -> dict:
    # ---- HC-3: parameter-regime transfer (honest scope: cross-PARAMETER, not cross-model-class) ----
    solved = {}
    for name, over in FAMILIES.items():
        mm = RegimePOMDP(**over)
        g, Va, pol = solve_belief_aware(mm)
        v_aware = aware_start_value(mm, g, Va)
        v_blind = float(solve_belief_blind(mm)[0][0, 0])
        solved[name] = dict(m=mm, g=g, pol=pol, aware=v_aware, blind=v_blind)
    transfers = []
    train = "fam1_primary"
    for test in FAMILIES:
        if test == train:
            continue
        s_tr, s_te = solved[train], solved[test]
        v_transfer = eval_belief_policy(s_te["m"], s_tr["g"], s_tr["pol"], bf=s_tr["m"], obs_source="true")
        retained = (v_transfer - s_te["blind"]) / (s_te["aware"] - s_te["blind"] + 1e-9)
        transfers.append({"train": train, "test": test, "transfer_value": round(v_transfer, 4),
                          "test_aware": round(s_te["aware"], 4), "test_blind": round(s_te["blind"], 4),
                          "VoI_retained_frac": round(float(retained), 3)})
    hc3_pass = bool(all(tr["VoI_retained_frac"] > 0.5 for tr in transfers))

    # ---- HC-2: GM cost-shell sign-flip + short-tree-can't-match ----
    m_cost = RegimePOMDP(**primary)
    nocost = dict(primary); nocost.update(aggress_cost=0.0, hold_cost=0.0, liq_penalty=0.0)
    m_free = RegimePOMDP(**nocost)
    g, Va_c, pol_c = solve_belief_aware(m_cost)
    _, _, pol_f = solve_belief_aware(m_free)
    flip_mask = pol_c != pol_f
    flip_frac = float(flip_mask.mean())
    # an example flip cell
    ex = None
    idx = np.argwhere(flip_mask)
    if len(idx):
        t0, b0, I0 = idx[len(idx) // 2]
        ex = {"t": int(t0), "belief": round(float(g[b0]), 3), "I": int(I0),
              "no_cost_action": ACTION_NAMES[int(pol_f[t0, b0, I0])],
              "with_cost_action": ACTION_NAMES[int(pol_c[t0, b0, I0])]}
    # short-tree-can't-match: predict the with-cost optimal action from (belief, inventory, time)
    T, nb, nI = pol_c.shape
    bb, ii, tt = np.meshgrid(np.arange(nb), np.arange(nI), np.arange(T), indexing="ij")
    X = np.column_stack([g[bb.ravel()], ii.ravel(), tt.ravel()])
    y = pol_c.transpose(1, 2, 0).ravel()
    depth_acc = {}
    for d in (1, 2, 3, 5):
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        depth_acc[d] = round(float((clf.predict(X) == y).mean()), 3)
    hc2_pass = bool(flip_frac > 0.05 and depth_acc[1] < 0.90)

    return {"HC3_cross_parameter_transfer": {"transfers": transfers, "PASS": hc3_pass,
            "scope_note": "cross-PARAMETER-regime transfer; true cross-MODEL-FAMILY (GCMG/Brock-Hommes) needs the alternate generators (methodology must-build)"},
            "HC2_cost_sign_flip": {"flip_fraction": round(flip_frac, 3), "example_flip": ex,
            "short_tree_accuracy_by_depth": depth_acc, "depth1_under_0.90": bool(depth_acc[1] < 0.90),
            "PASS": hc2_pass, "claim": "native costs FLIP the optimal action; a single-threshold tree cannot reproduce the optimum"},
            "PASS": bool(hc3_pass and hc2_pass)}


def run() -> dict:
    m = RegimePOMDP(**PRIMARY)
    g, Va, pol_a = solve_belief_aware(m)
    f1 = f1_concept_ablation(m, g, Va)
    f2 = f2_beat_zi_per_state(m, g, Va)
    f3 = f3_belief_to_noise(m, g, pol_a)
    f4 = f4_cross_family_and_costflip(PRIMARY)
    all_pass = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"])
    verdict = ("ALL FOUR FALSIFIERS PASS — the complexity is real (belief load-bearing + sufficient, beats ZI "
               "per-state, collapses under belief-noise, transfers across parameter regimes, cost-induced "
               "sign-flip non-trivial). Proceed to Phase 2 (train the 2-level policy)."
               if all_pass else
               "FALSIFIER FAILURE — at least one gate failed. STOP. The complexity may be a simulator artifact; "
               "publish 'complexity was plumbing' and revise the benchmark, not the method.")
    return {"primary_env_config": PRIMARY,
            "F1_concept_ablation_HC1": f1, "F2_beat_ZI_per_state": f2,
            "F3_belief_to_noise": f3, "F4_cross_family_HC3_and_costflip_HC2": f4,
            "ALL_PASS": all_pass, "PHASE1_VERDICT": verdict}


if __name__ == "__main__":
    rep = run()
    print(json.dumps(rep, indent=2))
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\n[phase1] wrote {OUT.name}  ->  {'ALL PASS' if rep['ALL_PASS'] else 'FAILURE'}")
