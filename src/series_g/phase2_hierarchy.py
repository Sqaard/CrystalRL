"""Series-G Phase 2 — H3 (hierarchical factoring) + the rollout log for Phase 3.

H3 (SERIES_G_METHODOLOGY_AND_PLAN.md §5.2): factor whole-policy complexity into NAMED sub-controllers so a
faithful surrogate stays SHORT despite higher whole-policy complexity. Null to beat: a FLAT single-level
distillation matches the fidelity at equal-or-LOWER concept budget (⇒ hierarchy bought nothing).

We test it on the env's OPTIMAL belief-aware policy (Phase 1 gated env) — the cleanest ground-truth target a
trained 2-level CHRL approximates (no undertraining confound; a PPO CHRL would only approach it). The
2-level structure mirrors the project's PM→Trader split:
  MANAGER (level 1):  (belief, time)        → stance ∈ {PROVIDE, ABSTAIN, AGGRESS}   [the PM's intended mode]
  WORKER  (level 2):  (stance, inventory, time) → action                            [the Trader's execution]
vs a FLAT surrogate: (belief, inventory, time) → action.

Both are distilled with `DecisionTreeClassifier` at increasing leaf budgets; fidelity is
STATE-VISITATION-WEIGHTED (focus where the optimal policy operates, not unreachable cells). H3 passes iff the
hierarchical fidelity-vs-budget curve DOMINATES the flat one in the low-budget regime (more faithful per rule).

Also writes the optimal-policy ROLLOUT action log (`phase2_optimal_rollout.csv`) for Phase 3's L0 ruler.

Run: python -m src.series_g.phase2_hierarchy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.series_g.phase0_gate import solve_belief_aware
from src.series_g.phase1_falsifiers import PRIMARY
from src.series_g.regime_pomdp import (ACTION_NAMES, BENIGN, BURST, QUIET, TOXIC, RegimePOMDP)

HERE = Path(__file__).resolve().parent
OUT = HERE / "phase2_hierarchy_report.json"
ROLLOUT = HERE / "phase2_optimal_rollout.csv"
BUDGETS = [3, 4, 5, 6, 8, 10, 12, 16]


def visitation_weights(m: RegimePOMDP, g: np.ndarray, pol: np.ndarray) -> np.ndarray:
    """Forward-propagate P(belief_bin, inventory, t) under the optimal policy; return W[t, b_bin, I]."""
    nb = len(g)
    bpred = (1 - g) * m.M[BENIGN, TOXIC] + g * m.M[TOXIC, TOXIC]
    nxt = {}
    for o in (QUIET, BURST):
        p_o = (1 - bpred) * m.obs[BENIGN, o] + bpred * m.obs[TOXIC, o]
        bup = np.where(p_o > 1e-12, (bpred * m.obs[TOXIC, o]) / np.maximum(p_o, 1e-12), bpred)
        nxt[o] = np.clip(np.round(bup * (nb - 1)).astype(int), 0, nb - 1)
    W = np.zeros((m.T, nb, m.I_max + 1))
    P = np.zeros((2, m.I_max + 1, nb))
    b0 = int(round(m.prior_toxic * (nb - 1)))
    P[BENIGN, 0, b0] = 1 - m.prior_toxic
    P[TOXIC, 0, b0] = m.prior_toxic
    for t in range(m.T):
        Pn = np.zeros_like(P)
        for bi in range(nb):
            for I in range(m.I_max + 1):
                mass = P[BENIGN, I, bi] + P[TOXIC, I, bi]
                if mass <= 0:
                    continue
                W[t, bi, I] += mass
                a = int(pol[t, bi, I])
                In = m.inventory_next(a, I)
                for r in (BENIGN, TOXIC):
                    p = P[r, I, bi]
                    if p <= 0:
                        continue
                    for rn in (BENIGN, TOXIC):
                        ptr = p * m.M[r, rn]
                        Pn[rn, In, nxt[QUIET][bi]] += ptr * m.obs[rn, QUIET]
                        Pn[rn, In, nxt[BURST][bi]] += ptr * m.obs[rn, BURST]
        P = Pn
    return W


def _cells(m, g, pol, W):
    """Flatten the policy table into (features, action, weight) over all (t, belief, I) cells."""
    nb = len(g)
    rows = []
    for t in range(m.T):
        for bi in range(nb):
            for I in range(m.I_max + 1):
                rows.append((g[bi], I, t, int(pol[t, bi, I]), float(W[t, bi, I])))
    arr = np.array(rows, dtype=float)
    return arr[:, :3], arr[:, 3].astype(int), arr[:, 4]   # X=(belief,I,t), y=action, w=visitation


def _fid(clf, X, y, w):
    pred = clf.predict(X)
    return float((w * (pred == y)).sum() / (w.sum() + 1e-12))


def fit_flat(X, y, w, budget):
    """Fit a flat (belief, inventory, time)->action decision tree at the given leaf budget; return its visitation-weighted fidelity and the classifier."""
    clf = DecisionTreeClassifier(max_leaf_nodes=max(2, budget), random_state=0).fit(X, y, sample_weight=w + 1e-9)
    return _fid(clf, X, y, w), clf


def fit_hierarchical(m, g, pol, W, X, y, w, budget):
    """Manager (belief,t)->stance (=optimal action at I=0); Worker (stance,I,t)->action. Split the leaf budget
    between the two levels; report the best split for this total budget."""
    nb = len(g)
    # manager target = the optimal stance at reference inventory I=0
    stance_tbl = pol[:, :, 0]                                  # (T, nb)
    Xm = X[:, [0, 2]]                                          # (belief, t)
    ym = np.array([int(stance_tbl[int(t), int(round(b * (nb - 1)))]) for b, I, t in X])
    wm = w
    best = None
    for m_leaves in range(2, budget - 1):
        w_leaves = budget - m_leaves
        if w_leaves < 2:
            continue
        mgr = DecisionTreeClassifier(max_leaf_nodes=m_leaves, random_state=0).fit(Xm, ym, sample_weight=wm + 1e-9)
        # pass the manager's full LEAF PARTITION (the "mode", up to m_leaves values) to the worker — not the
        # 3-class stance collapse, which is an unfairly lossy bottleneck that discards belief magnitude.
        mode_pred = mgr.apply(Xm)
        Xw = np.column_stack([mode_pred, X[:, 1], X[:, 2]])    # (mode, I, t)
        wkr = DecisionTreeClassifier(max_leaf_nodes=w_leaves, random_state=0).fit(Xw, y, sample_weight=w + 1e-9)
        fid = _fid(wkr, Xw, y, w)
        if best is None or fid > best[0]:
            best = (fid, m_leaves, w_leaves, mgr, wkr)
    return best


def make_rollout(m: RegimePOMDP, g: np.ndarray, pol: np.ndarray, n_episodes: int = 80, seed: int = 0):
    """Stochastic rollout of the optimal policy → the action stream for Phase 3 L0."""
    rng = np.random.default_rng(seed)
    nb = len(g)
    recs = []
    step = 0
    for ep in range(n_episodes):
        r = TOXIC if rng.random() < m.prior_toxic else BENIGN
        I, b = 0, m.prior_toxic
        for t in range(m.T):
            bi = int(round(b * (nb - 1)))
            a = int(pol[t, bi, I])
            recs.append({"t_global": step, "episode": ep, "t": t, "action": a,
                         "belief_toxic": round(float(b), 4), "inventory": I, "regime": r,
                         "reward": round(m.reward(r, a, I), 4)})
            In = m.inventory_next(a, I)
            r = TOXIC if rng.random() < m.M[r, TOXIC] else BENIGN
            o = BURST if rng.random() < m.obs[r, BURST] else QUIET
            b = m.update(m.predict(b), o)
            I = In
            step += 1
    return pd.DataFrame(recs)


def run() -> dict:
    """Solve the optimal policy, run the H3 flat-vs-hierarchical compactness comparison across leaf budgets, emit the per-level FSMs and the Phase-3 rollout log, and return the report."""
    m = RegimePOMDP(**PRIMARY)
    g, Va, pol = solve_belief_aware(m)
    W = visitation_weights(m, g, pol)
    X, y, w = _cells(m, g, pol, W)

    flat_curve, hier_curve = {}, {}
    best_hier = None
    for B in BUDGETS:
        f_fid, _ = fit_flat(X, y, w, B)
        flat_curve[B] = round(f_fid, 4)
        bh = fit_hierarchical(m, g, pol, W, X, y, w, B)
        if bh:
            hier_curve[B] = round(bh[0], 4)
            if best_hier is None or bh[0] > best_hier[0]:
                best_hier = (bh[0], B, bh[1], bh[2], bh[3], bh[4])

    # H3 verdict: does hierarchy reach a fidelity the flat surrogate needs MORE budget to match?
    # For each budget, hier_fid[B] vs the smallest flat budget achieving >= hier_fid[B].
    dominance = []
    for B in BUDGETS:
        hf = hier_curve.get(B)
        if hf is None:
            continue
        flat_budget_needed = next((b2 for b2 in BUDGETS if flat_curve[b2] >= hf - 1e-9), None)
        dominance.append({"budget": B, "hier_fid": hf, "flat_fid_same_budget": flat_curve[B],
                          "flat_budget_to_match": flat_budget_needed,
                          "hier_more_compact": bool(flat_budget_needed is not None and flat_budget_needed > B)})
    # PASS if at >=1 low budget the hierarchy is strictly more compact-faithful AND not dominated elsewhere
    hier_wins = sum(d["hier_more_compact"] for d in dominance)
    hier_better_or_equal = all(hier_curve[B] >= flat_curve[B] - 0.02 for B in hier_curve)
    passed = bool(hier_wins >= 1 and hier_better_or_equal)

    # ---- the interpretable <=7-rule-per-level FSMs (best hierarchical config) ----
    fsm = {}
    if best_hier:
        _, B, m_leaves, w_leaves, mgr, wkr = best_hier
        fsm = {"total_budget": B, "manager_leaves": int(m_leaves), "worker_leaves": int(w_leaves),
               "manager_<=7_rules": (m_leaves <= 7), "worker_<=7_rules": (w_leaves <= 7),
               "manager_rules(belief,t->stance)": export_text(
                   mgr, feature_names=["belief", "time"]).replace("\n", " | ")[:600],
               "worker_rules(stance,I,t->action)": export_text(
                   wkr, feature_names=["stance", "inventory", "time"]).replace("\n", " | ")[:800]}

    # ---- rollout log for Phase 3 ----
    df = make_rollout(m, g, pol)
    df.to_csv(ROLLOUT, index=False)
    act_counts = df["action"].value_counts().to_dict()

    return {"env": "Phase-1 gated (config B + toxic-unwind enrichment)",
            "flat_fidelity_by_budget": flat_curve, "hierarchical_fidelity_by_budget": hier_curve,
            "best_hierarchical": {"fidelity": round(best_hier[0], 4), "total_budget": best_hier[1],
                                  "manager_leaves": best_hier[2], "worker_leaves": best_hier[3]} if best_hier else None,
            "compactness_dominance": dominance, "H3_PASS": passed,
            "H3_verdict": ("PASS — the 2-level (manager→worker) factoring reaches a fidelity the flat surrogate "
                           "needs MORE rules to match: hierarchy keeps the surrogate short (named sub-controllers)."
                           if passed else
                           "FAIL — flat distillation matches the hierarchical fidelity at equal-or-lower budget: "
                           "the hierarchy bought nothing here (H3 not supported on this env)."),
            "per_level_FSMs": fsm,
            "rollout_log": {"path": ROLLOUT.name, "n_steps": int(len(df)), "n_episodes": int(df['episode'].nunique()),
                            "action_counts": {ACTION_NAMES[int(k)]: int(v) for k, v in act_counts.items()}}}


if __name__ == "__main__":
    rep = run()
    print(json.dumps({k: v for k, v in rep.items() if k != "per_level_FSMs"}, indent=2))
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\n[phase2] H3: {'PASS' if rep['H3_PASS'] else 'FAIL'}  ->  wrote {OUT.name} + {ROLLOUT.name}")
