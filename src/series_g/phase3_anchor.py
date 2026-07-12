"""Series-G Phase 3 — HX: the second frontier anchor (two-axis read on the optimal policy).

Measure the env's optimal belief-aware policy on the SAME two axes as the R6c/csi500 anchors, so it lands on
ONE frontier (SERIES_G_METHODOLOGY_AND_PLAN.md §5):
  x  = BEHAVIORAL COMPLEXITY = L0 bits/action (h_mu / E / C_mu + the phase-shuffle structure test), computed
       by the SAME interpretability/cross_policy_crystal.py::behavioral_complexity_dynamic used for R6c, on the
       optimal-policy rollout action stream (native 3-symbol alphabet = R6c's cash A=3 → directly comparable).
  y  = SIMULATABILITY = a parsimony-bounded surrogate's fidelity. We report it TWO ways (the H1 contrast):
       - STATE→action  : depth-≤4 tree from (belief, inventory, time)  [is the policy reactive to its state?]
       - AUTOREGRESSIVE: order-k Markov from the action's own past      [is it self-predictable, like R6c?]

Reference points (verified, from this repo):
  R6c (frozen Dow-29): x h_mu ≈ 0.29–0.49 (3-symbol stance); STATE/obs→action 0.20 (NOT obs-reactive),
                       AUTOREGRESSIVE 0.95 (a self-predictable persister — L0/H1 reports).
  poker (design-judgment): ≈ (0.78, 0.62) normalized — qualitative; NOT yet in bits/action (reconciliation open).

HONESTY: the policy is the analytic optimum (the env's intrinsic frontier behavior a trained CHRL approaches),
NOT a PPO agent; Faithfulness/Controllability/Stability need per-policy machinery (N/A here). The x-axis is
directly comparable to R6c (same ruler, 3-symbol). The y recipes differ from R6c's latent→action CrystalScore,
so y is reported as the state/autoregressive surrogate fidelity (the H1 recipe both can produce), with that flag.

Run: python -m src.series_g.phase3_anchor
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "interpretability"))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

ROLLOUT = HERE / "phase2_optimal_rollout.csv"
OUT = HERE / "phase3_anchor_report.json"
R6C_REF = {"x_h_mu_3sym": [0.288, 0.487], "y_state_obs_to_action": 0.20, "y_autoregressive": 0.95}
POKER_REF = {"x_norm": 0.78, "y": 0.62, "note": "design-judgment, normalized; not yet bits/action"}


def _markov_simulatability(actions: np.ndarray, ep: np.ndarray, order: int) -> float:
    """Order-k Markov on the action's OWN past (within episodes), balanced accuracy — the autoregressive y."""
    from collections import defaultdict
    K = int(actions.max()) + 1
    table = defaultdict(lambda: np.zeros(K))
    for i in range(order, len(actions)):
        if ep[i] != ep[i - order]:
            continue
        table[tuple(actions[i - order:i])][actions[i]] += 1
    pred, true = [], []
    for i in range(order, len(actions)):
        if ep[i] != ep[i - order]:
            continue
        h = tuple(actions[i - order:i])
        pred.append(int(np.argmax(table[h])) if table[h].sum() > 0 else int(np.bincount(actions).argmax()))
        true.append(actions[i])
    return float(balanced_accuracy_score(true, pred))


def main() -> int:
    """Measure the optimal-policy rollout on both frontier axes (L0 bits/action complexity + surrogate simulatability), place the Series-G anchor vs R6c, and write the report."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    d = pd.read_csv(ROLLOUT)
    actions = d["action"].to_numpy(int)
    ep = d["episode"].to_numpy(int)

    # ---- x: L0 bits/action on the action stream (native 3-symbol, same machinery as R6c) ----
    x = behavioral_complexity_dynamic(actions, kind="discrete", dts=(1, 2), n_null=500, n_boot=500, seed=0)
    h_mu_range = x["h_mu_range"]

    # ---- y(state→action): depth-≤4 tree from (belief, inventory, time); rollout rows = visitation samples ----
    Xs = d[["belief_toxic", "inventory", "t"]].to_numpy(float)
    ystate = {}
    for label, kw in {"depth4": dict(max_depth=4), "K9": dict(max_leaf_nodes=9)}.items():
        clf = DecisionTreeClassifier(random_state=0, **kw).fit(Xs, actions)
        pred = clf.predict(Xs)
        ystate[label] = {"accuracy": round(float((pred == actions).mean()), 4),
                         "balanced_accuracy": round(float(balanced_accuracy_score(actions, pred)), 4),
                         "n_leaves": int(clf.get_n_leaves())}
    # completeness Pareto: fidelity vs leaf budget
    pareto = {}
    for k in (2, 3, 4, 6, 9, 12, 16):
        clf = DecisionTreeClassifier(max_leaf_nodes=k, random_state=0).fit(Xs, actions)
        pareto[k] = round(float((clf.predict(Xs) == actions).mean()), 4)

    # ---- y(autoregressive): order-k Markov from the action's own past ----
    y_auto = max(_markov_simulatability(actions, ep, o) for o in (1, 2))

    y_state_bal = ystate["depth4"]["balanced_accuracy"]
    # frontier placement (raw bits/action x; simulatability y in [0,1])
    more_complex_than_r6c = bool(min(h_mu_range) > max(R6C_REF["x_h_mu_3sym"]))
    interpretable = bool(y_state_bal >= 0.5)

    report = {
        "policy": "Series-G optimal belief-aware (Phase-1 gated env)", "n_steps": int(len(d)),
        "x_behavioral_complexity_bits_per_action": {
            "h_mu_range_3symbol": h_mu_range, "E_range": x["E_range"], "C_mu_Lstar_range": x["C_mu_Lstar_range"],
            "structure_vs_phase_shuffle": x["structure_present_configs"], "structure_present": x["structure_present"]},
        "y_simulatability": {
            "state_to_action_depth4": ystate["depth4"], "state_to_action_K9": ystate["K9"],
            "autoregressive_markov": round(y_auto, 4), "completeness_pareto_acc_by_leaves": pareto},
        "reference_points": {"R6c_frozen_Dow29": R6C_REF, "poker_design_judgment": POKER_REF},
        "frontier_placement": {
            "x_series_g_vs_r6c": f"h_mu {h_mu_range} (3-symbol) vs R6c {R6C_REF['x_h_mu_3sym']} — "
                                 + ("HIGHER (Series-G is more behaviorally complex)" if more_complex_than_r6c
                                    else "comparable/lower"),
            "mechanism_contrast": (f"Series-G is STATE-reactive (state→action {y_state_bal} >> autoregressive "
                                   f"{round(y_auto,3)}); R6c is the OPPOSITE — autoregressive {R6C_REF['y_autoregressive']} "
                                   f">> obs-reactive {R6C_REF['y_state_obs_to_action']}. Both interpretable, different mechanisms."),
            "high_complexity_AND_interpretable": bool(more_complex_than_r6c and interpretable)},
        "verdict": "",
        "honesty": ("Analytic optimum (not a PPO agent); x directly comparable to R6c (same L0 ruler, 3-symbol); "
                    "y is the state/autoregressive surrogate fidelity (H1 recipe), NOT R6c's latent→action CrystalScore "
                    "— flagged. H3 (hierarchical interpretability) did NOT hold here (Phase 2): this is a "
                    "flat-interpretable anchor. Poker's (0.78,0.62) is normalized design-judgment, not bits/action."),
    }
    report["verdict"] = (
        "SECOND ANCHOR PLACED. Series-G's optimal policy is more behaviorally complex than R6c on the identical "
        f"bits/action ruler ({h_mu_range} vs {R6C_REF['x_h_mu_3sym']}) AND highly simulatable from its belief-state "
        f"(state→action {y_state_bal}) — a high(er)-complexity-yet-interpretable point, reached by a STATE-reactive "
        "(not autoregressive) mechanism. It moves RIGHT along the frontier at high y vs R6c's low corner."
        if report["frontier_placement"]["high_complexity_AND_interpretable"] else
        "Series-G does NOT clearly exceed R6c's complexity on this minimal env — the anchor is comparable, not "
        "higher; the high-complexity corner needs the high-dim Interface-A env.")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["x_behavioral_complexity_bits_per_action"], indent=2))
    print(json.dumps(report["y_simulatability"], indent=2))
    print("\nFRONTIER:", json.dumps(report["frontier_placement"], indent=2))
    print("\nVERDICT:", report["verdict"])
    print(f"\n[phase3] wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
