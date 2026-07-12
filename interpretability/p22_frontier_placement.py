"""Place the GENUINE two-agent P22 policy on the bits/action frontier — the third real-policy mechanism.

P22's L0 complexity was measured in l0_csi500_anchors (cash h_mu high but structure 0/6 ⇒ noise; book-mode
structured 2/2). This computes its frontier coordinate on the SAME recipe as R6c/Series-G:
  x = L0 bits/action (3-symbol cash stance, for direct comparability) + the phase-shuffle structure test.
  y = best compact simulatability = max( REACTIVE obs->action , AUTOREGRESSIVE own-past ), over the cash and
      book-mode streams — the H1 contrast, so we see HOW (if at all) P22 is simulatable.
Observations are recovered without a re-roll by aligning the frozen stream to the panel's macro/regime
features by date order (the P22 PM's actual inputs: VIX, 10Y_Yield, Market_Regime, Regime_*, SP500_Trend,
turbulence). Run: python interpretability/p22_frontier_placement.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

MACRO = ["VIX", "10Y_Yield", "Market_Regime", "Regime_0_Prob", "Regime_1_Prob", "SP500_Trend", "turbulence"]
RUNS = [("P22 deadline", "_streams/p22_deadline_daily.csv", "data/adapters/_csi500_wide/csi300_model_ready.csv"),
        ("P22 PIT", "_streams/p22_pit_daily.csv", "data/adapters/_csi500_pit/csi300_model_ready.csv")]


def eq_bins(x, k=3):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)))
    return np.clip(np.digitize(x, qs[1:-1]), 0, max(0, qs.size - 2)).astype(int) if qs.size > 1 else np.zeros(len(x), int)


def reactive_sim(X, y):
    """obs->action depth-4 tree, temporal 60/40 holdout, balanced accuracy (state-reactive simulatability)."""
    n = len(y); cut = int(n * 0.6)
    if len(np.unique(y[cut:])) < 2:
        return float("nan")
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X[:cut], y[:cut])
    return float(balanced_accuracy_score(y[cut:], clf.predict(X[cut:])))


def autoreg_sim(y, order=1):
    """order-k Markov on the action's own past, temporal 60/40 holdout, balanced accuracy."""
    from collections import defaultdict
    n = len(y); cut = int(n * 0.6); K = int(y.max()) + 1
    tbl = defaultdict(lambda: np.zeros(K))
    for i in range(order, cut):
        tbl[tuple(y[i - order:i])][y[i]] += 1
    if len(np.unique(y[cut:])) < 2:
        return float("nan")
    pred = [int(np.argmax(tbl[tuple(y[i - order:i])])) if tbl[tuple(y[i - order:i])].sum() > 0
            else int(np.bincount(y[:cut]).argmax()) for i in range(cut, n)]
    return float(balanced_accuracy_score(y[cut:], pred))


def analyze(stream_path, panel_path):
    s = pd.read_csv(HERE / stream_path)
    cash = s["cash"].to_numpy(float)
    wcols = [c for c in s.columns if c.startswith("w_")]
    W = np.clip(s[wcols].to_numpy(float), 0, None)
    within = np.divide(W, W.sum(1, keepdims=True), out=np.full_like(W, 1.0 / W.shape[1]), where=W.sum(1, keepdims=True) > 1e-9)
    cash_sym = eq_bins(cash, 3)
    mode_sym = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(within)
    # obs (macro/regime) aligned by date order
    d = pd.read_csv(ROOT / panel_path, usecols=lambda c: c in (["date"] + MACRO))
    d["date"] = pd.to_datetime(d["date"])
    val = d[(d["date"] >= "2022-01-04") & (d["date"] <= "2023-02-28")].groupby("date").first().reset_index().sort_values("date")
    m = min(len(s), len(val))
    Xobs = StandardScaler().fit_transform(val[MACRO].to_numpy(float)[:m])
    out = {}
    for name, sym in [("cash_stance_3bin", cash_sym[:m]), ("book_mode_K6", mode_sym[:m])]:
        l0 = behavioral_complexity_dynamic(sym, kind="discrete", dts=(1, 2), n_null=400, n_boot=400, seed=0)
        react = reactive_sim(Xobs, sym)
        auto = max(autoreg_sim(sym, 1), autoreg_sim(sym, 2))
        out[name] = {"h_mu_range": l0["h_mu_range"], "structure": l0["structure_present_configs"],
                     "reactive_obs_to_action": round(react, 3) if np.isfinite(react) else None,
                     "autoregressive_own_past": round(auto, 3) if np.isfinite(auto) else None,
                     "best_simulatability": round(float(np.nanmax([react, auto])), 3)}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rows = {}
    for name, stream, panel in RUNS:
        rows[name] = analyze(stream, panel)
    report = {
        "policy": "GENUINE two-agent P22 (Beta-budget PM + latent-action Trader), csi500 frozen 2022-2023",
        "recipe": "x = L0 bits/action (3-symbol cash stance); y = best compact simulatability = max(reactive obs->action, autoregressive own-past)",
        "results": rows,
        "reference": {"R6c": {"x_cash_3sym": [0.288, 0.487], "y_best_sim": 0.95, "mechanism": "autoregressive persister"},
                      "Series-G_optimal": {"x": [0.853, 1.39], "y_best_sim": 0.93, "mechanism": "state-reactive controller"}},
    }
    # frontier placement: x = cash 3-bin h_mu; y = best simulatability across both streams
    for name in rows:
        r = rows[name]
        x = r["cash_stance_3bin"]["h_mu_range"]
        y = max(r["cash_stance_3bin"]["best_simulatability"], r["book_mode_K6"]["best_simulatability"])
        cash_struct = r["cash_stance_3bin"]["structure"]; mode_struct = r["book_mode_K6"]["structure"]
        mech = (f"cash-stance {('STRUCTURED' if cash_struct.split('/')[0]!='0' else 'NOISE')} ({cash_struct}); "
                f"book-mode {('STRUCTURED' if mode_struct.split('/')[0]!='0' else 'NOISE')} ({mode_struct}); "
                f"y from {'reactive' if (r['cash_stance_3bin']['reactive_obs_to_action'] or 0) >= (r['cash_stance_3bin']['autoregressive_own_past'] or 0) else 'autoregressive'}")
        report.setdefault("frontier_points", {})[name] = {"x_bits_per_action": x, "y_best_simulatability": round(y, 3), "mechanism": mech}
    (HERE / "p22_frontier_placement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))
    print("\nFRONTIER POINTS:", json.dumps(report["frontier_points"], indent=2))
    print(f"\n[p22] wrote p22_frontier_placement_report.json")


if __name__ == "__main__":
    main()
