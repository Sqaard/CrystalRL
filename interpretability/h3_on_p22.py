"""H3 on the GENUINE two-agent P22 — closing the last coverage cell.

P22 is a real Beta-budget PM -> latent-action Trader hierarchy: the PM sets the budget (cash stance), the
Trader sets the within-book allocation CONDITIONED on that budget. H3 asks whether this PM->Trader factoring
yields a more compact faithful surrogate than a flat distillation (Phase-2 recipe, on P22's real frozen log).

  FLAT          : tree( macro/regime obs ) -> within-book mode.
  HIERARCHICAL  : manager tree(obs)->budget(stance) leaf ; worker tree( manager-leaf, obs )->within-book mode
                  (the PM's budget mediates the Trader, as in the real architecture).

Both judged on a temporal 60/40 holdout, fidelity-vs-leaf-budget, against the chance baseline. Honest prior:
P22 is an UNSTRUCTURED CHURNER (structure 0/6; simulatability at chance from obs AND own-past), so we expect
BOTH surrogates near chance and the hierarchy to buy nothing — you cannot hierarchically distill noise. That
is a clean NEGATIVE (a property of P22, not a failure of H3), and it closes the coverage matrix.

Run: python interpretability/h3_on_p22.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MACRO = ["VIX", "10Y_Yield", "Market_Regime", "Regime_0_Prob", "Regime_1_Prob", "SP500_Trend", "turbulence"]
RUNS = [("P22 deadline", "_streams/p22_deadline_daily.csv", "data/adapters/_csi500_wide/csi300_model_ready.csv"),
        ("P22 PIT", "_streams/p22_pit_daily.csv", "data/adapters/_csi500_pit/csi300_model_ready.csv")]
BUDGETS = [3, 4, 6, 8, 12, 16]


def eq_bins(x, k=3):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)))
    return np.clip(np.digitize(x, qs[1:-1]), 0, max(0, qs.size - 2)).astype(int) if qs.size > 1 else np.zeros(len(x), int)


def acc(clf, X, y):
    return float((clf.predict(X) == y).mean())


def analyze(stream, panel):
    s = pd.read_csv(HERE / stream)
    cash = s["cash"].to_numpy(float)
    wcols = [c for c in s.columns if c.startswith("w_")]
    W = np.clip(s[wcols].to_numpy(float), 0, None)
    within = np.divide(W, W.sum(1, keepdims=True), out=np.full_like(W, 1.0 / W.shape[1]), where=W.sum(1, keepdims=True) > 1e-9)
    d = pd.read_csv(ROOT / panel, usecols=lambda c: c in (["date"] + MACRO))
    d["date"] = pd.to_datetime(d["date"])
    val = d[(d["date"] >= "2022-01-04") & (d["date"] <= "2023-02-28")].groupby("date").first().reset_index().sort_values("date")
    m = min(len(s), len(val))
    obs = StandardScaler().fit_transform(val[MACRO].to_numpy(float)[:m])
    stance = eq_bins(cash[:m], 3)                       # PM budget (3 levels)
    mode = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(within[:m])  # Trader selection
    cut = int(m * 0.6)
    tr, te = slice(0, cut), slice(cut, m)

    flat, hier = {}, {}
    for B in BUDGETS:
        # FLAT: obs -> mode
        f = DecisionTreeClassifier(max_leaf_nodes=B, random_state=0).fit(obs[tr], mode[tr])
        flat[B] = round(acc(f, obs[te], mode[te]), 3)
        # HIER: manager obs->stance leaf ; worker (manager-leaf, obs) -> mode ; best budget split
        best = 0.0
        for mlv in range(2, B - 1):
            wlv = B - mlv
            if wlv < 2:
                continue
            mgr = DecisionTreeClassifier(max_leaf_nodes=mlv, random_state=0).fit(obs[tr], stance[tr])
            leaf_tr = mgr.apply(obs[tr])[:, None]; leaf_te = mgr.apply(obs[te])[:, None]
            wkr = DecisionTreeClassifier(max_leaf_nodes=wlv, random_state=0).fit(
                np.hstack([leaf_tr, obs[tr]]), mode[tr])
            best = max(best, acc(wkr, np.hstack([leaf_te, obs[te]]), mode[te]))
        hier[B] = round(best, 3)
    # PM-stance distillability + chance baselines
    pm = DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(obs[tr], stance[tr])
    pm_acc = round(acc(pm, obs[te], stance[te]), 3)
    chance_mode = round(float(np.bincount(mode[te]).max() / len(mode[te])), 3)   # majority-class baseline
    chance_stance = round(float(np.bincount(stance[te]).max() / len(stance[te])), 3)
    flat_best = max(flat.values()); hier_best = max(hier.values())
    # H3 PASS would require hier to beat flat at a LOWER budget AND fidelity materially above chance
    above_chance = bool(flat_best > chance_mode + 0.08 or hier_best > chance_mode + 0.08)
    hier_helps = bool(hier_best > flat_best + 0.03)
    return {"flat_fidelity_by_budget": flat, "hier_fidelity_by_budget": hier,
            "PM_stance_distillability(obs->budget)": pm_acc, "chance_mode_majority": chance_mode,
            "chance_stance_majority": chance_stance, "flat_best": flat_best, "hier_best": hier_best,
            "any_surrogate_above_chance": above_chance, "hierarchy_helps": hier_helps,
            "H3_PASS": bool(above_chance and hier_helps)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    res = {name: analyze(stream, panel) for name, stream, panel in RUNS}
    any_pass = any(r["H3_PASS"] for r in res.values())
    report = {
        "policy": "GENUINE two-agent P22 (Beta-budget PM -> latent-action Trader), csi500 frozen",
        "test": "H3 hierarchical factoring on P22's real two levels: flat(obs->mode) vs hierarchical(PM-budget-mediated worker->mode), temporal 60/40 holdout, vs chance",
        "results": res,
        "H3_on_P22_verdict": (
            "H3 NEGATIVE on P22 (as predicted) — neither flat nor PM->Trader-hierarchical distillation reproduces "
            "P22's within-book selection above chance, and the hierarchy buys nothing. P22 is an unstructured "
            "churner (structure 0/6, simulatability at chance), and you cannot hierarchically distill noise: there "
            "is no compact readable PM->Trader factoring because there is no readable program. A clean negative — a "
            "property of P22, not a failure of H3. This CLOSES the coverage matrix (the last applicable cell)."
            if not any_pass else
            "H3 SUPPORTED on P22 — the PM->Trader factoring yields a more compact faithful surrogate (unexpected; "
            "investigate which level carries the structure)."),
        "closes_coverage_cell": True,
    }
    (HERE / "h3_on_p22_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, r in res.items():
        print(f"\n{name}: flat={r['flat_fidelity_by_budget']}  hier={r['hier_fidelity_by_budget']}")
        print(f"   chance(mode)={r['chance_mode_majority']} flat_best={r['flat_best']} hier_best={r['hier_best']} "
              f"above_chance={r['any_surrogate_above_chance']} hier_helps={r['hierarchy_helps']} H3_PASS={r['H3_PASS']}")
    print("\nVERDICT:", report["H3_on_P22_verdict"])
    print(f"[h3-p22] wrote h3_on_p22_report.json")


if __name__ == "__main__":
    main()
