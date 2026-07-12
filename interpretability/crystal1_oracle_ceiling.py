"""Oracle-ceiling decomposition: is the csi500/Dow null the SIGNAL's fault or the MODEL's (policy class)?

The user's hypothesis: the problem is the model + preprocessing. Decompose it with deliberately
future-peeking ORACLE signals fed through the honest v4 gate:

  O-A  oracle REGIME signal (forward-20d realized vol / forward drawdown state) through the SAME 4-knob
       threshold policy, reviewed by the REAL v4 gate. Certifies => the policy class + gate are fine and
       the bottleneck is the SIGNAL (the K=2 binary-|EW-ret| belief filter / preprocessing).
  O-B  stronger oracle (next-day sign) through the same class — the sanity ceiling.
  O-C  policy-class headroom with the CURRENT learned belief: exhaustive dev-grid over the 4 knobs,
       dev-best evaluated RAW on hold (no gate) — how much the class could ever give with today's signal.

Everything reuses hl_v4_over_crystal1's substrate + gate verbatim (strat, Crystal1V4Gate, windows).
Oracle beliefs are mapped to {0.05, 0.95} so the knob ranges (t1 in [.1,.6], t2 in [.4,.9]) can act on them.
Run: python interpretability/crystal1_oracle_ceiling.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import interpretability.hl_v4_over_crystal1 as V  # noqa: E402

OUT = HERE / "crystal1_oracle_ceiling_report.json"
B3 = {"t1": 0.30, "t2": 0.70, "lvl_reduced": 0.60, "lvl_defensive": 0.30}


def oracle_beliefs(r):
    """Deliberately future-peeking regime indicators, mapped onto the belief scale {0.05, 0.95}."""
    x = r.to_numpy()
    n = len(x)
    fwd_vol = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    for t in range(n):
        w = x[t + 1:t + 21]
        if len(w) >= 5:
            fwd_vol[t] = w.std()
            fwd_ret[t] = w.sum()
    volq = np.nanquantile(fwd_vol, 0.75)
    sign_next = np.concatenate([x[1:] < 0, [False]])
    return {
        "oracle_vol": pd.Series(np.where(fwd_vol >= volq, 0.95, 0.05), index=r.index),
        "oracle_dd": pd.Series(np.where(fwd_ret <= -0.03, 0.95, 0.05), index=r.index),
        "oracle_sign": pd.Series(np.where(sign_next, 0.95, 0.05), index=r.index),
    }


def window_o(r, bel, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    return r[m].to_numpy()[1:], bel[m].to_numpy()[:-1]


def gate_test(spec, r, bel_series, cand, label):
    """Fresh v4 gate (cheapest bar), REAL review() path: candidate vs the static-full anchor."""
    dev = window_o(r, bel_series, *spec["dev"])
    hold = window_o(r, bel_series, *spec["hold"])
    g = V.Crystal1V4Gate(dev, hold, spec["hold_win"])
    g.frontier = [(dict(V.ANCHOR), g.vec(V.ANCHOR))]
    verdict, info, _ = g.review(dict(cand), dict(V.ANCHOR))
    av, cv = g.vec(V.ANCHOR), g.vec(cand)
    hold_a = V.ann_dd(V.strat(V.ANCHOR, *hold))
    hold_c = V.ann_dd(V.strat(cand, *hold))
    return {"candidate": label, "verdict": verdict,
            "info": {k: v for k, v in info.items() if k != "cand"},
            "dev": {"anchor": {"ann": round(av["ann"], 3), "DD": round(av["maxDD"], 3)},
                    "cand": {"ann": round(cv["ann"], 3), "DD": round(cv["maxDD"], 3)}},
            "hold_raw": {"anchor": {"ann": round(hold_a[0], 3), "DD": round(hold_a[1], 3)},
                         "cand": {"ann": round(hold_c[0], 3), "DD": round(hold_c[1], 3)}}}


def dev_grid_best(spec, r, bel_series):
    """O-A/O-C helper: pick the dev-best 4-knob config by risk-adjusted dev score (ann + maxDD)."""
    dev = window_o(r, bel_series, *spec["dev"])
    best, best_s = None, -1e9
    for t1 in (0.10, 0.30, 0.50):
        for t2 in (0.50, 0.70, 0.90):
            if t2 <= t1:
                continue
            for lr in (0.25, 0.50, 0.75, 1.00):
                for ld in (0.00, 0.25, 0.50):
                    c = {"t1": t1, "t2": t2, "lvl_reduced": lr, "lvl_defensive": ld}
                    a, d = V.ann_dd(V.strat(c, *dev))
                    s = a + d                      # ann + maxDD (DD negative) = the Calmar-style dev score
                    if s > best_s:
                        best, best_s = c, s
    return best


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    report = {"design": "O-A oracle regime through the SAME 4-knob class + REAL v4 gate; "
                        "O-B next-day-sign oracle ceiling; O-C class headroom with the current learned belief.",
              "panels": {}}
    for name in ("csi500", "dow"):
        spec = V.PANELS[name]
        r, bel_learned = V.build_belief(spec["panel"], *spec["train"])
        oracles = oracle_beliefs(r)
        entry = {"O_A_gate": [], "O_B_gate": [], "O_C_headroom": {}}

        # O-A: oracle regime signals through the knob class, gate-tested (B3-shaped + dev-grid-best configs)
        for oname in ("oracle_vol", "oracle_dd"):
            ob = oracles[oname]
            gb = dev_grid_best(spec, r, ob)
            entry["O_A_gate"].append({**gate_test(spec, r, ob, B3, f"{oname}+B3"), "grid_best": None})
            entry["O_A_gate"].append({**gate_test(spec, r, ob, gb, f"{oname}+devgrid"), "grid_best": gb})

        # O-B: the sanity ceiling
        gbs = dev_grid_best(spec, r, oracles["oracle_sign"])
        entry["O_B_gate"].append({**gate_test(spec, r, oracles["oracle_sign"], gbs, "oracle_sign+devgrid"),
                                  "grid_best": gbs})

        # O-C: class headroom with the CURRENT learned belief (dev-select, raw hold eval, no gate)
        gbl = dev_grid_best(spec, r, bel_learned)
        dev = window_o(r, bel_learned, *spec["dev"]); hold = window_o(r, bel_learned, *spec["hold"])
        a_dev = V.ann_dd(V.strat(V.ANCHOR, *dev)); c_dev = V.ann_dd(V.strat(gbl, *dev))
        a_h = V.ann_dd(V.strat(V.ANCHOR, *hold)); c_h = V.ann_dd(V.strat(gbl, *hold))
        entry["O_C_headroom"] = {
            "dev_grid_best": gbl,
            "dev": {"anchor": {"ann": round(a_dev[0], 3), "DD": round(a_dev[1], 3)},
                    "best": {"ann": round(c_dev[0], 3), "DD": round(c_dev[1], 3)}},
            "hold_raw": {"anchor": {"ann": round(a_h[0], 3), "DD": round(a_h[1], 3)},
                         "best": {"ann": round(c_h[0], 3), "DD": round(c_h[1], 3)}},
        }
        report["panels"][name] = entry
        print(f"\n=== {name} ===")
        for e in entry["O_A_gate"] + entry["O_B_gate"]:
            print(f"  {e['candidate']:24s} -> {e['verdict']:22s} dev cand ann {e['dev']['cand']['ann']:+.3f} DD {e['dev']['cand']['DD']:+.3f}"
                  f" | hold raw cand ann {e['hold_raw']['cand']['ann']:+.3f} DD {e['hold_raw']['cand']['DD']:+.3f}"
                  f" (anchor {e['hold_raw']['anchor']['ann']:+.3f}/{e['hold_raw']['anchor']['DD']:+.3f})")
        oc = entry["O_C_headroom"]
        print(f"  O-C current-belief class headroom: dev best {oc['dev']['best']} vs anchor {oc['dev']['anchor']}"
              f" | hold raw {oc['hold_raw']['best']} vs {oc['hold_raw']['anchor']}")

    # the verdict logic
    oa = [e["verdict"].startswith("ACCEPTED") for p in report["panels"].values() for e in p["O_A_gate"]]
    report["verdict"] = (
        "ORACLE-THROUGH-KNOBS CERTIFIES on {}/{} gate tests: the 4-knob policy class + the v4 gate CAN monetize a "
        "good regime signal — the bottleneck is the SIGNAL (K=2 binary-|EW-ret| preprocessing), so upgrade the "
        "belief observation, not the shell.".format(sum(oa), len(oa))
        if any(oa) else
        "ORACLE-THROUGH-KNOBS DOES NOT CERTIFY on any gate test: even a perfect regime signal cannot be monetized "
        "by the 4-knob class under the honest gate — the POLICY CLASS (model) is the bottleneck.")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + report["verdict"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
