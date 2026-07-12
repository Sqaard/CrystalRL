"""E-13 / v8 — the 5-20d REBALANCE lane: test the E-08 macro signal at ITS OWN horizon through the frozen gate.

E-11's kill point was a HORIZON MISMATCH: the gate certified *daily* exposure switching, but the E-08 signal
(live-macro block on the clean Dow panel) is a 5-20d timing signal (IC +0.089@5d / +0.150@20d, nothing @1d).
v8 adds the one legitimate degree of freedom the diagnosis names: the policy REBALANCES EVERY H DAYS —
exposure is set from the belief at the rebalance date and HELD for H days (H itself a searched, priced knob,
5..20). Everything else is unchanged and imported verbatim: the v7 macro belief (train-frozen causal Gaussian
HMM over named live observables), the v4 gate (typed lanes, span-covering rotating holdout, alpha-wealth,
disjoint re-confirmation, stress adversary, canaries+freeze), the bandit proposer, single-shot OOS.

Multiplicity note: H is searched INSIDE the loop (every informative query priced by alpha-wealth) — not three
separate loops with a picked winner, which would be exactly the phase/grid selection mode this project polices.

Pre-registered falsifiers: shuffled-belief placebo (expect 0), +3bp positive control through the v8 strat
(expect ACCEPT), belief occupancy report. Verdict rule: accepts>0 AND placebo==0 -> the lane converts the
signal; 0 accepts with control accepted -> honest NULL (the conversion gap survives the horizon fix).

Run: python interpretability/hl_v8_rebalance_lane.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import interpretability.hl_v4_over_crystal1 as V4                    # noqa: E402
from interpretability.hl_v4_over_crystal1 import Crystal1V4Gate, ann_dd, COST  # noqa: E402
from interpretability.hl_v7_macro_belief import (                    # noqa: E402
    load_panel, build_macro_belief, window, DEV, HOLD, OOS, HOLD_WIN,
)
from src.hl.mechanism_bandit import MechanismBandit                  # noqa: E402

OUT = HERE / "hl_v8_rebalance_lane_report.json"
KNOBS8 = {"t1": (0.10, 0.60, 0.30), "t2": (0.40, 0.90, 0.70),
          "lvl_reduced": (0.10, 1.00, 1.00), "lvl_defensive": (0.00, 1.00, 1.00),
          "H": (5.0, 20.0, 10.0)}
ANCHOR8 = {k: v[2] for k, v in KNOBS8.items()}          # levels 1.0 -> behavior == buy-and-hold regardless of H
ARMS8 = ["t1", "t2", "lvl_reduced", "lvl_defensive", "H", "joint_defend"]
ROUNDS = 40


def _exposure(c, b):
    t1, t2 = c["t1"], max(c["t2"], c["t1"] + 1e-6)
    return 1.0 if b < t1 else (c["lvl_reduced"] if b < t2 else c["lvl_defensive"])


def strat_v8(c, ro, bl):
    """Exposure decided at rebalance dates (every H days) from the belief there, HELD in between."""
    H = int(round(c.get("H", ANCHOR8["H"])))
    H = max(1, H)
    ex = np.empty(len(ro))
    cur = 1.0
    for i in range(len(ro)):
        if i % H == 0:
            cur = _exposure(c, bl[i])
        ex[i] = cur
    costs = np.abs(np.diff(np.concatenate([[1.0], ex]))) * COST
    return ex * ro - costs


def n_off_anchor8(c):
    return float(sum(1 for k in KNOBS8 if abs(float(c.get(k, ANCHOR8[k])) - ANCHOR8[k]) > 1e-9))


def make_gate(dev, hold, strat_fn=strat_v8):
    """v4 gate with the v8 strat; authority counting patched to include H (module-level fn lookup)."""
    V4.n_off_anchor = n_off_anchor8
    gate = Crystal1V4Gate(dev, hold, HOLD_WIN, strat_fn=strat_fn)
    # the canary bank configs lack H -> they inherit the anchor H via c.get() in strat_v8 (bank unchanged)
    return gate


def run_loop(dev, hold, oos, tag, rounds=ROUNDS):
    gate = make_gate(dev, hold)
    current = dict(ANCHOR8)
    gate.frontier = [(dict(current), gate.vec(current))]
    bandit = MechanismBandit(arms=list(ARMS8))
    step = {k: 0.2 * (v[1] - v[0]) for k, v in KNOBS8.items()}
    direction = {k: -1.0 for k in KNOBS8}
    direction["H"] = -1.0                                            # start by shortening toward 5d (E-08's fastest lane)
    trail = []
    for rnd in range(rounds):
        if rnd > 0 and rnd % 10 == 0:
            gate.canary_check(current)
        arm = bandit.select(ARMS8)
        cand = dict(current)
        if arm == "joint_defend":
            cand["t2"] = float(np.clip(cand["t2"] - step["t2"], *KNOBS8["t2"][:2]))
            cand["lvl_defensive"] = float(np.clip(cand["lvl_defensive"] - step["lvl_defensive"], *KNOBS8["lvl_defensive"][:2]))
        else:
            lo, hi, _ = KNOBS8[arm]
            cand[arm] = float(np.clip(cand[arm] + step[arm] * direction[arm], lo, hi))
        verdict, info, current = gate.review(cand, current)
        ok = verdict.startswith("ACCEPTED")
        bandit.update(arm, 1.0 if ok else 0.0)
        if not ok and arm != "joint_defend":
            step[arm] *= 0.6; direction[arm] = -direction[arm]
        trail.append({"round": rnd, "arm": arm, "verdict": verdict,
                      **{k: v for k, v in info.items() if k != "cand"}})
    gate.canary_check(current)

    def perf(c, win):
        p = strat_v8(c, win[0], win[1]); a, d = ann_dd(p)
        return {"ann": round(a, 4), "maxDD": round(d, 4),
                "sharpe": round(float(np.sqrt(252) * p.mean() / (p.std() + 1e-12)), 3),
                "dsd_bp": round(float(np.sqrt((np.minimum(p, 0) ** 2).mean())) * 1e4, 2)}
    counts = {}
    for t in trail:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    return {"tag": tag, "accepts": counts.get("ACCEPTED_RISKMODE", 0), "gate_counts": counts,
            "audit": gate.audit, "gate_compromised": gate.compromised,
            "certified_coeffs": {k: round(float(current[k]), 3) for k in KNOBS8},
            "hold_full": {"anchor": perf(ANCHOR8, hold), "final": perf(current, hold)},
            "oos_single_shot": {"anchor": perf(ANCHOR8, oos), "final": perf(current, oos)},
            "trail": trail}


def positive_control(dev, hold):
    def boosted(c, ro, bl):
        base = strat_v8(c, ro, bl)
        if c.get("__boost__"):
            base = base + 0.0003 + 0.001 * np.sin(1e4 * ro)
        return base
    gate = make_gate(dev, hold, strat_fn=boosted)
    gate.frontier = [(dict(ANCHOR8), gate.vec(ANCHOR8))]
    sentinel = dict(ANCHOR8); sentinel["__boost__"] = 1
    verdict, info, _ = gate.review(sentinel, dict(ANCHOR8))
    return {"verdict": verdict, "accepted": verdict.startswith("ACCEPTED"),
            **{k: v for k, v in info.items() if k != "cand"}}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-13 / v8 — the 5-20d rebalance lane (H priced inside the loop) ===")
    r, macro = load_panel()
    bel, bel_meta = build_macro_belief(r, macro)
    dev, hold, oos = window(r, bel, *DEV), window(r, bel, *HOLD), window(r, bel, *OOS)

    main_run = run_loop(dev, hold, oos, "v8_rebalance_macro_belief")

    rng = np.random.default_rng(0)
    bl_all = bel.to_numpy().copy()
    blocks = [bl_all[i:i + 60] for i in range(0, len(bl_all), 60)]
    rng.shuffle(blocks)
    bel_pl = pd.Series(np.concatenate(blocks)[:len(bl_all)], index=bel.index)
    placebo = run_loop(window(r, bel_pl, *DEV), window(r, bel_pl, *HOLD), window(r, bel_pl, *OOS),
                       "placebo_shuffled_belief")
    control = positive_control(dev, hold)

    rep = {"experiment": "E-13 v8 5-20d rebalance lane (the E-11 horizon fix)",
           "belief": bel_meta, "knobs": {k: list(v) for k, v in KNOBS8.items()},
           "main": {k: v for k, v in main_run.items() if k != "trail"},
           "placebo": {k: v for k, v in placebo.items() if k != "trail"},
           "positive_control": control, "trail": main_run["trail"]}
    if main_run["accepts"] > 0 and placebo["accepts"] == 0:
        rep["verdict"] = (f"CONVERTED — {main_run['accepts']} certified accept(s) at the rebalance horizon; "
                          "belief load-bearing (placebo 0)")
    elif main_run["accepts"] > 0:
        rep["verdict"] = f"WARNING — accepts {main_run['accepts']} but placebo also {placebo['accepts']} (not belief-specific)"
    else:
        rep["verdict"] = ("NULL — the conversion gap SURVIVES the horizon fix: 0 accepts on the 5-20d lane too"
                          if control["accepted"] else "INCONCLUSIVE — positive control failed (dead gate)")
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    for run in (main_run, placebo):
        print(f"[{run['tag']}] accepts={run['accepts']} gate={run['gate_counts']}")
        print(f"    coeffs={run['certified_coeffs']}")
        print(f"    hold: anchor {run['hold_full']['anchor']} -> final {run['hold_full']['final']}")
        print(f"    OOS : anchor {run['oos_single_shot']['anchor']} -> final {run['oos_single_shot']['final']}")
    print("positive control:", control)
    print("\nVERDICT:", rep["verdict"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
