"""E-11 / v7 — CRYSTAL-1 belief over the LIVE macro block (the E-08 signal) on the VERIFIED-CLEAN Dow panel.

THE MAIN-PROBLEM CLOSURE ATTEMPT. The project's honest wall: the self-improvement engine certifies real changes
on the polygon but 0 on real daily data. The v5/v6 diagnosis said the SIGNAL was the bottleneck (the K=2
return-alphabet belief has negative class headroom OOS). E-08 then found the first capacity-fair OOS signal on
accessible data: the DOW30 panel's live US-macro block predicts 5-20d returns (IC +0.09/+0.15) — and E-10c
verified this panel is CLEAN (close/volume ground-truth vs Yahoo at lag0 corr 1.000, unlike csi500's mixed clock).

v7 = the v4-over-CRYSTAL-1 loop with ONE change (capacity-matched): the belief.
  v5-era belief : K=2 filter over the EW-return magnitude alphabet  -> certified 0 (return-NI failed ~20x)
  v7 belief     : causal Gaussian HMM over NAMED live macro observables [VIX, SP500_Trend, turbulence, 10Y_Yield],
                  train-frozen (2010-2016) standardizer + params, K in {2,3} by train-internal held-out LL,
                  bear state NAMED by highest VIX emission mean (observation-based, leak-free).
Everything else is imported VERBATIM from hl_v4_over_crystal1: the 4-knob exposure shell, the certified v4 gate
(typed RETURN/RISK lanes, span-covering rotating holdout, alpha-investing wealth, disjoint re-confirmation,
stress adversary, canaries + escape=>freeze), the bandit proposer schedule, single-shot OOS.

Pre-registered falsifiers (fixed before the run):
  F1 placebo   : block-shuffled belief (60d blocks) through the SAME loop -> must certify ~0 (belief load-bearing).
  F2 teeth     : the +3bp noisy sentinel through a fresh gate -> must be ACCEPTED (gate alive).
  F3 liveness  : report belief occupancy per window; a belief that never crosses the thresholds is honestly inert.
Literature priors (REFERENCES.md): Cederburg et al. 2020 — the modal failure is signal->weight instability OOS;
Ang-Bekaert 2002 — the value should express as cash-timing; our 2022-bear window was the maximally favorable draw.

Run: python interpretability/hl_v7_macro_belief.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v4_over_crystal1 import (          # noqa: E402  (the certified machinery, verbatim)
    Crystal1V4Gate, strat, ann_dd, mdl_axis_a, ANCHOR, KNOBS, ARMS,
)
from interpretability.hl_v6_crystal1_features import GaussianHMM  # noqa: E402
from src.hl.mechanism_bandit import MechanismBandit          # noqa: E402

OUT = HERE / "hl_v7_macro_belief_report.json"
PANEL = ROOT / "PPO_configurations_comparison" / "processed_final_fixed.csv"
MACRO = ["VIX", "SP500_Trend", "turbulence", "10Y_Yield"]    # named, live (E-07), raw observables
TRAIN = ("2010-01-01", "2016-12-31"); DEV = ("2017-01-01", "2018-12-31")
HOLD = ("2019-01-01", "2020-12-31"); OOS = ("2021-01-01", "2023-02-28")
HOLD_WIN = 120; ROUNDS = 40


def load_panel():
    df = pd.read_csv(PANEL, usecols=lambda c: c in set(["date", "tic", "close"] + MACRO))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "tic", "close"]).sort_values(["tic", "date"])
    df["ret"] = df.groupby("tic")["close"].pct_change()
    r = df.groupby("date")["ret"].mean().dropna()
    macro = df.drop_duplicates("date").set_index("date")[MACRO].sort_index().ffill()
    macro = macro.reindex(r.index).ffill()
    return r, macro


def build_macro_belief(r, macro, k_choices=(2, 3)):
    """Causal HMM belief over the macro observables; ALL statistics train-frozen."""
    m_tr = (macro.index >= pd.Timestamp(TRAIN[0])) & (macro.index <= pd.Timestamp(TRAIN[1]))
    X_tr = macro[m_tr].to_numpy(dtype=float)
    mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0) + 1e-9          # train-frozen standardizer
    Z_full = (macro.to_numpy(dtype=float) - mu) / sd
    Z_full = np.nan_to_num(Z_full, nan=0.0)
    Z_tr = Z_full[np.asarray(m_tr)]
    cut = int(len(Z_tr) * 0.8)
    best = None
    for K in k_choices:                                              # K by train-internal held-out LL
        h = GaussianHMM(K); h.fit(Z_tr[:cut], seed=0)
        _, ll = h.causal_filter(Z_tr[cut:])
        if best is None or ll > best[1]:
            best = (K, ll)
    K = best[0]
    hmm = GaussianHMM(K); hmm.fit(Z_tr, seed=0)                      # refit on the full train window
    bear = int(np.argmax(hmm.mu[:, 0]))                             # NAMED: highest standardized-VIX emission mean
    gamma, _ = hmm.causal_filter(Z_full)                             # past-only posterior over the full span
    return pd.Series(gamma[:, bear], index=macro.index), {"K": K, "heldout_ll": round(best[1], 3),
                                                          "bear_state_vix_mu": round(float(hmm.mu[bear, 0]), 2)}


def window(r, bel, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    return r[m].to_numpy()[1:], bel[m].to_numpy()[:-1]               # belief at t-1 gates exposure for day t


def occupancy(bl):
    return {"mean": round(float(np.mean(bl)), 3), "frac>0.3": round(float((bl > 0.3).mean()), 3),
            "frac>0.7": round(float((bl > 0.7).mean()), 3)}


def run_loop(dev, hold, oos, tag, rounds=ROUNDS):
    gate = Crystal1V4Gate(dev, hold, HOLD_WIN)
    current = dict(ANCHOR)
    gate.frontier = [(dict(current), gate.vec(current))]
    bandit = MechanismBandit(arms=list(ARMS))
    step = {k: 0.2 * (v[1] - v[0]) for k, v in KNOBS.items()}
    direction = {k: -1.0 for k in KNOBS}
    trail = []
    for rnd in range(rounds):
        if rnd > 0 and rnd % 10 == 0:
            gate.canary_check(current)
        arm = bandit.select(ARMS)
        cand = dict(current)
        if arm == "joint_defend":
            cand["t2"] = float(np.clip(cand["t2"] - step["t2"], *KNOBS["t2"][:2]))
            cand["lvl_defensive"] = float(np.clip(cand["lvl_defensive"] - step["lvl_defensive"], *KNOBS["lvl_defensive"][:2]))
        else:
            lo, hi, _ = KNOBS[arm]
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
        p = strat(c, win[0], win[1]); a, d = ann_dd(p)
        return {"ann": round(a, 4), "maxDD": round(d, 4),
                "sharpe": round(float(np.sqrt(252) * p.mean() / (p.std() + 1e-12)), 3),
                "dsd_bp": round(float(np.sqrt((np.minimum(p, 0) ** 2).mean())) * 1e4, 2)}
    counts = {}
    for t in trail:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    return {"tag": tag, "accepts": counts.get("ACCEPTED_RISKMODE", 0), "gate_counts": counts,
            "audit": gate.audit, "gate_compromised": gate.compromised,
            "certified_coeffs": {k: round(current[k], 3) for k in KNOBS},
            "belief_occupancy": {"dev": occupancy(dev[1]), "hold": occupancy(hold[1]), "oos": occupancy(oos[1])},
            "hold_full": {"anchor": perf(ANCHOR, hold), "final": perf(current, hold)},
            "oos_single_shot": {"anchor": perf(ANCHOR, oos), "final": perf(current, oos)},
            "mdl_axis_a_dev": {"anchor": round(mdl_axis_a(ANCHOR, *dev), 3), "final": round(mdl_axis_a(current, *dev), 3)},
            "trail": trail}


def positive_control(dev, hold):
    def boosted(c, ro, bl):
        base = strat(c, ro, bl)
        if c.get("__boost__"):
            base = base + 0.0003 + 0.001 * np.sin(1e4 * ro)
        return base
    gate = Crystal1V4Gate(dev, hold, HOLD_WIN, strat_fn=boosted)
    gate.frontier = [(dict(ANCHOR), gate.vec(ANCHOR))]
    sentinel = dict(ANCHOR); sentinel["__boost__"] = 1
    verdict, info, _ = gate.review(sentinel, dict(ANCHOR))
    return {"verdict": verdict, "accepted": verdict.startswith("ACCEPTED"),
            **{k: v for k, v in info.items() if k != "cand"}}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-11 / v7 — macro-belief CRYSTAL-1 loop on the verified-clean Dow panel ===")
    r, macro = load_panel()
    bel, bel_meta = build_macro_belief(r, macro)
    print(f"belief: K={bel_meta['K']} (held-out LL {bel_meta['heldout_ll']}), bear = argmax VIX emission "
          f"(mu_z={bel_meta['bear_state_vix_mu']})")
    dev, hold, oos = window(r, bel, *DEV), window(r, bel, *HOLD), window(r, bel, *OOS)

    main_run = run_loop(dev, hold, oos, "v7_macro_belief")

    # F1 placebo: 60d-block-shuffled belief through the identical loop
    rng = np.random.default_rng(0)
    bl_all = bel.to_numpy().copy()
    blocks = [bl_all[i:i + 60] for i in range(0, len(bl_all), 60)]
    rng.shuffle(blocks)
    bel_pl = pd.Series(np.concatenate(blocks)[:len(bl_all)], index=bel.index)
    dev_p, hold_p, oos_p = window(r, bel_pl, *DEV), window(r, bel_pl, *HOLD), window(r, bel_pl, *OOS)
    placebo = run_loop(dev_p, hold_p, oos_p, "placebo_shuffled_belief")

    # F2 teeth
    control = positive_control(dev, hold)

    rep = {"experiment": "E-11 v7 macro-belief loop (capacity-matched belief swap vs the v5-era return-belief run)",
           "panel_integrity": "DOW30 verified clean by E-10c (close/volume lag0 corr 1.000 vs Yahoo)",
           "belief": {"observables": MACRO, **bel_meta, "train": TRAIN},
           "windows": {"dev": DEV, "hold": HOLD, "oos": OOS, "hold_win_days": HOLD_WIN},
           "main": {k: v for k, v in main_run.items() if k != "trail"},
           "placebo": {k: v for k, v in placebo.items() if k != "trail"},
           "positive_control": control,
           "trail": main_run["trail"]}
    load_bearing = main_run["accepts"] > 0 and placebo["accepts"] == 0
    if main_run["accepts"] > 0:
        rep["verdict"] = (f"{main_run['accepts']} certified accept(s) with the macro belief"
                          + (" — belief LOAD-BEARING (placebo certifies 0)" if load_bearing
                             else f" — WARNING: placebo certified {placebo['accepts']} (not belief-specific)"))
    else:
        rep["verdict"] = ("NULL — the macro belief certifies 0 through the same gate "
                          "(consistent with Cederburg et al.: windowed IC does not convert to a certifiable policy)"
                          if control["accepted"] else
                          "INCONCLUSIVE — 0 accepts AND the positive control failed (dead gate, rerun)")
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    for run in (main_run, placebo):
        print(f"[{run['tag']}] accepts={run['accepts']} gate={run['gate_counts']}")
        print(f"    coeffs={run['certified_coeffs']} | occupancy hold {run['belief_occupancy']['hold']}")
        print(f"    hold: anchor {run['hold_full']['anchor']} -> final {run['hold_full']['final']}")
        print(f"    OOS : anchor {run['oos_single_shot']['anchor']} -> final {run['oos_single_shot']['final']}")
    print("positive control:", control)
    print("\nVERDICT:", rep["verdict"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
