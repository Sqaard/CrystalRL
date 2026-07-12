"""E-15 / v9 — the FRESH-OOS re-cut: the strongest legitimate test of the E-08 signal, on data no one has seen.

PRE-REGISTRATION (fixed before the run; rationale written first):
  Extended substrate : data/_dow_extended/dow_extended_panel.csv — 29 Dow names + [VIX, SP500_Trend,
                       turbulence, 10Y_Yield] rebuilt 2010..2026-07 from ONE source (Yahoo, one clock),
                       overlap-verified vs the clean old panel (corr 0.96-0.98; EW-return corr 1.000).
  belief train       : 2010-01-01..2018-12-31 (frozen; K by train-internal held-out LL; bear = argmax VIX
                       emission — observation-named).
  dev                : 2019-01-01..2021-12-31 — CONTAINS THE COVID BEAR. Rationale: E-11/E-13's kill point
                       was a calm dev window where no defensive move can show a Pareto gain; a bear-inclusive
                       dev fixes that WITHOUT touching the 2022 bear or the fresh OOS.
  hold               : 2022-01-01..2023-12-31 — the 2022 bear + 2023 recovery (the window where the E-08 IC
                       lived; now legitimately usable as the gate's rotating holdout, because the fresh OOS
                       replaces it as the untouched test).
  frozen OOS         : 2024-01-01..2026-07-07 (629 trading days) — untouched by ANY experiment, by E-08's
                       measurement, and by every prior loop. Touched ONCE at the end.
  machinery          : the v8 rebalance lane verbatim (H in 5-20d priced inside the loop), v4 gate, 40 rounds,
                       shuffled-belief placebo, +3bp positive control.
  verdict rule       : accepts>0 AND placebo==0 -> report the certified config's ONE-SHOT fresh-OOS numbers
                       vs buy-and-hold (that comparison IS the result, whatever it says); 0 accepts with the
                       control accepted -> the conversion gap survives even a bear-inclusive dev (terminal
                       NULL for this signal under this discipline).

Run: python interpretability/hl_v9_fresh_oos.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v8_rebalance_lane import run_loop, positive_control  # noqa: E402  (v8 machinery verbatim)
from interpretability.hl_v6_crystal1_features import GaussianHMM              # noqa: E402

OUT = HERE / "hl_v9_fresh_oos_report.json"
PANEL = ROOT / "data" / "_dow_extended" / "dow_extended_panel.csv"
MACRO = ["VIX", "SP500_Trend", "turbulence", "10Y_Yield"]
TRAIN = ("2010-01-01", "2018-12-31")
DEV = ("2019-01-01", "2021-12-31")
HOLD = ("2022-01-01", "2023-12-31")
OOS = ("2024-01-01", "2026-07-07")


def load_extended():
    df = pd.read_csv(PANEL)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "tic", "adjclose"]).sort_values(["tic", "date"])
    df["ret"] = df.groupby("tic")["adjclose"].pct_change()
    r = df.groupby("date")["ret"].mean().dropna()
    macro = df.drop_duplicates("date").set_index("date")[MACRO].sort_index().ffill()
    macro = macro.reindex(r.index).ffill()
    return r, macro


def build_belief(r, macro, k_choices=(2, 3)):
    m_tr = (macro.index >= pd.Timestamp(TRAIN[0])) & (macro.index <= pd.Timestamp(TRAIN[1]))
    X_tr = macro[m_tr].to_numpy(dtype=float)
    mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0) + 1e-9
    Z = np.nan_to_num((macro.to_numpy(dtype=float) - mu) / sd, nan=0.0)
    Z_tr = Z[np.asarray(m_tr)]
    cut = int(len(Z_tr) * 0.8)
    best = None
    for K in k_choices:
        h = GaussianHMM(K); h.fit(Z_tr[:cut], seed=0)
        _, ll = h.causal_filter(Z_tr[cut:])
        if best is None or ll > best[1]:
            best = (K, ll)
    K = best[0]
    hmm = GaussianHMM(K); hmm.fit(Z_tr, seed=0)
    bear = int(np.argmax(hmm.mu[:, 0]))
    gamma, _ = hmm.causal_filter(Z)
    return pd.Series(gamma[:, bear], index=macro.index), {"K": K, "heldout_ll": round(best[1], 3),
                                                          "bear_state_vix_mu": round(float(hmm.mu[bear, 0]), 2)}


def window(r, bel, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    return r[m].to_numpy()[1:], bel[m].to_numpy()[:-1]


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-15 / v9 — fresh-OOS re-cut (dev incl. COVID bear; OOS 2024-26 untouched) ===")
    r, macro = load_extended()
    bel, bel_meta = build_belief(r, macro)
    print(f"belief: K={bel_meta['K']} bear mu_z(VIX)={bel_meta['bear_state_vix_mu']} | "
          f"span {r.index.min().date()}..{r.index.max().date()}")
    dev, hold, oos = window(r, bel, *DEV), window(r, bel, *HOLD), window(r, bel, *OOS)
    print(f"windows: dev {len(dev[0])}d (incl. COVID), hold {len(hold[0])}d (2022 bear), OOS {len(oos[0])}d (fresh)")

    main_run = run_loop(dev, hold, oos, "v9_fresh_oos_macro_belief")

    rng = np.random.default_rng(0)
    bl = bel.to_numpy().copy()
    blocks = [bl[i:i + 60] for i in range(0, len(bl), 60)]
    rng.shuffle(blocks)
    bel_pl = pd.Series(np.concatenate(blocks)[:len(bl)], index=bel.index)
    placebo = run_loop(window(r, bel_pl, *DEV), window(r, bel_pl, *HOLD), window(r, bel_pl, *OOS),
                       "placebo_shuffled_belief")
    control = positive_control(dev, hold)

    rep = {"experiment": "E-15 v9 fresh-OOS re-cut (pre-registered; OOS 2024-01..2026-07 untouched)",
           "substrate": "dow_extended_panel.csv (one source, one clock, overlap-verified)",
           "preregistration": {"train": TRAIN, "dev": DEV + ("includes COVID bear",),
                               "hold": HOLD + ("includes 2022 bear",), "oos": OOS + ("fresh, one shot",)},
           "belief": bel_meta,
           "main": {k: v for k, v in main_run.items() if k != "trail"},
           "placebo": {k: v for k, v in placebo.items() if k != "trail"},
           "positive_control": control, "trail": main_run["trail"]}
    if main_run["accepts"] > 0 and placebo["accepts"] == 0:
        oo = main_run["oos_single_shot"]
        rep["verdict"] = (f"{main_run['accepts']} certified accept(s) with a bear-inclusive dev; FRESH-OOS "
                          f"one-shot: anchor {oo['anchor']} vs final {oo['final']} — this comparison is the result.")
    elif main_run["accepts"] > 0:
        rep["verdict"] = f"WARNING: accepts {main_run['accepts']} but placebo {placebo['accepts']} (not belief-specific)"
    else:
        rep["verdict"] = ("TERMINAL NULL for this signal under this discipline — 0 accepts even with a "
                          "bear-inclusive dev and a bear holdout" if control["accepted"]
                          else "INCONCLUSIVE — positive control failed")
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    for run in (main_run, placebo):
        print(f"[{run['tag']}] accepts={run['accepts']} gate={run['gate_counts']}")
        print(f"    coeffs={run['certified_coeffs']}")
        print(f"    hold(2022-23): anchor {run['hold_full']['anchor']} -> final {run['hold_full']['final']}")
        print(f"    FRESH OOS(2024-26): anchor {run['oos_single_shot']['anchor']} -> final {run['oos_single_shot']['final']}")
    print("positive control:", control)
    print("\nVERDICT:", rep["verdict"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
