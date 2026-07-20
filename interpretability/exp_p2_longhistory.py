"""P-2 / E-31 — EXTEND THE DATA BACK TO 2000 + the long-history multi-year simulation.

Ivan's instruction: we have data through 2026 — now extend BACKWARD to 2000-2010 as well. That gives a
26-year base (2000-2026) containing FOUR bear regimes of different mechanism: dot-com (2000-02, slow grind),
GFC (2008-09, credit crash), COVID (2020, flash), 2022 (inflation, the bond-hedge failure) — exactly the
variety the 2019-23 base lacked (the P-1 kill was BASE-CONDITIONAL for this reason).

STAGES
  1. EXTEND: one-source Yahoo fetch 1999-01..2026-07 (fresh cache data/_dow_extended/full2000/) for the 29
     panel names + ^GSPC/^VIX/^TNX/^IRX + the multi-asset sleeves (EFA/IEF/TLT/TIP/GLD). Same single-clock
     discipline (one chart response per series). SANITY: the EW book must show the known crisis years
     (2008 deeply negative; 2000-02 negative run). SURVIVORSHIP disclosed: the 29-name list is the MODERN
     panel's — using it back to 2000 inherits survivor bias (favors equity books; stated in every output).
  2. BELIEF: the frozen v9 recipe (train 2010-2018 standardizer+HMM) filtered CAUSALLY over the full span —
     pre-2010 beliefs are out-of-training-era but point-in-time (an honest stress of the recipe).
  3. SIMULATE (the P-2 core): STATIONARY BOOTSTRAP (Politis-Romano, geometric blocks, E[block]=63d) over the
     26y daily pnl of each book; horizons h in {1,3,5,10}y -> P(goal | h) curves for goals {inflation 3%,
     deposits 4.5%, 6%, 10%} (annualized over the horizon), plus DD distributions.
     CALIBRATION CROSS-CHECK (Cogneau-Zakamulin bias guard): bootstrap P(goal|h) vs the EMPIRICAL frequency
     over rolling historical h-year windows; report gaps.
  4. MULTI-ASSET RETRY (the P-1 re-queue condition, now satisfiable): the risky-core x cash fans vs the chord
     on the LONG base (sleeves exist from ~2005; the retry runs 2005-2026), same domination kill test.

HONESTY BAND (ships with every horizon curve): long-run equity safety is NOT a theorem — predictive variance
rises with horizon under parameter uncertainty [Pastor-Stambaugh 2012]; survivorship-corrected 30y real-loss
probability ~12% [Anarkulova et al. 2022]; our own base is survivor-listed. Numbers are bootstrap estimates.

Run: python interpretability/exp_p2_longhistory.py       (fetch ~40 series; simulation minutes)
"""
from __future__ import annotations
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v9_fresh_oos import TRAIN  # noqa: E402
from interpretability.hl_v6_crystal1_features import GaussianHMM  # noqa: E402
from interpretability.build_dow_extended_panel import turbulence_series  # noqa: E402
from interpretability.hl_v8_rebalance_lane import strat_v8  # noqa: E402

OUT = HERE / "exp_p2_longhistory_report.json"
CACHE = ROOT / "data" / "_dow_extended" / "full2000"; CACHE.mkdir(parents=True, exist_ok=True)
P1U, P2U = 915148800, 1783468800          # 1999-01-01 .. 2026-07-08 (SMA200/turbulence warmup before 2000)
CERT = {"t1": 0.30, "t2": 0.657, "lvl_reduced": 1.0, "lvl_defensive": 0.738, "H": 10.0}
GOALS = {"inflation_3%": 0.03, "deposits_4.5%": 0.045, "6%": 0.06, "10%": 0.10}
HORIZONS = [1, 3, 5, 10]
N_PATHS = {1: 3000, 3: 2000, 5: 1500, 10: 1000}
EBLOCK = 63


def fetch(sym):
    f = CACHE / f"{sym.replace('^', '_idx_')}.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={P1U}&period2={P2U}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for a in range(4):
        try:
            j = json.loads(urllib.request.urlopen(req, timeout=30).read())
            r = j["chart"]["result"][0]; q = r["indicators"]["quote"][0]
            adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
            d = pd.DataFrame({"date": pd.to_datetime(r["timestamp"], unit="s").normalize(),
                              "close": q["close"], "adjclose": adj}).dropna()
            d.to_csv(f, index=False)
            return d
        except Exception as e:
            if a == 3:
                raise RuntimeError(f"{sym}: {e}")
            time.sleep(2)


def build_long_panel():
    tics = sorted(pd.read_csv(ROOT / "PPO_configurations_comparison" / "processed_final_fixed.csv",
                              usecols=["tic"])["tic"].dropna().unique())
    px, failed = {}, []
    for t in tics:
        try:
            px[t] = fetch(t).set_index("date")["adjclose"]
        except Exception:
            failed.append(t)
    idx = {s: fetch(s).set_index("date")["close"] for s in ("^GSPC", "^VIX", "^TNX", "^IRX")}
    wide = pd.DataFrame(px).sort_index()
    ret = wide.pct_change()
    r = ret.mean(axis=1).dropna()
    r = r[r.index >= pd.Timestamp("2000-01-03")]
    sp = idx["^GSPC"].sort_index(); sma = sp.rolling(200).mean()
    macro = pd.DataFrame({"VIX": idx["^VIX"].sort_index(), "SP500_Trend": (sp - sma) / sma,
                          "turbulence": turbulence_series(ret), "10Y_Yield": idx["^TNX"].sort_index()})
    macro = macro.reindex(r.index).ffill()
    rf = (idx["^IRX"].reindex(r.index).ffill() / 100 / 252).fillna(0.0)
    names_per_day = ret.reindex(r.index).notna().sum(axis=1)
    return r, macro, rf, {"tics_fetched": len(px), "failed": failed,
                          "names_2000": int(names_per_day.iloc[0]), "names_2010": int(names_per_day[r.index >= "2010-01-01"].iloc[0]),
                          "span": (str(r.index.min().date()), str(r.index.max().date())), "days": len(r)}


def frozen_belief(r, macro):
    m_tr = (macro.index >= pd.Timestamp(TRAIN[0])) & (macro.index <= pd.Timestamp(TRAIN[1]))
    X = macro.to_numpy(dtype=float)
    mu, sd = np.nanmean(X[np.asarray(m_tr)], 0), np.nanstd(X[np.asarray(m_tr)], 0) + 1e-9
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    Ztr = Z[np.asarray(m_tr)]
    cut = int(len(Ztr) * 0.8)
    best = None
    for K in (2, 3):
        h = GaussianHMM(K); h.fit(Ztr[:cut], seed=0)
        _, ll = h.causal_filter(Ztr[cut:])
        if best is None or ll > best[1]:
            best = (K, ll)
    hmm = GaussianHMM(best[0]); hmm.fit(Ztr, seed=0)
    bear = int(np.argmax(hmm.mu[:, 0]))
    g, _ = hmm.causal_filter(Z)
    return pd.Series(g[:, bear], index=macro.index)


def stationary_boot(x, n_days, n_paths, seed):
    """Politis-Romano stationary bootstrap: geometric block lengths, E[block]=EBLOCK."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty((n_paths, n_days))
    for p in range(n_paths):
        idx = np.empty(n_days, dtype=int)
        t = 0
        while t < n_days:
            start = rng.integers(0, n)
            L = min(int(rng.geometric(1.0 / EBLOCK)), n_days - t)
            seg = (start + np.arange(L)) % n
            idx[t:t + L] = seg
            t += L
        out[p] = x[idx]
    return out


def horizon_stats(pnl, seed=31):
    res = {}
    for h in HORIZONS:
        paths = stationary_boot(pnl, 252 * h, N_PATHS[h], seed + h)
        eq = np.cumprod(1 + paths, axis=1)
        ann = eq[:, -1] ** (1.0 / h) - 1
        # the running peak includes initial wealth 1.0 (E-27c fix; omitting it understates maxDD)
        eq1 = np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)
        dd = (eq1 / np.maximum.accumulate(eq1, axis=1) - 1).min(axis=1)
        res[h] = {"E_ann": round(float(ann.mean()), 4),
                  "ann_p20": round(float(np.quantile(ann, 0.20)), 4),
                  "maxDD_p95": round(float(np.quantile(dd, 0.05)), 4),
                  **{f"P(>{g})": round(float((ann > v).mean()), 3) for g, v in GOALS.items()}}
    return res


def empirical_check(pnl, h):
    """Rolling historical h-year windows: empirical P(ann > goal) — the bias cross-check."""
    n = 252 * h
    if len(pnl) < n + 252:
        return None
    anns = []
    for s in range(0, len(pnl) - n, 63):
        eq = float(np.prod(1 + pnl[s:s + n]))
        anns.append(eq ** (1.0 / h) - 1)
    anns = np.array(anns)
    return {f"P(>{g})": round(float((anns > v).mean()), 3) for g, v in GOALS.items()} | {"n_windows": len(anns)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== P-2 / E-31 — the 2000-2026 extension + long-history simulation ===")
    r, macro, rf, meta = build_long_panel()
    print(f"panel: {meta['span'][0]}..{meta['span'][1]} ({meta['days']}d) | names in 2000: {meta['names_2000']}, "
          f"in 2010: {meta['names_2010']} | failed: {meta['failed']}")
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    print("sanity (EW book, survivor-listed): 2002 %.1f%%  2008 %.1f%%  2020 %.1f%%  2022 %.1f%%" %
          (yearly.get(2002, np.nan) * 100, yearly.get(2008, np.nan) * 100,
           yearly.get(2020, np.nan) * 100, yearly.get(2022, np.nan) * 100))

    bel = frozen_belief(r, macro)
    ro = r.to_numpy()[1:]; bl = bel.to_numpy()[:-1]; rfv = rf.to_numpy()[1:]
    books = {
        "BH_dow": ro.copy(),
        "CERT_dow": strat_v8(CERT, ro, bl),
    }
    books["CERT_c40"] = 0.6 * books["CERT_dow"] + 0.4 * rfv
    books["CERT_c80"] = 0.2 * books["CERT_dow"] + 0.8 * rfv

    print("--- horizon curves (stationary bootstrap on 2000-2026) ---")
    curves, calib = {}, {}
    for nm, pnl in books.items():
        curves[nm] = horizon_stats(pnl)
        emp5 = empirical_check(pnl, 5)
        calib[nm] = {"boot_5y": {k: v for k, v in curves[nm][5].items() if k.startswith("P(")},
                     "empirical_5y_rolling": emp5}
        c = curves[nm]
        print(f"[{nm:10s}] P(>infl): 1y {c[1]['P(>inflation_3%)']} | 3y {c[3]['P(>inflation_3%)']} | "
              f"5y {c[5]['P(>inflation_3%)']} | 10y {c[10]['P(>inflation_3%)']}   "
              f"P(>6%): 1y {c[1]['P(>6%)']} -> 10y {c[10]['P(>6%)']}")

    # ---- the multi-asset RETRY on the long base (sleeves exist from ~2005) ----
    print("--- multi-asset retry on the long base (2005-2026) ---")
    sleeves = {}
    for nm, sym in (("EQ_INTL", "EFA"), ("BOND_7_10", "IEF"), ("BOND_20Y", "TLT"), ("TIPS", "TIP"), ("GOLD", "GLD")):
        sleeves[nm] = fetch(sym).set_index("date")["adjclose"].reindex(r.index).ffill().pct_change().fillna(0.0).to_numpy()[1:]
    start = int(np.searchsorted(r.index[1:], np.datetime64("2005-01-03")))
    S = {"EQ_DOW_CERT": books["CERT_dow"][start:], "CASH": rfv[start:],
         **{k: v[start:] for k, v in sleeves.items()}}
    cores = {"CLASSIC_60_40": {"EQ_DOW_CERT": .45, "EQ_INTL": .15, "BOND_7_10": .20, "BOND_20Y": .10, "TIPS": .05, "GOLD": .05},
             "EQUAL_W": {k: 1 / 6 for k in ("EQ_DOW_CERT", "EQ_INTL", "BOND_7_10", "BOND_20Y", "TIPS", "GOLD")},
             "BALANCED_50_30_20": {"EQ_DOW_CERT": .40, "EQ_INTL": .10, "BOND_7_10": .20, "BOND_20Y": .10, "TIPS": .10, "GOLD": .10}}
    def core_pnl(wd):
        keys = list(wd); w = np.array([wd[k] for k in keys])
        R = np.column_stack([S[k] for k in keys])
        port = np.zeros(len(R)); hw = w.copy()
        for t in range(len(R)):
            if t % 21 == 0 and t > 0:
                port[t] -= float(np.abs(hw - w).sum()) * 0.001; hw = w.copy()
            port[t] += float(hw @ R[t]); hw = hw * (1 + R[t]); hw /= max(hw.sum(), 1e-12)
        return port
    retry = []
    chord_long = {}
    for cash in (0.0, 0.4, 0.8):
        pnl = (1 - cash) * S["EQ_DOW_CERT"] + cash * S["CASH"]
        h = horizon_stats(pnl, seed=77)
        chord_long[f"CHORD_c{int(cash*100)}"] = {"1y": h[1], "5y": h[5]}
    for nm, wd in cores.items():
        cp = core_pnl(wd)
        for cash in (0.0, 0.4, 0.8):
            pnl = (1 - cash) * cp + cash * S["CASH"]
            h = horizon_stats(pnl, seed=77)
            retry.append({"book": f"{nm}_c{int(cash*100)}", "1y": h[1], "5y": h[5]})
    dominated = True
    detail = []
    for cnm, ch in chord_long.items():
        cands = [b for b in retry if abs(b["1y"]["maxDD_p95"]) <= abs(ch["1y"]["maxDD_p95"]) + 1e-4]
        best = max(cands, key=lambda b: b["1y"]["E_ann"]) if cands else None
        ok = best is not None and best["1y"]["E_ann"] >= ch["1y"]["E_ann"] - 0.001
        dominated &= ok
        detail.append({"chord": cnm, "chord_1y": ch["1y"]["E_ann"], "chord_dd95": ch["1y"]["maxDD_p95"],
                       "best_fan": best["book"] if best else None,
                       "fan_1y": best["1y"]["E_ann"] if best else None, "dominates": bool(ok)})
        print(f"retry @{cnm:12s}: chord {ch['1y']['E_ann']:+.2%}@{ch['1y']['maxDD_p95']:.2%} vs fan "
              f"{(best['1y']['E_ann'] if best else float('nan')):+.2%} ({best['book'] if best else '-'}) -> {'OK' if ok else 'FAIL'}")

    rep = {"experiment": "P-2/E-31 backward extension to 2000 + long-history simulation + multi-asset retry",
           "panel_meta": meta,
           "survivorship_note": "the 29-name list is the MODERN panel's — pre-2010 use inherits survivor bias "
                                 "(favors equity books; every number here carries it)",
           "honesty_band": "long-run equity safety is NOT a theorem: Pastor-Stambaugh 2012 (parameter "
                            "uncertainty), Anarkulova 2022 (~12% 30y real-loss ex-survivorship)",
           "horizon_curves": curves, "calibration_5y_check": calib,
           "multiasset_retry_long_base": {"chord": chord_long, "fans": retry,
                                           "detail": detail, "fans_dominate": bool(dominated)},
           "verdict": None}
    infl_lift = curves["CERT_c40"][10]["P(>inflation_3%)"] - curves["CERT_c40"][1]["P(>inflation_3%)"]
    rep["verdict"] = (f"HORIZON is the big lever: P(beat inflation) on the certified+40%cash book rises "
                      f"{curves['CERT_c40'][1]['P(>inflation_3%)']:.0%} (1y) -> "
                      f"{curves['CERT_c40'][10]['P(>inflation_3%)']:.0%} (10y). Multi-asset retry on the long "
                      f"base: {'fans DOMINATE - P-1 reverses' if dominated else 'the chord still wins - P-1 kill CONFIRMED on 26y'}")
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nVERDICT:", rep["verdict"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
