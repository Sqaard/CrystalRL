"""Extend the (verified-clean) Dow substrate to 2026 — ONE source, ONE clock, overlap-verified.

The old DOW30 panel (`PPO_configurations_comparison/processed_final_fixed.csv`) ends 2023-02-28. Yahoo has
2023-03..2026-07 — data NO experiment in this project has ever touched: a genuinely fresh OOS. Per the E-14
lesson, we do NOT splice two sources: every series used by the v7/v8 substrate is REBUILT from Yahoo for the
WHOLE 2010-2026 span (prices per ticker + ^GSPC + ^VIX + ^TNX, each from one chart-API response = one clock),
then VERIFIED against the old panel on the 2010-2023 overlap (the old panel is clean per E-10c, so high overlap
correlation certifies the rebuild; the extension then inherits the same definitions by construction).

Rebuilt observables (exact panel definitions, `csi300_pipeline.py`/notebook cell 26 + FinRL):
  VIX          = ^VIX close (raw level)
  10Y_Yield    = ^TNX close, scale-matched to the panel on overlap (ratio check)
  SP500_Trend  = (^GSPC - SMA200(^GSPC)) / SMA200(^GSPC)              [trailing]
  turbulence   = trailing-252d Mahalanobis distance of the Dow names' daily return vector (FinRL formula,
                 pseudo-inverse for stability)                        [trailing]

Output: data/_dow_extended/dow_extended_panel.csv  (date, tic, adjclose, ret + the 4 macro cols per date)
Honest scope: prices + the 4 belief observables (the v7/v8 substrate). Fundamentals/GRU are NOT extended
(WRDS 2023+ not ingested; GRU quarantined anyway). WBA delisted 2025 (went private) — EW handles per-date
availability; counts reported.

Run: python interpretability/build_dow_extended_panel.py
"""
from __future__ import annotations
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
OLD = ROOT / "PPO_configurations_comparison" / "processed_final_fixed.csv"
CACHE = ROOT / "data" / "_dow_extended"; CACHE.mkdir(parents=True, exist_ok=True)
PANEL_OUT = CACHE / "dow_extended_panel.csv"
REPORT = HERE / "build_dow_extended_panel_report.json"
P1, P2 = 1246838400, 1783468800            # 2009-07-06 .. 2026-07-08 (SMA200 warmup before 2010)
INDEXES = {"^GSPC": "sp500", "^VIX": "vix", "^TNX": "tnx"}


def fetch(sym):
    f = CACHE / f"{sym.replace('^', '_idx_')}.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={P1}&period2={P2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            j = json.loads(urllib.request.urlopen(req, timeout=30).read())
            r = j["chart"]["result"][0]; q = r["indicators"]["quote"][0]
            adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
            d = pd.DataFrame({"date": pd.to_datetime(r["timestamp"], unit="s").normalize(),
                              "close": q["close"], "adjclose": adj}).dropna()
            d.to_csv(f, index=False)
            return d
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"{sym}: {e}")
            time.sleep(2)


def turbulence_series(ret_wide, win=252):
    """FinRL-style rolling Mahalanobis: d_t = (r_t - mu)' pinv(Sigma) (r_t - mu) on trailing `win` days."""
    dates = ret_wide.index
    out = np.full(len(dates), np.nan)
    X = ret_wide.to_numpy()
    for i in range(win, len(dates)):
        hist = X[i - win:i]
        ok = ~np.isnan(hist).any(axis=0)                      # names fully observed in the window
        if ok.sum() < 10 or np.isnan(X[i, ok]).any():
            continue
        h = hist[:, ok]
        mu = h.mean(axis=0)
        cov = np.cov(h, rowvar=False)
        diff = X[i, ok] - mu
        out[i] = float(diff @ np.linalg.pinv(cov) @ diff)
    return pd.Series(out, index=dates)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    tics = sorted(pd.read_csv(OLD, usecols=["tic"])["tic"].dropna().unique())
    print(f"=== extending the Dow substrate to 2026: {len(tics)} tickers + {list(INDEXES)} ===")
    px, failed = {}, []
    for t in tics:
        try:
            d = fetch(t)
            px[t] = d.set_index("date")["adjclose"]
        except Exception:
            failed.append(t)
    idx = {}
    for sym in INDEXES:
        idx[sym] = fetch(sym).set_index("date")["close"]
    print(f"fetched {len(px)}/{len(tics)} tickers (failed: {failed}); index series: "
          f"{ {s: (str(v.index.min().date()), str(v.index.max().date())) for s, v in idx.items()} }")

    # ---- macro block, whole span, trailing ----
    sp = idx["^GSPC"].sort_index()
    sma200 = sp.rolling(200).mean()
    trend = (sp - sma200) / sma200
    vix = idx["^VIX"].sort_index()
    tnx = idx["^TNX"].sort_index()
    ret_wide = pd.DataFrame(px).sort_index().pct_change()
    turb = turbulence_series(ret_wide)
    macro = pd.DataFrame({"VIX": vix, "SP500_Trend": trend, "turbulence": turb, "10Y_Yield": tnx}).sort_index()
    macro = macro[macro.index >= pd.Timestamp("2010-01-01")].ffill()

    # ---- overlap verification vs the OLD (clean) panel ----
    old = pd.read_csv(OLD, usecols=["date", "tic", "close", "VIX", "SP500_Trend", "turbulence", "10Y_Yield"])
    old["date"] = pd.to_datetime(old["date"], errors="coerce")
    old_m = old.drop_duplicates("date").set_index("date")[["VIX", "SP500_Trend", "turbulence", "10Y_Yield"]].sort_index()
    ver = {}
    for c in ("VIX", "SP500_Trend", "turbulence", "10Y_Yield"):
        m = pd.DataFrame({"new": macro[c], "old": old_m[c]}).dropna()
        ratio = float((m["old"] / m["new"]).median()) if len(m) else float("nan")
        ver[c] = {"overlap_days": len(m), "corr": round(float(m["new"].corr(m["old"])), 4),
                  "median_old_over_new": round(ratio, 4)}
        print(f"  overlap[{c:12s}] corr {ver[c]['corr']} | old/new median ratio {ver[c]['median_old_over_new']} ({len(m)}d)")
    # scale-match 10Y if the panel stored percent while ^TNX quotes the same/10 (apply the measured ratio)
    r10 = ver["10Y_Yield"]["median_old_over_new"]
    if np.isfinite(r10) and abs(r10 - 1.0) > 0.05:
        macro["10Y_Yield"] = macro["10Y_Yield"] * r10
        print(f"  10Y_Yield rescaled by the measured overlap ratio {r10}")
    # EW-return verification (prices)
    old["ret"] = old.sort_values(["tic", "date"]).groupby("tic")["close"].pct_change()
    ew_old = old.groupby("date")["ret"].mean()
    ew_new = ret_wide.mean(axis=1)
    m = pd.DataFrame({"new": ew_new, "old": ew_old}).dropna()
    ver["EW_return"] = {"overlap_days": len(m), "corr": round(float(m["new"].corr(m["old"])), 4)}
    print(f"  overlap[EW_return   ] corr {ver['EW_return']['corr']} ({len(m)}d)")

    # ---- write the long panel ----
    longp = pd.DataFrame(px).sort_index()
    longp = longp[longp.index >= pd.Timestamp("2010-01-01")]
    lp = longp.stack().rename("adjclose").reset_index()
    lp.columns = ["date", "tic", "adjclose"]
    lp = lp.merge(macro.reset_index().rename(columns={"index": "date"}), on="date", how="left")
    lp.to_csv(PANEL_OUT, index=False)
    span = (str(longp.index.min().date()), str(longp.index.max().date()))
    n_2024plus = int((longp.index >= pd.Timestamp("2024-01-01")).sum())
    print(f"wrote {PANEL_OUT.name}: {len(lp)} rows, {lp['tic'].nunique()} tics, span {span[0]}..{span[1]} "
          f"({n_2024plus} trading days in 2024+ = the fresh OOS)")

    rep = {"tickers": {"requested": len(tics), "fetched": len(px), "failed": failed},
           "span": span, "fresh_oos_days_2024plus": n_2024plus,
           "overlap_verification": ver, "panel": str(PANEL_OUT.relative_to(ROOT)),
           "scope_note": "prices + the 4 v7 belief observables; fundamentals/GRU NOT extended (out of scope)"}
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", REPORT.name)


if __name__ == "__main__":
    main()
