"""Do the pipeline's TECHNICAL INDICATORS predict the REGIME — incrementally over trailing-vol baselines?

Context: the panels (Dow 2010-23, csi500 2018-23) are preprocessed per Preprocessing/Data_preprocessing.ipynb
(pandas_ta indicators + HMM regime features + stationary transforms). The alpha-era law says nothing beat
trailing RV for FORWARD VOL; this asks the sharper, operationally-relevant question for risk-mode (B3):

  T1  regime STATE at t+5              (baseline: persistence — the regime is sticky ~0.98)
  T2  EARLY WARNING: benign->toxic within h∈{5,10} days, given benign now  (the valuable one)
  T3  EXIT: toxic->benign within h∈{5,10}, given toxic now                (B3's "unwind on clearing")

Feature blocks (market-level daily aggregates, leak-safe — indicators at t use data<=t, label at t+h uses
vol up to t+h-1):
  BASE: trailing vol20 (the label's own driver — anything must beat THIS), vol-change, EW ret 5/20d, state.
  TECH: cross-sectional mean+std of {rsi_30, macd, cci_30, dx_30, atr_rel, volume_ratio, obv_pct_change}.
  HMM : the pipeline's own regime features {Regime_1_Prob, turbulence, VIX, SP500_Trend} (macro, per-date).

Protocol: expanding walk-forward by YEAR; logistic (standardized) + HistGradientBoosting; incremental
AUC(BASE+X) - AUC(BASE) reported PER TEST-YEAR (the Simpson's/within-year guard) + pooled; verdict requires
sign-consistency across years, not one pooled number. Run: python interpretability/regime_predictability_tech.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.series_g.generators import _hysteresis  # noqa: E402

OUT = HERE / "regime_predictability_tech_report.json"
PANELS = {
    "Dow-29": ROOT / "artifacts/action_vq/A67_joint_hidden_action_controls_fullenv_from_R6c_v1/feature_scalers_frozen/fold_2021/model_ready.csv",
    "csi500": ROOT / "data/adapters/_csi500_wide/csi300_model_ready.csv",
}
TECH = ["rsi_30", "macd", "cci_30", "dx_30", "atr_rel", "volume_ratio", "obv_pct_change"]
HMM = ["Regime_1_Prob", "turbulence", "VIX", "SP500_Trend"]
HORIZONS = (5, 10)


def build_market_table(path):
    head = pd.read_csv(path, nrows=1)
    cols = list(head.columns)
    tech = [c for c in TECH if c in cols]
    hmm = [c for c in HMM if c in cols]
    use = ["date", "tic"] + (["daily_return"] if "daily_return" in cols else ["close"]) + tech + hmm
    d = pd.read_csv(path, usecols=[c for c in use if c in cols])
    d["date"] = pd.to_datetime(d["date"])
    if "daily_return" not in d.columns:
        d = d.sort_values(["tic", "date"])
        d["daily_return"] = d.groupby("tic")["close"].pct_change()
    g = d.groupby("date")
    t = pd.DataFrame({"ret": g["daily_return"].mean()})
    for c in tech:
        t[f"{c}_cs_mean"] = g[c].mean()
        t[f"{c}_cs_std"] = g[c].std()
    for c in hmm:
        t[c] = g[c].first()                      # macro columns are per-date constants
    t = t.sort_index()
    # label: trailing vol20 (PAST-only) -> hysteresis regimes (identical to WH2/B3)
    vol = t["ret"].rolling(20).std().shift(1)
    m = vol.notna()
    t = t[m].copy()
    t["vol20"] = vol[m]
    t["toxic"] = _hysteresis(t["vol20"].to_numpy(), 0.80, 0.55)
    # BASE features
    t["vol_chg5"] = t["vol20"].pct_change(5)
    t["ret5"] = t["ret"].rolling(5).sum()
    t["ret20"] = t["ret"].rolling(20).sum()
    t["state"] = t["toxic"]
    t = t.dropna()
    feats = {
        "BASE": ["vol20", "vol_chg5", "ret5", "ret20", "state"],
        "TECH": [f"{c}_cs_{s}" for c in tech for s in ("mean", "std")],
        "HMM": hmm,
    }
    return t, feats


def targets(t):
    tox = t["toxic"].to_numpy()
    out = {}
    out["T1_state_t5"] = (pd.Series(tox, index=t.index).shift(-5), np.ones(len(t), bool))
    for h in HORIZONS:
        fut_max = pd.Series(tox, index=t.index).shift(-1).rolling(h, min_periods=1).max().shift(-(h - 1))
        out[f"T2_entry_h{h}"] = (fut_max, tox == 0)                      # benign now -> toxic within h?
        fut_min = pd.Series(tox, index=t.index).shift(-1).rolling(h, min_periods=1).min().shift(-(h - 1))
        out[f"T3_exit_h{h}"] = (1 - fut_min, tox == 1)                   # toxic now -> benign within h?
    return out


def walk_forward_auc(t, X_cols, y, cond, min_train_years=3):
    d = t.copy()
    d["y"] = y
    d = d[cond & d["y"].notna()]
    if len(d) < 200:
        return {}
    years = sorted(d.index.year.unique())
    per_year = {}
    for i, yr in enumerate(years):
        tr = d[d.index.year < yr]
        te = d[d.index.year == yr]
        if len(tr) < 250 or te["y"].nunique() < 2 or len(te) < 30 or tr["y"].nunique() < 2:
            continue
        Xtr, Xte = tr[X_cols].to_numpy(float), te[X_cols].to_numpy(float)
        sc = StandardScaler().fit(Xtr)
        aucs = []
        for mdl in (LogisticRegression(max_iter=500, C=1.0),
                    HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08)):
            try:
                mdl.fit(sc.transform(Xtr), tr["y"].astype(int))
                p = mdl.predict_proba(sc.transform(Xte))[:, 1]
                aucs.append(roc_auc_score(te["y"].astype(int), p))
            except Exception:
                pass
        if aucs:
            per_year[int(yr)] = round(float(max(aucs)), 3)              # best of the two models per year
    return per_year


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    report = {}
    for name, path in PANELS.items():
        if not path.exists():
            continue
        t, feats = build_market_table(path)
        print(f"[{name}] {len(t)} days, toxic_rate={t['toxic'].mean():.3f}, tech_feats={len(feats['TECH'])}, hmm={feats['HMM']}")
        res = {}
        for tgt_name, (y, cond) in targets(t).items():
            sets = {"BASE": feats["BASE"],
                    "BASE+TECH": feats["BASE"] + feats["TECH"],
                    "BASE+TECH+HMM": feats["BASE"] + feats["TECH"] + feats["HMM"]}
            aucs = {k: walk_forward_auc(t, cols, y, cond) for k, cols in sets.items()}
            years = sorted(set(aucs["BASE"]) & set(aucs["BASE+TECH"]))
            if not years:
                continue
            d_tech = [aucs["BASE+TECH"][y2] - aucs["BASE"][y2] for y2 in years]
            d_hmm = [aucs["BASE+TECH+HMM"][y2] - aucs["BASE"][y2] for y2 in years if y2 in aucs["BASE+TECH+HMM"]]
            res[tgt_name] = {
                "per_year_BASE": aucs["BASE"], "per_year_BASE+TECH": aucs["BASE+TECH"],
                "per_year_BASE+TECH+HMM": aucs["BASE+TECH+HMM"],
                "incr_TECH_mean": round(float(np.mean(d_tech)), 3),
                "incr_TECH_years_positive": f"{sum(x > 0 for x in d_tech)}/{len(d_tech)}",
                "incr_TECH+HMM_mean": round(float(np.mean(d_hmm)), 3) if d_hmm else None,
                "incr_TECH+HMM_years_positive": f"{sum(x > 0 for x in d_hmm)}/{len(d_hmm)}" if d_hmm else None,
            }
            print(f"  {tgt_name:14s}: BASE {np.mean(list(aucs['BASE'].values())):.3f} | "
                  f"+TECH {res[tgt_name]['incr_TECH_mean']:+.3f} ({res[tgt_name]['incr_TECH_years_positive']} yrs+) | "
                  f"+TECH+HMM {res[tgt_name]['incr_TECH+HMM_mean']} ({res[tgt_name]['incr_TECH+HMM_years_positive']} yrs+)")
        report[name] = res
    # verdict: incremental only if mean>+0.02 AND positive in >=2/3 of years, on a target x market
    findings = []
    for mk, res in report.items():
        for tg, v in res.items():
            for blk in ("TECH", "TECH+HMM"):
                mkey, ykey = f"incr_{blk}_mean", f"incr_{blk}_years_positive"
                if v.get(mkey) is not None and v[mkey] > 0.02:
                    a, b = map(int, v[ykey].split("/"))
                    if b >= 3 and a / b >= 2 / 3:
                        findings.append(f"{mk}/{tg}/{blk}: +{v[mkey]} AUC ({v[ykey]} years)")
    report["verdict"] = (("INCREMENTAL SIGNAL FOUND: " + "; ".join(findings)) if findings else
                          "NO consistent incremental signal: tech/HMM features do not beat the trailing-vol "
                          "baseline for regime prediction (state, entry, or exit) with within-year consistency — "
                          "the alpha-era law extends to regime-classification framing.")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nVERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
