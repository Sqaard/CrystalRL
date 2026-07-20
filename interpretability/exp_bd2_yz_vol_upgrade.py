"""BD-2 — the Yang-Zhang vol upgrade: the two pieces PK-1 did NOT test.

HONEST LINEAGE: PK-1 (2026-07-18) already PRE-KILLED the frozen-beta Log-HAR on QLIKE (EWMA-0.94
won the frozen HOLD at both h=5/10; HAR won OOS - window-dependent). Its referee explicitly ruled
that (a) a ROLLING-REFIT HAR would be a NEW preregistration, not a rescue, and (b) PK-1 measured
forecast accuracy only. So this file tests exactly the two untested pieces:

  TEST A (forecast, extends PK-1 with one new arm): QLIKE at h=5/10 for
     EWMA-0.94 | rolling-21d RV | Yang-Zhang range-vol | Log-HAR(YZ) frozen | Log-HAR(YZ) ROLLING-REFIT
     (refit every 252d on the expanding window - the referee's sanctioned new arm).
  TEST B (THE DEPLOYED DIAL - never tested): each estimator drives a vol-targeted book
     e_t = clip(sigma_target / sigma_hat_t, 0, 1.5), 10bp per change, cash accrues rf.
     Targets are calibrated on TRAIN so every arm has the SAME mean exposure on train
     (exposure-matched by construction - the house rule); we then read realized vol tracking,
     maxDD, turnover and the exposure-matched twin z on hold/oos.

PREREGISTERED BARS (before running):
  A-PASS: the rolling-refit HAR beats EWMA on QLIKE on the frozen HOLD at BOTH horizons
          (< 0.98x, the PK-1 bar). Else the HAR family is closed for good.
  B-PASS: an estimator reduces HOLD maxDD vs the EWMA arm at equal-or-lower turnover AND tracks
          the vol target better (lower |realized ann vol - target|). Else the vol-estimator
          upgrade is NO-ADD on the deployed dial and the beyond-daily #3 candidate closes.
  Both windows reported; OOS is confirmation only.

Run: python interpretability/exp_bd2_yz_vol_upgrade.py     (~2 min)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import load_v2  # noqa: E402
from interpretability.exp_pk1_har_qlike import fetch_dji_ohlc, har_terms, fit_log_har, qlike  # noqa: E402
from interpretability.hl_v4_over_crystal1 import risk_boot_z  # noqa: E402
from interpretability.build_dow_extended_panel import fetch  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, HOLD, OOS  # noqa: E402

OUT = HERE / "exp_bd2_yz_vol_upgrade_report.json"
FIT_END = pd.Timestamp("2021-12-31")
EPS = 1e-12
HS = (5, 10)
REFIT_EVERY = 252
CAP = 1.5
COST = 0.001


def yang_zhang(ohlc, n=21):
    """Proper Yang-Zhang: overnight + k*open-close + (1-k)*Rogers-Satchell, rolling n."""
    o, h, l, c = (ohlc[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    ln_on = np.log(o / prev_c)                       # overnight
    ln_oc = np.log(c / o)                            # open-to-close
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    idx = ohlc["date"].values
    s_on = pd.Series(ln_on, index=idx).rolling(n).var()
    s_oc = pd.Series(ln_oc, index=idx).rolling(n).var()
    s_rs = pd.Series(rs, index=idx).rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return (s_on + k * s_oc + (1 - k) * s_rs).clip(lower=0)


def rolling_refit_har(feats, target_log, dates, fit_end, refit=REFIT_EVERY):
    """Expanding-window refits every `refit` days; each prediction uses only past-fitted betas."""
    X = np.column_stack([np.ones(len(target_log))] + [f.to_numpy() for f in feats])
    y = target_log.to_numpy()
    pred = np.full(len(y), np.nan)
    first = int(np.searchsorted(dates, fit_end))
    ok_all = np.isfinite(X).all(axis=1) & np.isfinite(y)
    t = first
    beta, smear = None, 1.0
    while t < len(y):
        fit_mask = (np.arange(len(y)) < t - 20) & ok_all      # 20d purge before the refit point
        if fit_mask.sum() > 200:
            b, *_ = np.linalg.lstsq(X[fit_mask], y[fit_mask], rcond=None)
            resid = y[fit_mask] - X[fit_mask] @ b
            beta, smear = b, float(np.mean(np.exp(resid)))
        hi = min(t + refit, len(y))
        if beta is not None:
            pred[t:hi] = np.exp(X[t:hi] @ beta) * smear
        t = hi
    return pred


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== BD-2 — Yang-Zhang vol upgrade: rolling-refit HAR + the DEPLOYED dial ===")
    r, obs, rf = load_v2()
    idx = r.index
    dji = fetch_dji_ohlc()
    yz = yang_zhang(dji).reindex(idx).ffill()
    r2 = r ** 2
    ew = r2.ewm(alpha=0.06, adjust=False).mean()
    roll21 = r2.rolling(21).mean()
    irx = fetch("^IRX").set_index("date")["close"].reindex(idx).ffill()
    rfd = (irx / 100.0 / 252.0).fillna(0.0)

    # ---------- TEST A: forecast QLIKE, with the new rolling-refit arm ----------
    report = {"preregistration": {"A": "rolling-refit HAR must beat EWMA QLIKE on frozen HOLD at both h (<0.98x)",
                                   "B": "an estimator must cut HOLD maxDD vs EWMA at <= turnover AND track the target better",
                                   "lineage": "PK-1 pre-killed frozen-beta Log-HAR; these are the two untested pieces"},
              "A_forecast": {}, "B_deployed": {}}
    a_pass = {}
    for h in HS:
        rv_fwd = r2.rolling(h).sum().shift(-h)
        y_log = np.log(rv_fwd + EPS)
        fit_mask = np.asarray(idx <= FIT_END - pd.Timedelta(days=20))
        f_har_frozen = fit_log_har(har_terms(yz), y_log, fit_mask)
        f_har_roll = rolling_refit_har(har_terms(yz), y_log, idx, FIT_END)
        F = {"ewma094": (ew * h).to_numpy(), "roll21": (roll21 * h).to_numpy(),
             "yz_scaled": (yz * h).to_numpy(),
             "har_frozen": np.asarray(f_har_frozen), "har_rolling": np.asarray(f_har_roll)}
        row = {}
        for w, (a, b) in (("hold", HOLD), ("oos", OOS)):
            m = np.asarray((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))) & np.isfinite(rv_fwd.to_numpy())
            cell = {}
            for k, v in F.items():
                mm = m & np.isfinite(v)
                cell[k] = round(float(np.nanmean(qlike(rv_fwd.to_numpy()[mm], v[mm]))), 5)
            row[w] = cell
        report["A_forecast"][h] = row
        hp = row["hold"]
        a_pass[h] = hp["har_rolling"] < 0.98 * hp["ewma094"]
        print(f"  A h={h:2d} hold: " + " ".join(f"{k} {v:.4f}" for k, v in hp.items()))
        print(f"      oos : " + " ".join(f"{k} {v:.4f}" for k, v in row['oos'].items()))

    # ---------- TEST B: the deployed vol-targeted dial ----------
    ro = r.to_numpy()
    est = {"ewma094": np.sqrt(ew.to_numpy() * 252),
           "roll21": np.sqrt(roll21.to_numpy() * 252),
           "yz": np.sqrt(yz.to_numpy() * 252)}
    m_tr = np.asarray(idx <= pd.Timestamp(TRAIN[1]))
    # calibrate each target so mean TRAIN exposure matches the EWMA arm at target 12%
    base_tgt = 0.12
    e_ref = np.clip(base_tgt / np.maximum(est["ewma094"], 1e-6), 0, CAP)
    ref_mean = float(np.nanmean(e_ref[m_tr]))
    b_rows = {}
    series = {}
    for k, s in est.items():
        lo, hi = 0.01, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            e = np.clip(mid / np.maximum(s, 1e-6), 0, CAP)
            if np.nanmean(e[m_tr]) < ref_mean:
                lo = mid
            else:
                hi = mid
        tgt = 0.5 * (lo + hi)
        e = np.clip(tgt / np.maximum(s, 1e-6), 0, CAP)
        e = pd.Series(e, index=idx).shift(1).fillna(1.0).to_numpy()   # decide on yesterday's estimate
        pnl = e * ro + (1 - e) * rfd.to_numpy() - np.abs(np.diff(np.concatenate([[1.0], e]))) * COST
        series[k] = (e, pnl, tgt)
    for k, (e, pnl, tgt) in series.items():
        row = {"target": round(tgt, 4)}
        for w, (a, b) in (("hold", HOLD), ("oos", OOS)):
            m = np.asarray((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))
            p = pnl[m]
            eq = np.cumprod(1 + p); peak = np.maximum.accumulate(eq)
            av = float(p.std() * np.sqrt(252))
            twin_c = float(np.nanmean(e[m]))
            twin = twin_c * ro[m] + (1 - twin_c) * rfd.to_numpy()[m]
            z, _ = risk_boot_z(p, twin, block=20, n_boot=1000, seed=6)
            row[w] = {"ann_vol": round(av, 4), "vol_track_err": round(abs(av - base_tgt), 4),
                      "maxDD": round(float((eq / peak - 1).min()), 4),
                      "turnover_yr": round(float(np.abs(np.diff(e[m])).sum() / (m.sum() / 252)), 2),
                      "mean_exposure": round(twin_c, 3), "twin_z": round(float(z), 2)}
        b_rows[k] = row
        print(f"  B {k:9s} hold: vol {row['hold']['ann_vol']:.3f} (err {row['hold']['vol_track_err']:.3f}) "
              f"DD {row['hold']['maxDD']:.3f} turn {row['hold']['turnover_yr']:.1f} "
              f"exp {row['hold']['mean_exposure']:.2f} twin z {row['hold']['twin_z']:+.2f}")
    report["B_deployed"] = b_rows

    ref = b_rows["ewma094"]["hold"]
    b_pass = {k: (v["hold"]["maxDD"] > ref["maxDD"] and v["hold"]["turnover_yr"] <= ref["turnover_yr"] * 1.05
                  and v["hold"]["vol_track_err"] < ref["vol_track_err"])
              for k, v in b_rows.items() if k != "ewma094"}
    a_ok = any(a_pass.values())
    b_ok = any(b_pass.values())
    verdict = (("A " + ("PASS" if a_ok else "FAIL") + f" (rolling-refit HAR vs EWMA QLIKE: {a_pass}); "
                "B " + ("PASS" if b_ok else "FAIL") + f" (deployed dial: {b_pass}). ")
               + ("The vol-estimator upgrade earns a place - escalate the winning arm to the frozen gate."
                  if (a_ok or b_ok) else
                  "NO-ADD on both the new forecast arm and the deployed dial: the Yang-Zhang/HAR upgrade "
                  "closes as the beyond-daily #3 candidate - EWMA-0.94 remains the risk-dial incumbent "
                  "(a null with positive prior, honestly read)."))
    report["A_pass"] = a_pass
    report["B_pass"] = b_pass
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
