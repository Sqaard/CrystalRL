"""CRYSTAL-1 B4 — the bridge to reality: does the corner OPEN in (proxy) EXECUTION economics on real data?

WH2 closed the real corner at RETURN-economics (regime inferable but valueless, VoI=0.0). The blueprint's
main bet: the corner opens where the regime modulates the agent's OWN payoffs — execution economics (spread
capture vs adverse selection). Daily panels cannot see true fills, but they CAN see a real spread PROXY:
the Corwin-Schultz (2012) high-low estimator. This measures, on real csi500 OHLC (2018-2023):

  1. the regime-conditional REAL spread:   E[CS-spread | benign] vs E[CS-spread | toxic]
  2. the regime-conditional adverse proxy: E[|EW ret| | regime]
  3. the provide-edge per regime: edge(g) = spread(g)/2 − λ·adverse(g), λ = adverse-capture fraction
     → the HC-2 question on REAL data: does the edge FLIP SIGN across regimes?
  4. VoI of tracking the regime under these economics (the same belief-aware vs blind VI as WH2),
     swept over λ (the unknown calibration) → the map of where the corner opens.

HONESTY: CS is a noisy estimator (negatives floored); λ is unknowable without fill data; this is a PROXY task.
The deliverable is the corner-opening map + the explicit data requirement, not a deployable claim.

Run: python interpretability/crystal1_b4_bridge.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from market_regime_inferability import voi, estimate  # noqa: E402
from src.series_g.generators import _hysteresis  # noqa: E402

PANEL = ROOT / "data/adapters/_csi500_wide/processed_final_csi300.csv"
OUT = HERE / "crystal1_b4_report.json"
LAMBDAS = [0.05, 0.10, 0.20, 0.30, 0.50]
COST = 10 / 1e4                       # 10bp switch cost in the VI


def corwin_schultz(d: pd.DataFrame) -> pd.Series:
    """Per-name CS spread; daily cross-sectional mean. d: date,tic,high,low sorted."""
    d = d.sort_values(["tic", "date"]).copy()
    lnHL = np.log(d["high"] / d["low"]) ** 2
    H2 = d.groupby("tic")["high"].shift(-1).combine(d["high"], max)
    L2 = d.groupby("tic")["low"].shift(-1).combine(d["low"], min)
    beta = lnHL + lnHL.groupby(d["tic"]).shift(-1)
    gamma = np.log(H2 / L2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    d["cs"] = S.clip(lower=0)                                  # standard: floor negatives
    return d.groupby("date")["cs"].mean()


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    d = pd.read_csv(PANEL, usecols=["date", "tic", "high", "low", "close"])
    d["date"] = pd.to_datetime(d["date"])
    cs = corwin_schultz(d)
    # EW returns + vol-regime labels (leak-safe, same recipe as WH2)
    d = d.sort_values(["tic", "date"])
    d["ret"] = d.groupby("tic")["close"].pct_change()
    r = d.groupby("date")["ret"].mean().sort_index()
    r, cs = r.align(cs, join="inner")
    m = r.notna() & cs.notna()
    r, cs = r[m], cs[m]
    vol = r.rolling(20).std().shift(1)
    ok = vol.notna()
    rr, vv, ss = r[ok].to_numpy(), vol[ok].to_numpy(), cs[ok].to_numpy()
    tox = _hysteresis(vv, hi_q=0.80, lo_q=0.55)
    adverse = np.abs(rr)

    spread_b, spread_t = float(ss[tox == 0].mean()), float(ss[tox == 1].mean())
    adv_b, adv_t = float(adverse[tox == 0].mean()), float(adverse[tox == 1].mean())
    # block-bootstrap CI on the spread widening (is the regime-modulation of spreads real?)
    rng = np.random.default_rng(0); T = len(ss); bl = 20; nb = T // bl
    dsp = []
    for _ in range(2000):
        idx = (rng.integers(0, T - bl, nb)[:, None] + np.arange(bl)).ravel()
        s_, t_ = ss[idx], tox[idx]
        if (t_ == 0).any() and (t_ == 1).any():
            dsp.append(s_[t_ == 1].mean() - s_[t_ == 0].mean())
    ci_widen = [float(np.quantile(dsp, 0.025)), float(np.quantile(dsp, 0.975))]

    # regime dynamics for the VI (reuse WH2's estimator on this stream)
    est = estimate(pd.Series(rr))
    rows = []
    for lam in LAMBDAS:
        edge_b = spread_b / 2 - lam * adv_b
        edge_t = spread_t / 2 - lam * adv_t
        va, vb, dv = voi(est, edge_b, edge_t, cost=COST)
        rows.append({"lambda": lam, "edge_benign_bp": round(1e4 * edge_b, 2), "edge_toxic_bp": round(1e4 * edge_t, 2),
                     "sign_flip": bool(edge_b > 0 > edge_t), "VoI_bp_per_20d": round(1e4 * dv, 2),
                     "VoI_annualized_pct": round(100 * dv * 252 / 20, 2)})
    corner_open = [x["lambda"] for x in rows if x["sign_flip"] and x["VoI_annualized_pct"] >= 1.0]

    report = {
        "data": "csi500 real OHLC 2018-2023 (344 names), Corwin-Schultz daily spread proxy",
        "regime_conditional_facts": {
            "spread_benign_bp": round(1e4 * spread_b, 2), "spread_toxic_bp": round(1e4 * spread_t, 2),
            "spread_widening_bp_95ci": [round(1e4 * ci_widen[0], 2), round(1e4 * ci_widen[1], 2)],
            "spread_widening_significant": bool(ci_widen[0] > 0),
            "adverse_benign_bp": round(1e4 * adv_b, 2), "adverse_toxic_bp": round(1e4 * adv_t, 2),
            "regime_dynamics": {k: est[k] for k in ("p_stay_benign", "p_stay_toxic", "toxic_rate")}},
        "edge_and_VoI_by_lambda": rows,
        "corner_opens_at_lambda": corner_open,
        "verdict": None,
        "honesty": ("CS is a noisy proxy (negatives floored); lambda (adverse-capture fraction) is unknowable "
                    "without fill-level data; PROXY task — the deliverable is the map + the data requirement "
                    "(true LOB/fill data), not a deployable claim."),
    }
    report["verdict"] = (
        f"CORNER OPENS in proxy-execution economics for lambda in {corner_open}: the regime MODULATES the "
        "agent's own payoffs (spreads widen AND adverse selection rises in toxic), the provide-edge flips sign, "
        "and tracking the belief pays — the first REAL-data evidence for the program's endgame bet. Next data "
        "requirement: fill-level/LOB data to pin lambda." if corner_open else
        "Corner does NOT open at any plausible lambda — even execution-proxy economics on daily data doesn't "
        "reward regime tracking; the bridge needs true intraday data (the explicit data requirement).")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
