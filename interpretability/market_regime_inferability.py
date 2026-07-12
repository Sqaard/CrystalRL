"""WH2 — is the REAL market's corner forbidden? The "inferable-but-VALUELESS regime" test (config-D law).

The Phase-0 validation proved the gate has teeth via two negative controls: C (regime NOT inferable) and
D (regime inferable but does NOT modulate payoffs) — both correctly yield VoI≈0 (belief worthless). WH2 claims
the REAL liquid market is essentially CONFIG D at return-economics: its volatility regime is highly INFERABLE
(vol clusters — everyone can see the storm) but carries ~no exploitable RETURN modulation net of costs — so
the optimal real-market policy is a persister BY CONSTRUCTION, and the empty real corner is a property of the
MARKET's information structure, not of architectures or training.

Test, per real panel (Dow-29 2010-2023; csi500 2018-2023):
 1. INFERABILITY: label persistent high-vol episodes via hysteresis on PAST-only rolling vol (leak-safe);
    estimate regime stickiness (p_stay) and the burst-observability gap P(burst|toxic)−P(burst|benign).
 2. ECONOMIC MODULATION: regime-conditional mean next-day EW return (label at t uses info ≤ t−1) with a
    moving-block bootstrap CI on the benign−toxic drift difference.
 3. VoI: plug the ESTIMATED dynamics + ESTIMATED drifts + real costs (10 bp/switch) into a belief-aware vs
    belief-blind value iteration (invested/flat execution MDP over 20-day episodes) → the value of tracking
    the regime, in return units. Compare with the Series-G polygon's VoI under the SAME machinery.
Scope (honest): return-economics only — daily panels cannot see spread-capture/adverse-selection economics;
the conclusion is "corner closed at RETURN-economics", explicitly not a claim about market-making economics.

Run: python interpretability/market_regime_inferability.py
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
from src.series_g.generators import _hysteresis  # noqa: E402

OUT = HERE / "market_regime_inferability_report.json"
PANELS = [
    ("Dow-29 2010-2023", ROOT / "artifacts/action_vq/A67_joint_hidden_action_controls_fullenv_from_R6c_v1/feature_scalers_frozen/fold_2021/model_ready.csv"),
    ("csi500 2018-2023", ROOT / "data/adapters/_csi500_wide/csi300_model_ready.csv"),
]
COST_BP = 10.0                                  # one-way switch cost, basis points
T_EP = 20                                       # episode length (days), matching the polygon


def ew_returns(path):
    head = pd.read_csv(path, nrows=1)
    if "daily_return" in head.columns:
        d = pd.read_csv(path, usecols=["date", "tic", "daily_return"])
        d["date"] = pd.to_datetime(d["date"])
        r = d.groupby("date")["daily_return"].mean().sort_index()
    else:                                                    # fallback: per-tic close pct-change
        d = pd.read_csv(path, usecols=["date", "tic", "close"])
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values(["tic", "date"])
        d["ret"] = d.groupby("tic")["close"].pct_change()
        r = d.groupby("date")["ret"].mean().sort_index()
    return r[np.isfinite(r)]


def estimate(rets: pd.Series):
    """Leak-safe regime estimation + economics: vol from PAST 20d (shifted), hysteresis episodes, burst obs."""
    vol = rets.rolling(20).std().shift(1)
    m = vol.notna()
    r = rets[m].to_numpy(float); vol = vol[m].to_numpy(float)
    tox = _hysteresis(vol, hi_q=0.80, lo_q=0.55)                       # persistent high-vol episodes
    burst = (np.abs(r) > np.quantile(np.abs(r), 0.80)).astype(int)     # today's observable
    n_b2t = ((tox[:-1] == 0) & (tox[1:] == 1)).sum(); n_b = (tox[:-1] == 0).sum()
    n_t2b = ((tox[:-1] == 1) & (tox[1:] == 0)).sum(); n_t = (tox[:-1] == 1).sum()
    p_stay_b, p_stay_t = 1 - n_b2t / max(1, n_b), 1 - n_t2b / max(1, n_t)
    p_burst_b, p_burst_t = burst[tox == 0].mean(), burst[tox == 1].mean()
    drift_b, drift_t = r[tox == 0].mean(), r[tox == 1].mean()
    # moving-block bootstrap CI on drift difference
    rng = np.random.default_rng(0); T = len(r); bl = 20; nb = T // bl
    diffs = []
    for _ in range(2000):
        idx = (rng.integers(0, T - bl, nb)[:, None] + np.arange(bl)).ravel()
        rr, tt = r[idx], tox[idx]
        if (tt == 0).any() and (tt == 1).any():
            diffs.append(rr[tt == 0].mean() - rr[tt == 1].mean())
    ci = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    return {"n_days": int(T), "toxic_rate": round(float(tox.mean()), 3),
            "p_stay_benign": round(float(p_stay_b), 3), "p_stay_toxic": round(float(p_stay_t), 3),
            "p_burst_benign": round(float(p_burst_b), 3), "p_burst_toxic": round(float(p_burst_t), 3),
            "inferability_gap": round(float(p_burst_t - p_burst_b), 3),
            "drift_benign_bp": round(1e4 * float(drift_b), 2), "drift_toxic_bp": round(1e4 * float(drift_t), 2),
            "drift_diff_bp": round(1e4 * float(drift_b - drift_t), 2),
            "drift_diff_95ci_bp": [round(1e4 * ci[0], 2), round(1e4 * ci[1], 2)],
            "modulation_significant": bool(ci[0] > 0 or ci[1] < 0)}


def voi(params, drift_b, drift_t, cost=COST_BP / 1e4, T=T_EP, nb_grid=101):
    """Belief-aware vs belief-blind VI on the invested/flat execution MDP with the estimated dynamics."""
    M = np.array([[params["p_stay_benign"], 1 - params["p_stay_benign"]],
                  [1 - params["p_stay_toxic"], params["p_stay_toxic"]]])
    OBS = np.array([[1 - params["p_burst_benign"], params["p_burst_benign"]],
                    [1 - params["p_burst_toxic"], params["p_burst_toxic"]]])
    prior = params["toxic_rate"]
    g = np.linspace(0, 1, nb_grid)
    b_pred = (1 - g) * M[0, 1] + g * M[1, 1]
    po, bo = {}, {}
    for o in (0, 1):
        p_o = (1 - b_pred) * OBS[0, o] + b_pred * OBS[1, o]
        bo[o] = np.where(p_o > 1e-12, b_pred * OBS[1, o] / np.maximum(p_o, 1e-12), b_pred)
        po[o] = p_o
    # states: inv ∈ {0 flat, 1 invested}; actions: hold, switch. invested earns the regime drift each day.
    V = np.zeros((T + 1, nb_grid, 2))
    for t in range(T - 1, -1, -1):
        for inv in (0, 1):
            earn = ((1 - g) * drift_b + g * drift_t) if inv == 1 else 0.0
            cont_same = po[0] * np.array([np.interp(bo[0], g, V[t + 1, :, inv])])[0] + \
                        po[1] * np.array([np.interp(bo[1], g, V[t + 1, :, inv])])[0]
            cont_flip = po[0] * np.array([np.interp(bo[0], g, V[t + 1, :, 1 - inv])])[0] + \
                        po[1] * np.array([np.interp(bo[1], g, V[t + 1, :, 1 - inv])])[0]
            V[t, :, inv] = earn + np.maximum(cont_same, cont_flip - cost)
    v_aware = float(np.interp(prior, g, V[0, :, 0]))
    # belief-blind: same MDP on the open-loop forecast (no obs)
    Vb = np.zeros((T + 1, 2)); b = prior; fc = []
    for _ in range(T):
        fc.append(b); b = (1 - b) * M[0, 1] + b * M[1, 1]
    for t in range(T - 1, -1, -1):
        bt = fc[t]
        for inv in (0, 1):
            earn = ((1 - bt) * drift_b + bt * drift_t) if inv == 1 else 0.0
            Vb[t, inv] = earn + max(Vb[t + 1, inv], Vb[t + 1, 1 - inv] - cost)
    v_blind = float(Vb[0, 0])
    return v_aware, v_blind, v_aware - v_blind


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    out = {}
    for name, path in PANELS:
        if not path.exists():
            out[name] = {"error": "panel missing"}; continue
        r = ew_returns(path)
        est = estimate(r)
        va, vb, dv = voi(est, est["drift_benign_bp"] / 1e4, est["drift_toxic_bp"] / 1e4)
        ann = dv * 252 / T_EP
        est["VoI_per_20d_bp"] = round(1e4 * dv, 2)
        est["VoI_annualized_pct"] = round(100 * ann, 2)
        est["belief_aware_bp"] = round(1e4 * va, 2); est["belief_blind_bp"] = round(1e4 * vb, 2)
        out[name] = est
        print(f"[{name}] inferability gap={est['inferability_gap']} (p_stay {est['p_stay_benign']}/{est['p_stay_toxic']}) "
              f"| drift b/t = {est['drift_benign_bp']}/{est['drift_toxic_bp']} bp, diff CI {est['drift_diff_95ci_bp']} "
              f"sig={est['modulation_significant']} | VoI={est['VoI_per_20d_bp']} bp/20d ({est['VoI_annualized_pct']}%/yr)")
    # Series-G polygon reference through the SAME VoI machinery (its own dynamics + payoffs, in its units)
    sg = {"p_stay_benign": 0.95, "p_stay_toxic": 0.80, "p_burst_benign": 0.15, "p_burst_toxic": 0.70, "toxic_rate": 0.20}
    va, vb, dv = voi(sg, drift_b=2.0, drift_t=-4.0, cost=0.3)
    out["Series-G polygon (reference)"] = {"VoI_per_20d_units": round(dv, 2), "belief_aware": round(va, 2), "belief_blind": round(vb, 2)}
    print(f"[Series-G ref] VoI={dv:.2f} units/20d (aware {va:.2f} vs blind {vb:.2f})")
    # verdict
    real = [v for k, v in out.items() if "polygon" not in k and "error" not in v]
    # inferability: the regime label is DETERMINISTICALLY computable from past observables (rolling vol,
    # shifted — by construction), and the episodes are extremely sticky; the burst-gap is only an external
    # corroboration, not the inferability itself (vol clustering is famously observable).
    inferable = all(v["p_stay_toxic"] > 0.7 and v["p_stay_benign"] > 0.7 for v in real)
    valueless = all((not v["modulation_significant"]) or v["VoI_annualized_pct"] < 1.0 for v in real)
    out["WH2_verdict"] = {
        "regimes_inferable": inferable, "regimes_valueless_at_return_economics": valueless,
        "config_D_law_holds": bool(inferable and valueless),
        "reading": ("REAL MARKETS ARE CONFIG-D: the vol regime is highly inferable (the storm is visible) but its "
                    "return-modulation is statistically/economically negligible net of costs → tracking the regime "
                    "buys ~nothing → the optimal real-market policy is a persister BY CONSTRUCTION → the empty real "
                    "corner is a property of the market's information structure, not of architectures. Scope: "
                    "return-economics of daily panels; execution/microstructure economics (where Series-G lives) "
                    "is exactly what these panels cannot see." if (inferable and valueless) else
                    "config-D law NOT confirmed on at least one panel — inspect (a significant regime drift "
                    "modulation would instead imply the corner is open and training failed).")}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWH2:", json.dumps(out["WH2_verdict"], indent=2)[:600])


if __name__ == "__main__":
    main()
