"""CRYSTAL-1 B3 — risk-mode on REAL daily panels: the belief drives ONLY the drawdown budget.

The config-D law (WH2) says: on daily return-economics the regime is inferable but valueless → the optimal
form is a transparent persister; the belief's legitimate REAL use is the RISK objective (drawdown), per the
alpha-era law "de-risking cuts DD, not Sharpe". B3 deploys CRYSTAL-1's L1 on real data in exactly that mode:

  L1: the structured Bayes filter trained SELF-SUPERVISED on the real market's observation stream
      (burst = |EW ret| > train-window 80th pct; params frozen after the train window — leak-safe), then run
      CAUSALLY through the OOS stream. Its learned world model is READABLE (p_stay, burst signatures printed).
  L2 (risk-mode, PRE-REGISTERED, no fitting): three named exposure modes on the EW book
      FULL 1.0 if b_toxic < 0.3 | REDUCED 0.6 if < 0.7 | DEFENSIVE 0.3 else;  position lags belief by 1 day;
      10bp cost per unit exposure change. The book itself stays equal-weight (the certified-persister form).

GATES (blueprint B3): on the frozen OOS window — maxDD improved materially vs EW buy&hold, Calmar up, Sharpe
non-degraded (>= EW − 0.05); honest incumbent comparison vs trailing-RV target-vol (risk inputs must face the
incumbent); HC-1 ablation (noise belief → the DD improvement must disappear). Battery: belief-N7 on the real
learned belief stream; L0+structure on the exposure stance.

Run: python interpretability/crystal1_b3_riskmode.py
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
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from series_g_corner_test import perm_irreversibility, shuffle_pct  # noqa: E402
from src.crystal.belief_filter import train_filter  # noqa: E402

OUT = HERE / "crystal1_b3_report.json"
COST = 0.001                       # 10bp per unit exposure change
PANELS = [
    ("Dow-29", "artifacts/action_vq/A67_joint_hidden_action_controls_fullenv_from_R6c_v1/feature_scalers_frozen/fold_2021/model_ready.csv",
     "2010-01-01", "2016-12-31", "2017-01-01", "2023-02-28"),
    ("csi500", "data/adapters/_csi500_wide/csi300_model_ready.csv",
     "2018-01-01", "2020-12-31", "2021-01-01", "2023-03-01"),
]


def ew_returns(path):
    head = pd.read_csv(path, nrows=1)
    if "daily_return" in head.columns:
        d = pd.read_csv(path, usecols=["date", "tic", "daily_return"]).rename(columns={"daily_return": "ret"})
    else:
        d = pd.read_csv(path, usecols=["date", "tic", "close"])
        d = d.sort_values(["tic", "date"]); d["ret"] = d.groupby("tic")["close"].pct_change()
    d["date"] = pd.to_datetime(d["date"])
    r = d.groupby("date")["ret"].mean().sort_index()
    return r[np.isfinite(r)]


def perf(rets):
    rets = np.asarray(rets, float)
    eq = np.cumprod(1 + rets)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = float(eq[-1] ** (252 / len(rets)) - 1)
    sh = float(np.sqrt(252) * rets.mean() / (rets.std() + 1e-12))
    return {"ann_return": round(ann, 4), "sharpe": round(sh, 3), "maxDD": round(dd, 4),
            "calmar": round(ann / abs(dd) if dd < 0 else float("inf"), 3)}


def exposure_from_belief(b):
    return 1.0 if b < 0.3 else (0.6 if b < 0.7 else 0.3)


def strategy(rets_next, exposures):
    """Position (already lagged) applied to next-day returns, minus switching costs."""
    exposures = np.asarray(exposures, float)
    costs = np.abs(np.diff(np.concatenate([[1.0], exposures]))) * COST
    return exposures * np.asarray(rets_next, float) - costs


def run_panel(name, path, ts, te, os_, oe):
    r = ew_returns(path)
    tr = r[(r.index >= ts) & (r.index <= te)]
    oo = r[(r.index >= os_) & (r.index <= oe)]
    thr = float(np.quantile(np.abs(tr), 0.80))                  # burst threshold: TRAIN only
    obs_tr = (np.abs(tr.to_numpy()) > thr).astype(int)
    # train the filter on train-window chunks (self-supervised)
    L = 60; n = len(obs_tr) // L
    seqs = obs_tr[:n * L].reshape(n, L)
    f = train_filter(seqs, K=2, A_obs=2, epochs=500, seed=0, verbose=False)
    T_l, E_l, p0 = f.numpy_params()
    tox = int(np.argmax(E_l[:, 1]))
    readable = {"p_stay": np.diag(T_l).round(3).tolist(), "E_burst": E_l[:, 1].round(3).tolist(),
                "burst_thr_daily_pct": round(100 * thr, 3)}
    # causal belief over the FULL stream with frozen params
    full = r[(r.index >= ts) & (r.index <= oe)]
    obs_full = (np.abs(full.to_numpy()) > thr).astype(int)
    b = p0.copy(); bel = np.empty(len(full))
    for i, o in enumerate(obs_full):
        bp = b @ T_l; j = bp * E_l[:, o]; b = j / max(j.sum(), 1e-12)
        bel[i] = b[tox]
    bel = pd.Series(bel, index=full.index)
    bo = bel[(bel.index >= os_) & (bel.index <= oe)]
    # align: position for day t = f(belief at t-1)
    rets_next = oo.to_numpy()[1:]
    b_lag = bo.to_numpy()[:-1]
    expo = np.array([exposure_from_belief(x) for x in b_lag])
    # incumbent: trailing-RV target-vol (target = train-window median rv20; pre-registered)
    rv20_full = full.rolling(20).std()
    tv = float(rv20_full[(rv20_full.index >= ts) & (rv20_full.index <= te)].median())
    rv_lag = rv20_full[(rv20_full.index >= os_) & (rv20_full.index <= oe)].to_numpy()[:-1]
    expo_rv = np.clip(tv / np.where(rv_lag > 1e-12, rv_lag, np.inf), 0.0, 1.0)
    expo_rv[~np.isfinite(expo_rv)] = 1.0
    # HC-1 noise ablation: same 3-mode policy on uniform-noise belief
    rng = np.random.default_rng(0)
    expo_noise = np.array([exposure_from_belief(x) for x in rng.random(len(b_lag))])

    res = {"EW_buyhold": perf(rets_next), "CRYSTAL1_belief_mode": perf(strategy(rets_next, expo)),
           "trailingRV_incumbent": perf(strategy(rets_next, expo_rv)),
           "HC1_noise_belief": perf(strategy(rets_next, expo_noise))}
    # battery on the risk-mode agent
    stance = np.array([{1.0: 0, 0.6: 1, 0.3: 2}[e] for e in expo])
    l0 = behavioral_complexity_dynamic(stance, kind="discrete", dts=(1, 2), n_null=200, n_boot=200, seed=0)
    _, _, pct = shuffle_pct(bo.to_numpy(), lambda z: perm_irreversibility(z, 3))
    dbel = np.diff(bo.to_numpy())
    _, _, pct_i = shuffle_pct(dbel, lambda z: perm_irreversibility(z, 3))
    ew, cb, nz = res["EW_buyhold"], res["CRYSTAL1_belief_mode"], res["HC1_noise_belief"]
    gates = {"maxDD_improved_20pct": bool(cb["maxDD"] > ew["maxDD"] * 0.8 if ew["maxDD"] < 0 else False),
             "calmar_up": bool(cb["calmar"] > ew["calmar"]),
             "sharpe_non_degraded": bool(cb["sharpe"] >= ew["sharpe"] - 0.05),
             "HC1_noise_loses_DD_gain": bool(nz["maxDD"] < cb["maxDD"])}
    return {"panel": name, "train": [ts, te], "oos": [os_, oe], "n_oos_days": int(len(rets_next)),
            "L1_readable_world_model": readable, "performance": res,
            "battery": {"stance_x_h_mu": l0["h_mu_range"], "stance_structure": l0["structure_present_configs"],
                         "belief_N7_pct": round(pct, 1), "belief_N7_increments_pct": round(pct_i, 1),
                         "mean_exposure": round(float(expo.mean()), 3),
                         "mode_days": {m: int((stance == i).sum()) for i, m in enumerate(("FULL", "REDUCED", "DEFENSIVE"))}},
            "gates": gates, "pass": all(gates.values())}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    out = {"policy": "CRYSTAL-1 risk-mode (pre-registered 3-mode exposure over EW book; L1 trained on train window only)",
           "panels": [run_panel(*p) for p in PANELS]}
    for p in out["panels"]:
        print(f"\n=== {p['panel']} (OOS {p['oos'][0]}..{p['oos'][1]}, {p['n_oos_days']}d) ===")
        print("  L1 world model:", p["L1_readable_world_model"])
        for k, v in p["performance"].items():
            print(f"  {k:24s} {v}")
        print("  battery:", p["battery"])
        print("  gates:", p["gates"], "-> PASS" if p["pass"] else "-> not all")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[b3] wrote {OUT.name}")


if __name__ == "__main__":
    main()
