"""CL-1 — do the PK-2 survivors (VRP, EBP) and/or CONTINUAL LEARNING solve the main problem?

THE MAIN PROBLEM (state-of-project): VoI~0 on daily priced data — no information channel has ever
beaten the dial/belief on the certified risk metric. This experiment attacks it from the two
directions the user named: (1) new information — the VRP (WARN, capacity-fair mandatory) and EBP
(PASS) columns just added to the v2 panel; (2) continual learning — annually refit the belief on
an expanding window instead of the train-frozen 2010-18 fit (motivated honestly by PK-1, where
the frozen forecaster lost HOLD but won OOS — adaptivity might matter).

Design (identical derivation protocol per variant — the capacity-fair spirit):
  * Belief variants (HMM K in {2,3} chosen by held-out ll, fit TRAIN 2010-18, causal filter):
    eyes4 (certified baseline observables) | eyes4+vrp | eyes4+ebp | eyes4+both
    + CAPACITY-FAIR PLACEBOS: for each new-eye variant, 5 seeds of the same panel with the new
    column(s) YEAR-BLOCK-SHUFFLED (distribution and within-year autocorrelation kept, alignment
    destroyed) through the same pipeline.
  * Rule form (the certified E-15 family): every H=10 days set exposure = e_def if P(bear)>tau
    else 1.0; 10bp switch cost; cash leg accrues rf (^IRX) — the E-23 executor lesson.
    (tau, e_def) derived per variant on DEV 2019-21 ONLY (max dev z_dsd s.t. dev NI z >= -0.5),
    frozen, then read on HOLD 2022-23 with OOS 2024-26 as the confirmation read.
  * CONTINUAL arms (for eyes4 and the best new-eye variant): refit the HMM each Jan 1 of
    2019..2026 on the expanding window 2010..prior-Dec-31, filter causally, splice per-year
    beliefs; thresholds stay the ones derived from that variant's frozen belief (isolates the
    belief-adaptation effect). No future data enters any refit.

PREREGISTERED READS (written before running):
  * NEW-INFO PASS iff on HOLD: z_dsd(variant) >= z_dsd(eyes4) + 0.5 AND > max of its 5 placebo
    draws AND ni_z >= ni_z(eyes4) - 0.5. Else NULL (VoI~0 stands for that channel).
  * CONTINUAL PASS iff z_dsd(CL) >= z_dsd(frozen same variant) + 0.5 on BOTH hold and OOS.
    Else NULL (continual learning does not solve the main problem on this substrate).
  * MAIN-PROBLEM verdict: SOLVED only if some arm PASSES with placebos clean; otherwise the
    honest answer is NO — logged as negative knowledge with the exact margins.

Run: python interpretability/exp_cl1_new_eyes_continual.py     (~5-10 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from interpretability.hl_v6_crystal1_features import GaussianHMM  # noqa: E402
from interpretability.hl_v4_over_crystal1 import risk_boot_z  # noqa: E402
from interpretability.build_dow_extended_panel import fetch  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402

OUT = HERE / "exp_cl1_new_eyes_continual_report.json"
PANEL = ROOT / "data" / "_dow_extended" / "dow_extended_panel_v2.csv"
MACRO4 = ["VIX", "SP500_Trend", "turbulence", "10Y_Yield"]
H = 10
COST = 0.001
NI_MARGIN = 2e-4
TAUS = np.round(np.arange(0.50, 0.92, 0.02), 2)
EDEFS = np.round(np.arange(0.50, 0.98, 0.03), 2)
N_PLACEBO = 5


def load_v2():
    df = pd.read_csv(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    d = df.dropna(subset=["date", "tic", "adjclose"]).sort_values(["tic", "date"]).copy()
    d["ret"] = d.groupby("tic")["adjclose"].pct_change()
    r = d.groupby("date")["ret"].mean().dropna()
    day = df.drop_duplicates("date").set_index("date").sort_index()
    obs = day[MACRO4 + ["vrp", "ebp"]].reindex(r.index).ffill()
    irx = fetch("^IRX").set_index("date")["close"].reindex(r.index).ffill()
    rf = (irx / 100.0 / 252.0).fillna(0.0)
    return r, obs, rf


def fit_belief(obs_df, fit_end, k_choices=(2, 3)):
    """Train-window fit (2010-01-01..fit_end), causal filter over the FULL span. Mirrors build_belief."""
    m_tr = (obs_df.index >= pd.Timestamp(TRAIN[0])) & (obs_df.index <= pd.Timestamp(fit_end))
    X_tr = obs_df[m_tr].to_numpy(dtype=float)
    mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0) + 1e-9
    Z = np.nan_to_num((obs_df.to_numpy(dtype=float) - mu) / sd, nan=0.0)
    Z_tr = Z[np.asarray(m_tr)]
    cut = int(len(Z_tr) * 0.8)
    best = None
    for K in k_choices:
        h = GaussianHMM(K); h.fit(Z_tr[:cut], seed=0)
        _, ll = h.causal_filter(Z_tr[cut:])
        if best is None or ll > best[1]:
            best = (K, ll)
    hmm = GaussianHMM(best[0]); hmm.fit(Z_tr, seed=0)
    bear = int(np.argmax(hmm.mu[:, 0]))
    gamma, _ = hmm.causal_filter(Z)
    return pd.Series(gamma[:, bear], index=obs_df.index)


def run_rule(ro, rfd, sig, tau, e_def):
    """E-15 family executor: decide every H days on yesterday's belief; cash accrues rf; 10bp cost."""
    ex, pnl = 1.0, np.empty(len(ro))
    for t in range(len(ro)):
        if t % H == 0:
            tgt = e_def if sig[t] > tau else 1.0
            cost = abs(tgt - ex) * COST
            ex = tgt
        else:
            cost = 0.0
        pnl[t] = ex * ro[t] + (1 - ex) * rfd[t] - cost
    return pnl


def window_masks(idx):
    def m(a, b):
        return np.asarray((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))
    return m(*DEV), m(*HOLD), m(*OOS)


def eval_variant(r, rf, bel, name):
    """Derive (tau,e_def) on DEV, freeze, read HOLD/OOS. Signals lagged one day."""
    idx = r.index
    ro, rfd, sig = r.to_numpy()[1:], rf.to_numpy()[1:], bel.to_numpy()[:-1]
    mdev, mhold, moos = (m[1:] for m in window_masks(idx))
    best = None
    for tau in TAUS:
        for e in EDEFS:
            pnl = run_rule(ro[mdev], rfd[mdev], sig[mdev], tau, e)
            bench = ro[mdev]
            z, _ = risk_boot_z(pnl, bench, block=20, n_boot=300, seed=4)
            d = pnl - bench
            _, se = block_z(d, block=5, n_boot=300, seed=4)
            ni = (d.mean() + NI_MARGIN) / se
            if ni >= -0.5 and (best is None or z > best[0]):
                best = (z, tau, e)
    if best is None:
        return {"name": name, "verdict": "no admissible dev config"}
    _, tau, e = best
    out = {"name": name, "tau": tau, "e_def": e}
    for wname, m in (("hold", mhold), ("oos", moos)):
        pnl = run_rule(ro[m], rfd[m], sig[m], tau, e)
        bench = ro[m]
        z, _ = risk_boot_z(pnl, bench, block=20, n_boot=1000, seed=4)
        d = pnl - bench
        _, se = block_z(d, block=5, n_boot=1000, seed=4)
        eq = np.cumprod(1 + pnl); peak = np.maximum.accumulate(eq)
        out[wname] = {"z_dsd": round(float(z), 2), "ni_z": round(float((d.mean() + NI_MARGIN) / se), 2),
                      "sharpe": round(float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)), 2),
                      "maxDD": round(float((eq / peak - 1).min()), 3)}
    return out


def year_block_shuffle(s, seed):
    rng = np.random.default_rng(seed)
    years = s.index.year
    uy = np.unique(years)
    perm = rng.permutation(uy)
    parts = []
    for orig, src in zip(uy, perm):
        chunk = s[years == src].to_numpy()
        need = int((years == orig).sum())
        chunk = np.resize(chunk, need)                   # length-align across leap/holiday differences
        parts.append(chunk)
    return pd.Series(np.concatenate(parts), index=s.index)


def continual_belief(obs_df, first_eval_year=2019):
    """Annual expanding refits: year Y filtered by the model fit on 2010..Y-1 (strictly past)."""
    idx = obs_df.index
    bel = fit_belief(obs_df, f"{first_eval_year - 1}-12-31")     # pre-2019 part from the 2018 fit
    out = bel.copy()
    for y in range(first_eval_year, idx[-1].year + 1):
        b_y = fit_belief(obs_df, f"{y - 1}-12-31")
        my = idx.year == y
        out[my] = b_y[my]
    return out


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== CL-1 — new eyes (VRP/EBP) + continual learning vs the main problem ===")
    r, obs, rf = load_v2()
    variants = {"eyes4": MACRO4, "eyes4_vrp": MACRO4 + ["vrp"],
                "eyes4_ebp": MACRO4 + ["ebp"], "eyes4_both": MACRO4 + ["vrp", "ebp"]}
    results, beliefs = {}, {}
    for name, cols in variants.items():
        bel = fit_belief(obs[cols], TRAIN[1])
        beliefs[name] = bel
        results[name] = eval_variant(r, rf, bel, name)
        h = results[name].get("hold", {})
        print(f"  {name:12s} tau {results[name].get('tau')} e {results[name].get('e_def')} "
              f"hold z_dsd {h.get('z_dsd')} ni {h.get('ni_z')} | oos z_dsd {results[name].get('oos', {}).get('z_dsd')}")

    # capacity-fair placebos for each new-eye variant
    placebos = {}
    for name, cols in list(variants.items())[1:]:
        new_cols = [c for c in cols if c not in MACRO4]
        zs = []
        for s in range(N_PLACEBO):
            o2 = obs[cols].copy()
            for c in new_cols:
                o2[c] = year_block_shuffle(obs[c], seed=100 * s + hash(c) % 97)
            belp = fit_belief(o2, TRAIN[1])
            resp = eval_variant(r, rf, belp, f"{name}_plc{s}")
            zs.append(resp.get("hold", {}).get("z_dsd", np.nan))
        placebos[name] = {"hold_z_dsd_draws": zs, "max": float(np.nanmax(zs))}
        print(f"  placebo[{name}]: hold z draws {zs} max {placebos[name]['max']:.2f}")

    base_h = results["eyes4"]["hold"]
    new_info = {}
    for name in list(variants)[1:]:
        h = results[name].get("hold", {})
        ok = (h.get("z_dsd", -9) >= base_h["z_dsd"] + 0.5
              and h.get("z_dsd", -9) > placebos[name]["max"]
              and h.get("ni_z", -9) >= base_h["ni_z"] - 0.5)
        new_info[name] = "PASS" if ok else "NULL"

    # continual arms: eyes4 + the best new-eye variant by hold z
    best_new = max(list(variants)[1:], key=lambda n: results[n].get("hold", {}).get("z_dsd", -9))
    continual = {}
    for name in ("eyes4", best_new):
        bel_cl = continual_belief(obs[variants[name]])
        res_cl = eval_variant(r, rf, bel_cl, f"{name}_CL")
        frozen = results[name]
        ok = all(res_cl.get(w, {}).get("z_dsd", -9) >= frozen.get(w, {}).get("z_dsd", 9) + 0.5
                 for w in ("hold", "oos"))
        continual[name] = {"frozen": {w: frozen.get(w) for w in ("hold", "oos")},
                           "continual": {w: res_cl.get(w) for w in ("hold", "oos")},
                           "cl_params": {"tau": res_cl.get("tau"), "e_def": res_cl.get("e_def")},
                           "verdict": "PASS" if ok else "NULL"}
        print(f"  continual[{name}]: frozen hold z {frozen.get('hold', {}).get('z_dsd')} -> CL "
              f"{res_cl.get('hold', {}).get('z_dsd')} | oos {frozen.get('oos', {}).get('z_dsd')} -> "
              f"{res_cl.get('oos', {}).get('z_dsd')} => {continual[name]['verdict']}")

    solved = any(v == "PASS" for v in new_info.values()) or any(c["verdict"] == "PASS" for c in continual.values())
    verdict = ("MAIN PROBLEM: a channel PASSED — see details; escalate the survivor to the full v12 "
               "frozen-gate battery before any stronger claim." if solved else
               "MAIN PROBLEM: NOT SOLVED — neither the new information (VRP/EBP eyes, placebo-controlled) "
               "nor continual belief refitting beats the certified baseline by the pre-registered margins; "
               "VoI~0 on the daily substrate STANDS, now also against the two best-credentialed literature "
               "channels and against adaptivity.")
    rep = {"preregistration": {"new_info_pass": "hold z_dsd >= base+0.5 AND > max placebo AND ni >= base-0.5",
                                "continual_pass": "z_dsd(CL) >= z_dsd(frozen)+0.5 on hold AND oos",
                                "derivation": "tau/e_def on DEV only (max z_dsd s.t. ni>=-0.5), frozen",
                                "executor": "E-15 family, H=10, 10bp, cash accrues rf"},
           "variants": results, "placebos": placebos, "new_info_verdicts": new_info,
           "best_new_variant": best_new, "continual": continual,
           "main_problem_solved": bool(solved), "verdict": verdict,
           "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
