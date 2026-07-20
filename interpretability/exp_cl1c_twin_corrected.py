"""CL-1c — the CORRECTED main-problem test: belief VoI measured against exposure-matched twins.

The CL-1 referee proved the raw z_dsd objective degenerate for near-constant trims (dsd is
1-homogeneous: the trim size cancels; on signal-free noise ANY constant trim scores z~19.6 and
day-coverage noise alone spans z 12-18). Every CL-1 separation sat inside that band → the VRP
"PASS" was an artifact and the grid corner (tau=.5, e=.95) its systematic optimum.

THE FIX (this script): the statistic is the EXPOSURE-MATCHED TWIN z — the rule's pnl vs a
constant dial holding the SAME mean exposure over the window (twin pays no switch costs — a
conservative handicap for the rule). A constant-behavior rule scores ~0 by construction; only
being defensive on the RIGHT days beats its twin. This is the project's E-23 graduation
statistic (the certified backwardation rule graduated at twin z 2.49) applied to CL-1's question.

Protocol per variant (eyes4 | eyes4+vrp | eyes4+ebp | eyes4+both; HMM frozen fit 2010-18):
  * derive (tau, e_def) on DEV 2019-21 ONLY, objective = dev twin z, with a coverage guard
    (defensive share in [5%, 95%] — outside it the twin test is vacuous);
  * freeze; read HOLD 2022-23 twin z (primary) + OOS 2024-26 (confirmation);
  * capacity-fair placebos: 5 year-block-shuffled draws per new-eye variant (fixed integer seeds
    — the referee flagged CL-1's salted-hash seeds as irreproducible), same full pipeline;
  * positive-control anchor: the certified E-15 config (tau .66, e .74) on the eyes4 belief —
    the harness must reproduce a positive twin z there or the harness itself is broken;
  * CONTINUAL arm (repaired per the referee): annual expanding HMM refits with STATE MATCHING
    (bear = the state whose filtered path correlates best with the frozen belief over the fit
    window — fixes the relabeling break that froze CL-1's arm at exposure 1.0), and per-year
    (tau, e) re-derivation on the trailing 3 years ending Y-1 (walk-forward, no future data).

PREREGISTERED READS (before running):
  * NEW-INFO PASS iff HOLD twin z(variant) >= 2.0 AND > max of its 5 placebo draws AND
    >= twin z(eyes4) + 0.5. Else NULL.
  * CONTINUAL PASS iff HOLD and OOS twin z(continual) >= twin z(frozen eyes4) + 0.5 on both.
  * MAIN PROBLEM: solved only by a PASS with placebos clean; expected honest outcome per all
    priors: NO.

Run: python interpretability/exp_cl1c_twin_corrected.py     (~10 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import (load_v2, fit_belief, run_rule,
                                                          year_block_shuffle, window_masks,
                                                          MACRO4, TAUS, EDEFS)  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN  # noqa: E402
from interpretability.hl_v6_crystal1_features import GaussianHMM  # noqa: E402
from interpretability.hl_v4_over_crystal1 import risk_boot_z  # noqa: E402

OUT = HERE / "exp_cl1c_twin_corrected_report.json"
N_PLACEBO = 5
CERT = (0.66, 0.74)


def twin_z(ro, rfd, sig, tau, e_def, n_boot=1000, seed=6):
    pnl = run_rule(ro, rfd, sig, tau, e_def)
    # reconstruct the exposure path to get the matched constant
    ex, exs = 1.0, np.empty(len(ro))
    for t in range(len(ro)):
        if t % 10 == 0:
            ex = e_def if sig[t] > tau else 1.0
        exs[t] = ex
    c = float(exs.mean())
    twin = c * ro + (1 - c) * rfd
    z, _ = risk_boot_z(pnl, twin, block=20, n_boot=n_boot, seed=seed)
    cover = float((exs < 1.0).mean())
    return float(z), cover, c


def derive_and_read(r, rf, bel, name, dev_boot=300):
    idx = r.index
    ro, rfd, sig = r.to_numpy()[1:], rf.to_numpy()[1:], bel.to_numpy()[:-1]
    mdev, mhold, moos = (m[1:] for m in window_masks(idx))
    best = None
    for tau in TAUS:
        for e in EDEFS:
            z, cover, _ = twin_z(ro[mdev], rfd[mdev], sig[mdev], tau, e, n_boot=dev_boot, seed=6)
            if 0.05 <= cover <= 0.95 and (best is None or z > best[0]):
                best = (z, tau, e)
    if best is None:
        return {"name": name, "verdict": "no admissible config (coverage guard)"}
    zdev, tau, e = best
    out = {"name": name, "tau": tau, "e_def": e, "dev_twin_z": round(zdev, 2)}
    for wname, m in (("hold", mhold), ("oos", moos)):
        z, cover, c = twin_z(ro[m], rfd[m], sig[m], tau, e)
        out[wname] = {"twin_z": round(z, 2), "defensive_share": round(cover, 3),
                      "matched_const": round(c, 3)}
    return out


def continual_belief_matched(obs_df, frozen_bel, first_year=2019):
    """Annual expanding refits with STATE MATCHING to the frozen belief (fixes relabeling)."""
    idx = obs_df.index
    out = frozen_bel.copy()
    for y in range(first_year, idx[-1].year + 1):
        fit_end = f"{y - 1}-12-31"
        m_tr = (idx >= pd.Timestamp(TRAIN[0])) & (idx <= pd.Timestamp(fit_end))
        X_tr = obs_df[m_tr].to_numpy(dtype=float)
        mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0) + 1e-9
        Z = np.nan_to_num((obs_df.to_numpy(dtype=float) - mu) / sd, nan=0.0)
        Z_tr = Z[np.asarray(m_tr)]
        cut = int(len(Z_tr) * 0.8)
        best = None
        for K in (2, 3):
            h = GaussianHMM(K); h.fit(Z_tr[:cut], seed=0)
            _, ll = h.causal_filter(Z_tr[cut:])
            if best is None or ll > best[1]:
                best = (K, ll)
        hmm = GaussianHMM(best[0]); hmm.fit(Z_tr, seed=0)
        gamma, _ = hmm.causal_filter(Z)
        # bear = the state whose filtered path best tracks the FROZEN belief on the fit window
        fb = frozen_bel.to_numpy()[np.asarray(m_tr)]
        cors = [np.corrcoef(gamma[np.asarray(m_tr), k], fb)[0, 1] for k in range(gamma.shape[1])]
        bear = int(np.nanargmax(cors))
        my = idx.year == y
        out[my] = gamma[my, bear]
    return out


def continual_read(r, rf, obs_cols_df, frozen_bel):
    """Walk-forward: per-year config re-derived on the trailing 3y ending Y-1; twin z per window."""
    idx = r.index
    bel = continual_belief_matched(obs_cols_df, frozen_bel)
    ro, rfd, sig = r.to_numpy()[1:], rf.to_numpy()[1:], bel.to_numpy()[:-1]
    idx1 = idx[1:]
    pnl_all, twin_all = [], []
    for y in range(2022, idx1[-1].year + 1):
        dtr = np.asarray((idx1 >= pd.Timestamp(f"{y-3}-01-01")) & (idx1 <= pd.Timestamp(f"{y-1}-12-31")))
        my = np.asarray(idx1.year == y)
        best = None
        for tau in TAUS:
            for e in EDEFS:
                z, cover, _ = twin_z(ro[dtr], rfd[dtr], sig[dtr], tau, e, n_boot=200, seed=6)
                if 0.05 <= cover <= 0.95 and (best is None or z > best[0]):
                    best = (z, tau, e)
        if best is None:
            continue
        _, tau, e = best
        pnl_y = run_rule(ro[my], rfd[my], sig[my], tau, e)
        ex, exs = 1.0, np.empty(my.sum())
        sg = sig[my]
        for t in range(my.sum()):
            if t % 10 == 0:
                ex = e if sg[t] > tau else 1.0
            exs[t] = ex
        c = exs.mean()
        pnl_all.append((y, pnl_y)); twin_all.append((y, c * ro[my] + (1 - c) * rfd[my]))
    def zwin(a, b):
        pa = np.concatenate([p for yy, p in pnl_all if a <= yy <= b])
        tw = np.concatenate([t for yy, t in twin_all if a <= yy <= b])
        z, _ = risk_boot_z(pa, tw, block=20, n_boot=1000, seed=6)
        return round(float(z), 2)
    return {"hold_twin_z": zwin(2022, 2023), "oos_twin_z": zwin(2024, 2026)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== CL-1c — belief VoI vs exposure-matched twins (the corrected main-problem test) ===")
    r, obs, rf = load_v2()
    variants = {"eyes4": MACRO4, "eyes4_vrp": MACRO4 + ["vrp"],
                "eyes4_ebp": MACRO4 + ["ebp"], "eyes4_both": MACRO4 + ["vrp", "ebp"]}
    results, beliefs = {}, {}
    for name, cols in variants.items():
        bel = fit_belief(obs[cols], TRAIN[1])
        beliefs[name] = bel
        results[name] = derive_and_read(r, rf, bel, name)
        h = results[name].get("hold", {})
        print(f"  {name:12s} tau {results[name].get('tau')} e {results[name].get('e_def')} "
              f"dev z {results[name].get('dev_twin_z')} | hold twin z {h.get('twin_z')} "
              f"(cover {h.get('defensive_share')}) | oos {results[name].get('oos', {}).get('twin_z')}")

    # positive control: the certified config on eyes4 through the same twin statistic
    idx = r.index
    ro, rfd, sig = r.to_numpy()[1:], rf.to_numpy()[1:], beliefs["eyes4"].to_numpy()[:-1]
    _, mhold, moos = (m[1:] for m in window_masks(idx))
    zc_h, cov_h, _ = twin_z(ro[mhold], rfd[mhold], sig[mhold], *CERT)
    zc_o, _, _ = twin_z(ro[moos], rfd[moos], sig[moos], *CERT)
    print(f"  positive control (certified {CERT}): hold twin z {zc_h:.2f} (cover {cov_h:.2f}) oos {zc_o:.2f}")

    placebos = {}
    for name, cols in list(variants.items())[1:]:
        new_cols = [c for c in cols if c not in MACRO4]
        zs = []
        for s in range(N_PLACEBO):
            o2 = obs[cols].copy()
            for ci, c in enumerate(new_cols):
                o2[c] = year_block_shuffle(obs[c], seed=7000 + 41 * s + 13 * ci)
            belp = fit_belief(o2, TRAIN[1])
            resp = derive_and_read(r, rf, belp, f"{name}_plc{s}")
            zs.append(resp.get("hold", {}).get("twin_z", np.nan))
        placebos[name] = {"hold_twin_z_draws": zs, "max": float(np.nanmax(zs))}
        print(f"  placebo[{name}]: {zs} max {placebos[name]['max']:.2f}")

    base_z = results["eyes4"]["hold"]["twin_z"]
    new_info = {}
    for name in list(variants)[1:]:
        h = results[name].get("hold", {})
        ok = (h.get("twin_z", -9) >= 2.0 and h.get("twin_z", -9) > placebos[name]["max"]
              and h.get("twin_z", -9) >= base_z + 0.5)
        new_info[name] = "PASS" if ok else "NULL"

    continual = {}
    for name in ("eyes4", "eyes4_vrp"):
        cr = continual_read(r, rf, obs[variants[name]], beliefs[name])
        ok = (cr["hold_twin_z"] >= base_z + 0.5 and cr["oos_twin_z"] >= base_z + 0.5)
        continual[name] = {**cr, "verdict": "PASS" if ok else "NULL"}
        print(f"  continual[{name}]: hold twin z {cr['hold_twin_z']} oos {cr['oos_twin_z']} => {continual[name]['verdict']}")

    solved = any(v == "PASS" for v in new_info.values()) or any(c["verdict"] == "PASS" for c in continual.values())
    verdict = ("MAIN PROBLEM: a channel PASSES the twin-corrected test — escalate to the full v12 gate."
               if solved else
               "MAIN PROBLEM: NOT SOLVED (twin-corrected). Neither VRP nor EBP eyes nor continual "
               "refitting beats the exposure-matched constant dial by the pre-registered margins — "
               "VoI~0 on the daily substrate STANDS against the two best-credentialed literature "
               "channels and against adaptivity, now measured with a non-degenerate statistic.")
    rep = {"preregistration": {"statistic": "exposure-matched constant-dial twin z (E-23 graduation form)",
                                "new_info_pass": "hold twin z >= 2.0 AND > max placebo AND >= eyes4 + 0.5",
                                "continual_pass": "hold AND oos twin z >= eyes4 + 0.5",
                                "coverage_guard": "defensive share in [5%,95%]",
                                "context": "replaces CL-1, whose raw z_dsd objective was proven degenerate"},
           "variants": results, "positive_control_certified": {"hold_twin_z": round(zc_h, 2),
                                                                "oos_twin_z": round(zc_o, 2)},
           "placebos": placebos, "new_info_verdicts": new_info, "continual": continual,
           "main_problem_solved": bool(solved), "verdict": verdict,
           "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
