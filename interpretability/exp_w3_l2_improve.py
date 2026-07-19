"""W3 (part 1) — improve L2 or discard it: the temporal fix, through the OOS-inclusive KT-C bar.

W2's diagnosis (referee-corrected): the GMM-based L2 classifies each day INDEPENDENTLY — no
temporal persistence — so it re-encodes the HMM decision noisily (R2 .63, 89% agreement) and its
independent 11% degrades OOS (twin z -1.2 vs HMM +1.0). The minimal principled fix: give L2 the
same temporal machinery the certified belief has — a Gaussian HMM with a CAUSAL FILTER (sticky
states, no look-ahead) — over the same month-block feature projections. Two variants:

  L2-tHMM      : GaussianHMM (K in {2,3} by held-out ll, fit TRAIN 2010-18) over the daily
                 trailing-21d block-feature PCA-8 projections; P(bear) = filtered posterior of
                 the state with the most negative ret_sum loading.
  L2-tHMM+L1   : same, over [PCA-8 block features ⊕ slow-pooled L1] (mean of the last 4 completed
                 5d-block L1 PCA-16 reps, causal as-of) — the methodology's s2 ≈ slow-pool(s1).

PREREGISTERED BAR (tightened exactly where W2 failed): an improved L2 PASSES iff through the
CL-1c twin harness BOTH windows hold: hold twin z >= z_HMM(hold) - 0.5 AND oos twin z >=
z_HMM(oos) - 0.5. If NO variant passes -> DISCARD L2-as-substitute (binding W2 guidance): L2
remains the KT-B representation level; the rule's command surface stays the certified HMM belief;
W3 memory work proceeds on L1 representations (it never depended on the substitute).

Run: python interpretability/exp_w3_l2_improve.py     (~4 min)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import load_v2, fit_belief, MACRO4  # noqa: E402
from interpretability.exp_cl1c_twin_corrected import derive_and_read  # noqa: E402
from interpretability.exp_w1_ktb_v2 import build_blocks  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN  # noqa: E402
from interpretability.hl_v6_crystal1_features import GaussianHMM  # noqa: E402
from sklearn.decomposition import PCA

OUT = HERE / "exp_w3_l2_improve_report.json"
B = 21
D = 8
FEATS = ["ret_sum", "real_vol", "max_dd", "dVIX", "VIX", "trend", "turb", "y10", "vrp", "ebp"]


def daily_block_proj(r, obs):
    """Daily trailing-21d block features -> train-frozen scaler + PCA-8 (as W2, refactored)."""
    idx = r.index
    rv = r.to_numpy()
    rows, keep = [], []
    for t in range(B - 1, len(idx)):
        s = t - B + 1
        blk = rv[s:t + 1]
        lvl, lvl0 = obs.iloc[t], obs.iloc[s]
        need = lvl[["VIX", "SP500_Trend", "turbulence", "10Y_Yield", "vrp", "ebp"]]
        if not (np.isfinite(blk).all() and np.isfinite(need.to_numpy(dtype=float)).all()):
            continue
        eqb = np.cumprod(1 + blk)
        rows.append([float(blk.sum()), float(blk.std()), float((eqb / np.maximum.accumulate(eqb) - 1).min()),
                     float(lvl["VIX"] - lvl0["VIX"]), float(lvl["VIX"]), float(lvl["SP500_Trend"]),
                     float(lvl["turb" "ulence"]), float(lvl["10Y_Yield"]), float(lvl["vrp"]), float(lvl["ebp"])])
        keep.append(t)
    X = np.array(rows, dtype=np.float32)
    dates = idx[keep]
    m_tr = np.asarray(dates <= pd.Timestamp(TRAIN[1]))
    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-9
    Z = (X - mu) / sd
    pca = PCA(n_components=D, random_state=0).fit(Z[m_tr])
    S = pca.transform(Z).astype(np.float32)
    # ret_sum direction in PCA space, for bear-state identification
    ret_dir = pca.components_[:, 0]
    return S, dates, m_tr, ret_dir


def pooled_l1_daily(r, obs, dates):
    """Causal slow-pool of L1: mean of the last 4 COMPLETED 5d-block PCA-16 reps, as-of daily."""
    bl = build_blocks(r, obs, 5, 16)
    S1 = bl["S"]; d1 = bl["dates_all"]
    valid = sorted(bl["valid"])
    ends = d1[valid]
    reps = S1[[bl["remap"][i] for i in valid]]
    out = np.zeros((len(dates), reps.shape[1]), dtype=np.float32)
    j = 0
    for i, d in enumerate(dates):
        while j < len(ends) and ends[j] <= d:
            j += 1
        lo = max(0, j - 4)
        out[i] = reps[lo:j].mean(0) if j > lo else 0.0
    return out


def hmm_belief_from(Sfull, dates, m_tr, ret_dir_proj):
    """GaussianHMM (K by held-out ll) on train, causal filter over all; bear = most negative
    projected ret_sum mean."""
    Z_tr = Sfull[np.asarray(m_tr)]
    cut = int(len(Z_tr) * 0.8)
    best = None
    for K in (2, 3):
        h = GaussianHMM(K); h.fit(Z_tr[:cut], seed=0)
        _, ll = h.causal_filter(Z_tr[cut:])
        if best is None or ll > best[1]:
            best = (K, ll)
    hmm = GaussianHMM(best[0]); hmm.fit(Z_tr, seed=0)
    proj = hmm.mu[:, :len(ret_dir_proj)] @ ret_dir_proj
    bear = int(np.argmin(proj))
    gamma, _ = hmm.causal_filter(Sfull)
    return pd.Series(gamma[:, bear], index=dates), best[0]


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W3 part 1 — improved L2 (temporal HMM) through the OOS-inclusive KT-C bar ===")
    r, obs, rf = load_v2()
    bel_hmm = fit_belief(obs[MACRO4], TRAIN[1])
    res_hmm = derive_and_read(r, rf, bel_hmm, "hmm_eyes4")
    zh_h, zh_o = res_hmm["hold"]["twin_z"], res_hmm["oos"]["twin_z"]
    print(f"  baseline HMM: hold twin z {zh_h} oos {zh_o}")

    S, dates, m_tr, ret_dir = daily_block_proj(r, obs)
    variants = {}
    bel1, K1 = hmm_belief_from(S, dates, m_tr, ret_dir)
    variants["L2_tHMM"] = (bel1.reindex(r.index).ffill().fillna(0.5), K1)
    L1p = pooled_l1_daily(r, obs, dates)
    Sfull = np.concatenate([S, L1p], axis=1)
    bel2, K2 = hmm_belief_from(Sfull, dates, m_tr, ret_dir)
    variants["L2_tHMM_L1pool"] = (bel2.reindex(r.index).ffill().fillna(0.5), K2)

    rows, passes = {}, {}
    for name, (bel, K) in variants.items():
        res = derive_and_read(r, rf, bel, name)
        zl_h, zl_o = res["hold"]["twin_z"], res["oos"]["twin_z"]
        ok = (zl_h >= zh_h - 0.5) and (zl_o >= zh_o - 0.5)
        corr = float(pd.concat([bel.rename("l2"), bel_hmm.rename("hmm")], axis=1).dropna().corr().iloc[0, 1])
        rows[name] = {"K": K, "result": res, "corr_with_hmm": round(corr, 3), "pass_both_windows": bool(ok)}
        passes[name] = ok
        print(f"  {name} (K={K}): hold twin z {zl_h} oos {zl_o} corr(HMM) {corr:.3f} -> {'PASS' if ok else 'FAIL'}")

    any_pass = any(passes.values())
    if any_pass:
        best = max([n for n in passes if passes[n]], key=lambda n: rows[n]["result"]["hold"]["twin_z"])
        verdict = (f"L2 IMPROVED — {best} passes the OOS-inclusive bar (hold "
                   f"{rows[best]['result']['hold']['twin_z']} vs {zh_h}, oos "
                   f"{rows[best]['result']['oos']['twin_z']} vs {zh_o}): the temporal fix cures the W2 "
                   "OOS degradation; W3 memory work may reference this L2. Improvement claims beyond "
                   "parity still need the v12 gate.")
    else:
        verdict = ("DISCARD L2-AS-SUBSTITUTE (per the binding W2 guidance) — no temporal variant passes "
                   "the OOS-inclusive bar: the certified HMM belief remains the rule's command surface; "
                   "L2 stays as the KT-B representation level only; W3 memory work proceeds on L1 "
                   "representations (never depended on the substitute). Logged honestly as the second "
                   "confirmation that the month-level substitute lacks independent OOS content.")
    rep = {"preregistration": {"bar": "hold AND oos twin z >= HMM - 0.5 (tightened where W2 failed)",
                                "variants": "temporal GaussianHMM causal filter; + slow-pooled L1",
                                "on_fail": "DISCARD substitute; keep certified HMM; L2 = KT-B level only"},
           "baseline_hmm": res_hmm, "variants": rows, "verdict": verdict,
           "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
