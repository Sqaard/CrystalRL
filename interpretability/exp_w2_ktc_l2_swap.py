"""W2 (part 2) — KT-C: swap the L2 representation into the certified rule + the L2 card.

Per business/CRYSTAL_WORLD_METHODOLOGY.md KT-C, pre-registered:
  * Build a DAILY L2-derived belief: for each day t, the trailing 21d window ending at t ->
    the W1 block-feature recipe -> L2 train-frozen standardization + PCA-8 -> a 2-component
    GaussianMixture fit on TRAIN (2010-18) daily projections; bear component = the one whose
    mean has the more negative block-return loading; P(bear)_L2(t) = its posterior. (Overlap is
    fine at INFERENCE — the certified HMM belief is likewise daily-updated; the overlap artifact
    concerned train/eval pairing, not state production.)
  * KT-C SWAP BAR: through the CL-1c exposure-matched-twin machinery (identical derivation
    protocol, dev-derived (tau, e_def) per belief), hold twin z of the L2-swapped rule must be
    >= twin z of the certified 4-eye HMM belief MINUS 0.5. Any claimed IMPROVEMENT would need
    the full v12 gate (not claimed here). OOS reported as confirmation.
  * THE L2 CARD (Joseph's battery, compact): corr(P(bear)_L2, P(bear)_HMM); corr with VIX and
    EWMA vol (the re-encoding question asked openly); PCA-dim loadings named; GMM bear-component
    profile named; contrast-write probe on the swapped rule (P=0.2 vs 0.8 at matched states must
    drop exposure — F of the rule form, boundary-aware per BH1 stage 3).

Run: python interpretability/exp_w2_ktc_l2_swap.py     (~3 min)
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
from interpretability.exp_w1_ktb_hierarchy import block_features  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

OUT = HERE / "exp_w2_ktc_l2_swap_report.json"
B = 21
D = 8
FEAT_NAMES = ["ret_sum", "real_vol", "max_dd", "dVIX", "VIX", "trend", "turb", "y10", "vrp", "ebp"]


def daily_l2_belief(r, obs):
    """Trailing-21d window -> block features -> train-frozen scaler+PCA -> GMM posterior."""
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
                     float(lvl["turbulence"]), float(lvl["10Y_Yield"]), float(lvl["vrp"]), float(lvl["ebp"])])
        keep.append(t)
    X = np.array(rows, dtype=np.float32)
    dates = idx[keep]
    m_tr = np.asarray(dates <= pd.Timestamp(TRAIN[1]))
    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-9
    Z = (X - mu) / sd
    pca = PCA(n_components=D, random_state=0).fit(Z[m_tr])
    S = pca.transform(Z)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0, n_init=3).fit(S[m_tr])
    # bear = component whose mean maps to more negative ret_sum in feature space
    comp_feat = pca.inverse_transform(gmm.means_) * sd + mu
    bear = int(np.argmin(comp_feat[:, 0]))
    p = gmm.predict_proba(S)[:, bear]
    bel = pd.Series(p, index=dates).reindex(r.index).ffill().fillna(0.5)
    return bel, {"pca": pca, "gmm": gmm, "mu": mu, "sd": sd, "bear_comp_features":
                 {n: round(float(v), 2) for n, v in zip(FEAT_NAMES, comp_feat[bear])},
                 "bull_comp_features": {n: round(float(v), 2) for n, v in zip(FEAT_NAMES, comp_feat[1 - bear])},
                 "pca_dim1_loadings": {n: round(float(v), 2) for n, v in zip(FEAT_NAMES, pca.components_[0])}}


def contrast_write_rule(tau, e_def):
    """The rule form's boundary-aware F: the write pair must SPAN the fitted threshold (the BH1
    stage-3 lesson, relearned here: a fixed 0.2/0.8 pair is blind when dev picks tau=0.9)."""
    lo_w, hi_w = max(0.0, tau - 0.1), min(1.0, tau + 0.1)
    lo = e_def if lo_w > tau else 1.0
    hi = e_def if hi_w > tau else 1.0
    return 1.0 if hi < lo else 0.0


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W2/KT-C — L2 swap into the certified rule + the L2 card ===")
    r, obs, rf = load_v2()
    bel_hmm, _ = fit_belief(obs[MACRO4], TRAIN[1]), None
    bel_hmm = bel_hmm[0] if isinstance(bel_hmm, tuple) else bel_hmm
    bel_l2, card = daily_l2_belief(r, obs)

    res_hmm = derive_and_read(r, rf, bel_hmm, "hmm_eyes4")
    res_l2 = derive_and_read(r, rf, bel_l2, "l2_gmm")
    zh, zl = res_hmm["hold"]["twin_z"], res_l2["hold"]["twin_z"]
    print(f"  HMM eyes4: tau {res_hmm['tau']} e {res_hmm['e_def']} hold twin z {zh} oos {res_hmm['oos']['twin_z']}")
    print(f"  L2 GMM   : tau {res_l2['tau']} e {res_l2['e_def']} hold twin z {zl} oos {res_l2['oos']['twin_z']}")

    ktc_pass = zl >= zh - 0.5
    # the card
    both = pd.concat([bel_l2.rename("l2"), bel_hmm.rename("hmm")], axis=1).dropna()
    sig_ewma = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean())
    aligned = pd.concat([bel_l2.rename("l2"), obs["VIX"].rename("vix"), sig_ewma.rename("evol")], axis=1).dropna()
    card_out = {"corr_with_hmm_belief": round(float(both.corr().iloc[0, 1]), 3),
                "corr_with_VIX": round(float(aligned[["l2", "vix"]].corr().iloc[0, 1]), 3),
                "corr_with_ewma_vol": round(float(aligned[["l2", "evol"]].corr().iloc[0, 1]), 3),
                "bear_component_profile": card["bear_comp_features"],
                "bull_component_profile": card["bull_comp_features"],
                "pca_dim1_loadings": card["pca_dim1_loadings"],
                "rule_contrast_write_F": contrast_write_rule(res_l2["tau"], res_l2["e_def"])}
    print(f"  card: corr(HMM) {card_out['corr_with_hmm_belief']} corr(VIX) {card_out['corr_with_VIX']} "
          f"corr(evol) {card_out['corr_with_ewma_vol']} | rule F {card_out['rule_contrast_write_F']}")
    print(f"  bear state: {card_out['bear_component_profile']}")

    verdict = ((f"KT-C PASS — the L2-swapped rule holds twin z {zl} vs HMM {zh} (bar: >= {zh}-0.5): the "
                "learned month-level representation is a viable substitute command surface; no "
                "improvement claimed (that would need the v12 gate).") if ktc_pass else
               (f"KT-C FAIL — L2 swap degrades the certified rule (twin z {zl} vs {zh}, bar -0.5): the "
                "L2 representation does not yet carry the belief's risk-timing content; W3 must "
                "improve L2 before any memory work builds on it."))
    rep = {"preregistration": {"bar": "hold twin z(L2 swap) >= twin z(HMM) - 0.5; improvement claims need v12",
                                "belief_construction": "trailing 21d block -> train-frozen PCA-8 -> train-fit GMM-2 posterior"},
           "hmm": res_hmm, "l2": res_l2, "ktc_pass": bool(ktc_pass), "l2_card": card_out,
           "verdict": verdict, "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
