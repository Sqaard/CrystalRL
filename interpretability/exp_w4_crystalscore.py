"""W4 — Joseph's interpretability battery on the CRYSTAL-WORLD representations + the frontier.

Per business/CRYSTAL_WORLD_METHODOLOGY.md W4. The surviving representation stack after W0-W3:
L1 = PCA-16 of 5d-block features (linear by the W0 amendment); the command surface = the
certified HMM belief (L2 substitute discarded, W3). The battery:

  * NAMING: each retained dim auto-named from its top loadings (the born-legible discipline);
  * SIMULATABILITY S: can a SPARSE story (top-3 features per dim) reproduce the representation?
    S = explained-variance-weighted mean R^2 of each dim vs its 3-feature linear story, measured
    OUT of the naming sample (dev+hold+oos);
  * STABILITY St: subspace overlap (mean squared cosine of principal angles) of the top-8 PCA
    dims fitted on the two train halves (2010-14 vs 2015-18) — does the representation exist, or
    is it sample noise?
  * FAITHFULNESS F: the L1->prediction pathway is LINEAR, hence born-faithful (a write of +d on a
    named dim moves the prediction by the corresponding ridge coefficients, monotone by
    construction) — reported as F=1.0 WITH the honest note that this is a property of choosing
    linearity, not an achievement; the certified rule's boundary-aware F=1.0 carries from W2.
  * CRYSTALSCORE = F x S x St for the representation; and THE FRONTIER: KT-A-protocol quality
    (k=1 hold margin vs the strong null family, dev-tuned, the exp_w1_ktb_v2 cell machinery
    verbatim) at D in {2, 4, 8, 16} — quality vs parsimony, the program's legibility curve.
  * TAIL CLOSED: the KT-B v2 flat-twin caveat — the pooled-vs-flat comparison at L2 k=4 gets its
    missing paired block-bootstrap z here.

Run: python interpretability/exp_w4_crystalscore.py     (~3 min)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import load_v2  # noqa: E402
from interpretability.exp_w1_ktb_hierarchy import block_features  # noqa: E402
from interpretability.exp_w1_ktb_v2 import build_blocks, assemble, cell  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression

OUT = HERE / "exp_w4_crystalscore_report.json"
FEATS = ["ret_sum", "real_vol", "max_dd", "dVIX", "VIX", "trend", "turb", "y10", "vrp", "ebp"]
D_FRONTIER = (2, 4, 8, 10)     # 10 = the full feature space (the true L1 dimensionality)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W4 — interpretability battery + CrystalScore frontier ===")
    r, obs, rf = load_v2()

    # raw standardized block features (5d), train stats
    rows, b_end, idx = block_features(r, obs, 5)
    valid = [i for i, x in enumerate(rows) if x is not None]
    X = np.array([rows[i] for i in valid], dtype=np.float32)
    dates = idx[b_end[valid]]
    m_tr = np.asarray(dates <= pd.Timestamp(TRAIN[1]))
    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-9
    Z = (X - mu) / sd
    # HONEST NOTE surfaced by this very script: the 5d block-feature space has only 10 features,
    # and build_blocks silently capped D=16 to 10 all week — "L1 PCA-16" was always PCA-10
    # (a full rotation). Recorded in the results doc; nothing changes numerically upstream.
    pca = PCA(n_components=min(16, Z.shape[1]), random_state=0).fit(Z[m_tr])
    S_rep = pca.transform(Z)
    evr = pca.explained_variance_ratio_

    # ---- NAMING + SIMULATABILITY (top-8 dims) ----
    m_eval = ~m_tr
    names, r2s = {}, []
    for i in range(8):
        load = pca.components_[i]
        top = np.argsort(-np.abs(load))[:3]
        names[f"dim{i+1}"] = " ".join(f"{'+' if load[j] > 0 else '-'}{FEATS[j]}" for j in top)
        reg = LinearRegression().fit(Z[m_tr][:, top], S_rep[m_tr, i])
        pred = reg.predict(Z[m_eval][:, top])
        ss_res = ((S_rep[m_eval, i] - pred) ** 2).sum()
        ss_tot = ((S_rep[m_eval, i] - S_rep[m_eval, i].mean()) ** 2).sum()
        r2s.append(max(0.0, 1 - ss_res / (ss_tot + 1e-12)))
    w = evr[:8] / evr[:8].sum()
    S_score = float((w * np.array(r2s)).sum())
    print("  dims:", {k: v for k, v in list(names.items())[:4]}, "...")
    print(f"  simulatability S = {S_score:.3f} (per-dim R2 {[round(x, 2) for x in r2s]})")

    # ---- STABILITY: two train halves, top-8 subspace overlap ----
    half = np.asarray(dates <= pd.Timestamp("2014-12-31")) & m_tr
    half2 = m_tr & ~half
    p1 = PCA(n_components=8, random_state=0).fit(Z[half])
    p2 = PCA(n_components=8, random_state=0).fit(Z[half2])
    M = p1.components_ @ p2.components_.T
    St_score = float((np.linalg.svd(M, compute_uv=False) ** 2).mean())
    print(f"  stability St (subspace overlap, train halves) = {St_score:.3f}")

    F_score = 1.0  # linear pathway: born-faithful by construction (see docstring honesty note)
    crystal = F_score * S_score * St_score
    print(f"  CrystalScore(L1 representation) = F {F_score} x S {S_score:.3f} x St {St_score:.3f} = {crystal:.3f}")

    # ---- FRONTIER: KT-A-protocol quality at D in {2,4,8,16} (identified protocol verbatim) ----
    frontier = {}
    for D in D_FRONTIER:
        bl = build_blocks(r, obs, 5, D)
        asm = assemble(bl, 2, (1, 2, 4, 8))
        c = cell(bl, asm, 1, boot_block=2)
        frontier[D] = {"hold_margin": c["hold"]["margin"], "z": c["hold"]["z"],
                       "null": c["null"], "oos_margin": c["oos"]["margin"]}
        print(f"  frontier D={D:2d}: hold margin {c['hold']['margin']} z {c['hold']['z']} (null {c['null']})")

    # ---- TAIL: flat-twin paired z at L2 k=4 (the KT-B v2 caveat) ----
    bl2 = build_blocks(r, obs, 21, 8)
    asm2 = assemble(bl2, 2, (1, 2, 4))
    S2 = bl2["S"]
    tr_all = np.where(asm2["masks"]["train"])[0]
    purged = tr_all[asm2["tgt_dates"][4 if 4 in asm2["tgt_of"] else 2][tr_all] <= pd.Timestamp(TRAIN[1])] \
        if 4 in asm2["tgt_of"] else tr_all
    k2 = 4 if 4 in asm2["tgt_of"] else 2
    reg_p = Ridge(alpha=1000).fit(S2[asm2["ctx_of"][purged]].reshape(len(purged), -1), S2[asm2["tgt_of"][k2][purged]])
    e_pool = ((reg_p.predict(S2[asm2["ctx_of"]].reshape(len(asm2["samples"]), -1)) - S2[asm2["tgt_of"][k2]]) ** 2).sum(1)
    # flat arm: L1 blocks spanning the same 42d lookback, PCA to same reg input dim (2*8=16)
    bl1 = build_blocks(r, obs, 5, 16)
    l1_valid = sorted(bl1["valid"]); l1_dates = bl1["dates_all"][l1_valid]
    l1_reps = bl1["S"][[bl1["remap"][i] for i in l1_valid]]
    pos = np.searchsorted(l1_dates, asm2["dates"], side="right") - 1
    keep, flat_rows = [], []
    for j, p in enumerate(pos):
        if p >= 8 - 1:
            flat_rows.append(l1_reps[p - 7:p + 1].ravel()); keep.append(j)
    keep = np.array(keep); Xf_raw = np.stack(flat_rows).astype(np.float32)
    trk = np.where(asm2["masks"]["train"][keep])[0]
    pfl = PCA(n_components=16, random_state=0).fit(Xf_raw[trk])
    Xf = pfl.transform(Xf_raw).astype(np.float32)
    reg_f = Ridge(alpha=1000).fit(Xf[trk], S2[asm2["tgt_of"][k2][keep][trk]])
    e_flat = ((reg_f.predict(Xf) - S2[asm2["tgt_of"][k2][keep]]) ** 2).sum(1)
    hold_k = asm2["masks"]["hold"][keep]
    d_arr = e_flat[hold_k] - e_pool[keep][hold_k]
    _, se = block_z(d_arr, block=2, n_boot=1000, seed=7)
    flat_z = float(d_arr.mean() / se)
    print(f"  flat-twin paired z (pooled better if >0) at L2 k={k2}: {flat_z:+.2f} (n={int(hold_k.sum())})")

    verdict = (f"W4 battery complete: L1 CrystalScore {crystal:.2f} (F 1.0 born-linear x S {S_score:.2f} "
               f"x St {St_score:.2f}); the frontier shows quality vs parsimony explicitly; the KT-B "
               f"flat-twin caveat is closed with paired z {flat_z:+.2f} "
               + ("(pooled representation advantage SIGNIFICANT)" if flat_z >= 1.28 else
                  "(pooled advantage NOT individually significant - the FULL-PASS reading of KT-B "
                  "softens to 'ordering PASS + directional representational dividend')") + ".")
    rep = {"names": names, "simulatability": {"S": round(S_score, 3), "per_dim_r2": [round(x, 3) for x in r2s]},
           "stability": {"St": round(St_score, 3), "method": "top-8 subspace overlap across train halves"},
           "faithfulness": {"F": 1.0, "note": "linear pathway born-faithful by construction; not an achievement"},
           "crystalscore_L1": round(crystal, 3),
           "frontier": {str(D): v for D, v in frontier.items()},
           "flat_twin_paired_z": {"k": k2, "z": round(flat_z, 2), "n_hold": int(hold_k.sum())},
           "verdict": verdict, "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
