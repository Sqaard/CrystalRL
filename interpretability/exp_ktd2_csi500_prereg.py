"""KT-D v2 PREREGISTRATION — the semantics-stable enrichment operator, on a market it never saw.

W3's KT-D KILL was invalidated (channel-permutation handicap); the referee's post-hoc probe showed
the SEMANTICS-STABLE operator (schemas frozen at train-end; only per-schema forward statistics
grow) passing on Dow 2022-26. That evidence is post-hoc. THIS is the clean confirmation: the same
operator, PREREGISTERED here before running, on csi500 — which no KT-D analysis has ever touched.

PREREGISTERED PROTOCOL (fixed before running):
  * L1 = 5d blocks of the csi500 price/volume-native features (the exp_ktb3 recipe, no raw
    levels), PCA-8, TRAIN 2017-2022 / DEV 2023 (alpha tuning) / walk-forward years 2024, 2025,
    2026 (the KT-B-csi500 hold+oos span — fresh for KT-D regardless of the KT-B read, since
    KT-D measures a DIFFERENT contrast: growing vs frozen memory).
  * Memory: K=6 KMeans schemas fit ONCE on train block representations (semantics FROZEN
    forever); per-schema forward-delta statistics computed from data <= Dec-31 of Y-1 for year Y
    (GROWING arm) vs from train only (FROZEN arm). Ridge coefficients fit once on train with
    frozen-arm features, alpha tuned on DEV, then untouched.
  * READ (symmetric, the W3 criterion lesson): pooled 2024-26 paired error diff at k=1 and k=4:
      PASS iff growing beats frozen with block-boot z >= +1.28 and margin > 0 at either k;
      KILL iff z <= -1.96 and margin < 0 at both k;
      NULL otherwise (uninformative — logged as such).

Run: python interpretability/exp_ktd2_csi500_prereg.py     (~2 min)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_ktb3_csi500_prereg import load_csi, build_level, assemble, SPLITS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

OUT = HERE / "exp_ktd2_csi500_prereg_report.json"
K_SCHEMAS = 6
YEARS = (2024, 2025, 2026)
CELLS = {1: {"CTX": 2}, 4: {"CTX": 4}}
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)
TRAIN_END = pd.Timestamp(SPLITS["train"][1])


def mem_features(reps, centroids, temp, fwd):
    d = np.linalg.norm(reps[:, None, :] - centroids[None, :, :], axis=2)
    sim = np.exp(-d / temp); sim = sim / (sim.sum(1, keepdims=True) + 1e-12)
    return np.concatenate([sim, sim @ fwd], axis=1).astype(np.float32)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== KT-D v2 (PREREGISTERED) — semantics-stable enrichment on csi500 ===")
    F = load_csi()
    bl = build_level(F, 5, 8)
    results = {}
    for k, cfg in CELLS.items():
        asm = assemble(bl, cfg["CTX"], (1, 2, 4))
        S = bl["S"]
        reps = S[asm["cur_of"]]
        delta = S[asm["tgt_of"][k]] - S[asm["cur_of"]]
        dates = asm["dates"]
        y = S[asm["tgt_of"][k]]
        Xctx = S[asm["ctx_of"]].reshape(len(asm["samples"]), -1)
        m_tr_all = np.where(asm["masks"]["train"])[0]
        purged = m_tr_all[asm["tgt_dates"][k][m_tr_all] <= TRAIN_END]
        di = np.where(asm["masks"]["dev"])[0]

        m_tr_dates = np.asarray(dates <= TRAIN_END)
        km = KMeans(n_clusters=K_SCHEMAS, n_init=10, random_state=0).fit(reps[m_tr_dates])
        cents = km.cluster_centers_
        temp = float(np.median(np.linalg.norm(reps[m_tr_dates] - cents[km.labels_], axis=1))) + 1e-9

        def fwd_stats(cutoff):
            m = np.asarray(dates <= cutoff)
            lab = km.predict(reps[m])
            return np.stack([delta[m][lab == j].mean(0) if (lab == j).any() else np.zeros(delta.shape[1])
                             for j in range(K_SCHEMAS)])
        fwd_frozen = fwd_stats(TRAIN_END)
        Ffr = mem_features(reps, cents, temp, fwd_frozen)
        Xfr = np.concatenate([Xctx, Ffr], axis=1)
        best = None
        for a in ALPHAS:
            reg = Ridge(alpha=a).fit(Xfr[purged], y[purged])
            e = ((reg.predict(Xfr[di]) - y[di]) ** 2).sum(1)
            if best is None or e.mean() < best[0]:
                best = (e.mean(), a, reg)
        _, alpha, reg = best
        e_frozen = ((reg.predict(Xfr) - y) ** 2).sum(1)
        e_grow = e_frozen.copy()
        per_year = {}
        for Y in YEARS:
            fwdY = fwd_stats(pd.Timestamp(f"{Y - 1}-12-31"))
            FY = mem_features(reps, cents, temp, fwdY)
            my = np.asarray(dates.year == Y)
            if my.sum() < 4:
                continue
            e_grow[my] = ((reg.predict(np.concatenate([Xctx, FY], axis=1)[my]) - y[my]) ** 2).sum(1)
            d_arr = e_frozen[my] - e_grow[my]
            _, se = block_z(d_arr, block=max(2, min(cfg["CTX"], my.sum() // 4)), n_boot=1000, seed=7)
            per_year[Y] = {"n": int(my.sum()),
                           "margin": round(float(d_arr.mean() / (e_frozen[my].mean() + 1e-12)), 4),
                           "z": round(float(d_arr.mean() / se), 2)}
        ev = np.asarray(dates.year >= YEARS[0])
        d_pool = e_frozen[ev] - e_grow[ev]
        _, se = block_z(d_pool, block=cfg["CTX"], n_boot=1000, seed=7)
        results[k] = {"alpha": alpha, "per_year": per_year,
                      "pooled": {"margin": round(float(d_pool.mean() / (e_frozen[ev].mean() + 1e-12)), 4),
                                  "z": round(float(d_pool.mean() / se), 2), "n": int(ev.sum())}}
        print(f"  k={k}: pooled margin {results[k]['pooled']['margin']:+.4f} z {results[k]['pooled']['z']:+.2f} "
              f"| years " + " ".join(f"{Y}:{v['margin']:+.4f}(z{v['z']:+.1f})" for Y, v in per_year.items()))

    p_pass = any(results[k]["pooled"]["z"] >= 1.28 and results[k]["pooled"]["margin"] > 0 for k in CELLS)
    p_kill = all(results[k]["pooled"]["z"] <= -1.96 and results[k]["pooled"]["margin"] < 0 for k in CELLS)
    verdict = ("KT-D v2 PASS (preregistered, clean market) — semantics-stable enrichment replicates: "
               "growing per-schema statistics beat frozen memory out of market; wire gated consolidation "
               "into the loop." if p_pass else
               ("KT-D v2 KILL (preregistered) — growing memory significantly hurts on the clean market: "
                "enrichment does not transfer; the Dow post-hoc positive was market-specific or spurious."
                if p_kill else
                "KT-D v2 NULL (preregistered) — no significant effect either way on the clean market: "
                "the Dow post-hoc positive is not confirmed; enrichment remains unproven (frozen store "
                "stays the default; the honest cumulative read = one post-hoc positive, one preregistered "
                "null)."))
    rep = {"preregistration": "see module docstring (fixed before running)",
           "cells": {str(k): v for k, v in results.items()}, "verdict": verdict,
           "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
