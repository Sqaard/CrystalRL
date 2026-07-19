"""W3 (part 2) — the semantic store v0 + KT-D: does GROWING memory beat FROZEN memory?

Per business/CRYSTAL_WORLD_METHODOLOGY.md §3 + KT-D. The mechanical, objective form of
"enrichment": the semantic store holds SCHEMAS — KMeans centroids of L1 5d-block representations
with per-schema forward statistics ("in a state like this, the next block tends to move like
that") — consolidated ANNUALLY with provenance and rollback (versioned entries; G10 gating).

KT-D DESIGN (the CL-1c separation, applied to memory): ridge coefficients are fit ONCE on TRAIN
(2010-18) with memory-as-of-2018 features and FROZEN forever; only the MEMORY CONTENT grows
(schemas recomputed from data <= Dec-31 of Y-1 for each walk-forward year Y in 2022..2026).
Growing-memory vs frozen-2018-memory arms are therefore identical in every respect except what
the store knows — isolating enrichment from refitting (the trap CL-1c documented).

Features per sample: [context PCA reps (as W1)] ⊕ [softmax similarity to the K=6 schemas]
⊕ [memory-expected next-block delta (similarity-weighted schema forward-deltas, 16 dims)].

PREREGISTERED READS (before running; the CL-1c-informed prior is NULL):
  KT-D PASS iff pooled 2022-2026 paired error diff (growing vs frozen) favors growing with
  block-bootstrap z >= 1.28 at L1 k=1 or k=4. KILL if growing <= frozen (enrichment fails on
  this substrate at this scale — informative, pre-registered as a live risk in methodology §8
  death #2). SECONDARY: memory-vs-no-memory (does schema memory help at all).
Also writes data/_crystal_world/semantic_store_v0.json: curated seed entries (named regimes,
negative knowledge, decay clocks — the human-readable world picture) + the mechanical schema
sections per consolidation year with provenance.

Run: python interpretability/exp_w3_ktd_memory.py     (~2 min)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import load_v2  # noqa: E402
from interpretability.exp_w1_ktb_v2 import build_blocks, assemble  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

OUT = HERE / "exp_w3_ktd_memory_report.json"
STORE = ROOT / "data" / "_crystal_world" / "semantic_store_v0.json"
K_SCHEMAS = 6
CELLS = {1: {"CTX": 2, "alpha_grid": (0.1, 1, 10, 100, 1000, 10000)},
         4: {"CTX": 4, "alpha_grid": (0.1, 1, 10, 100, 1000, 10000)}}
YEARS = (2022, 2023, 2024, 2025, 2026)


def build_memory(reps, rep_dates, next_delta, cutoff):
    """Schemas from data <= cutoff: centroids + similarity temperature + forward-delta means."""
    m = np.asarray(rep_dates <= pd.Timestamp(cutoff))
    X = reps[m]; D = next_delta[m]
    km = KMeans(n_clusters=K_SCHEMAS, n_init=10, random_state=0).fit(X)
    lab = km.labels_
    fwd = np.stack([D[lab == j].mean(0) if (lab == j).any() else np.zeros(D.shape[1])
                    for j in range(K_SCHEMAS)])
    temp = float(np.median(np.linalg.norm(X - km.cluster_centers_[lab], axis=1))) + 1e-9
    return {"centroids": km.cluster_centers_, "fwd": fwd, "temp": temp,
            "n": int(m.sum()), "cutoff": str(cutoff)}


def mem_features(reps, mem):
    d = np.linalg.norm(reps[:, None, :] - mem["centroids"][None, :, :], axis=2)
    sim = np.exp(-d / mem["temp"])
    sim = sim / (sim.sum(1, keepdims=True) + 1e-12)
    exp_delta = sim @ mem["fwd"]
    return np.concatenate([sim, exp_delta], axis=1).astype(np.float32)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W3 part 2 — semantic store v0 + KT-D (growing vs frozen memory) ===")
    r, obs, rf = load_v2()
    bl = build_blocks(r, obs, 5, 16)
    results, mem_entries = {}, {}
    for k, cfg in CELLS.items():
        asm = assemble(bl, cfg["CTX"], (1, 2, 4, 8))
        S = bl["S"]
        reps_cur = S[asm["cur_of"]]
        delta = S[asm["tgt_of"][k]] - S[asm["cur_of"]]
        dates = asm["dates"]
        Xctx = S[asm["ctx_of"]].reshape(len(asm["samples"]), -1)
        tr_all = np.where(asm["masks"]["train"])[0]
        purged = tr_all[asm["tgt_dates"][k][tr_all] <= pd.Timestamp(TRAIN[1])]
        di = np.where(asm["masks"]["dev"])[0]
        y = S[asm["tgt_of"][k]]

        mem18 = build_memory(reps_cur, dates, delta, f"{2018}-12-31")
        F18 = mem_features(reps_cur, mem18)

        def tune(X):
            best = None
            for a in cfg["alpha_grid"]:
                reg = Ridge(alpha=a).fit(X[purged], y[purged])
                e = ((reg.predict(X[di]) - y[di]) ** 2).sum(1)
                if best is None or e.mean() < best[0]:
                    best = (e.mean(), a, reg)
            return best[1], best[2]

        a0, reg0 = tune(Xctx)                                  # no-memory arm
        a1, reg1 = tune(np.concatenate([Xctx, F18], axis=1))   # memory arm, coefficients FROZEN
        e_nomem = ((reg0.predict(Xctx) - y) ** 2).sum(1)
        e_frozen = ((reg1.predict(np.concatenate([Xctx, F18], axis=1)) - y) ** 2).sum(1)
        e_grow = e_frozen.copy()
        per_year = {}
        for Y in YEARS:
            memY = build_memory(reps_cur, dates, delta, f"{Y - 1}-12-31")
            FY = mem_features(reps_cur, memY)
            my = np.asarray(dates.year == Y)
            if my.sum() == 0:
                continue
            e_grow[my] = ((reg1.predict(np.concatenate([Xctx, FY], axis=1)[my]) - y[my]) ** 2).sum(1)
            d_arr = e_frozen[my] - e_grow[my]
            _, se = block_z(d_arr, block=max(2, min(cfg["CTX"], my.sum() // 4)), n_boot=1000, seed=7)
            per_year[Y] = {"n": int(my.sum()),
                           "margin_grow_vs_frozen": round(float(d_arr.mean() / (e_frozen[my].mean() + 1e-12)), 4),
                           "z": round(float(d_arr.mean() / se), 2),
                           "mem_n": memY["n"]}
            if k == 1:
                mem_entries[Y] = {"n_episodes": memY["n"], "cutoff": memY["cutoff"],
                                  "centroid_norms": [round(float(np.linalg.norm(c)), 2)
                                                     for c in memY["centroids"]]}
        ev = np.asarray((dates.year >= YEARS[0]))
        d_pool = e_frozen[ev] - e_grow[ev]
        _, se = block_z(d_pool, block=cfg["CTX"], n_boot=1000, seed=7)
        z_pool = float(d_pool.mean() / se)
        m_pool = float(d_pool.mean() / (e_frozen[ev].mean() + 1e-12))
        d_mem = e_nomem[ev] - e_frozen[ev]
        _, se2 = block_z(d_mem, block=cfg["CTX"], n_boot=1000, seed=7)
        results[k] = {"alpha_nomem": a0, "alpha_mem": a1, "per_year": per_year,
                      "pooled": {"margin": round(m_pool, 4), "z": round(z_pool, 2)},
                      "memory_vs_nomemory": {"margin": round(float(d_mem.mean() / (e_nomem[ev].mean() + 1e-12)), 4),
                                              "z": round(float(d_mem.mean() / se2), 2)}}
        print(f"  k={k}: pooled grow-vs-frozen margin {m_pool:+.4f} (z {z_pool:+.2f}) | "
              f"memory-vs-none {results[k]['memory_vs_nomemory']['margin']:+.4f} "
              f"(z {results[k]['memory_vs_nomemory']['z']:+.2f})")
        for Y, row in per_year.items():
            print(f"    {Y}: n {row['n']} margin {row['margin_grow_vs_frozen']:+.4f} z {row['z']:+.2f} "
                  f"(mem episodes {row['mem_n']})")

    ktd_pass = any(results[k]["pooled"]["z"] >= 1.28 and results[k]["pooled"]["margin"] > 0 for k in CELLS)
    ktd_kill = all(results[k]["pooled"]["margin"] <= 0 for k in CELLS)
    verdict = ("KT-D PASS — growing memory beats frozen memory on walk-forward representation skill: "
               "enrichment is real on this substrate; wire the gated consolidation into the loop."
               if ktd_pass else
               ("KT-D KILL — growing memory does not beat frozen memory (the CL-1c pattern at the "
                "world-picture level, pre-registered as methodology §8 death #2): enrichment at this "
                "scale/operator adds nothing; the semantic store remains valuable as the AUDITED "
                "world picture (curated entries), but automatic annual schema consolidation is not "
                "yet earning its keep." if ktd_kill else
                "KT-D INCONCLUSIVE — mixed signs below the bar; report margins, keep the frozen store "
                "as default, revisit with a better consolidation operator."))

    # ---- semantic store v0: curated seed + mechanical schema sections ----
    store = {"version": "v0", "date": "2026-07-19",
             "governance": "appends only with provenance; every entry reversible; consolidation gated (G10)",
             "curated": [
                 {"type": "named_regime", "name": "bear_month_state",
                  "content": "elevated VIX (~21), rising dVIX (+0.4), turbulence ~56, in-block dd ~-5%",
                  "provenance": "exp_w2_ktc_l2_swap L2 card (GMM bear component)", "evidence": "Strong-but-Contextual"},
                 {"type": "mechanism", "name": "voi_zero_daily",
                  "content": "daily price-level prediction carries ~zero deployable info; certified value is risk-shaped at the regime level (10-20d)",
                  "provenance": "E-15/E-16, LIT-1, CL-1c, W0 KT-A", "evidence": "Established (in-house, multi-confirmed)"},
                 {"type": "negative_knowledge", "name": "continual_lt_frozen",
                  "content": "naive continual refitting underperforms the frozen champion (belief AND tails AND config re-derivation)",
                  "provenance": "CL-1c, CL-2; CLS theory retro-justifies", "evidence": "Established (in-house)"},
                 {"type": "decay_clock", "name": "csi500_reversal",
                  "content": "csi500 5d reversal replicated at HALF magnitude on fresh 2024-26 data and is priced out net of costs",
                  "provenance": "E-14b/E-15b", "evidence": "Established (in-house, untouched-data replication)"},
                 {"type": "mechanism", "name": "hierarchy_horizon",
                  "content": "week-level structure predictable to ~40d, month-level to ~84d (strong nulls, placebo-guarded); part representational (pooled>flat)",
                  "provenance": "W1 KT-B v2 (third-window prereg pending)", "evidence": "Strong-but-Contextual"}],
             "schemas_by_year": mem_entries,
             "ktd_status": verdict.split(" — ")[0]}
    STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")

    rep = {"preregistration": {"ktd_pass": "pooled 2022-26 grow-vs-frozen z >= 1.28 at k=1 or k=4",
                                "design": "coefficients frozen at train/mem-2018; only memory content grows",
                                "prior": "NULL per CL-1c"},
           "cells": results, "verdict": verdict,
           "semantic_store": str(STORE.relative_to(ROOT)), "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name, "and", STORE.name)


if __name__ == "__main__":
    main()
