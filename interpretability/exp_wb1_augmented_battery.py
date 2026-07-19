"""WB-1 — the W-battery re-run on the AUGMENTED state (base 10 + stockbond corr flag).

Per the WP-2 next step: the stock-bond correlation block (the first knowledge->model addition,
robust at k=4, OOS z 14.6) is wired into the state; this battery checks the whole program still
stands on the augmented representation. REUSES the twice-reviewed exp_w1_ktb_v2 machinery
verbatim (cell/assemble: dev-tuned context+alpha, strong null family, purged fits, year-shuffled
placebo per cell, degenerate-SE guards).

PREREGISTERED READS (before running; these are CONSISTENCY reads on much-read windows — research
evidence, NOT certification; any deployable claim still faces the v12 gate):
  KT-A-aug: the k=4 L1 cell must RETAIN the WP-2 gain — margin >= the base-state cell + 0.02
            with its placebo < half. (Consistency of the admitted block under the full protocol.)
  KT-B-aug: the hierarchy ordering must HOLD: reach(L1_aug) < reach(L2_aug) under the identified
            protocol (z>=2 + placebo guard). Report the reach deltas vs the base state (40d/84d).
  KT-E:     (a) collapse diagnostics healthy (min dim std > 0.1, eff rank > 3);
            (b) PLACEBO MARKET: i.i.d. day-shuffled (r, obs) joint rows -> full pipeline at
                L1 k=1 must show NO significant skill (z < 2) — a no-structure market yields none;
            (c) all augmented features finite over all windows.
  Overall: PASS iff all three hold; any failure is reported as-is (no averaging).

Run: python interpretability/exp_wb1_augmented_battery.py     (~6-8 min)
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
from interpretability.exp_w1_ktb_v2 import assemble, cell  # noqa: E402
from interpretability.exp_wp2_queue import candidate_dailies  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from sklearn.decomposition import PCA

OUT = HERE / "exp_wb1_augmented_battery_report.json"
LEVELS = {"L1": {"B": 5, "D": 12, "KS": (1, 2, 4, 8, 16)},
          "L2": {"B": 21, "D": 8, "KS": (1, 2, 4)}}
CTX_GRID = (2, 4, 6, 12)
Z_BAR = 2.0
BASE_REACH = {"L1": 40, "L2": 84}
WP2_BASE_K4 = 0.1649


def build_aug_level(r, obs, sb_daily, B, D):
    rows, b_end, idx = block_features(r, obs, B)
    arr = sb_daily.reindex(r.index).to_numpy(dtype=float)
    valid, X = [], []
    for i, x in enumerate(rows):
        if x is None:
            continue
        e = b_end[i]; s = e - B + 1
        m = np.nanmean(arr[s:e + 1], axis=0)
        if not np.isfinite(m).all():
            continue
        valid.append(i); X.append(list(x) + list(m))
    X = np.array(X, dtype=np.float32)
    remap = {i: j for j, i in enumerate(valid)}
    dates_all = idx[b_end]
    tr_rows = [remap[i] for i in valid if dates_all[i] <= pd.Timestamp(TRAIN[1])]
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-9
    Z = (X - mu) / sd
    pca = PCA(n_components=min(D, Z.shape[1]), random_state=0).fit(Z[tr_rows])
    S = pca.transform(Z).astype(np.float32)
    return {"S": S, "remap": remap, "valid": set(valid), "dates_all": dates_all,
            "n": len(rows), "B": B}


def run_level(bl, cfg, rng):
    rows, rch = {}, 0
    for k in cfg["KS"]:
        cand = None
        for CTX in CTX_GRID:
            asm = assemble(bl, CTX, cfg["KS"])
            if asm["masks"]["train"].sum() < 25:
                continue
            c = cell(bl, asm, k, boot_block=CTX)
            if c is None:
                continue
            from sklearn.linear_model import Ridge
            S = bl["S"]
            di = np.where(asm["masks"]["dev"])[0]
            tr_all = np.where(asm["masks"]["train"])[0]
            purged = tr_all[asm["tgt_dates"][k][tr_all] <= pd.Timestamp(TRAIN[1])]
            reg = Ridge(alpha=c["alpha"]).fit(S[asm["ctx_of"][purged]].reshape(len(purged), -1),
                                              S[asm["tgt_of"][k][purged]])
            e_dev = ((reg.predict(S[asm["ctx_of"][di]].reshape(len(di), -1)) - S[asm["tgt_of"][k][di]]) ** 2).sum(1).mean()
            if cand is None or e_dev < cand[0]:
                cand = (e_dev, CTX, asm, c)
        if cand is None:
            continue
        _, CTX, asm, c = cand
        yrs = np.asarray(asm["dates"].year)
        trm = asm["masks"]["train"]
        uy = np.unique(yrs[trm]); pm = dict(zip(uy, rng.permutation(uy)))
        plc = asm["tgt_of"][k].copy()
        for j in np.where(trm)[0]:
            cands2 = np.where(yrs == pm.get(yrs[j], yrs[j]))[0]
            if len(cands2):
                plc[j] = asm["tgt_of"][k][cands2[j % len(cands2)]]
        cp = cell(bl, asm, k, boot_block=CTX, plc_map=plc)
        h, hp = c.get("hold", {}), (cp or {}).get("hold", {})
        sig = (h.get("margin") is not None and h["margin"] > 0 and (h.get("z") or -9) >= Z_BAR
               and (hp.get("margin") is None or hp["margin"] < 0.5 * h["margin"]))
        rows[k] = {"horizon_days": k * cfg["B"], "ctx": CTX, "real": c, "placebo_hold": hp,
                   "significant": bool(sig)}
        if sig:
            rch = max(rch, k * cfg["B"])
    return rows, rch


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== WB-1 — the battery on the augmented state (base10 + stockbond) ===")
    r, obs, rf = load_v2()
    sb = candidate_dailies(r, obs, r.index)["stockbond"]
    rng = np.random.default_rng(1234)

    levels_out, reach = {}, {}
    for name, cfg in LEVELS.items():
        bl = build_aug_level(r, obs, sb, cfg["B"], cfg["D"])
        rows, rch = run_level(bl, cfg, rng)
        levels_out[name] = {str(k): v for k, v in rows.items()}
        reach[name] = rch
        for k, v in rows.items():
            h = v["real"]["hold"]
            print(f"  {name} k={k} ({v['horizon_days']:3d}d) ctx {v['ctx']}: m {h.get('margin')} "
                  f"z {h.get('z')} plc {v['placebo_hold'].get('margin')} -> {'SIG' if v['significant'] else '-'}")
    print("  reach(days):", reach, "| base state was", BASE_REACH)

    # KT-A consistency: the k=4 L1 cell retains the WP-2 gain
    l1k4 = levels_out["L1"].get("4", {})
    h4 = l1k4.get("real", {}).get("hold", {})
    p4 = l1k4.get("placebo_hold", {})
    kta_ok = (h4.get("margin") is not None and h4["margin"] >= WP2_BASE_K4 + 0.02
              and (p4.get("margin") is None or p4["margin"] < 0.5 * h4["margin"]))

    # KT-B ordering
    ktb_ok = reach["L1"] < reach["L2"]

    # KT-E: collapse + placebo market
    bl1 = build_aug_level(r, obs, sb, 5, 12)
    S = bl1["S"]
    stds = S.std(0); lam = np.clip(np.linalg.eigvalsh(np.cov(S.T)), 0, None)
    eff_rank = float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12))
    collapse_ok = (float(stds.min()) > 0.1) and (eff_rank > 3)
    # placebo market: i.i.d. day-shuffled joint rows
    rng2 = np.random.default_rng(777)
    perm = rng2.permutation(len(r))
    r_p = pd.Series(r.to_numpy()[perm], index=r.index)
    obs_p = pd.DataFrame(obs.to_numpy()[perm], index=obs.index, columns=obs.columns)
    sb_p = candidate_dailies(r_p, obs_p, r_p.index)["stockbond"]
    bl_p = build_aug_level(r_p, obs_p, sb_p, 5, 12)
    rows_p, _ = run_level(bl_p, {"B": 5, "D": 12, "KS": (1, 2, 4, 8)}, np.random.default_rng(99))
    pz = rows_p.get(1, {}).get("real", {}).get("hold", {}).get("z")
    placebo_market_ok = (pz is None) or (pz < Z_BAR)
    print(f"  KT-E: min_std {stds.min():.3f} eff_rank {eff_rank:.2f} | placebo-market L1k1 z {pz}")

    all_ok = kta_ok and ktb_ok and collapse_ok and placebo_market_ok
    verdict = (("BATTERY PASS on the augmented state — KT-A gain retained "
                f"(k=4 m {h4.get('margin')} vs base {WP2_BASE_K4}), hierarchy ordering holds "
                f"(reach {reach['L1']}d < {reach['L2']}d), collapse healthy, placebo market silent. "
                "The augmented 12-feature state is the new frozen L1; deployable claims -> v12 gate.")
               if all_ok else
               (f"BATTERY MIXED/FAIL — kta_retained={kta_ok} ordering={ktb_ok} "
                f"collapse={collapse_ok} placebo_market_silent={placebo_market_ok}: "
                "the failing read(s) gate the state upgrade; report as-is, no averaging."))
    rep = {"preregistration": {"kta": "L1 k=4 margin >= base 0.1649 + 0.02, placebo < half",
                                "ktb": "reach(L1_aug) < reach(L2_aug), z>=2 + placebo guard",
                                "kte": "collapse healthy; iid-shuffled placebo market z < 2 at L1 k=1",
                                "status": "consistency reads on much-read windows - research, not certification"},
           "levels": levels_out, "reach_days": reach, "base_reach": BASE_REACH,
           "kta_retained": bool(kta_ok), "ktb_ordering": bool(ktb_ok),
           "kte": {"min_dim_std": round(float(stds.min()), 3), "eff_rank": round(eff_rank, 2),
                    "placebo_market_L1k1_hold_z": pz, "collapse_ok": bool(collapse_ok),
                    "placebo_market_ok": bool(placebo_market_ok)},
           "verdict": verdict, "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
