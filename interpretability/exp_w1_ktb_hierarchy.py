"""W1/KT-B — THE HIERARCHY SIGNATURE: does abstraction level buy prediction horizon on markets?

The centerpiece read of CRYSTAL-WORLD (business/CRYSTAL_WORLD_METHODOLOGY.md, KT-B), the direct
market test of T5 ("quantum seconds -> thermodynamics hours"). Publishable either way.

DESIGN (all-linear per the W0 amendment "L1 is linear until proven otherwise"; homogeneous
construction across levels so the curve is not confounded by architecture):
  * Level ℓ = PCA-D_ℓ of WITHIN-BLOCK features of non-overlapping blocks of B_ℓ days:
      L1: B=5d  (week),    D=16, context 12 blocks (60d),  k in {1,2,4,8}  -> horizons 5..40d
      L2: B=21d (month),   D=8,  context 12 blocks (252d), k in {1,2,4,8}  -> horizons 21..168d
      L3: B=63d (quarter), D=4,  context 4 blocks (252d),  k in {1,2}      -> horizons 63,126d
    Features per block (the W0-v4 recipe, no window crosses a block boundary): [ret sum, realized
    vol, max drawdown, dVIX, end-levels VIX/trend/turb/y10/vrp/ebp].
  * Predictor per (ℓ,k): ridge, alpha tuned on DEV; null per (ℓ,k) = better-on-DEV of
    {persistence, train-mean}; skill = relative error reduction vs the null; z = paired block
    bootstrap with block length = the context overlap span in samples.
  * REACH(ℓ) = the largest horizon in DAYS whose skill has margin>0 AND z>=1.28 on HOLD, scanning
    k upward while significant from k=1 (reach 0 if k=1 fails). OOS reported as confirmation.
  * CONTROLS:
      - placebo (year-shuffled targets) per level: reach must be 0;
      - FLAT TWIN (Nachum/G8): predict the L2/L3 target from a FLAT L1-block context spanning
        the same lookback days, PCA-compressed to the same regression input dim as the pooled
        arm, same tuning protocol. Separates "abstract representation helps" from "coarse targets
        are just smoother".
      - MATCHED-HORIZON secondary read: skill(L2@21d) vs skill(L1@20d), skill(L3@63d) vs
        skill(L2@63d) — the sharpest T5 statement (same distance, different abstraction).

PREREGISTERED OUTCOMES (before running):
  * FULL PASS:   reach(L1) < reach(L2) on HOLD (primary; L3 directional only — power caveat
                 below) AND the pooled L2 arm beats its flat twin (margin higher at k=1).
  * MECHANICAL:  reach ordering holds but the flat twin matches the pooled arm — the horizon
                 dividend comes from TARGET aggregation alone (Simon's aggregation is real; the
                 representation hierarchy adds nothing). T5 half-confirmed.
  * KILL:        reach(L1) >= reach(L2) (flat or inverted curve) — T5 fails on markets beyond
                 the already-known regime result; major theory-file amendment.
POWER CAVEAT (declared): L3 has ~8 hold blocks — its reach is directional color, NOT part of the
bar. The primary bar is L1-vs-L2 only.

Run: python interpretability/exp_w1_ktb_hierarchy.py     (~1 min, sklearn only)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import load_v2  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

OUT = HERE / "exp_w1_ktb_hierarchy_report.json"
LEVELS = {"L1": {"B": 5, "D": 16, "CTX": 12, "KS": (1, 2, 4, 8)},
          "L2": {"B": 21, "D": 8, "CTX": 12, "KS": (1, 2, 4, 8)},
          "L3": {"B": 63, "D": 4, "CTX": 4, "KS": (1, 2)}}
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)


def block_features(r, obs, B):
    idx = r.index
    rv = r.to_numpy()
    ends = list(range(len(idx) - 1, B - 2, -B))[::-1]
    rows, b_end = [], []
    for e in ends:
        s = e - B + 1
        if s < 0:
            continue
        blk = rv[s:e + 1]
        lvl, lvl0 = obs.iloc[e], obs.iloc[s]
        need = lvl[["VIX", "SP500_Trend", "turbulence", "10Y_Yield", "vrp", "ebp"]]
        if not (np.isfinite(blk).all() and np.isfinite(need.to_numpy(dtype=float)).all()):
            rows.append(None); b_end.append(e); continue
        eqb = np.cumprod(1 + blk)
        rows.append([float(blk.sum()), float(blk.std()), float((eqb / np.maximum.accumulate(eqb) - 1).min()),
                     float(lvl["VIX"] - lvl0["VIX"]), float(lvl["VIX"]), float(lvl["SP500_Trend"]),
                     float(lvl["turbulence"]), float(lvl["10Y_Yield"]), float(lvl["vrp"]), float(lvl["ebp"])])
        b_end.append(e)
    return rows, np.array(b_end), idx


def build_level(r, obs, B, D, CTX, KS):
    rows, b_end, idx = block_features(r, obs, B)
    valid = [i for i, x in enumerate(rows) if x is not None]
    Xraw = np.array([rows[i] for i in valid], dtype=np.float32)
    remap = {i: j for j, i in enumerate(valid)}
    n = len(rows)
    samples = [i for i in range(CTX - 1, n - max(KS))
               if all(rows[j] is not None for j in list(range(i - CTX + 1, i + 1)) + [i + k for k in KS])]
    samples = np.array(samples)
    dates = idx[b_end[samples]]
    masks = {w: np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
             for w, (a, b) in dict(train=TRAIN, dev=DEV, hold=HOLD, oos=OOS).items()}
    tr_rows = [remap[i] for i in samples[masks["train"]]]
    mu, sd = Xraw[tr_rows].mean(0), Xraw[tr_rows].std(0) + 1e-9
    Xb = ((Xraw - mu) / sd).astype(np.float32)
    ctx_of = np.array([[remap[j] for j in range(i - CTX + 1, i + 1)] for i in samples])
    tgt_of = {k: np.array([remap[i + k] for i in samples]) for k in KS}
    tri = np.where(masks["train"])[0]
    pca = PCA(n_components=min(D, Xb.shape[1]), random_state=0).fit(Xb[np.unique(ctx_of[tri].reshape(-1))])
    S = pca.transform(Xb).astype(np.float32)
    return {"Xb": Xb, "S": S, "samples": samples, "dates": dates, "masks": masks,
            "ctx_of": ctx_of, "cur_of": ctx_of[:, -1], "tgt_of": tgt_of, "b_end": b_end,
            "remap": remap, "rows_valid": valid, "CTX": CTX, "KS": KS, "B": B}


def fit_eval(S, ctx_of, cur_of, tgt_of_k, masks, boot_block, plc_map=None):
    """Dev-tuned ridge + dev-chosen null; returns per-window (margin, z). plc_map optionally
    replaces TRAIN targets (placebo)."""
    tri, di = np.where(masks["train"])[0], np.where(masks["dev"])[0]
    tgt_tr = plc_map[tri] if plc_map is not None else tgt_of_k[tri]
    best = None
    for a in ALPHAS:
        reg = Ridge(alpha=a).fit(S[ctx_of[tri]].reshape(len(tri), -1), S[tgt_tr])
        e = ((reg.predict(S[ctx_of[di]].reshape(len(di), -1)) - S[tgt_of_k[di]]) ** 2).sum(1)
        if best is None or e.mean() < best[0]:
            best = (e.mean(), a, reg)
    _, alpha, reg = best
    mean_v = S[np.unique(tgt_tr)].mean(0)
    ep = ((S[cur_of[di]] - S[tgt_of_k[di]]) ** 2).sum(1)
    em = ((mean_v[None, :] - S[tgt_of_k[di]]) ** 2).sum(1)
    null_kind = "pers" if ep.mean() <= em.mean() else "mean"
    out = {"alpha": alpha, "null": null_kind}
    for w in ("hold", "oos"):
        ii = np.where(masks[w])[0]
        if len(ii) < 6:
            out[w] = {"margin": None, "z": None, "n": int(len(ii))}
            continue
        e_pred = ((reg.predict(S[ctx_of[ii]].reshape(len(ii), -1)) - S[tgt_of_k[ii]]) ** 2).sum(1)
        ep_ = ((S[cur_of[ii]] - S[tgt_of_k[ii]]) ** 2).sum(1)
        em_ = ((mean_v[None, :] - S[tgt_of_k[ii]]) ** 2).sum(1)
        e_null = ep_ if null_kind == "pers" else em_
        d = e_null - e_pred
        _, se = block_z(d, block=min(boot_block, max(2, len(ii) // 4)), n_boot=1000, seed=7)
        out[w] = {"margin": round(float(d.mean() / (e_null.mean() + 1e-12)), 4),
                  "z": round(float(d.mean() / se), 2), "n": int(len(ii))}
    return out


def flat_twin(l1, lvl, k, boot_block):
    """Predict the pooled level's k-target from a FLAT L1 context spanning the same lookback,
    PCA-compressed to the pooled arm's regression input dim. Alignment: for each pooled sample,
    find the L1 sample whose context-end day equals (or last precedes) the pooled context end."""
    span_days = lvl["CTX"] * lvl["B"]
    n_l1_blocks = span_days // l1["B"]
    reg_dim = lvl["CTX"] * lvl["S"].shape[1]
    # map pooled sample -> l1 sample index by date
    l1_dates = l1["dates"]
    pos = np.searchsorted(l1_dates, lvl["dates"], side="right") - 1
    ok = pos >= 0
    # flat context: last n_l1_blocks L1 block rows before/at that l1 sample's current block
    l1_all_rows = l1["ctx_of"]  # per-sample 12 blocks; we need longer: use block indices directly
    # reconstruct per-l1-sample the trailing n_l1_blocks block ids
    # l1 sample i corresponds to block index l1['samples'][i]; its trailing blocks:
    blocks_ok = set(l1["remap"].keys())
    flat_ids = []
    for j, p in enumerate(pos):
        if not ok[j]:
            flat_ids.append(None); continue
        bi = l1["samples"][p]
        ids = list(range(bi - n_l1_blocks + 1, bi + 1))
        if any(i not in blocks_ok for i in ids):
            flat_ids.append(None); continue
        flat_ids.append([l1["remap"][i] for i in ids])
    keep = np.array([j for j, f in enumerate(flat_ids) if f is not None])
    Xflat = np.stack([np.asarray(l1["S"][flat_ids[j]]).ravel() for j in keep]).astype(np.float32)
    masks = {w: lvl["masks"][w][keep] for w in lvl["masks"]}
    tgt = {k: lvl["tgt_of"][k][keep]}
    cur = lvl["cur_of"][keep]
    tri = np.where(masks["train"])[0]
    p2 = PCA(n_components=min(reg_dim, Xflat.shape[1], len(tri) - 2), random_state=0).fit(Xflat[tri])
    Xf = p2.transform(Xflat).astype(np.float32)
    # ridge from flat context to the POOLED level's target representation
    di = np.where(masks["dev"])[0]
    best = None
    for a in ALPHAS:
        reg = Ridge(alpha=a).fit(Xf[tri], lvl["S"][tgt[k][tri]])
        e = ((reg.predict(Xf[di]) - lvl["S"][tgt[k][di]]) ** 2).sum(1)
        if best is None or e.mean() < best[0]:
            best = (e.mean(), a, reg)
    _, alpha, reg = best
    mean_v = lvl["S"][np.unique(tgt[k][tri])].mean(0)
    ep = ((lvl["S"][cur[di]] - lvl["S"][tgt[k][di]]) ** 2).sum(1)
    em = ((mean_v[None, :] - lvl["S"][tgt[k][di]]) ** 2).sum(1)
    null_kind = "pers" if ep.mean() <= em.mean() else "mean"
    out = {"alpha": alpha, "null": null_kind}
    for w in ("hold", "oos"):
        ii = np.where(masks[w])[0]
        if len(ii) < 6:
            out[w] = {"margin": None, "z": None, "n": int(len(ii))}
            continue
        e_pred = ((reg.predict(Xf[ii]) - lvl["S"][tgt[k][ii]]) ** 2).sum(1)
        ep_ = ((lvl["S"][cur[ii]] - lvl["S"][tgt[k][ii]]) ** 2).sum(1)
        em_ = ((mean_v[None, :] - lvl["S"][tgt[k][ii]]) ** 2).sum(1)
        e_null = ep_ if null_kind == "pers" else em_
        d = e_null - e_pred
        _, se = block_z(d, block=min(boot_block, max(2, len(ii) // 4)), n_boot=1000, seed=7)
        out[w] = {"margin": round(float(d.mean() / (e_null.mean() + 1e-12)), 4),
                  "z": round(float(d.mean() / se), 2), "n": int(len(ii))}
    return out


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W1/KT-B — the hierarchy signature: reach(level) on disjoint blocks ===")
    r, obs, rf = load_v2()
    built = {}
    for name, cfg in LEVELS.items():
        built[name] = build_level(r, obs, cfg["B"], cfg["D"], cfg["CTX"], cfg["KS"])
        m = built[name]["masks"]
        print(f"  {name}: blocks B={cfg['B']}d, samples {len(built[name]['samples'])} "
              f"(tr {m['train'].sum()} dev {m['dev'].sum()} hold {m['hold'].sum()} oos {m['oos'].sum()})")

    rng = np.random.default_rng(1234)
    results, reach = {}, {}
    for name, lvl in built.items():
        results[name] = {}
        # placebo map (year-shuffled train targets), one map per level reused across k
        yrs = np.asarray(lvl["dates"].year)
        trm = lvl["masks"]["train"]
        uy = np.unique(yrs[trm]); pm = dict(zip(uy, rng.permutation(uy)))
        boot = lvl["CTX"]
        rch, rch_ok = 0, True
        for k in lvl["KS"]:
            plc = lvl["tgt_of"][k].copy()
            for j in np.where(trm)[0]:
                cands = np.where(yrs == pm.get(yrs[j], yrs[j]))[0]
                if len(cands):
                    plc[j] = lvl["tgt_of"][k][cands[j % len(cands)]]
            res = fit_eval(lvl["S"], lvl["ctx_of"], lvl["cur_of"], lvl["tgt_of"][k], lvl["masks"], boot)
            resp = fit_eval(lvl["S"], lvl["ctx_of"], lvl["cur_of"], lvl["tgt_of"][k], lvl["masks"], boot, plc_map=plc)
            results[name][k] = {"horizon_days": k * lvl["B"], "real": res, "placebo": resp}
            h = res.get("hold", {})
            print(f"    {name} k={k} ({k*lvl['B']:3d}d): margin {h.get('margin')} z {h.get('z')} "
                  f"(null {res['null']}, a {res['alpha']}) | plc {resp.get('hold', {}).get('margin')}")
            if rch_ok and h.get("margin") is not None and h["margin"] > 0 and (h.get("z") or -9) >= 1.28:
                rch = k * lvl["B"]
            else:
                rch_ok = False
        reach[name] = rch
    print("  reach(days):", reach)

    # flat twins for L2/L3 at k=1
    flats = {}
    for name in ("L2", "L3"):
        flats[name] = flat_twin(built["L1"], built[name], 1, built[name]["CTX"])
        print(f"  flat-twin {name}@k=1: hold margin {flats[name]['hold'].get('margin')} "
              f"z {flats[name]['hold'].get('z')} vs pooled {results[name][1]['real']['hold'].get('margin')}")

    # matched-horizon secondary
    matched = {"L2@21d_vs_L1@20d": {"L2": results["L2"][1]["real"]["hold"],
                                     "L1": results["L1"][4]["real"]["hold"]},
               "L3@63d_vs_L2@63d": {"L3": results["L3"][1]["real"]["hold"],
                                     "L2": results["L2"][3 if 3 in results["L2"] else 4]["real"]["hold"]
                                     if 4 in results["L2"] else None}}

    ordering = reach["L1"] < reach["L2"]
    l2_pool = results["L2"][1]["real"]["hold"].get("margin") or -9
    l2_flat = flats["L2"]["hold"].get("margin")
    pooled_beats_flat = l2_flat is not None and l2_pool > l2_flat
    if ordering and pooled_beats_flat:
        verdict = (f"FULL PASS — reach(L1)={reach['L1']}d < reach(L2)={reach['L2']}d AND the pooled "
                   "L2 arm beats its capacity-matched flat twin: the hierarchy dividend is real in "
                   "BOTH the target and the representation. T5-on-markets supported beyond the "
                   "regime data point. L3 directional color in the report.")
    elif ordering:
        verdict = (f"MECHANICAL — reach(L1)={reach['L1']}d < reach(L2)={reach['L2']}d but the flat "
                   "twin matches the pooled arm: the horizon dividend comes from TARGET aggregation "
                   "alone (Simon's aggregation real; representation hierarchy adds nothing yet). "
                   "T5 half-confirmed; the theory file gets the split verdict.")
    else:
        verdict = (f"KILL — reach ordering fails (L1 {reach['L1']}d vs L2 {reach['L2']}d): no "
                   "hierarchy dividend beyond the known regime result; major theory amendment "
                   "pre-registered in the methodology (section 8, death #1).")
    rep = {"preregistration": {"primary_bar": "reach(L1) < reach(L2) on HOLD; L3 directional only (power)",
                                "outcomes": ["FULL PASS (+pooled>flat)", "MECHANICAL (flat matches)", "KILL"],
                                "reach_def": "largest horizon days with margin>0 AND z>=1.28, scanned from k=1",
                                "controls": "year-shuffled placebo per level; capacity-matched flat twin; matched-horizon"},
           "levels": {n: {str(k): v for k, v in results[n].items()} for n in results},
           "reach_days": reach, "flat_twins": flats, "matched_horizon": matched,
           "verdict": verdict, "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
