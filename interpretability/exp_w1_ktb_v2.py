"""KT-B v2 — the identified hierarchy-signature protocol (every referee fix, fixed before running).

The v1 run's KILL was invalidated on BOTH sides (exp_w1_ktb_review.json): L1's 40d reach was ~84%
frozen-mean-null mechanics under the 2022-23 level shift (the positive placebo was the symptom),
and L2's zero reach was capacity starvation (dev itself prefers CTX=2, which yields z 6.7@21d and
survives a strong null family). The ordering was UNDECIDED — it flipped across defensible choices.
This v2 fixes the protocol so the ordering is identified:

  * CONTEXT LENGTH dev-tuned per (level, k) over the pre-registered grid {2, 4, 6, 12} blocks;
  * STRONG NULL FAMILY dev-chosen per (level, k): {persistence; train-mean; shrunk-persistence
    lam*s_cur+(1-lam)*mean, lam in {0.25,.5,.75}; EWMA of context blocks, halflife in {2,4}};
  * EXTENDED k-grids so no level's reach is capped by construction:
    L1 k in {1,2,4,8,16,24} (5..120d), L2 k in {1,2,4,8} (21..168d), L3 k in {1,2} (63,126d);
  * PURGE: tuning/fitting uses only train samples whose TARGET block ends inside TRAIN;
  * ROBUST REACH (pre-registered): the LARGEST horizon (days) whose cell has margin>0 AND
    z >= 2.0 AND placebo-margin < half the real margin (max-significant, no consecutive scan;
    z=2.0 is the single-cell bar chosen to keep the ~6-cell family error near 10%);
  * FLAT TWIN at L2/L3's best significant k (dev-chosen contexts): pooled must beat flat for the
    FULL-PASS reading; matched-horizon secondary under the SAME null family per space.

PREREGISTERED OUTCOMES: FULL PASS reach(L1)<reach(L2) AND pooled>flat at L2's best cell;
MECHANICAL if ordering holds but flat matches; KILL if reach(L1)>=reach(L2). L3 reported only
(n=8 hold blocks — no statistical content; the v1 referee's sign-flip p=0.14 note stands).

Run: python interpretability/exp_w1_ktb_v2.py     (~3-5 min, sklearn only)
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
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

OUT = HERE / "exp_w1_ktb_v2_report.json"
LEVELS = {"L1": {"B": 5, "D": 16, "KS": (1, 2, 4, 8, 16, 24)},
          "L2": {"B": 21, "D": 8, "KS": (1, 2, 4, 8)},
          "L3": {"B": 63, "D": 4, "KS": (1, 2)}}
CTX_GRID = (2, 4, 6, 12)
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)
Z_BAR = 2.0


def build_blocks(r, obs, B, D):
    rows, b_end, idx = block_features(r, obs, B)
    valid = [i for i, x in enumerate(rows) if x is not None]
    Xraw = np.array([rows[i] for i in valid], dtype=np.float32)
    remap = {i: j for j, i in enumerate(valid)}
    dates_all = idx[b_end]
    # standardize + PCA on train blocks
    tr_blocks = [remap[i] for i in valid if dates_all[i] <= pd.Timestamp(TRAIN[1])]
    mu, sd = Xraw[tr_blocks].mean(0), Xraw[tr_blocks].std(0) + 1e-9
    Xb = ((Xraw - mu) / sd).astype(np.float32)
    pca = PCA(n_components=min(D, Xb.shape[1]), random_state=0).fit(Xb[tr_blocks])
    S = pca.transform(Xb).astype(np.float32)
    return {"S": S, "remap": remap, "valid": set(valid), "dates_all": dates_all, "n": len(rows), "B": B}


def assemble(bl, CTX, KS):
    n, remap, valid = bl["n"], bl["remap"], bl["valid"]
    samples = [i for i in range(CTX - 1, n - max(KS))
               if all(j in valid for j in list(range(i - CTX + 1, i + 1)) + [i + k for k in KS])]
    samples = np.array(samples)
    dates = bl["dates_all"][samples]
    tgt_dates = {k: bl["dates_all"][samples + k] for k in KS}
    masks = {w: np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
             for w, (a, b) in dict(train=TRAIN, dev=DEV, hold=HOLD, oos=OOS).items()}
    ctx_of = np.array([[remap[j] for j in range(i - CTX + 1, i + 1)] for i in samples])
    tgt_of = {k: np.array([remap[i + k] for i in samples]) for k in KS}
    return {"samples": samples, "dates": dates, "tgt_dates": tgt_dates, "masks": masks,
            "ctx_of": ctx_of, "cur_of": ctx_of[:, -1], "tgt_of": tgt_of, "CTX": CTX}


def null_family(S, asm, k, fit_sel):
    """Candidate trivial predictors; returns dict name -> per-sample predictions fn(indices)."""
    mean_v = S[np.unique(asm["tgt_of"][k][fit_sel])].mean(0)
    def pers(ii):
        return S[asm["cur_of"][ii]]
    def mean_(ii):
        return np.repeat(mean_v[None, :], len(ii), 0)
    def shrunk(lam):
        def f(ii):
            return lam * S[asm["cur_of"][ii]] + (1 - lam) * mean_v[None, :]
        return f
    def ewma(hl):
        w = 0.5 ** (np.arange(asm["ctx_of"].shape[1])[::-1] / hl)
        w = w / w.sum()
        def f(ii):
            return np.einsum("c,icd->id", w.astype(np.float32), S[asm["ctx_of"][ii]])
        return f
    fam = {"pers": pers, "mean": mean_, "shrunk.25": shrunk(0.25), "shrunk.5": shrunk(0.5),
           "shrunk.75": shrunk(0.75), "ewma2": ewma(2), "ewma4": ewma(4)}
    return fam


def cell(bl, asm, k, boot_block, plc_map=None):
    """Fit dev-tuned ridge (purged train), dev-choose the null; hold/oos margins vs that null."""
    S = bl["S"]
    tr_all = np.where(asm["masks"]["train"])[0]
    purged = tr_all[asm["tgt_dates"][k][tr_all] <= pd.Timestamp(TRAIN[1])]
    di = np.where(asm["masks"]["dev"])[0]
    if len(purged) < 20 or len(di) < 6:
        return None
    tgt_tr = plc_map[purged] if plc_map is not None else asm["tgt_of"][k][purged]
    best = None
    for a in ALPHAS:
        reg = Ridge(alpha=a).fit(S[asm["ctx_of"][purged]].reshape(len(purged), -1), S[tgt_tr])
        e = ((reg.predict(S[asm["ctx_of"][di]].reshape(len(di), -1)) - S[asm["tgt_of"][k][di]]) ** 2).sum(1)
        if best is None or e.mean() < best[0]:
            best = (e.mean(), a, reg)
    _, alpha, reg = best
    fam = null_family(S, asm, k, purged)
    nbest = None
    for nm, f in fam.items():
        e = ((f(di) - S[asm["tgt_of"][k][di]]) ** 2).sum(1)
        if nbest is None or e.mean() < nbest[0]:
            nbest = (e.mean(), nm, f)
    _, null_name, null_f = nbest
    out = {"alpha": alpha, "null": null_name, "ctx": asm["CTX"]}
    for w in ("hold", "oos"):
        ii = np.where(asm["masks"][w])[0]
        if len(ii) < 6:
            out[w] = {"margin": None, "z": None, "n": int(len(ii))}
            continue
        e_pred = ((reg.predict(S[asm["ctx_of"][ii]].reshape(len(ii), -1)) - S[asm["tgt_of"][k][ii]]) ** 2).sum(1)
        e_null = ((null_f(ii) - S[asm["tgt_of"][k][ii]]) ** 2).sum(1)
        d = e_null - e_pred
        _, se = block_z(d, block=min(boot_block, max(2, len(ii) // 4)), n_boot=1000, seed=7)
        out[w] = {"margin": round(float(d.mean() / (e_null.mean() + 1e-12)), 4),
                  "z": round(float(d.mean() / se), 2), "n": int(len(ii))}
    return out


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== KT-B v2 — identified protocol (dev-tuned ctx, strong nulls, extended grids) ===")
    r, obs, rf = load_v2()
    rng = np.random.default_rng(1234)
    report = {"preregistration": {"reach": "max horizon with margin>0 AND z>=2.0 AND placebo<0.5x",
                                   "ctx_grid": CTX_GRID, "null_family": "pers/mean/shrunk(.25/.5/.75)/ewma(2/4)",
                                   "purge": "train fits use targets ending inside TRAIN",
                                   "outcomes": "FULL PASS / MECHANICAL / KILL; L3 reported only"},
              "levels": {}}
    reach, best_cells = {}, {}
    for name, cfg in LEVELS.items():
        bl = build_blocks(r, obs, cfg["B"], cfg["D"])
        rows = {}
        rch = 0
        for k in cfg["KS"]:
            # dev-tune CTX per (level,k)
            cand = None
            for CTX in CTX_GRID:
                asm = assemble(bl, CTX, cfg["KS"])
                if asm["masks"]["train"].sum() < 25:
                    continue
                c = cell(bl, asm, k, boot_block=CTX)
                if c is None:
                    continue
                # dev margin for selection: refit dev-eval via same cell (dev used in tuning already);
                # select CTX by dev null-relative error: recompute on dev
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
            # placebo at the chosen CTX
            yrs = np.asarray(asm["dates"].year)
            trm = asm["masks"]["train"]
            uy = np.unique(yrs[trm]); pm = dict(zip(uy, rng.permutation(uy)))
            plc = asm["tgt_of"][k].copy()
            for j in np.where(trm)[0]:
                cands = np.where(yrs == pm.get(yrs[j], yrs[j]))[0]
                if len(cands):
                    plc[j] = asm["tgt_of"][k][cands[j % len(cands)]]
            cp = cell(bl, asm, k, boot_block=CTX, plc_map=plc)
            h, hp = c.get("hold", {}), (cp or {}).get("hold", {})
            rows[k] = {"horizon_days": k * cfg["B"], "real": c, "placebo_hold": hp}
            sig = (h.get("margin") is not None and h["margin"] > 0 and (h.get("z") or -9) >= Z_BAR
                   and (hp.get("margin") is None or hp["margin"] < 0.5 * h["margin"]))
            rows[k]["significant"] = bool(sig)
            if sig:
                rch = max(rch, k * cfg["B"])
                best_cells[name] = (k, asm, c)
            print(f"  {name} k={k} ({k*cfg['B']:3d}d) ctx {CTX}: margin {h.get('margin')} z {h.get('z')} "
                  f"null {c['null']} plc {hp.get('margin')} -> {'SIG' if sig else '-'}")
        reach[name] = rch
        report["levels"][name] = {str(k): v for k, v in rows.items()}
    print("  reach(days):", reach)

    # flat twin at L2's best significant cell (pooled arm vs flat L1-context arm, same protocol)
    flat_note = None
    if "L2" in best_cells:
        from interpretability.exp_w1_ktb_hierarchy import flat_twin, build_level
        k2, asm2, c2 = best_cells["L2"]
        l1full = build_level(r, obs, 5, 16, 12, (1, 2, 4, 8))
        lvl2 = {"CTX": asm2["CTX"], "B": 21, "S": build_blocks(r, obs, 21, 8)["S"],
                "dates": asm2["dates"], "masks": asm2["masks"], "ctx_of": asm2["ctx_of"],
                "cur_of": asm2["cur_of"], "tgt_of": asm2["tgt_of"]}
        ft = flat_twin(l1full, lvl2, k2, asm2["CTX"])
        flat_note = {"k": k2, "pooled_hold_margin": c2["hold"]["margin"],
                     "flat_hold": ft.get("hold")}
        print(f"  flat twin at L2 best k={k2}: pooled {c2['hold']['margin']} vs flat {ft.get('hold', {}).get('margin')}")

    ordering = reach["L1"] < reach["L2"]
    pooled_beats = (flat_note is not None and flat_note["flat_hold"] and
                    flat_note["flat_hold"].get("margin") is not None and
                    flat_note["pooled_hold_margin"] > flat_note["flat_hold"]["margin"])
    if ordering and pooled_beats:
        verdict = (f"FULL PASS — reach(L1)={reach['L1']}d < reach(L2)={reach['L2']}d under the "
                   "identified protocol, and the pooled L2 arm beats its capacity-matched flat "
                   "twin: the hierarchy dividend is real in both target and representation. "
                   "T5-on-markets supported beyond the regime data point (L3 underpowered, reported only).")
    elif ordering:
        verdict = (f"MECHANICAL — reach(L1)={reach['L1']}d < reach(L2)={reach['L2']}d but the flat "
                   "twin matches the pooled arm: the horizon dividend is target aggregation alone.")
    else:
        verdict = (f"KILL — reach(L1)={reach['L1']}d >= reach(L2)={reach['L2']}d under the "
                   "identified protocol: no hierarchy dividend; theory amendment per methodology §8.")
    report["reach_days"] = reach
    report["flat_twin"] = flat_note
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
