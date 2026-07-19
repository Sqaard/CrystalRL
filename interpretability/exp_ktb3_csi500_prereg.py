"""KT-B THIRD-WINDOW PREREGISTRATION — cross-market replication of the hierarchy signature on csi500.

WHY csi500 IS THE CLEAN WINDOW: every Dow window (dev/hold/oos) was touched by KT-B protocol
selection or referee probing. No KT-B analysis has EVER touched the rebuilt single-clock csi500
panel. A cross-market replication is also the stronger read: the hierarchy-horizon principle
should not be a Dow idiosyncrasy.

PREREGISTERED PROTOCOL (fixed in this docstring BEFORE running; the identified KT-B v2 protocol
with two pre-declared adaptations):
  * DATA: data/_csi500_rebuilt/csi500_rebuilt_panel_2026.csv (403 names, 2017-01..2026-07).
    Book = EW mean of daily_return. FEATURES are price/volume-native and contain NO RAW LEVELS
    (the W3 L2 lesson applied): per block [ret_sum, real_vol, max_dd, mean cross-sectional
    dispersion, up-day fraction, log volume change vs prior block].
  * SPLITS: TRAIN 2017-01-01..2022-12-31, DEV 2023-01-01..2023-12-31,
    HOLD 2024-01-01..2025-06-30 (primary), OOS 2025-07-01..2026-07-07 (confirmation).
  * LEVELS: L1 5d blocks D=8; L2 21d blocks D=4. (L3 skipped — nowhere near the samples.)
    k grids: L1 (1,2,4,8,16) [5..80d], L2 (1,2,4) [21..84d]. Context grid {2,4,6} (thin dev).
  * The v2 machinery verbatim: dev-tuned context+alpha, the strong null family
    (pers/mean/shrunk/ewma) dev-chosen, purged fits, year-shuffled placebo per cell,
    REACH = max horizon with margin>0 AND z>=2.0 AND placebo<half.
  * OUTCOMES (symmetric, the W3 criterion lesson):
      REPLICATES:   reach(L1) < reach(L2) on HOLD;
      FAILS:        reach(L1) > reach(L2) with at least one significant L1 cell beyond L2's
                    reach (evidence against cross-market generality);
      UNDERPOWERED: no level has any significant cell — not evidence either way (declared
                    possible: hold has ~17 monthly samples).
  This read closes the KT-B prereg either way; the theory file records whichever verdict.

Run: python interpretability/exp_ktb3_csi500_prereg.py     (~3 min)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

OUT = HERE / "exp_ktb3_csi500_prereg_report.json"
PANEL = ROOT / "data" / "_csi500_rebuilt" / "csi500_rebuilt_panel_2026.csv"
SPLITS = {"train": ("2017-01-01", "2022-12-31"), "dev": ("2023-01-01", "2023-12-31"),
          "hold": ("2024-01-01", "2025-06-30"), "oos": ("2025-07-01", "2026-07-07")}
LEVELS = {"L1": {"B": 5, "D": 8, "KS": (1, 2, 4, 8, 16)},
          "L2": {"B": 21, "D": 4, "KS": (1, 2, 4)}}
CTX_GRID = (2, 4, 6)
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)
Z_BAR = 2.0


def load_csi():
    d = pd.read_csv(PANEL, usecols=["date", "tic", "daily_return", "volume"])
    d["date"] = pd.to_datetime(d["date"])
    g = d.groupby("date")
    r = g["daily_return"].mean().rename("r")
    disp = g["daily_return"].std().rename("disp")
    vol = g["volume"].sum().rename("vol")
    up = d.assign(u=(d["daily_return"] > 0).astype(float)).groupby("date")["u"].mean().rename("up")
    F = pd.concat([r, disp, vol, up], axis=1).dropna()
    return F


def block_features_csi(F, B):
    idx = F.index
    rv, dv, vv, uv = (F[c].to_numpy() for c in ("r", "disp", "vol", "up"))
    ends = list(range(len(idx) - 1, B - 2, -B))[::-1]
    rows, b_end, prev_vol = [], [], None
    for e in ends:
        s = e - B + 1
        if s < 0:
            continue
        blk = rv[s:e + 1]
        if not np.isfinite(blk).all():
            rows.append(None); b_end.append(e); prev_vol = None; continue
        eqb = np.cumprod(1 + blk)
        tot_vol = float(vv[s:e + 1].sum())
        dlv = float(np.log(tot_vol / prev_vol)) if (prev_vol and prev_vol > 0) else 0.0
        rows.append([float(blk.sum()), float(blk.std()), float((eqb / np.maximum.accumulate(eqb) - 1).min()),
                     float(np.nanmean(dv[s:e + 1])), float(np.nanmean(uv[s:e + 1])), dlv])
        b_end.append(e); prev_vol = tot_vol
    return rows, np.array(b_end), idx


def build_level(F, B, D):
    rows, b_end, idx = block_features_csi(F, B)
    valid = [i for i, x in enumerate(rows) if x is not None]
    X = np.array([rows[i] for i in valid], dtype=np.float32)
    remap = {i: j for j, i in enumerate(valid)}
    dates_all = idx[b_end]
    tr_rows = [remap[i] for i in valid if dates_all[i] <= pd.Timestamp(SPLITS["train"][1])]
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-9
    Z = (X - mu) / sd
    pca = PCA(n_components=min(D, Z.shape[1]), random_state=0).fit(Z[tr_rows])
    S = pca.transform(Z).astype(np.float32)
    return {"S": S, "remap": remap, "valid": set(valid), "dates_all": dates_all, "n": len(rows), "B": B}


def assemble(bl, CTX, KS):
    n, remap, valid = bl["n"], bl["remap"], bl["valid"]
    samples = np.array([i for i in range(CTX - 1, n - max(KS))
                        if all(j in valid for j in list(range(i - CTX + 1, i + 1)) + [i + k for k in KS])])
    dates = bl["dates_all"][samples]
    tgt_dates = {k: bl["dates_all"][samples + k] for k in KS}
    masks = {w: np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
             for w, (a, b) in SPLITS.items()}
    ctx_of = np.array([[remap[j] for j in range(i - CTX + 1, i + 1)] for i in samples])
    return {"samples": samples, "dates": dates, "tgt_dates": tgt_dates, "masks": masks,
            "ctx_of": ctx_of, "cur_of": ctx_of[:, -1],
            "tgt_of": {k: np.array([remap[i + k] for i in samples]) for k in KS}, "CTX": CTX}


def cell(bl, asm, k, boot_block, plc_map=None):
    S = bl["S"]
    tr_all = np.where(asm["masks"]["train"])[0]
    purged = tr_all[asm["tgt_dates"][k][tr_all] <= pd.Timestamp(SPLITS["train"][1])]
    di = np.where(asm["masks"]["dev"])[0]
    if len(purged) < 20 or len(di) < 5:
        return None
    tgt_tr = plc_map[purged] if plc_map is not None else asm["tgt_of"][k][purged]
    best = None
    for a in ALPHAS:
        reg = Ridge(alpha=a).fit(S[asm["ctx_of"][purged]].reshape(len(purged), -1), S[tgt_tr])
        e = ((reg.predict(S[asm["ctx_of"][di]].reshape(len(di), -1)) - S[asm["tgt_of"][k][di]]) ** 2).sum(1)
        if best is None or e.mean() < best[0]:
            best = (e.mean(), a, reg)
    _, alpha, reg = best
    mean_v = S[np.unique(tgt_tr)].mean(0)
    cands = {"pers": lambda ii: S[asm["cur_of"][ii]],
             "mean": lambda ii: np.repeat(mean_v[None, :], len(ii), 0)}
    for lam in (0.25, 0.5, 0.75):
        cands[f"shrunk{lam}"] = (lambda l: lambda ii: l * S[asm["cur_of"][ii]] + (1 - l) * mean_v[None, :])(lam)
    for hl in (2, 4):
        w_ = 0.5 ** (np.arange(asm["ctx_of"].shape[1])[::-1] / hl); w_ = w_ / w_.sum()
        cands[f"ewma{hl}"] = (lambda ww: lambda ii: np.einsum("c,icd->id", ww.astype(np.float32), S[asm["ctx_of"][ii]]))(w_)
    nbest = None
    for nm, f in cands.items():
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
        blk = min(boot_block, max(2, len(ii) // 4))
        if len(ii) - blk + 1 < 2:
            out[w] = {"margin": None, "z": None, "n": int(len(ii)), "note": "degenerate bootstrap"}
            continue
        d = e_null - e_pred
        _, se = block_z(d, block=blk, n_boot=1000, seed=7)
        out[w] = {"margin": round(float(d.mean() / (e_null.mean() + 1e-12)), 4),
                  "z": round(float(d.mean() / se), 2), "n": int(len(ii))}
    return out


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== KT-B third window (PREREGISTERED) — csi500 cross-market replication ===")
    F = load_csi()
    print(f"  book: {len(F)} days {F.index[0].date()} -> {F.index[-1].date()}")
    rng = np.random.default_rng(1234)
    reach, levels_out, any_sig = {}, {}, {}
    for name, cfg in LEVELS.items():
        bl = build_level(F, cfg["B"], cfg["D"])
        rows, rch, sig_any = {}, 0, False
        for k in cfg["KS"]:
            cand = None
            for CTX in CTX_GRID:
                asm = assemble(bl, CTX, cfg["KS"])
                if asm["masks"]["train"].sum() < 25:
                    continue
                c = cell(bl, asm, k, boot_block=CTX)
                if c is None:
                    continue
                S = bl["S"]
                di = np.where(asm["masks"]["dev"])[0]
                tr_all = np.where(asm["masks"]["train"])[0]
                purged = tr_all[asm["tgt_dates"][k][tr_all] <= pd.Timestamp(SPLITS["train"][1])]
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
                   and (hp.get("margin") is None or (h["margin"] > 0 and hp["margin"] < 0.5 * h["margin"])))
            rows[k] = {"horizon_days": k * cfg["B"], "ctx": CTX, "real": c,
                       "placebo_hold": hp, "significant": bool(sig)}
            if sig:
                rch = max(rch, k * cfg["B"]); sig_any = True
            print(f"  {name} k={k} ({k*cfg['B']:3d}d) ctx {CTX}: m {h.get('margin')} z {h.get('z')} "
                  f"null {c['null']} plc {hp.get('margin')} -> {'SIG' if sig else '-'}")
        reach[name], any_sig[name] = rch, sig_any
        levels_out[name] = {str(k): v for k, v in rows.items()}
    print("  reach(days):", reach)

    if not any(any_sig.values()):
        verdict = ("UNDERPOWERED — no level has any significant cell on csi500 hold (declared possible "
                   "at these sample sizes): NOT evidence for or against the hierarchy; the KT-B prereg "
                   "is closed as underpowered-cross-market and the Dow FULL PASS stands unreplicated.")
    elif reach["L1"] < reach["L2"]:
        verdict = (f"REPLICATES — reach(L1)={reach['L1']}d < reach(L2)={reach['L2']}d on a market no "
                   "KT-B analysis ever touched: the hierarchy signature is cross-market; T5-on-markets "
                   "upgrades to 'measured and replicated'.")
    elif reach["L1"] > reach["L2"]:
        verdict = (f"FAILS — reach(L1)={reach['L1']}d > reach(L2)={reach['L2']}d on csi500: the Dow "
                   "hierarchy result does not generalize cross-market; the theory file records the split "
                   "(Dow PASS, csi500 FAIL) and T5 stays market-conditional.")
    else:
        verdict = (f"TIED/AMBIGUOUS — reach L1 {reach['L1']} = L2 {reach['L2']}: no ordering read; "
                   "closed as non-confirming, non-refuting.")
    rep = {"preregistration": "see module docstring (fixed before running)",
           "levels": levels_out, "reach_days": reach, "verdict": verdict,
           "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
