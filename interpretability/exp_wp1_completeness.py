"""WP-1 — the representation-completeness test: do knowledge-derived state blocks improve
target-representation prediction? (The empirical answer to Ivan's sufficiency question.)

A state representation is never provably complete (Grossman-Stiglitz; the VoI≈0 floor). The
testable question is INCREMENTAL completeness: for a candidate state block named by the world
picture, does adding it to the context improve prediction of the target representation, against
a capacity-fair noise twin? This script runs the first knowledge-derived candidate — MARKET
BREADTH/DISPERSION (the world-picture corpus names breadth/participation as a first-order
market-state variable; our current 10-feature state has NO cross-sectional information).

CANDIDATE BLOCK (computed from the per-ticker Dow v2 panel, within-block only, no raw levels):
  per day: frac_above_50dma, frac_at_20d_high, frac_at_20d_low, cross-sectional return
  dispersion, up-name fraction -> per 5d block: means of the five dailies.
ARMS (identified protocol: dev-tuned alpha, strong null family, purged fits, boot block=CTX):
  BASE        context = PCA-10 of the standard 10-feature blocks (the W-series state);
  +BREADTH    context = PCA of [10 features ⊕ 5 breadth features] (15 -> PCA-12);
  +NOISE      same dims, breadth columns year-block-shuffled (capacity-fair twin), 3 seeds.
PREREGISTERED READ (before running): the breadth block ADDS information iff on HOLD at k=1 or
k=4: margin(+BREADTH) >= margin(BASE) + 0.02 AND margin(+BREADTH) > max seed margin(+NOISE).
Otherwise the state is incrementally complete w.r.t. breadth (a real answer, not a failure).
OOS reported as confirmation.

Run: python interpretability/exp_wp1_completeness.py     (~4 min)
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
from interpretability.exp_w1_ktb_v2 import build_blocks  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

OUT = HERE / "exp_wp1_completeness_report.json"
PANEL = ROOT / "data" / "_dow_extended" / "dow_extended_panel_v2.csv"
B = 5
KS = (1, 4)
CTX = {1: 2, 4: 4}
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)
N_NOISE = 3


def breadth_daily():
    df = pd.read_csv(PANEL, usecols=["date", "tic", "adjclose"])
    df["date"] = pd.to_datetime(df["date"])
    w = df.pivot_table(index="date", columns="tic", values="adjclose", aggfunc="first").sort_index()
    ma50 = w.rolling(50).mean()
    hi20 = w.rolling(20).max()
    lo20 = w.rolling(20).min()
    r = w.pct_change()
    out = pd.DataFrame({
        "frac_above_ma": (w > ma50).mean(axis=1),
        "frac_hi20": (w >= hi20).mean(axis=1),
        "frac_lo20": (w <= lo20).mean(axis=1),
        "xdisp": r.std(axis=1),
        "up_frac": (r > 0).mean(axis=1)})
    return out


def block_means(daily, idx, b_end, valid):
    """Per valid block: within-block means of the daily breadth columns."""
    D = daily.reindex(idx)
    arr = D.to_numpy(dtype=float)
    rows = []
    for i in valid:
        e = b_end[i]; s = e - B + 1
        seg = arr[s:e + 1]
        rows.append(np.nanmean(seg, axis=0))
    return np.array(rows, dtype=np.float32)


def strong_null_margin(S, ctx_of, cur_of, tgt_of_k, tgt_dates_k, masks, k, boot, seed_tag=7):
    tr_all = np.where(masks["train"])[0]
    purged = tr_all[tgt_dates_k[tr_all] <= pd.Timestamp(TRAIN[1])]
    di = np.where(masks["dev"])[0]
    best = None
    for a in ALPHAS:
        reg = Ridge(alpha=a).fit(S[ctx_of[purged]].reshape(len(purged), -1), S[tgt_of_k[purged]])
        e = ((reg.predict(S[ctx_of[di]].reshape(len(di), -1)) - S[tgt_of_k[di]]) ** 2).sum(1)
        if best is None or e.mean() < best[0]:
            best = (e.mean(), a, reg)
    _, alpha, reg = best
    mean_v = S[np.unique(tgt_of_k[purged])].mean(0)
    cands = {"pers": lambda ii: S[cur_of[ii]], "mean": lambda ii: np.repeat(mean_v[None, :], len(ii), 0)}
    for lam in (0.25, 0.5, 0.75):
        cands[f"sh{lam}"] = (lambda l: lambda ii: l * S[cur_of[ii]] + (1 - l) * mean_v[None, :])(lam)
    nbest = None
    for nm, f in cands.items():
        e = ((f(di) - S[tgt_of_k[di]]) ** 2).sum(1)
        if nbest is None or e.mean() < nbest[0]:
            nbest = (e.mean(), nm, f)
    _, null_name, null_f = nbest
    out = {"alpha": alpha, "null": null_name}
    for w in ("hold", "oos"):
        ii = np.where(masks[w])[0]
        e_pred = ((reg.predict(S[ctx_of[ii]].reshape(len(ii), -1)) - S[tgt_of_k[ii]]) ** 2).sum(1)
        e_null = ((null_f(ii) - S[tgt_of_k[ii]]) ** 2).sum(1)
        d = e_null - e_pred
        _, se = block_z(d, block=boot, n_boot=1000, seed=seed_tag)
        out[w] = {"margin": round(float(d.mean() / (e_null.mean() + 1e-12)), 4),
                  "z": round(float(d.mean() / se), 2), "n": int(len(ii))}
    return out


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== WP-1 — incremental completeness: the breadth/dispersion knowledge block ===")
    r, obs, rf = load_v2()
    rows, b_end, idx = block_features(r, obs, B)
    valid = [i for i, x in enumerate(rows) if x is not None]
    X10 = np.array([rows[i] for i in valid], dtype=np.float32)
    dates_all = idx[b_end]
    bre = breadth_daily()
    XB = block_means(bre, idx, b_end, valid)
    ok = np.isfinite(XB).all(axis=1)
    X10, XB = X10[ok], XB[ok]
    valid = [v for v, o in zip(valid, ok) if o]
    remap = {i: j for j, i in enumerate(valid)}
    dates_v = dates_all[valid]
    m_tr_rows = np.asarray(dates_v <= pd.Timestamp(TRAIN[1]))

    def make_S(Xfull, D):
        mu, sd = Xfull[m_tr_rows].mean(0), Xfull[m_tr_rows].std(0) + 1e-9
        Z = (Xfull - mu) / sd
        pca = PCA(n_components=min(D, Z.shape[1]), random_state=0).fit(Z[m_tr_rows])
        return pca.transform(Z).astype(np.float32)

    def assemble_local(ctx_len):
        n = max(valid) + 1
        vset = set(valid)
        samples = np.array([i for i in range(ctx_len - 1, n - max(KS))
                            if all(j in vset for j in list(range(i - ctx_len + 1, i + 1)) + [i + k for k in KS])])
        dates = dates_all[samples]
        masks = {w: np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
                 for w, (a, b) in dict(train=TRAIN, dev=DEV, hold=HOLD, oos=OOS).items()}
        ctx_of = np.array([[remap[j] for j in range(i - ctx_len + 1, i + 1)] for i in samples])
        return {"samples": samples, "masks": masks, "ctx_of": ctx_of, "cur_of": ctx_of[:, -1],
                "tgt_of": {k: np.array([remap[i + k] for i in samples]) for k in KS},
                "tgt_dates": {k: dates_all[samples + k] for k in KS}}

    rng = np.random.default_rng(1234)
    yrs_rows = np.asarray(pd.DatetimeIndex(dates_v).year)
    report = {"preregistration": {"bar": "hold margin(+BREADTH) >= BASE + 0.02 AND > max noise-twin margin, at k=1 or k=4",
                                   "block": "frac_above_50dma, frac_hi20, frac_lo20, xdisp, up_frac (within-block means)",
                                   "protocol": "identified (dev-tuned alpha, strong nulls, purged, capacity-fair noise twins x3)"},
              "cells": {}}
    adds = {}
    for k in KS:
        asm = assemble_local(CTX[k])
        boot = CTX[k]
        S_base = make_S(X10, 10)
        base = strong_null_margin(S_base, asm["ctx_of"], asm["cur_of"], asm["tgt_of"][k],
                                  asm["tgt_dates"][k], asm["masks"], k, boot)
        S_br = make_S(np.concatenate([X10, XB], axis=1), 12)
        breadth = strong_null_margin(S_br, asm["ctx_of"], asm["cur_of"], asm["tgt_of"][k],
                                     asm["tgt_dates"][k], asm["masks"], k, boot)
        noise_margins = []
        for s in range(N_NOISE):
            XBn = XB.copy()
            uy = np.unique(yrs_rows); pm = dict(zip(uy, np.random.default_rng(500 + s).permutation(uy)))
            for y in uy:
                src = pm[y]
                src_rows = np.where(yrs_rows == src)[0]
                dst_rows = np.where(yrs_rows == y)[0]
                XBn[dst_rows] = XB[np.resize(src_rows, len(dst_rows))]
            S_n = make_S(np.concatenate([X10, XBn], axis=1), 12)
            nres = strong_null_margin(S_n, asm["ctx_of"], asm["cur_of"], asm["tgt_of"][k],
                                      asm["tgt_dates"][k], asm["masks"], k, boot)
            noise_margins.append(nres["hold"]["margin"])
        h_b, h_base = breadth["hold"]["margin"], base["hold"]["margin"]
        added = (h_b >= h_base + 0.02) and (h_b > max(noise_margins))
        adds[k] = added
        report["cells"][k] = {"base": base, "breadth": breadth,
                              "noise_hold_margins": noise_margins, "adds_information": bool(added)}
        print(f"  k={k}: BASE hold m {h_base} z {base['hold']['z']} | +BREADTH {h_b} z {breadth['hold']['z']} "
              f"| noise {noise_margins} -> {'ADDS' if added else 'no add'}")

    verdict = (("THE STATE WAS INCOMPLETE: the breadth block adds target-prediction information beyond "
                "capacity (knowledge->model binding works; wire the block into the state and re-run the "
                "battery) at k=" + ",".join(str(k) for k in KS if adds[k]))
               if any(adds.values()) else
               ("INCREMENTALLY COMPLETE w.r.t. breadth: the cross-sectional block adds nothing beyond "
                "capacity-fair noise on this substrate — the 10-feature state already spans what breadth "
                "carries at these horizons (a real answer to the sufficiency question, not a failure)."))
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
