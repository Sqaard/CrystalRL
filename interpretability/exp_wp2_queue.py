"""WP-2 — the preregistered candidate queue through the completeness harness (all four blocks).

Per experience/ECONOMIC_WORLD_PICTURE_V2_FOR_BOT.md §9. Same protocol as WP-1 (identified:
dev-tuned alpha, strong null family, purged fits, capacity-fair year-shuffled noise twins x3),
same PREREGISTERED BAR per candidate: ADDS iff on HOLD at k=1 or k=4
margin(+CAND) >= margin(BASE) + 0.02 AND > max noise-twin margin. OOS = confirmation only.

CANDIDATES (daily constructions -> within-block means; no raw levels):
  credit    : hyg_lqd_21 = 21d adj-return spread HYG-LQD; lqd_ief_21 = LQD-IEF (duration strip).
              W-G2 note: adj returns embed carry - acceptable here because the test is
              representation-INFORMATION (with capacity twins), not pnl; any deployable use
              still faces the carry-matched battery at the gate.
  stockbond : rolling 63d corr(book ret, TLT ret) and (book, IEF) - the inflation-regime flag.
  plumbing  : mean pairwise 5d correlation of Dow names; 1d VIX %change; VIX/VIX3M ratio.
  complacency: normalized run-length of days with VIX < rolling 504d median; VIX3M/VIX.

Run: python interpretability/exp_wp2_queue.py     (~8 min)
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
from interpretability.exp_wp1_completeness import strong_null_margin  # noqa: E402
from interpretability.build_dow_extended_panel import fetch  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from sklearn.decomposition import PCA

OUT = HERE / "exp_wp2_queue_report.json"
PANEL = ROOT / "data" / "_dow_extended" / "dow_extended_panel_v2.csv"
B = 5
KS = (1, 4)
CTX = {1: 2, 4: 4}
N_NOISE = 3
D_AUG = 12


def etf_ret(sym, idx):
    d = fetch(sym).set_index("date")["adjclose"].reindex(idx).ffill()
    return d.pct_change()


def candidate_dailies(r, obs, idx):
    """All candidate daily columns, causal at t."""
    out = {}
    hyg, lqd, ief, tlt = (etf_ret(s, idx) for s in ("HYG", "LQD", "IEF", "TLT"))
    out["credit"] = pd.DataFrame({
        "hyg_lqd_21": hyg.rolling(21).sum() - lqd.rolling(21).sum(),
        "lqd_ief_21": lqd.rolling(21).sum() - ief.rolling(21).sum()})
    out["stockbond"] = pd.DataFrame({
        "corr63_tlt": r.rolling(63).corr(tlt), "corr63_ief": r.rolling(63).corr(ief)})
    df = pd.read_csv(PANEL, usecols=["date", "tic", "adjclose"])
    df["date"] = pd.to_datetime(df["date"])
    w = df.pivot_table(index="date", columns="tic", values="adjclose", aggfunc="first").sort_index()
    rr = w.pct_change().reindex(idx)
    R = rr.to_numpy()
    pc = np.full(len(idx), np.nan)
    for t in range(5, len(idx)):
        seg = R[t - 4:t + 1]
        ok = ~np.isnan(seg).any(axis=0)
        if ok.sum() < 10:
            continue
        Z = seg[:, ok]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        n = Z.shape[1]
        s = Z.sum(1)
        pc[t] = float(((s ** 2).sum() / 5 - n) / (n * (n - 1)))
    vix = obs["VIX"]
    vix3 = fetch("^VIX3M").set_index("date")["close"].reindex(idx).ffill()
    out["plumbing"] = pd.DataFrame({
        "paircorr5": pd.Series(pc, index=idx),
        "dvix1": vix.pct_change(), "bwd": vix / vix3.replace(0, np.nan)})
    med504 = vix.rolling(504).median()
    below = (vix < med504).astype(float)
    run = below.copy()
    run_v = run.to_numpy()
    for t in range(1, len(run_v)):
        if run_v[t] > 0:
            run_v[t] = run_v[t - 1] + 1
    out["complacency"] = pd.DataFrame({
        "calm_clock": np.log1p(run_v) / np.log(504.0),
        "vix3_ratio": (vix3 / vix.replace(0, np.nan))})
    return out


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== WP-2 — the preregistered candidate queue through the completeness harness ===")
    r, obs, rf = load_v2()
    idx = r.index
    rows, b_end, idx_b = block_features(r, obs, B)
    valid0 = [i for i, x in enumerate(rows) if x is not None]
    dailies = candidate_dailies(r, obs, idx)

    report = {"preregistration": {"bar": "hold margin(+CAND) >= BASE + 0.02 AND > max of 3 noise twins, at k=1 or k=4",
                                   "protocol": "WP-1 identified harness; D_aug=12 uniform; within-block means; no raw levels",
                                   "wg2_note": "credit adj-returns embed carry - information test only; deployable use faces the carry battery"},
              "candidates": {}}
    verdicts = {}
    for cname, D in dailies.items():
        arr = D.reindex(idx).to_numpy(dtype=float)
        Xc_rows, ok_rows = [], []
        for i in valid0:
            e = b_end[i]; s = e - B + 1
            seg = arr[s:e + 1]
            m = np.nanmean(seg, axis=0)
            Xc_rows.append(m); ok_rows.append(np.isfinite(m).all())
        Xc = np.array(Xc_rows, dtype=np.float32)
        ok = np.array(ok_rows)
        valid = [v for v, o in zip(valid0, ok) if o]
        X10 = np.array([rows[i] for i in valid], dtype=np.float32)
        Xc = Xc[ok]
        remap = {i: j for j, i in enumerate(valid)}
        dates_v = idx_b[b_end[valid]]
        m_tr_rows = np.asarray(dates_v <= pd.Timestamp(TRAIN[1]))
        yrs_rows = np.asarray(pd.DatetimeIndex(dates_v).year)

        def make_S(Xfull, Dd):
            mu, sd = Xfull[m_tr_rows].mean(0), Xfull[m_tr_rows].std(0) + 1e-9
            Z = (Xfull - mu) / sd
            p = PCA(n_components=min(Dd, Z.shape[1]), random_state=0).fit(Z[m_tr_rows])
            return p.transform(Z).astype(np.float32)

        def assemble_local(ctx_len):
            n = max(valid) + 1
            vset = set(valid)
            samples = np.array([i for i in range(ctx_len - 1, n - max(KS))
                                if all(j in vset for j in list(range(i - ctx_len + 1, i + 1)) + [i + k for k in KS])])
            dts = idx_b[b_end[samples]]
            masks = {w: np.asarray((dts >= pd.Timestamp(a)) & (dts <= pd.Timestamp(b)))
                     for w, (a, b) in dict(train=TRAIN, dev=DEV, hold=HOLD, oos=OOS).items()}
            ctx_of = np.array([[remap[j] for j in range(i - ctx_len + 1, i + 1)] for i in samples])
            return {"masks": masks, "ctx_of": ctx_of, "cur_of": ctx_of[:, -1],
                    "tgt_of": {k: np.array([remap[i + k] for i in samples]) for k in KS},
                    "tgt_dates": {k: idx_b[b_end[samples + k]] for k in KS}}

        crow = {}
        added_any = False
        for k in KS:
            asm = assemble_local(CTX[k])
            boot = CTX[k]
            base = strong_null_margin(make_S(X10, 10), asm["ctx_of"], asm["cur_of"], asm["tgt_of"][k],
                                      asm["tgt_dates"][k], asm["masks"], k, boot)
            cand = strong_null_margin(make_S(np.concatenate([X10, Xc], axis=1), D_AUG),
                                      asm["ctx_of"], asm["cur_of"], asm["tgt_of"][k],
                                      asm["tgt_dates"][k], asm["masks"], k, boot)
            noise = []
            for s in range(N_NOISE):
                Xn = Xc.copy()
                uy = np.unique(yrs_rows)
                pm = dict(zip(uy, np.random.default_rng(700 + s).permutation(uy)))
                for y in uy:
                    src_rows = np.where(yrs_rows == pm[y])[0]
                    dst_rows = np.where(yrs_rows == y)[0]
                    Xn[dst_rows] = Xc[np.resize(src_rows, len(dst_rows))]
                nres = strong_null_margin(make_S(np.concatenate([X10, Xn], axis=1), D_AUG),
                                          asm["ctx_of"], asm["cur_of"], asm["tgt_of"][k],
                                          asm["tgt_dates"][k], asm["masks"], k, boot)
                noise.append(nres["hold"]["margin"])
            hb, hc = base["hold"]["margin"], cand["hold"]["margin"]
            adds = (hc >= hb + 0.02) and (hc > max(noise))
            added_any = added_any or adds
            crow[k] = {"base": base, "cand": cand, "noise_hold": noise, "adds": bool(adds)}
            print(f"  {cname:11s} k={k}: BASE {hb} z {base['hold']['z']} | +CAND {hc} z {cand['hold']['z']} "
                  f"| noise {noise} -> {'ADDS' if adds else 'no'}")
        verdicts[cname] = added_any
        report["candidates"][cname] = crow

    survivors = [c for c, v in verdicts.items() if v]
    verdict = ((f"SURVIVORS: {survivors} — wire into the frozen state and re-run the W-battery "
                "(the picture's §9 step 2-3); the rest are incrementally-spanned.")
               if survivors else
               ("ALL FOUR NO-ADD: the 10-feature state is incrementally complete w.r.t. the entire "
                "knowledge-derived candidate queue at 5-20d horizons — a strong, honest sufficiency "
                "statement: what the corpus names as first-order state is already spanned (largely "
                "via VIX/vol/ebp channels) at these timescales. The binding pipeline WORKS (it "
                "correctly refuses redundant knowledge); enrichment of the STATE moves to horizons "
                "or data we do not yet have (monthly cells were power-limited here)."))
    report["survivors"] = survivors
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
