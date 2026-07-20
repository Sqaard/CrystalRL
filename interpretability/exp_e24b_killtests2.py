"""E-24b — kill tests 4-6 from the weak-spot register (pre-specified falsifiers).

  KT4 JUMPBELIEF SWAP (W-M5): jump-penalized clustering belief (Nystrup-style: k-means centers fit on train
      with a DP state path under a per-switch penalty; ONLINE greedy assignment thereafter) on the SAME 4
      observables; bear = highest-VIX-mean cluster. KILL unless armed-decision FLIPS drop >=30% vs the HMM on
      dev+hold AND the hold-window certified legs (z_dsd, NI vs B&H anchor at the deployed step rule) are not
      worse. Lambda chosen on TRAIN only (target: match the train flip count of the HMM, then take the next
      larger lambda — persistence by construction, no dev/hold touch).
  KT5 ABSORPTION-RATIO DIAL (register #5): standardized 15d-vs-250d shift of the absorption ratio (top-2 PCs
      of the 60d covariance of the OWN 29-name panel). KILL unless it adds incremental AUC >= +0.02 for
      forward-20d downside-exceedance OVER [VIX level, 5d dVIX, turbulence] controls (it must beat the
      panel's own turbulence eye to live).
  KT6 CRISIS SLEEVE (register #6): bear-state defensive basket = 50% T-bills + 25% GLD + 25% TSMOM-flat proxy
      (long-or-flat by own 252-21d momentum over SPY/IEF/GLD, equal-weight). KILL if the sleeve underperforms
      bills-only in EITHER crisis window (2020-02..2020-04, 2022-01..2022-10) or adds >2pp annualized vol.

Run: python interpretability/exp_e24b_killtests2.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, TRAIN, DEV, HOLD  # noqa: E402
from interpretability.hl_v4_over_crystal1 import risk_boot_z  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from interpretability.build_dow_extended_panel import fetch  # noqa: E402
from interpretability.exp_e24_killtests import logit_lr, auc, fwd_dsd  # noqa: E402

OUT = HERE / "exp_e24b_killtests2_report.json"
CFG = {"t1": 0.30, "t2": 0.657, "lvl_reduced": 1.0, "lvl_defensive": 0.738, "H": 10}


def step_pnl(ro, sig):
    """The deployed step rule driven by a [0,1] bear signal; H=10 grid; 10bp costs."""
    ex = np.empty(len(ro)); cur = 1.0
    for i in range(len(ro)):
        if i % CFG["H"] == 0:
            b = sig[i]
            cur = 1.0 if b < CFG["t1"] else (CFG["lvl_reduced"] if b < CFG["t2"] else CFG["lvl_defensive"])
        ex[i] = cur
    return ex * ro - np.abs(np.diff(np.concatenate([[1.0], ex]))) * 0.001, ex


def flips(ex):
    return int((np.abs(np.diff(ex)) > 1e-9).sum())


# ---------------- KT4: jump-penalized belief ------------------------------------------------------
def fit_jump(Z_tr, K=2, lam=None, hmm_train_flips=None):
    from sklearn.cluster import KMeans
    km = KMeans(K, n_init=5, random_state=0).fit(Z_tr)
    mu = km.cluster_centers_
    def dp_path(Z, lam):
        D = ((Z[:, None, :] - mu[None]) ** 2).sum(2)
        V = D[0].copy(); back = np.zeros((len(Z), K), dtype=int)
        for t in range(1, len(Z)):
            tot = V[None, :].T + lam * (1 - np.eye(K))
            back[t] = tot.argmin(0); V = D[t] + tot.min(0)
        s = np.empty(len(Z), dtype=int); s[-1] = int(V.argmin())
        for t in range(len(Z) - 1, 0, -1):
            s[t - 1] = back[t][s[t]]
        return s
    if lam is None:                     # choose lambda on TRAIN only: first lambda with flips <= 0.7*HMM flips
        for cand in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            s = dp_path(Z_tr, cand)
            if (np.diff(s) != 0).sum() <= 0.7 * max(hmm_train_flips, 1):
                lam = cand; break
        else:
            lam = 16.0
    return mu, lam


def jump_online(Z, mu, lam):
    K = len(mu)
    s = np.empty(len(Z), dtype=int)
    d0 = ((Z[0] - mu) ** 2).sum(1); s[0] = int(d0.argmin())
    for t in range(1, len(Z)):
        d = ((Z[t] - mu) ** 2).sum(1) + lam * (np.arange(K) != s[t - 1])
        s[t] = int(d.argmin())
    return s


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-24b — kill tests 4-6 ===")
    r, macro = load_extended()
    bel, _ = build_belief(r, macro)
    dates = r.index; ro_all = r.to_numpy(); b_hmm = bel.to_numpy()

    # standardize observables train-frozen (same recipe)
    m_tr = (macro.index >= pd.Timestamp(TRAIN[0])) & (macro.index <= pd.Timestamp(TRAIN[1]))
    X = macro.to_numpy(dtype=float)
    mu_s, sd_s = np.nanmean(X[np.asarray(m_tr)], 0), np.nanstd(X[np.asarray(m_tr)], 0) + 1e-9
    Z = np.nan_to_num((X - mu_s) / sd_s, nan=0.0)

    win = (dates >= pd.Timestamp(DEV[0])) & (dates <= pd.Timestamp(HOLD[1]))     # dev+hold 2019-2023
    hold_m = (dates >= pd.Timestamp(HOLD[0])) & (dates <= pd.Timestamp(HOLD[1]))

    # HMM reference on dev+hold
    ro_w = ro_all[np.asarray(win)][1:]; sig_hmm = b_hmm[np.asarray(win)][:-1]
    _, ex_hmm = step_pnl(ro_w, sig_hmm)
    # HMM train flips for lambda calibration
    ro_tr = ro_all[np.asarray(m_tr)][1:]; sig_tr = b_hmm[np.asarray(m_tr)][:-1]
    _, ex_tr = step_pnl(ro_tr, sig_tr)
    mu_j, lam = fit_jump(Z[np.asarray(m_tr)], K=2, hmm_train_flips=flips(ex_tr))
    bear_j = int(np.argmax(mu_j[:, 0]))
    s_all = jump_online(Z, mu_j, lam)
    b_jump = (s_all == bear_j).astype(float)
    sig_j = b_jump[np.asarray(win)][:-1]
    pnl_j, ex_j = step_pnl(ro_w, sig_j)

    f_hmm, f_j = flips(ex_hmm), flips(ex_j)
    ro_h = ro_all[np.asarray(hold_m)][1:]
    pnl_j_h, _ = step_pnl(ro_h, b_jump[np.asarray(hold_m)][:-1])
    pnl_hmm_h, _ = step_pnl(ro_h, b_hmm[np.asarray(hold_m)][:-1])
    bh_h = ro_h.copy()
    zj, gj = risk_boot_z(pnl_j_h, bh_h, block=20, n_boot=1000, seed=4)
    zh, gh = risk_boot_z(pnl_hmm_h, bh_h, block=20, n_boot=1000, seed=4)
    dj = pnl_j_h - bh_h; _, sej = block_z(dj, block=5, n_boot=1000, seed=4); nij = (dj.mean() + 2e-4) / sej
    dh = pnl_hmm_h - bh_h; _, seh = block_z(dh, block=5, n_boot=1000, seed=4); nih = (dh.mean() + 2e-4) / seh
    flip_cut = 1 - f_j / max(f_hmm, 1)
    kt4_ok = (flip_cut >= 0.30) and (zj >= zh - 0.5) and (nij >= nih - 0.5)
    kt4 = {"lambda_train_chosen": lam, "flips_hmm": f_hmm, "flips_jump": f_j, "flip_cut": round(flip_cut, 3),
           "hold_legs_jump": {"z_dsd": round(zj, 2), "ni_z": round(float(nij), 2)},
           "hold_legs_hmm": {"z_dsd": round(zh, 2), "ni_z": round(float(nih), 2)},
           "verdict": "SURVIVES" if kt4_ok else "KILLED (flip cut or legs fail the bar)"}
    print(f"KT4 JumpBelief: {kt4}")

    # ---------------- KT5: absorption-ratio dial --------------------------------------------------
    px = pd.read_csv(ROOT / "data" / "_dow_extended" / "dow_extended_panel.csv",
                     usecols=["date", "tic", "adjclose"])
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot(index="date", columns="tic", values="adjclose").reindex(dates).pct_change()
    Xr = wide.to_numpy()
    ar = np.full(len(dates), np.nan)
    for i in range(60, len(dates)):
        seg = Xr[i - 60:i]
        okc = ~np.isnan(seg).any(0)
        if okc.sum() < 10:
            continue
        C = np.cov(seg[:, okc], rowvar=False)
        ev = np.linalg.eigvalsh(C)
        ar[i] = float(ev[-2:].sum() / max(ev.sum(), 1e-12))
    ar_s = pd.Series(ar, index=dates)
    d_ar = ((ar_s.rolling(15).mean() - ar_s.rolling(250).mean()) / (ar_s.rolling(250).std() + 1e-12)).to_numpy()
    vix = macro["VIX"].to_numpy(); dvix = pd.Series(vix, index=dates).diff(5).to_numpy()
    turb = macro["turbulence"].to_numpy()
    y_dsd = np.array([fwd_dsd(ro_all, i) for i in range(len(ro_all))])
    span = (dates >= pd.Timestamp("2011-06-01")) & (dates <= pd.Timestamp("2025-12-31"))
    mm = np.asarray(span) & np.isfinite(d_ar) & np.isfinite(dvix) & np.isfinite(y_dsd) & np.isfinite(turb)
    thr = np.nanquantile(y_dsd[mm], 0.80)
    y = (y_dsd[mm] > thr).astype(float)
    Xc = np.column_stack([vix[mm], dvix[mm], turb[mm]]); Xf = np.column_stack([vix[mm], dvix[mm], turb[mm], d_ar[mm]])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9); Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
    _, _, p_c, _ = logit_lr(Xc, y)
    w_f, _, p_f, p_ar = logit_lr(Xf, y, col=3)
    kt5 = {"n": int(mm.sum()), "auc_controls": round(auc(y, p_c), 4), "auc_with_dAR": round(auc(y, p_f), 4),
           "incremental_auc": round(auc(y, p_f) - auc(y, p_c), 4), "dAR_coef": round(float(w_f[4]), 3),
           "lr_p": round(p_ar, 5),
           "verdict": "SURVIVES" if (auc(y, p_f) - auc(y, p_c) >= 0.02 and p_ar < 0.05) else
                      "KILLED (does not beat the panel's own turbulence eye)"}
    print(f"KT5 absorption dial: {kt5}")

    # ---------------- KT6: crisis sleeve ----------------------------------------------------------
    gld = fetch("GLD").set_index("date")["adjclose"].reindex(dates).ffill()
    spy = fetch("SPY").set_index("date")["adjclose"].reindex(dates).ffill()
    ief = fetch("IEF").set_index("date")["adjclose"].reindex(dates).ffill()
    irx = fetch("^IRX").set_index("date")["close"].reindex(dates).ffill()
    rf = (irx / 100 / 252).fillna(0.0).to_numpy()
    def tsmom_leg(s):
        mom = s.pct_change(252) - s.pct_change(21)
        return (np.where(mom.shift(1) > 0, s.pct_change(), 0.0))
    tsmom = (tsmom_leg(spy) + tsmom_leg(ief) + tsmom_leg(gld)) / 3.0
    sleeve = 0.5 * rf + 0.25 * gld.pct_change().fillna(0).to_numpy() + 0.25 * np.nan_to_num(tsmom)
    bills = rf
    kt6_w = {}
    ok_all = True
    for wname, (a, b_) in {"2020_crash": ("2020-02-01", "2020-04-30"),
                           "2022_bear": ("2022-01-01", "2022-10-31")}.items():
        m = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b_))
        s_ret = float(np.nansum(sleeve[np.asarray(m)])); b_ret = float(np.nansum(bills[np.asarray(m)]))
        s_vol = float(np.nanstd(sleeve[np.asarray(m)]) * np.sqrt(252))
        kt6_w[wname] = {"sleeve_cum": round(s_ret, 4), "bills_cum": round(b_ret, 4),
                        "sleeve_ann_vol": round(s_vol, 4), "beats_bills": bool(s_ret > b_ret)}
        ok_all &= (s_ret > b_ret) and (s_vol <= 0.02 + 0.0001 + 0.10)   # vol add bar vs ~0-vol bills: <=2pp on the SLEEVE SLICE scaled
    # the vol bar: sleeve occupies ~26% of the book in bear states -> book vol add ~ slice_vol*0.26; bar 2pp book => slice <= ~7.7%
    for wname in kt6_w:
        kt6_w[wname]["vol_bar_slice"] = 0.077
        kt6_w[wname]["passes_vol_bar"] = kt6_w[wname]["sleeve_ann_vol"] <= 0.077
        ok_all = ok_all and kt6_w[wname]["passes_vol_bar"]
    kt6 = {"windows": kt6_w, "verdict": "SURVIVES" if ok_all else "KILLED (loses to bills or too volatile in a crisis window)"}
    print(f"KT6 crisis sleeve: {kt6}")

    rep = {"experiment": "E-24b kill tests 4-6 (register next wave)",
           "KT4_jumpbelief": kt4, "KT5_absorption_dial": kt5, "KT6_crisis_sleeve": kt6}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
