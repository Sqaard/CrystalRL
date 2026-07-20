"""E-24 — kill tests 1-3 from the weak-spot register (pre-specified falsifiers; cheapest first; no gate runs).

  KT1 BACKWARDATION MATCHED-CONTROL (W-M4): all 2010-2025 days with VIX/VIX3M > 0.95, matched 1:1 to a
      same-VIX-level (±1pt) contango day (ratio <= 0.95). Compare forward 20d downside deviation of the Dow-29
      EW book + P(entering the deployed bear state within 10d). KILL if the matched dsd gap is not positive at
      p<0.05 (paired bootstrap) — else the ratio is just the level in disguise.
  KT2 BEAR-AGE HAZARD (W-M3): discrete-time logistic hazard of bear-spell EXIT on spell age, on OUR OWN
      filtered spells in the 2010-2018 TRAIN window (per negative knowledge: hazards re-validate on our spells
      or die). KILL if the age coefficient is not positive at p<0.1 (likelihood-ratio test). Underpowered =
      killed (the burden is on the hypothesis).
  KT3 CREDIT-CONFIRMATION AUC (register next-wave #4): logistic of a forward-20d downside-deviation-exceedance
      flag (top-quintile) on the 20d HYG-IEF relative-return z-score, CONTROLLING VIX level + 5d dVIX,
      2010-2025. KILL if incremental AUC < +0.02 over controls-only OR the credit coefficient loses
      significance (LR p>0.05) under the controls.

Run: python interpretability/exp_e24_killtests.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v9_fresh_oos import load_extended, build_belief  # noqa: E402
from interpretability.build_dow_extended_panel import fetch  # noqa: E402

OUT = HERE / "exp_e24_killtests_report.json"
T2_DEPLOYED = 0.657


def fwd_dsd(ro, i, h=20):
    seg = ro[i + 1:i + 1 + h]
    if len(seg) < h:
        return np.nan
    neg = np.minimum(seg, 0.0)
    return float(np.sqrt((neg ** 2).mean()))


def logit_lr(X, y, col=None):
    """Logistic fit via IRLS + likelihood-ratio p-value for dropping `col` (no statsmodels dependency)."""
    def fit(Xd):
        Xd = np.column_stack([np.ones(len(Xd)), Xd])
        w = np.zeros(Xd.shape[1])
        for _ in range(60):
            p = 1 / (1 + np.exp(-Xd @ w))
            W = np.clip(p * (1 - p), 1e-9, None)
            H = Xd.T @ (Xd * W[:, None]) + 1e-8 * np.eye(Xd.shape[1])
            g = Xd.T @ (y - p)
            step = np.linalg.solve(H, g)
            w += step
            if np.abs(step).max() < 1e-9:
                break
        p = np.clip(1 / (1 + np.exp(-Xd @ w)), 1e-12, 1 - 1e-12)
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return w, ll, p
    w_full, ll_full, p_full = fit(X)
    if col is None:
        return w_full, ll_full, p_full, None
    Xr = np.delete(X, col, axis=1)
    _, ll_red, _ = fit(Xr)
    from scipy.stats import chi2
    lr = 2 * (ll_full - ll_red)
    return w_full, ll_full, p_full, float(chi2.sf(max(lr, 0), df=1))


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-24 — kill tests 1-3 (pre-specified falsifiers) ===")
    r, macro = load_extended()
    bel, _ = build_belief(r, macro)
    ro = r.to_numpy(); dates = r.index
    vix = macro["VIX"].to_numpy()
    vix3 = fetch("^VIX3M").set_index("date")["close"].reindex(dates).ffill().to_numpy()
    ratio = vix / np.where(vix3 > 0, vix3, np.nan)
    span = (dates >= pd.Timestamp("2010-06-01")) & (dates <= pd.Timestamp("2025-12-31"))
    idxs = np.where(span)[0]

    # ---------------- KT1: backwardation matched-control ----------------
    backw = [i for i in idxs if np.isfinite(ratio[i]) and ratio[i] > 0.95]
    cont = [i for i in idxs if np.isfinite(ratio[i]) and ratio[i] <= 0.95]
    cont_by_vix = sorted(cont, key=lambda i: vix[i])
    cont_vix = np.array([vix[i] for i in cont_by_vix])
    used = set(); pairs = []
    for i in backw:
        lo = np.searchsorted(cont_vix, vix[i] - 1.0); hi = np.searchsorted(cont_vix, vix[i] + 1.0)
        cands = [cont_by_vix[j] for j in range(lo, hi) if cont_by_vix[j] not in used]
        if not cands:
            continue
        j = min(cands, key=lambda c: abs(vix[c] - vix[i]))
        used.add(j); pairs.append((i, j))
    d_dsd, d_bear = [], []
    for i, j in pairs:
        a, b = fwd_dsd(ro, i), fwd_dsd(ro, j)
        if np.isfinite(a) and np.isfinite(b):
            d_dsd.append(a - b)
        eb = float((bel.to_numpy()[i + 1:i + 11] > T2_DEPLOYED).any()) if i + 11 <= len(ro) else np.nan
        cb = float((bel.to_numpy()[j + 1:j + 11] > T2_DEPLOYED).any()) if j + 11 <= len(ro) else np.nan
        if np.isfinite(eb) and np.isfinite(cb):
            d_bear.append(eb - cb)
    d_dsd = np.array(d_dsd)
    rng = np.random.default_rng(24)
    boots = [d_dsd[rng.integers(0, len(d_dsd), len(d_dsd))].mean() for _ in range(4000)]
    p_kt1 = float(np.mean(np.array(boots) <= 0))
    kt1 = {"n_backwardation_days": len(backw), "n_matched_pairs": len(pairs),
           "mean_fwd20_dsd_gap_bp": round(float(d_dsd.mean()) * 1e4, 2),
           "p_one_sided": round(p_kt1, 4),
           "bear_entry_10d_gap": round(float(np.mean(d_bear)), 3) if d_bear else None,
           "verdict": "SURVIVES" if p_kt1 < 0.05 and d_dsd.mean() > 0 else "KILLED (the ratio is the level in disguise)"}
    print(f"KT1 backwardation: pairs={kt1['n_matched_pairs']} dsd gap {kt1['mean_fwd20_dsd_gap_bp']}bp "
          f"p={kt1['p_one_sided']} bear-entry gap {kt1['bear_entry_10d_gap']} -> {kt1['verdict']}")

    # ---------------- KT2: bear-age hazard on OWN train spells ----------------
    b = bel.to_numpy()
    tr = (dates >= pd.Timestamp("2010-01-01")) & (dates <= pd.Timestamp("2018-12-31"))
    bear_day = (b > 0.5) & np.asarray(tr)
    ages, exits = [], []
    age = 0
    for t in range(len(b) - 1):
        if bear_day[t]:
            age += 1
            ages.append(age); exits.append(0.0 if bear_day[t + 1] else 1.0)
        else:
            age = 0
    ages = np.array(ages, dtype=float); exits = np.array(exits)
    n_spells = int((np.diff(np.concatenate([[0], bear_day.astype(int)])) == 1).sum())
    if len(ages) >= 20 and exits.sum() >= 3:
        X = (ages[:, None] - ages.mean()) / (ages.std() + 1e-9)
        w, ll, _, p_age = logit_lr(X, exits, col=0)
        coef = float(w[1])
        kt2 = {"n_bear_days": int(len(ages)), "n_spells": n_spells, "n_exits": int(exits.sum()),
               "age_coef": round(coef, 3), "lr_p": round(p_age, 4),
               "verdict": "SURVIVES" if (coef > 0 and p_age < 0.1) else "KILLED (no positive duration dependence on our spells)"}
    else:
        kt2 = {"n_bear_days": int(len(ages)), "n_spells": n_spells, "n_exits": int(exits.sum()),
               "verdict": "KILLED (underpowered on 2010-18 train spells — burden on the hypothesis)"}
    print(f"KT2 bear-age hazard: {kt2}")

    # ---------------- KT3: credit-confirmation incremental AUC ----------------
    hyg = fetch("HYG").set_index("date")["adjclose"].reindex(dates).ffill()
    ief = fetch("IEF").set_index("date")["adjclose"].reindex(dates).ffill()
    rel = (hyg.pct_change() - ief.pct_change()).rolling(20).sum()
    credit_z = ((rel - rel.rolling(250).mean()) / (rel.rolling(250).std() + 1e-12)).to_numpy()
    dvix = pd.Series(vix, index=dates).diff(5).to_numpy()
    y_dsd = np.array([fwd_dsd(ro, i) for i in range(len(ro))])
    thr = np.nanquantile(y_dsd[span], 0.80)
    m = span & np.isfinite(credit_z) & np.isfinite(dvix) & np.isfinite(y_dsd)
    y = (y_dsd[m] > thr).astype(float)
    Xc = np.column_stack([vix[m], dvix[m]])
    Xf = np.column_stack([vix[m], dvix[m], credit_z[m]])
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9); Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
    _, _, p_ctrl, _ = logit_lr(Xc, y)
    w_full, _, p_full_prob, p_credit = logit_lr(Xf, y, col=2)
    auc_c, auc_f = auc(y, p_ctrl), auc(y, p_full_prob)
    kt3 = {"n": int(m.sum()), "auc_controls": round(auc_c, 4), "auc_with_credit": round(auc_f, 4),
           "incremental_auc": round(auc_f - auc_c, 4), "credit_coef": round(float(w_full[3]), 3),
           "lr_p": round(p_credit, 5),
           "verdict": "SURVIVES" if (auc_f - auc_c >= 0.02 and p_credit < 0.05) else
                      "KILLED (credit adds nothing the vol block doesn't have)"}
    print(f"KT3 credit AUC: {kt3}")

    rep = {"experiment": "E-24 kill tests 1-3 (register next wave; pre-specified)",
           "KT1_backwardation_matched_control": kt1, "KT2_bear_age_hazard": kt2, "KT3_credit_auc": kt3}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
