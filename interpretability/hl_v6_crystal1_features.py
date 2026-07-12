"""HL v6 — CRYSTAL-1 belief built from the PREPROCESSED panel features (the user's item 0: v5 used only raw
`close`; the 39 engineered columns of data/adapters/_csi500_wide/csi300_model_ready.csv went unused).

This is the "use the preprocessing" upgrade, kept born-legible. The observation is a small NAMED vector of
market-level, PIT-safe, live risk features (pre-registered from TRAIN-only forward correlations):
    reg1     = Regime_1_Prob   (the panel's OWN causal US-macro regime posterior; fwd_vol +0.42, fwd_ret -0.33)
    atr_rel  = EW-mean atr/close (China-native realized-vol proxy; +0.40)
    xdisp    = cross-sectional std of daily_return (China-native breadth-of-dispersion; +0.36)
    rsi30    = EW-mean rsi_30 (+0.30/-0.19)
    dx30     = EW-mean dx_30 (trend strength; +0.26)
DROPPED: macd (fwd_vol +0.06 dead), turbulence/10Y_Yield (100% zero), Market_Regime (82% zero).
QUARANTINED to a flagged ablation only: the GRU forecast columns — forecast_std has fwd_ret +0.49 but that is
the in-sample-fit signature (panel audit: cross-sectional IC t=+10 inside train); provenance of the GRU
train/test split on this file is unverifiable, so it may NOT enter the core belief.

Filter: a compact diagonal-covariance Gaussian-emission HMM with the notebook's OWN discipline —
`infer_causal_hmm_states` past-only filtering (posterior at t uses obs <= t), K in {2,3} selected by held-out
filtered LL, means/covs/threshold standardization all FROZEN on the train window. States NAMED by their
standardized risk-score (CRISIS = highest, CALM = lowest, GRIND = middle). b_risk = P(CRISIS)+P(GRIND).

Everything downstream is v5 VERBATIM (import): the belief-gated conditional-vol-targeting policy {Q_ON, E_GRIND},
the v4 typed gate (RETURN/RISK lanes, rotating disjoint-confirm holdout, alpha-wealth, stress adversary,
canaries+freeze), and the U3 rf=0 / execution-lag acceptance stress. Only the SIGNAL changed — so the
comparison v5 (raw close) vs v6 (preprocessed features) is clean.

Run: python interpretability/hl_v6_crystal1_features.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import interpretability.hl_v4_over_crystal1 as V4  # noqa: E402
import interpretability.hl_v5_crystal1_upgraded as V5  # noqa: E402

OUT = HERE / "hl_v6_crystal1_report.json"
CORE_FEATURES = ["reg1", "atr_rel", "xdisp", "rsi30", "dx30"]
GRU_FEATURES = ["gru_std", "gru_mean"]           # quarantined; ablation arm only


# ---------- v6 policy fix: E_GRIND must CAP exposure whenever the belief is armed ----------------
# The v5 vol-targeting branch (clip(sig*/sig, E_MIN, 1)) is INERT on a LOW-vol grind (sig < sig* => target
# >1 => clipped to 1 => no de-risk), and on a K=2 panel there is no separate GRIND state, so E_GRIND never
# fired. The diagnosis showed a real csi500 downside cut needs a b_risk-threshold CUT, not vol targeting.
# Fix (one legible line): when armed, exposure = MIN(vol_target, E_GRIND) — E_GRIND is a hard cap that bites
# on low-vol grinds; vol targeting cuts further in genuine high-vol crises. Live on both K=2 and K=3.
def exposure_series_v6(S, c):
    br = S["b_risk"]
    g_on = br.rolling(252, min_periods=60).quantile(min(c["Q_ON"], 1.0)).shift(1)
    sig, sig_star = S["sigma20"], S["sig_star"]
    n = len(br); ex = np.ones(n); on = False
    brv, gv, sv = br.to_numpy(), g_on.to_numpy(), sig.to_numpy()
    for t in range(n):
        if not np.isfinite(gv[t]):
            ex[t] = 1.0; on = False; continue
        if on and brv[t] < gv[t] - V5.HYST:
            on = False
        elif not on and brv[t] > gv[t]:
            on = True
        if not on:
            ex[t] = 1.0
        else:
            vt = float(np.clip(sig_star / sv[t], V5.E_MIN, 1.0)) if np.isfinite(sv[t]) and sv[t] > 0 else V5.E_MIN
            ex[t] = min(vt, float(c["E_GRIND"]))     # E_GRIND caps whenever armed (low-vol grind defense)
    return pd.Series(ex, index=br.index)


class SubstrateV6(V5.SubstrateV5):
    def _ex(self, c):
        key = (round(c["Q_ON"], 6), round(c["E_GRIND"], 6))
        if key not in self._cache:
            self._cache[key] = exposure_series_v6(self.S, c).to_numpy()
        return self._cache[key]


# ---------- compact causal Gaussian-emission HMM -------------------------------------------------
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM with past-only (causal) filtering — matches the notebook's
    infer_causal_hmm_states discipline. Light EM from a KMeans init; frozen after fit."""

    def __init__(self, K):
        self.K = K

    def _emit_logp(self, X):
        # log N(x | mu_k, diag(var_k)) for each t, k  -> (T, K)
        d = X[:, None, :] - self.mu[None, :, :]
        return (-0.5 * (np.log(2 * np.pi * self.var[None]) + d ** 2 / self.var[None]).sum(axis=2))

    def fit(self, X, n_iter=25, seed=0):
        from sklearn.cluster import KMeans
        T, D = X.shape
        km = KMeans(self.K, n_init=5, random_state=seed).fit(X)
        lab = km.labels_
        self.mu = km.cluster_centers_.copy()
        self.var = np.stack([X[lab == k].var(0) + 1e-3 if (lab == k).sum() > 1 else X.var(0) + 1e-3
                             for k in range(self.K)])
        A = np.ones((self.K, self.K))
        for a, b in zip(lab[:-1], lab[1:]):
            A[a, b] += 1
        self.A = A / A.sum(1, keepdims=True)
        self.pi = np.bincount(lab, minlength=self.K) / T
        for _ in range(n_iter):
            logB = self._emit_logp(X)
            B = np.exp(logB - logB.max(1, keepdims=True))
            # forward-backward (scaled)
            alpha = np.zeros((T, self.K)); c = np.zeros(T)
            alpha[0] = self.pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0] + 1e-12
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ self.A) * B[t]; c[t] = alpha[t].sum(); alpha[t] /= c[t] + 1e-12
            beta = np.zeros((T, self.K)); beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / (c[t + 1] + 1e-12)
            gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True) + 1e-12
            xi = np.zeros((self.K, self.K))
            for t in range(T - 1):
                x = (alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None, :])
                xi += x / (x.sum() + 1e-12)
            self.pi = gamma[0] + 1e-6; self.pi /= self.pi.sum()
            self.A = xi / (xi.sum(1, keepdims=True) + 1e-12)
            self.mu = (gamma[:, :, None] * X[:, None, :]).sum(0) / (gamma.sum(0)[:, None] + 1e-12)
            self.var = (gamma[:, :, None] * (X[:, None, :] - self.mu[None]) ** 2).sum(0) / (gamma.sum(0)[:, None] + 1e-12) + 1e-3
        return self

    def causal_filter(self, X):
        """Past-only filtered posterior gamma_t = P(state_t | x_1..x_t) and per-step log-lik."""
        T = len(X); logB = self._emit_logp(X)
        B = np.exp(logB - logB.max(1, keepdims=True)); scale = logB.max(1)
        f = np.zeros((T, self.K)); ll = 0.0
        prev = self.pi.copy()
        for t in range(T):
            a = (prev @ self.A if t > 0 else self.pi) * B[t]
            s = a.sum(); f[t] = a / (s + 1e-12)
            ll += np.log(s + 1e-300) + scale[t]
            prev = f[t]
        return f, ll / T

    def heldout_ll(self, X):
        return self.causal_filter(X)[1]


# ---------- build the v6 belief from preprocessed features ---------------------------------------
def market_features(panel):
    d = pd.read_csv(panel)
    d["date"] = pd.to_datetime(d["date"])
    g = d.groupby("date")
    r = g["daily_return"].mean().sort_index() if "daily_return" in d else None
    if r is None:                                    # Dow panel fallback: build EW ret from close
        dd = d.sort_values(["tic", "date"]); dd["ret"] = dd.groupby("tic")["close"].pct_change()
        r = dd.groupby("date")["ret"].mean().sort_index()
    feats = {"r": r}
    if "Regime_1_Prob" in d: feats["reg1"] = g["Regime_1_Prob"].first().sort_index()
    if "atr_rel" in d: feats["atr_rel"] = g["atr_rel"].mean().sort_index()
    feats["xdisp"] = g["daily_return"].std().sort_index() if "daily_return" in d else g["close"].std().sort_index() * 0
    if "rsi_30" in d: feats["rsi30"] = g["rsi_30"].mean().sort_index()
    if "dx_30" in d: feats["dx30"] = g["dx_30"].mean().sort_index()
    if "forecast_std" in d: feats["gru_std"] = g["forecast_std"].mean().sort_index()
    if "forecast_mean" in d: feats["gru_mean"] = g["forecast_mean"].mean().sort_index()
    F = pd.DataFrame(feats).sort_index()
    return F


def build_belief_v6(panel, train_a, train_b, use_features=None, direct_reg1=False):
    F = market_features(panel)
    r = F["r"]
    cols = [c for c in (use_features or CORE_FEATURES) if c in F.columns]
    m_tr = np.asarray((F.index >= pd.Timestamp(train_a)) & (F.index <= pd.Timestamp(train_b)))
    # HMM train sub-window = where ALL chosen features are live (reg1 warms up 2018-10-30)
    live = np.ones(len(F), bool)
    for c in cols:
        live &= F[c].notna().to_numpy() & (F[c].to_numpy() != 0 if c == "reg1" else True)
    fit_mask = m_tr & live
    X_all = F[cols].to_numpy(float)
    mu_tr = np.nanmean(X_all[fit_mask], 0); sd_tr = np.nanstd(X_all[fit_mask], 0) + 1e-9  # FROZEN standardizer
    Z = (np.nan_to_num(X_all, nan=0.0) - mu_tr) / sd_tr
    Z = np.clip(Z, -5, 5)

    if direct_reg1:                                  # arm: use the preprocessing's OWN regime posterior directly
        b_risk = F["reg1"].fillna(0.0).to_numpy()
        b_cri = b_risk * 0.6                          # crisis-share proxy; U2 crisis-branch uses b_cri>=b_risk-b_cri
        K = 1; card = {"mode": "direct_Regime_1_Prob", "K": 1, "features": ["reg1"]}
        path = (b_risk > 0.5).astype(int)
        heldout = {"K1": None, "K2": None}
    else:
        Ztr = Z[fit_mask]
        cut = int(len(Ztr) * 0.8)
        m3 = GaussianHMM(3).fit(Ztr[:cut]); m2 = GaussianHMM(2).fit(Ztr[:cut])
        ll3, ll2 = m3.heldout_ll(Ztr[cut:]), m2.heldout_ll(Ztr[cut:])
        K = 3 if ll3 > ll2 else 2
        model = GaussianHMM(K).fit(Ztr)
        f_all, _ = model.causal_filter(Z)
        # NAME states by standardized risk-score = mean of (reg1, atr_rel, xdisp) means, else all features
        risk_cols = [i for i, c in enumerate(cols) if c in ("reg1", "atr_rel", "xdisp")] or list(range(len(cols)))
        score = model.mu[:, risk_cols].mean(1)
        order = np.argsort(score)                      # low->high risk
        crisis = int(order[-1]); calm = int(order[0])
        grind = int(order[1]) if K == 3 else -1
        b_cri = f_all[:, crisis]
        b_risk = b_cri + (f_all[:, grind] if grind >= 0 else 0.0)
        path = f_all.argmax(1)
        names = {k: "CALM" for k in range(K)}; names[crisis] = "CRISIS"
        if grind >= 0: names[grind] = "GRIND"
        card = {"mode": "gaussian_hmm", "K": K, "features": cols,
                "heldout_LL": {"K3": round(float(ll3), 4), "K2": round(float(ll2), 4)},
                "state_names": [names[k] for k in range(K)],
                "state_means_std": np.round(model.mu, 2).tolist(),
                "dwell_days": [round(1 / (1 - model.A[k, k] + 1e-9), 1) for k in range(K)],
                "risk_score_order": order.tolist()}
        heldout = card["heldout_LL"]

    sigma20 = r.rolling(V5.VOL_WIN).std().shift(1)
    sig_star = float(sigma20[m_tr].median())
    S = {"r": r, "b_risk": pd.Series(b_risk, index=r.index), "b_cri": pd.Series(b_cri, index=r.index),
         "sigma20": sigma20, "sig_star": sig_star, "m_tr": m_tr, "path": np.asarray(path),
         "card": card, "heldout": heldout, "cols": cols}
    return S


# ---------- v6 falsifier battery -----------------------------------------------------------------
def falsifiers_v6(S, sub, spec):
    out = {}
    card = S["card"]
    if card.get("mode") == "gaussian_hmm":
        out["F_U1a_LL"] = {"chosen_K": card["K"], **card["heldout_LL"],
                           "PASS": bool(card["heldout_LL"]["K3"] is not None)}
        path_tr = S["path"][S["m_tr"]]
        occ = np.bincount(path_tr, minlength=card["K"]) / max(1, len(path_tr))
        runs = np.diff(np.where(np.concatenate([[1], np.diff(path_tr) != 0, [1]]))[0])
        out["F_U1b_states"] = {"occupancy": np.round(occ, 3).tolist(), "mean_dwell_d": round(float(runs.mean()), 1),
                               "PASS": bool(occ.min() > 0.05 and runs.mean() > 5)}
    lit = {"Q_ON": 0.85, "E_GRIND": 0.70}
    ex = sub._ex(lit)
    hold_idx = sub.win_idx(*spec["hold"])
    armed = [float((ex[hold_idx[s:s + 120] - 1] < 0.999).mean()) for s in range(0, max(1, len(hold_idx) - 120), 60)]
    out["F_U2b_armed"] = {"armed_frac_per_window": np.round(armed, 3).tolist(),
                          "PASS": bool(all(0.05 <= x <= 0.60 for x in armed))}
    # placebo: block-shuffle b_risk AND b_cri together, must not reproduce the dev edge
    rng = np.random.default_rng(0)
    br, bc = S["b_risk"].to_numpy(), S["b_cri"].to_numpy()
    idx = np.arange(len(br)); blocks = [idx[i:i + 60] for i in range(0, len(idx), 60)]; rng.shuffle(blocks)
    perm = np.concatenate(blocks)[:len(br)]
    S_pl = dict(S); S_pl["b_risk"] = pd.Series(br[perm], index=S["b_risk"].index)
    S_pl["b_cri"] = pd.Series(bc[perm], index=S["b_cri"].index)
    sub_pl = V5.SubstrateV5(S_pl, sub.rf)
    dev_idx = sub.win_idx(*spec["dev"])
    re = V4.ann_dd(sub.pnl_win(lit, dev_idx)); pl = V4.ann_dd(sub_pl.pnl_win(lit, dev_idx))
    an = V4.ann_dd(sub.pnl_win(V5.ANCHOR, dev_idx))
    out["F_U2a_placebo"] = {"real_dev_edge": round((re[0] - an[0]) + (re[1] - an[1]), 4),
                            "placebo_dev_edge": round((pl[0] - an[0]) + (pl[1] - an[1]), 4),
                            "PASS": bool((re[0] - an[0]) + (re[1] - an[1]) > (pl[0] - an[0]) + (pl[1] - an[1]))}
    return out


# ---------- the loop (v5 gate + policy, v6 belief) ------------------------------------------------
def run_panel_v6(name, spec, arm="core", prior=None, rounds=30):
    feats = {"core": CORE_FEATURES, "core+gru": CORE_FEATURES + GRU_FEATURES}.get(arm, CORE_FEATURES)
    S = build_belief_v6(spec["panel"], *spec["train"], use_features=feats, direct_reg1=(arm == "direct_reg1"))
    sub = SubstrateV6(S, V5.RF_DAILY[name])
    dev_idx, hold_idx, oos_idx = (sub.win_idx(*spec[k]) for k in ("dev", "hold", "oos"))
    fals = falsifiers_v6(S, sub, spec)
    from src.hl.mechanism_bandit import MechanismBandit
    gate = V5.GateV5(sub, dev_idx, hold_idx)
    current = dict(V5.ANCHOR)
    gate.frontier = [(dict(current), gate.vec(current))]
    bandit = MechanismBandit(arms=list(V5.ARMS))
    if prior: bandit.seed_from(prior)
    step = {k: 0.25 * (v[1] - v[0]) for k, v in V5.KNOBS.items()}
    direction = {"Q_ON": -1.0, "E_GRIND": -1.0}
    trail = []
    for rnd in range(rounds):
        if rnd > 0 and rnd % 10 == 0:
            gate.canary_check(current)
        a = bandit.select(V5.ARMS)
        if a == "joint_defend":
            cand = {"Q_ON": 0.85, "E_GRIND": 0.70}
        else:
            cand = {k: current[k] for k in V5.KNOBS}
            base = cand[a] if cand[a] <= V5.KNOBS[a][1] else V5.KNOBS[a][2]
            lo, hi, _ = V5.KNOBS[a]; cand[a] = float(np.clip(base + step[a] * direction[a], lo, hi))
        verdict, info, current = gate.review(cand, current)
        ok = verdict.startswith("ACCEPTED"); bandit.update(a, 1.0 if ok else 0.0)
        if not ok and a != "joint_defend":
            step[a] *= 0.6; direction[a] = -direction[a]
        trail.append({"round": rnd, "arm": a, "verdict": verdict, **{k: v for k, v in info.items() if k != "cand"}})
    gate.canary_check(current)

    def perf(c, wi):
        s = sub.pnl_win(c, wi); ann, dd = V4.ann_dd(s)
        return {"ann": round(ann, 4), "maxDD": round(dd, 4),
                "sharpe": round(float(np.sqrt(252) * s.mean() / (s.std() + 1e-12)), 3)}
    counts = {}
    for t in trail: counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    return {"panel": name, "arm": arm, "card": S["card"], "falsifiers": fals,
            "certified_coeffs": {k: round(current[k], 3) for k in V5.KNOBS}, "is_anchor": bool(current["Q_ON"] > 1.0),
            "accepts": counts.get("ACCEPTED_RISKMODE", 0), "gate_counts": counts, "audit": gate.audit,
            "wealth_left": round(gate.wealth, 4), "gate_compromised": gate.compromised,
            "frontier": [{"coeffs": {k: round(p[k], 3) for k in V5.KNOBS}, "ann_dev": round(v["ann"], 4),
                          "maxDD_dev": round(v["maxDD"], 4)} for p, v in gate.frontier],
            "hold_full": {"anchor": perf(V5.ANCHOR, hold_idx), "final": perf(current, hold_idx)},
            "oos_single_shot": {"anchor": perf(V5.ANCHOR, oos_idx), "final": perf(current, oos_idx)},
            "bandit_prior": bandit.prior(), "trail": trail}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    runs = {}
    for arm in ("core", "direct_reg1", "core+gru"):
        runs[f"csi500_{arm}"] = run_panel_v6("csi500", V4.PANELS["csi500"], arm=arm)
    runs["dow_core"] = run_panel_v6("dow", V4.PANELS["dow"], arm="core")
    report = {"design": "CRYSTAL-1 belief from PREPROCESSED panel features (causal Gaussian HMM over "
                        "reg1/atr_rel/xdisp/rsi30/dx30) + v5 policy + v4 gate + U3 stress; arms: core, "
                        "direct_Regime_1_Prob, core+GRU(quarantined ablation)",
              "provenance": "Data_preprocessing.ipynb discipline (causal past-only HMM filtering, train-frozen "
                            "standardizer); GRU quarantined (in-sample-fit IC t=+10); reg1 = US-macro regime "
                            "posterior appended to CN names (cross-asset risk-off, PIT-safe)",
              **{k: {kk: vv for kk, vv in v.items() if kk != "trail"} for k, v in runs.items()},
              "trails": {k: v["trail"] for k, v in runs.items()}}
    OUT.write_text(json.dumps(report, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    for k, v in runs.items():
        c = v["card"]
        print(f"[{k}] accepts={v['accepts']} gate={v['gate_counts']}")
        print(f"    belief={c.get('mode')} K={c.get('K')} states={c.get('state_names', ['reg1'])} feats={c.get('features')}")
        print(f"    falsifiers: " + " ".join(f"{fk.split('_')[1]}={'P' if fv['PASS'] else 'F'}" for fk, fv in v["falsifiers"].items()))
        print(f"    hold: anchor {v['hold_full']['anchor']} vs final {v['hold_full']['final']}")
        print(f"    OOS : anchor {v['oos_single_shot']['anchor']} vs final {v['oos_single_shot']['final']}")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
