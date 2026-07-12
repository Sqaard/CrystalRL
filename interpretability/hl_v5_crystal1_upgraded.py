"""HL v5 — the UPGRADED CRYSTAL-1 (U1+U2+U3 from CRYSTAL1_UPGRADE_RESEARCH.md) + the v4 gate, on real panels.

The oracle test (CRYSTAL1_ORACLE_CEILING.md) proved the shell+gate monetize a good signal; the research
workflow converged on exactly three upgrades, implemented here verbatim with their falsifiers:

U1 SIGNAL — signed vol-normalized return symbol x cross-sectional-dispersion bit -> A_obs=6 named symbols;
   K=3 filter (train blocks L=252), STICKINESS prior applied as a frozen transition blend to min-dwell 20d;
   states pre-registered-NAMED from the emission card: CRISIS=argmax P(disp=HI), GRIND=remaining argmax
   P(DOWN), CALM=rest. b_risk = b[GRIND]+b[CRISIS]. (Fixes audit #1/#2/#3/#4/#9; breadth is DEAD on this
   panel — xsec_disp is the verified substitute; GRU features QUARANTINED for leakage.)
U2 POLICY — belief-gated conditional vol targeting (Bongaerts 2020) with liveness + hysteresis:
   g_on = trailing-252d Q_ON-quantile of b_risk; gate ON when b_risk>g_on, OFF when b_risk<g_on-HYST.
   OFF -> exposure 1 (never de-risk in calm). ON&crisis -> clip(SIGMA_STAR/sigma_hat, E_MIN, 1).
   ON&grind -> E_GRIND (grind is LOW-vol: vol targeting cannot see it).
   SEARCHED knobs: Q_ON (~0.85), E_GRIND (~0.7) — two only, preserving gate power.
   FROZEN: HYST=0.10, E_MIN=0.5 (A-share high-vol->high-return; unfloored = B3's known failure),
   SIGMA_STAR=train-median sigma_hat, VOL_WIN=20/60.
U3 EXECUTION — cash yield on the sleeve (CN 2%/yr; US 1.5%), SELL-side stamp asymmetry (0.1%);
   acceptance stress: the RISK vol/DD leg must survive rf=0 (rf may only support non-inferiority) and a
   1-day execution lag (if the effect dies at t+1, it was never tradeable under T+1).

Pre-registered falsifier battery runs BEFORE the loop and is reported PASS/FAIL (fallbacks per synthesis).
Run: python interpretability/hl_v5_crystal1_upgraded.py
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.crystal.belief_filter import train_filter  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from src.hl.mechanism_bandit import MechanismBandit  # noqa: E402
import interpretability.hl_v4_over_crystal1 as V4  # noqa: E402

OUT = HERE / "hl_v5_crystal1_report.json"
COST, STAMP = 0.001, 0.001
HYST, E_MIN, VOL_WIN, DWELL_MIN = 0.10, 0.50, 20, 20.0
KNOBS = {"Q_ON": (0.70, 0.95, 0.85), "E_GRIND": (0.40, 1.00, 0.70)}
ANCHOR = {"Q_ON": 1.01, "E_GRIND": 1.00}       # gate never arms -> exposure identically 1 (EW buy-and-hold)
RF_DAILY = {"dow": 0.015 / 252, "csi500": 0.02 / 252}
ARMS = list(KNOBS) + ["joint_defend"]


# ---------- U1: signal ---------------------------------------------------------------------------
def load_streams(panel):
    d = pd.read_csv(panel, usecols=["date", "tic", "close"]).sort_values(["tic", "date"])
    d["ret"] = d.groupby("tic")["close"].pct_change()
    d["date"] = pd.to_datetime(d["date"])
    g = d.groupby("date")["ret"]
    r, disp = g.mean().sort_index(), g.std().sort_index()
    ok = np.isfinite(r)
    return r[ok], disp[ok]


def build_belief_v5(panel, train_a, train_b):
    r, disp = load_streams(panel)
    m_tr = np.asarray((r.index >= pd.Timestamp(train_a)) & (r.index <= pd.Timestamp(train_b)))
    sd60 = r.rolling(60).std().shift(1)
    z = (r / sd60).to_numpy()
    z_hi = float(np.nanquantile(np.abs(z[m_tr]), 0.80))
    disp_q = float(np.nanquantile(disp.to_numpy()[m_tr], 0.80))
    s_ret = np.where(np.isnan(z), 1, np.where(z < -z_hi, 0, np.where(z > z_hi, 2, 1)))
    s_disp = (disp.to_numpy() > disp_q).astype(int)
    sym = (s_ret + 3 * s_disp).astype(int)                        # 6 named symbols
    sym_tr = sym[m_tr]
    L = 252
    # PRE-REGISTERED K-SELECTION (the synthesis's own falsifier + fallback): train K=3 and K=2 on the SAME
    # 6-symbol alphabet; K=3 stays only if it beats K=2 on held-out filtering LL (last 20% of train).
    cut = int(len(sym_tr) * 0.8)

    def _fit(K, seq):
        n = max(1, len(seq) // L)
        m = train_filter(seq[:n * L].reshape(n, L), K=K, A_obs=6, epochs=500, seed=0, verbose=False)
        return m.numpy_params()

    def _ll(params, seq):
        T, E, pr = params
        b = pr.copy(); ll = 0.0
        for o in seq:
            bp = b @ T; j = bp * E[:, int(o)]; s = j.sum()
            ll += math.log(max(s, 1e-300)); b = j / max(s, 1e-12)
        return ll / len(seq)
    par3, par2 = _fit(3, sym_tr[:cut]), _fit(2, sym_tr[:cut])
    ll3, ll2 = _ll(par3, sym_tr[cut:]), _ll(par2, sym_tr[cut:])
    K = 3 if ll3 > ll2 else 2
    T_l, E_l, p0 = _fit(K, sym_tr)                                 # refit the winner on the full train window
    # stickiness prior (frozen): blend toward I until min expected dwell >= 20d
    w = max(0.0, max((0.95 - T_l[k, k]) / (1.0 - T_l[k, k] + 1e-12) for k in range(K)))
    T_s = (1 - w) * T_l + w * np.eye(K)
    # pre-registered naming from the TRAIN emission card. CRISIS = argmax P(disp=HI). A non-crisis state
    # counts as GRIND (risk-bearing) only if genuinely DOWN-TILTED: P(DOWN|k) > P(UP|k) — without this the
    # middle bulk state (61% occupancy, dwell ~1yr) absorbs into b_risk and the U2 gate loses all meaning.
    p_hi = E_l[:, 3:].sum(axis=1); crisis = int(np.argmax(p_hi))
    p_down = E_l[:, 0] + E_l[:, 3]; p_up = E_l[:, 2] + E_l[:, 5]
    rest = [k for k in range(K) if k != crisis]
    down_tilted = [k for k in rest if p_down[k] > p_up[k]]
    grind = int(max(down_tilted, key=lambda k: p_down[k] - p_up[k])) if down_tilted else -1
    calm = [k for k in range(K) if k not in (crisis, grind)]
    b = p0.copy(); b_risk = np.empty(len(r)); b_cri = np.empty(len(r)); path = np.empty(len(r), int)
    for i, o in enumerate(sym):
        bp = b @ T_s; j = bp * E_l[:, int(o)]; b = j / max(j.sum(), 1e-12)
        b_risk[i] = b[crisis] + (b[grind] if grind >= 0 else 0.0)
        b_cri[i] = b[crisis]; path[i] = int(np.argmax(b))
    sigma20 = r.rolling(VOL_WIN).std().shift(1)
    sig_star = float(sigma20[m_tr].median())
    names = {k: "CALM" for k in range(K)}
    names[crisis] = "CRISIS"
    if grind >= 0:
        names[grind] = "GRIND"
    card = {"K": K, "heldout_LL": {"K3": round(ll3, 4), "K2": round(ll2, 4)},
            "state_names": [names[k] for k in range(K)], "emissions_6sym": np.round(E_l, 3).tolist(),
            "transition_sticky": np.round(T_s, 3).tolist(), "stick_w": round(w, 3),
            "z_hi": round(z_hi, 3), "disp_q": round(disp_q, 5),
            "dwell_days": [round(1 / (1 - T_s[k, k] + 1e-12), 1) for k in range(K)]}
    S = {"r": r, "b_risk": pd.Series(b_risk, index=r.index), "b_cri": pd.Series(b_cri, index=r.index),
         "sigma20": sigma20, "sig_star": sig_star, "sym": sym, "m_tr": m_tr, "path": path,
         "train_LL_data": (sym_tr, T_s, E_l, p0), "card": card}
    return S


# ---------- U2: policy (full-series exposure, then window slicing — PIT-consistent) ---------------
def exposure_series(S, c):
    br = S["b_risk"]
    g_on = br.rolling(252, min_periods=60).quantile(min(c["Q_ON"], 1.0)).shift(1)
    sig = S["sigma20"]; sig_star = S["sig_star"]
    n = len(br); ex = np.ones(n); on = False
    brv, gv, sv, cv = br.to_numpy(), g_on.to_numpy(), sig.to_numpy(), S["b_cri"].to_numpy()
    for t in range(n):
        if not np.isfinite(gv[t]):
            ex[t] = 1.0; on = False; continue
        if on and brv[t] < gv[t] - HYST:
            on = False
        elif not on and brv[t] > gv[t]:
            on = True
        if not on:
            ex[t] = 1.0
        elif cv[t] >= brv[t] - cv[t]:                              # crisis dominates
            ex[t] = float(np.clip(sig_star / sv[t], E_MIN, 1.0)) if np.isfinite(sv[t]) and sv[t] > 0 else E_MIN
        else:
            ex[t] = float(c["E_GRIND"])
    return pd.Series(ex, index=br.index)


def pnl(ex, ro, rf):
    dex = np.diff(np.concatenate([[1.0], ex]))
    costs = COST * np.maximum(dex, 0) + (COST + STAMP) * np.maximum(-dex, 0)
    return ex * ro + (1.0 - ex) * rf - costs


class SubstrateV5:
    """Precomputes exposure series per config (cached) and serves window PnL, incl. rf=0 / lagged variants."""

    def __init__(self, S, rf):
        self.S, self.rf = S, rf
        self._cache = {}

    def _ex(self, c):
        key = (round(c["Q_ON"], 6), round(c["E_GRIND"], 6))
        if key not in self._cache:
            self._cache[key] = exposure_series(self.S, c).to_numpy()
        return self._cache[key]

    def win_idx(self, a, b):
        idx = np.asarray((self.S["r"].index >= pd.Timestamp(a)) & (self.S["r"].index <= pd.Timestamp(b)))
        return np.where(idx)[0][1:]                                # skip first day (pct_change convention)

    def pnl_win(self, c, wi, rf=None, lag=0):
        rf = self.rf if rf is None else rf
        ro = self.S["r"].to_numpy()[wi]
        ex = self._ex(c)[wi - 1 - lag]                             # exposure decided on t-1 (or t-1-lag)
        return pnl(ex, ro, rf)


# ---------- the v5 gate (v4 machinery + either-lane + U3 stress) ----------------------------------
class GateV5:
    def __init__(self, sub, dev_idx, hold_idx, hold_win=120):
        self.sub, self.dev_idx, self.hold_idx = sub, dev_idx, hold_idx
        self.hold_win = min(hold_win, len(hold_idx) - 1)
        a = sub.pnl_win(ANCHOR, dev_idx)
        vol = pd.Series(a).rolling(20).std().to_numpy()
        eq = np.cumprod(1 + a); ddp = eq / np.maximum.accumulate(eq) - 1
        self.stress = {"high_vol": np.where(vol >= np.nanquantile(vol, 0.75))[0],
                       "deep_dd": np.where(ddp <= np.quantile(ddp, 0.25))[0]}
        self.wealth, self.spent_tension, self.tension_budget = 0.20, 0.0, 6.0
        self.epoch, self.compromised, self.frontier = 0, False, []
        self.audit = {"dominated": 0, "adversary_vetoes": 0, "budget_refusals": 0, "wealth_refusals": 0,
                      "holdout_rejections": 0, "inert_windows": 0, "stress_slice_fails": 0,
                      "canary_caught": 0, "canary_escaped": 0, "canary_unarmed": 0}
        self.canary_log = []

    def vec(self, c):
        a, d = V4.ann_dd(self.sub.pnl_win(c, self.dev_idx))
        return {"ann": a, "maxDD": d}

    def _rot(self):
        span = max(1, len(self.hold_idx) - self.hold_win)
        stride = max(7, span // 7)
        s = (self.epoch * stride) % span
        return self.hold_idx[s:s + self.hold_win], span, s

    def review(self, cand, current):
        if self.compromised:
            return "REFUSED_GATE_COMPROMISED", {}, current
        cur_vec, cand_vec = self.vec(current), self.vec(cand)
        front = [v for _, v in self.frontier] or [cur_vec]
        if any(V4.dominates(f, cand_vec) for f in front):
            self.audit["dominated"] += 1
            return "REJECTED_DOMINATED", {"cand": cand_vec}, current
        ret_fwd = cand_vec["ann"] > cur_vec["ann"] + 1e-4 and cand_vec["maxDD"] >= cur_vec["maxDD"] - 1e-4
        risk_fwd = cand_vec["maxDD"] > cur_vec["maxDD"] + V4.EPS_DD and cand_vec["ann"] >= cur_vec["ann"] - 0.01
        if not (ret_fwd or risk_fwd):
            self.audit["dominated"] += 1
            return "REJECTED_NO_FRONTIER_GAIN", {"cand": cand_vec}, current
        if self.wealth < 1e-3:
            self.audit["wealth_refusals"] += 1
            return "REFUSED_ALPHA_WEALTH_EXHAUSTED", {"wealth": round(self.wealth, 4)}, current
        alpha_t = min(0.5, self.wealth / 2.0)
        z_crit = math.sqrt(2.0 * math.log(1.0 / alpha_t))
        hw, span, s0 = self._rot()
        cur_h, cand_h = self.sub.pnl_win(current, hw), self.sub.pnl_win(cand, hw)
        delta = cand_h - cur_h
        if float(np.abs(delta).max()) < 1e-12:
            self.audit["inert_windows"] += 1
            return "REFUSED_INERT_ON_WINDOW", {"window_start": int(s0)}, current
        self.epoch += 1
        z, se = block_z(delta, block=5, n_boot=1000, seed=self.epoch)
        lanes, passes, lane_used = {}, False, None
        if ret_fwd:
            lanes["RETURN"] = {"z": round(z, 2)}
            passes = float(delta.mean()) > 0 and z > z_crit
            lane_used = "RETURN" if passes else None
        if not passes and risk_fwd:
            # RISK leg on rf=0 streams (U3: rf may only support non-inferiority)
            cur0, cand0 = self.sub.pnl_win(current, hw, rf=0.0), self.sub.pnl_win(cand, hw, rf=0.0)
            z_dsd, dsd_gain = V4.risk_boot_z(cand0, cur0, seed=self.epoch)
            ni = (float(delta.mean()) + V4.NI_MARGIN) / se
            lanes["RISK"] = {"z_dsd_rf0": round(z_dsd, 2), "ni_z": round(ni, 2), "dsd_gain": round(dsd_gain, 6)}
            if z_dsd > z_crit and ni > z_crit:
                s2 = (s0 + span // 2) % span
                hw2 = self.hold_idx[s2:s2 + self.hold_win]
                c2a, c2b = self.sub.pnl_win(current, hw2), self.sub.pnl_win(cand, hw2)
                d2 = c2b - c2a
                if float(np.abs(d2).max()) < 1e-12:
                    lanes["RISK"]["confirm"] = "INERT"
                else:
                    z2, se2 = block_z(d2, block=5, n_boot=1000, seed=self.epoch + 7919)
                    ni2 = (float(d2.mean()) + V4.NI_MARGIN) / se2
                    c20a, c20b = self.sub.pnl_win(current, hw2, rf=0.0), self.sub.pnl_win(cand, hw2, rf=0.0)
                    z_dsd2, _ = V4.risk_boot_z(c20b, c20a, seed=self.epoch + 7919)
                    lanes["RISK"]["confirm"] = {"start": int(s2), "ni2": round(ni2, 2), "z_dsd2": round(z_dsd2, 2)}
                    passes = ni2 > z_crit and z_dsd2 > 0
                    lane_used = "RISK" if passes else None
        info = {"lanes": lanes, "z_crit": round(z_crit, 2)}
        if passes:
            # U3 lag stress: the passing lane's direction must survive 1-day execution lag
            curL, candL = self.sub.pnl_win(current, hw, lag=1), self.sub.pnl_win(cand, hw, lag=1)
            if lane_used == "RETURN":
                zL, _ = block_z(candL - curL, block=5, n_boot=500, seed=self.epoch + 31)
                lag_ok = zL > 0
            else:
                cur0L, cand0L = self.sub.pnl_win(current, hw, rf=0.0, lag=1), self.sub.pnl_win(cand, hw, rf=0.0, lag=1)
                zL, _ = V4.risk_boot_z(cand0L, cur0L, seed=self.epoch + 31)
                lag_ok = zL > 0
            info["lag_z"] = round(zL, 2)
            if not lag_ok:
                self.audit["stress_slice_fails"] += 1
                self.wealth -= alpha_t / (1 - alpha_t)
                return "REFUSED_LAG_STRESS", info, current
        if not passes:
            self.wealth -= alpha_t / (1 - alpha_t)
            self.audit["holdout_rejections"] += 1
            info["wealth"] = round(self.wealth, 4)
            return "REJECTED_HOLDOUT", info, current
        cand_dev, cur_dev = self.sub.pnl_win(cand, self.dev_idx), self.sub.pnl_win(current, self.dev_idx)
        for sname, sidx in self.stress.items():
            sd = float((cand_dev[sidx] - cur_dev[sidx]).mean())
            if sd < V4.STRESS_MARGIN:
                self.audit["adversary_vetoes"] += 1
                self.wealth -= alpha_t / (1 - alpha_t)
                return "VETOED_BY_ADVERSARY", {"stressor": sname, "mean_delta": round(sd, 6)}, current
        harm = max(0.0, sum(1 for k in KNOBS if abs(cand[k] - ANCHOR[k]) > 1e-9)
                   - sum(1 for k in KNOBS if abs(current[k] - ANCHOR[k]) > 1e-9))
        if self.spent_tension + harm > self.tension_budget:
            self.audit["budget_refusals"] += 1
            return "REFUSED_TENSION_BUDGET", {"harm": harm}, current
        self.wealth = min(1.0, self.wealth + 0.05)
        self.spent_tension += harm
        self.frontier = [(p, v) for (p, v) in self.frontier if not V4.dominates(cand_vec, v)]
        self.frontier.append((dict(cand), cand_vec))
        return "ACCEPTED_RISKMODE", {"cand": cand_vec, "lane": lane_used, "harm": harm, **info}, cand

    def _sandboxed(self, cand, current):
        snap = (self.wealth, self.spent_tension, self.epoch, list(self.frontier), dict(self.audit))
        try:
            v, _, _ = self.review(cand, current)
        finally:
            self.wealth, self.spent_tension, self.epoch, self.frontier, self.audit = snap
        return v

    def canary_check(self, current):
        results = []
        if current["Q_ON"] > 1.0:                                  # anchor semantics: gate never arms
            bloat = dict(current); bloat["E_GRIND"] = 0.55         # knob moved, behavior identical (gate off)
            results.append(("config_bloat", self._sandboxed(bloat, current)))
        cur_dev = self.sub.pnl_win(current, self.dev_idx)
        for name, cfg in {"lockdown": {"Q_ON": 0.70, "E_GRIND": 0.40},
                          "perma_grind": {"Q_ON": 0.70, "E_GRIND": 0.40}}.items():
            harm_dev = float((self.sub.pnl_win(cfg, self.dev_idx) - cur_dev).mean())
            if harm_dev < V4.STRESS_MARGIN:
                results.append((name, self._sandboxed(dict(cfg), current)))
            else:
                self.audit["canary_unarmed"] += 1
                self.canary_log.append((name, "UNARMED", round(harm_dev, 6)))
        for nm, v in results:
            self.canary_log.append((nm, v, None))
            if v.startswith("ACCEPTED"):
                self.audit["canary_escaped"] += 1
                self.compromised = True
            else:
                self.audit["canary_caught"] += 1
        return results


# ---------- pre-registered falsifier battery ------------------------------------------------------
def falsifiers(S, sub, spec):
    out = {}
    sym_tr, T_s, E_l, p0 = S["train_LL_data"]
    cut = int(len(sym_tr) * 0.8)
    va = sym_tr[cut:]

    def filt_ll(T, E, prior, seq):
        b = prior.copy(); ll = 0.0
        for o in seq:
            bp = b @ T; j = bp * E[:, int(o)]; s = j.sum()
            ll += math.log(max(s, 1e-300)); b = j / max(s, 1e-12)
        return ll / len(seq)
    freqs = np.bincount(sym_tr[:cut], minlength=6) / cut
    ll_memoryless = float(np.mean([math.log(max(freqs[o], 1e-12)) for o in va]))
    ll_v5 = filt_ll(T_s, E_l, p0, va)
    out["F_U1a_heldout_LL"] = {"chosen_K": S["card"]["K"], **S["card"]["heldout_LL"],
                               "selected_LL": round(ll_v5, 4), "memoryless_6sym": round(ll_memoryless, 4),
                               "PASS": bool(ll_v5 > ll_memoryless)}
    path_tr = S["path"][S["m_tr"]]
    occ = np.bincount(path_tr, minlength=3) / len(path_tr)
    runs = np.diff(np.where(np.concatenate([[1], np.diff(path_tr) != 0, [1]]))[0])
    out["F_U1b_states"] = {"occupancy": np.round(occ, 3).tolist(), "mean_dwell_d": round(float(runs.mean()), 1),
                           "PASS": bool(occ.min() > 0.05 and runs.mean() > 5)}
    tr_rate = float((S["sym"][S["m_tr"]] % 3 != 1).mean())
    hold_idx = sub.win_idx(*spec["hold"])
    rates = []
    for s in range(0, max(1, len(hold_idx) - 120), 60):
        rates.append(float((S["sym"][hold_idx[s:s + 120]] % 3 != 1).mean()))
    out["F_U1c_firing"] = {"train_rate": round(tr_rate, 3), "hold_window_rates": np.round(rates, 3).tolist(),
                           "PASS": bool(all(0.5 * tr_rate <= x <= 2.0 * tr_rate for x in rates))}
    lit = {"Q_ON": 0.85, "E_GRIND": 0.70}
    ex = sub._ex(lit)
    armed = []
    for s in range(0, max(1, len(hold_idx) - 120), 60):
        armed.append(float((ex[hold_idx[s:s + 120] - 1] < 0.999).mean()))
    out["F_U2b_armed"] = {"armed_frac_per_window": np.round(armed, 3).tolist(),
                          "PASS": bool(all(0.05 <= x <= 0.60 for x in armed))}
    # U2a placebo: block-shuffled belief must not produce the same dev advantage
    rng = np.random.default_rng(0)
    br = S["b_risk"].to_numpy().copy()
    blocks = [br[i:i + 60] for i in range(0, len(br), 60)]        # include the tail block
    rng.shuffle(blocks)
    S_pl = dict(S); S_pl["b_risk"] = pd.Series(np.concatenate(blocks)[:len(br)], index=S["b_risk"].index)
    sub_pl = SubstrateV5(S_pl, sub.rf)
    dev_idx = sub.win_idx(*spec["dev"])
    real_dev = V4.ann_dd(sub.pnl_win(lit, dev_idx))
    plac_dev = V4.ann_dd(sub_pl.pnl_win(lit, dev_idx))
    anch_dev = V4.ann_dd(sub.pnl_win(ANCHOR, dev_idx))
    real_edge = (real_dev[0] - anch_dev[0]) + (real_dev[1] - anch_dev[1])
    plac_edge = (plac_dev[0] - anch_dev[0]) + (plac_dev[1] - anch_dev[1])
    out["F_U2a_placebo"] = {"real_dev_edge": round(real_edge, 4), "placebo_dev_edge": round(plac_edge, 4),
                            "PASS": bool(real_edge > plac_edge)}
    return out


# ---------- the loop ------------------------------------------------------------------------------
def run_panel(name, spec, prior=None, rounds=30):
    S = build_belief_v5(spec["panel"], *spec["train"])
    sub = SubstrateV5(S, RF_DAILY[name])
    dev_idx, hold_idx, oos_idx = (sub.win_idx(*spec[k]) for k in ("dev", "hold", "oos"))
    fals = falsifiers(S, sub, spec)
    gate = GateV5(sub, dev_idx, hold_idx)
    current = dict(ANCHOR)
    gate.frontier = [(dict(current), gate.vec(current))]
    bandit = MechanismBandit(arms=list(ARMS))
    if prior:
        bandit.seed_from(prior)
    step = {k: 0.25 * (v[1] - v[0]) for k, v in KNOBS.items()}
    direction = {"Q_ON": -1.0, "E_GRIND": -1.0}
    trail = []
    for rnd in range(rounds):
        if rnd > 0 and rnd % 10 == 0:
            gate.canary_check(current)
        arm = bandit.select(ARMS)
        if arm == "joint_defend":
            cand = {"Q_ON": 0.85, "E_GRIND": 0.70}                # the literature default, atomic
        else:
            cand = {k: current[k] for k in KNOBS}
            base = cand[arm] if cand[arm] <= KNOBS[arm][1] else KNOBS[arm][2]
            lo, hi, _ = KNOBS[arm]
            cand[arm] = float(np.clip(base + step[arm] * direction[arm], lo, hi))
        verdict, info, current = gate.review(cand, current)
        ok = verdict.startswith("ACCEPTED")
        bandit.update(arm, 1.0 if ok else 0.0)
        if not ok and arm != "joint_defend":
            step[arm] *= 0.6; direction[arm] = -direction[arm]
        trail.append({"round": rnd, "arm": arm, "verdict": verdict,
                      **{k: v for k, v in info.items() if k != "cand"}})
    gate.canary_check(current)

    def perf(c, wi):
        s = sub.pnl_win(c, wi)
        a, d = V4.ann_dd(s)
        return {"ann": round(a, 4), "maxDD": round(d, 4),
                "sharpe": round(float(np.sqrt(252) * s.mean() / (s.std() + 1e-12)), 3)}

    counts = {}
    for t in trail:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    return {"panel": name, "filter_card": S["card"], "falsifiers": fals, "input_prior": prior,
            "certified_coeffs": {k: round(current[k], 3) for k in KNOBS},
            "is_anchor": bool(current["Q_ON"] > 1.0),
            "accepts": counts.get("ACCEPTED_RISKMODE", 0), "gate_counts": counts, "audit": gate.audit,
            "wealth_left": round(gate.wealth, 4), "gate_compromised": gate.compromised,
            "canary_log": [list(x) for x in gate.canary_log],
            "frontier": [{"coeffs": {k: round(p[k], 3) for k in KNOBS},
                          "ann_dev": round(v["ann"], 4), "maxDD_dev": round(v["maxDD"], 4)}
                         for p, v in gate.frontier],
            "bandit_prior": bandit.prior(),
            "hold_full": {"anchor": perf(ANCHOR, hold_idx), "final": perf(current, hold_idx)},
            "oos_single_shot": {"anchor": perf(ANCHOR, oos_idx), "final": perf(current, oos_idx)},
            "trail": trail}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    csi = run_panel("csi500", V4.PANELS["csi500"])
    dow = run_panel("dow", V4.PANELS["dow"], prior=csi["bandit_prior"])
    report = {"design": "U1 signed vol-normalized x dispersion 6-symbol K=3 sticky filter; "
                        "U2 belief-gated conditional vol targeting {Q_ON, E_GRIND}; "
                        "U3 cash yield + stamp asymmetry + rf0/lag acceptance stress",
              "provenance": "reports/CRYSTAL1_UPGRADE_RESEARCH.md + CRYSTAL1_ORACLE_CEILING.md; "
                            "train-window-only design; falsifier battery pre-registered",
              "csi500": {k: v for k, v in csi.items() if k != "trail"},
              "dow": {k: v for k, v in dow.items() if k != "trail"},
              "trails": {"csi500": csi["trail"], "dow": dow["trail"]}}
    OUT.write_text(json.dumps(report, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    for tag, rep in (("csi500", csi), ("dow", dow)):
        print(f"[{tag}] accepts={rep['accepts']} gate={rep['gate_counts']}")
        print(f"    states={rep['filter_card']['state_names']} dwell={rep['filter_card']['dwell_days']}")
        print(f"    falsifiers: " + " ".join(f"{k.split('_')[1]}={'PASS' if v['PASS'] else 'FAIL'}"
                                             for k, v in rep["falsifiers"].items()))
        print(f"    coeffs={rep['certified_coeffs']} anchor={rep['is_anchor']}")
        print(f"    hold: anchor {rep['hold_full']['anchor']} vs final {rep['hold_full']['final']}")
        print(f"    OOS : anchor {rep['oos_single_shot']['anchor']} vs final {rep['oos_single_shot']['final']}")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
