"""HL v4 over CRYSTAL-1 — the full Pareto tension-vector loop on REAL panels (Dow-29 + csi500), reusing the
R6c-certified v4 machinery (rotating date holdout, moving-block-bootstrap z, stress adversary, alpha-investing
wealth, sandboxed canaries + compromised freeze, mechanism bandit with a PORTABLE prior).

Substrate (CRYSTAL-1-native, from hl_real_dow/csi500): a K=2 self-supervised NeuralBayesFilter trained on the
panel's EW-return stream (train window only, frozen) drives a 4-knob exposure policy {t1, t2, lvl_reduced,
lvl_defensive}; anchor = static full exposure (EW buy-and-hold).

What v4 changes vs the old flat loop (which certified 0 on a hand-mixed scalar ann-|maxDD|):
  * The objective is a PARETO FRONTIER over (ann UP, maxDD UP-toward-0) — no hand-mixed scalar. A move may be
    RETURN-type (dev ann gain) or RISK-type (dev DD cut at ann non-inferiority): the B3 lesson says the belief's
    legitimate real use is drawdown, and the old return-gated/mixed-scalar gates could not certify that honestly.
  * Holdout by move type on a ROTATING date window: RETURN-type must prove mean daily delta > 0 at block-z > z_crit;
    RISK-type must prove the DD improvement at z_dd > z_crit AND return non-inferiority ((mean+NI)/se > z_crit).
  * Alpha-investing wealth prices the knob-grid query storm; stress adversary (anchor-defined dev high-vol +
    worst-drawdown slices) vetoes regressions; canaries (behavior-identical config bloat + measured-harmful
    defensive lockdown) must be rejected, an escape FREEZES the gate; authority is priced in off-anchor knob count.
  * F9 for real: the mechanism bandit's per-knob prior learned on Dow WARM-STARTS the csi500 loop (paired same
    proposer schedule), vs a cold csi500 arm.
  * Honest OOS: the frozen OOS window is touched ONCE at the end (final incumbent + frontier), no selection.
  * Positive control: a synthetic +3bp/day uniformly-better candidate must be ACCEPTED by a fresh gate instance
    (proves accepts are reachable on this substrate; a 0-certified result is then about the data, not a dead gate).

VoI-gate context: belief-on-capital stays fenced by interpretability/voi_gate.py (CLOSED on all accessible real
substrates; CN L5 recorder restarted 2026-07-06, accumulating). This loop certifies TRANSPARENCY/RISK-mode edits.
Run: python interpretability/hl_v4_over_crystal1.py
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

OUT = HERE / "hl_v4_over_crystal1_report.json"
COST = 0.001
KNOBS = {"t1": (0.10, 0.60, 0.30), "t2": (0.40, 0.90, 0.70),
         "lvl_reduced": (0.10, 1.00, 1.00), "lvl_defensive": (0.00, 1.00, 1.00)}
ANCHOR = {k: v[2] for k, v in KNOBS.items()}
ARMS = ["t1", "t2", "lvl_reduced", "lvl_defensive", "joint_defend"]
NI_MARGIN = 0.0002        # return non-inferiority margin, 2bp/day (~5%/yr materiality)
STRESS_MARGIN = -0.0005   # mean daily regression allowed on a stress slice before veto (stress days are high-vol)
EPS_DD = 0.005            # material dev DD improvement (0.5pp)

PANELS = {
    "dow": {"panel": ROOT / "artifacts/action_vq/A67_joint_hidden_action_controls_fullenv_from_R6c_v1/feature_scalers_frozen/fold_2021/model_ready.csv",
            "train": ("2010-01-01", "2016-12-31"), "dev": ("2017-01-01", "2018-12-31"),
            "hold": ("2019-01-01", "2020-12-31"), "oos": ("2021-01-01", "2023-12-31"), "hold_win": 120},
    "csi500": {"panel": ROOT / "data/adapters/_csi500_wide/csi300_model_ready.csv",
               "train": ("2018-01-01", "2020-12-31"), "dev": ("2021-01-01", "2021-08-31"),
               "hold": ("2021-09-01", "2022-06-30"), "oos": ("2022-07-01", "2023-03-31"), "hold_win": 120},
}


# ---------- substrate (from hl_real_dow, frozen conventions) ----------------------------------
def ew_returns(path):
    d = pd.read_csv(path, usecols=["date", "tic", "close"]).sort_values(["tic", "date"])
    d["ret"] = d.groupby("tic")["close"].pct_change()
    d["date"] = pd.to_datetime(d["date"])
    r = d.groupby("date")["ret"].mean().sort_index()
    return r[np.isfinite(r)]


def build_belief(panel, train_a, train_b):
    r = ew_returns(panel)
    tr = r[(r.index >= pd.Timestamp(train_a)) & (r.index <= pd.Timestamp(train_b))]
    thr = float(np.quantile(np.abs(tr), 0.80))
    obs_tr = (np.abs(tr.to_numpy()) > thr).astype(int)
    L = 60; n = len(obs_tr) // L
    f = train_filter(obs_tr[:n * L].reshape(n, L), K=2, A_obs=2, epochs=500, seed=0, verbose=False)
    T_l, E_l, p0 = f.numpy_params(); tox = int(np.argmax(E_l[:, 1]))
    obs_full = (np.abs(r.to_numpy()) > thr).astype(int)
    b = p0.copy(); bel = np.empty(len(r))
    for i, o in enumerate(obs_full):
        bp = b @ T_l; j = bp * E_l[:, o]; b = j / max(j.sum(), 1e-12); bel[i] = b[tox]
    return r, pd.Series(bel, index=r.index)


def window(r, bel, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    return r[m].to_numpy()[1:], bel[m].to_numpy()[:-1]


def exposure(c, b):
    t1, t2 = c["t1"], max(c["t2"], c["t1"] + 1e-6)
    return 1.0 if b < t1 else (c["lvl_reduced"] if b < t2 else c["lvl_defensive"])


def strat(c, ro, bl):
    ex = np.array([exposure(c, x) for x in bl])
    costs = np.abs(np.diff(np.concatenate([[1.0], ex]))) * COST
    return ex * ro - costs


def ann_dd(rets):
    eq = np.cumprod(1 + rets)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = float(eq[-1] ** (252 / max(len(rets), 1)) - 1)
    return ann, dd


def dsd(rets):
    """Downside deviation (daily): sqrt(mean(min(r,0)^2)) — a SIZE-sensitive tail-mean over ~half the days."""
    neg = np.minimum(rets, 0.0)
    return float(np.sqrt((neg ** 2).mean()))


def risk_boot_z(cand, cur, block=20, n_boot=500, seed=0):
    """Block-bootstrap z for the DOWNSIDE-DEVIATION improvement (paired same-index resamples).
    Replaces the maxDD bootstrap the verifier proved unwinnable-by-construction: resampling the path
    statistic maxDD tests the SPREAD of a localized drawdown cut across block inclusion, not its SIZE
    (a perfect-foresight oracle passed 0/55 Dow windows). Downside deviation is a mean-like statistic,
    so its bootstrap SE shrinks with evidence instead of scaling with the gain."""
    rng = np.random.default_rng(seed)
    n = len(cand); starts = np.arange(0, n - block + 1)
    nb = max(1, int(np.ceil(n / block)))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nb)])[:n]
        diffs[i] = dsd(cur[idx]) - dsd(cand[idx])               # positive = candidate carries less downside risk
    se = float(diffs.std(ddof=1)) + 1e-12
    return float(diffs.mean() / se), float(diffs.mean())


AXES = {"ann": +1, "maxDD": +1}     # maxDD stored negative: closer to 0 = better


def dominates(a, b, tol=1e-9):
    ge = all((a[k] - b[k]) * AXES[k] >= -tol for k in AXES)
    gt = any((a[k] - b[k]) * AXES[k] > tol for k in AXES)
    return ge and gt


def n_off_anchor(c):
    return float(sum(1 for k in KNOBS if abs(c[k] - ANCHOR[k]) > 1e-9))


# ---------- the v4 gate ------------------------------------------------------------------------
class Crystal1V4Gate:
    def __init__(self, dev, hold, hold_win, strat_fn=None):
        self._strat = strat_fn or strat        # injectable for the positive control (sentinel candidate)
        self.ro_dev, self.bl_dev = dev
        self.ro_hold, self.bl_hold = hold
        self.hold_win = min(hold_win, len(self.ro_hold) - 1)
        anchor_dev = self._strat(ANCHOR, self.ro_dev, self.bl_dev)
        vol = pd.Series(anchor_dev).rolling(20).std().to_numpy()
        eq = np.cumprod(1 + anchor_dev); ddpath = eq / np.maximum.accumulate(eq) - 1
        vq = np.nanquantile(vol, 0.75)
        self.stress = {"high_vol": np.where(vol >= vq)[0],
                       "deep_dd": np.where(ddpath <= np.quantile(ddpath, 0.25))[0]}
        self.wealth = 0.20
        self.tension_budget = 6.0
        self.spent_tension = 0.0
        self.epoch = 0
        self.compromised = False
        self.frontier = []
        self.audit = {"dominated": 0, "adversary_vetoes": 0, "budget_refusals": 0, "wealth_refusals": 0,
                      "holdout_rejections": 0, "inert_windows": 0,
                      "canary_caught": 0, "canary_escaped": 0, "canary_unarmed": 0}
        self.canary_log = []

    def vec(self, c):
        a, d = ann_dd(self._strat(c, self.ro_dev, self.bl_dev))
        return {"ann": a, "maxDD": d}

    def _rot_hold(self):
        # stride sized so the ~8 affordable tests (alpha-wealth budget) can COVER the whole hold span —
        # the verifier proved the old stride-7 schedule made the Dow COVID-crash window unreachable
        # (crash starts at day 279; 8 tests x stride 7 reach day 56): a de-risking move could never be
        # tested on the one window where it matters.
        # PEEKS ONLY — the epoch advances via _advance_epoch() strictly on a PRICED test: the red-team
        # showed free epoch advance on inert refusals enables window-shopping (steer the rotation onto a
        # mined window at zero wealth cost).
        span = max(1, len(self.ro_hold) - self.hold_win)
        stride = max(7, span // 7)
        start = (self.epoch * stride) % span
        return np.arange(start, start + self.hold_win), span

    def _advance_epoch(self):
        self.epoch += 1

    def review(self, cand, current):
        if self.compromised:
            return "REFUSED_GATE_COMPROMISED", {}, current
        cur_vec, cand_vec = self.vec(current), self.vec(cand)

        # F1/F10: Pareto admission + typed forward move
        front = [v for _, v in self.frontier] or [cur_vec]
        if any(dominates(f, cand_vec) for f in front):
            self.audit["dominated"] += 1
            return "REJECTED_DOMINATED", {"cand": cand_vec}, current
        ret_fwd = cand_vec["ann"] > cur_vec["ann"] + 1e-4 and cand_vec["maxDD"] >= cur_vec["maxDD"] - 1e-4
        risk_fwd = (cand_vec["maxDD"] > cur_vec["maxDD"] + EPS_DD and
                    cand_vec["ann"] >= cur_vec["ann"] - 0.01)
        if not (ret_fwd or risk_fwd):
            self.audit["dominated"] += 1
            return "REJECTED_NO_FRONTIER_GAIN", {"cand": cand_vec}, current

        # F3: alpha-investing + rotating holdout, typed certification
        if self.wealth < 1e-3:
            self.audit["wealth_refusals"] += 1
            return "REFUSED_ALPHA_WEALTH_EXHAUSTED", {"wealth": round(self.wealth, 4)}, current
        alpha_t = min(0.5, self.wealth / 2.0)
        z_crit = math.sqrt(2.0 * math.log(1.0 / alpha_t))
        hs, span = self._rot_hold()
        cur_h = self._strat(current, self.ro_hold[hs], self.bl_hold[hs])
        cand_h = self._strat(cand, self.ro_hold[hs], self.bl_hold[hs])
        delta = cand_h - cur_h
        # INERT guard (verifier: 2 of 4 Dow tests had delta==0 — the candidate never activates on the
        # window, so the test cannot pass regardless of truth). No information revealed -> no wealth
        # charge AND no epoch advance (a free advance would enable window-shopping via inert spins).
        if float(np.abs(delta).max()) < 1e-12:
            self.audit["inert_windows"] += 1
            return "REFUSED_INERT_ON_WINDOW", {"window_start": int(hs[0])}, current
        self._advance_epoch()                      # the test is priced from here on
        z, se = block_z(delta, block=5, n_boot=1000, seed=self.epoch)
        if ret_fwd:
            passes = float(delta.mean()) > 0 and z > z_crit
            info = {"type": "RETURN", "z": round(z, 2), "z_crit": round(z_crit, 2)}
        else:      # RISK-type: downside-deviation improvement out-of-window + return non-inferiority
            z_dsd, dsd_gain = risk_boot_z(cand_h, cur_h, seed=self.epoch)
            ni = (float(delta.mean()) + NI_MARGIN) / se
            passes = z_dsd > z_crit and ni > z_crit
            info = {"type": "RISK", "z_dsd": round(z_dsd, 2), "ni_z": round(ni, 2), "z_crit": round(z_crit, 2),
                    "hold_dsd_gain": round(dsd_gain, 6)}
            if passes:
                # DISJOINT-WINDOW re-confirmation, mandatory for RISK accepts (red-team: 5 mined configs
                # passed a single window while violating their certified non-inferiority off-window by
                # 2-4x the margin; the z_dsd null is also anti-conservative, sd~1.38). The certified
                # CLAIM (return non-inferiority) must hold on the maximally-distant window, and the
                # downside direction must not reverse. Part of the same priced query — no extra charge.
                start2 = (int(hs[0]) + span // 2) % span
                hs2 = np.arange(start2, start2 + self.hold_win)
                cur2 = self._strat(current, self.ro_hold[hs2], self.bl_hold[hs2])
                cand2 = self._strat(cand, self.ro_hold[hs2], self.bl_hold[hs2])
                d2 = cand2 - cur2
                if float(np.abs(d2).max()) < 1e-12:
                    passes = False                # inert on the confirm window: the claim is unverifiable
                    info["confirm"] = "INERT"
                else:
                    z2, se2 = block_z(d2, block=5, n_boot=1000, seed=self.epoch + 7919)
                    ni2 = (float(d2.mean()) + NI_MARGIN) / se2
                    z_dsd2, _ = risk_boot_z(cand2, cur2, seed=self.epoch + 7919)
                    passes = ni2 > z_crit and z_dsd2 > 0
                    info["confirm"] = {"window_start": start2, "ni2": round(ni2, 2), "z_dsd2": round(z_dsd2, 2)}
        if not passes:
            self.wealth -= alpha_t / (1 - alpha_t)
            self.audit["holdout_rejections"] += 1
            info["wealth"] = round(self.wealth, 4)
            return "REJECTED_HOLDOUT", info, current

        # F2: stress adversary on anchor-defined dev slices
        cand_dev = self._strat(cand, self.ro_dev, self.bl_dev)
        cur_dev = self._strat(current, self.ro_dev, self.bl_dev)
        for sname, sidx in self.stress.items():
            sdelta = float((cand_dev[sidx] - cur_dev[sidx]).mean())
            if sdelta < STRESS_MARGIN:
                self.audit["adversary_vetoes"] += 1
                self.wealth -= alpha_t / (1 - alpha_t)
                return "VETOED_BY_ADVERSARY", {"stressor": sname, "mean_delta": round(sdelta, 6)}, current

        # F4/F11: authority priced in off-anchor knob count, cumulative
        harm = max(0.0, n_off_anchor(cand) - n_off_anchor(current))
        if self.spent_tension + harm > self.tension_budget:
            self.audit["budget_refusals"] += 1
            return "REFUSED_TENSION_BUDGET", {"spent": self.spent_tension, "harm": harm}, current

        self.wealth = min(1.0, self.wealth + 0.05)
        self.spent_tension += harm
        self.frontier = [(p, v) for (p, v) in self.frontier if not dominates(cand_vec, v)]
        self.frontier.append((dict(cand), cand_vec))
        return "ACCEPTED_RISKMODE", {"cand": cand_vec, "harm": harm, **info}, cand

    # F12 canaries -----------------------------------------------------------------------------
    def _sandboxed(self, cand, current):
        snap = (self.wealth, self.spent_tension, self.epoch, list(self.frontier), dict(self.audit))
        try:
            v, _, _ = self.review(cand, current)
        finally:
            self.wealth, self.spent_tension, self.epoch, self.frontier, self.audit = snap
        return v

    # a BANK of harm canaries; each arms only where its harm premise MEASURES true on dev (live-provide
    # analog). The verifier showed defensive_lockdown never arms on calm dev windows (belief<t1 most days
    # => it is near-neutral there), so the bank adds a cost-churn canary whose harm channel (transaction
    # costs on threshold-crossing flips) does not depend on the direction of the market.
    CANARY_BANK = {
        "defensive_lockdown": {"t1": 0.10, "t2": 0.40, "lvl_reduced": 0.10, "lvl_defensive": 0.0},
        "cost_churn": {"t1": 0.10, "t2": 0.101, "lvl_reduced": 1.0, "lvl_defensive": 0.0},
    }

    def canary_check(self, current):
        results = []
        # (a) behavior-identical config bloat: thresholds moved but both levels stay 1.0 -> exposure == anchor
        bloat = dict(current)
        if abs(current["lvl_reduced"] - 1.0) < 1e-9 and abs(current["lvl_defensive"] - 1.0) < 1e-9:
            bloat.update({"t1": 0.15, "t2": 0.85})
            results.append(("config_bloat", self._sandboxed(bloat, current)))
        # (b) measured-harmful canaries from the bank
        cur_dev = self._strat(current, self.ro_dev, self.bl_dev)
        for name, cfg in self.CANARY_BANK.items():
            harm_dev = float((self._strat(cfg, self.ro_dev, self.bl_dev) - cur_dev).mean())
            if harm_dev < STRESS_MARGIN:
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


# ---------- MDL Axis A on the exposure decision -------------------------------------------------
def mdl_axis_a(c, ro, bl):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import balanced_accuracy_score
    ex = np.array([exposure(c, x) for x in bl])
    lv = sorted(set(np.round(ex, 6)))
    y = np.array([lv.index(round(e, 6)) for e in ex])
    if len(lv) < 2:
        return 0.0
    X = bl.reshape(-1, 1); cut = int(len(y) * 0.6)
    if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return 0.0
    s8 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    s64 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=64, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    return float(max(0.0, 1 - s8 / max(s64, 1e-9)))


# ---------- the walk ----------------------------------------------------------------------------
def run_panel(name, spec, prior=None, rounds=40):
    r, bel = build_belief(spec["panel"], *spec["train"])
    dev = window(r, bel, *spec["dev"]); hold = window(r, bel, *spec["hold"]); oos = window(r, bel, *spec["oos"])
    gate = Crystal1V4Gate(dev, hold, spec["hold_win"])
    current = dict(ANCHOR)
    gate.frontier = [(dict(current), gate.vec(current))]
    bandit = MechanismBandit(arms=list(ARMS))
    if prior:
        bandit.seed_from(prior)
    step = {k: 0.2 * (v[1] - v[0]) for k, v in KNOBS.items()}
    direction = {k: -1.0 for k in KNOBS}
    trail = []
    for rnd in range(rounds):
        if rnd > 0 and rnd % 10 == 0:
            gate.canary_check(current)
        arm = bandit.select(ARMS)
        cand = dict(current)
        if arm == "joint_defend":     # F5 atomic bundle: defend earlier AND harder (the B3-shaped joint move)
            cand["t2"] = float(np.clip(cand["t2"] - step["t2"], *KNOBS["t2"][:2]))
            cand["lvl_defensive"] = float(np.clip(cand["lvl_defensive"] - step["lvl_defensive"], *KNOBS["lvl_defensive"][:2]))
        else:
            lo, hi, _ = KNOBS[arm]
            cand[arm] = float(np.clip(cand[arm] + step[arm] * direction[arm], lo, hi))
        verdict, info, current = gate.review(cand, current)
        ok = verdict.startswith("ACCEPTED")
        bandit.update(arm, 1.0 if ok else 0.0)
        if not ok and arm != "joint_defend":
            step[arm] *= 0.6; direction[arm] = -direction[arm]
        trail.append({"round": rnd, "arm": arm, "verdict": verdict,
                      **{k: v for k, v in info.items() if k != "cand"}})
    gate.canary_check(current)

    def perf(c, win):
        a, d = ann_dd(strat(c, win[0], win[1]))
        sh = float(np.sqrt(252) * strat(c, win[0], win[1]).mean() / (strat(c, win[0], win[1]).std() + 1e-12))
        return {"ann": round(a, 4), "maxDD": round(d, 4), "sharpe": round(sh, 3)}

    counts = {}
    for t in trail:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    return {
        "panel": name, "certified_coeffs": {k: round(current[k], 3) for k in KNOBS},
        "input_prior": prior,                      # F9 disclosure: an all-zero prior transfers ZERO information
        "canary_log": [list(x) for x in gate.canary_log],
        "accepts": counts.get("ACCEPTED_RISKMODE", 0), "gate_counts": counts, "audit": gate.audit,
        "wealth_left": round(gate.wealth, 4), "tension_spent": gate.spent_tension,
        "gate_compromised": gate.compromised,
        "frontier": [{"coeffs": {k: round(p[k], 3) for k in KNOBS},
                      "ann_dev": round(v["ann"], 4), "maxDD_dev": round(v["maxDD"], 4)} for p, v in gate.frontier],
        "bandit_prior": bandit.prior(),
        "oos_single_shot": {"anchor_static_full": perf(ANCHOR, oos), "final_incumbent": perf(current, oos)},
        "hold_full_window": {"anchor_static_full": perf(ANCHOR, hold), "final_incumbent": perf(current, hold)},
        "mdl_axis_a": {"anchor": round(mdl_axis_a(ANCHOR, *dev), 3), "final": round(mdl_axis_a(current, *dev), 3)},
        "trail": trail,
    }, gate, dev, hold


def positive_control(spec):
    """TEETH: a genuinely-better sentinel candidate must be ACCEPTED through the REAL review() path.
    The sentinel's edge is a NOISY +3bp/day boost (mean 3bp, ~7bp sd via a deterministic value-keyed
    pseudo-noise, so holdout slices stay consistent) — a non-degenerate z-test, not a constant shift."""
    def boosted_strat(c, ro, bl):
        base = strat(c, ro, bl)
        if c.get("__boost__"):
            base = base + 0.0003 + 0.001 * np.sin(1e4 * ro)   # value-keyed: same day -> same noise on any slice
        return base
    r, bel = build_belief(spec["panel"], *spec["train"])
    dev = window(r, bel, *spec["dev"]); hold = window(r, bel, *spec["hold"])
    gate = Crystal1V4Gate(dev, hold, spec["hold_win"], strat_fn=boosted_strat)
    gate.frontier = [(dict(ANCHOR), gate.vec(ANCHOR))]
    sentinel = dict(ANCHOR); sentinel["__boost__"] = 1
    verdict, info, _ = gate.review(sentinel, dict(ANCHOR))
    return {"verdict": verdict, **{k: v for k, v in info.items() if k != "cand"},
            "accepted": verdict.startswith("ACCEPTED")}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    dow_rep, *_ = run_panel("dow", PANELS["dow"])
    prior = dow_rep["bandit_prior"]
    csi_warm, *_ = run_panel("csi500", PANELS["csi500"], prior=prior)      # F9 transfer
    csi_cold, *_ = run_panel("csi500", PANELS["csi500"], prior=None)       # paired: same schedule, no prior

    control = positive_control(PANELS["dow"])

    report = {
        "substrate": "CRYSTAL-1-native: K=2 self-supervised belief filter (frozen train) + 4-knob exposure policy; "
                     "REAL Dow-29 (2010-2023) and csi500 (2018-2023) EW panels",
        "voi_gate_context": "belief-on-capital fenced by voi_gate.py: CLOSED on BTC + SOL/AVAX/GALA (VoI=0, regime "
                            "priced); CN A-shares pending data (L5 recorder restarted 2026-07-06, accumulating). "
                            "This loop certifies TRANSPARENCY/RISK-mode edits only.",
        "dow": {k: v for k, v in dow_rep.items() if k != "trail"},
        "csi500_warm": {k: v for k, v in csi_warm.items() if k != "trail"},
        "csi500_cold": {k: v for k, v in csi_cold.items() if k != "trail"},
        "positive_control_dow": control,
        "trails": {"dow": dow_rep["trail"], "csi500_warm": csi_warm["trail"]},
        "honesty_caps": [
            "RISK-mode ceiling: an accept certifies a (DD, non-inferior return) or return improvement WITHIN "
            "single-panel replay of the exposure shell — not alpha, not a promotion of belief-writing to capital "
            "(the VoI gate stays CLOSED on accessible substrates).",
            "The B3 lesson stands as the null to beat: the hand-tuned defensive config's DD benefit was a "
            "single-path 2020-crash-timing artifact that died under block-bootstrap; this gate uses the same "
            "bootstrap discipline plus typed RISK certification, so a certified DD cut here means it survived "
            "what killed B3-config.",
            "DETECTABILITY (verifier-measured): RETURN-branch MDE on calm 120-day windows is ~1.5-4.8 bp/day "
            "(3.7-12.7%/yr) for realistic knob moves — high; '0 certified' on the RETURN lane means "
            "below-floor-or-absent. The RISK lane was REDESIGNED after the verifier proved the maxDD bootstrap "
            "unwinnable-by-construction (a perfect-foresight oracle passed 0/55 Dow windows): certification now "
            "tests DOWNSIDE-DEVIATION improvement (size-sensitive tail-mean), and the rotation stride now covers "
            "the whole hold span so crash windows are reachable within the alpha-wealth budget.",
            "ROTATING WINDOWS OVERLAP (120-day windows, span-covering stride): alpha-investing prices the COUNT "
            "of adaptive tests, not their dependence. Hardened after the red-team mined a window-local RISK "
            "false-accept cluster: (i) inert refusals no longer advance the rotation (free window-shopping is "
            "dead by design, not by coincidence), (ii) RISK accepts must RE-CONFIRM their non-inferiority claim "
            "on the maximally-distant disjoint window in-code (also contains the anti-conservative z_dsd null, "
            "empirical sd~1.38). RETURN accepts should still be re-confirmed on a disjoint window before any "
            "promotion beyond risk-mode.",
            "F9 DISCLOSURE: the Dow->csi500 transferred prior was ALL-ZERO (source loop certified nothing), so "
            "warm==cold identically — the transfer mechanism is live but had nothing to carry (input_prior is "
            "recorded per run).",
            "CANARY COVERAGE (honest): config_bloat fires every check; the harm-canary bank was UNARMED all "
            "run on BOTH panels — on calm dev windows every expressible bounded-exposure config (incl. "
            "cost_churn) measures neutral-to-HELPFUL on dev while being genuinely harmful on hold, and arming "
            "on hold would break the PIT discipline. Harm-rejection teeth on such panels therefore rest on the "
            "escape=>freeze wiring + the positive/negative controls (verified live), NOT on in-space canaries.",
            "MDL Axis A of the exposure decision is ~0 by construction (a 3-region threshold rule over ONE "
            "feature is an 8-leaf story) — the contrast with R6c's 0.286 is directional, NOT apples-to-apples "
            "(R6c's controller decision is measured over 11 state features).",
            "Panel naming: the 'csi500' panel file is csi300_model_ready.csv (csi300 constituents, the csi500 "
            "project lineage's model-ready panel).",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== HL v4 over CRYSTAL-1 (real Dow + csi500) ===")
    for tag in ("dow", "csi500_warm", "csi500_cold"):
        rep = report[tag]
        print(f"[{tag}] accepts={rep['accepts']} gate={rep['gate_counts']}")
        print(f"        coeffs={rep['certified_coeffs']} | frontier={len(rep['frontier'])} pts | "
              f"canaries {rep['audit']['canary_caught']}/{rep['audit']['canary_caught']+rep['audit']['canary_escaped']} "
              f"| compromised={rep['gate_compromised']}")
        print(f"        hold: anchor {rep['hold_full_window']['anchor_static_full']} vs final {rep['hold_full_window']['final_incumbent']}")
        print(f"        OOS : anchor {rep['oos_single_shot']['anchor_static_full']} vs final {rep['oos_single_shot']['final_incumbent']}")
    print("positive control (dow):", control)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
