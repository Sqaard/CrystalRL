"""E-21 — TRANSPARENCY & CONTROLLABILITY audit of the deployed model (the CrystalScore discipline on v9).

The engineering is done (E-12..E-20); this measures how transparent and how agent-controllable the deployed
model actually is, with the same do-not-assume discipline as the original CrystalScore work:

  T1 BELIEF LEGIBILITY (MDL axis): can a small tree (<=8 leaves) reproduce the belief's ARMED/NOT decision
     (P(bear) > t2) from the 4 NAMED observables, vs a 64-leaf ceiling? deficit = 1 - bal_acc8/bal_acc64.
  T2 NAMING FAITHFULNESS: the bear state is NAMED by its VIX emission. Inject counterfactual VIX shocks
     (+5/+10/+20 pts over 20-day windows) into the causal filter: P(bear) must respond monotonically.
     Faithfulness = fraction of windows with monotone response + mean dP(bear)/+10pts.
  T3 MULTI-SEED STABILITY (the ROADMAP item): refit the HMM with seeds 0..9 — pairwise corr of bear-prob
     series, % days with the SAME certified-rule exposure decision, and K-choice stability.
  C1 RE-DISCOVERY: three fresh gate lifetimes from ANCHOR (step-scale 0.15/0.20/0.25) on dev 2019-21 /
     hold 2022-23 (OOS NEVER touched — the loop runner is wrapped to exclude it): does the agent re-certify
     into the neighborhood of the E-15 rule? (accepts, final-config distance).
  C2 REPAIR: hand the loop a deliberately MIS-SET incumbent (over-defensive: t2=0.45, lvl_def=0.20) — can the
     gate-led agent walk it back? (Prediction written first: the typed-forward Pareto admission is a RATCHET
     toward defense — un-defending moves worsen dev maxDD and should be REFUSED; if so, that is an honest,
     measured controllability LIMIT, not a success.)

Run: python interpretability/exp_e21_transparency_audit.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, window, DEV, HOLD  # noqa: E402
from interpretability.hl_v6_crystal1_features import GaussianHMM  # noqa: E402
import interpretability.hl_v8_rebalance_lane as V8  # noqa: E402
from interpretability.hl_v8_rebalance_lane import KNOBS8, ANCHOR8  # noqa: E402

OUT = HERE / "exp_e21_transparency_audit_report.json"
CERTIFIED = {"t1": 0.30, "t2": 0.657, "lvl_reduced": 1.0, "lvl_defensive": 0.738, "H": 10.0}
TRAIN = ("2010-01-01", "2018-12-31")


def belief_from_Z(Z, hmm, bear):
    gamma, _ = hmm.causal_filter(Z)
    return gamma[:, bear]


def fit_frozen(macro, seed=0, k_choices=(2, 3)):
    m_tr = (macro.index >= pd.Timestamp(TRAIN[0])) & (macro.index <= pd.Timestamp(TRAIN[1]))
    X_tr = macro[m_tr].to_numpy(dtype=float)
    mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0) + 1e-9
    Z = np.nan_to_num((macro.to_numpy(dtype=float) - mu) / sd, nan=0.0)
    Z_tr = Z[np.asarray(m_tr)]
    cut = int(len(Z_tr) * 0.8)
    best = None
    for K in k_choices:
        h = GaussianHMM(K); h.fit(Z_tr[:cut], seed=seed)
        _, ll = h.causal_filter(Z_tr[cut:])
        if best is None or ll > best[1]:
            best = (K, ll)
    K = best[0]
    hmm = GaussianHMM(K); hmm.fit(Z_tr, seed=seed)
    bear = int(np.argmax(hmm.mu[:, 0]))
    return Z, hmm, bear, K, (mu, sd)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-21 — transparency & controllability audit ===")
    r, macro = load_extended()
    Z, hmm, bear, K, (mu, sd) = fit_frozen(macro, seed=0)
    b = belief_from_Z(Z, hmm, bear)

    # ---- T1: belief legibility (MDL 8 vs 64 leaves) on the ARMED decision -----------------------
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import balanced_accuracy_score
    X = macro.to_numpy(dtype=float)
    y = (b > CERTIFIED["t2"]).astype(int)
    cut = int(len(y) * 0.6)
    ok = np.isfinite(X).all(1)
    Xa, ya = X[ok], y[ok]; cut = int(len(ya) * 0.6)
    t1 = {}
    if len(np.unique(ya[:cut])) > 1 and len(np.unique(ya[cut:])) > 1:
        s8 = balanced_accuracy_score(ya[cut:], DecisionTreeClassifier(max_leaf_nodes=8, random_state=0)
                                     .fit(Xa[:cut], ya[:cut]).predict(Xa[cut:]))
        s64 = balanced_accuracy_score(ya[cut:], DecisionTreeClassifier(max_leaf_nodes=64, random_state=0)
                                      .fit(Xa[:cut], ya[:cut]).predict(Xa[cut:]))
        t1 = {"bal_acc_8leaf": round(float(s8), 3), "bal_acc_64leaf": round(float(s64), 3),
              "mdl_deficit": round(float(max(0.0, 1 - s8 / max(s64, 1e-9))), 3),
              "armed_base_rate": round(float(ya.mean()), 4)}
    print(f"T1 belief legibility: {t1}")

    # ---- T2: naming faithfulness (counterfactual VIX shocks through the causal filter) ----------
    rng = np.random.default_rng(0)
    starts = rng.choice(np.arange(2500, len(Z) - 40), size=40, replace=False)   # post-2019-ish windows
    mono, dmean = 0, []
    for s in starts:
        resp = []
        for shock in (0.0, 5.0, 10.0, 20.0):
            Zs = Z.copy()
            Zs[s:s + 20, 0] = Zs[s:s + 20, 0] + shock / sd[0]      # +pts of VIX, standardized
            bb = belief_from_Z(Zs, hmm, bear)
            resp.append(float(bb[s:s + 20].mean()))
        if resp[0] <= resp[1] <= resp[2] <= resp[3]:
            mono += 1
        dmean.append((resp[2] - resp[0]))
    t2 = {"monotone_frac": round(mono / len(starts), 3),
          "mean_dPbear_per_+10VIXpts": round(float(np.mean(dmean)), 3), "n_windows": len(starts)}
    print(f"T2 naming faithfulness: {t2}")

    # ---- T3: multi-seed stability ----------------------------------------------------------------
    beliefs, Ks = [], []
    for seed in range(10):
        _, h_s, bear_s, K_s, _ = fit_frozen(macro, seed=seed)
        beliefs.append(belief_from_Z(Z, h_s, bear_s)); Ks.append(K_s)
    cors, agree = [], []
    base_dec = (beliefs[0] > CERTIFIED["t2"])
    for i in range(len(beliefs)):
        for j in range(i + 1, len(beliefs)):
            cors.append(float(np.corrcoef(beliefs[i], beliefs[j])[0, 1]))
        if i > 0:
            agree.append(float(((beliefs[i] > CERTIFIED["t2"]) == base_dec).mean()))
    t3 = {"K_choices": {str(k): Ks.count(k) for k in set(Ks)},
          "pairwise_belief_corr_mean": round(float(np.mean(cors)), 3),
          "pairwise_belief_corr_min": round(float(np.min(cors)), 3),
          "decision_agreement_vs_seed0_mean": round(float(np.mean(agree)), 4)}
    print(f"T3 multi-seed stability: {t3}")

    # ---- C1/C2: controllability lifetimes (OOS never touched: pass hold as the report window) ----
    bel = pd.Series(b, index=macro.index)
    dev, hold = window(r, bel, *DEV), window(r, bel, *HOLD)

    def lifetime(start_cfg, step_scale, tag, rounds=40):
        saved = None
        try:
            import interpretability.hl_v8_rebalance_lane as m
            gate = m.make_gate(dev, hold)
            current = dict(start_cfg)
            gate.frontier = [(dict(current), gate.vec(current))]
            from src.hl.mechanism_bandit import MechanismBandit
            bandit = MechanismBandit(arms=list(m.ARMS8))
            step = {k: step_scale * (v[1] - v[0]) for k, v in KNOBS8.items()}
            direction = {k: -1.0 for k in KNOBS8}
            counts = {}
            for rnd in range(rounds):
                arm = bandit.select(m.ARMS8)
                cand = dict(current)
                if arm == "joint_defend":
                    cand["t2"] = float(np.clip(cand["t2"] - step["t2"], *KNOBS8["t2"][:2]))
                    cand["lvl_defensive"] = float(np.clip(cand["lvl_defensive"] - step["lvl_defensive"], *KNOBS8["lvl_defensive"][:2]))
                else:
                    lo, hi, _ = KNOBS8[arm]
                    cand[arm] = float(np.clip(cand[arm] + step[arm] * direction[arm], lo, hi))
                verdict, info, current = gate.review(cand, current)
                ok = verdict.startswith("ACCEPTED")
                bandit.update(arm, 1.0 if ok else 0.0)
                if not ok and arm != "joint_defend":
                    step[arm] *= 0.6; direction[arm] = -direction[arm]
                counts[verdict] = counts.get(verdict, 0) + 1
            dist = float(np.sqrt(sum(((current[k] - CERTIFIED[k]) / (KNOBS8[k][1] - KNOBS8[k][0])) ** 2
                                     for k in KNOBS8)))
            return {"tag": tag, "accepts": counts.get("ACCEPTED_RISKMODE", 0), "gate_counts": counts,
                    "final": {k: round(float(current[k]), 3) for k in KNOBS8},
                    "dist_to_certified_norm": round(dist, 3)}
        finally:
            saved = saved

    c1 = [lifetime(ANCHOR8, s, f"rediscovery_step{s}") for s in (0.15, 0.20, 0.25)]
    for run in c1:
        print(f"C1 {run['tag']}: accepts={run['accepts']} final={run['final']} dist={run['dist_to_certified_norm']}")
    misset = {"t1": 0.30, "t2": 0.45, "lvl_reduced": 1.0, "lvl_defensive": 0.20, "H": 10.0}
    c2 = lifetime(misset, 0.20, "repair_from_overdefensive")
    d0 = float(np.sqrt(sum(((misset[k] - CERTIFIED[k]) / (KNOBS8[k][1] - KNOBS8[k][0])) ** 2 for k in KNOBS8)))
    c2["dist_start"] = round(d0, 3)
    print(f"C2 repair: accepts={c2['accepts']} start_dist={c2['dist_start']} -> final_dist={c2['dist_to_certified_norm']} "
          f"final={c2['final']} counts={c2['gate_counts']}")

    rediscover_rate = sum(1 for run in c1 if run["accepts"] > 0) / len(c1)
    scorecard = {
        "transparency": {"belief_mdl_deficit": t1.get("mdl_deficit"),
                          "naming_faithfulness_monotone": t2["monotone_frac"],
                          "seed_stability_decision_agreement": t3["decision_agreement_vs_seed0_mean"]},
        "controllability": {"rediscovery_certify_rate": rediscover_rate,
                             "rediscovery_mean_dist": round(float(np.mean([x["dist_to_certified_norm"] for x in c1])), 3),
                             "repair_improved_distance": bool(c2["dist_to_certified_norm"] < c2["dist_start"]),
                             "repair_accepts": c2["accepts"]},
    }
    rep = {"experiment": "E-21 transparency & controllability audit (deployed v9 model)",
           "T1_belief_legibility": t1, "T2_naming_faithfulness": t2, "T3_multiseed_stability": t3,
           "C1_rediscovery": c1, "C2_repair": c2, "scorecard": scorecard}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nSCORECARD:", json.dumps(scorecard, indent=2))
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
