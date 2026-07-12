"""IV-10 EPISTASIS de-risk: do two belief/lever WRITES on a real trained policy interact NON-ADDITIVELY?

Context: the HL5 synthesis (K/HLX/TB/GV/IV FINALs + braid) converged on ONE weakest construct — IV unit-10's
cumulative-authority CROSS-TERM (the epistasis budget  D_total = Σ|α_i|·τ_i + Σ|ε_ij|·co_τ_ij).  Its
*phenomenon* is Tier-A (Fisher 1918 / Bliss / Loewe / Berenbaum 1989) but its *quantitative* claim was
[PLAUS], unmeasured on any project model.  Per the governing rule ("if the reports' info is insufficient,
do a full run yourself"), this measures it directly on the Series-G corner PPO — the exact belief-write
surface certified_battery_v2.py already uses.

Operationalization: a 2-factor factorial on two WRITES —
    factor A = a belief-write  (force belief b_lo -> b_hi)
    factor B = an inventory-write (force inv iv_lo -> iv_hi)
Outcome y = P(PROVIDE) from the policy's action distribution.  For each context (t, burst) and contrast we
read the 4 cells {neither, A, B, both} and compute the INTERACTION residual against a no-interaction null:
    logit null (log-odds additive, absorbs each single lever's monotone curvature):  eps_logit = L11-(L10+L01-L00)
    linear-prob null (Bliss-like):                                                   eps_prob  = y11-(y10+y01-y00)

Three IV-10 claims tested EMPIRICALLY:
  (1) NON-ADDITIVITY  -> is |eps_logit| materially != 0 ?  (=> per-command C4 budgets don't sum)
  (2) SIGN EPISTASIS  -> does sign(eps_logit) FLIP across the envelope with material magnitude ?
                         (=> the cross-term is envelope-bound, cannot extrapolate OOS)
  (3) NULL DEPENDENCE -> do the logit-null and prob-null DISAGREE on the sign of eps ?  (Berenbaum 1989)
Controls: deterministic zero-check (identical probe twice -> eps==0) and a within-lever "sham pair"
(compose belief 0.1->0.5 and 0.5->0.9 vs 0.1->0.9) giving the single-lever curvature reference — if the
cross-term isn't larger than this sham, it isn't special.

Run: python interpretability/iv10_epistasis_pairedwrite.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402

OUT = HERE / "iv10_epistasis_pairedwrite_report.json"
EPS = 1e-4


def logit(p):
    p = min(1 - EPS, max(EPS, float(p)))
    return float(np.log(p / (1 - p)))


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    import torch
    from stable_baselines3 import PPO
    model = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")
    env = MultiAssetRegimePOMDP(n_assets=1, seed=0)

    def obs_vec(b, t, iv, burst):
        return np.array([2 * b - 1, 2 * t / env.T - 1, 2 * (iv / env.m.I_max) - 1, 1.0 if burst else -1.0],
                        dtype=np.float32)

    def probs(b, t, iv, burst):
        """policy action distribution [P(PROVIDE=0), P(ABSTAIN=1), P(AGGRESS=2)]."""
        with torch.no_grad():
            ot = torch.as_tensor(obs_vec(b, t, iv, burst)).unsqueeze(0)
            dist = model.policy.get_distribution(ot)
            d = dist.distribution
            cat = d[0] if isinstance(d, (list, tuple)) else d   # MultiCategorical -> list of Categorical
            pr = cat.probs.detach().cpu().numpy().reshape(-1)
        return pr

    def yprov(b, t, iv, burst):
        return float(probs(b, t, iv, burst)[0])

    # ---- contexts (nuisance) and the two writable factors' contrasts ----
    CTX = [(t, burst) for t in (2, 6, 10, 14, 18) for burst in (False, True)]
    BPAIRS = [(0.10, 0.50), (0.50, 0.90), (0.10, 0.90)]     # belief-write contrasts (probe sign along belief axis)
    IVPAIRS = [(0, 1), (0, 2), (1, 2)]                       # inventory-write contrasts (all within visited envelope inv<=2)

    cells_logit, cells_prob = [], []
    per = []
    for (t, burst) in CTX:
        for (blo, bhi) in BPAIRS:
            for (ivlo, ivhi) in IVPAIRS:
                y00 = yprov(blo, t, ivlo, burst)      # neither
                y10 = yprov(bhi, t, ivlo, burst)      # A only (belief)
                y01 = yprov(blo, t, ivhi, burst)      # B only (inventory)
                y11 = yprov(bhi, t, ivhi, burst)      # both
                L00, L10, L01, L11 = logit(y00), logit(y10), logit(y01), logit(y11)
                eps_logit = L11 - (L10 + L01 - L00)
                eps_prob = y11 - (y10 + y01 - y00)
                cells_logit.append(eps_logit); cells_prob.append(eps_prob)
                per.append({"t": t, "burst": burst, "bpair": [blo, bhi], "ivpair": [ivlo, ivhi],
                            "y": [round(y00, 4), round(y10, 4), round(y01, 4), round(y11, 4)],
                            "eps_logit": round(eps_logit, 4), "eps_prob": round(eps_prob, 4)})

    el = np.array(cells_logit); ep = np.array(cells_prob)
    MAT = 0.5   # material log-odds shift threshold (~1.65x odds)
    n = len(el)
    n_pos = int(np.sum(el > MAT)); n_neg = int(np.sum(el < -MAT))
    frac_material = round(float(np.mean(np.abs(el) > MAT)), 3)
    # null-model sign disagreement (logit vs linear-prob), among cells where BOTH are non-trivial
    nz = (np.abs(el) > 0.2) & (np.abs(ep) > 0.02)
    disagree = float(np.mean(np.sign(el[nz]) != np.sign(ep[nz]))) if nz.sum() else 0.0

    # ---- control 1: deterministic zero-check ----
    z = yprov(0.3, 6, 1, False); z2 = yprov(0.3, 6, 1, False)
    zero_ok = abs(z - z2) < 1e-9

    # ---- control 2: within-lever SHAM pair (single-lever curvature reference on the logit scale) ----
    # compose belief 0.1->0.5 and 0.5->0.9 vs the direct 0.1->0.9, holding inv/t/burst; residual = curvature
    sham = []
    for (t, burst) in CTX:
        for iv in (0, 1, 2):
            La = logit(yprov(0.10, t, iv, burst)); Lm = logit(yprov(0.50, t, iv, burst)); Lb = logit(yprov(0.90, t, iv, burst))
            # additive-in-two-halves prediction of the full move on logit scale = (Lm-La)+(Lb-Lm) = Lb-La -> residual 0 by identity;
            # the meaningful curvature proxy is the deviation of the MIDPOINT from the chord: Lm - (La+Lb)/2
            sham.append(abs(Lm - 0.5 * (La + Lb)))
    sham_curv = float(np.median(sham))

    med_abs_cross = float(np.median(np.abs(el)))

    # ---- verdicts ----
    non_additive = med_abs_cross > 0.2 and frac_material >= 0.2
    sign_epistasis = (n_pos >= max(2, int(0.05 * n))) and (n_neg >= max(2, int(0.05 * n)))
    null_dependent = disagree >= 0.1
    cross_exceeds_sham = med_abs_cross > 1.5 * sham_curv

    verdict = []
    verdict.append(("NON-ADDITIVITY", "CONFIRMED" if non_additive else "NOT-SUPPORTED",
                    f"median|eps_logit|={med_abs_cross:.3f}, material(>{MAT}) in {frac_material:.0%} of {n} cells"))
    verdict.append(("SIGN-EPISTASIS", "CONFIRMED" if sign_epistasis else "NOT-SUPPORTED",
                    f"material eps both signs: +{n_pos} / -{n_neg} of {n}"))
    verdict.append(("NULL-MODEL-DEPENDENCE", "CONFIRMED" if null_dependent else "NOT-SUPPORTED",
                    f"logit-vs-prob sign disagreement = {disagree:.0%} of non-trivial cells"))
    verdict.append(("CROSS-TERM > SINGLE-LEVER-CURVATURE", "YES" if cross_exceeds_sham else "NO",
                    f"median|cross|={med_abs_cross:.3f} vs sham-curvature={sham_curv:.3f}"))

    iv10_supported = non_additive and sign_epistasis
    report = {
        "substrate": "src/series_g/corner_ppo_n1.zip (Series-G corner PPO; belief-write surface of certified_battery_v2)",
        "design": "2-factor factorial: A=belief-write, B=inventory-write; outcome P(PROVIDE); "
                  f"{len(CTX)} contexts x {len(BPAIRS)} belief-pairs x {len(IVPAIRS)} inv-pairs = {n} interaction cells",
        "controls": {"deterministic_zero_check": bool(zero_ok),
                     "within_lever_sham_curvature_median_logit": round(sham_curv, 4)},
        "stats": {"median_abs_eps_logit": round(med_abs_cross, 4),
                  "frac_material_gt_%.2f" % MAT: frac_material,
                  "n_material_pos": n_pos, "n_material_neg": n_neg,
                  "null_model_sign_disagreement": round(disagree, 3)},
        "verdicts": [{"claim": c, "result": r, "evidence": e} for c, r, e in verdict],
        "IV10_cross_term_supported": bool(iv10_supported),
        "headline": (
            "IV-10 EMPIRICALLY SUPPORTED on a real trained policy: two writes interact non-additively AND the "
            "interaction sign-flips across the belief/inventory envelope — so per-command certificates provably do "
            "NOT compose and the epistasis cross-term is envelope-bound (cannot extrapolate). The combination "
            "certificate + co-activation cap are empirically justified as design; the magnitude, however, is a "
            "property of THIS policy on the polygon, not a market-validated number."
            if iv10_supported else
            "IV-10 cross-term NOT empirically supported on this substrate: interactions are near-additive / "
            "sign-stable — the combination certificate is a low-priority safeguard here (revisit on a higher-dim "
            "belief surface before it gates real changes)."),
        "per_cell": per,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== IV-10 epistasis paired-write de-risk ===")
    print(report["design"])
    print(f"controls: zero_check={zero_ok}  sham_curvature(logit)={sham_curv:.3f}")
    for c, r, e in verdict:
        print(f"  {c:38s}: {r:13s} | {e}")
    print(f"\nIV-10 cross-term supported: {iv10_supported}")
    print("HEADLINE:", report["headline"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
