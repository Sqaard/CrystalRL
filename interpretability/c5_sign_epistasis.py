"""C-5 (part) — the owed >=2-BELIEF-DIM SIGN-EPISTASIS run. IV-10's cross-term was measured on the corner
(belief x INVENTORY, 1-belief-dim): non-additive but SIGN-STABLE. The scary half — sign-epistasis / OOS sign
reversal, which motivated the co-activation cap + forbidden-pair registry — needs TWO genuine BELIEF coordinates.
The G-regime family env (src/series_g/family_env.py) gives a G-simplex belief written directly into the obs.

Substrate: family_G4 PPO (src/crystal/_b2/family_G4_s1.zip) on RegimeRotationEnv(G=4). Belief is a 4-simplex; a
"write to venue i" puts mass on b_i. 2-factor factorial: factor A = mass on venue i (lo/hi), factor B = mass on
venue j (lo/hi); outcome y = P(provide @ venue i). Interaction on the log-odds scale:
    eps_logit = L(both) - (L(A_only)+L(B_only)-L(neither)).
Run over many (venue-pair x context) cells and ask: does sign(eps_logit) FLIP (sign epistasis), unlike the corner?
Run: python interpretability/c5_sign_epistasis.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.family_env import RegimeRotationEnv  # noqa: E402

OUT = HERE / "c5_sign_epistasis_report.json"
EPS = 1e-4


def logit(p):
    p = min(1 - EPS, max(EPS, float(p))); return float(np.log(p / (1 - p)))


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    import torch
    from stable_baselines3 import PPO
    G = 4
    env = RegimeRotationEnv(G=G, seed=0)
    model = PPO.load(str(ROOT / "src/crystal/_b2/family_G4_s1.zip"), device="cpu")
    sev = env.sev

    def obs_of(b, t, inv):
        y = float(np.dot(b, sev))                              # belief-consistent severity signal
        return np.concatenate([b, [2.0 * t / env.T - 1.0, inv / env.I_max, y]]).astype(np.float32)

    def probs(b, t, inv):
        with torch.no_grad():
            d = model.policy.get_distribution(torch.as_tensor(obs_of(b, t, inv)).unsqueeze(0)).distribution
            cat = d[0] if isinstance(d, (list, tuple)) else d
            return cat.probs.detach().cpu().numpy().reshape(-1)

    def belief(vec_i, vi, vec_j, vj, base=0.05):
        raw = np.full(G, base); raw[vec_i] = vi; raw[vec_j] = vj
        return raw / raw.sum()

    LO, HI = 0.05, 0.55
    pairs = [(i, j) for i in range(G) for j in range(G) if i < j]     # 6 venue pairs
    ctxs = [(t, inv) for t in (2, 6, 10, 14) for inv in (0, 1, 2)]
    # THREE outcomes so sign-epistasis gets a fair chance: own venue i (competition -> tends negative), a THIRD
    # venue k (not mechanically constrained), and ABSTAIN (=G). k/abstain can flip sign freely.
    cells = []
    for (i, j) in pairs:
        k = next(v for v in range(G) if v not in (i, j))             # a third venue
        for (t, inv) in ctxs:
            P00 = probs(belief(i, LO, j, LO), t, inv)
            P10 = probs(belief(i, HI, j, LO), t, inv)
            P01 = probs(belief(i, LO, j, HI), t, inv)
            P11 = probs(belief(i, HI, j, HI), t, inv)
            for name, idx in (("own_i", i), ("third_k", k), ("abstain", env.ABSTAIN)):
                el = logit(P11[idx]) - (logit(P10[idx]) + logit(P01[idx]) - logit(P00[idx]))
                cells.append({"pair": [i, j], "outcome": name, "idx": int(idx), "t": t, "inv": inv,
                              "eps_logit": round(float(el), 4)})

    MAT = 0.5
    by_outcome = {}
    for name in ("own_i", "third_k", "abstain"):
        e = np.array([c["eps_logit"] for c in cells if c["outcome"] == name])
        np_, nn_ = int(np.sum(e > MAT)), int(np.sum(e < -MAT))
        by_outcome[name] = {"n": len(e), "median_abs": round(float(np.median(np.abs(e))), 4),
                            "frac_material": round(float(np.mean(np.abs(e) > MAT)), 3),
                            "n_material_pos": np_, "n_material_neg": nn_,
                            "both_signs_material": bool(np_ >= 2 and nn_ >= 2)}
    el_all = np.array([c["eps_logit"] for c in cells])
    n = len(el_all)
    med_abs = round(float(np.median(np.abs(el_all))), 4)
    frac_material = round(float(np.mean(np.abs(el_all) > MAT)), 3)
    n_pos = int(np.sum(el_all > MAT)); n_neg = int(np.sum(el_all < -MAT))
    # sign-epistasis if ANY outcome shows material cross-terms of BOTH signs
    sign_epistasis = any(v["both_signs_material"] for v in by_outcome.values())
    non_additive = med_abs > 0.2 and frac_material >= 0.2

    report = {
        "substrate": "family_G4 PPO on RegimeRotationEnv(G=4); belief = 4-simplex; factor A/B = mass on venue i/j; outcome P(provide@i)",
        "design": f"{len(pairs)} venue-pairs x {len(ctxs)} contexts = {n} belief x belief interaction cells",
        "stats_all": {"n_cells": n, "median_abs_eps_logit": med_abs, "frac_material_gt_0.5": frac_material,
                      "n_material_pos": n_pos, "n_material_neg": n_neg},
        "by_outcome": by_outcome,
        "NON_ADDITIVITY": bool(non_additive),
        "SIGN_EPISTASIS": bool(sign_epistasis),
        "contrast_with_corner": ("corner (belief x inventory, 1-belief-dim): non-additive but SIGN-STABLE (+68/-0). "
                                 "This G=4 belief x belief run is the >=2-belief-dim test IV-10 owed."),
        "verdict": (
            "SIGN-EPISTASIS CONFIRMED on a >=2-belief-dim surface: the cross-term takes BOTH signs "
            f"(+{n_pos}/-{n_neg} of {n} material cells). So two belief-writes can interact with a sign that reverses "
            "across venue-pairs/context — the OOS-reversal IV-10 warned about is REAL on genuine multi-belief-dim "
            "writes (the corner's belief x inventory could not show it). => the co-activation cap + forbidden-pair "
            "registry + worst-case cross-term bounds move from [PLAUS] toward [SBC, measured]; per-command certs "
            "provably do not compose, and the combination certificate must carry a SIGNED, non-commutative cross-term."
            if sign_epistasis else
            "NO sign-epistasis even on >=2-belief-dim: material cross-terms are one-signed like the corner. The "
            "sign-reversal machinery stays [PLAUS]; non-additivity " + ("holds" if non_additive else "is weak") + "."),
        "caveats": ["Forced beliefs may sit off the natural manifold (the governor handles that in deployment); a "
                    "single trained family policy; polygon, teeth only where VoI>0."],
        "cells": cells,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== C-5 belief x belief sign-epistasis (G=4) ===")
    print(report["design"] + " x 3 outcomes")
    print(f"ALL: median|eps_logit|={med_abs}  material(>{MAT}) {frac_material:.0%}  signs +{n_pos}/-{n_neg} of {n}")
    for name, v in by_outcome.items():
        print(f"  outcome {name:8s}: median|eps|={v['median_abs']:.3f}  material={v['frac_material']:.0%}  "
              f"+{v['n_material_pos']}/-{v['n_material_neg']}  both_signs={v['both_signs_material']}")
    print(f"NON_ADDITIVITY={non_additive}  SIGN_EPISTASIS={sign_epistasis}")
    print(report["verdict"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
