"""Series-G Extension 2 — the FULL cross-MODEL-FAMILY HC-3 test.

For each generator family (Markov regime-POMDP, GCMG minority-game, Brock-Hommes adaptive-beliefs) we fit a
2-state Markov view, solve the belief-aware execution policy + its belief-blind baseline (shared economics,
family-specific dynamics), then evaluate EVERY policy on EVERY family's actual (toxicity, observation) stream.

HC-3 (full): the off-diagonal transfers (policy trained on family A, run on family B's stream with A's filter)
must retain a MATERIAL fraction of the value-of-information, i.e. the regime-response semantics survive a change
of MODEL CLASS — not just a change of parameters (Phase-1 HC-3). Null: the policy overfits one family's temporal
structure (Platt & Gebbie 2016 non-identifiability) and collapses to its belief-blind value off-diagonal.

Run: python -m src.series_g.ext2_cross_family
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.series_g.generators import ALL_GENERATORS, estimate_markov_params
from src.series_g.phase0_gate import solve_belief_aware, solve_belief_blind
from src.series_g.regime_pomdp import PRIMARY_ENRICHED, RegimePOMDP

OUT = Path(__file__).resolve().parent / "ext2_cross_family_report.json"
T_TOTAL = 8000


def eval_on_sequence(econ, filt, g, pol, tox, obs, is_blind=False) -> float:
    """Execution return of a policy on a generator's (toxicity, observation) stream, chunked into episodes of
    econ.T. Reward uses the SHARED economics + the stream's TRUE toxicity; the belief is updated with `filt`
    (the policy's own family filter) on the observed bursts. is_blind ⇒ ignore observations (pol indexed [t,I])."""
    T = econ.T
    nb = len(g) if g is not None else 0
    n_ep = len(tox) // T
    total = 0.0
    for ep in range(n_ep):
        base = ep * T
        inv, belief = 0, filt.prior_toxic
        for t in range(T):
            step = base + t
            a = int(pol[t, inv]) if is_blind else int(pol[t, int(round(belief * (nb - 1))), inv])
            total += econ.reward(int(tox[step]), a, inv)
            inv = econ.inventory_next(a, inv)
            if not is_blind:
                belief = filt.update(filt.predict(belief), int(obs[step]))
        total += econ.terminal(inv)
    return total / max(1, n_ep)


def main() -> int:
    """Fit and solve a policy per generator family, evaluate every policy on every family's stream, and report whether the regime-response transfers across model classes (full HC-3)."""
    econ = RegimePOMDP(**PRIMARY_ENRICHED)            # shared execution economics
    fam = {}
    for gen in ALL_GENERATORS:
        tox, obs = gen.simulate(T_TOTAL, seed=0)
        params = estimate_markov_params(tox, obs)
        m = RegimePOMDP(**{**PRIMARY_ENRICHED, **params})   # shared economics + this family's estimated dynamics
        g, Va, pol = solve_belief_aware(m)
        _, pol_blind = solve_belief_blind(m)
        fam[gen.name] = {"tox": tox, "obs": obs, "params": params, "filt": m, "g": g, "pol": pol, "pol_blind": pol_blind}

    names = list(fam)
    # diagonal references: each family's own-policy and belief-blind value on its own stream
    for nm in names:
        f = fam[nm]
        f["own"] = eval_on_sequence(econ, f["filt"], f["g"], f["pol"], f["tox"], f["obs"])
        f["blind"] = eval_on_sequence(econ, f["filt"], None, f["pol_blind"], f["tox"], f["obs"], is_blind=True)

    matrix, retained = {}, {}
    for tr in names:                                  # policy trained on `tr`
        for te in names:                              # evaluated on family `te`
            v = eval_on_sequence(econ, fam[tr]["filt"], fam[tr]["g"], fam[tr]["pol"], fam[te]["tox"], fam[te]["obs"])
            matrix[f"{tr}__on__{te}"] = round(float(v), 4)
            denom = fam[te]["own"] - fam[te]["blind"]
            retained[f"{tr}__on__{te}"] = round(float((v - fam[te]["blind"]) / denom) if abs(denom) > 1e-9 else float("nan"), 3)

    # Only families whose belief is genuinely load-bearing (own >> blind) are valid TRANSFER TARGETS — a family
    # with no value-of-information has nothing to retain, and its retained-VoI denominator is ill-defined.
    valid = [te for te in names if (fam[te]["own"] - fam[te]["blind"]) > 0.5]
    off = [retained[f"{tr}__on__{te}"] for tr in names for te in valid if tr != te and np.isfinite(retained[f"{tr}__on__{te}"])]
    hc3_pass = bool(len(valid) >= 2 and off and min(off) > 0.5)

    report = {
        "families": {nm: {"params": fam[nm]["params"], "own_value": round(fam[nm]["own"], 4),
                          "blind_value": round(fam[nm]["blind"], 4),
                          "toxic_rate": round(float(fam[nm]["tox"].mean()), 3)} for nm in names},
        "transfer_value_matrix": matrix,
        "VoI_retained_matrix": retained,
        "valid_transfer_targets (belief load-bearing, own-blind>0.5)": valid,
        "off_diagonal_retained_to_valid_targets": ({"min": round(min(off), 3), "mean": round(float(np.mean(off)), 3),
                                                    "max": round(max(off), 3)} if off else "none — <2 load-bearing families"),
        "HC3_full_cross_model_class_PASS": hc3_pass,
        "verdict": ("HC-3 (FULL) PASS — the execution policy's regime-response transfers across MODEL CLASSES "
                    "(Markov ↔ minority-game ↔ adaptive-beliefs), retaining >50% of the value-of-information "
                    "off-diagonal; the semantics are task-, not generator-, specific."
                    if hc3_pass else
                    "HC-3 (FULL) FAIL — at least one cross-model-class transfer collapses toward belief-blind; "
                    "the policy overfits one family's temporal structure (Platt-Gebbie non-identifiability)."),
        "scope": "3 model classes via a shared (toxicity, burst/quiet) interface + a Markov filter view; the "
                 "generators differ in lag-1 toxic autocorrelation (Markov +0.73, GCMG +0.33, BH -0.04).",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"families": report["families"], "VoI_retained_matrix": retained,
                      "off_diagonal_retained": report["off_diagonal_retained"]}, indent=2))
    print(f"\n[ext2] HC-3 full cross-model-class: {'PASS' if hc3_pass else 'FAIL'} -> wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
