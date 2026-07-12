"""Series-G Phase 0 — gate VALIDATION SUITE (does the gate have teeth?).

A gate that only ever says PASS proves nothing. This sweeps four environments and checks the gate
discriminates correctly — the §8 honesty guard that the Phase-0 gate is a real test, not a rubber stamp:

  A  default            — regime informative + modulates economics            → must PASS
  B  nontrivial_blind   — providing is profitable on average, so the belief-blind baseline is a WORKING
                          policy (not do-nothing); the belief must STILL materially dominate it   → must PASS
  C  control_uninformative_obs  — bursts barely distinguish regimes (P(burst|benign)≈P(burst|toxic)),
                          so the belief cannot be inferred                      → must FAIL (VoI≈0)
  D  control_no_modulation      — toxic-PROVIDE payoff == benign-PROVIDE (adverse = −spread), so the regime
                          does not change the payoff of ANY action; the belief is economically worthless
                          → must FAIL (VoI≈0).  [NB: adverse==spread is NOT this control — it sign-flips the
                          payoff (+s benign / −s toxic), which IS modulation; the suite caught that mistake.]

The env is validated for Phase 1 iff {A,B} PASS and {C,D} FAIL — i.e. the belief is load-bearing exactly
when the regime is both inferable AND economically relevant, and not otherwise.

Run: python -m src.series_g.phase0_validate
"""
from __future__ import annotations

import json
from pathlib import Path

from src.series_g.phase0_gate import run
from src.series_g.regime_pomdp import RegimePOMDP

OUT = Path(__file__).resolve().parent / "phase0_validation_report.json"

CONFIGS = {
    "A_default": dict(),
    "B_nontrivial_blind": dict(spread=2.0, adverse=4.0, aggress_cost=0.3, hold_cost=0.05),
    "C_control_uninformative_obs": dict(p_burst_benign=0.40, p_burst_toxic=0.45),
    "D_control_no_modulation": dict(spread=1.0, adverse=-1.0),  # toxic PROVIDE == benign PROVIDE (+1 both)
}
EXPECTED = {"A_default": True, "B_nontrivial_blind": True,
            "C_control_uninformative_obs": False, "D_control_no_modulation": False}


def main() -> int:
    results, ok_all = {}, True
    print(f"{'config':30s} {'aware':>8s} {'blind':>8s} {'ZI_u':>8s} {'VoI':>8s} {'gate':>5s} {'expect':>6s} {'disc?':>5s}")
    for name, over in CONFIGS.items():
        rep = run(RegimePOMDP(**over))
        v = rep["values"]; gate = rep["GATE"]
        passed = gate["material_margin"]
        expect = EXPECTED[name]
        discriminates = (passed == expect)
        ok_all &= discriminates
        results[name] = {"overrides": over, "values": v, "gate": gate,
                         "expected_pass": expect, "discriminates": discriminates,
                         "gm": rep["gm_validation"]}
        print(f"{name:30s} {v['belief_aware_optimum']:8.3f} {v['belief_blind_optimum']:8.3f} "
              f"{v['ZI_floor_unconstrained']:8.3f} {gate['value_of_information']:8.3f} "
              f"{('PASS' if passed else 'FAIL'):>5s} {('PASS' if expect else 'FAIL'):>6s} "
              f"{('yes' if discriminates else 'NO!'):>5s}")

    verdict = ("VALIDATED — gate has teeth: PASSES when regime is inferable AND economically relevant, "
               "FAILS otherwise; the default env is sound for Phase 1"
               if ok_all else
               "BROKEN — gate did not discriminate as expected on >=1 config; do NOT proceed to Phase 1")
    report = {"suite": results, "all_discriminate": ok_all, "PHASE0_VERDICT": verdict}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[phase0-validate] {'ALL DISCRIMINATE' if ok_all else 'DISCRIMINATION FAILURE'}")
    print(f"[phase0-validate] {verdict}")
    print(f"[phase0-validate] wrote {OUT.name}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
