"""E-27c — full action-probability diagnostics of the SAVED E-27/E-28 PPO heads (no retraining).

The external audit (2026-07-10) claimed the saved heads are near-uniform and their "constant dials" are
deterministic argmax tie-breaks, not learned preferences. This script verifies that claim against the saved
models and persists the evidence: per-head mean max-prob, entropy vs log(5), argmax variation over a fixed
54-point obs grid, and the full per-leaf action probabilities that E-27 originally failed to save.

Expected reading (from the interactive verification): every cold head sits at max-prob 0.204-0.221 with
entropy ~1.609/1.6094 and a grid-constant argmax => E-27's "converged to constant dials" must be downgraded
to "did not train away from uniform; the dial identity is a tie-break artifact". The one genuinely
structured head is e28T_warm (teacher-BC then PPO): the TEACHER lever is the only lever that demonstrably
shaped a policy on this substrate.

Run: python interpretability/exp_e27c_prob_diagnostics.py       (seconds; loads saved zips only)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO  # noqa: E402
from interpretability.crystal_ppo import MODELS, policy_prob_diagnostics  # noqa: E402

OUT = HERE / "exp_e27c_prob_diagnostics_report.json"
UNIFORM_BAND = 0.03      # |max_prob - 1/5| below this + grid-constant argmax => tie-break verdict


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-27c — action-probability diagnostics of the saved PPO heads ===")
    heads = {}
    for mz in sorted(MODELS.glob("*.zip")):
        m = PPO.load(mz, device="cpu")
        d = policy_prob_diagnostics(m)
        near_uniform = abs(d["mean_max_prob"] - 0.2) < UNIFORM_BAND and not d["argmax_varies_over_grid"]
        d["verdict"] = ("NEAR-UNIFORM: the deterministic dial is an argmax tie-break, not a learned policy"
                        if near_uniform else "structured policy (trained signal present)")
        heads[mz.stem] = d
        print(f"[{mz.stem:12s}] max-prob {d['mean_max_prob']:.4f} | entropy {d['mean_entropy']:.4f}/"
              f"{d['max_entropy']:.4f} | argmax varies: {d['argmax_varies_over_grid']} -> {d['verdict'][:14]}")
    n_uni = sum(1 for d in heads.values() if d["verdict"].startswith("NEAR-UNIFORM"))
    rep = {"experiment": "E-27c action-probability diagnostics (saved models; no retraining)",
           "obs_grid": "P(bear) x prev_ex x dd_state = 6x3x3 = 54 points",
           "heads": heads,
           "verdict": (f"{n_uni}/{len(heads)} heads NEAR-UNIFORM => E-27's 'constant dials' were argmax "
                       f"tie-breaks of near-untrained policies; the exception(s) "
                       f"{[k for k, d in heads.items() if not d['verdict'].startswith('NEAR-UNIFORM')]} "
                       f"carry real structure (the teacher lever). E-28 lever-R (budget->dial mapping) and "
                       f"lever-I (+/-2 nats on near-zero logits) are DOWNGRADED accordingly; lever-T stands.")}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nVERDICT:", rep["verdict"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
