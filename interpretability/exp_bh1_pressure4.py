"""BH1 stage 4 — the corrected Pressure Hypothesis test: does pressure SUSTAIN belief-use?

Stages 1-3 established: (i) the designed pressure is real (oracle +282% ann vs +2% best dial at
c=2); (ii) cold PPO discovers nothing at ANY pressure in 30-90k steps (discovery failure, E-27
again); (iii) the strict fixed-delta fidelity probe is BLIND on saturated-belief step policies —
metric artifact proven — so fidelity here is measured with the boundary-aware CONTRAST-WRITE probe
(write P(bear)=0.2 vs 0.8 at matched states) plus the behavioral regime gap.

The well-posed Richens-Everitt question that remains testable with a discovery-limited optimizer:
start every head from the SAME belief-using teacher (BC of the threshold rule, acc ~0.97) and let
PPO fine-tune under each pressure level. Where using the belief pays (high contrast), fine-tuning
should RETAIN it; where it only costs churn (c=0, the belief is uninformative noise near 0.5),
fine-tuning should ERODE it toward a constant dial.

PREREGISTERED READ (before running): SUPPORTED if contrast-write fidelity after fine-tune rises
with contrast (Spearman > 0) with retention at c=2 at least 0.4 above c=0. KILLED if retention is
flat (pressure does not govern whether belief-use survives optimization). The teacher's own
fidelity (~0.85+) is the ceiling; erosion at c=0 is EXPECTED and is part of the hypothesis, not a
failure.

Run: python interpretability/exp_bh1_pressure4.py     (~10 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_bh1_pressure import gen_market, behavioral_voi, CONTRASTS, SEEDS  # noqa: E402
from interpretability.exp_bh1_pressure2 import bc_oracle_teacher, train  # noqa: E402
from interpretability.exp_bh1_pressure3 import contrast_write_probe  # noqa: E402

OUT = HERE / "exp_bh1_pressure4_report.json"
STEPS = 30_000


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== BH1 stage 4 — retention(pressure): warm-started heads, contrast-write fidelity ===")
    levels = {}
    for c in CONTRASTS:
        per_seed = []
        for seed in SEEDS:
            tr, _ = gen_market(c, seed)
            ev, z_ev = gen_market(c, seed + 500)
            streams = {"train": tr, "dev": ev, "hold": ev}
            t0 = time.time()
            sd, acc = bc_oracle_teacher(streams, seed=seed)
            # teacher fidelity before fine-tune (the ceiling for this seed/contrast)
            from interpretability.crystal_ppo import ExposureEnv, SoftTreeActorCriticPolicy
            from stable_baselines3 import PPO
            env = ExposureEnv(*streams["train"], budget=0.08, lam=2.0, seed=seed)
            m_t = PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=seed,
                      policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0,
                                     "critic_arch": (32, 32)})
            m_t.policy.load_state_dict(sd)
            fid_teacher = contrast_write_probe(m_t)
            m = train(streams, seed, STEPS, 0.005, init_from=sd)
            fid = contrast_write_probe(m)
            eb, er = behavioral_voi(m, ev, z_ev)
            per_seed.append({"seed": seed, "bc_acc": round(acc, 3),
                             "fid_teacher": round(fid_teacher, 3), "fid_after": round(fid, 3),
                             "retention": round(fid - fid_teacher, 3),
                             "regime_gap": round(eb - er, 3), "s": int(time.time() - t0)})
            print(f"  c={c:3.1f} seed {seed}: teacher {fid_teacher:.2f} -> after-PPO {fid:.2f} "
                  f"(gap {eb-er:+.2f}) ({per_seed[-1]['s']}s)")
        levels[str(c)] = {"per_seed": per_seed,
                          "fid_after_mean": round(float(np.mean([s["fid_after"] for s in per_seed])), 3),
                          "fid_teacher_mean": round(float(np.mean([s["fid_teacher"] for s in per_seed])), 3),
                          "gap_mean": round(float(np.mean([s["regime_gap"] for s in per_seed])), 3)}

    f_after = [levels[str(c)]["fid_after_mean"] for c in CONTRASTS]
    from scipy.stats import spearmanr
    rho, _ = spearmanr(CONTRASTS, f_after)
    supported = (rho > 0) and (f_after[-1] - f_after[0] >= 0.4)
    killed = abs(f_after[-1] - f_after[0]) < 0.15
    verdict = (f"SUPPORTED: pressure governs whether belief-use survives optimization — post-PPO "
               f"contrast-write fidelity {f_after[0]:.2f}@c=0 -> {f_after[-1]:.2f}@c=2 (Spearman {rho:+.2f}). "
               "Richens-Everitt holds on the designed market: belief->action causality is kept where shift "
               "pressure makes it pay and eroded where it does not. Bridge implication for the real panel: "
               "the 3%-fidelity head is what a no-pressure substrate makes; raise pressure (or oversample "
               "transitions) before blaming the architecture." if supported else
               (f"KILLED: retention flat ({f_after[0]:.2f} -> {f_after[-1]:.2f}) — pressure does not govern "
                "belief-use survival here; the E-27 collapse must have another cause." if killed else
                f"INCONCLUSIVE: partial slope ({f_after[0]:.2f} -> {f_after[-1]:.2f}, rho {rho:+.2f})."))
    rep = {"experiment": "BH1 stage 4 — retention(pressure) with the boundary-aware probe",
           "preregistration": {"supported": "Spearman>0 AND fid(c=2)-fid(c=0) >= 0.4",
                                "killed": "|fid(c=2)-fid(c=0)| < 0.15",
                                "probe": "contrast-write (0.2 vs 0.8), validated in stage 3"},
           "levels": levels, "spearman": round(float(rho), 3), "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
