"""BH1 stage 3 — is the strict monotone write-test BLIND on saturated-belief step policies?

Stage 2 found warm heads that BEHAVE like the oracle (regime gap +0.87/+0.90 vs oracle 1.0) yet
score fidelity 0.00-0.03 on the strict E-28 monotone probe. Suspected mechanism: at c=2 the Bayes
belief saturates near 0/1; the probe's fixed deltas (+0.2/+0.4, capped at 1.0) then never cross the
policy's 0.5 threshold, so a PERFECT belief-conditioned step policy reads as flat.

Decisive check on a policy that IS the rule by construction — the BC-distilled oracle (acc 0.97):
  (a) the strict E-28 probe on it          -> if ~0, the metric artifact is PROVEN;
  (b) a CONTRAST-WRITE probe (write P(bear)=0.2 vs 0.8 at the same state; count exposure drops)
                                            -> if ~1, the same policy is fully causal under a
                                               boundary-aware measurement;
  (c) belief saturation stats at each contrast (fraction of eval days with bl in (0.1, 0.5), the
      only region where the fixed-delta probe can register a crossing).
Also (d): the contrast-write probe on a COLD stage-1-style head (constant dial) must read ~0 —
otherwise the new probe is confounded.

Run: python interpretability/exp_bh1_pressure3.py    (~2 min CPU)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.crystal_ppo import LEVELS  # noqa: E402
from interpretability.exp_bh1_pressure import gen_market, fidelity_probe, CONTRASTS  # noqa: E402
from interpretability.exp_bh1_pressure2 import bc_oracle_teacher, train  # noqa: E402

OUT = HERE / "exp_bh1_pressure3_report.json"
C_TOP = 2.0


def contrast_write_probe(model, n_probes=60, seed=28):
    """Write P(bear)=0.2 vs 0.8 at matched states; count strict exposure drops."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_probes):
        pex = float(rng.choice(LEVELS)); dd = float(-rng.uniform(0, 0.06))
        lo = np.array([[0.2, pex, dd]], dtype=np.float32)
        hi = np.array([[0.8, pex, dd]], dtype=np.float32)
        a_lo, _ = model.predict(lo, deterministic=True)
        a_hi, _ = model.predict(hi, deterministic=True)
        e_lo = float(LEVELS[int(np.asarray(a_lo).reshape(-1)[0])])
        e_hi = float(LEVELS[int(np.asarray(a_hi).reshape(-1)[0])])
        hits += int(e_lo > e_hi)
    return hits / n_probes


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    import torch
    from stable_baselines3 import PPO as _  # noqa: F401
    print("=== BH1 stage 3 — probing the probe: strict-delta vs contrast-write on known policies ===")

    # (c) belief saturation per contrast
    sat = {}
    for c in CONTRASTS:
        (_, bl, _), _ = gen_market(c, 700)
        sat[str(c)] = round(float(((bl > 0.1) & (bl < 0.5)).mean()), 3)
    print("  saturation — fraction of days with bl in (0.1,0.5):", sat)

    tr, _ = gen_market(C_TOP, 0)
    ev, _ = gen_market(C_TOP, 500)
    streams = {"train": tr, "dev": ev, "hold": ev}

    # (a)+(b): the BC-distilled oracle — the rule by construction
    sd, acc = bc_oracle_teacher(streams, seed=0)
    # wrap the state dict in a predictable model: rebuild via a 0-step train
    m_bc = train(streams, 0, 0, 0.005, init_from=sd) if False else None
    # cheaper: load into a fresh PPO shell without learning
    from interpretability.crystal_ppo import ExposureEnv, SoftTreeActorCriticPolicy
    from stable_baselines3 import PPO
    env = ExposureEnv(*streams["train"], budget=0.08, lam=2.0, seed=0)
    m_bc = PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=0,
               policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0,
                              "critic_arch": (32, 32)})
    m_bc.policy.load_state_dict(sd)
    strict_bc = fidelity_probe(m_bc, ev[1])
    contrast_bc = contrast_write_probe(m_bc)
    print(f"  BC oracle (acc {acc:.2f}): strict-delta probe {strict_bc:.2f} | contrast-write probe {contrast_bc:.2f}")

    # (d) negative control: a cold constant-dial head (stage-1 conditions, 30k)
    m_cold = train(streams, 0, 30_000, 0.005)
    strict_cold = fidelity_probe(m_cold, ev[1])
    contrast_cold = contrast_write_probe(m_cold)
    print(f"  cold 30k head: strict {strict_cold:.2f} | contrast-write {contrast_cold:.2f}")

    artifact_proven = (acc >= 0.9) and (strict_bc <= 0.15) and (contrast_bc >= 0.8) and (contrast_cold <= 0.2)
    verdict = ("METRIC ARTIFACT PROVEN: a policy that IS the oracle rule by construction "
               f"(BC acc {acc:.2f}) scores {strict_bc:.2f} on the strict fixed-delta probe but "
               f"{contrast_bc:.2f} on the contrast-write probe, while the constant-dial control stays at "
               f"{contrast_cold:.2f}. The strict E-28 probe is BLIND on saturated-belief step policies "
               "(fixed +0.2/+0.4 deltas never cross the threshold when bl sits near 0/1). "
               "Stage-1's flat fidelity curve is VOID as a reading of the Pressure Hypothesis; "
               "fidelity must be measured boundary-aware (contrast-write) alongside behavioral regime gap."
               if artifact_proven else
               f"NOT PROVEN: acc {acc:.2f}, strict {strict_bc:.2f}, contrast {contrast_bc:.2f}, "
               f"cold-control {contrast_cold:.2f} — the artifact story fails its own kill test.")
    rep = {"experiment": "BH1 stage 3 — metric-blindness check (strict fixed-delta vs contrast-write)",
           "belief_saturation_frac_in_probe_window": sat,
           "bc_oracle": {"acc": round(acc, 3), "strict_probe": round(strict_bc, 3),
                          "contrast_write_probe": round(contrast_bc, 3)},
           "cold_control": {"strict_probe": round(strict_cold, 3),
                             "contrast_write_probe": round(contrast_cold, 3)},
           "artifact_proven": bool(artifact_proven), "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
