"""BH1 stage 5 — the transplant test: does PPO ERODE a working rule when pressure is removed?

Stage 4's rising curve (0.00@c=0 -> 0.92@c=2) conflates two mechanisms: at c=0 the belief never
leaves ~0.5, so the TEACHER itself cannot encode the rule (install fails, ceiling 0.00) — the
erosion prediction was never actually tested. And wherever the teacher DID install the rule,
fine-tune changed nothing (0.92->0.92 at every level) — consistent with a lazy/inert fine-tune.

The clean retention test: transplant the c=2 teacher (rule installed, contrast-write ~0.9) into
the c=0 market (belief = uninformative noise near 0.5; conditioning on it pays nothing and costs
churn) and PPO fine-tune 30k. Per Richens-Everitt, removed pressure should ERODE the rule.

PREREGISTERED READ: EROSION if post-transplant contrast-write fidelity drops by >= 0.4 from the
teacher's; INERT-OPTIMIZER if it stays within 0.15 (then stage 4's "retention under pressure" is
vacuous — fine-tune preserves everything, and the honest BH1 story is install-only); in between:
partial. Control: the same teacher fine-tuned on c=2 (stage-4 arm) stayed at 0.92.

Run: python interpretability/exp_bh1_pressure5.py    (~3 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_bh1_pressure import gen_market  # noqa: E402
from interpretability.exp_bh1_pressure2 import bc_oracle_teacher, train  # noqa: E402
from interpretability.exp_bh1_pressure3 import contrast_write_probe  # noqa: E402

OUT = HERE / "exp_bh1_pressure5_report.json"


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from interpretability.crystal_ppo import ExposureEnv, SoftTreeActorCriticPolicy
    from stable_baselines3 import PPO
    print("=== BH1 stage 5 — transplant: c=2 teacher fine-tuned on the c=0 (no-pressure) market ===")
    per_seed = []
    for seed in (0, 1):
        tr2, _ = gen_market(2.0, seed)            # rule-rich market: teacher source
        tr0, _ = gen_market(0.0, seed)            # pressure-free market: fine-tune target
        s2 = {"train": tr2, "dev": tr2, "hold": tr2}
        s0 = {"train": tr0, "dev": tr0, "hold": tr0}
        t0 = time.time()
        sd, acc = bc_oracle_teacher(s2, seed=seed)
        env = ExposureEnv(*s2["train"], budget=0.08, lam=2.0, seed=seed)
        m_t = PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=seed,
                  policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0,
                                 "critic_arch": (32, 32)})
        m_t.policy.load_state_dict(sd)
        fid_before = contrast_write_probe(m_t)
        m = train(s0, seed, 30_000, 0.005, init_from=sd)   # fine-tune where the rule pays NOTHING
        fid_after = contrast_write_probe(m)
        per_seed.append({"seed": seed, "bc_acc": round(acc, 3), "fid_teacher": round(fid_before, 3),
                         "fid_after_c0_finetune": round(fid_after, 3),
                         "drop": round(fid_before - fid_after, 3), "s": int(time.time() - t0)})
        print(f"  seed {seed}: teacher(c=2) {fid_before:.2f} -> after 30k fine-tune on c=0: {fid_after:.2f}")
    drop = float(np.mean([s["drop"] for s in per_seed]))
    fb = float(np.mean([s["fid_teacher"] for s in per_seed]))
    fa = float(np.mean([s["fid_after_c0_finetune"] for s in per_seed]))
    if drop >= 0.4:
        verdict = (f"EROSION CONFIRMED: removing pressure erodes a working rule ({fb:.2f} -> {fa:.2f}). "
                   "Together with stage 4, BOTH halves of the Pressure Hypothesis hold on the designed "
                   "market: pressure is needed to install belief-use AND to keep it.")
    elif drop <= 0.15:
        verdict = (f"INERT OPTIMIZER: the rule survives fine-tune even where it pays nothing "
                   f"({fb:.2f} -> {fa:.2f}) — stage 4's flat retention rows were vacuous; the honest BH1 "
                   "claim is INSTALL-ONLY: pressure governs what a teacher can encode (belief information "
                   "content), while 30k-step PPO fine-tune is behaviorally conservative either way. "
                   "The real-panel implication narrows to: warm-start installs causality (E-28 cross-check "
                   "0.77-0.83) and short fine-tunes will not remove it — but they will not create it either.")
    else:
        verdict = f"PARTIAL EROSION ({fb:.2f} -> {fa:.2f}, drop {drop:.2f}) — pressure matters but slowly."
    rep = {"experiment": "BH1 stage 5 — transplant erosion test (c=2 teacher fine-tuned on c=0)",
           "preregistration": {"erosion": "drop >= 0.4", "inert": "drop <= 0.15",
                                "control": "same teacher fine-tuned on c=2 stayed 0.92 (stage 4)"},
           "per_seed": per_seed, "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
