"""B5b — the law-testing contrast: self-crystallization bonus at K=14 (= G+2, the budget MATCHING the task).

B5 at K=9 (< C*≈G+2=14) was weak/mixed — consistent with "you cannot train into a budget below C*".
If that reading is right, the SAME bonus at K=14 should work: sim@K14(arm) > sim@K14(baselines).
Run: python interpretability/b5b_k14.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from b5_crystallize import CrystallizeWrapper, rollout, distill, sim_at  # noqa: E402
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from src.series_g.family_env import RegimeRotationEnv  # noqa: E402

G, LEAVES, SEED = 12, 14, 1
OUT = HERE / "b5b_k14_report.json"


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    # sim@K14 baselines from the three existing G=12 models
    base = []
    for tag, p in [("s0", ROOT / "src/series_g/family_ppo_G12.zip"),
                   ("s1", ROOT / "src/crystal/_b2/family_G12_s1.zip"),
                   ("s2", ROOT / "src/crystal/_b2/family_G12_s2.zip")]:
        if not p.exists():
            continue
        m = PPO.load(str(p), device="cpu")
        d, ret = rollout(m, G, n=120)
        base.append({"tag": tag, "sim_K14": sim_at(d, G, LEAVES), "return": round(ret, 2)})
    print("[b5b] baselines sim@K14:", base)

    # the arm: identical schedule to B5, bonus tree at K=14
    wrap = CrystallizeWrapper(RegimeRotationEnv(G=G, seed=SEED))
    env = Monitor(wrap)
    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                policy_kwargs=dict(net_arch=[128, 128]), seed=SEED, device="cpu", verbose=0)
    model.learn(total_timesteps=40_000)
    agree = []
    for rnd in (1, 2, 3):
        d, _ = rollout(model, G, n=100)
        wrap.tree = distill(d, G, leaves=LEAVES)
        model.learn(total_timesteps=30_000, reset_num_timesteps=False)
        d2, _ = rollout(model, G, n=60)
        feats = [f"b{i}" for i in range(G)] + ["t", "inv"]
        agree.append(round(float((wrap.tree.predict(d2[feats].to_numpy(float)) == d2["action"].to_numpy(int)).mean()), 3))
    d, ret = rollout(model, G, n=150)
    s14 = sim_at(d, G, LEAVES)
    l0 = behavioral_complexity_dynamic(d["action"].to_numpy(int), kind="discrete", dts=(1, 2),
                                       n_null=200, n_boot=200, seed=0)
    best_base = max(b["sim_K14"] for b in base)
    rep = {"baselines": base, "arm": {"seed": SEED, "sim_K14": s14, "return": round(ret, 2),
           "x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
           "story_agreement_by_round": agree},
           "arm_beats_all_baselines": bool(s14 is not None and s14 > best_base + 0.03),
           "law_reading": None}
    rep["law_reading"] = ("CONFIRMS the constructive C*≈K reading: the bonus WORKS when the budget matches the "
                          "task (K=14=G+2) though it failed below C* (K=9)." if rep["arm_beats_all_baselines"]
                          else "arm does not beat baselines at K=14 either — the bonus itself is weak, not (only) the budget.")
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
