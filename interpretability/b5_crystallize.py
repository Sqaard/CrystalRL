"""CRYSTAL-1 B5 — the constructive turn: the battery as a TRAINING objective (self-crystallization).

Test bed: the G=12 rotation family, where the explanation budget FAILS (sim@K9 baselines 0.583-0.611 across
3 seeds while sim@K64 ~0.93 — the C*≈K bend). The move: round-based training where the policy is periodically
DISTILLED into its own K=9-leaf story tree, then continues training with a small bonus for AGREEING with its
own story — "become the policy your story says you are."

    round 0: plain PPO 40k  →  distill K9 tree
    rounds 1..3: PPO 30k each with reward += λ·1[action == tree(state)]  →  re-distill between rounds
    final eval WITHOUT bonus, identical protocol to the B2 family jobs.

Anti-Goodhart gates (the collapse-to-trivial failure mode is priced in):
    sim@K9 rises materially (> 0.70 vs baselines ≤ 0.611)   [the objective]
    return not collapsed (≥ 3.5; baseline range 3.99-9.14)   [competence held]
    h_mu stays high (> 0.6) AND structure passes the shuffle [no persister collapse]

Run: python interpretability/b5_crystallize.py   (2 seeds, ~15 min)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from src.series_g.family_env import RegimeRotationEnv  # noqa: E402

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym

G = 12
LAM = 0.5
OUT = HERE / "b5_crystallize_report.json"
BASELINES = {"sim_K9": [0.611, 0.594, 0.583], "return": [3.99, 9.14, 4.6]}   # B2 family G=12, seeds 0,1,2


class CrystallizeWrapper(gym.Wrapper):
    """Adds λ·1[action == own-story(state)] to the reward. The story (a K9 tree) is swapped in between rounds."""

    def __init__(self, env, lam=LAM):
        super().__init__(env)
        self.lam = lam
        self.tree = None

    def step(self, action):
        bonus = 0.0
        if self.tree is not None:
            feats = [*[float(b) for b in self.env.belief], float(self.env.t), float(self.env.inv)]
            bonus = self.lam if int(self.tree.predict([feats])[0]) == int(action) else 0.0
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r + bonus, term, trunc, info


def rollout(model, G, n=150, seed0=7_000):
    env = RegimeRotationEnv(G=G, seed=3)
    rows, rets = [], []
    for epi in range(n):
        obs, _ = env.reset(seed=seed0 + epi)
        done, tot = False, 0.0
        while not done:
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            rows.append({"action": a, **{f"b{i}": float(env.belief[i]) for i in range(G)},
                         "t": env.t, "inv": env.inv})
            obs, r, term, trunc, _ = env.step(a); tot += r; done = term or trunc
        rets.append(tot)
    return pd.DataFrame(rows), float(np.mean(rets))


def distill(d, G, leaves=9):
    feats = [f"b{i}" for i in range(G)] + ["t", "inv"]
    X = d[feats].to_numpy(float); y = d["action"].to_numpy(int)
    return DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(X, y)


def sim_at(d, G, leaves):
    feats = [f"b{i}" for i in range(G)] + ["t", "inv"]
    X = d[feats].to_numpy(float); y = d["action"].to_numpy(int); cut = int(len(y) * 0.6)
    if len(np.unique(y[cut:])) < 2:
        return float("nan")
    c = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(X[:cut], y[:cut])
    return round(float(balanced_accuracy_score(y[cut:], c.predict(X[cut:]))), 3)


def run_seed(seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    wrap = CrystallizeWrapper(RegimeRotationEnv(G=G, seed=seed))
    env = Monitor(wrap)
    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                policy_kwargs=dict(net_arch=[128, 128]), seed=seed, device="cpu", verbose=0)
    print(f"[b5 s{seed}] round 0: plain PPO 40k")
    model.learn(total_timesteps=40_000)
    story_agreements = []
    for rnd in (1, 2, 3):
        d, _ = rollout(model, G, n=100)
        wrap.tree = distill(d, G)
        print(f"[b5 s{seed}] round {rnd}: distilled story; PPO 30k with self-agreement bonus")
        model.learn(total_timesteps=30_000, reset_num_timesteps=False)
        # story stability: agreement of the NEW behavior with the story it was trained against
        d2, _ = rollout(model, G, n=60)
        feats = [f"b{i}" for i in range(G)] + ["t", "inv"]
        agree = float((wrap.tree.predict(d2[feats].to_numpy(float)) == d2["action"].to_numpy(int)).mean())
        story_agreements.append(round(agree, 3))
    # final eval WITHOUT bonus, identical to family protocol
    d, ret = rollout(model, G, n=150)
    s9, s64 = sim_at(d, G, 9), sim_at(d, G, 64)
    l0 = behavioral_complexity_dynamic(d["action"].to_numpy(int), kind="discrete", dts=(1, 2),
                                       n_null=200, n_boot=200, seed=0)
    row = {"seed": seed, "sim_K9": s9, "sim_K64": s64, "gap": round((s64 or 0) - (s9 or 0), 3),
           "return": round(ret, 2), "x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
           "story_agreement_by_round": story_agreements,
           "gates": {"sim_up": bool(s9 is not None and s9 > 0.70), "return_ok": bool(ret >= 3.5),
                      "h_mu_ok": bool(min(l0["h_mu_range"]) > 0.6), "structured": l0["structure_present"]}}
    row["pass"] = all(row["gates"].values())
    print(f"[b5 s{seed}] FINAL: simK9={s9} (baselines<=0.611) simK64={s64} ret={ret:.2f} "
          f"h_mu={l0['h_mu_range']} struct={l0['structure_present_configs']} pass={row['pass']}")
    return row


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    rows = [run_seed(s) for s in (1, 2)]
    report = {"env": f"RegimeRotation G={G}", "lambda": LAM, "baselines_B2": BASELINES, "arms": rows,
              "verdict": ("SELF-CRYSTALLIZATION WORKS: training toward its own story raises sim@K9 materially "
                          "without collapsing competence or complexity (anti-Goodhart gates held)."
                          if all(r["pass"] for r in rows) else
                          "not confirmed on all seeds — inspect arms (which gate failed).")}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
