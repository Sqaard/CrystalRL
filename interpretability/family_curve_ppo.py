"""LEARNED frontier curve on the Series-G rotation family — the PPO test of the C*≈K law (W3) + the
belief-bottleneck correlation (W1).

The other agent's ANALYTIC sweep (series_g_family_sweep.py) found: interpretability holds ~ceiling while the
number of regimes G ≤ the concept budget K, then bends (C*≈K). This trains REAL PPO policies at G∈{2,4,8,12}
on the rotation family and measures the same curve on LEARNED behavior:
  x  = bits/action (L0, native Discrete(G+2) alphabet) + phase-shuffle structure
  y9 = simulatability @ K=9 leaves (tree on [belief…, t, inv] → action, temporal 60/40 holdout)
  y64= simulatability @ 64 leaves (the "policy is learnable at all" ceiling — separates budget-insufficient
       from policy-noisy: the C*≈K claim is the y64−y9 GAP opening at G>K)
  H(belief) mean bits + E(actions) — the belief-bottleneck pair (W1: they should co-grow).

Run: python interpretability/family_curve_ppo.py   (trains/loads 4 PPO models, ~10-15 min)
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
ROOT = HERE.parents[0].parent if HERE.name == "interpretability" else HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from src.series_g.family_env import RegimeRotationEnv, oracle_policy  # noqa: E402

GS = [2, 4, 8, 12]
BUDGET = 100_000
OUT = HERE / "family_curve_ppo_report.json"


def rollout(model, G, n_episodes=150, seed=3):
    env = RegimeRotationEnv(G=G, seed=seed)
    rng = np.random.default_rng(seed)
    rows, rets = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done, tot = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            rows.append({"ep": ep, "t": env.t, "action": int(a), "inv": env.inv,
                         "belief_entropy": float(-(env.belief * np.log2(np.clip(env.belief, 1e-12, 1))).sum()),
                         **{f"b{i}": float(env.belief[i]) for i in range(G)}})
            obs, r, term, trunc, _ = env.step(int(a)); tot += r; done = term or trunc
        rets.append(tot)
    return pd.DataFrame(rows), float(np.mean(rets))


def sim_at(df, G, leaves):
    feats = [f"b{i}" for i in range(G)] + ["t", "inv"]
    X = df[feats].to_numpy(float); y = df["action"].to_numpy(int)
    cut = int(len(y) * 0.6)
    if len(np.unique(y[cut:])) < 2:
        return float("nan")
    clf = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(X[:cut], y[:cut])
    return round(float(balanced_accuracy_score(y[cut:], clf.predict(X[cut:]))), 3)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    rows = []
    for G in GS:
        mp = ROOT / f"src/series_g/family_ppo_G{G}.zip"
        env = Monitor(RegimeRotationEnv(G=G, seed=0))
        if mp.exists():
            model = PPO.load(str(mp), env=env, device="cpu"); print(f"[family] G={G}: loaded")
        else:
            model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                        policy_kwargs=dict(net_arch=[128, 128]), seed=0, device="cpu", verbose=0)
            print(f"[family] G={G}: training PPO ({BUDGET})...")
            model.learn(total_timesteps=BUDGET); model.save(str(mp))
        df, ret = rollout(model, G)
        # oracle + random reference
        envr = RegimeRotationEnv(G=G, seed=9); rng = np.random.default_rng(1)
        def run(policy_fn, eps=80):
            tot = 0.0
            for ep in range(eps):
                envr.reset(seed=7000 + ep); done = False
                while not done:
                    _, r, term, trunc, _ = envr.step(policy_fn(envr)); tot += r; done = term or trunc
            return tot / eps
        r_orc = run(oracle_policy); r_rnd = run(lambda e: int(rng.integers(e.G + 2)))
        act = df["action"].to_numpy(int)
        l0 = behavioral_complexity_dynamic(act, kind="discrete", dts=(1, 2), n_null=250, n_boot=250, seed=0)
        row = {"G": G, "n_actions": G + 2, "ppo_return": round(ret, 2), "oracle_return": round(r_orc, 2),
               "random_return": round(r_rnd, 2),
               "learned_frac": round((ret - r_rnd) / (r_orc - r_rnd + 1e-9), 2),
               "x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
               "sim_K9": sim_at(df, G, 9), "sim_K64": sim_at(df, G, 64),
               "H_belief_bits": round(float(df["belief_entropy"].mean()), 3),
               "E_actions": l0["E_range"]}
        row["budget_gap_K9"] = round((row["sim_K64"] or 0) - (row["sim_K9"] or 0), 3)
        rows.append(row)
        print(f"[family] G={G:2d}: ret={ret:6.2f} (orc {r_orc:5.2f})  x={l0['h_mu_range']}  "
              f"simK9={row['sim_K9']}  simK64={row['sim_K64']}  gap={row['budget_gap_K9']}  "
              f"H_belief={row['H_belief_bits']}")
    # W1 belief-bottleneck: co-growth of H(belief) and complexity/E across G
    xs = [np.mean(r["x_h_mu"]) for r in rows]; hb = [r["H_belief_bits"] for r in rows]
    eact = [np.mean(r["E_actions"]) for r in rows]
    from scipy.stats import spearmanr
    rho_x = float(spearmanr(hb, xs).statistic); rho_e = float(spearmanr(hb, eact).statistic)
    # C*≈K verdict: the y64-y9 gap must OPEN as G+2 exceeds ~9
    small = [r["budget_gap_K9"] for r in rows if r["G"] + 2 <= 9]
    large = [r["budget_gap_K9"] for r in rows if r["G"] + 2 > 9]
    bend = bool(large and small and (np.mean(large) > np.mean(small) + 0.05))
    report = {"rows": rows,
              "W3_learned_bend": {"gap_small_G": round(float(np.mean(small)), 3) if small else None,
                                   "gap_large_G": round(float(np.mean(large)), 3) if large else None,
                                   "bend_at_C_star_approx_K": bend},
              "W1_belief_bottleneck": {"spearman_Hbelief_vs_hmu": round(rho_x, 3),
                                        "spearman_Hbelief_vs_Eactions": round(rho_e, 3),
                                        "co_grow": bool(rho_x > 0.7)},
              "verdict": (("LEARNED BEND CONFIRMED (C*≈K survives learning): the K9-vs-K64 simulatability gap opens "
                           "once the number of named modes G+2 exceeds the budget K=9." if bend else
                           "no clear learned bend at K=9 — inspect rows."))}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nW1:", report["W1_belief_bottleneck"]); print("W3:", report["W3_learned_bend"])
    print("VERDICT:", report["verdict"]); print(f"[family] wrote {OUT.name}")


if __name__ == "__main__":
    main()
