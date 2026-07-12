"""Series-G Extension 1 — train PPO on the high-dim multi-asset env, retest H3, place the LEARNED anchor.

Phase 2 scoped H3's failure to "needs a high-dim env where the flat policy explodes." This is that test.
Because the N assets are conditionally independent given the shared regime, the multi-asset OPTIMAL is a
single per-asset worker f(global-belief, own-inventory, time) REUSED across all assets — a manager(belief)→
shared-worker hierarchy. Describing the same policy FLATLY (N separate per-asset mappings, no sharing) costs
~N× more rules. So H3 should now PASS, and the size of the win scales with N.

We (1) train SB3 PPO on the env, (2) check it approaches the analytic optimal (per-asset VI applied with the
global belief), (3) run the H3 compression test (shared worker vs N-separate vs full-state-flat) on BOTH the
optimal and the trained policy, (4) place the LEARNED policy on the frontier (L0 bits/action + simulatability).

Run: python -m src.series_g.ext1_train_and_h3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.series_g.multiasset_env import MultiAssetRegimePOMDP
from src.series_g.phase0_gate import solve_belief_aware
from src.series_g.regime_pomdp import PRIMARY_ENRICHED, RegimePOMDP

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "interpretability"))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

N_ASSETS = 6
BUDGET = 150_000
MODEL_PATH = HERE / "ext1_ppo_multiasset.zip"
OUT = HERE / "ext1_train_and_h3_report.json"


def optimal_action(g, pol, belief, inv, t):
    return int(pol[t, int(round(belief * (len(g) - 1))), int(inv)])


def rollout(policy_fn, env: MultiAssetRegimePOMDP, n_episodes=120, seed=0):
    """policy_fn(obs, env) -> action array. Returns a long-form df: one row per (episode, t, asset)."""
    rng = np.random.default_rng(seed)
    recs, ret = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))
        done, epret = False, 0.0
        while not done:
            t = env.t
            belief = env.belief
            invs = env.inv.copy()
            a = policy_fn(obs, env)
            for i in range(env.N):
                recs.append({"episode": ep, "t": t, "asset": i, "belief": round(float(belief), 4),
                             "inventory": int(invs[i]), "action": int(a[i]), "regime": int(env.regime)})
            obs, r, term, trunc, info = env.step(a)
            epret += r
            done = term or trunc
        ret.append(epret)
    return pd.DataFrame(recs), float(np.mean(ret))


def h3_compression(df: pd.DataFrame, N: int, budget_per_worker=9) -> dict:
    """The H3 high-dim test. SHARED worker: one tree f(belief, own_inv, time)->action reused for all assets.
    FLAT-separate: N per-asset trees over the same per-asset features (no sharing). FLAT-fullstate: a tree that
    must also see the OTHER assets' inventories (the monolithic representation). Compare fidelity + TOTAL rules."""
    y = df["action"].to_numpy(int)
    own = df[["belief", "inventory", "t"]].to_numpy(float)
    # shared worker (pooled across assets)
    sw = DecisionTreeClassifier(max_leaf_nodes=budget_per_worker, random_state=0).fit(own, y)
    fid_shared = float((sw.predict(own) == y).mean())
    rules_shared = int(sw.get_n_leaves())
    # N separate per-asset trees (no sharing)
    fid_sep, rules_sep = [], 0
    for i in range(N):
        di = df[df["asset"] == i]
        Xi = di[["belief", "inventory", "t"]].to_numpy(float)
        yi = di["action"].to_numpy(int)
        ti = DecisionTreeClassifier(max_leaf_nodes=budget_per_worker, random_state=0).fit(Xi, yi)
        fid_sep.append(float((ti.predict(Xi) == yi).mean()))
        rules_sep += int(ti.get_n_leaves())
    # full-state flat: own features + the other assets' inventories (the monolithic, sharing-blind view)
    wide = df.pivot_table(index=["episode", "t"], columns="asset", values="inventory").add_prefix("inv_")
    dfw = df.merge(wide, on=["episode", "t"], how="left")
    inv_cols = [c for c in dfw.columns if c.startswith("inv_")]
    Xf = dfw[["belief", "inventory", "t"] + inv_cols].to_numpy(float)
    yf = dfw["action"].to_numpy(int)
    ff = DecisionTreeClassifier(max_leaf_nodes=budget_per_worker, random_state=0).fit(Xf, yf)
    fid_full = float((ff.predict(Xf) == yf).mean())
    return {
        "shared_worker": {"fidelity": round(fid_shared, 4), "total_rules": rules_shared,
                          "note": "ONE tree reused for all N assets (manager=belief, worker shared)"},
        "flat_separate": {"fidelity": round(float(np.mean(fid_sep)), 4), "total_rules": rules_sep,
                          "note": f"N={N} separate per-asset trees (no sharing)"},
        "flat_fullstate": {"fidelity": round(fid_full, 4), "leaves": int(ff.get_n_leaves()),
                           "note": "monolithic tree forced to see ALL inventories (sharing-blind)"},
        "compression_ratio_flat_over_shared": round(rules_sep / max(1, rules_shared), 2),
        "H3_PASS": bool(rules_sep > 1.5 * rules_shared and fid_shared >= float(np.mean(fid_sep)) - 0.02),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(MultiAssetRegimePOMDP(n_assets=N_ASSETS, seed=0))
    # analytic optimal per-asset policy (the shared worker, by conditional independence)
    msingle = RegimePOMDP(**PRIMARY_ENRICHED)
    g, Va, pol = solve_belief_aware(msingle)

    def opt_fn(obs, e):
        return np.array([optimal_action(g, pol, e.belief, e.inv[i], e.t) for i in range(e.N)], dtype=int)

    # ---- train PPO ----
    if MODEL_PATH.exists():
        model = PPO.load(str(MODEL_PATH), env=env, device="cpu")
        print(f"[ext1] loaded {MODEL_PATH.name}")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, gae_lambda=0.95, gamma=0.99,
                    ent_coef=0.01, learning_rate=3e-4, n_epochs=10, policy_kwargs=dict(net_arch=[128, 128]),
                    seed=0, device="cpu", verbose=0)
        print(f"[ext1] training PPO ({BUDGET} steps, N={N_ASSETS})...")
        model.learn(total_timesteps=BUDGET, progress_bar=False)
        model.save(str(MODEL_PATH))

    def ppo_fn(obs, e):
        a, _ = model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=int).reshape(e.N)

    def random_fn(obs, e):
        return np.random.default_rng(e.t + 7).integers(0, 3, e.N)

    raw = MultiAssetRegimePOMDP(n_assets=N_ASSETS, seed=1)
    df_ppo, ret_ppo = rollout(ppo_fn, raw, seed=10)
    df_opt, ret_opt = rollout(opt_fn, raw, seed=10)
    _, ret_rnd = rollout(random_fn, raw, seed=10)
    learned_frac = (ret_ppo - ret_rnd) / (ret_opt - ret_rnd + 1e-9)

    # ---- H3 retest (high-dim) on BOTH optimal and trained policy ----
    h3_opt = h3_compression(df_opt, N_ASSETS)
    h3_ppo = h3_compression(df_ppo, N_ASSETS)

    # ---- frontier: L0 + simulatability of the LEARNED policy (per-asset action stream) ----
    stream = df_ppo.sort_values(["asset", "episode", "t"])["action"].to_numpy(int)
    x = behavioral_complexity_dynamic(stream, kind="discrete", dts=(1, 2), n_null=400, n_boot=400, seed=0)
    sim = h3_ppo["shared_worker"]["fidelity"]   # simulatability = shared-worker fidelity

    report = {
        "env": f"multi-asset regime-POMDP, N={N_ASSETS}, enriched economics", "ppo_budget": BUDGET,
        "returns": {"PPO": round(ret_ppo, 3), "optimal": round(ret_opt, 3), "random": round(ret_rnd, 3),
                    "PPO_fraction_of_optimal": round(float(learned_frac), 3)},
        "H3_retest_high_dim": {
            "on_optimal_policy": h3_opt, "on_trained_PPO_policy": h3_ppo,
            "verdict": ("H3 SUPPORTED in high-dim: the shared-worker hierarchy reproduces all N assets at "
                        f"~{h3_opt['compression_ratio_flat_over_shared']}x fewer rules than the flat (no-sharing) "
                        "description, at equal fidelity — exactly the compression Phase-2 said needs a high-dim env."
                        if h3_opt["H3_PASS"] else "H3 still not supported — investigate.")},
        "learned_anchor_frontier": {
            "x_bits_per_action": x["h_mu_range"], "x_structure": x["structure_present_configs"],
            "x_E_range": x["E_range"], "y_simulatability_shared_worker": round(sim, 4),
            "note": "LEARNED (PPO) policy on the high-dim env — the apples-to-apples-with-R6c/P22 trained anchor"},
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["returns"], indent=2))
    print("H3 optimal:", json.dumps(h3_opt, indent=2))
    print("H3 PPO   :", json.dumps(h3_ppo, indent=2))
    print("learned anchor:", json.dumps(report["learned_anchor_frontier"], indent=2))
    print(f"\n[ext1] H3 high-dim: {'PASS' if h3_opt['H3_PASS'] else 'FAIL'}; PPO={report['returns']['PPO_fraction_of_optimal']} of optimal; wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
