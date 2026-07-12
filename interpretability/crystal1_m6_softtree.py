"""M6 — train the jointly-trained belief-only SOFT-TREE head and certify it closes the C-1 leak in the ACTION
DISTRIBUTION (not just in return, as the C-4 distill pilot did).

The actor is a differentiable soft decision tree (src/crystal/soft_tree_policy.py) reading ONLY [belief, inv, time]
(obs indices [0,2,1]) — structurally blind to the raw `burst` observable (obs idx 3). Trained end-to-end with PPO on
the Series-G corner polygon. Certifications:
  1. RETURN PARITY vs the frozen MLP corner PPO (src/series_g/corner_ppo_n1.zip) on shared held-out episodes.
  2. BURST-LEAK CLOSED IN ACTION DISTRIBUTION: TV(action dist | burst=0 vs burst=1) at fixed (belief,t,inv) == 0
     exactly, by construction — measured, not assumed.
  3. MONOTONE dose (P(PROVIDE) non-increasing in toxic-belief) + a legible leaf table.

Usage: python interpretability/crystal1_m6_softtree.py [total_timesteps]   (default 300000; use 3000 for a smoke test)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402
from src.crystal.soft_tree_policy import SoftTreeActorCriticPolicy  # noqa: E402

OUT = HERE / "crystal1_m6_softtree_report.json"
MODEL_OUT = ROOT / "src/series_g/crystal1_m6_softtree.zip"
FEAT_IDX = [0, 2, 1]   # belief, inv, time  (obs = [belief, time, inv, burst]) — burst (idx 3) EXCLUDED
TREE_DEPTH = 3         # 8 leaves, matching the C-4 winning budget


def rollout_return(env, policy_fn, seeds):
    rets = []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False; R = 0.0
        while not done:
            obs, r, term, trunc, _ = env.step([policy_fn(obs)]); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    import torch
    from stable_baselines3 import PPO
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0     # gate sharpness (higher = crisper, closer to a hard tree)
    smoke = total_timesteps < 20_000

    env = MultiAssetRegimePOMDP(n_assets=1, seed=0)
    model = PPO(SoftTreeActorCriticPolicy, env, verbose=0, seed=0,
                n_steps=1024, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.003, learning_rate=3e-4, max_grad_norm=0.5,
                policy_kwargs=dict(feat_idx=FEAT_IDX, tree_depth=TREE_DEPTH, beta=beta, critic_arch=(64, 64)))

    # ---- baseline MLP corner PPO ----
    base_model = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")

    # ---- (optional) BC WARM-START: behavior-clone the soft tree to the MLP's actions on [belief,inv,time] BEFORE PPO.
    # This is the TB_gen_08 lesson (BC warm-start of an RL fine-tune). The distilled hard tree (C-4) proved parity is
    # representable; a BC warm-start gives PPO a near-parity crisp init so soft-mixture optimization doesn't strand it. ----
    warmstart = ("warmstart" in sys.argv)
    if warmstart:
        OBS, ACT = [], []
        for epi in range(300):
            obs, _ = env.reset(seed=50_000 + epi); done = False
            while not done:
                a = int(np.asarray(base_model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
                OBS.append(np.asarray(obs, float).copy()); ACT.append(a)
                obs, r, term, trunc, _ = env.step([a]); done = term or trunc
        Xb = torch.as_tensor(np.array(OBS), dtype=torch.float32)[:, FEAT_IDX]
        Yb = torch.as_tensor(np.array(ACT), dtype=torch.long)
        opt = torch.optim.Adam(model.policy.tree.parameters(), lr=5e-2)
        for ep in range(400):
            opt.zero_grad()
            logits = model.policy.tree(Xb)              # log-mixture-probs = valid CE logits
            loss = torch.nn.functional.cross_entropy(logits, Yb)
            loss.backward(); opt.step()
        print(f"[warmstart] BC pretrain done: final CE={float(loss):.3f} acc={float((logits.argmax(1)==Yb).float().mean()):.3f}")

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(str(MODEL_OUT))
    pol = model.policy

    def mlp_act(obs): return int(np.asarray(base_model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
    def tree_act(obs): return int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])

    seeds = [20_000 + i for i in range(300 if not smoke else 40)]
    base = rollout_return(env, mlp_act, seeds); tree = rollout_return(env, tree_act, seeds)
    base_mean, base_sem = float(base.mean()), float(base.std(ddof=1) / np.sqrt(len(base)))
    tree_mean, tree_sem = float(tree.mean()), float(tree.std(ddof=1) / np.sqrt(len(tree)))
    gap = round(tree_mean - base_mean, 3)
    parity = tree_mean >= base_mean - 0.5 * base_sem
    # paired difference
    diff = tree - base; dmean = float(diff.mean()); dsem = float(diff.std(ddof=1) / np.sqrt(len(diff)))

    # ---- burst-leak in the ACTION DISTRIBUTION: TV(dist|burst=0 vs burst=1) at fixed (belief,t,inv) ----
    def obs_vec(b, t, iv, burst):
        return np.array([2 * b - 1, 2 * t / env.T - 1, 2 * (iv / env.m.I_max) - 1, 1.0 if burst else -1.0], dtype=np.float32)

    def tree_probs(b, t, iv, burst):
        with torch.no_grad():
            ot = torch.as_tensor(obs_vec(b, t, iv, burst)).unsqueeze(0)
            dist = pol.get_distribution(ot).distribution
            cat = dist[0] if isinstance(dist, (list, tuple)) else dist
            return cat.probs.detach().cpu().numpy().reshape(-1)

    probes = [(b, t, iv) for b in np.linspace(0.02, 0.98, 13) for t in (2, 6, 10, 14, 18) for iv in (0, 1, 2)]
    tvs = [0.5 * float(np.abs(tree_probs(b, t, iv, False) - tree_probs(b, t, iv, True)).sum()) for b, t, iv in probes]
    tv_burst_mean, tv_burst_max = round(float(np.mean(tvs)), 6), round(float(np.max(tvs)), 6)

    # ---- dose: P(PROVIDE) vs belief at a mid context, monotonicity ----
    dose = {round(float(b), 2): round(float(tree_probs(b, 8, 0, False)[0]), 3) for b in np.linspace(0.02, 0.98, 13)}
    dv = list(dose.values()); dose_monotone = all(dv[i + 1] <= dv[i] + 0.02 for i in range(len(dv) - 1))
    thr = next((b for b, p in dose.items() if p < 0.5), None)

    # ---- leaf legibility ----
    leaf_tbl = pol.tree.leaf_report(n_actions=int(np.asarray(env.action_space.nvec).reshape(-1)[0]))

    passed = parity and (tv_burst_max < 1e-6)
    report = {
        "substrate": "Series-G corner polygon; jointly-trained PPO soft-tree actor, burst-EXCLUDED (feat_idx=[belief,inv,time])",
        "total_timesteps": total_timesteps, "smoke": smoke, "tree_depth": TREE_DEPTH, "n_leaves": pol.tree.n_leaves,
        "return": {"mlp_baseline_mean": round(base_mean, 3), "mlp_sem": round(base_sem, 3),
                   "softtree_mean": round(tree_mean, 3), "softtree_sem": round(tree_sem, 3),
                   "gap": gap, "parity_floor": round(base_mean - 0.5 * base_sem, 3), "PARITY": bool(parity),
                   "paired_diff_mean": round(dmean, 3), "paired_diff_sem": round(dsem, 3)},
        "burst_leak_action_distribution": {"tv_mean": tv_burst_mean, "tv_max": tv_burst_max,
                                           "CLOSED_by_construction": bool(tv_burst_max < 1e-6)},
        "dose": dose, "dose_monotone": bool(dose_monotone), "command_threshold_belief": thr,
        "leaf_table": leaf_tbl,
        "M6_PASS": bool(passed),
        "verdict": (
            "PASS — the jointly-trained belief-only soft-tree head reaches MLP return parity AND has TV=0 under "
            "burst-flip: the C-1 residual leak is now closed IN THE ACTION DISTRIBUTION (structural, not just in "
            "return). A legible tree IS the policy; commands are leaf edits; the raw observable has zero authority."
            if passed else
            "PARTIAL/FAIL — see return parity and TV. If parity fails, the jointly-trained tree costs return vs the "
            "MLP (the B5 risk at the head level); if TV>0 something reads burst (should be impossible by construction)."),
        "caveats": ["K=2 polygon, deterministic + sampled dist both from the same burst-blind head; teeth only where "
                    "VoI>0; critic sees full obs (training-only, does not affect the deployed action distribution)."],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== M6 soft-tree head ===")
    print(f"timesteps={total_timesteps} smoke={smoke} leaves={pol.tree.n_leaves}")
    print(f"return: MLP {base_mean:.3f}(SEM {base_sem:.3f}) | soft-tree {tree_mean:.3f}(SEM {tree_sem:.3f}) | gap {gap:+.3f} | PARITY={parity}")
    print(f"paired diff {dmean:+.3f} (SEM {dsem:.3f})")
    print(f"burst-leak in ACTION DIST: TV_mean={tv_burst_mean} TV_max={tv_burst_max} -> CLOSED={tv_burst_max < 1e-6}")
    print(f"dose monotone={dose_monotone} threshold_belief={thr}")
    print(f"dose={dose}")
    print(f"\nM6_PASS={passed}\n{report['verdict']}")
    print("wrote", OUT.name, "| model", MODEL_OUT.name)


if __name__ == "__main__":
    main()
