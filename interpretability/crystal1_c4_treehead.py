"""C-4 (pilot) — STORY-TREE AS THE POLICY HEAD: can a <=K-leaf tree over the belief BE the policy at return parity?

Milestone C-4 (the pivot) of the CRYSTAL-1 controllability plan. B5 falsified reward-shaping-for-legibility; the
plan therefore rests on a STRUCTURAL mechanism: make the small decision tree over the belief the ACTUAL action head,
so a command is a diffable leaf edit — *if* that costs no return. This pilot tests the core hypothesis cheaply by
DISTILL-AND-ROLLOUT (fit a leaf-budgeted tree on the frozen corner PPO's state->action, then roll the TREE out in the
env as the policy) instead of a full jointly-trained differentiable head.

The decisive comparison ties back to C-1: C-1 found the named belief write is causal but leaks residually through the
raw `burst` observable. So the key question is whether a **belief-only** (or belief+book, NO burst) small tree head
reaches the MLP's return — which would CLOSE the C-1 leakage (proves burst was not needed) at full legibility.

Design: feature sets = subsets of the 4-d obs [belief, time, inv, burst]:
    belief_only=[0] · belief_book=[0,2] · belief_book_time=[0,1,2] · all_incl_burst=[0,1,2,3]
x leaf budgets K in {2,3,4,6,8}. Baseline = the MLP corner PPO. All evaluated on the SAME N held-out episodes.
GATE (pre-registered): a tree head reaches PARITY iff its mean return >= MLP_mean - 0.5*SEM_MLP. The milestone's
structural-legibility claim PASSES iff SOME belief-only or belief+book (burst-free) tree with <=K<=8 leaves reaches
parity. Also report the winning tree's belief->action dose profile (monotone?).
Run: python interpretability/crystal1_c4_treehead.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402

OUT = HERE / "crystal1_c4_treehead_report.json"
FEATSETS = {"belief_only": [0], "belief_book": [0, 2], "belief_book_time": [0, 1, 2], "all_incl_burst": [0, 1, 2, 3]}
LEAVES = [2, 3, 4, 6, 8]
N_EVAL = 300


def rollout_return(env, policy_fn, seeds):
    """mean/steps of episodic return under an action policy_fn(obs)->int."""
    rets = []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False; R = 0.0
        while not done:
            a = policy_fn(obs)
            obs, r, term, trunc, _ = env.step([a]); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    model = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")
    env = MultiAssetRegimePOMDP(n_assets=1, seed=0)
    seeds = [20_000 + i for i in range(N_EVAL)]

    def mlp_act(obs):
        return int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])

    # ---- distillation set: natural rollout obs -> MLP action ----
    OBS, ACT = [], []
    for epi in range(200):
        obs, _ = env.reset(seed=40_000 + epi); done = False
        while not done:
            a = mlp_act(obs)
            OBS.append(np.asarray(obs, float).copy()); ACT.append(a)
            obs, r, term, trunc, _ = env.step([a]); done = term or trunc
    OBS = np.array(OBS); ACT = np.array(ACT, int)

    # ---- baseline MLP return ----
    base = rollout_return(env, mlp_act, seeds)
    base_mean = float(base.mean()); base_sem = float(base.std(ddof=1) / np.sqrt(len(base)))
    parity_floor = base_mean - 0.5 * base_sem

    results = []
    best_bf = None  # best burst-free config reaching parity
    for fname, idx in FEATSETS.items():
        for K in LEAVES:
            tree = DecisionTreeClassifier(max_leaf_nodes=K, min_samples_leaf=5, random_state=0)
            tree.fit(OBS[:, idx], ACT)
            train_acc = float((tree.predict(OBS[:, idx]) == ACT).mean())
            tr = rollout_return(env, lambda obs, tr=tree, ix=idx: int(tr.predict(np.asarray(obs, float)[ix].reshape(1, -1))[0]), seeds)
            tr_mean = float(tr.mean()); gap = round(tr_mean - base_mean, 3)
            parity = tr_mean >= parity_floor
            n_leaves = int(tree.get_n_leaves())
            row = {"features": fname, "max_leaves": K, "n_leaves": n_leaves, "distill_acc": round(train_acc, 3),
                   "return_mean": round(tr_mean, 3), "gap_vs_mlp": gap, "parity": bool(parity)}
            results.append(row)
            burst_free = 3 not in idx
            if burst_free and parity and (best_bf is None or tr_mean > best_bf["return_mean_raw"]):
                best_bf = {**row, "return_mean_raw": tr_mean, "idx": idx, "tree": tree}

    # ---- structural-legibility verdict ----
    passed = best_bf is not None
    dose = None; tree_text = None
    if passed:
        tree = best_bf["tree"]; idx = best_bf["idx"]
        # belief->action dose profile at a mid context (fill non-belief features at typical values)
        prof = {}
        for b in np.linspace(0.02, 0.98, 13):
            feat = []
            for j in idx:
                feat.append(2 * b - 1 if j == 0 else (-0.4 if j == 1 else (2 * (0 / env.m.I_max) - 1 if j == 2 else -1.0)))
            prof[round(float(b), 2)] = int(tree.predict(np.array(feat, float).reshape(1, -1))[0])
        dose = prof
        # monotone: P(PROVIDE=0) should be non-increasing in belief (provide less as toxic-belief rises)
        vals = [1 if prof[b] == 0 else 0 for b in sorted(prof)]
        dose_monotone = all(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))
        names = [["belief", "time", "inv", "burst"][j] for j in idx]
        tree_text = export_text(tree, feature_names=names).replace("\n", " | ")[:600]
        best_bf = {k: v for k, v in best_bf.items() if k not in ("tree", "idx", "return_mean_raw")}
    else:
        dose_monotone = None

    report = {
        "substrate": "src/series_g/corner_ppo_n1.zip; distill-and-rollout pilot (tree AS head, not surrogate)",
        "n_eval_episodes": N_EVAL,
        "baseline_mlp": {"return_mean": round(base_mean, 3), "sem": round(base_sem, 3), "parity_floor": round(parity_floor, 3)},
        "grid": results,
        "best_burst_free_parity": best_bf,
        "winning_tree_belief_dose": dose,
        "winning_tree_dose_monotone": dose_monotone,
        "winning_tree": tree_text,
        "STRUCTURAL_LEGIBILITY_PASS": bool(passed),
        "verdict": (
            "PASS — a small BURST-FREE tree head reaches MLP return parity, so the story tree can BE the policy head "
            "at no return cost. This delivers the B5-mandated STRUCTURAL legibility AND closes the C-1 residual burst "
            "leakage (a belief-based head needs no raw observable). Commands are now diffable leaf edits."
            if passed else
            "FAIL — no burst-free tree head reaches parity within the pre-registered band. Structural legibility as a "
            "policy HEAD costs return here (the B5 risk materializes at the head level); fall back to tree-as-certified-"
            "surrogate + belief-only leakage closure. Report the smallest gap and whether burst was required."),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== C-4 pilot — story-tree as policy head ===")
    print(f"baseline MLP return={base_mean:.3f} (SEM {base_sem:.3f}); parity_floor={parity_floor:.3f}")
    print(f"{'features':18s} {'K':>2s} {'leaves':>6s} {'dacc':>5s} {'return':>8s} {'gap':>7s} parity")
    for r in results:
        print(f"{r['features']:18s} {r['max_leaves']:2d} {r['n_leaves']:6d} {r['distill_acc']:5.2f} "
              f"{r['return_mean']:8.3f} {r['gap_vs_mlp']:+7.3f} {r['parity']}")
    print(f"\nSTRUCTURAL LEGIBILITY PASS={passed}")
    if passed:
        print(f"best burst-free parity: {best_bf}")
        print(f"belief->action dose: {dose}  monotone={dose_monotone}")
    print(report["verdict"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
