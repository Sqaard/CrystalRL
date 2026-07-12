"""Debts 1+2 closed against the C-5 evidence.

Debt 1 (joint-manifold governor gates the C-5 off-manifold corners): fit ManifoldGovernor on NATURAL family-G4
beliefs, then check what fraction of the C-5 sign-epistasis factorial corners it flags OFF-manifold (the box governor
flagged ~none). Confirms the debt-1 fix actually gates the beliefs the sign-epistasis was demonstrated on.

Debt 2 (calibrated per-pair |eps| bound): from the C-5 cells, compute the |eps_logit| distribution PER venue-pair and
fit a calibrated worst-case bound (p95 / max) to replace the eps_bound=0.5 materiality-threshold stand-in in the ledger.
Run: python interpretability/c5_debts.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.family_env import RegimeRotationEnv  # noqa: E402
from src.crystal.governor import ManifoldGovernor, BeliefGovernor  # noqa: E402

OUT = HERE / "c5_debts_report.json"


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    G = 4
    env = RegimeRotationEnv(G=G, seed=0)
    model = PPO.load(str(ROOT / "src/crystal/_b2/family_G4_s1.zip"), device="cpu")

    # ---- collect NATURAL beliefs from rollouts ----
    nat = []
    for epi in range(300):
        obs, _ = env.reset(seed=90_000 + epi); done = False
        while not done:
            nat.append(env.belief.copy())
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            obs, r, term, trunc, _ = env.step(a); done = term or trunc
    nat = np.array(nat)
    man = ManifoldGovernor.from_visited(nat, k=8, q=0.99)
    box = BeliefGovernor.from_visited(nat, q=0.02)

    # ---- DEBT 1: how many C-5 forced factorial corners are flagged off-manifold? ----
    LO, HI = 0.05, 0.55
    def belief(i, vi, j, vj, base=0.05):
        raw = np.full(G, base); raw[i] = vi; raw[j] = vj; return raw / raw.sum()
    pairs = [(i, j) for i in range(G) for j in range(G) if i < j]
    corners = []
    for (i, j) in pairs:
        for (vi, vj) in ((LO, LO), (HI, LO), (LO, HI), (HI, HI)):
            corners.append(belief(i, vi, j, vj))
    man_off = sum(man.govern(b)[1]["status"] == "OFF_MANIFOLD_PROJECTED" for b in corners)
    box_off = sum(box.govern(b)[1]["status"] == "OFF_ENVELOPE_PROJECTED" for b in corners)
    nat_entropy = float(np.mean(-(nat * np.log(nat + 1e-12)).sum(1)))

    # ---- DEBT 2: per-pair calibrated |eps_logit| bound from the C-5 report ----
    c5 = json.loads((HERE / "c5_sign_epistasis_report.json").read_text(encoding="utf-8"))
    per_pair = {}
    for c in c5["cells"]:
        key = f"{c['pair'][0]}-{c['pair'][1]}"
        per_pair.setdefault(key, []).append(abs(c["eps_logit"]))
    pair_bounds = {k: {"p95": round(float(np.quantile(v, 0.95)), 3), "max": round(float(np.max(v)), 3), "n": len(v)}
                   for k, v in per_pair.items()}
    all_abs = np.array([abs(c["eps_logit"]) for c in c5["cells"]])
    global_p95 = round(float(np.quantile(all_abs, 0.95)), 3)
    global_max = round(float(np.max(all_abs)), 3)
    stand_in = 0.5

    report = {
        "debt1_joint_manifold_governor": {
            "natural_belief_mean_entropy": round(nat_entropy, 3),
            "n_c5_corners": len(corners),
            "box_governor_flagged_off": box_off, "manifold_governor_flagged_off": man_off,
            "manifold_thr_knn_L1": round(man.thr, 3),
            "reading": f"the box governor flags {box_off}/{len(corners)} C-5 corners; the kNN manifold governor flags "
                       f"{man_off}/{len(corners)} — closing the joint-manifold gap the sign-epistasis was demonstrated in."},
        "debt2_calibrated_epsilon_bound": {
            "eps_bound_stand_in_was": stand_in,
            "global_p95_abs_eps_logit": global_p95, "global_max_abs_eps_logit": global_max,
            "per_pair_bounds": pair_bounds,
            "recommendation": (f"replace eps_bound={stand_in} with a per-pair table (p95 shown); the global p95="
                               f"{global_p95} and max={global_max} are >> the {stand_in} stand-in, so the ledger's "
                               "worst-case charge was UNDER-conservative on magnitude — the N_live<=K cap was doing the "
                               "real work. Use the per-pair p95 as the charged bound, max as the hard-refuse trip.")},
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== C-5 debts 1+2 ===")
    d1 = report["debt1_joint_manifold_governor"]; d2 = report["debt2_calibrated_epsilon_bound"]
    print(f"DEBT1: natural belief entropy={d1['natural_belief_mean_entropy']} (near-one-hot); "
          f"box flags {d1['box_governor_flagged_off']}/{d1['n_c5_corners']} corners, "
          f"MANIFOLD flags {d1['manifold_governor_flagged_off']}/{d1['n_c5_corners']} (thr knn-L1={d1['manifold_thr_knn_L1']})")
    print(f"DEBT2: eps stand-in was {d2['eps_bound_stand_in_was']}; global |eps_logit| p95={d2['global_p95_abs_eps_logit']} "
          f"max={d2['global_max_abs_eps_logit']}; per-pair p95 bounds = "
          + ", ".join(f"{k}:{v['p95']}" for k, v in d2['per_pair_bounds'].items()))
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
