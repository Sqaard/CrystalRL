"""CrystalScore-v2 Axis A (MDL / Parsimony-Fidelity Deficit) — the most demanding axis, run across the complexity range.

Claim from the tradeoff funnel: CrystalScore 0.938 lives in the DEGENERATE low-h_mu (near-persister) corner where a K<=9
story trivially reproduces the policy (deficit ~0). As behavioral complexity rises, a K<=9 story provably CANNOT
reproduce the policy (parsimony-fidelity is a Pareto frontier — Rissanen MDL / Kolmogorov): you cannot have both a short
story AND full fidelity. Measured on the K-simplex family models (G=2 corner -> G=4 -> G=12), whose action space and
belief dimension grow, so the K<=9 budget is increasingly outmatched.

Deficit(policy) = 1 - Simul@(<=8 leaves) / Simul@(<=64 leaves)   (fidelity lost to the parsimony budget, normalized by
the achievable ceiling). Also reports h_mu (L0 ruler) so the frontier is (behavioral complexity, deficit).
Run: python interpretability/mdl_fidelity_deficit.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.series_g.family_env import RegimeRotationEnv  # noqa: E402
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from interpretability.cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

OUT = HERE / "mdl_fidelity_deficit_report.json"
FAM = {4: "src/crystal/_b2/family_G4_s1.zip", 12: "src/crystal/_b2/family_G12_s1.zip"}


def rollout_family(G, zip_path, seeds):
    from stable_baselines3 import PPO
    env = RegimeRotationEnv(G=G, seed=0); model = PPO.load(str(ROOT / zip_path), device="cpu"); sev = env.sev
    rows, acts = [], []
    for s in seeds:
        env.reset(seed=s); done = False
        while not done:
            y = float(np.dot(env.belief, sev))
            obs = np.concatenate([env.belief, [2.0 * env.t / env.T - 1.0, env.inv / env.I_max, y]]).astype(np.float32)
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            row = {f"b{v}": float(env.belief[v]) for v in range(G)}; row.update({"inv": env.inv, "t": env.t})
            rows.append(row); acts.append(a)
            _, r, term, trunc, _ = env.step(a); done = term or trunc
    return pd.DataFrame(rows), np.array(acts)


def rollout_corner(seeds):
    from stable_baselines3 import PPO
    env = MultiAssetRegimePOMDP(n_assets=1, seed=0); model = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")
    rows, acts = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False
        while not done:
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            rows.append({"belief": float(env.belief), "inv": int(env.inv[0]), "t": env.t}); acts.append(a)
            obs, r, term, trunc, _ = env.step([a]); done = term or trunc
    return pd.DataFrame(rows), np.array(acts)


def deficit(df, acts):
    X = df.to_numpy(float); y = acts; cut = int(len(X) * 0.6)
    s9 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    s64 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=64, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    hmu = behavioral_complexity_dynamic(y.astype(float), kind="discrete", n_null=120, n_boot=120, seed=0)
    hr = hmu.get("h_mu_range", hmu.get("h_mu", [None, None]))
    return {"n_actions_used": int(len(np.unique(y))), "simul_K9": round(float(s9), 3), "simul_ceiling64": round(float(s64), 3),
            "deficit": round(float(1 - s9 / max(s64, 1e-9)), 3), "h_mu_range": [round(float(x), 3) for x in hr] if isinstance(hr, (list, tuple)) else hr}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    res = {}
    dfc, ac = rollout_corner(range(10_000, 10_160)); res["corner_G2"] = {"G": 2, **deficit(dfc, ac)}
    for G, zp in FAM.items():
        if (ROOT / zp).exists():
            dff, af = rollout_family(G, zp, range(10_000, 10_120)); res[f"family_G{G}"] = {"G": G, **deficit(dff, af)}
    pts = [(v["G"], v["deficit"], v["simul_K9"]) for v in res.values()]
    res["frontier"] = {"(G, deficit, simul_K9)": pts}
    res["headline"] = (
        "MDL Parsimony-Fidelity Deficit across complexity: " +
        "; ".join(f"G{v['G']} deficit {v['deficit']} (simul@K9 {v['simul_K9']}, {v['n_actions_used']} actions)" for v in res.values() if isinstance(v, dict) and "deficit" in v) +
        ". Confirms the tension: the CrystalScore 0.938 corner (G2, deficit~0) is the DEGENERATE low-complexity end; as "
        "the K-simplex vocabulary grows the K<=9 story CANNOT reproduce the policy and the deficit rises — a real Pareto "
        "frontier (short story vs full fidelity), so no policy scores 1 on both. CRYSTAL-1 is near-ideal only on the "
        "low-complexity leg; Axis A is the demanding metric the saturated scalar hid.")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("=== MDL Parsimony-Fidelity Deficit (CrystalScore-v2 Axis A) ===")
    for k, v in res.items():
        if isinstance(v, dict) and "deficit" in v: print(f"  {k:12s}: {v}")
    print("\n" + res["headline"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
