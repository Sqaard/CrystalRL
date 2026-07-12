"""CRYSTAL-1 B2 — multi-seed replication harness (kills the n=1 caveat).

Twelve training jobs, one sequential background process, incremental JSONL (partial progress survives).
Eval protocol is FIXED (env seed 1, episode seeds 10_000+); only the TRAINING pipeline seeds vary
(SSL data offset, filter torch seed, PPO seed, training env seed) — replicating the pipeline, not the eval.

Claims under replication:
  C1 corner profile with LEARNED belief (crystal v3 × seeds 1,2,3; + the existing seed-0 run) — 6-gate pass.
  C2 privileged reference profile (× seeds 1,2,3; + existing seed-0).
  C3 uniqueness-law ENDPOINTS: Rashomon(v1, 320-episode filter) > Rashomon(v3, 8000) in every seed.
  C4 C*≈K learned bend: budget gap(G=12) > gap(G=4) in every seed pair (+ existing seed-0 sweep).
  C5 price of learned beliefs ≈ 0: return(crystal v3) ≈ return(privileged) per seed.

Run: python interpretability/b2_multiseed.py    (~60-90 min; writes b2_multiseed_results.jsonl + report)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import series_g_corner_test as sct  # noqa: E402
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from certified_battery_v2 import n7_grouped  # noqa: E402
from crystal1_b1 import LearnedBeliefEnv  # noqa: E402
from src.crystal.belief_filter import train_filter  # noqa: E402
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402
from src.series_g.family_env import RegimeRotationEnv  # noqa: E402
from src.series_g.regime_pomdp import PRIMARY_ENRICHED, RegimePOMDP  # noqa: E402

MDIR = ROOT / "src/crystal/_b2"; MDIR.mkdir(parents=True, exist_ok=True)
JL = HERE / "b2_multiseed_results.jsonl"
OUT = HERE / "b2_multiseed_report.json"

JOBS = ([{"kind": "crystal", "fid": "v3", "seed": s} for s in (1, 2, 3)] +
        [{"kind": "privileged", "seed": s} for s in (1, 2, 3)] +
        [{"kind": "crystal", "fid": "v1", "seed": s} for s in (1, 2)] +
        [{"kind": "family", "G": G, "seed": s} for G in (4, 12) for s in (1, 2)])
N_SSL = {"v1": 320, "v3": 8000}


def battery_lite(d, model=None, env_ctor=None):
    """Shared corner battery: L0, react/auto, Rashomon (2 rng-seeds x N=150, median), belief-N7, HC-1."""
    act = d["action"].to_numpy(int); ep = d["ep"].to_numpy(int)
    bel = d["belief"].to_numpy(float); inv = d["inv"].to_numpy(float)
    l0 = behavioral_complexity_dynamic(act, kind="discrete", dts=(1, 2), n_null=200, n_boot=200, seed=0)
    X = d[["belief", "inv", "t"]].to_numpy(float); cut = int(len(act) * 0.6)
    st = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X[:cut], act[:cut])
    react = round(float(balanced_accuracy_score(act[cut:], st.predict(X[cut:]))), 3)
    auto = sct.autoreg_sim(act, ep)
    rsh = []
    for s in (1, 2):
        sct._rng = np.random.default_rng(s)
        rsh.append(sct.rashomon(inv, bel, N=150)["ratio_e25"])
    _, b_pct, _ = n7_grouped(bel, ep, reps=300)
    return {"x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
            "struct_ok": l0["structure_present"], "react": react, "auto": auto,
            "rashomon_med": round(float(np.median(rsh)), 3), "rashomon": rsh, "belief_n7_pct": round(b_pct, 1)}


def rollout_corner(model, env, n=160, seed0=10_000, learned=False):
    rows, rets = [], []
    for epi in range(n):
        obs, _ = env.reset(seed=seed0 + epi)
        done, tot = False, 0.0
        while not done:
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            inner = env.inner if learned else env
            rows.append({"ep": epi, "t": inner.t, "belief": (env.belief_learned if learned else float(inner.belief)),
                         "inv": int(inner.inv[0]), "action": a})
            obs, r, term, trunc, _ = env.step([a]); tot += r; done = term or trunc
    # NB: per-episode return accumulated inside the loop tail
        rets.append(tot)
    return pd.DataFrame(rows), float(np.mean(rets))


def job_crystal(fid, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    n_ssl = N_SSL[fid]
    env0 = MultiAssetRegimePOMDP(n_assets=1, seed=seed)
    seqs = np.empty((n_ssl, env0.T), dtype=int)
    base = 100_000 + 50_000 * seed
    for e in range(n_ssl):
        env0.reset(seed=base + e)
        for t in range(env0.T):
            env0.step([1]); seqs[e, t] = int(env0.last_obs[0])
    f = train_filter(seqs, K=2, A_obs=2, epochs=(300 if fid == "v1" else 800), seed=seed, verbose=False)
    T_l, E_l, p0 = f.numpy_params(); tox = int(np.argmax(E_l[:, 1]))
    # fidelity vs analytic
    m = RegimePOMDP(**PRIMARY_ENRICHED); errs = []
    for e in range(100):
        env0.reset(seed=900_000 + e); b_a = m.prior_toxic; b_l = p0.copy()
        for t in range(env0.T):
            env0.step([1]); o = int(env0.last_obs[0])
            b_a = m.update(m.predict(b_a), o)
            bp = b_l @ T_l; j = bp * E_l[:, o]; b_l = j / max(j.sum(), 1e-12)
            errs.append(abs(b_a - b_l[tox]))
    mae = round(float(np.mean(errs)), 4)
    mp = MDIR / f"crystal_{fid}_s{seed}.zip"
    env = Monitor(LearnedBeliefEnv(T_l, E_l, p0, tox, seed=seed))
    if mp.exists():
        model = PPO.load(str(mp), env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=seed, device="cpu", verbose=0)
        model.learn(total_timesteps=120_000); model.save(str(mp))
    d, ret = rollout_corner(model, LearnedBeliefEnv(T_l, E_l, p0, tox, seed=1), learned=True)
    _, ret_noise = rollout_corner(model, LearnedBeliefEnv(T_l, E_l, p0, tox, seed=1, belief_mode="noise"), learned=True)
    b = battery_lite(d)
    gates = {"hc": bool(min(b["x_h_mu"]) > 0.6), "st": b["struct_ok"], "ra": bool(b["react"] > b["auto"] + 0.1),
             "rc": bool(b["rashomon_med"] <= 0.3), "n7": bool(b["belief_n7_pct"] >= 95),
             "ab": bool((ret - ret_noise) > 0.25 * abs(ret))}
    return {"kind": "crystal", "fid": fid, "seed": seed, "mae": mae, "return": round(ret, 2),
            "return_noise": round(ret_noise, 2), **b, "gates": gates, "pass6": all(gates.values())}


def job_privileged(seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    mp = MDIR / f"priv_s{seed}.zip"
    env = Monitor(MultiAssetRegimePOMDP(n_assets=1, seed=seed))
    if mp.exists():
        model = PPO.load(str(mp), env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=seed, device="cpu", verbose=0)
        model.learn(total_timesteps=120_000); model.save(str(mp))
    d, ret = rollout_corner(model, MultiAssetRegimePOMDP(n_assets=1, seed=1))
    b = battery_lite(d)
    gates = {"hc": bool(min(b["x_h_mu"]) > 0.6), "st": b["struct_ok"], "ra": bool(b["react"] > b["auto"] + 0.1),
             "rc": bool(b["rashomon_med"] <= 0.3), "n7": bool(b["belief_n7_pct"] >= 95)}
    return {"kind": "privileged", "seed": seed, "return": round(ret, 2), **b, "gates": gates,
            "pass5": all(gates.values())}


def job_family(G, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    mp = MDIR / f"family_G{G}_s{seed}.zip"
    env = Monitor(RegimeRotationEnv(G=G, seed=seed))
    if mp.exists():
        model = PPO.load(str(mp), env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[128, 128]), seed=seed, device="cpu", verbose=0)
        model.learn(total_timesteps=100_000); model.save(str(mp))
    envr = RegimeRotationEnv(G=G, seed=3)
    rows, rets = [], []
    for epi in range(150):
        obs, _ = envr.reset(seed=7_000 + epi)
        done, tot = False, 0.0
        while not done:
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            rows.append({"action": a, **{f"b{i}": float(envr.belief[i]) for i in range(G)},
                         "t": envr.t, "inv": envr.inv})
            obs, r, term, trunc, _ = envr.step(a); tot += r; done = term or trunc
        rets.append(tot)
    d = pd.DataFrame(rows)
    feats = [f"b{i}" for i in range(G)] + ["t", "inv"]
    X = d[feats].to_numpy(float); y = d["action"].to_numpy(int); cut = int(len(y) * 0.6)
    def sim(leaves):
        if len(np.unique(y[cut:])) < 2:
            return float("nan")
        c = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(X[:cut], y[:cut])
        return round(float(balanced_accuracy_score(y[cut:], c.predict(X[cut:]))), 3)
    s9, s64 = sim(9), sim(64)
    l0 = behavioral_complexity_dynamic(y, kind="discrete", dts=(1, 2), n_null=200, n_boot=200, seed=0)
    return {"kind": "family", "G": G, "seed": seed, "return": round(float(np.mean(rets)), 2),
            "x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
            "sim_K9": s9, "sim_K64": s64, "gap": round((s64 or 0) - (s9 or 0), 3)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    done = set()
    if JL.exists():
        for line in JL.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["kind"], r.get("fid", r.get("G", "")), r["seed"]))
    for j in JOBS:
        key = (j["kind"], j.get("fid", j.get("G", "")), j["seed"])
        if key in done:
            print(f"[b2] skip {key} (done)"); continue
        t0 = time.time()
        if j["kind"] == "crystal":
            row = job_crystal(j["fid"], j["seed"])
        elif j["kind"] == "privileged":
            row = job_privileged(j["seed"])
        else:
            row = job_family(j["G"], j["seed"])
        row["minutes"] = round((time.time() - t0) / 60, 1)
        with JL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[b2] {key} done in {row['minutes']}m: " + json.dumps({k: v for k, v in row.items() if k in
              ('return', 'react', 'auto', 'rashomon_med', 'pass6', 'pass5', 'gap', 'mae')}))

    # ---------- aggregate ----------
    rows = [json.loads(x) for x in JL.read_text(encoding="utf-8").splitlines()]
    cv3 = [r for r in rows if r["kind"] == "crystal" and r["fid"] == "v3"]
    cv1 = [r for r in rows if r["kind"] == "crystal" and r["fid"] == "v1"]
    pv = [r for r in rows if r["kind"] == "privileged"]
    fam = [r for r in rows if r["kind"] == "family"]
    # existing seed-0 anchors (from earlier runs)
    seed0 = {"cv3_rashomon": 0.145, "cv3_return": 6.90, "priv_return": 7.11, "cv1_rashomon": 0.395,
             "gap_G4": 0.077, "gap_G12": 0.314}
    def ms(vals): return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
    rep = {
        "C1_crystal_v3_profile": {"n": len(cv3), "pass6_rate": f"{sum(r['pass6'] for r in cv3)}/{len(cv3)}",
            "react": ms([r["react"] for r in cv3]), "auto": ms([r["auto"] for r in cv3]),
            "rashomon_med": ms([r["rashomon_med"] for r in cv3]), "return": ms([r["return"] for r in cv3]),
            "per_seed_gates": {r["seed"]: r["gates"] for r in cv3}},
        "C2_privileged_profile": {"n": len(pv), "pass5_rate": f"{sum(r['pass5'] for r in pv)}/{len(pv)}",
            "react": ms([r["react"] for r in pv]), "rashomon_med": ms([r["rashomon_med"] for r in pv]),
            "return": ms([r["return"] for r in pv])},
        "C3_uniqueness_endpoints": {"v1_rashomon": [r["rashomon_med"] for r in cv1] + [seed0["cv1_rashomon"]],
            "v3_rashomon": [r["rashomon_med"] for r in cv3] + [seed0["cv3_rashomon"]],
            "v1_mae": [r["mae"] for r in cv1], "v3_mae": [r["mae"] for r in cv3],
            "all_v1_gt_all_v3": bool(min([r["rashomon_med"] for r in cv1] + [seed0["cv1_rashomon"]]) >
                                     max([r["rashomon_med"] for r in cv3] + [seed0["cv3_rashomon"]]))},
        "C4_bend": {"gap_G4": [r["gap"] for r in fam if r["G"] == 4] + [seed0["gap_G4"]],
                     "gap_G12": [r["gap"] for r in fam if r["G"] == 12] + [seed0["gap_G12"]],
                     "bend_replicates_all_seeds": bool(
                         min([r["gap"] for r in fam if r["G"] == 12] + [seed0["gap_G12"]]) >
                         max([r["gap"] for r in fam if r["G"] == 4] + [seed0["gap_G4"]]))},
        "C5_price_of_learned_beliefs": {"crystal_v3_returns": [r["return"] for r in cv3] + [seed0["cv3_return"]],
                                          "privileged_returns": [r["return"] for r in pv] + [seed0["priv_return"]]},
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
