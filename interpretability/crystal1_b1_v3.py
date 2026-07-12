"""CRYSTAL-1 B1 v3 (4th dose-response point) — the uniqueness-tracks-belief-fidelity test.

B1 v1 finding: with an L1 filter trained on 320 episodes, the agent keeps return (−4%) and all axes but LOSES
explanation uniqueness (state-aware Rashomon 0.02→0.40, stable across seeds). Hypothesis: uniqueness tracks
BELIEF FIDELITY — a better filter (5x SSL data) should re-crisp the explanation. If confirmed, L1 quality is
an INTERPRETABILITY lever, not just a competence lever (a design law for the blueprint).

Run: python interpretability/crystal1_b1_v2.py
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
import series_g_corner_test as sct  # noqa: E402
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from certified_battery_v2 import n7_grouped  # noqa: E402
from crystal1_b1 import LearnedBeliefEnv  # noqa: E402
from src.crystal.belief_filter import train_filter  # noqa: E402
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402
from src.series_g.regime_pomdp import PRIMARY_ENRICHED, RegimePOMDP  # noqa: E402

MODEL = ROOT / "src/crystal/crystal1_b1v3_ppo.zip"
OUT = HERE / "crystal1_b1_v3_report.json"
N_SSL = 8000
EPOCHS = 800


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    # ---- L1 v2: 5x SSL data ----
    env0 = MultiAssetRegimePOMDP(n_assets=1, seed=0)
    seqs = np.empty((N_SSL, env0.T), dtype=int)
    for e in range(N_SSL):
        env0.reset(seed=100_000 + e)
        for t in range(env0.T):
            env0.step([1])
            seqs[e, t] = int(env0.last_obs[0])
    f = train_filter(seqs, K=2, A_obs=2, epochs=EPOCHS, seed=0, verbose=False)
    T_l, E_l, p0_l = f.numpy_params()
    tox = int(np.argmax(E_l[:, 1]))
    m = RegimePOMDP(**PRIMARY_ENRICHED)
    # filter fidelity vs analytic (fresh streams)
    errs = []
    for e in range(120):
        env0.reset(seed=900_000 + e)
        b_a = m.prior_toxic; b_l = p0_l.copy()
        for t in range(env0.T):
            env0.step([1]); o = int(env0.last_obs[0])
            b_a = m.update(m.predict(b_a), o)
            bp = b_l @ T_l; j = bp * E_l[:, o]; b_l = j / max(j.sum(), 1e-12)
            errs.append(abs(b_a - b_l[tox]))
    mae = float(np.mean(errs))
    print(f"[B1v3] L1: p_stay={np.diag(T_l).round(3)} E_burst={E_l[:,1].round(3)} | belief MAE vs analytic = {mae:.4f} (v1: 0.024)")

    # ---- L2 v2 ----
    env = Monitor(LearnedBeliefEnv(T_l, E_l, p0_l, tox, seed=0))
    if MODEL.exists():
        model = PPO.load(str(MODEL), env=env, device="cpu"); print("[B1v3] loaded PPO")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=0, device="cpu", verbose=0)
        print("[B1v3] training PPO (120k)...")
        model.learn(total_timesteps=120_000); model.save(str(MODEL))

    # ---- rollouts + battery ----
    def rollout(env_, n=160, seed0=10_000, record=True):
        rows, rets = [], []
        for epi in range(n):
            obs, _ = env_.reset(seed=seed0 + epi)
            done, tot = False, 0.0
            while not done:
                a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
                if record:
                    rows.append({"ep": epi, "t": env_.inner.t, "belief": env_.belief_learned,
                                 "inv": int(env_.inner.inv[0]), "action": a})
                obs, r, term, trunc, _ = env_.step([a]); tot += r; done = term or trunc
            rets.append(tot)
        return (pd.DataFrame(rows) if record else None), float(np.mean(rets))

    d, ret = rollout(LearnedBeliefEnv(T_l, E_l, p0_l, tox, seed=1))
    _, ret_noise = rollout(LearnedBeliefEnv(T_l, E_l, p0_l, tox, seed=1, belief_mode="noise"), record=False)
    act = d["action"].to_numpy(int); ep = d["ep"].to_numpy(int)
    bel = d["belief"].to_numpy(float); inv = d["inv"].to_numpy(float)
    l0 = behavioral_complexity_dynamic(act, kind="discrete", dts=(1, 2), n_null=300, n_boot=300, seed=0)
    X = d[["belief", "inv", "t"]].to_numpy(float)
    cut = int(len(act) * 0.6)
    st = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X[:cut], act[:cut])
    y_react = round(float(balanced_accuracy_score(act[cut:], st.predict(X[cut:]))), 3)
    y_auto = sct.autoreg_sim(act, ep)
    rsh_seeds = []
    for s in (1, 2, 3):
        sct._rng = np.random.default_rng(s)
        rsh_seeds.append(sct.rashomon(inv, bel, N=200)["ratio_e25"])
    _, b_pct, b_verdict = n7_grouped(bel, ep)
    ablation_drop = ret - ret_noise

    gates = {"high_complexity": bool(min(l0["h_mu_range"]) > 0.6), "structured": l0["structure_present"],
             "reactive>>autoregressive": bool(y_react > y_auto + 0.1),
             "rashomon_crisp(state-aware, median of 3 seeds)": bool(float(np.median(rsh_seeds)) <= 0.3),
             "belief_N7_asym": bool(b_pct >= 95), "HC1_ablation_hurts": bool(ablation_drop > 0.25 * abs(ret))}
    report = {"L1_v2": {"n_ssl": N_SSL, "belief_MAE_vs_analytic": round(mae, 4), "v1_MAE": 0.024},
              "returns": {"CRYSTAL1_v3": round(ret, 2), "noise_belief": round(ret_noise, 2), "v1": 6.84, "privileged": 7.11},
              "battery": {"x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
                           "y_reactive": y_react, "y_autoregressive": y_auto,
                           "rashomon_ratio_3seeds": rsh_seeds, "v1_rashomon_3seeds": [0.375, 0.395, 0.43],
                           "privileged_rashomon_3seeds": [0.02, 0.02, 0.105],
                           "belief_N7": {"pct": b_pct, "verdict": b_verdict}},
              "gates": gates, "B1v3_PASS": all(gates.values()),
              "uniqueness_tracks_fidelity": None}
    v1_med, v2_med = 0.395, float(np.median(rsh_seeds))
    report["uniqueness_tracks_fidelity"] = bool(mae < 0.024 and v2_med < v1_med - 0.1)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
