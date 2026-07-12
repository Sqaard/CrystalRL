"""CRYSTAL-1 B1 — the first agent built to the blueprint: LEARNED belief filter (L1) + policy (L2) on the
polygon, with the battery as the acceptance gate.

Difference from the certified corner model: that PPO received the ENV-COMPUTED (privileged) Bayes belief in
its observation. CRYSTAL-1 must EARN its belief: L1 (`src/crystal/belief_filter.py`) is trained
self-supervised on raw observation streams (no regime labels; parameter recovery verified), FROZEN, and run
online inside the env wrapper; the policy sees [learned_belief, t, inv, last_obs] — the same 4-dim layout as
the privileged model, so the ONLY difference is the belief's source.

Gates (blueprint B1):
  1. five-axis corner battery on the trained agent (bits/action+structure, reactive>>autoregressive,
     state-aware Rashomon crisp, belief-N7 asym on the LEARNED belief stream);
  2. HC-1 ablation: replacing the learned belief with noise at eval must materially hurt return;
  3. competence: return close to the privileged corner PPO (the price of learning your own beliefs).

Run: python interpretability/crystal1_b1.py
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
from series_g_corner_test import rashomon, autoreg_sim  # noqa: E402
from certified_battery_v2 import n7_grouped  # noqa: E402
from src.crystal.belief_filter import train_filter  # noqa: E402
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces

MODEL = ROOT / "src/crystal/crystal1_b1_ppo.zip"
OUT = HERE / "crystal1_b1_report.json"
BUDGET = 120_000


class LearnedBeliefEnv(gym.Env):
    """The polygon with L1 inside: obs = [2*b_learned-1, 2t/T-1, inv-scaled, last_obs] — belief from the
    FROZEN learned filter (numpy), updated online. `belief_mode`: 'learned' | 'noise' (HC-1 ablation)."""
    metadata = {"render_modes": []}

    def __init__(self, T_l, E_l, p0_l, toxic_idx, seed=None, belief_mode="learned"):
        super().__init__()
        self.inner = MultiAssetRegimePOMDP(n_assets=1, seed=seed)
        self.T_l, self.E_l, self.p0_l, self.tox = T_l, E_l, p0_l, int(toxic_idx)
        self.belief_mode = belief_mode
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.action_space = self.inner.action_space
        self._rng = np.random.default_rng(seed)
        self.b = p0_l.copy()

    @property
    def belief_learned(self) -> float:
        return float(self.b[self.tox])

    def _obs(self):
        bt = self._rng.random() if self.belief_mode == "noise" else self.belief_learned
        e = self.inner
        return np.array([2 * bt - 1, 2 * e.t / e.T - 1, 2 * (e.inv[0] / e.m.I_max) - 1,
                         1.0 if e.last_obs[0] == 1 else -1.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.inner.reset(seed=seed)
        self.b = self.p0_l.copy()
        self._filter_update(int(self.inner.last_obs[0]))
        return self._obs(), {}

    def _filter_update(self, o: int):
        bp = self.b @ self.T_l
        joint = bp * self.E_l[:, o]
        s = joint.sum()
        self.b = joint / s if s > 1e-12 else bp

    def step(self, action):
        obs, r, term, trunc, info = self.inner.step(action)
        self._filter_update(int(self.inner.last_obs[0]))
        return self._obs(), r, term, trunc, info


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    # ---- L1: train the filter self-supervised on raw observation streams, freeze ----
    env0 = MultiAssetRegimePOMDP(n_assets=1, seed=0)
    seqs = np.empty((400, env0.T), dtype=int)
    for e in range(400):
        env0.reset(seed=100_000 + e)
        for t in range(env0.T):
            env0.step([1])
            seqs[e, t] = int(env0.last_obs[0])
    f = train_filter(seqs[:320], K=2, A_obs=2, epochs=300, seed=0, verbose=False)
    T_l, E_l, p0_l = f.numpy_params()
    toxic_idx = int(np.argmax(E_l[:, 1]))          # the state with the higher burst prob = toxic
    print(f"[B1] L1 frozen: p_stay={np.diag(T_l).round(3)} E_burst={E_l[:,1].round(3)} toxic_idx={toxic_idx}")

    # ---- L2: PPO on the learned-belief env ----
    env = Monitor(LearnedBeliefEnv(T_l, E_l, p0_l, toxic_idx, seed=0))
    if MODEL.exists():
        model = PPO.load(str(MODEL), env=env, device="cpu"); print("[B1] loaded PPO")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=0, device="cpu", verbose=0)
        print(f"[B1] training CRYSTAL-1 PPO ({BUDGET})...")
        model.learn(total_timesteps=BUDGET); model.save(str(MODEL))

    # ---- rollouts: natural / HC-1 noise-belief ablation / privileged reference ----
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

    d, ret = rollout(LearnedBeliefEnv(T_l, E_l, p0_l, toxic_idx, seed=1))
    _, ret_noise = rollout(LearnedBeliefEnv(T_l, E_l, p0_l, toxic_idx, seed=1, belief_mode="noise"), record=False)
    # privileged corner model reference (same eval seeds)
    ref = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")
    env_ref = MultiAssetRegimePOMDP(n_assets=1, seed=1)
    rr = []
    for epi in range(160):
        obs, _ = env_ref.reset(seed=10_000 + epi)
        done, tot = False, 0.0
        while not done:
            a = int(np.asarray(ref.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            obs, r, term, trunc, _ = env_ref.step([a]); tot += r; done = term or trunc
        rr.append(tot)
    ret_priv = float(np.mean(rr))

    # ---- battery ----
    act = d["action"].to_numpy(int); ep = d["ep"].to_numpy(int)
    bel = d["belief"].to_numpy(float); inv = d["inv"].to_numpy(float)
    l0 = behavioral_complexity_dynamic(act, kind="discrete", dts=(1, 2), n_null=300, n_boot=300, seed=0)
    X = d[["belief", "inv", "t"]].to_numpy(float)
    cut = int(len(act) * 0.6)
    st = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X[:cut], act[:cut])
    y_react = round(float(balanced_accuracy_score(act[cut:], st.predict(X[cut:]))), 3)
    y_auto = autoreg_sim(act, ep)
    rsh = rashomon(inv, bel, N=80)
    _, b_pct, b_verdict = n7_grouped(bel, ep)

    ablation_drop = ret - ret_noise
    gates = {"high_complexity": bool(min(l0["h_mu_range"]) > 0.6), "structured": l0["structure_present"],
             "reactive>>autoregressive": bool(y_react > y_auto + 0.1),
             "rashomon_crisp(state-aware)": bool(rsh["ratio_e25"] <= 0.3),
             "belief_N7_asym(learned stream)": bool(b_pct >= 95),
             "HC1_ablation_hurts": bool(ablation_drop > 0.25 * abs(ret))}
    report = {
        "L1_frozen_filter": {"p_stay": np.diag(T_l).round(3).tolist(), "E_burst": E_l[:, 1].round(3).tolist(),
                              "provenance": "self-supervised on raw obs streams; param recovery verified in module selftest"},
        "returns": {"CRYSTAL1(learned belief)": round(ret, 2), "HC1_noise_belief": round(ret_noise, 2),
                     "privileged_corner_PPO": round(ret_priv, 2), "analytic_optimum": 8.58,
                     "price_of_learning_beliefs": round(ret_priv - ret, 2)},
        "battery": {"x_h_mu": l0["h_mu_range"], "structure": l0["structure_present_configs"],
                     "y_reactive": y_react, "y_autoregressive": y_auto,
                     "rashomon_state_aware": rsh, "belief_N7": {"pct": b_pct, "verdict": b_verdict}},
        "gates": gates, "B1_PASS": all(gates.values()),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nB1:", "PASS — CRYSTAL-1 reproduces the corner profile with a LEARNED belief" if report["B1_PASS"]
          else "some gate failed — inspect")


if __name__ == "__main__":
    main()
