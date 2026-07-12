"""R / WH5 — ONTOGENY: does legibility CRYSTALLIZE (jump) or grow smoothly during training?

Retrains the corner PPO with dense checkpoints (every 4096 steps to ~61k — exactly the 0→50k window the
other agent's checkpoint audit flagged as unobserved) and measures, per checkpoint:
  belief_usage   — intervention meter: fraction of envelope probes where forcing belief 0.1→0.9 flips the
                   action (does the policy USE the belief at all?)
  reactive_sim   — depth-4 story fidelity on a natural rollout (legibility)
  autoreg_sim    — own-past predictability (persister-ness)
  h_mu           — bits/action (single config, cheap)
  mean_return    — competence
Verdict: sharp-vs-smooth (max single-step jump vs the median step), with the honest mirage guard (a smooth
curve under a nonlinear readout can fake a jump — we report the raw curves).

Run: python interpretability/ontogeny_corner.py
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
from cross_policy_crystal import _entropy_rate_complexity  # noqa: E402
from series_g_corner_test import autoreg_sim  # noqa: E402
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402

CKPT_DIR = ROOT / "src/series_g/_ontogeny_ckpts"
OUT = HERE / "ontogeny_corner_report.json"
SAVE_EVERY = 4096
TOTAL = 61_440


def measure(model, env):
    rows = []
    for epi in range(60):
        obs, _ = env.reset(seed=40_000 + epi)
        done, tot = False, 0.0
        while not done:
            a = int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])
            rows.append({"ep": epi, "t": env.t, "belief": float(env.belief), "inv": int(env.inv[0]), "action": a})
            obs, r, term, trunc, _ = env.step([a]); tot += r; done = term or trunc
        rows[-1]["ret"] = tot
    d = pd.DataFrame(rows)
    act = d["action"].to_numpy(int); ep = d["ep"].to_numpy(int)
    X = d[["belief", "inv", "t"]].to_numpy(float)
    cut = int(len(act) * 0.6)
    if len(np.unique(act[cut:])) >= 2 and len(np.unique(act[:cut])) >= 2:
        st = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X[:cut], act[:cut])
        react = round(float(balanced_accuracy_score(act[cut:], st.predict(X[cut:]))), 3)
    else:
        react = float("nan")                                   # degenerate (single-action) phase
    auto = autoreg_sim(act, ep)
    h = _entropy_rate_complexity(act, int(act.max()) + 1, 6, 5.0, 0.15, with_cmu=False)["h_mu"]
    # belief-usage: forced-belief action-flip rate on the envelope
    def ov(b, t, iv, burst):
        return np.array([2 * b - 1, 2 * t / env.T - 1, 2 * (iv / env.m.I_max) - 1, 1.0 if burst else -1.0], dtype=np.float32)
    flips, n = 0, 0
    for t in (2, 6, 10, 14):
        for iv in (0, 1, 2):
            for burst in (False, True):
                a_lo = int(np.asarray(model.predict(ov(0.1, t, iv, burst), deterministic=True)[0]).reshape(-1)[0])
                a_hi = int(np.asarray(model.predict(ov(0.9, t, iv, burst), deterministic=True)[0]).reshape(-1)[0])
                flips += int(a_lo != a_hi); n += 1
    return {"reactive_sim": react, "autoreg_sim": auto, "h_mu": round(float(h), 3),
            "belief_usage": round(flips / n, 3), "mean_return": round(float(d.groupby("ep")["ret"].last().mean()), 2)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    env = Monitor(MultiAssetRegimePOMDP(n_assets=1, seed=7))
    if not list(CKPT_DIR.glob("onto_*_steps.zip")):
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=7, device="cpu", verbose=0)
        cb = CheckpointCallback(save_freq=SAVE_EVERY, save_path=str(CKPT_DIR), name_prefix="onto")
        print(f"[onto] training with dense checkpoints (every {SAVE_EVERY} to {TOTAL})...")
        model.learn(total_timesteps=TOTAL, callback=cb)
    envm = MultiAssetRegimePOMDP(n_assets=1, seed=8)
    rows = []
    for f in sorted(CKPT_DIR.glob("onto_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1])):
        step = int(f.stem.split("_")[1])
        m = PPO.load(str(f), device="cpu")
        r = {"step": step, **measure(m, envm)}
        rows.append(r)
        print(f"[onto] {step:>6}: react={r['reactive_sim']} auto={r['autoreg_sim']} "
              f"belief_usage={r['belief_usage']} h_mu={r['h_mu']} ret={r['mean_return']}")
    bu = [r["belief_usage"] for r in rows]
    jumps = np.abs(np.diff(bu))
    verdict = ("SHARP (crystallization-like): one step carries most of the belief-usage rise"
               if len(jumps) and jumps.max() > 0.35 and jumps.max() > 3 * np.median(jumps) else
               "SMOOTH: belief-usage/legibility grow gradually — no phase-transition jump in this window")
    rep = {"rows": rows, "max_jump_belief_usage": round(float(jumps.max()), 3) if len(jumps) else None,
           "median_step": round(float(np.median(jumps)), 3) if len(jumps) else None, "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
