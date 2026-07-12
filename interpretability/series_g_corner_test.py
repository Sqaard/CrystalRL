"""SERIES-G CORNER TEST — train the anchor model, then the FULL acceptance battery.

The frontier's empty corner is "high behavioral-complexity AND interpretable AND genuinely state-reactive."
R6c (persister) and P22 (churner) both fail it. This trains a real PPO policy on the Series-G env and asks
— through the whole battery, not just x/y — whether it OCCUPIES the corner:

  x  bits/action (L0) + phase-shuffle STRUCTURE test   → high h_mu WITH structure (not churn)
  y  simulatability, REACTIVE (obs/state→action) vs AUTOREGRESSIVE (own-past)  → reactive >> autoregressive
     (the state-reactive fingerprint; R6c is the opposite, P22 is chance on both)
  Rashomon (uniqueness)                                 → SMALL/crisp set (not arbitrary like the churner)
  N7 time-reversal (the pre-registered decider)         → ASYMMETRIC (a ratchet R6c & P22 both lack),
     with an analytic BELIEF-BLIND control on the same env to isolate genuine reactivity from the
     episodic terminal-liquidation confound, and the known-truth c_ratchet/c_persister/c_iid controls.

The Rashomon (`rashomon`) and N7 (`perm_irreversibility`/`shuffle_pct`/`codelen_asym`/`c_*`) functions are
copied VERBATIM from interpretability/{k_rashomon,n7_time_reversal}.py (the other agent's validated measures)
to reuse the exact instruments without import side-effects. Run: python interpretability/series_g_corner_test.py
"""
from __future__ import annotations
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402
from src.series_g.phase0_gate import solve_belief_aware, solve_belief_blind  # noqa: E402
from src.series_g.regime_pomdp import PRIMARY_ENRICHED, RegimePOMDP  # noqa: E402

_rng = np.random.default_rng(7)


# ---- N7 time-reversal (verbatim from interpretability/n7_time_reversal.py) ----
def ordinal_patterns(x, m=3):
    x = np.asarray(x, float) + 1e-9 * _rng.standard_normal(np.asarray(x).size)
    perms = {p: i for i, p in enumerate(itertools.permutations(range(m)))}
    idx = [perms[tuple(np.argsort(x[k:k + m], kind='stable'))] for k in range(len(x) - m + 1)]
    return np.array(idx), perms


def perm_irreversibility(x, m=3):
    idx, perms = ordinal_patterns(x, m)
    if idx.size < 12:
        return np.nan
    P = np.bincount(idx, minlength=len(perms)).astype(float); P /= P.sum()
    inv = {v: k for k, v in perms.items()}
    mirror = np.array([perms[tuple(reversed(inv[i]))] for i in range(len(perms))])
    return 0.5 * np.abs(P - P[mirror]).sum()


def shuffle_pct(x, stat, reps=400):
    real = stat(x)
    null = np.array([stat(_rng.permutation(x)) for _ in range(reps)])
    null = null[~np.isnan(null)]
    return real, float(np.nanmean(null)), (null <= real).mean() * 100


def c_ratchet(n=280):
    x = np.zeros(n); v = 0.0
    for t in range(n):
        v += 0.05
        if v > 1 or _rng.random() < 0.04:
            v = _rng.uniform(0, 0.2)
        x[t] = v
    return x


def c_persister(n=280):
    x = np.zeros(n); x[0] = _rng.random()
    for t in range(1, n):
        x[t] = x[t - 1] if _rng.random() < 0.85 else _rng.random()
    return x


def c_iid(n=280):
    return _rng.random(n)


# ---- Rashomon (verbatim from interpretability/k_rashomon.py) ----
def _lag(a, k):
    return np.concatenate([np.full(k, np.nan), a[:-k]])


def _build_state(x, r=None):
    df = {'l1': _lag(x, 1), 'l2': _lag(x, 2), 'l3': _lag(x, 3),
          'rm5': pd.Series(x).rolling(5).mean().shift(1).to_numpy()}
    if r is not None:
        df['r1'] = _lag(r, 1); df['r2'] = _lag(r, 2)
    S = pd.DataFrame(df); y = pd.Series(x); m = S.notna().all(axis=1)
    return S[m].to_numpy(), y[m].to_numpy()


def _oos_loss(S, y, feats, K, seed, folds=3):
    Xf = S[:, feats]; kf = KFold(folds, shuffle=True, random_state=seed); err = 0.0; nte = 0
    for tr, te in kf.split(Xf):
        mu = Xf[tr].mean(0); sd = Xf[tr].std(0) + 1e-9
        km = KMeans(K, n_init=2, random_state=seed).fit((Xf[tr] - mu) / sd)
        cmean = np.array([y[tr][km.labels_ == c].mean() if (km.labels_ == c).any() else y[tr].mean() for c in range(K)])
        pred = cmean[km.predict((Xf[te] - mu) / sd)]
        err += np.sum((y[te] - pred) ** 2); nte += len(te)
    return err / nte


def rashomon(x, r=None, N=100):
    S, y = _build_state(np.asarray(x, float), None if r is None else np.asarray(r, float))
    base = np.var(y); losses = []; nf = S.shape[1]
    for i in range(N):
        K = int(_rng.integers(2, 10)); k = int(_rng.integers(1, nf + 1))
        feats = sorted(_rng.choice(nf, k, replace=False).tolist())
        losses.append(_oos_loss(S, y, feats, K, seed=i))
    losses = np.array(losses); Lstar = min(losses.min(), base)
    struct = max(0.0, 1 - Lstar / base)
    ratio25 = float((losses <= Lstar * 1.25).mean())
    return {'n': int(len(y)), 'structure_frac': round(float(struct), 3), 'ratio_e25': round(ratio25, 3)}


# ---- simulatability (reactive vs autoregressive) ----
def reactive_sim(X, y):
    n = len(y); cut = int(n * 0.6)
    if len(np.unique(y[cut:])) < 2:
        return float("nan")
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X[:cut], y[:cut])
    return round(float(balanced_accuracy_score(y[cut:], clf.predict(X[cut:]))), 3)


def autoreg_sim(y, ep=None):
    best = 0.0
    for order in (1, 2):
        n = len(y); cut = int(n * 0.6); K = int(y.max()) + 1
        tbl = defaultdict(lambda: np.zeros(K))
        for i in range(order, cut):
            if ep is not None and ep[i] != ep[i - order]:
                continue
            tbl[tuple(y[i - order:i])][y[i]] += 1
        if len(np.unique(y[cut:])) < 2:
            continue
        pred = [int(np.argmax(tbl[tuple(y[i - order:i])])) if tbl[tuple(y[i - order:i])].sum() > 0
                else int(np.bincount(y[:cut]).argmax()) for i in range(cut, n)]
        best = max(best, float(balanced_accuracy_score(y[cut:], pred)))
    return round(best, 3)


# ---- Series-G rollout of a policy on the N=1 env ----
def rollout(policy_fn, n_episodes=160, seed=1):
    env = MultiAssetRegimePOMDP(n_assets=1, seed=seed)
    recs = []
    for ep in range(n_episodes):
        env.reset(seed=seed * 1000 + ep)
        done = False
        while not done:
            a = int(policy_fn(env))
            recs.append({"ep": ep, "t": env.t, "belief": float(env.belief), "inv": int(env.inv[0]),
                         "action": a, "regime": int(env.regime)})
            _, r, term, trunc, _ = env.step([a])
            recs[-1]["reward"] = float(r)
            done = term or trunc
    return pd.DataFrame(recs)


def battery(df, name, reactive=True):
    act = df["action"].to_numpy(int)
    ep = df["ep"].to_numpy(int) if "ep" in df.columns else None
    l0 = behavioral_complexity_dynamic(act, kind="discrete", dts=(1, 2), n_null=300, n_boot=300, seed=0)
    row = {"stream": name, "x_bits_per_action": l0["h_mu_range"], "structure": l0["structure_present_configs"],
           "sim_autoregressive": autoreg_sim(act, ep)}
    if reactive:
        X = StandardScaler().fit_transform(df[["belief", "inv", "t"]].to_numpy(float))
        row["sim_reactive_state"] = reactive_sim(X, act)
    # Rashomon on the exposure/inventory stance
    stance = df["inv"].to_numpy(float)
    rsh = rashomon(stance, df["reward"].to_numpy(float) if "reward" in df.columns else None)
    row["rashomon_structure_frac"] = rsh["structure_frac"]; row["rashomon_ratio_e25"] = rsh["ratio_e25"]
    # N7 on the inventory trajectory
    _, _, pct = shuffle_pct(stance, lambda z: perm_irreversibility(z, 3))
    row["N7_shuffle_pct"] = round(pct, 1)
    row["N7_verdict"] = "ASYMMETRIC" if pct >= 95 else ("borderline" if pct >= 90 else "symmetric")
    return row


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    MODEL = HERE.parent / "src/series_g/corner_ppo_n1.zip"

    # analytic policies (belief-aware optimum + belief-blind control)
    m = RegimePOMDP(**PRIMARY_ENRICHED)
    g, Va, pol = solve_belief_aware(m); _, pol_blind = solve_belief_blind(m)
    nb = len(g)

    env = Monitor(MultiAssetRegimePOMDP(n_assets=1, seed=0))
    if MODEL.exists():
        model = PPO.load(str(MODEL), env=env, device="cpu"); print(f"[corner] loaded {MODEL.name}")
    else:
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256, ent_coef=0.01, gamma=0.99,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=0, device="cpu", verbose=0)
        print("[corner] training single-asset Series-G PPO (120k)...")
        model.learn(total_timesteps=120_000); model.save(str(MODEL))

    def ppo_fn(e):
        a, _ = model.predict(e._obs(), deterministic=True); return int(np.asarray(a).reshape(-1)[0])

    def aware_fn(e):
        return int(pol[e.t, int(round(e.belief * (nb - 1))), int(e.inv[0])])

    def blind_fn(e):
        return int(pol_blind[e.t, int(e.inv[0])])

    df_ppo = rollout(ppo_fn); df_aware = rollout(aware_fn); df_blind = rollout(blind_fn)
    # PPO return vs optimal vs blind (sanity)
    ret = {k: round(float(d.groupby("ep")["reward"].sum().mean()), 3)
           for k, d in [("PPO", df_ppo), ("optimal", df_aware), ("belief_blind", df_blind)]}

    rows = [battery(df_ppo, "Series-G PPO (MODEL)"), battery(df_aware, "Series-G analytic-optimum"),
            battery(df_blind, "Series-G belief-BLIND (N7 control)")]

    # reference real policies (R6c persister, P22 churner) on their cash streams (no obs join needed)
    for nm, path in [("R6c csi500 (persister ref)", "_streams/r6c_deadline_daily.csv"),
                     ("P22 csi500 (churner ref)", "_streams/p22_deadline_daily.csv")]:
        d = pd.read_csv(HERE / path)
        cash = d["cash"].to_numpy(float)
        cash_sym = np.clip(np.digitize(cash, np.quantile(cash, [1/3, 2/3])), 0, 2)
        l0 = behavioral_complexity_dynamic(cash_sym, kind="discrete", dts=(1, 2), n_null=300, n_boot=300, seed=0)
        rsh = rashomon(cash, d["net_return"].to_numpy(float) if "net_return" in d.columns else None)
        _, _, pct = shuffle_pct(cash, lambda z: perm_irreversibility(z, 3))
        rows.append({"stream": nm, "x_bits_per_action": l0["h_mu_range"], "structure": l0["structure_present_configs"],
                     "sim_autoregressive": autoreg_sim(cash_sym), "sim_reactive_state": "(prior: R6c~0.2 / P22~0.33)",
                     "rashomon_structure_frac": rsh["structure_frac"], "rashomon_ratio_e25": rsh["ratio_e25"],
                     "N7_shuffle_pct": round(pct, 1), "N7_verdict": "ASYMMETRIC" if pct >= 95 else ("borderline" if pct >= 90 else "symmetric")})

    # known-truth N7 controls (instrument validation)
    ctrls = {}
    for cn, cf in [("c_ratchet(state-reactive*)", c_ratchet), ("c_persister*", c_persister), ("c_iid*", c_iid)]:
        _, _, pct = shuffle_pct(cf(), lambda z: perm_irreversibility(z, 3))
        ctrls[cn] = {"N7_shuffle_pct": round(pct, 1), "verdict": "ASYMMETRIC" if pct >= 95 else ("borderline" if pct >= 90 else "symmetric")}

    ppo = rows[0]; blind = rows[2]
    occupies = bool(min(ppo["x_bits_per_action"]) > 0.6 and ppo["structure"].split("/")[0] != "0"
                    and isinstance(ppo["sim_reactive_state"], float) and ppo["sim_reactive_state"] >= 0.5
                    and ppo["sim_reactive_state"] > ppo["sim_autoregressive"]
                    and ppo["rashomon_ratio_e25"] <= 0.4
                    and ppo["N7_shuffle_pct"] >= 95 and ppo["N7_shuffle_pct"] > blind["N7_shuffle_pct"])
    report = {"returns_PPO_vs_optimal_vs_blind": ret, "battery": rows, "N7_controls": ctrls,
              "corner_occupied": occupies,
              "verdict": ("OCCUPIES THE CORNER — the trained Series-G model is high-complexity WITH structure, "
                          "simulatable from STATE (reactive > autoregressive), crisp (small Rashomon), and TIME-"
                          "ASYMMETRIC beyond the belief-blind control — the state-reactive corner R6c & P22 leave empty."
                          if occupies else
                          "does NOT cleanly occupy the corner on all axes — inspect which gate failed (see battery).")}
    (HERE / "series_g_corner_test_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\nreturns:", ret)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nN7 controls:", json.dumps(ctrls))
    print("\nCORNER OCCUPIED:", occupies, "\n", report["verdict"])
    print("[corner] wrote series_g_corner_test_report.json")


if __name__ == "__main__":
    main()
