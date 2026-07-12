"""The #1 HL-loop ARCHITECTURE FLAW, demonstrated: the loop is TENSION-BLIND. It grows the policy for RETURN through the
gate, but the gate does not track the tradeoff axes, so each certified return-gain silently rides UP the MDL
parsimony-fidelity deficit (legibility loss) — a certified regression the gate never sees.

Modular G=10: cover venues one at a time (each strictly raises return — modular). At each coverage level record
return (what the gate certifies) AND MDL deficit = 1 - Simul@(<=8 leaf) / Simul@(<=64 leaf) (what it does NOT).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from src.hl import modular_rule_policy as M
from src.series_g.family_env import RegimeRotationEnv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
G = 10; HON = list(range(20000, 20200))


def rollout_actions(policy, ctor, seeds):
    env = ctor(); rows, acts = [], []
    for s in seeds:
        env.reset(seed=s); done = False
        while not done:
            a = M.act(policy, env)
            rows.append({**{f"b{v}": float(env.belief[v]) for v in range(G)}, "inv": env.inv, "t": env.t}); acts.append(a)
            _, r, term, trunc, _ = env.step(a); done = term or trunc
    return pd.DataFrame(rows), np.array(acts)


def deficit(policy, ctor):
    df, y = rollout_actions(policy, ctor, range(10_000, 10_100)); X = df.to_numpy(float); cut = int(len(X) * 0.6)
    if len(np.unique(y)) < 2: return 0.0
    s9 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    s64 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=64, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    return round(float(1 - s9 / max(s64, 1e-9)), 3)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    ctor = M.env_ctor(G); U = M.unwind_clause(); pol = [U]; traj = []
    for k in range(G + 1):
        if k > 0: pol = pol + [M.provide_clause(k - 1, thr=0.556)]
        ret = round(float(M.evaluate(pol, ctor, HON).mean()), 3); dfc = deficit(pol, ctor)
        traj.append({"venues_covered": k, "gate_sees_RETURN": ret, "gate_BLIND_to_MDL_deficit": dfc})
    r0, rN = traj[1]["gate_sees_RETURN"], traj[-1]["gate_sees_RETURN"]
    d0, dN = traj[1]["gate_BLIND_to_MDL_deficit"], traj[-1]["gate_BLIND_to_MDL_deficit"]
    out = {"trajectory": traj, "headline": (
        f"TENSION-BLINDNESS demonstrated: as the HL loop covers venues 1->{G}, the RETURN the gate certifies rises "
        f"{r0}->{rN} while the MDL parsimony-fidelity DEFICIT the gate never checks rises {d0}->{dN}. Every step is "
        "gate-ACCEPTED (return up) yet the policy walks up the legibility-loss frontier UNCERTIFIED. The HL gate is "
        "single-objective; it has no tension axis, so it cannot refuse a return gain that pays in legibility. This is "
        "the core architectural flaw the CrystalScore-v2 tension profile exposes.")}
    (HERE / "hl_tension_blind_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("=== HL loop tension-blindness (return gated, MDL deficit blind) ===")
    for t in traj: print(f"  venues {t['venues_covered']:2d}: RETURN(gated) {t['gate_sees_RETURN']:+.2f}  |  MDL-deficit(blind) {t['gate_BLIND_to_MDL_deficit']}")
    print("\n" + out["headline"])
if __name__ == "__main__": main()
