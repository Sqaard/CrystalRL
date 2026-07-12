"""HL v4 — the TENSION VECTOR + Pareto machinery (fixes F1/F10: the objective is a frontier, not scalar return).

A candidate is scored on a VECTOR of axes that cannot all be maximized at once (CrystalScore-v2). On the modular
substrate the two load-bearing, cheaply-measurable axes are:
  return          (MAXIMIZE) — the alpha/capability axis
  description_len (MINIMIZE) — the legibility axis = # clauses (the exact MDL numerator; more coverage => longer story)
plus a periodically-measured true MDL DEFICIT (Simul@K<=9 / Simul@ceiling) used for archive binning + honest reporting.
Pareto dominance is the admission relation: a move is admissible only if it is NON-DOMINATED (you cannot add return by
paying legibility unless it expands the frontier). This is F1/F10 fixed at the representation level.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from src.hl import modular_rule_policy as M

# axis directions: +1 = higher is better, -1 = lower is better
AXES = {"return": +1, "description_len": -1}


def description_len(policy):
    return float(sum(1 for c in policy if c[0] == "provide") + (1 if any(c[0] == "unwind" for c in policy) else 0))


def tension_vector(policy, ctor, seeds):
    return {"return": float(M.evaluate(policy, ctor, seeds).mean()), "description_len": description_len(policy)}


def dominates(a, b, tol=1e-6):
    """a Pareto-dominates b: a is >= on every axis (in the good direction) and strictly better on at least one."""
    ge = all((a[k] - b[k]) * AXES[k] >= -tol for k in AXES)
    gt = any((a[k] - b[k]) * AXES[k] > tol for k in AXES)
    return ge and gt


def non_dominated(vec, frontier, tol=1e-6):
    """True if `vec` is not dominated by any point already on the frontier (i.e. it is admissible / expands it)."""
    return not any(dominates(f, vec, tol) for f in frontier)


def G_of(ctor):
    return ctor().G


def mdl_deficit(policy, ctor, G, n_seeds=80):
    """True legibility axis: 1 - Simul@(<=8 leaf, K<=9) / Simul@(<=64 leaf ceiling) over the named belief state."""
    env = ctor(); rows, acts = [], []
    for s in range(11_000, 11_000 + n_seeds):
        env.reset(seed=s); done = False
        while not done:
            a = M.act(policy, env)
            rows.append({**{f"b{v}": float(env.belief[v]) for v in range(G)}, "inv": env.inv, "t": env.t}); acts.append(a)
            _, r, term, trunc, _ = env.step(a); done = term or trunc
    y = np.array(acts)
    if len(np.unique(y)) < 2:
        return 0.0
    X = pd.DataFrame(rows).to_numpy(float); cut = int(len(X) * 0.6)
    s9 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=8, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    s64 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=64, random_state=0).fit(X[:cut], y[:cut]).predict(X[cut:]))
    return float(max(0.0, 1 - s9 / max(s64, 1e-9)))
