"""HL v4 over R6c — the tension adapter: real HCS policy-forward daily logs -> CrystalScore-v2 tension vectors.

Substrate: a completed HCS run dir (per-candidate validation_daily.csv, 251 aligned fold-2021 trading days).
Axes on this substrate:
  sharpe       (MAXIMIZE)  — annualized net-return Sharpe on a DATE SLICE (the capability axis)
  mdl_deficit  (MINIMIZE)  — REAL CrystalScore-v2 Axis A for the controller: 1 - Simul@8-leaf/Simul@64-leaf trees
                             predicting the daily controller decision (cash_trade_direction in {-1,0,1}) from
                             NAMED state columns — how much of the controller's behavior a K<=9 story reproduces.
Authority (tension-harm) unit: CONFIG DESCRIPTION LENGTH = # knob overrides vs pf_original (the registry diff).
Statistics: paired daily deltas vs the incumbent with a MOVING-BLOCK bootstrap SE (honest under autocorrelation).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

STATE_COLS = ["risk_stress", "recovery_score", "drawdown", "confidence_derisk", "confidence_rerisk",
              "risk_score", "cash_duration_score", "target_strength", "q_anchor", "cash_anchor", "stop_active"]
KNOB_EXCLUDE = {"generation", "family", "name"}
AXES = {"sharpe": +1, "mdl_deficit": -1}


def load_run(run_dir):
    """Load an HCS run dir: per-candidate daily logs, the candidate registry, and summary; assert date alignment."""
    run = Path(run_dir)
    logs = {p.parent.name: pd.read_csv(p) for p in run.glob("daily/**/validation_daily.csv")}
    reg = pd.read_csv(run / "candidate_registry.csv")
    summary = pd.read_csv(run / "summary.csv")
    dates = None
    for name, d in logs.items():
        if dates is None:
            dates = d["date"].to_numpy()
        elif not (d["date"].to_numpy() == dates).all():
            raise ValueError(f"date misalignment in {name}")     # paired stats require aligned days
    return logs, reg, summary


def config_len(reg, cand):
    """Description length of the config diff vs pf_original (# overridden knobs). None if unknown -> deny-by-default."""
    base = reg[reg["name"] == "pf_original"]
    row = reg[reg["name"] == cand]
    if not len(row) or not len(base):
        return None
    base, row = base.iloc[0], row.iloc[0]
    n = 0
    for c in reg.columns:
        if c in KNOB_EXCLUDE:
            continue
        a, b = row[c], base[c]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or a != b:
            n += 1
    return float(n)


def sharpe(returns):
    """Annualized Sharpe of a daily return series (0 if the series is degenerate)."""
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    return 0.0 if sd < 1e-12 else float(r.mean() / sd * np.sqrt(252))


def mdl_deficit(daily, idx):
    """Axis A on the given day-slice: small-tree vs big-tree simulatability of the controller decision."""
    d = daily.iloc[idx]
    y = d["cash_trade_direction"].to_numpy()
    X = d[STATE_COLS].to_numpy(float)
    cut = int(len(d) * 0.6)
    if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return 0.0
    s8 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=8, random_state=0)
                                 .fit(X[:cut], y[:cut]).predict(X[cut:]))
    s64 = balanced_accuracy_score(y[cut:], DecisionTreeClassifier(max_leaf_nodes=64, random_state=0)
                                  .fit(X[:cut], y[:cut]).predict(X[cut:]))
    return float(max(0.0, 1 - s8 / max(s64, 1e-9)))


def tension_vector(daily, dev_idx):
    """Score a candidate on the day-slice: {sharpe (capability), mdl_deficit (controller legibility)}."""
    return {"sharpe": sharpe(daily["net_return"].to_numpy()[dev_idx]),
            "mdl_deficit": mdl_deficit(daily, dev_idx)}


def dominates(a, b, tol=1e-6):
    """True if vector a Pareto-dominates b over AXES (>= on every axis, strictly better on at least one)."""
    ge = all((a[k] - b[k]) * AXES[k] >= -tol for k in AXES)
    gt = any((a[k] - b[k]) * AXES[k] > tol for k in AXES)
    return ge and gt


def non_dominated(vec, frontier, tol=1e-6):
    """True if `vec` is dominated by no point on the frontier (admissible)."""
    return not any(dominates(f, vec, tol) for f in frontier)


def block_z(delta, block=5, n_boot=2000, seed=0):
    """z = mean/SE with a moving-block bootstrap SE — paired daily deltas are autocorrelated; an i.i.d. SE would
    overstate significance, which is exactly the optimism this gate exists to refuse."""
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, float)
    n = len(delta)
    nb = max(1, int(np.ceil(n / block)))
    starts = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nb)])[:n]
        means[i] = delta[idx].mean()
    se = float(means.std(ddof=1)) + 1e-12
    return float(delta.mean() / se), se
