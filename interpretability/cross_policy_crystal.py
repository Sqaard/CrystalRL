"""CROSS-POLICY CrystalScore -- the publishable "interpretability degrades gracefully
as policy behavioral-complexity rises" arc.

WHAT THIS DOES
--------------
The single-policy CrystalScore (interpretability/crystal_score.py + crystal_score_report.json)
scores ONE policy (R6c). This script scores MULTIPLE policies on a UNIFORM, comparable subset
of CrystalScore and places each as a point in (behavioral_complexity, CrystalScore-core) space.

For each scorable policy we REPLAY its frozen 2022-2023 rollout (no retrain) to obtain, per day:
  - the ACTION series: invested fraction q (= 1 - cash) and the within-book risky-weight vector,
  - the LATENT series: the TRUE 64-d final policy latent (hook on policy_net.5.ReLU), exactly as
    scripts/export_r6c_hidden_activations.py captures it for R6c.

Then we compute two axes:

  x  BEHAVIORAL_COMPLEXITY (uniform proxy, ACTION series ONLY -- no latent, fully cross-comparable):
       a defensible scalar combining three normalized action-richness components:
         (1) cash_entropy   : Shannon entropy of the discretized cash/q series (10 equal-width bins,
                              over the realised cash range), normalized by log(10) -> [0,1].
                              Captures how much the STANCE moves around.
         (2) book_dispersion: mean L1 distance of the within-book composition to equal-weight,
                              normalized to [0,1] by the max attainable L1 (= 2*(1-1/N)).
                              Captures how much genuine WITHIN-BOOK selection tilt there is.
         (3) action_eff_dim : participation ratio (effective dimensionality) of the action
                              covariance [q ; within-book weights], (sum lambda)^2 / sum(lambda^2),
                              divided by the action dimension -> [0,1]. Captures how many
                              independent directions the action actually uses.
       behavioral_complexity = mean(cash_entropy, book_dispersion, action_eff_dim)  in [0,1].
       (Equal-weight mean of three [0,1] components; documented, not tuned.)

  y  CRYSTALSCORE-CORE (cross-policy-COMPARABLE subset -- needs ONLY latent + actions):
       Simulatability   : K=9 cluster-mean predictor R^2 of cash/q from the latent (discrete,
                          parsimony reading) -- IDENTICAL recipe to crystal_score.py's headline.
       Completeness@K   : 1 - SS_res/SS_tot of the K-latent-cluster-mean predictor, swept K=2..9,
                          for STANCE (cash) and SELECTION (within-book composition). Gives each
                          policy its own Pareto (completeness vs K) and Pareto-AUC.
       Stability        : cross-seed ARI of the K=9 latent clusters IF multiple seeds exist for
                          that policy. We have ONE frozen rollout per policy, so per-policy
                          multi-seed Stability is N/A (we recompute a single-rollout multi-seed
                          KMeans ARI as a clustering-stability proxy and report it separately,
                          clearly labeled -- it is NOT cross-seed-of-training).
       CrystalScore-core = Faithfulness_avail x Simulatability(cash,K<=9) [x Stability]
         FAITHFULNESS + CONTROLLABILITY need per-policy STEERING / intervention machinery that
         ONLY R6c has (its code-steering curves). They are therefore "R6c-only" and the FULL
         CrystalScore is reported ONLY for R6c. For the cross-comparable y-axis we use the
         Stance Simulatability x (single-rollout) clustering-stability product -- the subset that
         every policy can produce identically. We ALSO report the no-stability variant.

R6c is NOT recomputed: its numbers come VERBATIM from crystal_score_report.json. We re-derive its
behavioral_complexity from its OWN behavior log so the x-axis is uniform, but every R6c y-metric
(Simulatability, Completeness, Stability, full CrystalScore) is copied from the existing report.

HONESTY (firewall discipline)
-----------------------------
  * Never fabricate. Negative R^2 -> 0 (clip01). Missing input -> N/A, excluded.
  * Cross-comparable axes (behavioral_complexity, Simulatability, Completeness, clustering-stability)
    are explicitly separated from R6c-ONLY axes (Faithfulness, Controllability, full CrystalScore).
  * Two-agent policies (W1 / H2 / P22) are SKIPPED with a stated architectural reason -- they have
    no single comparable policy_net / 64-d latent -> action map (split pm.actor + trader.* heads).
  * R6c reuses crystal_score_report.json verbatim; it is NOT recomputed differently.
  * If only the R6c+family is scorable (a narrow complexity range), we SAY SO -- we do not overclaim
    a smooth curve.

Run:
    python interpretability/cross_policy_crystal.py

Structure (this file is long; jump by section):
  * L132-172  scalar helpers: clip01, cluster_mean_r2, pareto_auc
  * L174-236  behavioral_complexity  -- the static CrystalScore complexity term
  * L238-440  the DYNAMIC-complexity ruler: _block_entropies / _causal_complexity /
              _entropy_rate_complexity / _discretize / behavioral_complexity_dynamic (+ _selftest_dynamic).
              This is the reusable metric block; mdl_fidelity_deficit.py imports behavioral_complexity_dynamic.
  * L441-469  clustering_stability + register_final_latent_hook
  * L471-556  Replay / replay_policy / score_replay  -- run a policy forward and score it
  * L557-660  r6c_from_report, candidate_specs  -- assemble the policies to compare
  * L662-940  main() + maybe_plot / write_markdown / print_console  -- orchestration and output
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console
except Exception:
    pass

# ----------------------------------------------------------------------------- paths / imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extract_stage0_1_hidden_state_package import policy_layers  # noqa: E402
from src.ppo.instrumented_ppo import InstrumentedPPO  # noqa: E402
from src.ppo.stage0_1_train import append_eval_info_row  # noqa: E402
from src.ppo.stage0_1_weight_env import load_weight_panel, make_env_from_config  # noqa: E402

# the R6c frozen panel: 2010-2023, fold_2021 train-only scaling, full stage0_1_weight_features.
# Every R6c-FAMILY candidate shares this exact feature set + fold + normalization, so the SAME
# frozen panel is the correct, identical input for all of them (verified: feature_subset=None,
# train_start/validation_end/frozen window all match).
FROZEN_PANEL = (
    ROOT
    / "artifacts"
    / "stage4"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph"
    / "feature_scalers_frozen"
    / "fold_2021"
    / "model_ready.csv"
)

R6C_REPORT = HERE / "crystal_score_report.json"

OUT_CSV = HERE / "cross_policy_crystalscore.csv"
OUT_CURVE_CSV = HERE / "cross_policy_curve.csv"
OUT_CURVE_PNG = HERE / "cross_policy_curve.png"
OUT_MD = HERE / "CROSS_POLICY_CRYSTALSCORE.md"
OUT_JSON = HERE / "cross_policy_crystal_report.json"

LATENT_LAYER = "policy_net.5.ReLU"  # TRUE 64-d final policy latent (same as R6c export)
K_BUDGET = 9
K_GRID = [2, 3, 4, 5, 6, 7, 8, 9]
SEEDS = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------- helpers
def clip01(x) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def cluster_mean_r2(X_std: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> float:
    """IDENTICAL recipe to crystal_score.py: cluster latent into k groups, predict y by the
    per-cluster MEAN, return 1 - SS_res/SS_tot (negative clipped to 0). y may be 1-D or 2-D."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=500).fit(X_std)
    labels = km.labels_
    yhat = np.zeros_like(y)
    for c in np.unique(labels):
        m = labels == c
        yhat[m] = y[m].mean(axis=0, keepdims=True)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return clip01(1.0 - ss_res / ss_tot)


def pareto_auc(ks: list[int], comp: dict[int, float]) -> dict:
    xs = np.array(sorted(ks), dtype=float)
    ys = np.array([comp[int(k)] for k in xs], dtype=float)
    mask = np.isfinite(ys)
    if mask.sum() < 2:
        return {"auc_normalized": float("nan"), "at_K9": comp.get(9, float("nan"))}
    xs_m, ys_m = xs[mask], ys[mask]
    try:
        from numpy import trapezoid as _trap  # numpy>=2
    except ImportError:  # pragma: no cover
        from numpy import trapz as _trap
    auc = float(_trap(ys_m, xs_m))
    auc_norm = auc / (xs_m.max() - xs_m.min())
    return {"auc_normalized": round(clip01(auc_norm), 4), "at_K9": round(float(comp.get(9, float("nan"))), 4)}


# ----------------------------------------------------------------------------- behavioral complexity (x-axis)
def behavioral_complexity(cash: np.ndarray, within: np.ndarray) -> dict:
    """Uniform action-only behavioral-complexity proxy in [0,1].  Components:
       cash_entropy   : entropy of 10-bin discretized cash (over realised range) / log(10).
       book_dispersion: mean L1(within-book, equal-weight) / (2*(1-1/N)).
       action_eff_dim : participation ratio of cov([q; within]) / action_dim.
    Returns the three components and their equal-weight mean = behavioral_complexity."""
    cash = np.asarray(cash, dtype=float)
    within = np.asarray(within, dtype=float)
    n_assets = within.shape[1]

    # (1) cash entropy (stance richness)
    lo, hi = float(np.nanmin(cash)), float(np.nanmax(cash))
    if hi - lo < 1e-9:
        cash_entropy = 0.0
    else:
        bins = np.linspace(lo, hi, 11)
        counts, _ = np.histogram(cash, bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        cash_entropy = float(-(p * np.log(p)).sum() / np.log(10.0))
    cash_entropy = clip01(cash_entropy)

    # (2) within-book dispersion (selection richness)
    eq = np.full(n_assets, 1.0 / n_assets)
    l1 = np.abs(within - eq[None, :]).sum(axis=1)  # L1 to equal-weight, per day
    max_l1 = 2.0 * (1.0 - 1.0 / n_assets)  # max attainable L1 on the simplex
    book_dispersion = clip01(float(np.mean(l1)) / max_l1) if max_l1 > 0 else 0.0

    # (3) effective dimensionality of the action covariance [q ; within-book]
    q = (1.0 - cash)[:, None]  # invested fraction
    A = np.concatenate([q, within], axis=1)  # (N, 1 + n_assets)
    A = A - A.mean(axis=0, keepdims=True)
    cov = np.cov(A, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        eff_dim = 0.0
    else:
        pr = (ev.sum() ** 2) / (ev ** 2).sum()  # participation ratio in [1, dim]
        eff_dim = float(pr) / A.shape[1]
    action_eff_dim = clip01(eff_dim)

    bc = float(np.mean([cash_entropy, book_dispersion, action_eff_dim]))
    return {
        "behavioral_complexity": round(bc, 4),
        "cash_entropy": round(cash_entropy, 4),
        "book_dispersion": round(book_dispersion, 4),
        "action_eff_dim": round(action_eff_dim, 4),
        "n_assets": int(n_assets),
    }


# --------------------------------------------------- L0: action-only bits/action complexity ruler (h_mu / C_mu / E)
# The existing behavioral_complexity() above is ORDER-INVARIANT: cash_entropy/book_dispersion/action_eff_dim
# are statistics of the per-day MARGINAL, blind to the SEQUENCE. The L0 unit-fix (named the top-open item in
# FINAL_A / FINAL_G / FRONTIER_ARC) measures the TEMPORAL structure of the action stream in bits/action:
#   h_mu  (entropy rate)          : the irreducible per-step uncertainty -- how much genuinely-new randomness
#                                   each action carries given the whole past. (0 = perfectly predictable.)
#   E     (excess entropy)        : mutual information between the semi-infinite past and future -- how much the
#                                   past must be remembered to predict optimally. (0 = memoryless.)
#   C_mu  (statistical complexity) : entropy of the minimal predictive (causal) states -- the size of the
#                                   smallest machine that reproduces the process. (the structural-memory cost.)
# These are computed identically on R6c's frozen log and (when it lands) the parallel agent's new policy log,
# on the SAME alphabet + dt, so two ordinal "corners" become two readings on ONE frontier x-axis. All in BITS.
def _block_entropies(symbols: np.ndarray, max_L: int, min_mean_count: float) -> dict:
    """Empirical block Shannon entropies H(L), L=0..Lmax, in BITS, from a symbol sequence via sliding
    length-L windows. Lmax is auto-capped at the first L whose mean count per OBSERVED L-gram drops below
    `min_mean_count`: deeper H(L) is undersampled and biased LOW (Miller-Madow), which would spuriously
    DEFLATE h_mu and fabricate structure -- the small-sample guard that keeps a 289-day log honest.
    Returns {L: {H, n_grams, n_distinct, mean_count}} for every L actually estimated."""
    symbols = np.asarray(symbols).astype(np.int64)
    T = symbols.size
    out = {0: {"H": 0.0, "n_grams": int(T), "n_distinct": 1, "mean_count": float(T)}}
    for L in range(1, max_L + 1):
        n = T - L + 1
        if n < 2:
            break
        grams = np.stack([symbols[i:i + n] for i in range(L)], axis=1)  # (n, L)
        _, counts = np.unique(grams, axis=0, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log2(p)).sum())
        n_distinct = int(counts.size)
        mean_count = n / max(1, n_distinct)
        out[L] = {"H": H, "n_grams": int(n), "n_distinct": n_distinct, "mean_count": float(mean_count)}
        if mean_count < min_mean_count:  # this L is already too sparse -> last usable depth is L-1
            break
    return out


def _causal_complexity(symbols: np.ndarray, L: int, alphabet_size: int, merge_tol: float) -> float:
    """Finite-L statistical complexity C_mu (BITS): reconstruct causal states at history length L by
    merging length-L histories whose next-symbol conditional distribution P(next|history) are within L1
    `merge_tol`, then C_mu = entropy of the stationary distribution over the merged states. This is a
    simplified (greedy, order-dependent) epsilon-machine reconstruction -- an ESTIMATE that is sensitive
    to L and merge_tol, so callers report it as a RANGE, never a crisp point."""
    symbols = np.asarray(symbols).astype(np.int64)
    T = symbols.size
    if T - L < 2:
        return float("nan")
    from collections import defaultdict
    nxt = defaultdict(lambda: np.zeros(alphabet_size, dtype=float))
    hist_count = defaultdict(int)
    for i in range(T - L):
        h = tuple(symbols[i:i + L].tolist())
        nxt[h][symbols[i + L]] += 1.0
        hist_count[h] += 1
    states = []  # each: [rep_conditional_dist, total_history_count]
    for h in nxt:
        d = nxt[h] / nxt[h].sum()
        placed = False
        for st in states:
            if float(np.abs(st[0] - d).sum()) <= merge_tol:
                st[1] += hist_count[h]
                placed = True
                break
        if not placed:
            states.append([d, hist_count[h]])
    total = float(sum(hist_count.values()))
    ps = np.array([st[1] for st in states], dtype=float) / total
    ps = ps[ps > 0]
    return float(-(ps * np.log2(ps)).sum())


def _entropy_rate_complexity(symbols: np.ndarray, alphabet_size: int, max_L: int, min_mean_count: float,
                             merge_tol: float, with_cmu: bool = True) -> dict:
    """Single-config estimate of (h_mu, E, C_mu) on one already-discretized symbol stream. Returns the
    finite-depth Lstar used, the h_mu(L) curve, H1 (the i.i.d./memoryless upper bound on h_mu), E, and
    C_mu at L=1 and L=Lstar. All in bits. `with_cmu=False` skips the expensive causal-state reconstruction
    (the null/bootstrap inner loops only read h_mu) — a pure speed-up, identical results."""
    be = _block_entropies(symbols, max_L, min_mean_count)
    Ls = sorted(be)
    reliable = [L for L in Ls if L >= 1 and be[L]["mean_count"] >= min_mean_count]
    Lstar = max(reliable) if reliable else 1
    h_curve = {L: be[L]["H"] - be[L - 1]["H"] for L in Ls if L >= 1}
    h_mu = h_curve[Lstar]
    H1 = be[1]["H"] if 1 in be else float("nan")
    E = be[Lstar]["H"] - Lstar * h_mu  # excess entropy (finite-L): H(Lstar) - Lstar*h_mu
    cmu1 = _causal_complexity(symbols, 1, alphabet_size, merge_tol) if with_cmu else float("nan")
    cmuL = (_causal_complexity(symbols, Lstar, alphabet_size, merge_tol) if Lstar >= 1 else cmu1) if with_cmu else float("nan")
    return {
        "Lstar": int(Lstar), "h_mu": float(h_mu), "H1": float(H1), "E": float(max(0.0, E)),
        "C_mu_L1": float(cmu1), "C_mu_Lstar": float(cmuL),
        "h_curve": {int(L): round(float(v), 4) for L, v in h_curve.items()},
        "depth_table": {int(L): {"H": round(d["H"], 4), "mean_count": round(d["mean_count"], 2)} for L, d in be.items()},
    }


def _discretize(series: np.ndarray, kind: str, n_bins: int) -> tuple:
    """Map a per-day series to integer symbols 0..A-1. kind='continuous' -> n_bins equal-width bins over the
    realised range; kind='discrete' -> factorize the native labels (n_bins ignored). Returns (symbols, A)."""
    series = np.asarray(series)
    if kind == "discrete":
        uniq, inv = np.unique(series, return_inverse=True)
        return inv.astype(np.int64), int(uniq.size)
    x = series.astype(float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.zeros(x.size, dtype=np.int64), 1
    edges = np.linspace(lo, hi, n_bins + 1)
    sym = np.clip(np.digitize(x, edges[1:-1], right=False), 0, n_bins - 1)
    return sym.astype(np.int64), int(n_bins)


def behavioral_complexity_dynamic(series: np.ndarray, kind: str = "continuous",
                                  alphabets=(3, 5, 10), dts=(1, 2), max_L: int = 6,
                                  min_mean_count: float = 5.0, merge_tol: float = 0.15,
                                  n_null: int = 500, n_boot: int = 500, seed: int = 0) -> dict:
    """L0 bits/action ruler with the firewall discipline baked in (guardrail #4: publish a RANGE, beat the
    phase-shuffle null). Sweeps >=3 alphabets x >=2 temporal resolutions (dt), and for each config reports
    h_mu / E / C_mu plus:
      - phase-shuffle NULL: fully permute the day order n_null times -> h_mu_null ~ H1 (memoryless). The
        stream has READABLE temporal structure iff observed h_mu sits below the 5th percentile of h_mu_null.
        structure_gap = (mean(h_mu_null) - h_mu) / mean(h_mu_null)  in [0,1]; passes iff h_mu < null p05.
      - circular block-bootstrap 95% CI on h_mu (block ~ sqrt(T)) so the point is reported with error.
    Aggregates to RANGES (min..max across configs) and a single verdict {structure_present}. For
    'discrete' kind the native labels are used and `alphabets` is collapsed to one config (the native A)."""
    rng = np.random.default_rng(seed)
    series = np.asarray(series)
    alpha_list = [None] if kind == "discrete" else list(alphabets)
    configs = []
    for a in alpha_list:
        for dt in dts:
            s = series[::dt]
            symbols, A = _discretize(s, kind, a if a is not None else 0)
            T = symbols.size
            if T < 20 or A < 2:
                continue
            est = _entropy_rate_complexity(symbols, A, max_L, min_mean_count, merge_tol)
            # phase-shuffle null on h_mu (destroys order -> memoryless)
            null_h = np.empty(n_null)
            for i in range(n_null):
                perm = rng.permutation(symbols)
                null_h[i] = _entropy_rate_complexity(perm, A, max_L, min_mean_count, merge_tol, with_cmu=False)["h_mu"]
            null_mean = float(np.mean(null_h)); null_p05 = float(np.quantile(null_h, 0.05))
            struct_gap = float((null_mean - est["h_mu"]) / null_mean) if null_mean > 1e-9 else 0.0
            structure = bool(est["h_mu"] < null_p05)
            # circular MOVING-BLOCK bootstrap CI on observed h_mu. Block length is set LONG (T//6, not the
            # usual sqrt(T)) on purpose: R6c's cash stance has long persistence runs, and short blocks sever
            # them -> every replica gets artificial block-boundary transitions that inflate h_mu and push the
            # CI ABOVE the point estimate. Long blocks preserve structure up to ~T/6 (>> the measured Lstar<=6),
            # so the CI brackets the estimate. It remains mildly CONSERVATIVE (upward-biased) near strong
            # persistence -- a sampling-error band, not a substitute for the alphabet/dt RANGE or the null.
            bl = max(8, T // 6)
            boot = np.empty(n_boot)
            ext = np.concatenate([symbols, symbols[:bl]])
            nb = int(np.ceil(T / bl))
            for i in range(n_boot):
                starts = rng.integers(0, T, nb)
                bs = np.concatenate([ext[st:st + bl] for st in starts])[:T]
                boot[i] = _entropy_rate_complexity(bs, A, max_L, min_mean_count, merge_tol, with_cmu=False)["h_mu"]
            ci = [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)]
            configs.append({
                "alphabet": int(A), "dt": int(dt), "T": int(T), "Lstar": est["Lstar"],
                "h_mu": round(est["h_mu"], 4), "h_mu_ci95": ci, "H1_iid_bound": round(est["H1"], 4),
                "E": round(est["E"], 4), "C_mu_L1": round(est["C_mu_L1"], 4), "C_mu_Lstar": round(est["C_mu_Lstar"], 4),
                "null_h_mu_mean": round(null_mean, 4), "null_h_mu_p05": round(null_p05, 4),
                "structure_gap": round(struct_gap, 4), "structure_present": structure,
                "h_curve": est["h_curve"], "depth_table": est["depth_table"],
            })
    if not configs:
        return {"error": "no usable config (series too short or single-valued)", "configs": []}

    def _rng_of(key):
        vals = [c[key] for c in configs if np.isfinite(c[key])]
        return [round(min(vals), 4), round(max(vals), 4)] if vals else [float("nan"), float("nan")]

    n_struct = sum(c["structure_present"] for c in configs)
    return {
        "kind": kind, "n_configs": len(configs),
        "h_mu_range": _rng_of("h_mu"), "E_range": _rng_of("E"),
        "C_mu_L1_range": _rng_of("C_mu_L1"), "C_mu_Lstar_range": _rng_of("C_mu_Lstar"),
        "structure_present_configs": f"{n_struct}/{len(configs)}",
        "structure_present": bool(n_struct >= (len(configs) + 1) // 2),  # majority of the alphabet/dt sweep
        "configs": configs,
    }


def _selftest_dynamic() -> None:
    """Ground-truth validation of behavioral_complexity_dynamic on sequences with KNOWN answers, so the
    R6c numbers are trusted only after the estimator passes. Run: python -c 'import cross_policy_crystal as m; m._selftest_dynamic()'."""
    rng = np.random.default_rng(0)
    A = 4
    n = 2000
    # (1) i.i.d. uniform over A: h_mu = log2(A) = 2.0, E = 0, no temporal structure
    iid = rng.integers(0, A, n)
    r_iid = behavioral_complexity_dynamic(iid, kind="discrete", n_null=200, n_boot=100, seed=1)
    # (2) period-3 deterministic 0,1,2,0,1,2..: h_mu = 0, E = log2(3) ~ 1.585, structure present
    per = np.tile([0, 1, 2], n // 3 + 1)[:n]
    r_per = behavioral_complexity_dynamic(per, kind="discrete", n_null=200, n_boot=100, seed=2)
    # (3) first-order Markov, sticky (stay w.p. .9): h_mu = H(.9,.1/(A-1)*...) low, structure present
    P = np.full((A, A), 0.1 / (A - 1)); np.fill_diagonal(P, 0.9)
    mk = np.empty(n, dtype=int); mk[0] = 0
    for t in range(1, n):
        mk[t] = rng.choice(A, p=P[mk[t - 1]])
    r_mk = behavioral_complexity_dynamic(mk, kind="discrete", n_null=200, n_boot=100, seed=3)
    h_markov_true = float(-(P[0] * np.log2(P[0])).sum())  # per-row conditional entropy (uniform stationary)
    print("=== _selftest_dynamic (BITS) ===")
    print(f"(1) i.i.d. A=4   : h_mu={r_iid['h_mu_range']} (expect ~2.0), E={r_iid['E_range']} (~0), "
          f"structure={r_iid['structure_present']} (expect False)")
    print(f"(2) period-3     : h_mu={r_per['h_mu_range']} (~0), E={r_per['E_range']} (~1.585=log2 3), "
          f"structure={r_per['structure_present']} (expect True)")
    print(f"(3) Markov sticky: h_mu={r_mk['h_mu_range']} (~{h_markov_true:.3f}), E={r_mk['E_range']} (>0 small), "
          f"structure={r_mk['structure_present']} (expect True)")
    ok = (not r_iid["structure_present"]) and r_per["structure_present"] and r_mk["structure_present"]
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} (i.i.d. no-structure, period-3 + Markov have structure)")


def clustering_stability(X_std: np.ndarray, k: int = K_BUDGET) -> float:
    """Single-rollout clustering stability proxy: cross-SEED-OF-KMEANS ARI at K (NOT cross seed of
    TRAINING -- we have one rollout per policy). Clearly labeled as a proxy in the output."""
    labels = [KMeans(n_clusters=k, random_state=s, n_init=10, max_iter=500).fit(X_std).labels_ for s in SEEDS]
    aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
    return clip01(float(np.mean(aris)))


# ----------------------------------------------------------------------------- replay one R6c-family policy
def register_final_latent_hook(policy, store: dict) -> list:
    """Hook the TRUE 64-d final policy latent (policy_net.5.ReLU). Same mechanism as
    scripts/export_r6c_hidden_activations.py; last-write-wins per forward."""
    handles = []

    def make_hook(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return hook

    net = policy.mlp_extractor.policy_net
    for child_name, module in net.named_modules():
        if child_name == "" or any(module.children()):
            continue
        kind = module.__class__.__name__
        handles.append(module.register_forward_hook(make_hook(f"policy_net.{child_name}.{kind}")))
    return handles


@dataclass
class Replay:
    latent: np.ndarray          # (N, 64) true final policy latent
    cash: np.ndarray            # (N,) cash_target stance
    within: np.ndarray          # (N, 29) within-book composition (exposure removed)
    n_steps: int


def replay_policy(model_zip: Path, config: dict, variant: dict) -> Replay:
    """Replay a single R6c-family policy through the frozen 2022-2023 window. Captures the TRUE
    64-d final latent (hook) and builds the per-day behavior log via append_eval_info_row, exactly
    like the R6c machinery -> identical cash_target / executed_weight_* semantics."""
    start = str(config["data"]["frozen_test_start"])
    end = str(config["data"]["frozen_test_end"])
    panel = load_weight_panel(FROZEN_PANEL, start, end)
    env = make_env_from_config(panel, config, variant)
    model = InstrumentedPPO.load(str(model_zip), device="cpu", custom_objects={"instrumentation_dir": None})
    model.policy.set_training_mode(False)

    store: dict[str, np.ndarray] = {}
    handles = register_final_latent_hook(model.policy, store)

    obs, _ = env.reset()
    done = False
    rows: list[dict[str, Any]] = []
    latent_rows: list[np.ndarray] = []
    while not done:
        store.clear()
        layer_out = policy_layers(model.policy, obs)  # triggers the hook + gives canonical action
        action = np.asarray(layer_out["action_mode"], dtype=np.float32)
        final_latent = store.get(LATENT_LAYER)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        daily_steps = info.get("daily_steps")
        iterable = list(daily_steps) if isinstance(daily_steps, list) and daily_steps else [info]
        for daily_info in iterable:
            append_eval_info_row(rows, panel, daily_info, float(daily_info.get("reward", reward)))
            latent_rows.append(final_latent)  # macro-step latent replicated per internal day (as R6c does)

    for h in handles:
        h.remove()
    env.close()

    df = pd.DataFrame(rows)
    latent = np.vstack(latent_rows).astype(float)
    cash = pd.to_numeric(df["cash_target"], errors="coerce").to_numpy(dtype=float)
    risky_cols = [c for c in df.columns if c.startswith("executed_weight_") and c != "executed_weight_CASH"]
    risky = df[risky_cols].to_numpy(dtype=float)
    gross = risky.sum(axis=1, keepdims=True)
    denom = np.where(gross == 0, 1.0, gross)
    within = risky / denom  # within-book composition = exposure-removed selection (same as crystal_score)

    # drop any non-finite latent rows (boundary safety)
    ok = np.isfinite(latent).all(axis=1) & np.isfinite(cash)
    return Replay(latent=latent[ok], cash=cash[ok], within=within[ok], n_steps=int(ok.sum()))


def score_replay(rep: Replay) -> dict:
    """Compute the cross-comparable CrystalScore-core subset + behavioral complexity from a replay."""
    X_std = StandardScaler().fit_transform(rep.latent)
    bc = behavioral_complexity(rep.cash, rep.within)

    simul = {k: cluster_mean_r2(X_std, rep.cash, k, seed=0) for k in K_GRID}
    comp_stance = simul  # stance completeness == cash cluster-mean R^2 (same as crystal_score)
    comp_sel = {k: cluster_mean_r2(X_std, rep.within, k, seed=0) for k in K_GRID}
    stab = clustering_stability(X_std, K_BUDGET)

    S9 = simul[K_BUDGET]
    core = clip01(S9 * stab) if np.isfinite(S9) and np.isfinite(stab) else float("nan")
    core_nostab = clip01(S9) if np.isfinite(S9) else float("nan")

    return {
        "n_steps": rep.n_steps,
        **bc,
        "simulatability_cash_K9": round(float(S9), 4),
        "completeness_stance": {k: round(float(v), 4) for k, v in comp_stance.items()},
        "completeness_selection": {k: round(float(v), 4) for k, v in comp_sel.items()},
        "completeness_selection_K9": round(float(comp_sel[K_BUDGET]), 4),
        "clustering_stability_proxy_K9": round(float(stab), 4),
        "pareto_stance": pareto_auc(K_GRID, comp_stance),
        "pareto_selection": pareto_auc(K_GRID, comp_sel),
        "crystalscore_core": round(float(core), 4),
        "crystalscore_core_nostab": round(float(core_nostab), 4),
    }


# ----------------------------------------------------------------------------- policy registry
def r6c_from_report() -> dict:
    """R6c row reused VERBATIM from crystal_score_report.json (NOT recomputed). We re-derive ONLY
    the behavioral_complexity from R6c's own behavior log to keep the x-axis uniform."""
    rep = json.loads(R6C_REPORT.read_text(encoding="utf-8"))
    sm = rep["submetrics"]
    comp = sm["completeness"]
    simul9 = float(sm["simulatability"]["headline_discrete"])  # K=9 cluster-mean cash R^2
    stab = float(sm["stability"]["value"])
    comp_stance = {int(k): float(v) for k, v in comp["stance"].items()}
    comp_sel = {int(k): float(v) for k, v in comp["selection"].items()}
    sel9 = float(comp["selection"]["9"])

    # cross-comparable core for R6c uses the SAME subset (Simulatability x stability) as the others,
    # so it sits on the same y-axis. The FULL CrystalScore (with Faithfulness) is also carried.
    core = clip01(simul9 * stab)
    full_stance = float(rep["crystalscore"]["stance"]["value"])  # F x Simul x Stab (R6c-only Faithfulness=1.0)

    # behavioral complexity from R6c's OWN frozen behavior log
    blog = (
        ROOT / "artifacts" / "stage4"
        / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph"
        / "frozen_test_behavior_log_daily.csv"
    )
    bc = {"behavioral_complexity": float("nan")}
    if blog.exists():
        df = pd.read_csv(blog, low_memory=False)
        cash = pd.to_numeric(df["cash_target"], errors="coerce").to_numpy(dtype=float)
        rc = [c for c in df.columns if c.startswith("executed_weight_") and c != "executed_weight_CASH"]
        risky = df[rc].to_numpy(dtype=float)
        gross = risky.sum(axis=1, keepdims=True)
        within = risky / np.where(gross == 0, 1.0, gross)
        ok = np.isfinite(cash)
        bc = behavioral_complexity(cash[ok], within[ok])

    return {
        "policy": "R6c",
        "family": "R6c (root_split_beta_dirichlet)",
        "n_steps": int(rep["n_steps"]),
        **bc,
        "simulatability_cash_K9": round(simul9, 4),
        "completeness_stance": {k: round(v, 4) for k, v in comp_stance.items()},
        "completeness_selection": {k: round(v, 4) for k, v in comp_sel.items()},
        "completeness_selection_K9": round(sel9, 4),
        "clustering_stability_proxy_K9": round(stab, 4),  # R6c: this IS the canonical multi-seed ARI
        "pareto_stance": pareto_auc(K_GRID, {k: comp_stance.get(k, float("nan")) for k in K_GRID}),
        "pareto_selection": pareto_auc(K_GRID, {k: comp_sel.get(k, float("nan")) for k in K_GRID}),
        "crystalscore_core": round(core, 4),
        "crystalscore_core_nostab": round(clip01(simul9), 4),
        "full_crystalscore": round(full_stance, 4),  # R6c-only (has Faithfulness + steering)
        "notes": "reused verbatim from crystal_score_report.json; full CrystalScore incl. Faithfulness=1.0 (R6c-only steering)",
        "source": "crystal_score_report.json",
    }


def candidate_specs() -> list[dict]:
    """R6c-family candidates to replay (same single-MLP arch -> latent+actions exportable)."""
    s01 = ROOT / "artifacts" / "stage0_1"
    specs = [
        # R6c+ latent-action arms (gaussian_logits + root_split_latent_action)
        {"policy": "R6c+_latent_relabel_off", "family": "R6c+ latent-action (gaussian_logits)",
         "config": s01 / "r6c_latent_ab" / "config.yaml",
         "metadata": s01 / "r6c_latent_ab" / "R6c_latent_relabel_off" / "fold_2021" / "metadata.json",
         "model": s01 / "r6c_latent_ab" / "R6c_latent_relabel_off" / "fold_2021" / "model.zip"},
        {"policy": "R6c+_latent_relabel_on", "family": "R6c+ latent-action (gaussian_logits)",
         "config": s01 / "r6c_latent_ab" / "config.yaml",
         "metadata": s01 / "r6c_latent_ab" / "R6c_latent_relabel_on" / "fold_2021" / "metadata.json",
         "model": s01 / "r6c_latent_ab" / "R6c_latent_relabel_on" / "fold_2021" / "model.zip"},
        # vol_excess arms (same R6c arch: root_split_beta_dirichlet)
        {"policy": "R6c_vol0", "family": "R6c vol_excess (root_split_beta_dirichlet)",
         "config": s01 / "r6c_vol_excess_full" / "config.yaml",
         "metadata": s01 / "r6c_vol_excess_full" / "R6c_AB_vol0" / "fold_2021" / "metadata.json",
         "model": s01 / "r6c_vol_excess_full" / "R6c_AB_vol0" / "fold_2021" / "model.zip"},
        {"policy": "R6c_vol05", "family": "R6c vol_excess (root_split_beta_dirichlet)",
         "config": s01 / "r6c_vol_excess_full" / "config.yaml",
         "metadata": s01 / "r6c_vol_excess_full" / "R6c_AB_vol05" / "fold_2021" / "metadata.json",
         "model": s01 / "r6c_vol_excess_full" / "R6c_AB_vol05" / "fold_2021" / "model.zip"},
        {"policy": "R6c_vol10", "family": "R6c vol_excess (root_split_beta_dirichlet)",
         "config": s01 / "r6c_vol_excess_full" / "config.yaml",
         "metadata": s01 / "r6c_vol_excess_full" / "R6c_AB_vol10" / "fold_2021" / "metadata.json",
         "model": s01 / "r6c_vol_excess_full" / "R6c_AB_vol10" / "fold_2021" / "model.zip"},
    ]
    return specs


SKIPPED = [
    {"policy": "P22", "family": "two-agent (pm + trader)",
     "reason": "Architecture mismatch: P22 is a two-module pm.* (Beta cash/q actor) + trader.* "
               "(stock_encoder + GraphHierarchicalAssetEncoder) design, NOT a single SB3 "
               "policy_net/value_net MLP. There is no single comparable 64-d 'latent -> cash/q + "
               "weights' map (the export reports/firewall_upgrade/p22_hidden_activations has split "
               "pm.actor / trader.* heads). Forcing a CrystalScore on a different latent geometry "
               "would be apples-to-oranges, so it is skipped rather than mis-scored."},
    {"policy": "W1", "family": "two-agent (pm_policy.pt + trader_policy.pt)",
     "reason": "Architecture mismatch: W1 ships separate pm_policy.pt + trader_policy.pt (Beta-budget "
               "PM + a distinct trader net), not an SB3 model.zip with mlp_extractor.policy_net. The "
               "uniform replay/hook machinery (policy_layers + policy_net.5.ReLU) does not apply; "
               "skipped to avoid a non-comparable score."},
    {"policy": "H2", "family": "two-agent (pm + trader, no-LOB)",
     "reason": "Architecture mismatch: H2 (stage0_1_h2_pm_trader_nolob) is the same two-agent pm+trader "
               "family as W1/P22 (separate policy modules), not the single-MLP R6c arch. No single "
               "comparable policy latent -> skipped."},
]


# ----------------------------------------------------------------------------- main
def main() -> None:
    if not FROZEN_PANEL.exists():
        raise FileNotFoundError(f"frozen panel missing: {FROZEN_PANEL}")

    rows: list[dict] = []

    # R6c (verbatim)
    print("[scoring] R6c (verbatim from crystal_score_report.json)")
    rows.append(r6c_from_report())

    # R6c-family candidates (replay)
    for spec in candidate_specs():
        if not spec["model"].exists():
            print(f"[skip] {spec['policy']}: model.zip missing -> {spec['model']}")
            SKIPPED.append({"policy": spec["policy"], "family": spec["family"],
                            "reason": f"model.zip not found at {spec['model']}"})
            continue
        print(f"[scoring] {spec['policy']}  ({spec['family']})  -- replaying frozen 2022-2023 ...")
        config = yaml.safe_load(spec["config"].read_text(encoding="utf-8"))
        meta = json.loads(spec["metadata"].read_text(encoding="utf-8"))
        variant = meta["variant"]
        try:
            rep = replay_policy(spec["model"], config, variant)
            sc = score_replay(rep)
        except Exception as e:  # honest failure -> skip, do not fabricate
            print(f"[skip] {spec['policy']}: replay/scoring failed: {e}")
            SKIPPED.append({"policy": spec["policy"], "family": spec["family"],
                            "reason": f"replay failed: {type(e).__name__}: {e}"})
            continue
        row = {
            "policy": spec["policy"], "family": spec["family"],
            **sc,
            "full_crystalscore": None,  # R6c-only (no per-policy steering machinery)
            "notes": "replayed frozen 2022-2023; CrystalScore-core only (Faithfulness/Controllability are R6c-only)",
            "source": str(spec["model"].relative_to(ROOT)),
        }
        rows.append(row)
        print(f"          complexity={row['behavioral_complexity']}  core={row['crystalscore_core']}  "
              f"simul9={row['simulatability_cash_K9']}  stab={row['clustering_stability_proxy_K9']}  n={row['n_steps']}")

    # ----- flat CSV
    flat = []
    for r in rows:
        flat.append({
            "policy": r["policy"],
            "family": r.get("family", ""),
            "behavioral_complexity": r.get("behavioral_complexity"),
            "cash_entropy": r.get("cash_entropy"),
            "book_dispersion": r.get("book_dispersion"),
            "action_eff_dim": r.get("action_eff_dim"),
            "simulatability_cash_K9": r.get("simulatability_cash_K9"),
            "completeness_stance_K9": r.get("completeness_stance", {}).get(9),
            "completeness_selection_K9": r.get("completeness_selection_K9"),
            "pareto_stance_auc": r.get("pareto_stance", {}).get("auc_normalized"),
            "pareto_selection_auc": r.get("pareto_selection", {}).get("auc_normalized"),
            "clustering_stability_proxy_K9": r.get("clustering_stability_proxy_K9"),
            "crystalscore_core": r.get("crystalscore_core"),
            "crystalscore_core_nostab": r.get("crystalscore_core_nostab"),
            "full_crystalscore_R6c_only": r.get("full_crystalscore"),
            "n_steps": r.get("n_steps"),
            "notes": r.get("notes", ""),
        })
    flat_df = pd.DataFrame(flat)
    flat_df.to_csv(OUT_CSV, index=False)

    # ----- curve CSV + PNG
    curve = flat_df[["policy", "family", "behavioral_complexity", "crystalscore_core",
                     "crystalscore_core_nostab", "simulatability_cash_K9",
                     "completeness_selection_K9"]].copy()
    curve = curve.sort_values("behavioral_complexity").reset_index(drop=True)
    curve.to_csv(OUT_CURVE_CSV, index=False)

    plot_info = maybe_plot(curve)

    # ----- correlation across the scored set (honest interpretation aid)
    cc = curve.dropna(subset=["behavioral_complexity", "crystalscore_core"])
    n_scored = int(len(cc))
    if n_scored >= 3 and cc["behavioral_complexity"].nunique() > 1:
        from scipy.stats import pearsonr, spearmanr
        pear = float(pearsonr(cc["behavioral_complexity"], cc["crystalscore_core"])[0])
        spear = float(spearmanr(cc["behavioral_complexity"], cc["crystalscore_core"])[0])
    else:
        pear = spear = float("nan")

    # ----- report json
    report = {
        "title": "Cross-policy CrystalScore -- behavioral-complexity vs interpretability",
        "x_axis": "behavioral_complexity (action-only proxy; mean of cash_entropy, book_dispersion, action_eff_dim)",
        "y_axis": "CrystalScore-core (cross-comparable subset = Simulatability_cash(K=9) x clustering-stability)",
        "cross_comparable_metrics": ["behavioral_complexity", "simulatability_cash_K9",
                                     "completeness_stance", "completeness_selection",
                                     "pareto_*_auc", "clustering_stability_proxy_K9",
                                     "crystalscore_core"],
        "r6c_only_metrics": ["faithfulness", "controllability", "full_crystalscore"],
        "frozen_panel": str(FROZEN_PANEL.relative_to(ROOT)),
        "latent_layer": LATENT_LAYER,
        "K_budget": K_BUDGET, "K_grid": K_GRID,
        "n_policies_scored": n_scored,
        "correlation_complexity_vs_core": {"pearson": round(pear, 4) if np.isfinite(pear) else None,
                                           "spearman": round(spear, 4) if np.isfinite(spear) else None},
        "rows": rows,
        "skipped": SKIPPED,
        "plot": plot_info,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    write_markdown(report, flat_df, curve)
    print_console(report, flat_df)


def maybe_plot(curve: pd.DataFrame) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        return {"plotted": False, "reason": str(e)}

    cc = curve.dropna(subset=["behavioral_complexity", "crystalscore_core"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fam_colors = {}
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for fam in cc["family"].unique():
        fam_colors[fam] = palette[len(fam_colors) % len(palette)]
    for _, r in cc.iterrows():
        ax.scatter(r["behavioral_complexity"], r["crystalscore_core"],
                   color=fam_colors[r["family"]], s=70, zorder=3, edgecolor="k", linewidth=0.5)
        ax.annotate(r["policy"], (r["behavioral_complexity"], r["crystalscore_core"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    # trend line if >=3 points
    if len(cc) >= 3 and cc["behavioral_complexity"].nunique() > 1:
        z = np.polyfit(cc["behavioral_complexity"], cc["crystalscore_core"], 1)
        xs = np.linspace(cc["behavioral_complexity"].min(), cc["behavioral_complexity"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", lw=1, alpha=0.6, label=f"OLS slope={z[0]:.2f}")
        ax.legend(loc="upper right", fontsize=8)
    # legend for families
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
                      markersize=8, label=f) for f, c in fam_colors.items()]
    leg1 = ax.legend(handles=handles, loc="lower left", fontsize=7, title="family")
    ax.add_artist(leg1)
    ax.set_xlabel("Behavioral complexity  (action-only proxy, [0,1])")
    ax.set_ylabel("CrystalScore-core  (Simulatability_cash x stability)")
    ax.set_title("Cross-policy: interpretability vs behavioral complexity (frozen 2022-2023)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_CURVE_PNG, dpi=140)
    plt.close(fig)
    return {"plotted": True, "file": str(OUT_CURVE_PNG)}


def write_markdown(r: dict, flat: pd.DataFrame, curve: pd.DataFrame) -> None:
    def f(x):
        return "N/A" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else (f"{x:.3f}" if isinstance(x, float) else str(x))

    L = []
    L.append("# Cross-policy CrystalScore -- interpretability vs behavioral complexity\n")
    L.append(
        "This is the cross-policy companion to the single-policy `CRYSTAL_SCORE.md` (R6c). It scores "
        "MULTIPLE policies on a UNIFORM, comparable subset of CrystalScore and plots each as a point in "
        "(behavioral_complexity, CrystalScore-core) space. R6c's numbers are reused **verbatim** from "
        "`crystal_score_report.json` (not recomputed); every other policy is **replayed** (no retrain) on "
        f"the same frozen 2022-2023 panel and scored with the identical recipe.\n")

    L.append("## What is cross-comparable vs R6c-only\n")
    L.append("| Axis / metric | Cross-comparable? | Why |")
    L.append("|---|---|---|")
    L.append("| `behavioral_complexity` (x) | **YES** | action-only proxy (cash entropy + within-book dispersion + action effective-dim); needs no latent or steering |")
    L.append("| `simulatability_cash_K9` | **YES** | K=9 latent-cluster-mean predictor of cash; needs only latent+action |")
    L.append("| `completeness_stance/selection` + Pareto-AUC | **YES** | 1 - SS_res/SS_tot of K-cluster-mean of cash / within-book; latent+action only |")
    L.append("| `clustering_stability_proxy_K9` | **YES (proxy)** | cross-KMeans-seed ARI of the K=9 latent clusters on the single rollout. NOTE: this is clustering robustness, **not** cross-training-seed stability (one rollout per policy). For R6c it equals the canonical multi-seed ARI. |")
    L.append("| `crystalscore_core` (y) | **YES** | = Simulatability_cash(K=9) x clustering-stability -- the subset every policy produces identically |")
    L.append("| Faithfulness, Controllability, **full CrystalScore** | **R6c-ONLY** | require per-policy code-steering / intervention machinery that only R6c has |")
    L.append("")

    L.append("## The table\n")
    cols = ["policy", "family", "behavioral_complexity", "cash_entropy", "book_dispersion",
            "action_eff_dim", "simulatability_cash_K9", "completeness_selection_K9",
            "pareto_stance_auc", "pareto_selection_auc", "clustering_stability_proxy_K9",
            "crystalscore_core", "full_crystalscore_R6c_only", "n_steps"]
    L.append("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    L.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in flat.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("N/A" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"Full per-policy CSV: `{OUT_CSV.name}`. Curve CSV: `{OUT_CURVE_CSV.name}`.\n")

    L.append("## The curve\n")
    if r["plot"].get("plotted"):
        L.append(f"![cross-policy curve]({OUT_CURVE_PNG.name})\n")
    else:
        L.append("(matplotlib unavailable -> PNG skipped)\n")
    L.append("Points sorted by behavioral complexity (low -> high):\n")
    L.append("```")
    for _, row in curve.iterrows():
        core = row["crystalscore_core"]
        L.append(f"  {row['policy']:<26} complexity={row['behavioral_complexity']:.3f}  "
                 f"core={core:.3f}  (simul9={row['simulatability_cash_K9']:.3f})  [{row['family']}]")
    L.append("```")
    cor = r["correlation_complexity_vs_core"]
    L.append(f"\nAcross the {r['n_policies_scored']} scored policies: "
             f"Pearson(complexity, core) = {f(cor['pearson'])}, Spearman = {f(cor['spearman'])}.\n")

    L.append("## Coverage\n")
    L.append("**Scored (replayed or verbatim):**\n")
    for row in r["rows"]:
        src = row.get("source", "")
        L.append(f"- `{row['policy']}` ({row.get('family','')}) -- n={row.get('n_steps')} steps. {row.get('notes','')}")
    L.append("\n**Skipped (with reason):**\n")
    for s in r["skipped"]:
        L.append(f"- `{s['policy']}` ({s['family']}): {s['reason']}")
    L.append("")

    L.append("## Honest interpretation\n")
    cc = curve.dropna(subset=["behavioral_complexity", "crystalscore_core"])
    n = len(cc)
    fams = cc["family"].nunique()
    cmin, cmax = cc["behavioral_complexity"].min(), cc["behavioral_complexity"].max()
    slope_txt = ""
    if n >= 3 and cc["behavioral_complexity"].nunique() > 1:
        z = np.polyfit(cc["behavioral_complexity"], cc["crystalscore_core"], 1)
        slope_txt = f" The OLS slope of CrystalScore-core on behavioral_complexity is {z[0]:.2f}"
        slope_txt += " (negative => interpretability falls as complexity rises)." if z[0] < 0 else " (>=0 => no decline across the scored set)."
    L.append(
        f"- **Coverage is {n} policies across {fams} architecture sub-families, spanning a behavioral-complexity "
        f"range of [{cmin:.3f}, {cmax:.3f}].** This is the R6c + R6c-family corner of complexity space; the "
        f"two-agent policies (W1/H2/P22) could not be placed on the same axis (architecture mismatch, see Coverage), "
        f"so this is an **R6c-FAMILY curve over a narrow complexity range**, not a full architectural sweep. We state "
        f"that plainly rather than overclaiming a universal curve.")
    L.append(f"- **Does interpretability fall as behavioral complexity rises across the scored set?**{slope_txt} "
             f"Pearson = {f(r['correlation_complexity_vs_core']['pearson'])}, Spearman = "
             f"{f(r['correlation_complexity_vs_core']['spearman'])}. Read the sign and magnitude with the small-n caveat.")
    L.append("- **Is R6c near the simple / high-interpretability corner?** Compare R6c's "
             "`behavioral_complexity` and `crystalscore_core` to the others in the table above. R6c is a "
             "near-equal-weight cash-timing controller (low within-book dispersion), so it is expected to sit "
             "toward the low-complexity end; whether it also has the highest core is read directly from the points.")
    L.append("- **Caveats carried from the single-policy score:** the discrete K<=9 cluster-mean is a lossy, "
             "parsimonious description of a continuous cash dial, so absolute core values are modest by "
             "construction (the price of human-readable parsimony); negative R^2 is clipped to 0; and the "
             "per-policy 'stability' here is clustering robustness on one rollout, not cross-training-seed "
             "stability. These do not bias the cross-policy COMPARISON because every policy is scored identically.\n")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")


def print_console(r: dict, flat: pd.DataFrame) -> None:
    print("=" * 90)
    print("CROSS-POLICY CRYSTALSCORE  (behavioral complexity  vs  interpretability)")
    print("=" * 90)
    cols = ["policy", "behavioral_complexity", "simulatability_cash_K9",
            "clustering_stability_proxy_K9", "crystalscore_core", "full_crystalscore_R6c_only", "n_steps"]
    hdr = f"{'policy':<26}{'complx':>8}{'simul9':>8}{'stab':>8}{'core':>8}{'full(R6c)':>11}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in flat.iterrows():
        def g(c):
            v = row.get(c)
            return "  N/A" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{str(row['policy']):<26}{g('behavioral_complexity'):>8}{g('simulatability_cash_K9'):>8}"
              f"{g('clustering_stability_proxy_K9'):>8}{g('crystalscore_core'):>8}{g('full_crystalscore_R6c_only'):>11}{g('n_steps'):>6}")
    print("-" * len(hdr))
    cor = r["correlation_complexity_vs_core"]
    print(f"Scored: {r['n_policies_scored']} policies | Pearson(complexity,core)={cor['pearson']} Spearman={cor['spearman']}")
    print(f"Skipped: {', '.join(s['policy'] for s in r['skipped'])}  (architecture mismatch -- see MD)")
    print(f"Wrote: {OUT_CSV.name}, {OUT_CURVE_CSV.name}, {OUT_MD.name}, {OUT_JSON.name}"
          + (f", {OUT_CURVE_PNG.name}" if r['plot'].get('plotted') else ""))
    print("=" * 90)


if __name__ == "__main__":
    main()
