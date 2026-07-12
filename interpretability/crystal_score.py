"""CrystalScore v1 for R6c -- the first numeric interpretability score for the
Interpretable-CHRL ("CrystalRL") policy.

WHAT THIS DOES
--------------
Computes, from EXISTING Stage-0-9 frozen-2022-2023 artifacts only (no retrain, no
new rollout), a panel of interpretability sub-metrics, a scalar CrystalScore, and
two diagnostic curves:

  sub-metrics (each in [0,1], negative R^2 clipped to 0):
    FAITHFULNESS     -- do the code steering curves move cash in their LABELED
                        direction, monotonically in alpha?
    SIMULATABILITY   -- can a parsimonious surrogate predict the policy's action
                        from the interpretable description?  (a) latent->cash
                        continuous ceiling (cv_r2_shuffled); (b) K-code cluster-mean
                        predictor of per-date cash  [HEADLINE = (b)].
    COMPLETENESS@K   -- 1 - SS_res/SS_tot of the K-cluster-mean predictor of the
                        action, computed separately for STANCE (cash) and SELECTION
                        (risky-weight vector).  Swept over K.
    CONTROLLABILITY  -- steering success x specificity x monotonicity, using the
                        EMPIRICAL controllability (commanded-vs-empirical cash
                        Spearman ~ 0.20), reported separately for cash and selection.
    STABILITY        -- cross-seed agreement (ARI) of the code clusters.

  scalars:
    CrystalScore = Faithfulness x Simulatability x Stability  at K<=9,
    computed SEPARATELY for STANCE and for SELECTION.  Controllability is reported
    as a co-equal axis (NOT multiplied in), because the honest reading is that the
    commanded dial is near-tautological.

  curves:
    Pareto     : x = K, y = completeness@K  (parsimony vs completeness) + AUC + @9.
    Complexity : (behavioral-complexity-captured, CrystalScore) approximated by
                 varying K within this ONE policy (a placeholder for the true
                 cross-policy curve, stated honestly).

HONESTY RULES (enforced, not decorative)
----------------------------------------
  * If an input is missing, the metric is marked N/A and excluded from the scalar.
  * Negative R^2 -> 0.
  * Controllability uses the EMPIRICAL (not commanded) cash Spearman.
  * Selection metrics are reported even though they are ~0 -- that is the finding.
  * STABILITY is taken from the multi-seed ARI in the code-layer manifest; if it
    were absent it would be N/A (we DO have it, so it is used; a no-stability
    CrystalScore is also reported for comparison).

Run:
    python interpretability/crystal_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console
except Exception:
    pass

# ----------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FW = ROOT / "reports" / "firewall_upgrade"
PROBE = FW / "r6c_latent_probe" / "latent_decodability.csv"
CODE_DICT = FW / "r6c_code_layer" / "r6c_code_dictionary.csv"
CODE_ASSIGN = FW / "r6c_code_layer" / "r6c_code_assignments.csv"
CRISP_SEARCH = FW / "r6c_code_layer" / "crisp_primitives_search.csv"
CODE_MANIFEST = FW / "r6c_code_layer" / "r6c_code_layer_manifest.json"
STEER = FW / "r6c_code_control" / "code_steering_curve.csv"
CTRL_REPORT = FW / "r6c_code_control" / "code_control_report.json"

STAGE4 = (
    ROOT
    / "artifacts"
    / "stage4"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph"
)
HIDDEN_NPZ = STAGE4 / "hidden_activations" / "r6c_frozen_hidden_activations.npz"
BEHAVIOR_LOG = STAGE4 / "frozen_test_behavior_log_daily.csv"

OUT_JSON = HERE / "crystal_score_report.json"
OUT_MD = HERE / "CRYSTAL_SCORE.md"
OUT_PARETO_CSV = HERE / "crystal_pareto_curve.csv"
OUT_COMPLEXITY_CSV = HERE / "crystal_complexity_curve.csv"
OUT_PARETO_PNG = HERE / "crystal_pareto_curve.png"
OUT_COMPLEXITY_PNG = HERE / "crystal_complexity_curve.png"

LATENT_LAYER = "policy_net.5.ReLU"  # TRUE 64-d final policy latent
K_BUDGET = 9  # human-readable parsimony budget for the headline CrystalScore
SEEDS = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------- helpers
def clip01(x: float) -> float:
    """Clip a value into [0,1]; non-finite -> nan stays nan."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def cluster_mean_r2(X_latent_std: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> float:
    """Completeness reading: cluster the latent into k groups, predict y by the
    per-cluster MEAN of y, return 1 - SS_res/SS_tot (negative clipped to 0).

    y may be 1-D (stance) or 2-D (selection vector); SS is summed over all columns
    so the multivariate completeness is a single scalar (variance-weighted, i.e. it
    naturally downweights the near-constant equal-weight book)."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=500).fit(X_latent_std)
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    r, _ = spearmanr(a, b)
    return float(r)


# ----------------------------------------------------------------------------- loaders
def load_inputs() -> dict:
    """Load every artifact; record presence so missing inputs become N/A, not fiction."""
    present = {}
    data = {}

    # latent + action (the only place the per-date 64-d latent and executed weights live)
    if HIDDEN_NPZ.exists() and BEHAVIOR_LOG.exists():
        z = np.load(HIDDEN_NPZ, allow_pickle=True)
        dates = np.asarray(z["dates"], dtype=str)
        latent = z[LATENT_LAYER].astype(float)  # (N, 64)
        df = pd.read_csv(BEHAVIOR_LOG, low_memory=False)
        df = df.set_index("date").reindex(dates).reset_index().rename(columns={"index": "date"})
        risky_cols = [c for c in df.columns if c.startswith("executed_weight_") and c != "executed_weight_CASH"]
        risky = df[risky_cols].to_numpy(dtype=float)  # (N, 29) raw executed risky weights
        cash_stance = df["cash_target"].to_numpy(dtype=float)  # policy's raw cash stance
        data["dates"] = dates
        data["latent"] = latent
        data["risky_raw"] = risky  # raw weights: variance dominated by GROSS EXPOSURE (a stance restatement)
        data["risky_cols"] = risky_cols
        data["cash_stance"] = cash_stance
        # within-book composition = renormalised weights -> isolates TRUE SELECTION
        # (removes the gross long-exposure component, which is really 1 - cash i.e. STANCE).
        gross = risky.sum(axis=1, keepdims=True)
        denom = np.where(gross == 0, 1.0, gross)
        within = risky / denom  # (N,29) sums to 1 -> pure composition
        data["within_book"] = within  # <-- this is what "SELECTION" means (no exposure leakage)
        data["risky_hhi"] = (within ** 2).sum(axis=1)  # within-book concentration
        # diagnostic: how much of raw-W variance is just the gross-exposure scalar?
        gss = float(((gross[:, 0] - gross[:, 0].mean()) ** 2).sum())
        wss = float(((risky - risky.mean(axis=0)) ** 2).sum())
        data["exposure_var_share"] = gss / wss if wss > 0 else float("nan")
        present["latent_and_action"] = True
    else:
        present["latent_and_action"] = False

    # decodability table
    if PROBE.exists():
        data["decode"] = pd.read_csv(PROBE)
        present["decodability"] = True
    else:
        present["decodability"] = False

    # code dictionary + assignments
    if CODE_DICT.exists():
        data["code_dict"] = pd.read_csv(CODE_DICT)
        present["code_dict"] = True
    else:
        present["code_dict"] = False
    if CODE_ASSIGN.exists():
        data["code_assign"] = pd.read_csv(CODE_ASSIGN)
        present["code_assign"] = True
    else:
        present["code_assign"] = False

    # crisp K-sweep (parsimony axis)
    if CRISP_SEARCH.exists():
        data["crisp"] = pd.read_csv(CRISP_SEARCH)
        present["crisp_search"] = True
    else:
        present["crisp_search"] = False

    # steering + control report
    if STEER.exists():
        data["steer"] = pd.read_csv(STEER)
        present["steering"] = True
    else:
        present["steering"] = False
    if CTRL_REPORT.exists():
        data["ctrl"] = json.loads(CTRL_REPORT.read_text())
        present["control_report"] = True
    else:
        present["control_report"] = False

    # code-layer manifest (multi-seed ARI -> STABILITY)
    if CODE_MANIFEST.exists():
        data["manifest"] = json.loads(CODE_MANIFEST.read_text())
        present["code_manifest"] = True
    else:
        present["code_manifest"] = False

    data["_present"] = present
    return data


# ----------------------------------------------------------------------------- metrics
def metric_faithfulness(data: dict) -> dict:
    """Fraction of steered codes whose commanded_cash moves in its LABELED direction
    monotonically in alpha.  to_high_cash => cash should INCREASE with alpha;
    to_low_cash => cash should DECREASE."""
    if not data["_present"]["steering"]:
        return {"value": float("nan"), "note": "N/A: steering curve missing", "detail": {}}
    steer = data["steer"]
    detail = {}
    hits = 0
    n = 0
    for (code, tag), g in steer.groupby(["target_code", "tag"]):
        g = g.sort_values("alpha")
        rho = spearman(g["alpha"].to_numpy(), g["commanded_cash"].to_numpy())
        want_up = "high_cash" in tag
        ok = (rho > 0) if want_up else (rho < 0)
        # monotonic in the labeled direction == |rho| == 1 with correct sign
        mono = abs(rho) >= 0.999
        match = bool(ok and mono)
        detail[f"code_{code}_{tag}"] = {
            "alpha_cash_spearman": round(rho, 4),
            "labeled_direction": "up" if want_up else "down",
            "sign_match": bool(ok),
            "monotonic": bool(mono),
            "pass": match,
        }
        hits += int(match)
        n += 1
    val = clip01(hits / n) if n else float("nan")
    return {
        "value": val,
        "note": f"{hits}/{n} steered codes move cash in their labeled direction monotonically",
        "detail": detail,
    }


def metric_simulatability(data: dict, X_std: np.ndarray | None) -> dict:
    """(a) continuous ceiling = latent->cash cv_r2_shuffled from the probe table.
    (b) HEADLINE = K-code cluster-mean predictor R^2 of per-date cash stance (K<=9)."""
    out = {"continuous_ceiling": None, "headline_discrete": None, "note": ""}

    # (a) continuous ceiling
    if data["_present"]["decodability"]:
        d = data["decode"]
        row = d[(d["latent"] == "final_64") & (d["target"] == "cash_target")]
        if len(row):
            out["continuous_ceiling"] = clip01(float(row["cv_r2_shuffled"].iloc[0]))

    # (b) discrete headline -- K-cluster-mean predictor of cash at the budget
    if X_std is not None and data["_present"]["latent_and_action"]:
        cash = data["cash_stance"]
        out["headline_discrete"] = cluster_mean_r2(X_std, cash, K_BUDGET, seed=0)
        out["note"] = (
            f"headline = {K_BUDGET}-cluster-mean predictor of cash_target (discrete/parsimony reading); "
            "continuous_ceiling = final_64->cash cv_r2_shuffled"
        )
    else:
        out["note"] = "N/A discrete: latent/action missing"

    out["value"] = out["headline_discrete"]  # the headline used in the scalar
    return out


def metric_completeness(data: dict, X_std: np.ndarray | None, ks: list[int]) -> dict:
    """completeness@K for STANCE (cash) and SELECTION (within-book composition).

    HONESTY NOTE (critical): the RAW 29-d executed weight vector is the WRONG target
    for "selection" -- its cross-date variance is dominated by GROSS LONG EXPOSURE
    (the book-sum = 1 - cash), which is just the STANCE restated 29 times. We verified
    the gross-exposure scalar carries ~28x the de-meaned raw-W variance, and raw-W
    completeness (0.34) tracks the gross-sum completeness (0.347), NOT selection.
    So SELECTION is measured on the WITHIN-BOOK COMPOSITION (renormalised weights),
    which removes the exposure component and isolates genuine stock tilts. We ALSO
    report raw-W completeness, clearly labeled as exposure-confounded, for transparency."""
    if X_std is None or not data["_present"]["latent_and_action"]:
        return {"stance": {}, "selection": {}, "selection_raw_confounded": {}, "note": "N/A: latent/action missing"}
    cash = data["cash_stance"]
    within = data["within_book"]
    raw = data["risky_raw"]
    stance, selection, selection_raw = {}, {}, {}
    for k in ks:
        stance[k] = cluster_mean_r2(X_std, cash, k, seed=0)
        selection[k] = cluster_mean_r2(X_std, within, k, seed=0)
        selection_raw[k] = cluster_mean_r2(X_std, raw, k, seed=0)
    return {
        "stance": stance,
        "selection": selection,  # within-book composition = TRUE selection (headline)
        "selection_raw_confounded": selection_raw,  # raw weights = exposure+composition (diagnostic only)
        "exposure_var_share_of_rawW": round(float(data.get("exposure_var_share", float("nan"))), 3),
        "note": (
            "stance=cash_target; SELECTION=within-book composition (renormalised, exposure removed); "
            "selection_raw_confounded=raw 29-d weights (variance dominated by gross exposure ~ stance, "
            "reported only to show the confound)."
        ),
    }


def metric_controllability(data: dict) -> dict:
    """EMPIRICAL controllability, split into cash and selection.

    cash_controllability   = steering_success x monotonicity x empirical_fidelity
        steering_success     : codes command DISTINCT cash actions (spread > 0)
        monotonicity         : mean |alpha->cash spearman| over steered codes
        empirical_fidelity   : commanded_vs_empirical_cash_spearman (~0.20)  <-- the honest discount
    selection_controllability= 0  (commanded_risky_hhi is FLAT across alpha and codes
                                    => steering cash does NOT move selection => selection
                                    is not controllable; specificity is degenerate)."""
    if not data["_present"]["control_report"]:
        return {"cash": float("nan"), "selection": float("nan"), "note": "N/A: control report missing"}
    ctrl = data["ctrl"]
    mono = ctrl.get("steering_monotonicity_spearman", {})
    mono_abs = np.mean([abs(v) for v in mono.values()]) if mono else float("nan")
    emp = float(ctrl.get("commanded_vs_empirical_cash_spearman", float("nan")))
    spread = float(ctrl.get("cash_command_spread_across_codes", float("nan")))
    steering_success = 1.0 if (np.isfinite(spread) and spread > 0) else 0.0

    # specificity of cash control: does forcing cash leave selection (risky_hhi) flat?
    # In the steering curve, commanded_risky_hhi is constant -> cash control is SPECIFIC
    # to cash (good for cash) but means selection cannot be steered (bad for selection).
    selection_steerable = 0.0
    hhi_flat = None
    if data["_present"]["steering"]:
        steer = data["steer"]
        hhi_unique = steer["commanded_risky_hhi"].round(6).nunique()
        hhi_flat = bool(hhi_unique <= 1)
        selection_steerable = 0.0 if hhi_flat else 1.0  # flat => not steerable

    cash_ctrl = clip01(steering_success * mono_abs * clip01(emp))
    # selection controllability: even if a "command" existed, the dial cannot move it
    selection_ctrl = clip01(selection_steerable)  # = 0 when hhi is flat

    return {
        "cash": cash_ctrl,
        "selection": selection_ctrl,
        "components": {
            "steering_success": steering_success,
            "monotonicity_mean_abs": round(float(mono_abs), 4) if np.isfinite(mono_abs) else None,
            "empirical_fidelity_spearman": round(emp, 4),
            "commanded_dial_spread": round(spread, 4),
            "selection_hhi_flat_across_steering": hhi_flat,
        },
        "note": (
            "cash uses EMPIRICAL fidelity (commanded-vs-empirical cash spearman ~0.20), NOT the "
            "near-tautological commanded dial. selection=0 because commanded_risky_hhi is FLAT "
            "(cash is controllable, selection is NOT)."
        ),
    }


def metric_stability(data: dict, X_std: np.ndarray | None) -> dict:
    """Cross-seed agreement (ARI) of the code clusters at the budget K.

    Primary source = the multi-seed stability_ari logged in the code-layer manifest
    (the canonical codebook).  We ALSO recompute a fresh multi-seed ARI at K_BUDGET
    on the standardized latent as an independent confirmation.  If neither existed,
    STABILITY would be N/A (we do NOT fabricate)."""
    manifest_ari = None
    if data["_present"]["code_manifest"]:
        for row in data["manifest"].get("k_selection", []):
            if int(row.get("k", -1)) == K_BUDGET:
                manifest_ari = float(row.get("stability_ari"))
                break
    recomputed = None
    if X_std is not None:
        labels = []
        for s in SEEDS:
            km = KMeans(n_clusters=K_BUDGET, random_state=s, n_init=10, max_iter=500).fit(X_std)
            labels.append(km.labels_)
        aris = [
            adjusted_rand_score(labels[i], labels[j])
            for i in range(len(SEEDS))
            for j in range(i + 1, len(SEEDS))
        ]
        recomputed = float(np.mean(aris))

    value = manifest_ari if manifest_ari is not None else recomputed
    if value is None:
        return {"value": float("nan"), "note": "N/A: no multi-seed code assignments", "available": False}
    return {
        "value": clip01(value),
        "manifest_ari_at_K9": manifest_ari,
        "recomputed_ari_at_K9": round(recomputed, 4) if recomputed is not None else None,
        "available": True,
        "note": "cross-seed ARI of K=9 KMeans code clusters (manifest value is primary; recomputed is a check)",
    }


# ----------------------------------------------------------------------------- curves
def pareto_auc(ks: list[int], comp: dict) -> dict:
    xs = np.array(sorted(ks), dtype=float)
    ys = np.array([comp[int(k)] for k in xs], dtype=float)
    mask = np.isfinite(ys)
    if mask.sum() < 2:
        return {"auc_normalized": float("nan"), "at_K9": comp.get(9, float("nan")), "points": list(zip(ks, ys.tolist()))}
    xs_m, ys_m = xs[mask], ys[mask]
    try:
        from numpy import trapezoid as _trap  # numpy>=2
    except ImportError:  # pragma: no cover
        from numpy import trapz as _trap
    auc = float(_trap(ys_m, xs_m))
    auc_norm = auc / (xs_m.max() - xs_m.min())  # normalize to mean-height in [0,1]
    return {
        "auc_raw": round(auc, 4),
        "auc_normalized": round(clip01(auc_norm), 4),
        "at_K9": round(float(comp.get(9, float("nan"))), 4) if np.isfinite(comp.get(9, float("nan"))) else None,
        "points": [(int(k), round(float(v), 4)) for k, v in zip(xs, ys)],
    }


def maybe_plot(ks, stance, selection, complexity_pts_stance, complexity_pts_sel):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        return {"plotted": False, "reason": str(e)}

    ksort = sorted(ks)
    # Pareto
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ksort, [stance[k] for k in ksort], "o-", label="STANCE (cash) completeness", color="#1f77b4")
    ax.plot(ksort, [selection[k] for k in ksort], "s--", label="SELECTION (risky weights) completeness", color="#d62728")
    ax.axvline(9, color="gray", ls=":", lw=1)
    ax.set_xlabel("K  (number of codes  =  description length / parsimony budget)")
    ax.set_ylabel("Completeness@K  (1 - SS_res/SS_tot)")
    ax.set_title("R6c CrystalScore -- Completeness vs Parsimony (Pareto)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PARETO_PNG, dpi=130)
    plt.close(fig)

    # Complexity vs CrystalScore (within-policy proxy)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cx_s = [p[0] for p in complexity_pts_stance]
    cy_s = [p[1] for p in complexity_pts_stance]
    cx_sel = [p[0] for p in complexity_pts_sel]
    cy_sel = [p[1] for p in complexity_pts_sel]
    ax.plot(cx_s, cy_s, "o-", label="STANCE", color="#1f77b4")
    ax.plot(cx_sel, cy_sel, "s--", label="SELECTION", color="#d62728")
    ax.set_xlabel("Behavioral complexity captured  (completeness@K)")
    ax.set_ylabel("CrystalScore(K)  = Faith x Simul(K) x Stab(K)")
    ax.set_title("R6c (complexity, CrystalScore) -- within-policy proxy (K-sweep)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_COMPLEXITY_PNG, dpi=130)
    plt.close(fig)
    return {"plotted": True, "files": [str(OUT_PARETO_PNG), str(OUT_COMPLEXITY_PNG)]}


# ----------------------------------------------------------------------------- main
def main() -> None:
    data = load_inputs()
    present = data["_present"]

    # standardized latent for all re-clustering (matches the codebook's StandardScaler)
    X_std = None
    if present["latent_and_action"]:
        X_std = StandardScaler().fit_transform(data["latent"])

    # K-grid = the values present in the crisp parsimony search (4..12), restricted
    # to those <= a reasonable readable cap; we use exactly the manifest K-grid.
    if present["crisp_search"]:
        ks = sorted({int(k) for k in data["crisp"]["k"].unique() if 2 <= int(k) <= 12})
    else:
        ks = [4, 5, 6, 7, 8, 9, 10, 12]
    if 9 not in ks:
        ks.append(9)
    ks = sorted(set(ks))

    # ----- sub-metrics
    faith = metric_faithfulness(data)
    simul = metric_simulatability(data, X_std)
    comp = metric_completeness(data, X_std, ks)
    ctrl = metric_controllability(data)
    stab = metric_stability(data, X_std)

    # ----- per-K simulatability (for the complexity curve): K-cluster-mean cash R^2
    simul_by_k_stance = {}
    if X_std is not None:
        for k in ks:
            simul_by_k_stance[k] = cluster_mean_r2(X_std, data["cash_stance"], k, seed=0)

    # ----- CrystalScore (Faithfulness x Simulatability x Stability) at K_BUDGET
    F = faith["value"]
    S_stance = simul["headline_discrete"]
    S_sel = comp["selection"].get(K_BUDGET) if comp.get("selection") else float("nan")
    St = stab["value"]

    def crystal(F, S, St):
        vals = [v for v in (F, S, St) if v is not None and np.isfinite(v)]
        if len(vals) < 2:  # need at least Faith & Simul
            return float("nan")
        prod = 1.0
        for v in vals:
            prod *= v
        return clip01(prod)

    crystal_stance_full = crystal(F, S_stance, St)
    crystal_sel_full = crystal(F, S_sel, St)
    crystal_stance_nostab = crystal(F, S_stance, None)
    crystal_sel_nostab = crystal(F, S_sel, None)

    # ----- curves
    pareto_stance = pareto_auc(ks, comp["stance"]) if comp.get("stance") else {}
    pareto_sel = pareto_auc(ks, comp["selection"]) if comp.get("selection") else {}

    # complexity proxy: x = completeness@K (behavioral complexity captured),
    #                   y = CrystalScore(K) = F x Simul(K) x Stab
    complexity_pts_stance = []
    complexity_pts_sel = []
    if X_std is not None and np.isfinite(F):
        st_val = St if (St is not None and np.isfinite(St)) else None
        for k in ks:
            cs = crystal(F, simul_by_k_stance[k], st_val)
            complexity_pts_stance.append((round(float(comp["stance"][k]), 4), round(float(cs), 4)))
            cs2 = crystal(F, comp["selection"][k], st_val)
            complexity_pts_sel.append((round(float(comp["selection"][k]), 4), round(float(cs2), 4)))

    plot_info = {"plotted": False}
    if comp.get("stance"):
        plot_info = maybe_plot(
            ks, comp["stance"], comp["selection"], complexity_pts_stance, complexity_pts_sel
        )

    # ----- write curve CSVs
    if comp.get("stance"):
        pd.DataFrame(
            {
                "K": ks,
                "completeness_stance_cash": [comp["stance"][k] for k in ks],
                "completeness_selection_withinbook": [comp["selection"][k] for k in ks],
                "completeness_selection_rawW_exposure_confounded": [comp["selection_raw_confounded"][k] for k in ks],
                "crystalscore_stance_at_K": [
                    crystal(F, simul_by_k_stance[k], St if (St and np.isfinite(St)) else None) for k in ks
                ],
                "crystalscore_selection_at_K": [
                    crystal(F, comp["selection"][k], St if (St and np.isfinite(St)) else None) for k in ks
                ],
            }
        ).to_csv(OUT_PARETO_CSV, index=False)

        pd.DataFrame(
            {
                "K": ks,
                "complexity_captured_stance": [comp["stance"][k] for k in ks],
                "crystalscore_stance": [p[1] for p in complexity_pts_stance],
                "complexity_captured_selection": [comp["selection"][k] for k in ks],
                "crystalscore_selection": [p[1] for p in complexity_pts_sel],
            }
        ).to_csv(OUT_COMPLEXITY_CSV, index=False)

    # ----- provenance map
    provenance = {
        "FAITHFULNESS": str(STEER.relative_to(ROOT)),
        "SIMULATABILITY_continuous": str(PROBE.relative_to(ROOT)),
        "SIMULATABILITY_discrete": f"{HIDDEN_NPZ.name} + {BEHAVIOR_LOG.name}",
        "COMPLETENESS": f"{HIDDEN_NPZ.name}[{LATENT_LAYER}] + {BEHAVIOR_LOG.name}[cash_target, executed_weight_* (within-book renorm)]",
        "CONTROLLABILITY": str(CTRL_REPORT.relative_to(ROOT)) + " + " + str(STEER.relative_to(ROOT)),
        "STABILITY": str(CODE_MANIFEST.relative_to(ROOT)),
        "K_grid_parsimony": str(CRISP_SEARCH.relative_to(ROOT)),
    }

    report = {
        "title": "CrystalScore v1 -- R6c (frozen 2022-2023)",
        "policy": "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12 (fold_2021 frozen rollout)",
        "n_steps": int(len(data["dates"])) if present["latent_and_action"] else None,
        "latent_layer": LATENT_LAYER,
        "K_budget": K_BUDGET,
        "K_grid": ks,
        "inputs_present": present,
        "submetrics": {
            "faithfulness": faith,
            "simulatability": simul,
            "completeness": comp,
            "controllability": ctrl,
            "stability": stab,
        },
        "crystalscore": {
            "stance": {
                "value": crystal_stance_full,
                "value_no_stability": crystal_stance_nostab,
                "formula": "Faithfulness x Simulatability(cash,K<=9) x Stability",
                "factors": {
                    "Faithfulness": F,
                    "Simulatability_cash": S_stance,
                    "Stability": St,
                },
            },
            "selection": {
                "value": crystal_sel_full,
                "value_no_stability": crystal_sel_nostab,
                "formula": "Faithfulness x Completeness_selection(K<=9) x Stability",
                "factors": {
                    "Faithfulness": F,
                    "Completeness_selection": S_sel,
                    "Stability": St,
                },
            },
            "controllability_coequal_axis": {
                "cash": ctrl.get("cash"),
                "selection": ctrl.get("selection"),
                "note": "reported alongside, NOT multiplied into CrystalScore (commanded dial is near-tautological)",
            },
        },
        "pareto": {
            "stance": pareto_stance,
            "selection": pareto_sel,
            "x": "K (codes / description length)",
            "y": "completeness@K",
        },
        "complexity_curve": {
            "stance": complexity_pts_stance,
            "selection": complexity_pts_sel,
            "x": "behavioral complexity captured (completeness@K)",
            "y": "CrystalScore(K)",
            "honest_caveat": (
                "ONE policy only -> the true (complexity, CrystalScore) curve needs MORE policies. "
                "This within-policy K-sweep is a placeholder proxy, not the cross-policy curve."
            ),
        },
        "provenance": provenance,
        "plot": plot_info,
        "interpretation": (
            "R6c is CRYSTAL-CLEAR on RISK-STANCE (cash) and EMPTY-BECAUSE-TRIVIAL on SELECTION. "
            "Faithfulness is perfect (every steered code moves cash monotonically in its labeled "
            "direction); cash is steerable but only at the EMPIRICAL fidelity level (~0.20, not the "
            "near-tautological commanded dial). The selection book is near-equal-weight (within-book risky "
            "HHI ~0.039 vs equal-weight 0.0345; mean L1 to equal-weight ~0.19), so there is almost no "
            "within-book tilt to explain: true within-book SELECTION completeness@9 is only ~0.14, and "
            "selection is NOT controllable (commanded_risky_hhi is flat across steering). The naive raw-29d "
            "weight completeness (~0.34) is an EXPOSURE ARTIFACT -- its variance is ~28x dominated by the "
            "gross long exposure (= 1 - cash), so it merely restates the STANCE; we therefore measure "
            "selection on the exposure-removed within-book composition. Net: STANCE CrystalScore is the "
            "meaningful axis, SELECTION CrystalScore is ~0 because selection is trivial. This ALIGNS the "
            "no-alpha null with the interpretability thesis: R6c has little selection alpha precisely because "
            "it does little selection -- it is a transparent cash-timing controller, and CrystalScore says so."
        ),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))
    write_markdown(report)
    print_console(report)


def write_markdown(r: dict) -> None:
    sm = r["submetrics"]
    F = sm["faithfulness"]["value"]
    Sc = sm["simulatability"]["headline_discrete"]
    Scont = sm["simulatability"]["continuous_ceiling"]
    St = sm["stability"]["value"]
    comp = sm["completeness"]
    ctrl = sm["controllability"]
    cs = r["crystalscore"]

    def f(x):
        return "N/A" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3f}"

    lines = []
    lines.append("# CrystalScore v1 -- R6c (frozen 2022-2023)\n")
    lines.append(f"Policy: `{r['policy']}`  \nLatent: `{r['latent_layer']}` ({r['n_steps']} steps)  \n")
    lines.append(
        "CrystalScore is the first numeric interpretability score for the Interpretable-CHRL "
        "(\"CrystalRL\") policy. Every number below is computed from EXISTING frozen-2022-2023 "
        "artifacts (no retrain). Each sub-metric is in [0,1]; negative R^2 is clipped to 0; missing "
        "inputs are marked N/A.\n"
    )

    lines.append("## Sub-metrics\n")
    lines.append("| Sub-metric | Value | Reading | Provenance |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| **Faithfulness** | {f(F)} | {sm['faithfulness']['note']} | `{r['provenance']['FAITHFULNESS']}` |"
    )
    lines.append(
        f"| **Simulatability** (headline, discrete code->cash, K<=9) | {f(Sc)} | "
        f"K=9 cluster-mean predictor of cash_target | "
        f"`{r['provenance']['SIMULATABILITY_discrete']}` |"
    )
    lines.append(
        f"| Simulatability (continuous ceiling, latent->cash cv_r2) | {f(Scont)} | "
        f"final_64 -> cash_target cv_r2_shuffled | `{r['provenance']['SIMULATABILITY_continuous']}` |"
    )
    lines.append(
        f"| **Completeness@9** (STANCE / cash) | {f(comp['stance'].get(9))} | "
        f"1 - SS_res/SS_tot, 9-cluster-mean of cash | `{r['provenance']['COMPLETENESS']}` |"
    )
    lines.append(
        f"| **Completeness@9** (SELECTION / within-book composition) | {f(comp['selection'].get(9))} | "
        f"1 - SS_res/SS_tot, 9-cluster-mean of renormalised (exposure-removed) weights | "
        f"`{r['provenance']['COMPLETENESS']}` |"
    )
    lines.append(
        f"| Completeness@9 (raw weights, exposure-confounded) | {f(comp['selection_raw_confounded'].get(9))} | "
        f"DIAGNOSTIC ONLY -- raw-W variance is ~{comp.get('exposure_var_share_of_rawW')}x gross-exposure; "
        f"tracks STANCE, not selection | `{r['provenance']['COMPLETENESS']}` |"
    )
    lines.append(
        f"| **Controllability** (cash, EMPIRICAL) | {f(ctrl['cash'])} | "
        f"success x monotonicity x empirical-fidelity({ctrl['components']['empirical_fidelity_spearman']}) | "
        f"`{r['provenance']['CONTROLLABILITY']}` |"
    )
    lines.append(
        f"| **Controllability** (SELECTION) | {f(ctrl['selection'])} | "
        f"commanded_risky_hhi FLAT across steering -> selection not steerable | "
        f"`{r['provenance']['CONTROLLABILITY']}` |"
    )
    lines.append(
        f"| **Stability** | {f(St)} | cross-seed ARI of K=9 code clusters | "
        f"`{r['provenance']['STABILITY']}` |"
    )
    lines.append("")

    lines.append("## CrystalScore (Faithfulness x Simulatability x Stability, K<=9)\n")
    lines.append("| Behavior | CrystalScore | (no-Stability variant) | Factors |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| **STANCE (cash/risk)** | **{f(cs['stance']['value'])}** | {f(cs['stance']['value_no_stability'])} | "
        f"F={f(cs['stance']['factors']['Faithfulness'])} x Simul={f(cs['stance']['factors']['Simulatability_cash'])} "
        f"x Stab={f(cs['stance']['factors']['Stability'])} |"
    )
    lines.append(
        f"| **SELECTION** | **{f(cs['selection']['value'])}** | {f(cs['selection']['value_no_stability'])} | "
        f"F={f(cs['selection']['factors']['Faithfulness'])} x Compl_sel={f(cs['selection']['factors']['Completeness_selection'])} "
        f"x Stab={f(cs['selection']['factors']['Stability'])} |"
    )
    lines.append("")
    lines.append(
        f"Controllability (co-equal axis, NOT multiplied in): cash = **{f(ctrl['cash'])}**, "
        f"selection = **{f(ctrl['selection'])}**. The commanded \"dial\" is near-tautological "
        f"(commanded-vs-empirical cash Spearman ~ {ctrl['components']['empirical_fidelity_spearman']}), "
        "so cash controllability is reported at the EMPIRICAL level, not the inflated commanded level.\n"
    )

    ps = r["pareto"]["stance"]
    psel = r["pareto"]["selection"]
    lines.append("## Pareto (Completeness vs Parsimony)\n")
    lines.append("| Curve | AUC (normalized mean-height) | Completeness@K*=9 |")
    lines.append("|---|---|---|")
    lines.append(f"| STANCE (cash) | {f(ps.get('auc_normalized'))} | {f(ps.get('at_K9'))} |")
    lines.append(f"| SELECTION (risky weights) | {f(psel.get('auc_normalized'))} | {f(psel.get('at_K9'))} |")
    lines.append("")
    lines.append("Per-K completeness:\n")
    lines.append("| K | completeness STANCE | completeness SELECTION (within-book) | raw-W (exposure-confounded, diag) |")
    lines.append("|---|---|---|---|")
    for k in r["K_grid"]:
        lines.append(
            f"| {k} | {f(comp['stance'].get(k))} | {f(comp['selection'].get(k))} | "
            f"{f(comp['selection_raw_confounded'].get(k))} |"
        )
    lines.append("")
    lines.append(f"Curve CSVs: `{OUT_PARETO_CSV.name}`, `{OUT_COMPLEXITY_CSV.name}`. ")
    if r["plot"].get("plotted"):
        lines.append(f"PNGs: `{OUT_PARETO_PNG.name}`, `{OUT_COMPLEXITY_PNG.name}`.\n")
    else:
        lines.append("(matplotlib unavailable -> PNGs skipped.)\n")

    lines.append("## (Complexity, CrystalScore) curve\n")
    lines.append(
        r["complexity_curve"]["honest_caveat"] + " Points (complexity_captured, CrystalScore) for STANCE:\n"
    )
    lines.append("```")
    lines.append("STANCE:    " + ", ".join(f"({a:.2f},{b:.2f})" for a, b in r["complexity_curve"]["stance"]))
    lines.append("SELECTION: " + ", ".join(f"({a:.2f},{b:.2f})" for a, b in r["complexity_curve"]["selection"]))
    lines.append("```\n")

    lines.append("## Reading the numbers (caveats)\n")
    lines.append(
        "- **Why the STANCE CrystalScore is only ~0.15 despite \"crystal-clear\" behavior.** "
        "The QUALITATIVE transparency is high (Faithfulness = 1.0; cash is monotonically steerable; the "
        "continuous latent decodes cash at cv_r2 = 0.82). The CrystalScore is modest because it deliberately "
        "scores the *parsimonious* description: a <=9-code discretisation of a genuinely CONTINUOUS cash dial "
        "is lossy (discrete completeness 0.244 vs continuous ceiling 0.820). So CrystalScore measures \"how "
        "well does a 9-symbol human story reproduce the policy\", not \"is the policy legible at all\" -- the "
        "gap (0.82 -> 0.24) is the price of human-readable parsimony, and it is reported, not hidden.\n"
        "- **Completeness is not perfectly monotonic in K** (e.g. STANCE dips at K=5). KMeans on a continuous "
        "manifold with a fixed seed does not guarantee nested partitions, so a coarser K can occasionally "
        "capture cash variance better than a slightly finer one. We report the raw sweep rather than forcing "
        "monotonicity.\n"
        "- **Stability is from the canonical multi-seed codebook ARI (0.619 at K=9)**; a fresh recompute on the "
        "standardised latent is logged in the JSON as an independent check. If no multi-seed assignments "
        "existed, Stability would be N/A and the no-Stability CrystalScores (STANCE 0.244 / SELECTION 0.141) "
        "would be the headline.\n"
    )
    lines.append("## Honest interpretation\n")
    lines.append(r["interpretation"] + "\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def print_console(r: dict) -> None:
    sm = r["submetrics"]
    cs = r["crystalscore"]
    ctrl = sm["controllability"]
    comp = sm["completeness"]

    def f(x):
        return "N/A" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3f}"

    print("=" * 72)
    print("CrystalScore v1 -- R6c (frozen 2022-2023)")
    print("=" * 72)
    print(f"n_steps={r['n_steps']}  latent={r['latent_layer']}  K_budget={r['K_budget']}")
    print("-" * 72)
    print("SUB-METRICS (each in [0,1]):")
    print(f"  Faithfulness            : {f(sm['faithfulness']['value'])}   ({sm['faithfulness']['note']})")
    print(f"  Simulatability (disc K9): {f(sm['simulatability']['headline_discrete'])}   [headline]")
    print(f"  Simulatability (cont.)  : {f(sm['simulatability']['continuous_ceiling'])}   (latent->cash cv_r2 ceiling)")
    print(f"  Completeness@9 STANCE   : {f(comp['stance'].get(9))}")
    print(f"  Completeness@9 SELECTION: {f(comp['selection'].get(9))}   (within-book composition = TRUE selection)")
    print(f"    [diag] raw-W completeness: {f(comp['selection_raw_confounded'].get(9))}   (exposure-confounded ~{comp.get('exposure_var_share_of_rawW')}x; tracks STANCE)")
    print(f"  Controllability cash    : {f(ctrl['cash'])}   (EMPIRICAL; emp-fidelity spearman={ctrl['components']['empirical_fidelity_spearman']})")
    print(f"  Controllability SELECTION: {f(ctrl['selection'])}   (commanded_risky_hhi flat -> not steerable)")
    print(f"  Stability               : {f(sm['stability']['value'])}   (cross-seed ARI K=9)")
    print("-" * 72)
    print("CRYSTALSCORE (Faithfulness x Simulatability x Stability, K<=9):")
    print(f"  STANCE   : {f(cs['stance']['value'])}   (no-stability: {f(cs['stance']['value_no_stability'])})")
    print(f"  SELECTION: {f(cs['selection']['value'])}   (no-stability: {f(cs['selection']['value_no_stability'])})")
    print(f"  Controllability (co-equal): cash={f(ctrl['cash'])}  selection={f(ctrl['selection'])}")
    print("-" * 72)
    ps, psel = r["pareto"]["stance"], r["pareto"]["selection"]
    print("PARETO (completeness vs parsimony):")
    print(f"  STANCE   : AUC(norm)={f(ps.get('auc_normalized'))}  completeness@9={f(ps.get('at_K9'))}")
    print(f"  SELECTION: AUC(norm)={f(psel.get('auc_normalized'))}  completeness@9={f(psel.get('at_K9'))}")
    print("-" * 72)
    print("INTERPRETATION:")
    print("  " + r["interpretation"].replace(" -> ", " -> "))
    print("-" * 72)
    print(f"Wrote: {OUT_JSON.name}, {OUT_MD.name}, {OUT_PARETO_CSV.name}, {OUT_COMPLEXITY_CSV.name}")
    if r["plot"].get("plotted"):
        print(f"       {OUT_PARETO_PNG.name}, {OUT_COMPLEXITY_PNG.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
