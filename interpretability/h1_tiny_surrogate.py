"""H1 -- "R6c is a tiny surrogate in disguise": does a depth-<=4 decision-list / FSM reproduce R6c's
FROZEN action log from the observation, at low KL, WITHOUT the simplicity being a market artifact?

THE TEST (crystalrl-interpretability route: Faithfulness x Simulatability x Parsimony)
-------------------------------------------------------------------------------------
Fit a depth in {1,2,3,4} DecisionTreeClassifier (the FSM / decision-list, parsimony = max_depth) on
obs -> action, on a TEMPORAL holdout (first 60% train / last 40% test -- never random CV, to honor the
frozen ordering), PLUS a regime holdout (low- vs high-drawdown halves) as robustness. Two action axes,
scored SEPARATELY so exposure cannot dominate:
  STANCE     : cash_target discretized into K equal-frequency bins.
  SELECTION  : within-book composition = executed_weight over the 29 names with CASH removed and
               renormalized to sum 1 (kills the ~28x exposure/cash dominance), KMeans-clustered into K
               selection modes. This is the completeness=0.141 axis the synthesis flagged as the hard one.

THE DECISIVE NULL (the one that decides "tiny surrogate" vs "simple market")
---------------------------------------------------------------------------
An order-k Markov surrogate predicts the action symbol from the PREVIOUS k action symbols ONLY (ignores
obs). If it reproduces the holdout log as well as the obs->action tree, the simplicity lived in the
action's own autocorrelation (a near-trivial optimal sequence on a liquid universe), NOT in a compact
readable policy program -- and "tiny surrogate in disguise" is REFUTED in favour of "simple market".
H1 passes only if the depth-<=4 tree (a) hits high fidelity / low KL AND (b) beats the order-k Markov null.

SCORER (new -- verified absent from interpretability/): balanced accuracy + TV + KL of the surrogate's
predicted action-symbol distribution vs the policy's, at the leaf level, on the held-out split.

MIRROR-OF-THE-HALL: R6c's OOS edge on Dow-29 is NULL. A high simulatability here means "this null-edge
cash-timing policy is easy to simulate", NOT "the surrogate understands a real signal". Stamped in output.

Run: python interpretability/h1_tiny_surrogate.py
Out: interpretability/h1_tiny_surrogate_report.json + interpretability/H1_TINY_SURROGATE.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JA67 = ROOT / "artifacts/action_vq/A67_joint_hidden_action_controls_fullenv_from_R6c_v1/ja67_joint_controls_daily.csv"
OUT_JSON = HERE / "h1_tiny_surrogate_report.json"
OUT_MD = HERE / "H1_TINY_SURROGATE.md"

DEPTHS = [1, 2, 3, 4]
K_ACT = 5          # action-symbol alphabet (stance bins / selection modes)
MARKOV_ORDERS = [1, 2]
SEED = 0
KL_EPS = 0.10      # the card's "KL <= eps" target (bits); reported, not a hard gate


def _tv_kl(true_sym: np.ndarray, pred_sym: np.ndarray, K: int) -> tuple:
    """TV and KL(true||pred) between the marginal action-symbol distributions on the holdout (bits)."""
    pt = np.bincount(true_sym, minlength=K).astype(float); pt /= pt.sum()
    pp = np.bincount(pred_sym, minlength=K).astype(float); pp /= pp.sum()
    tv = 0.5 * float(np.abs(pt - pp).sum())
    mask = pt > 0
    kl = float(np.sum(pt[mask] * np.log2(pt[mask] / np.clip(pp[mask], 1e-9, None))))
    return tv, kl


def _markov_predict(symbols: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, order: int, K: int) -> np.ndarray:
    """Order-k Markov null: P(next | previous k symbols) from the TRAIN slice, applied to TEST.
    Backs off to the train marginal mode for unseen histories. Predicts the action symbol from the
    action's OWN past only -- no observation. Aligned so prediction at day t uses symbols[t-order:t]."""
    from collections import defaultdict
    table = defaultdict(lambda: np.zeros(K))
    train_set = set(train_idx.tolist())
    for t in range(order, len(symbols)):
        if t in train_set:
            hist = tuple(symbols[t - order:t].tolist())
            table[hist][symbols[t]] += 1.0
    marg = np.bincount(symbols[train_idx], minlength=K).astype(float)
    marg = marg / marg.sum() if marg.sum() > 0 else np.ones(K) / K
    preds = np.empty(len(test_idx), dtype=int)
    for j, t in enumerate(test_idx):
        if t < order:
            preds[j] = int(np.argmax(marg)); continue
        hist = tuple(symbols[t - order:t].tolist())
        preds[j] = int(np.argmax(table[hist])) if hist in table and table[hist].sum() > 0 else int(np.argmax(marg))
    return preds


def _eq_freq_bins(x: np.ndarray, k: int) -> np.ndarray:
    """Equal-frequency (quantile) discretization into k classes, so the action alphabet is balanced."""
    qs = np.quantile(x, np.linspace(0, 1, k + 1))
    qs = np.unique(qs)
    if qs.size < 2:
        return np.zeros(x.size, dtype=int)
    return np.clip(np.digitize(x, qs[1:-1], right=False), 0, qs.size - 2).astype(int)


def _eval_axis(name: str, y: np.ndarray, X: np.ndarray, splits: dict, K: int) -> dict:
    """For one action axis (stance/selection): sweep tree depth on each holdout, compute fidelity/TV/KL,
    and the order-k Markov null. Returns per-split, per-depth metrics + the verdict."""
    out = {"axis": name, "K": int(K), "splits": {}}
    for split_name, (tr, te) in splits.items():
        n_test_classes = int(np.unique(y[te]).size)
        rec = {"n_train": int(tr.size), "n_test": int(te.size), "n_test_classes": n_test_classes,
               "depths": {}, "markov": {}}
        if n_test_classes < 2:
            # Degenerate: the policy's action collapses to ONE symbol on this holdout (e.g. a near-static
            # book), so any predictor trivially scores 1.0 -- the fidelity metric is uninformative, NOT a
            # surrogate result. Report the collapse honestly instead of a fake "perfect simulatability".
            rec["verdict"] = (f"DEGENERATE (test window is single-valued: the action collapses to 1 symbol over "
                              f"{te.size} days -> near-static behaviour, fidelity metric uninformative)")
            rec["tree_best_bal_acc"] = rec["markov_best_bal_acc"] = float("nan")
            out["splits"][split_name] = rec
            continue
        # tree surrogate (obs -> action) at each parsimony budget
        for depth in DEPTHS:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=SEED).fit(X[tr], y[tr])
            yhat = clf.predict(X[te])
            tv, kl = _tv_kl(y[te], yhat, K)
            rec["depths"][depth] = {
                "balanced_acc": round(float(balanced_accuracy_score(y[te], yhat)), 4),
                "acc": round(float((yhat == y[te]).mean()), 4),
                "TV": round(tv, 4), "KL_bits": round(kl, 4), "n_leaves": int(clf.get_n_leaves()),
            }
        # order-k Markov null (action autocorrelation only)
        symbols = y
        for order in MARKOV_ORDERS:
            mp = _markov_predict(symbols, tr, te, order, K)
            tv, kl = _tv_kl(y[te], mp, K)
            rec["markov"][order] = {
                "balanced_acc": round(float(balanced_accuracy_score(y[te], mp)), 4),
                "acc": round(float((mp == y[te]).mean()), 4), "TV": round(tv, 4), "KL_bits": round(kl, 4),
            }
        best_tree = max(rec["depths"][d]["balanced_acc"] for d in DEPTHS)
        best_d4 = rec["depths"][4]["balanced_acc"]
        best_markov = max(rec["markov"][o]["balanced_acc"] for o in MARKOV_ORDERS)
        rec["tree_best_bal_acc"] = round(best_tree, 4)
        rec["markov_best_bal_acc"] = round(best_markov, 4)
        rec["tree_beats_markov"] = bool(best_tree > best_markov + 0.02)  # obs adds readable info beyond autocorr
        rec["d4_kl_le_eps"] = bool(rec["depths"][4]["KL_bits"] <= KL_EPS)
        # Taxonomy distinguishes the FORM of the tiny surrogate (reactive vs autoregressive) from no-surrogate.
        if rec["tree_beats_markov"] and best_d4 >= 0.5:
            rec["verdict"] = "REACTIVE-TINY-SURROGATE (depth<=4 obs->action tree simulates AND beats the autocorrelation null)"
        elif best_markov >= 0.5:
            rec["verdict"] = ("AUTOREGRESSIVE-TINY-SURROGATE (a low-order Markov chain on the action's OWN past reproduces "
                              "the log; R6c is a tiny persistence/smoothing program, NOT a reactive obs->action map -- and "
                              "since the OOS edge is NULL this self-predictability is the policy's own autocorrelation, not "
                              "market signal)")
        else:
            rec["verdict"] = ("NO-READABLE-SURROGATE (neither the obs->action tree nor a short action-history predicts above "
                              "~chance on this holdout -- the structure does not transfer across this split)")
        out["splits"][split_name] = rec
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not JA67.exists():
        print(f"[H1] FATAL: {JA67} not found"); return 1
    df = pd.read_csv(JA67)
    if "counterfactual_variant" in df.columns:
        df = df[df["counterfactual_variant"] == "original_ppo"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].min() >= pd.Timestamp("2022-01-01"), "frozen-legality violated"
    T = len(df)
    win = (str(df["date"].min().date()), str(df["date"].max().date()))
    print(f"[H1] frozen original_ppo: {T} days {win[0]}..{win[1]}")

    # ---- predictors: the human-readable market/regime obs block the policy saw ----
    pred_cols = [c for c in df.columns if c.startswith("market_feature_")]
    X = StandardScaler().fit_transform(df[pred_cols].to_numpy(dtype=float))
    print(f"[H1] predictors: {len(pred_cols)} market_feature_* cols")

    # ---- STANCE target: cash_target -> equal-frequency bins ----
    y_stance = _eq_freq_bins(df["cash_target"].to_numpy(float), K_ACT)

    # ---- SELECTION target: within-book composition (CASH removed, renormalized) -> KMeans modes ----
    ew_cols = [c for c in df.columns if c.startswith("executed_weight_") and not c.endswith("CASH")]
    W = df[ew_cols].to_numpy(dtype=float)
    W = np.clip(W, 0, None)
    rs = W.sum(axis=1, keepdims=True)
    within = np.divide(W, rs, out=np.full_like(W, 1.0 / W.shape[1]), where=rs > 1e-9)  # within-book, exposure-free
    y_select = KMeans(n_clusters=K_ACT, random_state=SEED, n_init=10).fit_predict(within)
    print(f"[H1] selection: {len(ew_cols)} within-book names (CASH removed, renormalized)")

    # ---- holdouts: temporal 60/40 + regime (low vs high drawdown) ----
    idx = np.arange(T)
    cut = int(T * 0.6)
    temporal = (idx[:cut], idx[cut:])
    dd = df["drawdown"].to_numpy(float) if "drawdown" in df.columns else np.zeros(T)
    med = np.median(dd)
    lo, hi = idx[dd <= med], idx[dd > med]
    regime = (lo, hi) if (lo.size > 20 and hi.size > 20) else temporal
    splits = {"temporal_60_40": temporal, "regime_drawdown": regime}

    stance = _eval_axis("STANCE (cash)", y_stance, X, splits, K_ACT)
    select = _eval_axis("SELECTION (within-book, exposure-free)", y_select, X, splits, K_ACT)

    report = {
        "policy": "R6c (original_ppo)", "universe": "Dow-29", "frozen_window": win, "n_days": T,
        "predictors": f"{len(pred_cols)} market_feature_* (regime/macro obs the policy saw)",
        "action_alphabet_K": K_ACT, "tree_depths": DEPTHS, "markov_orders": MARKOV_ORDERS, "kl_eps_bits": KL_EPS,
        "mirror_of_the_hall": ("R6c's OOS edge on Dow-29 is NULL -- high simulatability means this null-edge "
                               "cash-timing policy is easy to simulate, NOT that the surrogate understands a real signal."),
        "decisive_null": ("order-k Markov on the action's OWN past: if it ties/beats the obs->action tree, the "
                          "simplicity is action-autocorrelation (simple market), not a readable policy program."),
        "stance": stance, "selection": select,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# H1 -- tiny-surrogate test (R6c, frozen 2022-2023)", "",
             f"- Policy **R6c (original_ppo)**, Dow-29, frozen {win[0]}..{win[1]} ({T} days). "
             f"Predictors: {len(pred_cols)} `market_feature_*` obs. Action alphabet K={K_ACT}.",
             f"- **Mirror-of-the-hall:** {report['mirror_of_the_hall']}",
             f"- **Decisive null:** {report['decisive_null']}", ""]
    for axis_obj in (stance, select):
        lines.append(f"## {axis_obj['axis']}")
        lines.append("| holdout | tree d4 bal-acc | best tree | KL@d4 (bits) | order-k Markov bal-acc | tree beats markov? | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for sp, rec in axis_obj["splits"].items():
            if not rec["depths"]:  # degenerate single-class test window
                lines.append(f"| {sp} | n/a | n/a | n/a | n/a | n/a | {rec['verdict'].split(' (')[0]} |")
                continue
            d4 = rec["depths"][4]; bt = max(rec["depths"][d]["balanced_acc"] for d in DEPTHS)
            bm = max(rec["markov"][o]["balanced_acc"] for o in MARKOV_ORDERS)
            lines.append(f"| {sp} | {d4['balanced_acc']} | {bt} | {d4['KL_bits']} | {bm} | "
                         f"{'YES' if rec['tree_beats_markov'] else 'no'} | {rec['verdict'].split(' (')[0]} |")
        lines.append("")
    lines.append("**Reading.** *Tiny-surrogate* holds only where a depth-≤4 tree both reaches usable fidelity "
                 "AND beats the order-k Markov (action-autocorrelation) null — i.e. the obs genuinely explain the "
                 "action beyond its own persistence. Where Markov ties/wins, the simplicity is the market's, not a "
                 "readable policy program. STANCE and SELECTION are scored separately; SELECTION is the exposure-free "
                 "within-book axis (the hard one).")
    lines.append("")
    lines.append("**Auto-applies to the new policy.** Re-run pointed at the parallel agent's PIT-retrained log to "
                 "get its simulatability on the same recipe.")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    for axis_obj in (stance, select):
        for sp, rec in axis_obj["splits"].items():
            extra = ("" if not rec["depths"] else
                     f" (d4 bal-acc {rec['depths'][4]['balanced_acc']}, markov {max(rec['markov'][o]['balanced_acc'] for o in MARKOV_ORDERS)})")
            print(f"[H1] {axis_obj['axis'][:22]:22s} {sp:16s} -> {rec['verdict'].split(' (')[0]}{extra}")
    print(f"[H1] wrote {OUT_JSON.name} + {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
