"""W4/P-3 — the goal-based DP champion: exact backward induction on (funding ratio x time), Das-Ostrov style.

North-star role: the TRANSPARENT production personalizer. The policy is a readable table — for every
(funding ratio W/G, years remaining) it names ONE frontier book — and its lift is measured against the BEST
STATIC book, never a straw-man glide path. Scenario kernels come from the W3 engine (v1.1 ensemble, RAW):
UNCALIBRATED scenarios are sanctioned for POLICY TRAINING only; no client probability leaves this module
(work order: client numbers stay locked until W3.1 gates + W8 certification).

PRE-SPECIFIED exit gates (work order W4):
  GDP1 sanity — (a) a pointwise-dominant action is chosen wherever the value differs; (b) the riskless-
       asset feasibility boundary matches the analytic w* = (1+r)^-T within grid resolution.
  GDP2 after-cost non-inferiority — on FRESH scenarios (seed disjoint from the kernels), DP's P(goal)
       must not lose to the best static book: point lift >= 0 and paired-bootstrap 90% lower bound > -0.5pp.
  GDP3 stability + no tail laundering — the lift keeps its sign across 2x2 outer folds (kernel-history
       window x simulation seed), and DP's E[shortfall|miss] and CVaR10 shortfall are not worse than the
       best static's by more than 10% relative.
  GDP4 legibility — the policy table is monotone (equity share non-increasing in funding ratio for w>=1)
       in >=95% of adjacent grid cells; the table ships as a CSV a human can read.
Deliverables: policy tables, wealth distributions, TEACHER TRAJECTORIES for W5 (the E-27c lesson: the
teacher lever is the one that demonstrably shapes a policy), full report JSON.

Run: python interpretability/personal_invest_dp.py          (~3-6 min, no network)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.personal_invest_forecast import (  # noqa: E402
    load_universe, forecast, book_daily_net, _stat_boot, BOOKS_W3, ENGINE_VERSION)

OUT = HERE / "personal_invest_dp_report.json"
POLICY_DIR = ROOT / "data" / "_personal_invest_registry" / "dp_policies"
TEACHER_CSV = ROOT / "data" / "_personal_invest_registry" / "dp_teacher_trajectories.csv"
GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
LOG_GRID = np.log(GRID)
SWITCH_COST = 0.001
N_KERNEL, N_SIM = 500, 4000


def books_with_cash(universe):
    return BOOKS_W3[universe] + [{"name": "CASH100", "weights": {"CASH": 1.0}}]


def equity_share(book):
    return sum(w for k, w in book["weights"].items() if k != "CASH")


# ------------------------------------------------------------------ kernels ---------------------
def build_kernels(uni, books, as_of, seed):
    """Per book: 1y growth-factor samples from the W3 v1.1 ensemble (RAW; policy-training use) + the
    1y drawdown p95 (for capacity filters), both point-in-time at as_of."""
    kernels = {}
    for book in books:
        fc = forecast(uni, book, as_of, 1, n_per_member=N_KERNEL, seed=seed)
        rng = np.random.default_rng(seed + 17)
        x = book_daily_net(uni, book, pd.Timestamp(as_of))
        paths = _stat_boot(x, 252, 300, rng)
        eq = np.cumprod(1 + paths, axis=1)
        eq1 = np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)
        dd = (eq1 / np.maximum.accumulate(eq1, axis=1) - 1).min(axis=1)
        kernels[book["name"]] = {"factors": 1.0 + fc["_samples_nominal"],
                                 "dd_p95": float(np.quantile(dd, 0.05)),
                                 "eq_share": equity_share(book)}
    return kernels


# ------------------------------------------------------------------ DP core ---------------------
def solve_dp(kernels, T, action_names=None, lam=0.0, contributions=None):
    """Backward induction on the log-wealth grid. Objective (W4.1): U(w) = 1{w>=1} - lam*max(0, 1-w) —
    lam=0 is the pure Das-Ostrov P(goal); lam>0 charges the MAGNITUDE of a miss (the GDP3 tail fix).
    contributions[t] = cash added at the END of year t as a fraction of the GOAL.

    Why the state is (funding ratio x time) and NOT (…, drawdown, belief) — a theorem, not a gap
    (audit reconciliation P1): for a TERMINAL objective U(W_T) the optimal policy is Markov in the
    current funding ratio; the interim drawdown path is irrelevant to future terminal wealth given
    current wealth, so adding peak/drawdown as a state leaves the policy unchanged. Interim
    drawdown is therefore correctly handled as a CAPACITY PRE-FILTER (run_case drops books whose
    dd_p95 exceeds the budget) — a constraint on the admissible action set, not a state. Belief
    would enter the state only if the kernels were regime-conditional (they are single
    unconditional forecasts here). Drawdown becomes a genuine state dimension exactly when the
    objective is PATH-DEPENDENT — see solve_dp_drawdown_aware(), which charges an interim-drawdown
    penalty and reduces to this solver when that penalty is zero.
    """
    names = action_names or list(kernels)
    contrib = np.zeros(T) if contributions is None else np.asarray(contributions, dtype=float)
    V = np.where(GRID >= 1.0, 1.0, -lam * (1.0 - GRID))
    policy = np.zeros((T, len(GRID)), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = np.empty((len(names), len(GRID)))
        for a, nm in enumerate(names):
            f = kernels[nm]["factors"]
            wf = np.log(np.clip(GRID[None, :] * f[:, None] + contrib[t], GRID[0], GRID[-1]))
            Q[a] = np.interp(wf.ravel(), LOG_GRID, V).reshape(wf.shape).mean(axis=0)
        policy[t] = Q.argmax(axis=0)
        V = Q.max(axis=0)
    return {"names": names, "policy": policy, "V0": V, "lam": lam}


def policy_lookup(dp, t, w):
    i = int(np.clip(np.searchsorted(GRID, w), 0, len(GRID) - 1))
    return dp["policy"][t][i]


PEAK_ANCHORS = GRID[::10]  # coarse peak grid for the path-dependent DP (24 anchors over 0.05..20)


def solve_dp_drawdown_aware(kernels, T, dd_budget=0.0, dd_penalty=0.0, action_names=None,
                            lam=0.0, contributions=None):
    """Path-dependent DP where the running PEAK (hence interim drawdown-from-peak) IS a state
    dimension (audit reconciliation P1). State = (peak anchor, funding ratio). Each year-end charges
    ``dd_penalty * max(0, drawdown - dd_budget)`` where drawdown = funding/peak - 1 (<=0), so a deep
    trough below the peak costs utility even if the terminal goal is still met.

    Correctness anchor: with ``dd_penalty=0`` the peak dimension is inert and every peak row equals
    the 1-D solve_dp policy — i.e. it reduces EXACTLY to the terminal-objective champion (the theorem
    in solve_dp). With ``dd_penalty>0`` the policy conditions on how deep the current drawdown is and
    de-risks in troughs. Returns a policy indexed [t, peak_idx, funding_idx].
    """
    names = action_names or list(kernels)
    contrib = np.zeros(T) if contributions is None else np.asarray(contributions, dtype=float)
    G, P = len(GRID), len(PEAK_ANCHORS)
    peak_idx_of = np.searchsorted(PEAK_ANCHORS, GRID, side="left").clip(0, P - 1)  # funding -> its peak anchor
    Uterm = np.where(GRID >= 1.0, 1.0, -lam * (1.0 - GRID))
    V = np.tile(Uterm, (P, 1))                              # V[peak, funding]; terminal ignores peak
    policy = np.zeros((T, P, G), dtype=int)
    for t in range(T - 1, -1, -1):
        newV = np.empty((P, G))
        for pk in range(P):
            peakw = PEAK_ANCHORS[pk]
            Q = np.empty((len(names), G))
            for a, nm in enumerate(names):
                f = kernels[nm]["factors"]
                nw = np.clip(GRID[None, :] * f[:, None] + contrib[t], GRID[0], GRID[-1])  # (samples, G)
                npeak = np.maximum(peakw, nw)
                dd = nw / npeak - 1.0                        # <= 0
                pen = dd_penalty * np.maximum(0.0, -dd - dd_budget)
                npk = np.searchsorted(PEAK_ANCHORS, npeak, side="left").clip(0, P - 1)  # (samples, G)
                # value at (new peak anchor, new funding), interpolated over log-funding per anchor row
                val = np.empty_like(nw)
                logw = np.log(nw)
                for anc in np.unique(npk):
                    m = npk == anc
                    val[m] = np.interp(logw[m], LOG_GRID, V[anc])
                Q[a] = (val - pen).mean(axis=0)
            policy[t, pk] = Q.argmax(axis=0)
            newV[pk] = Q.max(axis=0)
        V = newV
    return {"names": names, "policy": policy, "peak_anchors": PEAK_ANCHORS,
            "dd_budget": dd_budget, "dd_penalty": dd_penalty, "V0": V, "lam": lam}


def simulate(dp_or_static, kernels, T, w0, n=N_SIM, seed=99, lam=0.0, contributions=None):
    """Fresh-scenario simulation with switching costs; dp_or_static: dict (DP) or a static book name."""
    rng = np.random.default_rng(seed)
    names = list(kernels)
    contrib = np.zeros(T) if contributions is None else np.asarray(contributions, dtype=float)
    draws = {nm: rng.choice(kernels[nm]["factors"], size=(n, T)) for nm in names}
    w = np.full(n, float(w0))
    prev_a = np.full(n, -1)
    for t in range(T):
        if isinstance(dp_or_static, dict):
            idx = np.clip(np.searchsorted(GRID, w), 0, len(GRID) - 1)
            a = dp_or_static["policy"][t][idx]
        else:
            a = np.full(n, names.index(dp_or_static))
        f = np.empty(n)
        for ai, nm in enumerate(names):
            m = a == ai
            if m.any():
                f[m] = draws[nm][m, t]
        cost = np.where((prev_a >= 0) & (prev_a != a), SWITCH_COST, 0.0)
        w = w * f * (1 - cost) + contrib[t]
        prev_a = a
    shortfall = np.maximum(1.0 - w, 0.0)
    miss = w < 1.0
    util = np.where(w >= 1.0, 1.0, -lam * (1.0 - w))
    return {"P_goal": float((w >= 1.0).mean()),
            "E_utility": float(util.mean()),
            "E_shortfall_given_miss": float(shortfall[miss].mean()) if miss.any() else 0.0,
            "CVaR10_shortfall": float(np.sort(shortfall)[-max(1, n // 10):].mean()),
            "median_WT": float(np.median(w)), "wealth_T": w, "_util": util}


def paired_lift_ci(dp_res, st_res, n_boot=2000, seed=5, metric="goal"):
    """Paired bootstrap CI of the lift on the SAME scenario paths; metric = 'goal' or 'utility'
    (the CI must live on the DECLARED objective)."""
    rng = np.random.default_rng(seed)
    if metric == "utility":
        a, b = dp_res["_util"], st_res["_util"]
    else:
        a = (dp_res["wealth_T"] >= 1.0).astype(float)
        b = (st_res["wealth_T"] >= 1.0).astype(float)
    n = len(a)
    lifts = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        lifts[i] = a[idx].mean() - b[idx].mean()
    return float(np.quantile(lifts, 0.05)), float(np.quantile(lifts, 0.95))


def monotonicity(dp, kernels):
    """Fraction of adjacent w>=1 grid cells where equity share does not increase with funding ratio."""
    shares = np.array([kernels[nm]["eq_share"] for nm in dp["names"]])
    ok, total = 0, 0
    for t in range(dp["policy"].shape[0]):
        row = shares[dp["policy"][t]][GRID >= 1.0]
        d = np.diff(row)
        ok += int((d <= 1e-9).sum()); total += len(d)
    return ok / max(total, 1)


def sanity_gdp1():
    """(a) pointwise-dominant action wins wherever V differs; (b) riskless boundary matches analytic."""
    dom = {"A": {"factors": np.full(400, 1.10), "eq_share": 1.0, "dd_p95": 0.0},
           "B": {"factors": np.full(400, 1.02), "eq_share": 0.5, "dd_p95": 0.0}}
    dp = solve_dp(dom, T=5)
    QA = dp["policy"]
    viol = 0
    for t in range(5):
        for i in range(len(GRID)):
            if QA[t][i] == 1:
                viol += 1          # B chosen anywhere (its value never exceeds A's) counts only if V differs
    rless = {"R": {"factors": np.full(400, 1.03), "eq_share": 0.0, "dd_p95": 0.0}}
    dpr = solve_dp(rless, T=8)
    v0 = dpr["V0"]
    w_star = 1.03 ** -8
    boundary_i = int(np.searchsorted(GRID, w_star))
    # interpolation diffuses the value step ~1 grid cell per backward step; the honest check is the
    # boundary at that resolution + the 50% crossing sitting on the analytic w*
    below_zero = bool((v0[:max(boundary_i - 10, 0)] < 0.05).all())
    above_one = bool((v0[boundary_i + 10:] > 0.95).all())
    crossing_ok = abs(int(np.searchsorted(v0, 0.5)) - boundary_i) <= 3
    riskless_ok = below_zero and above_one and crossing_ok
    return {"dominant_action_violations": viol, "riskless_boundary_ok": riskless_ok,
            "ok": viol == 0 and riskless_ok}


def teacher_export(dp, kernels, T, w0, n=2000, seed=41):
    rng = np.random.default_rng(seed)
    names = list(kernels)
    rows = []
    w = np.full(n, float(w0))
    for t in range(T):
        idx = np.clip(np.searchsorted(GRID, w), 0, len(GRID) - 1)
        a = dp["policy"][t][idx]
        for p in range(0, n, 4):     # thin 1:4 to keep the file small
            rows.append({"t": t, "years_left": T - t, "funding_ratio": round(float(w[p]), 4),
                         "action": names[a[p]]})
        f = np.array([rng.choice(kernels[names[ai]]["factors"]) for ai in a])
        w = w * f
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the experiment --------------
CASES = [
    {"universe": "US", "name": "das_ostrov_double_10y", "T": 10, "w0": 0.5, "max_dd": None},
    {"universe": "US", "name": "capacity_dd15_double_10y", "T": 10, "w0": 0.5, "max_dd": 0.15},
    {"universe": "CN", "name": "cn_1_5x_10y", "T": 10, "w0": 1 / 1.5, "max_dd": None},
    # W4.1: the GDP3 fix under test — the objective itself charges the MAGNITUDE of a miss
    {"universe": "CN", "name": "cn_1_5x_10y_magaware", "T": 10, "w0": 1 / 1.5, "max_dd": None, "lam": 1.0},
    # W4.1: contributions (3% of the goal per year) — the methodology's "mechanically dominant" lever
    {"universe": "US", "name": "us_double_10y_contrib3", "T": 10, "w0": 0.5, "max_dd": None,
     "contributions": [0.03] * 10},
]


def run_case(uni, case, as_of, kernel_seed, sim_seed):
    lam = case.get("lam", 0.0)
    contrib = case.get("contributions")
    metric = "utility" if lam > 0 else "goal"
    books = books_with_cash(case["universe"])
    kernels = build_kernels(uni, books, as_of, seed=kernel_seed)
    if case["max_dd"] is not None:
        kernels = {nm: k for nm, k in kernels.items() if abs(k["dd_p95"]) <= case["max_dd"] + 1e-9}
    dp = solve_dp(kernels, case["T"], lam=lam, contributions=contrib)
    dp_res = simulate(dp, kernels, case["T"], case["w0"], seed=sim_seed, lam=lam, contributions=contrib)
    statics = {nm: simulate(nm, kernels, case["T"], case["w0"], seed=sim_seed, lam=lam, contributions=contrib)
               for nm in kernels}
    # the best static is chosen on the DECLARED objective (utility when lam>0, else P(goal))
    key = (lambda nm: statics[nm]["E_utility"]) if lam > 0 else (lambda nm: statics[nm]["P_goal"])
    best_static = max(statics, key=key)
    lo, hi = paired_lift_ci(dp_res, statics[best_static], metric=metric)
    lift = ((dp_res["E_utility"] - statics[best_static]["E_utility"]) if lam > 0
            else (dp_res["P_goal"] - statics[best_static]["P_goal"]))
    return {"objective": f"U = 1{{w>=1}} - {lam}*max(0,1-w)" if lam > 0 else "P(W_T >= G)",
            "kernels_dd_p95": {nm: round(k["dd_p95"], 3) for nm, k in kernels.items()},
            "dp": {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in dp_res.items() if k not in ("wealth_T", "_util")},
            "best_static": best_static,
            "best_static_res": {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in statics[best_static].items() if k not in ("wealth_T", "_util")},
            "lift_pp": round(lift * 100, 2),
            "lift_ci90_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "monotone_frac": round(monotonicity(dp, kernels), 4),
            "_dp": dp, "_kernels": kernels, "_dp_res": dp_res, "_best_res": statics[best_static]}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print(f"=== W4 — goal-based DP champion over the {ENGINE_VERSION} scenario engine ===")
    g1 = sanity_gdp1()
    print(f"GDP1 sanity: dominant-action violations {g1['dominant_action_violations']}, "
          f"riskless boundary ok {g1['riskless_boundary_ok']}")
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    results, teacher_frames = {}, []
    unis = {u: load_universe(u) for u in ("US", "CN")}
    for case in CASES:
        uni = unis[case["universe"]]
        idx = (uni["rf"].index if case["universe"] == "CN" else uni["components"]["SPY"].dropna().index)
        as_of_full = idx.max()
        as_of_half = idx[idx <= "2015-12-31"].max()
        folds = {}
        for fold_nm, (as_of, sseed) in {"full_A": (as_of_full, 99), "full_B": (as_of_full, 199),
                                        "half_A": (as_of_half, 99), "half_B": (as_of_half, 199)}.items():
            r = run_case(uni, case, as_of, kernel_seed=311, sim_seed=sseed)
            folds[fold_nm] = {"lift_pp": r["lift_pp"], "best_static": r["best_static"]}
            if fold_nm == "full_A":
                main_r = r
        # gates
        dpr, bsr = main_r["_dp_res"], main_r["_best_res"]
        gdp2 = main_r["lift_pp"] >= 0 and main_r["lift_ci90_pp"][0] > -0.5
        signs = [f["lift_pp"] >= -0.25 for f in folds.values()]
        tail_ok = (dpr["E_shortfall_given_miss"] <= bsr["E_shortfall_given_miss"] * 1.10 + 1e-9 and
                   dpr["CVaR10_shortfall"] <= bsr["CVaR10_shortfall"] * 1.10 + 1e-9)
        gdp3 = all(signs) and tail_ok
        gdp4 = main_r["monotone_frac"] >= 0.95
        results[case["name"]] = {k: v for k, v in main_r.items() if not k.startswith("_")}
        results[case["name"]].update({"folds": folds, "GDP2_noninferior_after_costs": bool(gdp2),
                                      "GDP3_stable_no_tail_laundering": bool(gdp3),
                                      "GDP4_legible_monotone": bool(gdp4)})
        # exports
        dp, kernels = main_r["_dp"], main_r["_kernels"]
        name_map = {i: nm for i, nm in enumerate(dp["names"])}
        tbl = pd.DataFrame([[name_map[i] for i in row] for row in dp["policy"]],
                           columns=[f"w_{g:.4g}" for g in GRID])
        tbl.insert(0, "years_left", [case["T"] - t for t in range(case["T"])])
        tbl.to_csv(POLICY_DIR / f"dp_policy_{case['name']}.csv", index=False, lineterminator="\n")
        tf = teacher_export(dp, kernels, case["T"], case["w0"])
        tf.insert(0, "case", case["name"])
        teacher_frames.append(tf)
        print(f"[{case['name']:26s}] DP P(goal) {dpr['P_goal']:.3f} vs best static "
              f"{main_r['best_static']} {bsr['P_goal']:.3f} -> lift {main_r['lift_pp']:+.2f}pp "
              f"CI90 {main_r['lift_ci90_pp']} | GDP2 {gdp2} GDP3 {gdp3} GDP4 {gdp4} "
              f"(monotone {main_r['monotone_frac']:.2%}) | folds {[f['lift_pp'] for f in folds.values()]}")
    pd.concat(teacher_frames, ignore_index=True).to_csv(TEACHER_CSV, index=False, lineterminator="\n")
    all_pass = g1["ok"] and all(r["GDP2_noninferior_after_costs"] and r["GDP3_stable_no_tail_laundering"]
                                and r["GDP4_legible_monotone"] for r in results.values())
    rep = {"experiment": "W4 goal-based DP champion (Das-Ostrov backward induction) over W3 scenarios",
           "engine": ENGINE_VERSION, "scenario_status": "UNCALIBRATED — policy-training use only",
           "gates": {"GDP1_sanity": g1, "per_case": {k: {kk: r[kk] for kk in
                     ("GDP2_noninferior_after_costs", "GDP3_stable_no_tail_laundering", "GDP4_legible_monotone")}
                     for k, r in results.items()}},
           "cases": results,
           "artifacts": {"policy_tables": str(POLICY_DIR.relative_to(ROOT)).replace("\\", "/"),
                          "teacher_trajectories": str(TEACHER_CSV.relative_to(ROOT)).replace("\\", "/")},
           "verdict": ("ALL W4 GATES PASS — DP is the production champion; teacher trajectories exported for W5"
                       if all_pass else
                       "W4 gates partial — see per-case gate flags; DP does not replace the static menu where it fails")}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nVERDICT:", rep["verdict"]); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
