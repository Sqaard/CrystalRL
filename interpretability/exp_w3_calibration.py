"""W3 exit gates — outer rolling-origin (prequential) calibration of the forecast engine, with teeth.

PRE-SPECIFIED gates (work order W3; written BEFORE the run):
  G1  proper score (v1.2 PRIMARY, declared before the v1.2 run): the ENSEMBLE's mean DOWNSIDE pinball
      (tau in {05,10,25,50} — the product decides on risk quantiles, so the decision-relevant score
      weights the downside) must not exceed the UNCONDITIONAL-bootstrap baseline's; the FULL-tau pinball
      (v1's G1) stays computed and reported as a secondary diagnostic.
  W (v1.2) prequential MEMBER WEIGHTS: at each origin the mixture weights are a softmax over each
      member's trailing RESOLVED downside pinball (>=10 resolved scores required; equal until then).
  G2  calibration bands, per universe x horizon: |cov50-0.50| <= 0.10, |cov80-0.80| <= 0.10, cov90 >= 0.80.
  G3  stress regimes: origins whose realization window intersects a named stress (2008, CN-2015, 2020, 2022)
      must show cov80 >= 0.70 (no systematic undercoverage exactly where it hurts).
  G4  CMA backcast (P-5 kill test): at h=5 the building-block central's MAE vs realized must beat the
      trailing sample-mean central's MAE, per universe.
  TEETH  a deliberately overconfident forecaster (samples shrunk 0.5x around the median) MUST fail G2 —
      if the harness cannot reject it, the harness itself is rejected.
Verdict: ALL gates green -> the engine's bands may be labeled CALIBRATED_V1 (within the measured error);
otherwise every probability stays UNCALIBRATED_RESEARCH_ESTIMATES. Either outcome is a result.

Honest power note: h=5 origins overlap heavily; the independent-window count (span/h) is reported next to
every number — this is a ~4-independent-window read on the 5y horizon, stated as such.

Run: python interpretability/exp_w3_calibration.py         (~15-25 min, no network)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.personal_invest_forecast import (  # noqa: E402
    load_universe, forecast, book_daily_net, cma_central, BOOKS_W3, GOALS, STRESS_WINDOWS, ENGINE_VERSION)

OUT = HERE / "exp_w3_calibration_report.json"
FIG = ROOT / "reports" / "figures" / "W3_calibration.png"
TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
DOWNSIDE_TAUS = (0.05, 0.10, 0.25, 0.50)
N_PER_MEMBER = 400
HORIZONS = (1, 5)
ORIGIN_START = {"US": "2005-01-01", "CN": "2010-01-01"}      # >=5y warmup before the first origin
BOOK_NAMES = {"US": ("SPY100", "SPY6040", "SPY_c40"), "CN": ("CSI500_TR", "CSI500_c40", "CSI500_c80")}


def realized_ann(uni, book, origin: pd.Timestamp, h: int):
    idx = (uni["rf"].index if uni["universe"] == "CN" else uni["components"]["SPY"].dropna().index)
    end = origin + pd.DateOffset(years=h)
    if idx.max() < end:
        return None
    x_full = book_daily_net(uni, book, idx.max())
    s = pd.Series(x_full, index=idx[-len(x_full):])
    w = s[(s.index > origin) & (s.index <= end)]
    if len(w) < 200 * h:
        return None
    return float(np.prod(1 + w.to_numpy()) ** (1.0 / h) - 1)


def pinball(qs: dict, realized: float, taus=TAUS) -> float:
    losses = []
    for tau in taus:
        q = qs[f"q{int(tau*100):02d}"]
        losses.append((tau - (realized < q)) * (realized - q))
    return float(np.mean(losses))


def pin_from_samples(samples: np.ndarray, realized: float, taus=DOWNSIDE_TAUS) -> float:
    losses = []
    for tau in taus:
        q = float(np.quantile(samples, tau))
        losses.append((tau - (realized < q)) * (realized - q))
    return float(np.mean(losses))


def _member_weights(rows: list, universe: str, h: int, origin: pd.Timestamp):
    """v1.2 prequential mixture weights: softmax over trailing RESOLVED downside pinballs per member;
    equal weights (None) until >=10 resolved scores exist."""
    resolved = [r for r in rows
                if r["universe"] == universe and r["h"] == h and "pin_m1_dn" in r
                and pd.Timestamp(r["origin"]) + pd.DateOffset(years=h) <= origin]
    if len(resolved) < 10:
        return None
    pins = {m: float(np.mean([r[f"pin_{m}_dn"] for r in resolved])) for m in ("m1", "m2", "m3")}
    scale = max(float(np.mean(list(pins.values()))), 1e-9)
    w = {m: float(np.exp(-pins[m] / (0.5 * scale))) for m in pins}
    s = sum(w.values())
    return {m: w[m] / s for m in w}


def stress_hit(origin: pd.Timestamp, h: int) -> bool:
    end = origin + pd.DateOffset(years=h)
    for a, b in STRESS_WINDOWS.values():
        if (origin <= pd.Timestamp(b)) and (end >= pd.Timestamp(a)):
            return True
    return False


def _preq_factor(rows: list, universe: str, h: int, origin: pd.Timestamp) -> float:
    """Prequential adaptive-conformal factor: RAW nonconformity scores of forecasts RESOLVED by the
    origin (origin' + h <= origin). Widen-only; needs >=10 resolved scores to move off 1.0."""
    scores = [abs(r["realized"] - r["q50_raw"]) / r["half80_raw"]
              for r in rows
              if r["universe"] == universe and r["h"] == h and r["half80_raw"] > 1e-9
              and pd.Timestamp(r["origin"]) + pd.DateOffset(years=h) <= origin]
    return float(max(1.0, np.quantile(scores, 0.80))) if len(scores) >= 10 else 1.0


def run_universe(universe: str, rows: list):
    uni = load_universe(universe)
    idx = (uni["rf"].index if universe == "CN" else uni["components"]["SPY"].dropna().index)
    for h in HORIZONS:
        origins = pd.date_range(ORIGIN_START[universe], idx.max() - pd.DateOffset(years=h), freq="6MS")
        for book in [b for b in BOOKS_W3[universe] if b["name"] in BOOK_NAMES[universe]]:
            for o in origins:
                o = idx[idx <= o].max()          # snap to a trading day
                real = realized_ann(uni, book, o, h)
                if real is None:
                    continue
                factor = _preq_factor(rows, universe, h, o)
                mw = _member_weights(rows, universe, h, o)
                try:
                    ens = forecast(uni, book, o, h, n_per_member=N_PER_MEMBER, seed=311,
                                   conformal=factor, member_weights=mw)
                    base = forecast(uni, book, o, h, n_per_member=N_PER_MEMBER, seed=311,
                                    members=("m1",), param_uncertainty=False)
                except ValueError:
                    continue
                samples = ens["_samples_nominal"]
                # trailing-sample-mean central (the G4 comparator)
                x_hist = book_daily_net(uni, book, o)
                sm_central = float(x_hist.mean() * 252)
                rows.append({
                    "universe": universe, "book": book["name"], "h": h, "origin": str(o.date()),
                    "realized": round(real, 4), "stress": stress_hit(o, h),
                    "q50_raw": ens["q50_raw"], "half80_raw": ens["half80_raw"],
                    "conformal_factor": round(float(factor), 4),
                    "pin_ens": pinball(ens["quantiles_net_nominal_ann"], real),
                    "pin_base": pinball(base["quantiles_net_nominal_ann"], real),
                    "pin_ens_dn": pinball(ens["quantiles_net_nominal_ann"], real, DOWNSIDE_TAUS),
                    "pin_base_dn": pinball(base["quantiles_net_nominal_ann"], real, DOWNSIDE_TAUS),
                    **{f"pin_{m}_dn": pin_from_samples(ens["_member_samples"][m], real)
                       for m in ("m1", "m2", "m3") if m in ens["_member_samples"]},
                    "member_weights": mw or "equal",
                    "pit": round(float((samples <= real).mean()), 4),
                    "in50": ens["quantiles_net_nominal_ann"]["q25"] <= real <= ens["quantiles_net_nominal_ann"]["q75"],
                    "in80": ens["quantiles_net_nominal_ann"]["q10"] <= real <= ens["quantiles_net_nominal_ann"]["q90"],
                    "in90": ens["quantiles_net_nominal_ann"]["q05"] <= real <= ens["quantiles_net_nominal_ann"]["q95"],
                    # the overconfident null (TEETH): samples shrunk 0.5x around the median
                    "in80_overconf": (lambda med, sh: np.quantile(sh, 0.10) <= real <= np.quantile(sh, 0.90))(
                        np.median(samples), np.median(samples) + 0.5 * (samples - np.median(samples))),
                    "cma_err": abs(ens["cma_central_net_nominal"] - real),
                    "sm_err": abs(sm_central - real),
                    "p_infl": ens["P_nominal"]["inflation"],
                    "hit_infl": real > GOALS[universe]["inflation"],
                })
        print(f"  [{universe}] h={h}: {sum(1 for r in rows if r['h'] == h and r['universe'] == universe)} scored origins",
              flush=True)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print(f"=== W3 calibration — rolling-origin prequential evaluation of {ENGINE_VERSION} ===")
    rows = []
    for universe in ("US", "CN"):
        run_universe(universe, rows)
    df = pd.DataFrame(rows)

    gates, detail = {}, {}
    for universe in ("US", "CN"):
        u = df[df["universe"] == universe]
        # v1.2 PRIMARY proper score = downside pinball (pre-declared); full pinball reported secondary
        g1 = float(u["pin_ens_dn"].mean()) <= float(u["pin_base_dn"].mean()) * 1.0001
        g1_full = float(u["pin_ens"].mean()) <= float(u["pin_base"].mean()) * 1.0001
        per_h = {}
        g2_all, g3_all = True, True
        for h in HORIZONS:
            s = u[u["h"] == h]
            cov50, cov80, cov90 = s["in50"].mean(), s["in80"].mean(), s["in90"].mean()
            ok2 = (abs(cov50 - 0.50) <= 0.10) and (abs(cov80 - 0.80) <= 0.10) and (cov90 >= 0.80)
            st = s[s["stress"]]
            cov80_st = float(st["in80"].mean()) if len(st) else float("nan")
            ok3 = (not len(st)) or cov80_st >= 0.70
            g2_all &= ok2; g3_all &= ok3
            per_h[h] = {"n_origins": int(len(s)), "independent_windows_approx": int(len(s) // (2 * h * len(BOOK_NAMES[universe])) + 1),
                        "cov50": round(float(cov50), 3), "cov80": round(float(cov80), 3),
                        "cov90": round(float(cov90), 3), "G2_ok": bool(ok2),
                        "stress_n": int(len(st)), "cov80_stress": round(cov80_st, 3) if len(st) else None,
                        "G3_ok": bool(ok3),
                        "pinball_ens": round(float(s["pin_ens"].mean()), 5),
                        "pinball_base": round(float(s["pin_base"].mean()), 5),
                        "pinball_dn_ens": round(float(s["pin_ens_dn"].mean()), 5),
                        "pinball_dn_base": round(float(s["pin_base_dn"].mean()), 5)}
        h5 = u[u["h"] == 5]
        g4 = float(h5["cma_err"].mean()) <= float(h5["sm_err"].mean())
        # TEETH at h=1: the shrunken forecaster must violate the cov80 band
        t1 = u[u["h"] == 1]
        cov80_over = float(t1["in80_overconf"].mean())
        teeth = abs(cov80_over - 0.80) > 0.10 and cov80_over < t1["in80"].mean()
        # Kitces reliability read (report-only): forecasts with P(infl) in [0.70, 0.97]
        kb = u[(u["p_infl"] >= 0.70) & (u["p_infl"] <= 0.97)]
        kitces = {"n": int(len(kb)), "mean_p": round(float(kb["p_infl"].mean()), 3) if len(kb) else None,
                  "realized_freq": round(float(kb["hit_infl"].mean()), 3) if len(kb) else None}
        gates[universe] = {"G1_proper_score": bool(g1), "G2_calibration_bands": bool(g2_all),
                           "G3_stress_coverage": bool(g3_all), "G4_cma_backcast": bool(g4),
                           "TEETH_overconf_rejected": bool(teeth)}
        detail[universe] = {"G1_full_pinball_secondary": bool(g1_full),
                            "per_horizon": per_h,
                            "g4_mae_cma": round(float(h5["cma_err"].mean()), 4),
                            "g4_mae_sample_mean": round(float(h5["sm_err"].mean()), 4),
                            "teeth_cov80_overconfident": round(cov80_over, 3),
                            "kitces_band_reliability": kitces}
        print(f"[{universe}] G1 {g1} | G2 {g2_all} | G3 {g3_all} | G4 {g4} (CMA {detail[universe]['g4_mae_cma']} "
              f"vs SM {detail[universe]['g4_mae_sample_mean']}) | TEETH {teeth} (overconf cov80 {cov80_over:.2f})")
        for h in HORIZONS:
            print(f"    h={h}: {per_h[h]}")

    all_pass = all(all(g.values()) for g in gates.values())
    verdict = ("ALL GATES PASS -> engine bands may be labeled CALIBRATED_V1 within the measured errors "
               "(honest power caveat: few independent long-horizon windows)"
               if all_pass else
               "GATES NOT ALL PASSED -> every engine probability stays UNCALIBRATED_RESEARCH_ESTIMATES; "
               "the failing gate(s) name the next work item")
    # figure
    fig_ok = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for j, universe in enumerate(("US", "CN")):
            u = df[df["universe"] == universe]
            ax = axes[0][j]
            labels, nominal, observed = [], [0.5, 0.8, 0.9], []
            for h in HORIZONS:
                s = u[u["h"] == h]
                for c, nm in ((s["in50"].mean(), f"50%@{h}y"), (s["in80"].mean(), f"80%@{h}y"),
                              (s["in90"].mean(), f"90%@{h}y")):
                    labels.append(nm); observed.append(c)
            ax.bar(range(len(labels)), observed, color="steelblue")
            for i, nom in enumerate([0.5, 0.8, 0.9] * len(HORIZONS)):
                ax.hlines(nom, i - 0.4, i + 0.4, color="red")
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_title(f"{universe}: interval coverage (red = nominal)"); ax.set_ylim(0, 1.05)
            ax2 = axes[1][j]
            ax2.hist(u["pit"], bins=10, range=(0, 1), color="darkseagreen", edgecolor="k")
            ax2.axhline(len(u) / 10, color="red", ls="--")
            ax2.set_title(f"{universe}: PIT histogram (flat = calibrated)")
        fig.suptitle("W3 prequential calibration (rolling origins; research diagnostic)", fontsize=11)
        fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=130)
        fig_ok = True
    except Exception as e:
        print("figure failed:", e)

    rep = {"experiment": "W3 rolling-origin prequential calibration + exit gates",
           "engine_version": ENGINE_VERSION, "n_scored": int(len(df)),
           "pre_specified_gates": {
               "G1": "ensemble mean pinball <= unconditional baseline, per universe",
               "G2": "|cov50-.5|<=.10, |cov80-.8|<=.10, cov90>=.80 per universe x horizon",
               "G3": "cov80 >= .70 on stress-intersecting origins",
               "G4": "CMA central MAE <= sample-mean central MAE at h=5",
               "TEETH": "0.5x-shrunk forecaster must violate the cov80 band"},
           "gates": gates, "detail": detail, "verdict": verdict,
           "figure": str(FIG.relative_to(ROOT)).replace("\\", "/") if fig_ok else None,
           "rows": df.drop(columns=["in80_overconf"]).to_dict("records")}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nVERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
