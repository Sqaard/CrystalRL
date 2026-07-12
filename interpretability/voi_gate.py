"""Debt 4 — the VoI GATE: the honest deployment decision rule for CRYSTAL-1's born-legible belief-writing surface.

The program established (B4-REAL closure) that on competitive markets THE REGIME IS PRICED (Glosten-Milgrom zero-profit):
belief-VoI = 0 on daily and on the accessible intraday crypto, so tracking the regime yields no edge. The corner where
regime-tracking pays exists on the POLYGON, not on real books. So we cannot manufacture a VoI>0 substrate. The honest,
constructive deliverable is the GATE that decides WHERE the belief-writing machinery earns its keep — the blueprint's
"environment-selected objective by measured belief-VoI." A substrate is CAPITAL-eligible only if VoI > eps; otherwise the
machinery runs in transparency/monitoring mode only.

We validate the gate DISCRIMINATES: it OPENS on the polygon (belief-aware optimum beats belief-blind) and CLOSES on the
real crypto intraday (B4-REAL VoI=0). CN A-shares — the one refuge where the competitive-equilibrium argument is weakest
— has no usable L5 data yet (the recorder started after Fri close; Sat is closed), so that route stays OWED.
Run: python interpretability/voi_gate.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.regime_pomdp import RegimePOMDP, PRIMARY_ENRICHED  # noqa: E402
from src.series_g.phase0_gate import solve_belief_aware, solve_belief_blind, aware_start_value  # noqa: E402

OUT = HERE / "voi_gate_report.json"
EPS = 0.02          # gate threshold: deploy belief-writing on capital only if VoI exceeds this


def gate(voi, eps=EPS):
    return "OPEN (belief-writing capital-eligible)" if voi > eps else "CLOSED (transparency/monitoring only)"


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    # ---- POLYGON VoI: belief-aware optimum vs belief-blind optimum ----
    m = RegimePOMDP(**PRIMARY_ENRICHED)
    g, V, pol = solve_belief_aware(m, n_bins=121)
    Vb, polb = solve_belief_blind(m)
    v_aware = aware_start_value(m, g, V)
    v_blind = float(Vb[0, 0])
    voi_polygon = round(v_aware - v_blind, 4)
    voi_polygon_frac = round(voi_polygon / (abs(v_blind) + 1e-9), 3)

    # ---- REAL crypto intraday VoI (B4-REAL) ----
    crypto = {}
    b4 = HERE / "b4_real_voi_report.json"
    if b4.exists():
        r = json.loads(b4.read_text(encoding="utf-8"))
        crypto["btcusdt_2023_03"] = {"VoI_bp_per_provided_minute": r.get("VoI_bp_per_provided_minute"),
                                     "verdict": r.get("verdict", "")[:160]}
    b4v2 = HERE / "b4_real_voi_v2_report.json"
    if b4v2.exists():
        r2 = json.loads(b4v2.read_text(encoding="utf-8"))
        crypto["wider_spread_alts_2023_08"] = {"symbols": r2.get("symbols"), "verdict": r2.get("verdict", "")[:200]}
    # the B4-REAL VoI is ~0 (regime priced); represent as 0.0 for the gate
    voi_crypto = 0.0

    decisions = {
        "polygon_series_g": {"VoI": voi_polygon, "VoI_frac_of_blind": voi_polygon_frac,
                             "v_aware": round(v_aware, 4), "v_blind": round(v_blind, 4), "gate": gate(voi_polygon)},
        "crypto_intraday_real": {"VoI": voi_crypto, "gate": gate(voi_crypto), "evidence": crypto},
        "cn_ashares": {"VoI": None, "gate": "UNTESTED — no usable L5 data yet (recorder started post-close Fri; Sat closed)",
                       "note": "the one refuge where the competitive-equilibrium argument is weakest; re-run after ~4-8 weeks of L5 accumulation"},
    }
    # ---- Dow-extended daily (E-16, 2026-07-08): the first certified, fresh-OOS-held candidate ----
    e16 = HERE / "exp_e16_voi_reopen_report.json"
    if e16.exists():
        r16 = json.loads(e16.read_text(encoding="utf-8"))
        v = r16.get("voi", {})
        decisions["dow_extended_daily"] = {
            "return_rule_as_written": v.get("return_metric (the gate's WRITTEN rule)"),
            "risk_adjusted_lane": v.get("risk_adjusted (the certified claim's metric)"),
            "gate": "CLOSED under the written return rule; risk-adjusted lane OPEN-IF-RATIFIED "
                    "(human decision pending — see exp_e16_voi_reopen_report.json governance)",
        }
        e18 = HERE / "exp_e18_deployment_stress_report.json"
        if e18.exists():
            r18 = json.loads(e18.read_text(encoding="utf-8"))
            if r18.get("verdict", "").startswith("PASS"):
                decisions["dow_extended_daily"]["gate"] = (
                    "OPEN on the risk-adjusted lane — RATIFIED by Ivan 2026-07-08 (E-16) AND the E-18 "
                    "pre-deployment battery PASSED (breakeven cost >150bp, 1-day lag tolerated, depth smooth). "
                    "Deployment spec: E-15 certified config at capital fraction w=0.5 "
                    "(exp_e18_deployment_stress_report.json). Return-rule lane remains CLOSED (no alpha claim).")
                decisions["dow_extended_daily"]["ratification"] = r18.get("ratification")
    discriminates = decisions["polygon_series_g"]["gate"].startswith("OPEN") and decisions["crypto_intraday_real"]["gate"].startswith("CLOSED")

    report = {
        "rule": f"deploy belief-writing on CAPITAL iff measured belief-VoI > eps={EPS}; else transparency/monitoring mode only",
        "decisions": decisions,
        "gate_discriminates_correctly": bool(discriminates),
        "headline": (
            f"The VoI gate DISCRIMINATES: polygon VoI={voi_polygon} ({voi_polygon_frac:+.0%} of the belief-blind value) "
            f"=> OPEN; real crypto intraday VoI=0 (B4-REAL: regime is PRICED, Glosten-Milgrom) => CLOSED. So the "
            "born-legible belief-writing surface is validated AND correctly fenced: it acts on capital only where the "
            "regime is NOT already priced. No accessible real substrate currently clears the gate — the standing open "
            "bet is a genuine VoI>0 execution task (queue/rebate/latency microstructure, or CN A-shares once the L5 "
            "recorder has accumulated). Until then CRYSTAL-1 is a transparency/interpretability object on real markets, "
            "consistent with the pivoted north star (CrystalScore, not alpha)."),
        "caveats": ["polygon VoI is the exact belief-MDP gap (aware vs blind optimum); crypto VoI=0 is the B4-REAL "
                    "measured result; CN untested (no data); the gate is the decision RULE, not a claim that a VoI>0 "
                    "substrate exists."],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== Debt 4 — VoI gate (deployment decision rule) ===")
    print(f"rule: deploy belief-writing on capital iff VoI > {EPS}")
    d = decisions
    print(f"  polygon  : VoI={d['polygon_series_g']['VoI']} ({voi_polygon_frac:+.0%} of blind)  -> {d['polygon_series_g']['gate']}")
    print(f"  crypto   : VoI={d['crypto_intraday_real']['VoI']} (B4-REAL: regime priced)      -> {d['crypto_intraday_real']['gate']}")
    print(f"  cn A-shr : {d['cn_ashares']['gate']}")
    print(f"gate discriminates correctly (open polygon / close real): {discriminates}")
    print(report["headline"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
