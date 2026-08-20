"""HL AGENT ABLATION — does the coding-agent loop actually help? WITHOUT vs WITH, >=50 cycles, applied
ITERATIVELY (each accepted candidate joins the incumbent; the next cycle builds on it).

Design (declared before the run):
  * CYCLE 0 = the model WITHOUT the coding-agent: the shipped base menu + the DP champion per profile
    (engine v1.4). This is the leftmost point of every learning curve.
  * Cycles 1..13 per profile (5 profiles x 13 = 65 >= 50): the agent's OWN candidate sequence, ordered as
    a real exploration (broad families first — the four diversifiers; then adaptive refinements that USE
    the lessons of earlier cycles: the best diversifier at a higher equity share, a three-asset mix, rung
    ladders, boundary re-sizing, execution deadzones, a TIP novelty). The sequence embodies the HL-LIVE
    session's strategy; adaptivity is explicit (tokens like BEST_DIV resolve from accepted history).
  * ACCEPT rule: the candidate's DEV q20-floor must beat the incumbent's by > 10bp (guards seed noise);
    accepted -> the incumbent mutates (boundary books REPLACE same-family predecessors; rungs/deadzones
    accumulate). Rejected -> the incumbent stands (curves show honest plateaus).
  * ANTI-OVERFIT rail: every accept is immediately re-read on the HOLD seed; both curves are plotted.
    A final full control read (dd-matched twin + sized matched-random) runs on the final incumbent.
  * Metrics per cycle (the plot series): q20 floor (the 80%-promise), median return, P(profile goal),
    E[shortfall|miss] — all from the incumbent AFTER the cycle's decision.
RESEARCH-ONLY on uncalibrated v1.4 scenarios; the promise's REALITY check remains the part-B backtest.

Run: python interpretability/exp_hl_agent_ablation.py        (~10-20 min, no network)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import interpretability.hl_profile_cycles as hpc  # noqa: E402
from interpretability.hl_profile_cycles import (  # noqa: E402
    PROFILES, evaluate, sized_book, probe_mix_dd, _PROBE_DD, EQ_KEY)
from interpretability.hl_live_session import get_uni  # noqa: E402
from interpretability.personal_invest_forecast import ENGINE_VERSION  # noqa: E402

OUT = HERE / "exp_hl_agent_ablation_report.json"
DEV_SEED, HOLD_SEED = 99, 199
# ACCEPT on the user's DECLARED objective J = E[U] = P(goal) - lam*E[shortfall] (the risk-adaptive
# model's contract metric); the q20 floor is reported alongside. The companion FLOOR-objective run
# (exp_hl_agent_ablation_report_floor.json) showed that axis nearly saturated post-v1.4 — itself a
# finding: the honest-promise axis is hard to move; the contract axis is where machinery lives.
ACCEPT_J = 0.005


def agent_sequence(prof):
    """The agent's exploration order for this profile: broad -> adaptive -> refine -> novel."""
    B = prof["budget"]
    return [
        {"id": "div_IEF",  "kind": "boundary", "div": "IEF",     "share": 0.45, "frac": 0.95},
        {"id": "div_GLD",  "kind": "boundary", "div": "GLD",     "share": 0.45, "frac": 0.95},
        {"id": "div_TLT",  "kind": "boundary", "div": "TLT",     "share": 0.45, "frac": 0.95},
        {"id": "div_MIX",  "kind": "boundary", "div": "IEF_GLD", "share": 0.45, "frac": 0.95},
        {"id": "best_at_65", "kind": "boundary", "div": "BEST",  "share": 0.65, "frac": 0.95},
        {"id": "tri_asset", "kind": "boundary", "div": "IEF_GLD", "share": 0.34, "frac": 0.95},
        {"id": "mid_rung",  "kind": "rung", "div": "BEST", "share": "BEST", "frac": 0.60},
        {"id": "low_rung",  "kind": "rung", "div": "BEST", "share": "BEST", "frac": 0.35},
        {"id": "tighten_098", "kind": "boundary", "div": "BEST", "share": "BEST", "frac": 0.98},
        {"id": "loosen_090",  "kind": "boundary", "div": "BEST", "share": "BEST", "frac": 0.90},
        {"id": "deadzone_010", "kind": "deadzone", "value": 0.10},
        {"id": "deadzone_015", "kind": "deadzone", "value": 0.15},
        {"id": "novel_TIP", "kind": "rung", "div": "TIP_MIX", "share": 0.45, "frac": 0.80},
    ]


def resolve(cand, state, uni, prof, as_of):
    """Materialize a candidate against the CURRENT incumbent (adaptive tokens resolve here)."""
    div = cand.get("div")
    share = cand.get("share")
    if div == "BEST":
        div = state["best_div"] or "IEF"
    if div == "TIP_MIX":
        hpc.DIVS["US"]["TIP_MIX"] = {"TIP": 0.5, state["best_div"] or "IEF": 0.5}
    if share == "BEST":
        share = state["best_share"] or 0.45
    if cand["kind"] == "deadzone":
        return state["books"], cand["value"]
    b = sized_book(uni, share, div, cand["frac"] * prof["budget"], as_of)
    books = [x for x in state["books"]
             if not (cand["kind"] == "boundary" and x.get("_tag") == "boundary")]
    b["_tag"] = "boundary" if cand["kind"] == "boundary" else "rung"
    return books + [b], state["deadzone"]


def metrics(uni, prof, books, deadzone, seed, as_of):
    clean = [{k: v for k, v in b.items() if k != "_tag"} for b in books]
    r = evaluate(uni, prof, clean, seed, as_of, deadzone=deadzone)
    return r


def run_profile(prof, uni, as_of):
    print(f"\n=== {prof['id']} (H={prof['H']}y, DD<={prof['budget']:.0%}) ===")
    _PROBE_DD.clear()
    state = {"books": [], "deadzone": 0.0, "best_div": None, "best_share": None}
    base = metrics(uni, prof, [], 0.0, DEV_SEED, as_of)
    base_hold = metrics(uni, prof, [], 0.0, HOLD_SEED, as_of)
    traj = [{"cycle": 0, "candidate": "NO-AGENT BASELINE", "accepted": None,
             "dev": base, "hold": base_hold}]
    incumbent_dev, incumbent_hold = base, base_hold
    for i, cand in enumerate(agent_sequence(prof), start=1):
        books, dz = resolve(cand, state, uni, prof, as_of)
        dev = metrics(uni, prof, books, dz, DEV_SEED, as_of)
        accepted = bool(dev and incumbent_dev and
                        dev["J_contract"] > incumbent_dev["J_contract"] + ACCEPT_J)
        hold = None
        if accepted:
            hold = metrics(uni, prof, books, dz, HOLD_SEED, as_of)
            state["books"], state["deadzone"] = books, dz
            incumbent_dev, incumbent_hold = dev, hold or incumbent_hold
            if cand["kind"] in ("boundary", "rung") and isinstance(cand.get("div"), str):
                b_last = books[-1]
                nm = b_last["name"].split("_")            # B_S45_IEF[_GLD]_aXX
                state["best_div"] = "_".join(nm[2:-1])
                state["best_share"] = float(nm[1][1:]) / 100
        traj.append({"cycle": i, "candidate": cand["id"], "accepted": accepted,
                     "dev": incumbent_dev, "hold": incumbent_hold,
                     "candidate_dev_J": dev["J_contract"] if dev else None})
        print(f"  cycle {i:02d} {cand['id']:14s}: cand J "
              f"{(dev['J_contract'] if dev else float('nan')):+.4f} -> "
              f"{'ACCEPT' if accepted else 'reject'} | incumbent J dev {incumbent_dev['J_contract']:+.4f}"
              f" hold {incumbent_hold['J_contract']:+.4f} | P(goal) {incumbent_hold['P_profile_goal']:.2f}")
    # final control read on the final incumbent (only if the agent changed anything)
    controls = None
    if state["books"]:
        eq = EQ_KEY["US"]
        from interpretability.hl_live_session import cached_kernels
        dd_eq = abs(cached_kernels(uni, [{"name": "EQ100p", "weights": {eq: 1.0}}], as_of)
                    ["EQ100p"]["dd_p95"])
        tw = min(0.95, 0.95 * prof["budget"] / dd_eq)
        twin = metrics(uni, prof, [{"name": "TWIN", "weights": {eq: round(tw, 2),
                                                                 "CASH": round(1 - tw, 2)}}],
                       0.0, HOLD_SEED, as_of)
        rng = np.random.default_rng(7 + prof["H"])
        x = rng.dirichlet([1, 1, 1])
        rnd_mix = {"name": "RND", "weights": {eq: round(x[0], 2), "IEF": round(x[1], 2),
                                               "CASH": round(1 - round(x[0], 2) - round(x[1], 2), 2)}}
        kr = cached_kernels(uni, [rnd_mix], as_of)
        al = min(1.0, 0.95 * prof["budget"] / abs(kr["RND"]["dd_p95"]))
        rnd_b = {"name": "RNDs", "weights": {k: round(v * al, 3) for k, v in rnd_mix["weights"].items()}}
        rnd_b["weights"]["CASH"] = round(rnd_b["weights"].get("CASH", 0) + (1 - al), 3)
        controls = {"twin_hold_floor": twin["floor_q20_ann"] if twin else None,
                    "matched_random_hold_floor":
                        (metrics(uni, prof, [rnd_b], 0.0, HOLD_SEED, as_of) or {}).get("floor_q20_ann")}
    n_acc = sum(1 for t in traj[1:] if t["accepted"])
    print(f"  SUMMARY: {n_acc} accepts / {len(traj)-1} cycles | J {base['J_contract']:+.4f} -> "
          f"dev {incumbent_dev['J_contract']:+.4f} / hold {incumbent_hold['J_contract']:+.4f} | "
          f"P(goal) {base['P_profile_goal']:.2f} -> {incumbent_hold['P_profile_goal']:.2f} "
          f"| controls {controls}")
    return {"profile": prof["id"], "H": prof["H"], "budget": prof["budget"],
            "trajectory": traj, "n_accepts": n_acc,
            "final_books": [{b['name']: {k: v for k, v in b['weights'].items()}}
                            for b in state["books"]],
            "final_deadzone": state["deadzone"], "final_controls": controls}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    hpc.UNIVERSE = "US"
    print(f"=== HL AGENT ABLATION: no-agent baseline vs 65 iterative agent cycles ({ENGINE_VERSION}) ===")
    uni = get_uni("US")
    as_of = uni["components"]["SPY"].dropna().index.max()
    report = {"experiment": "HL agent ablation — WITHOUT (cycle 0) vs WITH the coding-agent, iterative",
              "engine": ENGINE_VERSION, "status": "RESEARCH_ONLY_UNCALIBRATED_SCENARIOS",
              "accept_rule": f"dev J_contract improvement > {ACCEPT_J}; hold re-read on accept",
              "profiles": []}
    for prof in PROFILES:
        report["profiles"].append(run_profile(prof, uni, as_of))
    total = sum(len(p["trajectory"]) - 1 for p in report["profiles"])
    report["total_cycles"] = total
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\ntotal agent cycles: {total}; wrote {OUT.name}")


if __name__ == "__main__":
    main()
