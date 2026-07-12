"""E-04 — G12 (non-saturated) certified legibility-raising rule. Joseph's interpretability contribution.

E-01 ran the v4 Pareto loop on G6 (saturated: MDL deficit ~0, legibility is FREE) — that only CONFIRMED the
infrastructure. The real interpretability test is G12, where legibility is NOT free (12 hidden regimes, K<=9
concepts -> the belief->action map cannot be fully explained, so MDL deficit stays > 0).

This driver runs the loop on G12 ONLY, over a seed population, and — beyond the final frontier — logs the
per-ACCEPTED-move MDL-deficit trajectory. A "certified legibility-raising rule" = a gate-ACCEPTED move that
LOWERS the MDL deficit (raises Simulatability) WITHOUT regressing return. That is the PAPER Table 3 deliverable.

Honest: if NO accepted move lowers MDL deficit at flat/positive return, that is a valid NULL — on G12 legibility
is not freely raisable, and we say so (REFUSED/NULL Table 3), we do not manufacture a row.

Run:  python interpretability/hl_v4_g12.py [--smoke]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.hl import modular_rule_policy as M          # noqa: E402
from src.hl import tension as T                       # noqa: E402
from src.hl.pareto_gate import ParetoGate             # noqa: E402
from src.hl.mechanism_bandit import MechanismBandit   # noqa: E402
from interpretability.hl_v4_loop import ctors, OPS     # noqa: E402  (reuse the exact substrate ctors + operators)

MDL_NS = 20            # n_seeds for the per-move MDL deficit (speed); final frontier uses the loop default
LEG_MDL = 0.02         # a move "raises legibility" if MDL deficit drops by >= this
RET_TOL = 0.10         # ...without regressing return by more than this (in return units)
OUT = HERE / "hl_v4_g12_report.json"


def run_g12(rounds, seed):
    G = 12
    base, stress = ctors(G)
    gate = ParetoGate(dev_ctor=base, stressor_ctors=stress, G=G,
                      dev_seeds=list(range(200 + seed * 7, 400 + seed * 7)))
    bandit = MechanismBandit(arms=list(OPS))
    rng = np.random.default_rng(seed)
    current = [M.unwind_clause()]
    gate.frontier = [(list(current), gate._vec(current, gate.dev_seeds))]
    cur_vec = gate.frontier[0][1]
    cur_mdl = float(T.mdl_deficit(current, base, G, n_seeds=MDL_NS))
    traj, counts = [], {}
    for rnd in range(rounds):
        if rnd > 0 and rnd % 10 == 0:
            gate.canary_check(current)
        cov = sorted(T.M.covered_venues(current)); uncov = [v for v in range(G) if v not in cov]
        avail = [o for o in OPS if not (o == "add_rule" and not uncov) and not (o == "joint_move" and len(uncov) < 2)
                 and not (o == "retune" and not cov) and not (o == "recombine" and len(gate.frontier) < 2)]
        if not avail:
            break
        op = bandit.select(avail)
        if op == "add_rule":
            cand = M.add_provide(current, int(rng.choice(uncov)), thr=0.556)
        elif op == "joint_move":
            vs = list(rng.choice(uncov, 2, replace=False)); cand = current
            for v in vs:
                cand = M.add_provide(cand, int(v), thr=0.556)
        elif op == "retune":
            cand = list(current); idxs = [i for i, c in enumerate(cand) if c[0] == "provide"]; i = int(rng.choice(idxs))
            _, v, thr, cap = cand[i]; cand[i] = ("provide", v, float(min(0.95, thr + 0.12)), cap)
        else:
            fa, fb = (gate.frontier[i][0] for i in rng.choice(len(gate.frontier), 2, replace=False))
            cand = M.recombine(fa, fb)
        verdict, info, new = gate.review(cand, current)
        counts[verdict] = counts.get(verdict, 0) + 1
        accepted = verdict.startswith("ACCEPTED")
        bandit.update(op, 1.0 if accepted else 0.0)
        if accepted:
            new_vec = gate._vec(new, gate.dev_seeds)                 # robust: recompute the tension vector
            new_mdl = float(T.mdl_deficit(new, base, G, n_seeds=MDL_NS))
            traj.append({"round": rnd, "op": op,
                         "ret_before": round(float(cur_vec["return"]), 3), "ret_after": round(float(new_vec["return"]), 3),
                         "dlen_before": int(cur_vec["description_len"]), "dlen_after": int(new_vec["description_len"]),
                         "mdl_before": round(cur_mdl, 3), "mdl_after": round(new_mdl, 3),
                         "d_mdl": round(new_mdl - cur_mdl, 3), "d_ret": round(float(new_vec["return"]) - float(cur_vec["return"]), 3)})
            current = new; cur_vec = new_vec; cur_mdl = new_mdl
    front = sorted(gate.frontier, key=lambda pv: pv[1]["return"])
    front_pts = [{"return": round(v["return"], 2), "desc_len": int(v["description_len"]),
                  "mdl_deficit": round(T.mdl_deficit(p, base, G), 3)} for p, v in front if v["description_len"] > 0]
    return {"seed": seed, "G": G, "gate_counts": counts, "audit": gate.audit,
            "frontier": front_pts, "trajectory": traj, "gate_compromised": gate.compromised}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    smoke = "--smoke" in sys.argv
    seeds = [0] if smoke else [0, 1, 2]
    rounds = 6 if smoke else 45
    print(f"=== E-04 — G12 legibility-raising rule attempt (seeds={seeds}, rounds={rounds}, smoke={smoke}) ===")
    runs = [run_g12(rounds, s) for s in seeds]

    # certified legibility-raising moves: gate-ACCEPTED, MDL deficit down >= LEG_MDL, return not regressed > RET_TOL
    leg = []
    for r in runs:
        for m in r["trajectory"]:
            if m["d_mdl"] <= -LEG_MDL and m["d_ret"] >= -RET_TOL:
                leg.append({**m, "seed": r["seed"]})
    leg.sort(key=lambda m: (m["d_mdl"], -m["d_ret"]))               # most legibility-raising first

    def best_ret_mdl(r):
        if not r["frontier"]:
            return None
        p = max(r["frontier"], key=lambda x: x["return"]); return p["mdl_deficit"]
    nonsat = [best_ret_mdl(r) for r in runs]
    is_nonsat = any((x is not None and x > 0.05) for x in nonsat)

    all_counts = {}
    for r in runs:
        for k, v in r["gate_counts"].items():
            all_counts[k] = all_counts.get(k, 0) + v
    canary = {"caught": sum(r["audit"].get("canary_caught", 0) for r in runs),
              "escaped": sum(r["audit"].get("canary_escaped", 0) for r in runs),
              "compromised": [r["gate_compromised"] for r in runs]}

    rep = {"experiment": "E-04 G12 certified legibility-raising rule (Joseph, interpretability track)",
           "setup": {"G": 12, "seeds": seeds, "rounds": rounds, "mdl_n_seeds_traj": MDL_NS,
                     "legibility_rule_def": f"gate-ACCEPTED move with d_mdl<=-{LEG_MDL} AND d_ret>=-{RET_TOL}"},
           "nonsaturation_check": {"best_return_mdl_deficit_per_seed": nonsat,
                                    "verdict": "NON-SATURATED (legibility is NOT free on G12)" if is_nonsat
                                    else "inconclusive/saturated (MDL deficit ~0 at best return)"},
           "n_certified_legibility_moves": len(leg),
           "certified_legibility_raising_moves": leg[:5],
           "frontiers": {f"seed{r['seed']}": r["frontier"] for r in runs},
           "gate_counts_union": all_counts, "canary": canary,
           "table3_row": None, "verdict": None}
    if leg:
        b = leg[0]
        rep["table3_row"] = (f"G12 {b['op']} (seed {b['seed']}): MDL deficit {b['mdl_before']}->{b['mdl_after']} "
                             f"(Δ{b['d_mdl']}), return {b['ret_before']}->{b['ret_after']} (Δ{b['d_ret']}) — "
                             f"gate-certified: Simulatability RAISED with return non-regressed.")
        rep["verdict"] = "CONFIRMED — a legibility-raising rule certified on the non-saturated G12 substrate"
    else:
        rep["table3_row"] = "none certified — on G12 no gate-accepted move lowered MDL deficit without regressing return"
        rep["verdict"] = "NULL — no certified legibility-raising rule on G12 this run (legibility is not freely raisable)"

    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("non-saturation:", rep["nonsaturation_check"])
    print(f"certified legibility-raising moves: {len(leg)}")
    for m in leg[:5]:
        print(f"   seed{m['seed']} r{m['round']} {m['op']}: MDL {m['mdl_before']}->{m['mdl_after']} (Δ{m['d_mdl']}) "
              f"ret {m['ret_before']}->{m['ret_after']} (Δ{m['d_ret']})")
    print("verdict:", rep["verdict"])
    print("Table 3 row:", rep["table3_row"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
