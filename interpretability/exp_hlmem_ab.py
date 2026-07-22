"""HL-MEM A/B — the preregistered 4-arm test: does memory between attempts earn its keep?

Per reports/MEMORY_BETWEEN_ATTEMPTS_LIT_20260720.md. REUSES the reviewed v9 machinery verbatim
(Crystal1V4Gate, strat_v8/KNOBS8/ARMS8, MechanismBandit, dev 2019-21 + hold 2022-23; the frozen
OOS is NEVER loaded). The gate is IDENTICAL and UNCHANGED across arms (shared core). Canaries are
gate-integrity machinery orthogonal to the arm comparison and are skipped for runtime.

WHAT THIS MEASURES (honest scope): the procedural SKELETON of HL-MEM — exclusion, structured
deviation conditioning, hit-rate cards, portfolio islands. The LLM hypothesis-WRITER is not in
this A/B; per the lit analysis the deterministic components carry the largest measured share of
feedback value, so this is the right first claim to test.

ARMS (same base proposer lineage, additive increments; budget = 40 GATE QUERIES per run — every
gate.review() call counts, incl. free inert refusals; linter refusals do NOT consume gate budget
but count toward N and toward the 200-attempt cap):
  A  amnesiac control      = the v9 proposer as-is (bandit + step/direction hill-climb).
  B  A + LINTER            = near-duplicate + falsified-direction refusal before the gate.
  C  B + DEVIATION REPORT  = the frozen-vocabulary DEV record conditions the next proposal
                             (REPAIR map: step multiplier, preferred arms, direction flip).
  D  C + CARDS + PORTFOLIO = Thompson over (arm, direction, last_bar) hit-rate cards + island
                             rotation on quiet. Sub-arms: D_live (cards update all run) and
                             D_frozen (card updates freeze after 15 queries) — the KT-D switch.

PREREGISTERED READS (before running; 5 seeds per arm; primary endpoint = certified accepts per
40-query budget, secondary = mean wealth at end, attempts N, distinct configs explored):
  R1 (the theater test): mean accepts(D_live) > mean accepts(B), one-sided Wilcoxon/permutation
     p < 0.1 across seeds. If NOT, hypothesis-conditioning is theater at this scale and ONLY the
     linter ships.
  R2 (the location term): accepts(C) - accepts(B) isolates the deviation-report value.
  R3 (the exclusion term): accepts(B) - accepts(A) isolates the linter value (predicted >= 0:
     refusing known-dead proposals cannot lose accepts, only save budget).
  R4 (the KT-D switch): accepts(D_frozen) vs accepts(D_live) — if frozen >= live, the in-house
     KT-D pattern extends to this memory and LIVE updating must be admission-gated harder.
  All raw run records ledgered to exp_hlmem_ab_ledger.jsonl; N (incl. linter refusals) recorded
  per run for the (reporting-only) Bailey deflation hook.

Run: python interpretability/exp_hlmem_ab.py           (~30-50 min CPU)
     python interpretability/exp_hlmem_ab.py --smoke   (1 seed, 10 queries — plumbing check)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import interpretability.hl_v4_over_crystal1 as V4  # noqa: E402
from interpretability.hl_v4_over_crystal1 import Crystal1V4Gate  # noqa: E402
from interpretability.hl_v8_rebalance_lane import strat_v8, n_off_anchor8, KNOBS8, ARMS8  # noqa: E402
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, window, DEV, HOLD  # noqa: E402
from src.hl.mechanism_bandit import MechanismBandit  # noqa: E402
from src.hl.hlmem import (deviation_report, REPAIR, NegativeRegistry, ProposalLinter,  # noqa: E402
                          CardStore, PortfolioScheduler)

OUT = HERE / "exp_hlmem_ab_report.json"
LEDGER = HERE / "exp_hlmem_ab_ledger.jsonl"
CERTIFIED = {"t1": 0.30, "t2": 0.657, "lvl_reduced": 1.0, "lvl_defensive": 0.738, "H": 10.0}
# v4 (referee-prescribed): the INCUMBENT is DE-TUNED (anchor levels = buy-and-hold behavior) so the
# frontier has genuine headroom - the certified-incumbent runs measured polish-regime saturation
# (90% dominated rejections). The certified config remains the search TARGET, not the start.
INCUMBENT = {"t1": 0.30, "t2": 0.70, "lvl_reduced": 1.0, "lvl_defensive": 1.0, "H": 10.0}
HOLD_WIN = 120
ABSURD_SHARPE = 3.0
BUDGET = 40
ATTEMPT_CAP = 200
SEEDS = tuple(range(20))


def ledger(rec):
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def fresh_gate(dev, hold):
    V4.n_off_anchor = n_off_anchor8
    g = Crystal1V4Gate(dev, hold, HOLD_WIN, strat_fn=strat_v8)
    cur = dict(INCUMBENT)
    g.frontier = [(dict(cur), g.vec(cur))]
    return g, cur


def run_arm(arm_name, seed, dev, hold, budget=BUDGET):
    rng = np.random.default_rng(1000 + seed)
    gate, current = fresh_gate(dev, hold)
    bandit = MechanismBandit(arms=list(ARMS8))
    step = {k: 0.1 * (v[1] - v[0]) for k, v in KNOBS8.items()}
    direction = {k: -1.0 for k in KNOBS8}
    use_linter = arm_name in ("B", "C", "D_live", "D_frozen", "E_islands")
    use_dev = arm_name in ("C", "D_live", "D_frozen")
    use_cards = arm_name in ("D_live", "D_frozen")
    use_islands = use_cards or arm_name == "E_islands"
    registry = NegativeRegistry(k=2)
    linter = ProposalLinter(registry)
    cards = CardStore(rng, frozen_after=(15 if arm_name == "D_frozen" else None)) if use_cards else None
    portfolio = PortfolioScheduler(patience=8) if use_islands else None
    last_bar = "any"
    queries = attempts = accepts = migrations = 0
    consecutive_lint, forced_arm = 0, None

    while queries < budget and attempts < ATTEMPT_CAP:
        attempts += 1
        # ---- arm selection (epsilon-greedy over the bandit gives every arm REAL seed variance) ----
        avail = portfolio.arms() if use_islands else list(ARMS8)
        if forced_arm is not None:
            arm, forced_arm = forced_arm, None
        elif rng.random() < 0.25:
            arm = avail[int(rng.integers(len(avail)))]
        elif use_cards:
            arm = cards.thompson_pick(list(dict.fromkeys(avail + ["joint_defend"])), direction, last_bar)
        elif use_dev and REPAIR.get(last_bar, REPAIR["none"])["prefer"]:
            prefs = [a for a in REPAIR[last_bar]["prefer"] if a in ARMS8]
            arm = prefs[int(rng.integers(len(prefs)))] if (prefs and rng.random() < 0.6) else bandit.select(ARMS8)
        else:
            arm = bandit.select([a for a in avail if a in ARMS8] or list(ARMS8))
        # ---- candidate construction (v9 propose() + shared step jitter; no-op flip at bounds) ----
        jitter = float(rng.uniform(0.6, 1.4))
        cand = dict(current)
        if arm == "joint_defend":
            cand["t2"] = float(np.clip(cand["t2"] - step["t2"] * jitter, *KNOBS8["t2"][:2]))
            cand["lvl_defensive"] = float(np.clip(cand["lvl_defensive"] - step["lvl_defensive"] * jitter,
                                                  *KNOBS8["lvl_defensive"][:2]))
            sign = -1
        else:
            lo, hi, _ = KNOBS8[arm]
            nxt = float(np.clip(cand[arm] + step[arm] * jitter * direction[arm], lo, hi))
            if abs(nxt - cand[arm]) < 1e-9:               # clipped no-op at a bound: flip and retry once
                direction[arm] = -direction[arm]
                nxt = float(np.clip(cand[arm] + step[arm] * jitter * direction[arm], lo, hi))
            cand[arm] = nxt
            sign = int(np.sign(direction[arm]))
        # ---- linter (B/C/D): refuse without spending gate budget ----
        if use_linter:
            lint = linter.check(arm, sign, cand)
            if lint is not None:
                consecutive_lint += 1
                ledger({"arm_name": arm_name, "seed": seed, "attempt": attempts, "lint": lint,
                        "knob": arm})
                if arm != "joint_defend":
                    step[arm] = float(np.clip(step[arm] * float(rng.uniform(0.5, 2.0)), 1e-4,
                                              0.5 * (KNOBS8[arm][1] - KNOBS8[arm][0])))
                    if lint == "LINT_FALSIFIED_DIRECTION" or rng.random() < 0.5:
                        direction[arm] = -direction[arm]
                        if lint == "LINT_FALSIFIED_DIRECTION":
                            registry.clear(arm, sign)
                if consecutive_lint >= 3:                  # starvation guard: force a different knob
                    others = [a for a in ARMS8 if a != arm]
                    forced_arm = others[int(rng.integers(len(others)))]
                    consecutive_lint = 0
                continue
            consecutive_lint = 0
            linter.commit(cand)
        # ---- absurdity pre-check (v9 semantics; counts as a query like the loop counts it) ----
        devp = strat_v8(cand, dev[0], dev[1])
        dev_sharpe = float(np.sqrt(252) * devp.mean() / (devp.std() + 1e-12))
        if dev_sharpe > ABSURD_SHARPE:
            verdict, info = "REFUSED_ABSURDITY_ALARM", {"dev_sharpe": round(dev_sharpe, 2)}
        else:
            verdict, info, current = gate.review(cand, current)
        queries += 1
        ok = verdict.startswith("ACCEPTED")
        accepts += int(ok)
        dv = deviation_report(verdict, info)
        # ---- memory updates ----
        bandit.update(arm, 1.0 if ok else 0.0)
        if use_linter:
            registry.record(arm, sign, dv)
        if use_cards:
            cards.update(arm, sign, last_bar, ok)
        if use_islands and portfolio.note(ok):
            migrations += 1
        # ---- adaptation: ALL arms keep the v9 baseline (flip+shrink on reject) — the arms are
        # ADDITIVE increments; C/D only ADD conditioning on top (v3 fix: the first version
        # REPLACED the baseline adaptation and structurally zeroed C/D) ----
        if not ok and arm != "joint_defend":
            base_shrink = 0.6
            if use_dev:
                # v4: REPAIR step_mult FULLY WIRED - the baseline shrink is modulated by the
                # deviation class (dominated 0.6*1.6~0.96 hold, inert 0.6*1.8~1.08 grow,
                # ni/confirm shrink hard). The flip directive stays UNWIRED BY DESIGN in the
                # procedural skeleton (the baseline every-reject flip is retained; direction
                # information flows through the 'prefer' selection channel instead).
                rep_now = REPAIR.get(dv["first_failed_bar"], REPAIR["none"])
                base_shrink *= rep_now["step_mult"]
            step[arm] = float(np.clip(step[arm] * base_shrink, 1e-4,
                                      0.5 * (KNOBS8[arm][1] - KNOBS8[arm][0])))
            direction[arm] = -direction[arm]
        last_bar = dv["first_failed_bar"] if use_dev else "any"
        ledger({"arm_name": arm_name, "seed": seed, "attempt": attempts, "query": queries,
                "knob": arm, "verdict": verdict, "dev": dv, "wealth": round(gate.wealth, 4)})
        if gate.compromised:
            break
    return {"accepts": accepts, "queries": queries, "attempts_N": attempts,
            "wealth_end": round(gate.wealth, 4), "migrations": migrations,
            "lint_refusals": dict(linter.refusals), "distinct_configs": len(linter.seen),
            "audit": dict(gate.audit)}


def perm_p(a, b, n=20000, seed=7):
    """One-sided permutation p for mean(a) > mean(b), paired by seed."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (flips * d).mean(axis=1)
    return float((null >= obs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    seeds = (0,) if args.smoke else SEEDS
    budget = 10 if args.smoke else BUDGET
    print(f"=== HL-MEM A/B v4 — arms A/B/C/D_live/D_frozen/E_islands x {len(seeds)} seeds x budget {budget} ===")
    r, macro = load_extended()
    bel, _ = build_belief(r, macro)
    dev, hold = window(r, bel, *DEV), window(r, bel, *HOLD)     # OOS never loaded

    arms = ("A", "B", "C", "D_live", "D_frozen", "E_islands")
    results = {a: [] for a in arms}
    for seed in seeds:
        for a in arms:
            t1 = time.time()
            res = run_arm(a, seed, dev, hold, budget)
            results[a].append(res)
            print(f"  seed {seed} arm {a:8s}: accepts {res['accepts']:2d}/{res['queries']} "
                  f"N={res['attempts_N']} wealth {res['wealth_end']:.3f} "
                  f"lint {res['lint_refusals']} ({int(time.time() - t1)}s)")

    acc = {a: [x["accepts"] for x in results[a]] for a in arms}
    summary = {a: {"mean_accepts": round(float(np.mean(acc[a])), 2),
                   "sd": round(float(np.std(acc[a])), 2),
                   "mean_N": round(float(np.mean([x["attempts_N"] for x in results[a]])), 1),
                   "mean_wealth": round(float(np.mean([x["wealth_end"] for x in results[a]])), 3)}
               for a in arms}
    reads = {}
    if not args.smoke:
        reads = {"R1_theater_Dlive_gt_B": {"p": perm_p(acc["D_live"], acc["B"]),
                                            "delta": round(float(np.mean(acc["D_live"]) - np.mean(acc["B"])), 2)},
                 "R2_location_C_minus_B": round(float(np.mean(acc["C"]) - np.mean(acc["B"])), 2),
                 "R3_exclusion_B_minus_A": round(float(np.mean(acc["B"]) - np.mean(acc["A"])), 2),
                 "R4_ktd_frozen_minus_live": round(float(np.mean(acc["D_frozen"]) - np.mean(acc["D_live"])), 2),
                 "R5_islands_E_minus_B": {"p": perm_p(acc["E_islands"], acc["B"]),
                                           "delta": round(float(np.mean(acc["E_islands"]) - np.mean(acc["B"])), 2)},
                 "R6_cards_Dlive_minus_E": round(float(np.mean(acc["D_live"]) - np.mean(acc["E_islands"])), 2)}
        r1 = reads["R1_theater_Dlive_gt_B"]
        verdict = (("R1 PASS (p=%.3f, delta=+%.2f): hypothesis-conditioning memory beats the "
                    "exclusion-only null - the full HL-MEM skeleton earns its keep; wire the LLM "
                    "hypothesis-writer on top next." % (r1["p"], r1["delta"]))
                   if (r1["p"] < 0.1 and r1["delta"] > 0) else
                   ("R1 FAIL (p=%.3f, delta=%.2f): at this scale hypothesis-conditioning is "
                    "THEATER per the preregistration - only the linter ships; the LLM layer needs "
                    "a different lever or a bigger budget." % (r1["p"], r1["delta"])))
    else:
        verdict = "SMOKE ONLY - no reads"
    rep = {"preregistration": {"R1": "D_live > B, permutation p<0.1 (else only the linter ships)",
                                "R2_R3_R4": "component isolation reads",
                                "R5_R6": "v4: islands-vs-B isolation + cards-vs-islands split (the referee confound fix)",
                                "v4_amendment": "referee-prescribed BEFORE this run: (a) REPAIR step_mult wired into the "
                                                "reject adaptation (v3 shipped it as dead code), (b) joint_defend deduped "
                                                "in the thompson menu, (c) E_islands arm added (portfolio without cards), "
                                                "(d) 20 seeds, (e) DE-TUNED incumbent (anchor levels) for genuine frontier "
                                                "headroom - the v3 certified-incumbent run measured polish-regime saturation "
                                                "(90% dominated rejections) and is archived as _v3_certified_incumbent",
                                "budget": budget,
                                "seeds": list(seeds), "gate": "UNCHANGED v12-lineage, shared core",
                                "scope": "procedural skeleton only; LLM writer not in this A/B"},
           "summary": summary, "per_run": {a: results[a] for a in arms}, "reads": reads,
           "verdict": verdict, "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("SUMMARY:", json.dumps(summary))
    if reads:
        print("READS:", json.dumps(reads))
    print("VERDICT:", verdict)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
