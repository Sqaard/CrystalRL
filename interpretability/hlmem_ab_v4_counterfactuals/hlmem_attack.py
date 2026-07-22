"""Adversarial mechanism probe of exp_hlmem_ab v4. NOTHING in the repo is modified.
(1) budget-80 counterfactual with the VERBATIM run_arm (ledger redirected to scratchpad);
(2) instrumented copy of run_arm (B/C) logging step scales at every attempt.
"""
import sys, json, collections
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/Users/ivanp/RL for Time-Series Forecasting/data_RLagent_for_Joseph")
sys.path.insert(0, str(ROOT))
SCRATCH = Path(__file__).parent

import interpretability.exp_hlmem_ab as X
from interpretability.hl_v8_rebalance_lane import strat_v8, KNOBS8, ARMS8
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, window, DEV, HOLD
from src.hl.mechanism_bandit import MechanismBandit
from src.hl.hlmem import deviation_report, REPAIR, NegativeRegistry, ProposalLinter, CardStore, PortfolioScheduler

X.LEDGER = SCRATCH / "cf_ledger.jsonl"          # protect the real ledger
if X.LEDGER.exists(): X.LEDGER.unlink()

r, macro = load_extended()
bel, _ = build_belief(r, macro)
dev, hold = window(r, bel, *DEV), window(r, bel, *HOLD)

MODE = sys.argv[1] if len(sys.argv) > 1 else "cf80"

if MODE == "cf80":
    arms = ("A", "B", "C", "D_live", "D_frozen", "E_islands")
    out = {}
    for a in arms:
        out[a] = []
        for seed in range(20):
            res = X.run_arm(a, seed, dev, hold, budget=80)
            out[a].append(res["accepts"])
        print(a, out[a], "mean", np.mean(out[a]))
    json.dump(out, open(SCRATCH / "cf80.json", "w"))
    # accept curves from the scratch ledger
    recs = [json.loads(l) for l in open(X.LEDGER, encoding="utf-8")]
    gq = [x for x in recs if "verdict" in x]
    for a in arms:
        arr = np.zeros((20, 81))
        for x in gq:
            if x["arm_name"] == a and x["verdict"].startswith("ACCEPTED"):
                arr[x["seed"], x["query"]:] += 1
        print("curve", a, " ".join(f"{arr.mean(axis=0)[q]:.2f}" for q in (10, 20, 30, 40, 50, 60, 70, 80)))
    sys.exit(0)

# ---------------- MODE == "steps": instrumented copy of run_arm (verbatim + LOG lines) -------------
INCUMBENT = X.INCUMBENT; ABSURD_SHARPE = X.ABSURD_SHARPE; ATTEMPT_CAP = X.ATTEMPT_CAP
LOG = []

def run_arm_instr(arm_name, seed, dev, hold, budget=40):
    rng = np.random.default_rng(1000 + seed)
    gate, current = X.fresh_gate(dev, hold)
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
            if abs(nxt - cand[arm]) < 1e-9:
                direction[arm] = -direction[arm]
                nxt = float(np.clip(cand[arm] + step[arm] * jitter * direction[arm], lo, hi))
            cand[arm] = nxt
            sign = int(np.sign(direction[arm]))
        if use_linter:
            lint = linter.check(arm, sign, cand)
            if lint is not None:
                consecutive_lint += 1
                if arm != "joint_defend":
                    step[arm] = float(np.clip(step[arm] * float(rng.uniform(0.5, 2.0)), 1e-4,
                                              0.5 * (KNOBS8[arm][1] - KNOBS8[arm][0])))
                    if lint == "LINT_FALSIFIED_DIRECTION" or rng.random() < 0.5:
                        direction[arm] = -direction[arm]
                        if lint == "LINT_FALSIFIED_DIRECTION":
                            registry.clear(arm, sign)
                if consecutive_lint >= 3:
                    others = [a for a in ARMS8 if a != arm]
                    forced_arm = others[int(rng.integers(len(others)))]
                    consecutive_lint = 0
                continue
            consecutive_lint = 0
            linter.commit(cand)
        devp = strat_v8(cand, dev[0], dev[1])
        dev_sharpe = float(np.sqrt(252) * devp.mean() / (devp.std() + 1e-12))
        if dev_sharpe > ABSURD_SHARPE:
            verdict, info = "REFUSED_ABSURDITY_ALARM", {"dev_sharpe": round(dev_sharpe, 2)}
        else:
            cur_vec_log = gate.vec(current); cand_vec_log = gate.vec(cand)
            verdict, info, current = gate.review(cand, current)
        queries += 1
        ok = verdict.startswith("ACCEPTED")
        accepts += int(ok)
        dv = deviation_report(verdict, info)
        # ---- LOG ----
        LOG.append({"arm_name": arm_name, "seed": seed, "q": queries, "knob": arm,
                    "verdict": verdict, "bar": dv["first_failed_bar"],
                    "step_t2": step["t2"], "step_lvl": step["lvl_defensive"],
                    "step_own": (step[arm] if arm != "joint_defend" else None),
                    "jitter": jitter,
                    "d_ann": (cand_vec_log["ann"] - cur_vec_log["ann"]) if verdict != "REFUSED_ABSURDITY_ALARM" else None,
                    "d_dd": (cand_vec_log["maxDD"] - cur_vec_log["maxDD"]) if verdict != "REFUSED_ABSURDITY_ALARM" else None,
                    "wealth": gate.wealth})
        bandit.update(arm, 1.0 if ok else 0.0)
        if use_linter:
            registry.record(arm, sign, dv)
        if use_cards:
            cards.update(arm, sign, last_bar, ok)
        if use_islands and portfolio.note(ok):
            migrations += 1
        if not ok and arm != "joint_defend":
            base_shrink = 0.6
            if use_dev:
                rep_now = REPAIR.get(dv["first_failed_bar"], REPAIR["none"])
                base_shrink *= rep_now["step_mult"]
            step[arm] = float(np.clip(step[arm] * base_shrink, 1e-4,
                                      0.5 * (KNOBS8[arm][1] - KNOBS8[arm][0])))
            direction[arm] = -direction[arm]
        last_bar = dv["first_failed_bar"] if use_dev else "any"
        if gate.compromised:
            break
    return accepts

led_acc = {}  # determinism check vs the real ledger
for l in open(ROOT / "interpretability/exp_hlmem_ab_ledger.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if "verdict" in x and x["verdict"].startswith("ACCEPTED"):
        led_acc[(x["arm_name"], x["seed"])] = led_acc.get((x["arm_name"], x["seed"]), 0) + 1

mismatch = 0
for a in ("B", "C"):
    for seed in range(20):
        acc = run_arm_instr(a, seed, dev, hold, budget=40)
        if acc != led_acc.get((a, seed), 0):
            mismatch += 1
            print("DETERMINISM MISMATCH", a, seed, acc, led_acc.get((a, seed), 0))
print("determinism mismatches:", mismatch)
json.dump(LOG, open(SCRATCH / "steps_log.json", "w"))
print("logged", len(LOG), "priced attempts")
