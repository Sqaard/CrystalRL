"""Counterfactual decomposition of the C-arm harm in HL-MEM A/B v4.
Variants (all = B + deviation conditioning, differing in HOW REPAIR is wired):
  C_composed    : shipped v4 wiring  (step *= 0.6*step_mult on reject)   [reproduction]
  C_replace     : step *= step_mult  (REPAIR replaces the baseline shrink)
  C_prefer_only : step *= 0.6 (baseline), REPAIR only feeds the 'prefer' selection channel
  C_mult_only   : step *= 0.6*step_mult, but 'prefer' selection channel DISABLED (pure bandit)
  B_repro       : linter only (sanity reproduction)
Also logs the effective step size (step[knob]*jitter, or t2/lvl_def steps for joint_defend)
of every gate query, keyed by verdict, for arm B_repro — tests the 'small steps accept' premise.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/Users/ivanp/RL for Time-Series Forecasting/data_RLagent_for_Joseph")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "interpretability"))
import interpretability.hl_v4_over_crystal1 as V4
from interpretability.hl_v4_over_crystal1 import Crystal1V4Gate
from interpretability.hl_v8_rebalance_lane import strat_v8, n_off_anchor8, KNOBS8, ARMS8
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, window, DEV, HOLD
from src.hl.mechanism_bandit import MechanismBandit
from src.hl.hlmem import deviation_report, REPAIR, NegativeRegistry, ProposalLinter

INCUMBENT = {"t1": 0.30, "t2": 0.70, "lvl_reduced": 1.0, "lvl_defensive": 1.0, "H": 10.0}
HOLD_WIN = 120
ABSURD_SHARPE = 3.0
BUDGET = 40
ATTEMPT_CAP = 200


def fresh_gate(dev, hold):
    V4.n_off_anchor = n_off_anchor8
    g = Crystal1V4Gate(dev, hold, HOLD_WIN, strat_fn=strat_v8)
    cur = dict(INCUMBENT)
    g.frontier = [(dict(cur), g.vec(cur))]
    return g, cur


def run_arm(mode, seed, dev, hold, budget=BUDGET, steplog=None):
    rng = np.random.default_rng(1000 + seed)
    gate, current = fresh_gate(dev, hold)
    bandit = MechanismBandit(arms=list(ARMS8))
    step = {k: 0.1 * (v[1] - v[0]) for k, v in KNOBS8.items()}
    direction = {k: -1.0 for k in KNOBS8}
    use_dev = mode in ("C_composed", "C_replace", "C_prefer_only", "C_mult_only")
    use_prefer = mode in ("C_composed", "C_replace", "C_prefer_only")
    use_mult = mode in ("C_composed", "C_replace", "C_mult_only")
    replace = mode == "C_replace"
    registry = NegativeRegistry(k=2)
    linter = ProposalLinter(registry)
    last_bar = "any"
    queries = attempts = accepts = 0
    consecutive_lint, forced_arm = 0, None
    while queries < budget and attempts < ATTEMPT_CAP:
        attempts += 1
        avail = list(ARMS8)
        if forced_arm is not None:
            arm, forced_arm = forced_arm, None
        elif rng.random() < 0.25:
            arm = avail[int(rng.integers(len(avail)))]
        elif use_prefer and REPAIR.get(last_bar, REPAIR["none"])["prefer"]:
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
            eff = step["lvl_defensive"] * jitter
        else:
            lo, hi, _ = KNOBS8[arm]
            nxt = float(np.clip(cand[arm] + step[arm] * jitter * direction[arm], lo, hi))
            if abs(nxt - cand[arm]) < 1e-9:
                direction[arm] = -direction[arm]
                nxt = float(np.clip(cand[arm] + step[arm] * jitter * direction[arm], lo, hi))
            eff = abs(nxt - cand[arm])
            cand[arm] = nxt
            sign = int(np.sign(direction[arm]))
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
            verdict, info, current = gate.review(cand, current)
        queries += 1
        ok = verdict.startswith("ACCEPTED")
        accepts += int(ok)
        dv = deviation_report(verdict, info)
        bandit.update(arm, 1.0 if ok else 0.0)
        registry.record(arm, sign, dv)
        if steplog is not None:
            steplog.append({"arm": arm, "verdict": verdict, "eff_step": eff,
                            "s_lvl_def": step["lvl_defensive"], "s_t2": step["t2"]})
        if not ok and arm != "joint_defend":
            base_shrink = 0.6
            if use_mult:
                m = REPAIR.get(dv["first_failed_bar"], REPAIR["none"])["step_mult"]
                base_shrink = m if replace else 0.6 * m
            step[arm] = float(np.clip(step[arm] * base_shrink, 1e-4,
                                      0.5 * (KNOBS8[arm][1] - KNOBS8[arm][0])))
            direction[arm] = -direction[arm]
        last_bar = dv["first_failed_bar"] if use_dev else "any"
        if gate.compromised:
            break
    return accepts


def main():
    r, macro = load_extended()
    bel, _ = build_belief(r, macro)
    dev, hold = window(r, bel, *DEV), window(r, bel, *HOLD)
    modes = ("B_repro", "C_composed", "C_replace", "C_prefer_only", "C_mult_only")
    out = {}
    steplog = []
    for mode in modes:
        accs = []
        for seed in range(20):
            sl = steplog if (mode == "B_repro") else None
            accs.append(run_arm(mode, seed, dev, hold, steplog=sl))
        out[mode] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs)), "accs": accs}
        print(mode, out[mode]["mean"], "+-", round(out[mode]["sd"], 2), accs, flush=True)
    # step-size premise: accepted vs rejected effective steps (B_repro)
    import collections
    acc_steps = [x["eff_step"] for x in steplog if x["verdict"].startswith("ACCEPTED")]
    rej_steps = [x["eff_step"] for x in steplog if not x["verdict"].startswith("ACCEPTED")]
    jd_acc = [x["s_lvl_def"] for x in steplog if x["verdict"].startswith("ACCEPTED") and x["arm"] == "joint_defend"]
    print("B accepted eff_step: n=%d median=%.4f q10=%.4f q90=%.4f" % (
        len(acc_steps), np.median(acc_steps), np.quantile(acc_steps, .1), np.quantile(acc_steps, .9)))
    print("B rejected eff_step: n=%d median=%.4f q10=%.4f q90=%.4f" % (
        len(rej_steps), np.median(rej_steps), np.quantile(rej_steps, .1), np.quantile(rej_steps, .9)))
    print("B joint_defend accepts, step[lvl_defensive] at accept: median=%.4f" % np.median(jd_acc))
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
