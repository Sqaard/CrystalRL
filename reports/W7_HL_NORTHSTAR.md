# W7 — personalized-investing HL/HCS north star

**Status:** `W7_INCOMPLETE_RESEARCH_ONLY`, 2026-07-11.

**Authority:** [North Star](../PROJECT_NORTH_STAR.md) and
[Work Order](../PERSONAL_INVEST_WORK_ORDER.md).

**Machine-readable contract:** [W7_HL_NORTHSTAR.json](W7_HL_NORTHSTAR.json).

This replaces bare backtest return as the Heuristic Learning objective, but it does **not** authorize an
autonomous candidate or client probability. The old Dow/Sharpe loop remains outside the personalized
capital path.

## Result

| Gate | Result | Evidence |
|---|---:|---|
| fixed-complexity CV simulatability | PASS | flagship: in-sample 0.918, leave-one-horizon-row-out 0.918, trivial baseline 0.820, normalized 0.547 at ≤8 leaves |
| profile monotonicity + constraint faithfulness | PASS | 0 stricter-profile risk raises; 0/2410 capacity-table cells select a disallowed book |
| horizon consistency ≤1% | **FAIL** | flagship 3.3%; capacity table 0.7% |
| W6 evidence schema-bound | FAIL | operational core is fail-closed/replayable, but economic residual bounds and champion goal-probability parity are not established |
| complete CrystalScore inputs | **FAIL** | DP Faithfulness and true cross-seed Stability are unavailable; scalar is N/A, not fabricated |
| product replay→retrain calibration | **FAIL** | three W5 seeds are replicates of one policy family; minimum for promotion is ten independent candidates |
| north-star schema | PASS | version 2 validates |

Therefore W7 is a useful measurement and gate package, not a completed interpretability/HCS stage.

## Honest Crystal vector at fixed complexity

- behavioral complexity is measured on the unchanged DP action table;
- Simulatability is group-CV accuracy normalized against a train-only majority baseline;
- Completeness↔parsimony is reported for 2/4/8/16 leaves;
- capacity counterfactual supplies an empirical constraint-faithfulness/controllability diagnostic;
- Faithfulness and cross-seed Stability are `N/A`;
- the reduced Crystal scalar is consequently `N/A` under the CrystalRL missing-input rule.

This prevents the old Goodhart failure: simplifying the policy to raise a score cannot count as frontier
growth, because the underlying policy and its behavioral-complexity measurement are frozen.

## Product objective vector

The JSON contains calibration gate fraction, coverage error, proper-score delta, **certified** and research
frontier hypervolumes separately, the `{P(goal), E[shortfall|miss], CVaR}` triplet, W6 violations/cost,
DP regret, policy Stability and CrystalScore. Certified hypervolume remains exactly zero until W8.

Guards are lexicographic: calibration, constraints, stability, costs and DP regret cannot be traded away to
raise a research hypervolume or P(goal).

## Personalized HCS boundary

[personal_hcs_gate.py](../interpretability/personal_hcs_gate.py) now enforces:

- one of eight typed candidate families;
- SHA-bound matched-random, wrong-direction, OOD, dose and ghost result artifacts;
- positive, sign-stable control-adjusted lower bound;
- non-degradation of product guards;
- schema/hash-bound result, full-walk-forward and SkillLens artifacts;
- replay→retrain calibration over distinct independent candidates;
- at least one certified product growth axis at fixed behavioral complexity;
- no LLM write in the capital path.

The positive-control-shaped candidate is currently **rejected**: its synthetic booleans are not control
artifacts, three W5 seeds are one policy family rather than independent candidates, and policy Stability is
unavailable. See
[W7_HCS_GATE_RESULT.json](W7_HCS_GATE_RESULT.json) and [W7_SKILLLENS.md](W7_SKILLLENS.md).

## What closes W7

1. Close the two W6 exit gaps.
2. Measure DP/policy assignments across genuinely separate folds/seeds and compute Stability.
3. Add a valid faithfulness test without inventing latent primitives for a table policy.
4. Resolve or explicitly contract the 3.3% flagship horizon-inconsistency region.
5. Accumulate at least ten independent personalized candidates with cheap predictions and retrain/full-run truth.
6. Run held-out SkillOpt before adopting any change to shared skills or promotion rules.
