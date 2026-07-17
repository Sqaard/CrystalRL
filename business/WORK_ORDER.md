# Personal Investing — the Mandatory Work Order (English)

> English translation of [`PERSONAL_INVEST_WORK_ORDER.md`](../PERSONAL_INVEST_WORK_ORDER.md)
> (the Russian original remains authoritative). North star: [`NORTH_STAR.md`](NORTH_STAR.md).
> Part of the [business folder](STARTUP_OVERVIEW.md).

**Status:** user-accepted execution contract, 2026-07-10.

**Methodological basis:** `reports/PERSONAL_INVEST_METHODOLOGY.md` — the (incomplete) P-1…P-8
methodology. It remains part of this order but does not cancel the newer data, calibration, DP/PPO
and safety gates below.

## 1. Execution rules

1. The order encodes dependencies, not wishes: a stage is not `DONE` until its exit gate passes.
2. A later prototype may be built in parallel, but may not be used in product claims or menus until
   every preceding mandatory gate closes.
3. Every experiment pre-registers its claim, mechanism, cheapest kill test, null, split, costs,
   evidence rung and expected artifact.
4. Every result leaves a result artifact and a learning/failure-memory artifact.
5. Re-reading used OOS is never fresh confirmation. Final validation needs a future pre-registered
   period or an honest outer rolling-origin fold.
6. All probabilities are relative to a scenario-model version and carry a calibration/model-risk band.
7. The client menu may contain only policies/books with status `CERTIFIED_FOR_PROFILE_GRID`.

## 2. The canonical order (status as of 2026-07-11)

| Stage | Status | Depends on | Deliverable |
|---|---|---|---|
| W0. Governance reset | **DONE (2026-07-10)** | — | north star, work order, claims freeze |
| W1. Investor Contract + P-4 interface | **DONE — INTERFACE ONLY** | W0 | agreed schema/solver contract, no product claim |
| W2. PIT multi-asset data gate | **PASS V1 (2026-07-11)** | W0 | official-index-only China design, issuer-NAV sleeves, net cash path, access contract; stock panel quarantined |
| W3. Forecast/scenario engine | **v1.3: 7/10 gates; CAPE falsified; probabilities remain `UNCALIBRATED_RESEARCH_ESTIMATES`** | W1, W2 | forward wealth distribution + prequential calibration |
| W4. Goal-based DP/MPC champion | **VALIDATED + W4.1: magnitude-aware objective + contributions; P(goal)↔miss-depth coupling is STRUCTURAL → λ belongs to the Investor Contract; contributions +22pp** | W1, W3 | transparent dynamic policy + DP teacher |
| W5. Conditional constrained PPO | **v1 DELIVERED: one conditional policy + DP teacher + shield; near-uniform disease cured (max-prob 0.71 vs 0.20); budgets learned; regret 6.6pp → DP remains champion** | W2–W4 | one goal/risk-conditioned policy |
| W6. Safety shield + execution | **OPERATIONAL CORE 7/9; EXIT INCOMPLETE** | W1, W5 | fail-closed guards, costs/fills, paper replay; economic residual bounds open |
| W7. Interpretability + HL/HCS redesign | **`W7_INCOMPLETE_RESEARCH_ONLY`** | W4–W6 | product north-star registry live; Crystal inputs and replay calibration open |
| W8. Nested validation + certification | **`REJECTED_HONEST`** | W2–W7 | development battery complete; no profile grid certified |
| W9. Product contract + deployment | **FAIL-CLOSED / BLOCKED BY W8** | W8 | validated refusal, no client estimates |
| W10. Preference learning / multi-goal | **DEFERRED** | W9 evidence | bounded P-8-style adaptation |

## 3. Stages, deliverables and exit gates

### W0 — Governance reset

Closing artifacts: the claims↔evidence matrix, the unified OOS ledger, `ppo_menu_metrics.json`
(`RESEARCH_ONLY`, `NOT_CERTIFIED_FOR_PROFILE_GRID`, `recommendable=false`).

Do: embed the north star and this order into all skills/onboarding/roadmaps; freeze E-27 as a
`single-seed soft-tree PPO risk-dial prototype`; freeze E-28 as a lever demonstration (not causal
control); mark P-4/E-32 unfinished; remove PPO dials from any recommendable menu; open the
claims/evidence matrix (claim → split → controls → artifact → status).

Exit gate: no canonical document or runtime menu calls a PPO dial a client policy; all entry
documents reference the north star and this order; every used OOS period is listed in one ledger.

**Verdict 2026-07-10: `PASS`.** Canonical entry documents carry the authority override; E-27/E-28/
P-4 statuses frozen; no runtime code imports the historical PPO-dial registry; every registry row is
non-recommendable.

### W1 — Investor Contract + P-4 as an interface prototype only

Implementation: `investor_contract.py`, `personal_invest_scenarios.py`, `personal_invest_solver.py`.

Do: the schema (wealth, goals, currency, nominal/real, horizon, contributions/withdrawals, capacity,
tolerance, required risk, liquidity, instrument restrictions, probability target); capacity = hard
constraint, tolerance = preference, required risk = computed; merge risk-first and goal-first into
one joint solver; fix maxDD to include initial wealth=1; never replace the cash path with a scalar
mean; implement both consistent query forms; every probability ships with a shortfall/breach
magnitude pair; distinguish the annual drawdown cap from full-horizon maxDD.

Exit/kill gate: versioned reproducible schema round-trip; the goal genuinely constrains the
optimization; an infeasible contract yields only the allowed horizon/contribution/goal/probability
changes; property tests rule out budget/goal conflation; **until W8 it is forbidden** to call the
output a forecast or recommendation.

**Verdict 2026-07-10: `PASS AS INTERFACE PROTOTYPE`.** Round-trip, joint solver, goal/capacity
separation, initial-wealth DD, aligned cash path, inverse queries and probability↔magnitude pairs
pass the test/red-team suite. Dated cash-flow feasibility intentionally refuses until W3 supplies
pathwise wealth accounting. Runtime output stays non-recommendable/quarantined.

### W2 — PIT multi-asset data gate

Artifacts: `personal_invest_data_gate.py` + the data-quality report.

Do: obtain the official CSI500 and CSI300 total-return benchmarks; either reconstruct PIT membership
(delistings, dividends, corporate actions, suspensions, price limits) or do not use the late
403-name survivor list for historical claims; audit every adjusted-return jump; add China government
bonds, the money-market/deposit curve, gold, and an investable global/QDII sleeve; synchronize
calendars/as-of clocks, FX, fees, taxes, accessibility; store vintage and provenance per series;
pre-2005 SSE only as a separate regime proxy.

Exit/kill gate: zero unexplained extreme adjusted returns; PIT/official provenance passes an
independent audit; price vs total-return never mixed; a missing benchmark/sleeve blocks the
experiment rather than silently disappearing; survivor-biased diagnostic books are never
recommendable.

**Verdict 2026-07-11: `PASS V1`** for the registered index-only/multi-asset design. The four
original blockers closed under a machine re-audit: consumed-series jumps explained or excluded with
the quarantined stock panel; bond/gold/QDII verified against issuer NAV; a dated net cash path
built; a versioned fees/tax/accessibility contract adopted. Residual approximations explicitly
ledgered. This unlocks research W3 — not client probabilities, and not the survivor 403-name panel.

### W3 — Independent forecast/scenario engine

Do: the ensemble (PIT stationary/block bootstrap, regime/state-conditional member, building-block
expected returns, parameter/posterior uncertainty, explicit stresses); model inflation, deposit
rates, dividends, costs, taxes and FX jointly with asset returns; emit net nominal and net real
wealth distributions; lock every forecast before realization; run outer rolling-origin/prequential
evaluation; score Brier, CRPS/log score, pinball, PIT, empirical coverage 50/80/90%; adaptive
conformal only as a coverage wrapper; report the effective independent sample size, not path count.

Exit/kill gate: the ensemble beats the unconditional historical bootstrap on pre-chosen proper
scores; calibration error inside the pre-specified band separately per horizon/risk band; no
systematic undercoverage in stress regimes; the forward CMA backcast beats the sample-mean anchor —
otherwise probabilities stay `UNCALIBRATED_RESEARCH_ESTIMATES`.

**Verdict v1 → v1.2 (2026-07-11): `ENGINE + HARNESS DELIVERED; GATES 7/10`.** Built
`personal_invest_forecast.py` (3-member ensemble + parameter uncertainty + joint inflation + fees +
stresses + a locked forecast ledger) and `exp_w3_calibration.py` (370 rolling origins;
pinball/coverage/PIT/Brier + a TEETH test that rejects a deliberately overconfident null). Later
iterations added the valuation-reversion member, prequential member weights, and a downside-primary
proper score declared before the run. Headline: the CN CMA beats the sample mean 3.6× (the P-5
thesis proven); the honest residuals (US stress sharpness, CN overcoverage) are characterized, and a
believed upgrade — the real Shiller CAPE — was falsified and reverted. All probabilities remain
`UNCALIBRATED_RESEARCH_ESTIMATES`, enforced in code.

### W4 — Goal-based DP / stochastic MPC champion

Do: state = funding ratio `W/G`, remaining horizon, peak/DD, cash flows, liabilities, readable
belief; actions = certified multi-asset frontier points; objective = terminal/multi-goal utility
with probability and magnitude shortfall; capacity/liquidity/product restrictions as hard
constraints; export the policy table, the full wealth distribution and teacher trajectories;
compare against the best static frontier point, not a weak glide path.

Exit/kill gate: Bellman/solver checks and small-case known optima pass; DP is not worse than the
best static policy on the declared objective after costs; the dynamic lift is stable across outer
folds and not bought with tail risk; if state/action stay `one equity + cash` and PPO cannot add
complexity, DP remains the production champion.

**Verdict 2026-07-11: `MACHINERY VALIDATED; CHAMPION ON FEASIBLE UNCONSTRAINED GOALS`.** GDP1 PASS.
Flagship `das_ostrov_double_10y`: all gates PASS — DP P(goal) 0.472 vs best static 0.355, lift
+11.68pp (CI90 [+10.5,+13.0], folds stable, 97.5% monotone). The capacity case (DD≤15% + 2×@10y) is
an honest INFEASIBILITY (the product answer is a feasibility refusal, not auto-raised risk). The CN
case's +24.5pp lift was CAUGHT by GDP3 (miss magnitude worse than static — digital-option
aggression) → champion status withheld for CN until the magnitude-aware objective. **W4.1:** the
magnitude-aware objective U=1{w≥1}−λ·max(0,1−w) + contributions are live; the flagship reproduces
(+10.45pp); the P(goal)↔miss-depth coupling is STRUCTURAL → λ is an Investor Contract parameter;
contributions 3%/yr move P(goal) 0.60→0.82 (the dominant lever, confirmed numerically). Teacher
trajectories (15k rows) exported for W5.

### W5 — One goal/risk-conditioned constrained PPO

Do: replace per-DD heads with one conditional policy; inputs = belief, `W/G`, time-to-goal, DD/peak,
cash flows, current weights/costs, the Investor Contract; actor = sparse interpretable risk manager
+ simplex/Dirichlet sleeve allocator; critic stack = expected-value, quantile and separate cost
critics; DP-teacher BC/pretrain then constrained PPO fine-tune; separate reward from constraints
(Lagrangian/CPO); log full action distributions and diagnostics; multi-seed; stock selection only
after its own PIT firewall pass.

Exit/kill gate: PPO regret to DP within the pre-specified tolerance; added value survives a
matched-capacity comparison; budgets hold across outer folds/seeds (not only in shaped reward); the
deterministic output is not an argmax tie-break of a near-uniform policy; stricter capacity never
raises exposure/tail risk; unseen profile/horizon holdouts do not degrade beyond the band —
otherwise PPO stays challenger/research-only.

**Verdict v1 (2026-07-11): `MACHINERY DELIVERED; DP REMAINS CHAMPION`.** One conditional policy
(obs=[funding ratio, years-left, capacity flag], SoftTree depth-2) replaces the E-27 heads; BC from
the DP tables (82–85%) → PPO fine-tune × 3 seeds; shield-in-env. GW5-1 PASS (max-prob 0.706–0.718 —
the E-27c disease cured by the teacher, third confirmation of the teacher lever). GW5-3 PASS (0.0%
attempted violations — the budget is LEARNED). GW5-2 FAIL: regret 6.6pp — on the same discrete space
exact backward induction is unbeatable → GW5-4: DP is the production champion (a foreseen outcome).
PPO returns when the action space outgrows enumeration (continuous multi-sleeve).

### W6 — Safety shield and execution

Do: deterministic projection (simplex, leverage, wealth floor, liquidity, concentration, turnover,
restricted instruments); target→executed weight separation; anti-churn/deadzone and cost-aware
pacing; gap-risk, suspension, limit-up/down and stale-price behavior; safe fallback to a DP/static
certified policy on OOD, missing data or a calibration alarm; a separate paper book and audit log
per Investor Contract/model version.

Exit/kill gate: zero hard-rule violations in adversarial replay; ≥99% floor compliance alone is not
enough — every residual breach explained and bounded; the shield neither hides a weak PPO policy nor
destroys goal probability relative to the champion; target/executed attribution fully reproducible.

**Status:** OPERATIONAL CORE 7/9; exit incomplete (economic residual bounds + the paired
champion-degradation test remain open).

### W7 — Interpretability and Heuristic Learning/HCS under the product goal

Do: measure CrystalScore at fixed behavioral complexity; add goal-conditional simulatability,
profile monotonicity, constraint faithfulness and envelope coverage; the HL north star = calibration
error, certified frontier hypervolume, goal/shortfall metrics, constraint violations, DP regret,
costs, stability; type the candidates (scenario/calibration, goal state, reward, constraint, DP
teacher, architecture/action, primitive intervention, safety/execution); primitive interventions
only with matched-random, wrong-direction, OOD, dose and ghost controls; the LLM stays out of the
capital path; every skills/gates change goes through SkillLens → held-out SkillOpt.

Exit/kill gate: a frontier improvement is not policy-simplification-for-CrystalScore; the
control-adjusted effect is positive and stable; HCS replay is calibrated to retrain truth; no lever
demonstration is called causal without the full control suite.

**Status:** `W7_INCOMPLETE_RESEARCH_ONLY`.

### W8 — Nested validation and certification over the profile grid

Do: outer rolling-origin folds with the scenario model, DP and PPO fit separately inside each fold;
purging/embargo for overlapping horizons; multi-seed policy and forecast ensembles; holdouts by
profile, horizon, regime and universe; explicit model/data-vintage uncertainty; pre-register future
OOS (2024–2026 is already burned); evaluate the joint event (return quantile + capacity +
liquidity/goal constraints); compare against DP, static multi-asset, cash/deposit and simple
risk-overlay baselines.

Certification gate — a policy becomes `CERTIFIED_FOR_PROFILE_GRID` only if, simultaneously:
quantile/event forecasts are calibrated in-band; goal/shortfall metrics are stable; the constraint
violation rate is under the limit; stress/OOD fallback works; costs/taxes/FX included; profile
monotonicity and horizon consistency pass; PPO delivers declared value over DP or DP remains the
chosen champion; the full evidence package reproduces from committed artifacts.

**Verdict:** `REJECTED_HONEST` — the development battery is complete; no profile grid is certified.

### W9 — Product contract, P-6 messaging and monitored deployment

Do: serve `r*(p,H,u)` and the inverse `P(CAGR >= target)`; show mean/median/quantile, goal
probability, expected/CVaR shortfall, breach probability/magnitude and the uncertainty band; state
nominal/real, fees/taxes, horizon, data cutoff, model version; frequency framing passes a
comprehension test; an infeasible answer proposes only admissible contract changes; paper deployment
precedes any live capital; the calibration-drift monitor can demote status and trigger fallback.

Exit gate: no probability without a magnitude and uncertainty pair; no "guarantee" from a model
estimate; the live path uses only certified policies and a user-confirmed Investor Contract;
monitoring updates calibration but never rewrites past forecasts.

**Status:** FAIL-CLOSED / blocked by W8 (by design).

### W10 — Preference learning and multi-goal expansion

Start only after W9 outcome evidence accumulates. Behavioral data may refine tolerance but never
capacity; any Investor Contract change requires user confirmation; online learning never touches
live weights without the usual W7–W8 gates. **Status: DEFERRED.**

## 4. How the legacy P-experiments map into this order

| Legacy line | New role |
|---|---|
| P-1 multi-asset frontier | diagnostic precursor of W2/W3; repeat only on PIT/official TR data |
| P-2 long-history simulation | one scenario member inside W3, not a standalone forward forecast |
| P-3 GBWM DP | the W4 production champion |
| P-4 goal-first/shortfall/profile | W1 research-interface contract done; product gate closed until W2–W8 |
| P-5 CMA anchor | the core of W3 |
| P-6 messaging/calibration | W3 calibration + W9 product language |
| P-7 structural DD | a W6 safety-shield candidate |
| P-8 preference learning | W10, deferred |

## 5. Progress metrics and the HL learning curve

The primary learning curve is plotted against cumulative paid HL proposals — never against re-read
OOS. Axes: certified frontier hypervolume over `profile × horizon × lower-quantile return`;
calibration error / proper score; goal success and expected/CVaR shortfall; capacity breach rate;
regret to the DP/MPC champion; costs/turnover; the CrystalScore vector at fixed behavioral
complexity; replay-to-retrain prediction error. Frozen/future OOS enters the chart as one
pre-registered point, never as a dev curve.

## 6. The next allowed action

W0–W5 are closed within their declared scope; W6–W7 are measured but have not passed their exit
gates; W8 ended in an honest reject. The only allowed next actions are closing the binding evidence
gaps: W3 calibration 10/10; W6 economic residual bounds + the paired champion-degradation test;
horizon consistency; full Faithfulness/Stability inputs for CrystalScore; at least 10 independent
replay→retrain HCS pairs; and the future pre-registered outcomes of the W8 protocol
(`reports/preregistration_w8_2027.md`).

Until then W9 must return a fail-closed refusal with no personal return/probability, and W10 stays
deferred. The locked one-year forecasts are read no earlier than **2027-07-12**; a genuinely
untouched 3-year policy fold cannot mature before **July 2029**.
