# Project North Star — Personal Investing (English)

> English translation of [`PROJECT_NORTH_STAR.md`](../PROJECT_NORTH_STAR.md) (the Russian original
> remains authoritative in case of divergence). Part of the [business folder](STARTUP_OVERVIEW.md).

**Status:** user-ratified project contract, accepted 2026-07-10.

**Priority:** this document outranks historical roadmaps, the experiment logbook, reports, and any
earlier goal formulations.

**Changes:** only by direct user instruction. Agents may not silently redefine the goal based on the
outcome of any single experiment.

## 1. The user's original request

> The goal of the project is to build an alternative personal-investment instrument that adapts to:
> (a) the user's risk profile; (b) the investment horizon; and in return shows the achievable return
> with a stated probability. A transparent and controllable model must sit at the core of the
> risk-profile adaptation mechanism.

Accepted role clarifications:

- the RL layer maximizes the investment outcome **inside an explicitly specified admissible set**;
- the LLM agent adapts the research and explanation loop to the Investor Contract, but does not
  eyeball probabilities, does not expand the user's capacity to bear losses, and does not place trades;
- expected future return and its probability are estimated by a separate calibrated
  wealth-distribution engine, not by the PPO critic;
- goal-based DP/MPC is the transparent production champion; PPO is a goal/risk-conditioned
  challenger that must prove added value.

## 2. Product goal

The system accepts a versioned **Investor Contract** and returns a verifiable
**Return-and-Risk Contract**.

### Input: the Investor Contract

- current wealth and currency;
- goal or set of goals: amount / cash flow, nominal or real, priority;
- horizon and interim deadlines;
- contribution and withdrawal schedule;
- `capacity`: the hard financial ability to bear losses — liabilities, reserve, liquidity, wealth floor;
- `tolerance`: the psychological readiness to endure volatility and drawdown;
- `required risk`: the risk needed for the goal given wealth, contributions and horizon;
- required success probability;
- admissible instruments, concentration, liquidity, taxes and other constraints.

`capacity`, `tolerance` and `required risk` must never be collapsed into one score. Capacity is a
hard constraint; tolerance picks a comfortable point inside the admissible set; required risk is
computed. If required risk exceeds capacity, the system proposes extending the horizon / raising
contributions or lowering the goal / probability — it never raises risk automatically.

### Output: the Return-and-Risk Contract

The core object:

```text
r*(p, H, u) = max over certified policies of
              LCB[Q_(1-p)(net real CAGR over H)]
              subject to capacity, shortfall, liquidity,
              concentration, turnover and execution constraints.
```

At `p=80%` the system shows a conservative lower bound on the 20th-percentile CAGR — not the mean.

Every answer must contain:

- horizon, data cutoff date and model version;
- net nominal and/or net real CAGR — unambiguously defined;
- the lower quantile at the requested probability, the median and the mean;
- `P(goal)`;
- `E[shortfall | miss]` and CVaR shortfall;
- the probability of a capacity/drawdown breach and its magnitude when breached;
- fees, taxes, FX, cash-flow and inflation assumptions;
- a model-uncertainty band and the effective independent sample size;
- stress scenarios and evidence status;
- a statement that this is a conditional model estimate, not a guarantee.

Two query forms must be mutually consistent:

1. `(profile, horizon, probability) -> maximum conservative lower-quantile CAGR`;
2. `(profile, horizon, target return) -> calibrated P(CAGR >= target)`.

## 3. Target architecture

```text
Investor Contract
  -> feasibility/conflict resolver
  -> goal-based DP or stochastic MPC
  -> certified risk envelope + DP teacher
  -> one goal/risk-conditioned constrained PPO
  -> deterministic safety shield
  -> execution

PIT multi-asset data
  -> independent scenario/forecast engine
  -> independent wealth-distribution evaluator
  -> calibration monitor
  -> Return-and-Risk Contract

LLM + HCS remain offline:
  parse/explain/propose -> cheap-test -> firewall -> retrain -> future OOS.
```

### Mandatory PPO properties

- one conditional policy over the grid of goals, horizons and risk profiles — not one head per DD budget;
- the state includes belief, funding ratio `W/G`, remaining horizon, peak/drawdown, cash flows,
  current weights, costs and the Investor Contract;
- a sparse/legible risk manager;
- simplex/Dirichlet weights over investable asset sleeves;
- separate expected-value, quantile and cost critics;
- reward and constraints separated;
- a hard safety projection after the actor;
- stock selection stays off until its added value passes the firewall;
- full action probabilities, stochastic and deterministic behavior are logged;
- multi-seed stability, profile monotonicity and horizon consistency are mandatory.

## 4. Subsystem roles

- **Scenario/forecast engine:** estimates the forward wealth distribution from PIT data,
  building-block assumptions, regime/state models, parameter uncertainty and stress scenarios.
- **DP/MPC:** the transparent benchmark and strategic personalizer over wealth × time × goal.
- **PPO:** a sequential allocator inside the certified risk envelope; never the source of client forecasts.
- **Safety shield:** deterministically enforces leverage, wealth floor, liquidity, concentration,
  turnover and prohibited instruments.
- **Wealth evaluator:** independently recomputes the net wealth/CAGR distribution; a shaped-reward
  distribution is never presented as return.
- **CrystalRL:** the transparency-and-controllability guardrail at fixed behavioral complexity —
  not the sole product objective.
- **HCS/Heuristic Learning:** improves the calibrated certified `profile × horizon × return-quantile`
  frontier — not yet another backtest return.
- **LLM:** turns dialogue into a verifiable Investor Contract, explains infeasibility and proposes
  offline candidates; it is not in the capital path.

## 5. Data and universe

- The project remains universe-agnostic; fully locking onto the Dow or CSI500 is prohibited.
- CSI500 is an important China-equity sleeve and robustness domain — not the sole universe.
- "CSI500 2000–2026" must not be built from a late survivor list and called index history.
- Until PIT membership exists, use the official CSI500 total-return index from its available date;
  never mix price and total-return series in client numbers.
- The Chinese investable set must consider CSI500/CSI300 total return, government bonds, the
  money-market/deposit curve, gold, and an accessible global/QDII sleeve.
- The pre-2005 SSE Composite is allowed as a separate market-regime proxy, never as "CSI500".
- Corporate actions, dividends, delistings, suspensions, price limits, FX, taxes and instrument
  accessibility are part of the data contract.

## 6. Accepted facts at ratification

As of the acceptance date:

- E-27 is a genuine SB3 PPO implementation but only a `single-seed soft-tree PPO risk-dial prototype`;
- its deterministic dial arises on a near-uniform action distribution and is not a robust client policy;
- E-28 demonstrated that several knobs technically act, but did not prove causal primitive control
  or a robust improvement;
- P-4/CSI500 is unfinished and is not a result;
- P-2 is a historical-resampling diagnostic, not a calibrated forward forecast;
- 2024–2026 has already been read by earlier experiments and is not fresh frozen OOS for the
  personalization track;
- the current PPO dials are barred from the client menu until the new work order clears them.

The live status and mandatory sequence are in [`WORK_ORDER.md`](WORK_ORDER.md).

## 7. Prohibited goal substitutions

It is forbidden to:

- revert the project north star to alpha, Sharpe, or CrystalScore as the sole criterion;
- call a historical bootstrap a calibrated forecast of the future;
- count simulated paths as independent histories;
- present the mean return as "the return with probability p";
- treat a reward penalty as a drawdown guarantee;
- train a separate model per user;
- let the LLM change capacity, reward or live weights without a confirmed contract and the firewall;
- recommend uncertified, survivor-biased or diagnostic books;
- reuse already-read OOS as fresh confirmation;
- declare an infrastructure PASS to be evidence of investment value.

## 8. Rule for all agents

Before any analysis, experiment, model change, roadmap edit or report in this repository:

1. read this file in full;
2. read [`WORK_ORDER.md`](WORK_ORDER.md);
3. only then open the relevant methodologies and experiment artifacts;
4. name the current work-order stage and its exit gate;
5. do not execute a later stage before its mandatory dependencies close;
6. on document conflict, follow this north star and record the conflict explicitly.
