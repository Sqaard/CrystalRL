# Economic World Picture V2 — humanity's knowledge, bound to the bot

*Assembled 2026-07-19 from the full experience/ corpus (174 annual DeepSearch reports via the
per-report digest, both 52-year syntheses, the Technical Picture, WORLD OF BEST TRADING BOT) by a
9-reader extraction into 101 typed, evidence-tagged atoms — the machine layer lives in
[world_picture_v2_atoms.json](world_picture_v2_atoms.json), and the bindable subset feeds
`data/_crystal_world/semantic_store` directly. Replaces ECONOMIC_WORLD_PICTURE_FOR_AGENT.md as
the bot-facing picture. Status: external experience and PRIORS — never evidence of alpha
(the corpus's own rule: `corpus_priors_never_evidence`).*

## 1. The compressed worldview (one paragraph)

The economy is a human-made adaptive coordination system of promises, institutions, balance
sheets, behavior, capacity and politics. The recurring super-cycle: stability → lower perceived
risk → leverage/crowding → weaker underwriting → shock → liquidity-funding-narrative break →
forced adjustment → backstop → risk migrates somewhere new. For a daily-bar bot the translation
is fixed: **regime before rule, timing before signal, costs before alpha, liquidity before
confidence, failure conditions before promotion** — and the only near-unanimous daily-scale
regularity in 52 years of material is volatility clustering, which is why certified value lives
in the risk channel.

## 2. The state ontology — what the bot must SEE

| State variable | Binding (exact, daily-computable) | Horizon | Evidence | Panel status |
|---|---|---|---|---|
| Volatility state | per-ticker/book EWMA of squared returns (5d/21d half-lives) | days-weeks | Established | **IN** (the load-bearing variable; PK-1 validated EWMA) |
| Regime belief | 2-state causal HMM over macro observables | weeks | Established (in-house certified) | **IN** (the command surface) |
| Credit conditions | EBP level + 3m change (Fed CSV, lagged); credit-appetite spread = 21d adj-return HYG−LQD, duration-stripped via LQD−IEF | months-quarters | Established (the corpus's master amplifier) | EBP **IN** (PK-2 PASS); ETF spread = **CANDIDATE** |
| Vol term structure | VIX3M/VIX ratio (backwardation = stress marker) | days-weeks | Strong-but-Contextual | **IN** (the certified-queue 5th eye) |
| Stock-bond correlation regime | rolling 63d corr(book, TLT/IEF); >+0.2 sustained = inflation regime, bonds risk-additive, cash is the only hedge | months | Established (P-1 lived it in 2022) | **CANDIDATE** |
| Plumbing/liquidity stress | mean pairwise 5d correlation of names > ~0.6 AND (dVIX>+20% OR backwardation) → crash prices = flows, not news | days-weeks | Established | **CANDIDATE** (computable from panel) |
| Complacency clock | consecutive days VIX < trailing 504d median + spread tightness → leverage CAP only, never a timer | months-years build, days unwind | Strong-but-Contextual | **CANDIDATE** |
| Breadth/participation | frac above 50d MA, new-high/low fractions, dispersion | weeks | Strong-but-Contextual | **TESTED, NO-ADD** (WP-1: incrementally complete at 5-20d — the state already spans it) |
| Rate impulse | fast Δy10 (impulse, not level — levels killed L2 via OOD lock-in) | months | Strong-but-Contextual | Δ-form **IN** |

**The sufficiency doctrine** (the honest answer to "is the state complete?"): completeness is
unattainable in principle (Grossman-Stiglitz; `risk_migrates_beyond_daily_observables`); the
operational standard is **incremental completeness** — every candidate the picture names gets the
WP-1 harness (capacity-fair noise twins, identified protocol), survivors enter the state. Breadth
was the first test: NO-ADD. Queue: credit-appetite spread, stock-bond flag, plumbing flag,
complacency clock.

## 3. Regime taxonomy (what the picture says regimes ARE)

- **Two-phase financial cycle**: credit-expansion calm (long, grinding up, vol-selling wins) vs
  deleveraging stress (short, correlated, plumbing-driven). Our 2-state belief is the daily
  shadow of this cycle.
- **Inflation vs demand regimes** (`stock_bond_correlation_regime`): determines whether bonds
  hedge at all — regime-dependent diversification, the register's 2022 lesson from the corpus side.
- **Rate-impulse quadrant**: slow growth-driven rises absorbed; fast impulses surface hidden
  balance-sheet fragility (S&L, 1994, 2022-23).
- **Backstop states**: policy reflexes truncate left tails but move risk elsewhere
  (`policy_backstop_reflex`, `risk_migration_decay`) — after every backstop, re-ask "where did
  the leverage go?"

## 4. The mechanism library (top causal chains, full set in the atoms)

1. `credit_cycle_master_amplifier`: credit growth+quality → cycle amplitude — the single
   most-replicated claim across ~50 annual reports.
2. `crash_prices_are_flows_not_news`: margin spirals → forced selling → crash prices carry flow
   info; partial mean-reversion after deleveraging completes. Decision hook: while the plumbing
   flag is up, down-weight price-derived belief updates written to memory.
3. `extrapolation_momentum_overshoot`: narrative extrapolation → crowding → overshoot-reversal.
4. `intermediary_deleveraging_spiral` and `stress_exit_correlation_spike`: why diversification
   fails exactly when needed.
5. `regime_break_invalidates_learned_coefficients` (Lucas thread): learned relationships expire
   at regime boundaries → memory entries carry regime fingerprints; cross-boundary retrieval is
   down-weighted (wired into the semantic-store governance).

## 5. Negative knowledge and guardrails (the corpus converges on our own rules)

The strongest meta-finding: humanity's knowledge REDISCOVERS the project's hard-won discipline —
`search_freedom_manufactures_false_alpha` (= our loop governance), `one_oos_band_is_not_proof`
(= prereg culture), `optimizer_amplifies_estimation_error` (= RMT-1/Jagannathan-Ma),
`cost_capacity_audit_kills_gross_alpha` (= priced-out verdicts), `benchmark_score_is_not_live_
competence` (= the sim-to-market gap), `volume_is_not_liquidity`, `valuation_has_no_daily_timing
_power` (= W3 CAPE falsification), `calm_is_fragility_but_not_a_timer` (asymmetric use only),
`independent_risk_veto_and_safe_flat` (= fail-closed governance), `no_monocausal_gating`,
`point_in_time_clock_integrity` (= E-10 mixed-clock, PIT discipline). Full list: 36 guardrail +
negative-knowledge atoms.

## 6. Decay clocks

Every stored regularity carries: publication/crowding state, cost breakeven, and expiry triggers
(`alpha_half_life_vs_cost_clock`, `rule_coefficient_drift_monitor`, `no_timeless_coefficients`).
In-house instances already logged: csi500 reversal at half magnitude; McLean-Pontiff −26%/−58%.

## 7. Era memory (compressed; full per-era text in the corpus)

1975-92: inflation/credibility/plumbing — nominal numbers mislead across monetary regimes.
1993-2007: calm→leverage→fragility twice (LTCM, subprime); risk migrated to wherever regulation
wasn't. 2008-19: backstop era — policy reflex truncates tails, breeds moral hazard and duration
risk. 2020-26: pandemic + inflation return + 2022 correlated crash (the hedge that wasn't) +
backstop-credibility politics + AI narrative concentration. Lesson constant: **survivability
over prediction** — the systems that persist are those that cannot be killed by one regime.

## 8. The binding mechanism (knowledge → model), now standard

```
corpus atom (typed, evidence-tagged, bot_binding)          [world_picture_v2_atoms.json]
   → PRIOR in the semantic store (never evidence)          [data/_crystal_world/semantic_store]
   → candidate state block (exact computable construction)
   → WP-1 completeness harness: capacity-fair noise twins, identified protocol, prereg bar
   → survivor enters the state → W-series battery re-run (KT-A..KT-E)
   → any DEPLOYABLE claim → the v12 frozen gate + exposure-matched twins + paper track
```
Worked example (this session): breadth — named by the picture, built from the per-ticker panel,
tested, NO-ADD, recorded. The mechanism is the deliverable: priors flow in one end, certified
state variables come out the other, and nothing skips the middle.

## 9. Training and testing on the real universes — the preregistration

**Full-run protocol** (Dow-30 v2 panel primary; csi500 secondary with price/volume-native
features): (1) run the four CANDIDATE state blocks (credit-appetite spread, stock-bond flag,
plumbing flag, complacency clock) through WP-1; (2) freeze the surviving state; (3) re-run
KT-A/KT-B/KT-E on the frozen state (identified protocols, all guards); (4) KT-C against the
certified rule (exposure-matched twins); (5) horizon-gated (k=1 only) consolidation per KT-D v2;
(6) any deployable configuration goes through the v12 gate and joins the live paper track beside
the certified rule. Windows discipline unchanged (TRAIN 2010-18 / DEV 2019-21 / HOLD 2022-23 /
OOS 2024-26 read-once-per-claim; quarantined preregs untouched).

**The honest sentence about guarantees** (replacing "результаты, гарантирующие"): no historical
result can guarantee live performance — performativity, alpha decay and regime breaks are
Established atoms of this very picture (`one_oos_band_is_not_proof`,
`benchmark_score_is_not_live_competence`). What CAN be delivered and already partially exists:
**certified claims** (frozen-gate, placebo-controlled, twin-tested), **calibrated contracts**
(the personalization machinery), a **live paper track** accumulating toward pre-registered reads,
and **fail-closed governance**. That is the maximum honesty allows — and it is precisely the
product ("trust made auditable").
