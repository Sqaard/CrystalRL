# HL v4 over R6c — the Pareto tension gate on the real HCS harness

**Task:** point the red-team-certified v4 gate (`CRYSTAL1_HL_V4_REDESIGN.md`) at the REAL substrate —
R6c frozen-PPO policy-forward replay (fold 2021, 24 candidates × 251 aligned days) — replacing the
polygon's `description_len` legibility axis with **real CrystalScore-v2 axes**.

## The mapping (polygon → real substrate)

| v4 mechanism | Polygon | R6c / HCS real |
|---|---|---|
| Capability axis | env return | **dev-window Sharpe** (days 0–149) from `validation_daily.csv` |
| Legibility axis | description_len / MDL bins | **CrystalScore-v2 Axis A**: MDL simulatability of the controller decision (`cash_trade_direction` ∈ {−1,0,1}) from 11 named state columns, 8-leaf vs 64-leaf trees |
| Authority (tension-harm) unit | Δ#clauses | **config description length** = # knob overrides vs `pf_original` (registry diff) |
| Rotating holdout (F3) | seed slices 1000–1600 | **date windows**: 60 contiguous days over the last 101, stride 7 per query |
| Stressor adversary (F2) | different env physics | **regime slices**: top-quartile `risk_stress` days + worst-drawdown days, dev-only, defined on the INCUMBENT's log (candidate-independent → no selection leak) |
| z-test | i.i.d. seeds | **paired daily deltas + moving-block bootstrap** (block=5) — autocorrelation-honest |
| Mechanism bandit (F8/F9) | operators | **mechanism families** from the registry (global_topk / group_aware / risk_aware / confidence_coeffs / trigger_thresholds / no_incremental) |
| Canaries (F12) | harmful clauses | **config-bloat twin** (incumbent's own log +3 overrides) + the **measured-worst real candidate** (dev-precondition, sandboxed, escape ⇒ freeze) |
| Extra guards (kept from v1/HCS) | — | **R7 source-overprediction + best-of-N noise floor, REFUSAL-ONLY** (full-fold stats may stop a move, never help one) |

Code: [r6c_tension_adapter.py](../src/hl/r6c_tension_adapter.py), [hl_v4_over_r6c.py](../interpretability/hl_v4_over_r6c.py);
result JSON: `interpretability/hl_v4_over_r6c_report.json`.

## The run

- **23 real controller mutations walked; 0 certified — even at replay tier.**
- **8 died at the rotating holdout** (z = 1.63, 1.56, 0.11, 0.39, 0.98, −0.91, 0.81, 0.96 against an
  alpha-investing bar escalating 2.15 → 3.84); **15 refused as unaffordable** once the query storm had
  drained the alpha-wealth (0.20 → 0.0006). Canaries **3/3 caught**, gate never compromised.
- **The +30% replay lift decays out-of-window:** the top full-fold candidate
  (`pf_topk_b6_s8 vixShock/deriskMarket`) has dev-window edge **+456 bp/yr** but holdout-window
  (late-2021) edge **+173 bp/yr**, block-z **1.64** — real, positive, and *not certifiable at an honest
  autocorrelation-aware bar on 101 local days*.
- **Axis A on the real controller:** `pf_original` mdl_deficit = **0.286** — a K≤9 (8-leaf) story
  reproduces only ~71% of the 64-leaf-recoverable structure of R6c's daily cash decision. The polygon's
  CRYSTAL-1 sits near 0; this is the first Axis-A measurement of the production-line controller.
  *(Verification caveat: on 150 days this number is tree-seed-unstable — range [0, 0.33], the 64-leaf
  ceiling is a memorizing 16-leaf tree. A coarse indicator, and not load-bearing this run: every gate
  verdict fell on the Sharpe axis.)*

## What v4 adds over v1's flat refusal

v1 (`hl_over_r6c.py`) answered with one bit: `REFUSED_REPLAY_UNCALIBRATED` × 23. v4 returns a
**diagnosis with prices**:

1. **Where candidates die**: at out-of-window replication (holdout), not at noise floors or stress
   regressions — the mechanism class is dev-window-flattered, exactly the R7 over-crediting signature,
   now measured *within* one fold.
2. **The decay ratio** (456→173 bp/yr ≈ 2.6×) is a concrete calibration number for the HCS
   replay-optimism prior.
3. **The priced recommendation for cloud budget**: the only families whose z even approached the bar are
   `trigger_thresholds` and `confidence_coeffs` (z≈1.6); everything else is ≤1.0. If multi-fold cloud PPO
   budget is spent, spend it there — and the expected residual edge to confirm is ~170 bp/yr, not the
   +30% Sharpe the raw replay advertises.
4. **A governed walk, not a scan**: sequential testing on the same 101-day pool without the escalating
   alpha-bar would be a mining storm; the gate priced it to exhaustion instead.

## Honesty caps (all carried in the report JSON)

- **Replay-tier ceiling**: an accept here would still NOT be a promotion — single fold locally; cloud
  multi-fold PPO policy-forward truth remains the promotion gate (the v1 refusal, preserved).
- **Upstream selection**: the 23 candidates were themselves selected by the prior HCS run on *full-fold*
  scores, so the last-101-days holdout is not virgin; the escalating bar partially compensates, but this
  is one more reason the ceiling stands.
- **Power**: with ~60-day windows and this delta volatility, small true edges sit below the local
  detectability floor; "not certifiable" ≠ "zero".
- **Axis A small-sample**: MDL measured on 150 dev days (60/40 chronological split); treat coarsely.

## Adversarial verification (4 independent skeptics)

**LEAKAGE → CONFIRMED.** Instrumented full replay reproduces the shipped JSON bit-for-bit. All 8 holdout
windows inside days 150–250, zero dev overlap; stress slices are dev-only, incumbent-defined; within-family
ordering provably follows dev-Sharpe (diverges from full-fold selection_score order in 4/6 families); R7
guards structurally refusal-only; the canary alias mutates nothing (deep-compare of all 24 logs pre/post);
all 53 `mdl_deficit` calls saw dev indices only. Killer probes: a **dev-only mirage** (+10bp/day on dev days
only) explodes past the Pareto stage then dies at REJECTED_HOLDOUT with z=0.00 exactly; a **holdout-only
boost** dies at the dev-side admission — the two windows provably cannot reach each other's decisions.

**STATS → CONFIRMED.** All 8 trail z-values and all 8 z_crit values (2.146→3.837) reproduce exactly from
raw CSVs. The mining-storm simulation (AR(1) nulls matched to empirical autocorrelation, 400 runs): without
the alpha-investing schedule, 23 sequential tests on the 101-day pool give **FWER 0.84** (mean 1.65 false
accepts); with it, **0.06–0.08**. Bootstrap-seed shopping impossible (seed is candidate-independent) and
immaterial (z spans ±0.07 over 300 seeds — never crosses the bar).

**TEETH → CONFIRMED.** Positive control: a genuinely-better fake (+5bp/day everywhere) is **ACCEPTED** — the
gate is not dead; 0-certified is about the data. The stress adversary binds independently (a calibrated
stress-harmful candidate that survives Pareto *and* holdout is VETOED on the risk_stress slice). The bloat
canary is caught at the Pareto stage at any wealth (right reason, not a wealth accident). A forced canary
escape latches `compromised=True` and every subsequent review returns REFUSED_GATE_COMPROMISED.

**HONESTY → COSMETIC (wording, fixed above).** The verifier's three corrections are now in the report and
the JSON honesty_caps: (1) the **detectability floor** — per-window MDE 330–590 bp/yr (escalating with the
bar), so the top candidate's 173 bp/yr out-of-window edge is ~half the floor: **"0 certified" means
below-floor, not proven-dead**; a *stationary* 456 bp/yr edge would have certified (expected z≈2.96). The
dev→holdout decay itself is not significant (z≈1.1) — decay-consistent, not decay-proven. (2) Axis A
seed-instability stated. (3) The upstream-selection bias is leniency-direction (91% of candidates positive
on holdout vs ~50% null), so the null survives *a fortiori*. Bonus findings: the wealth design **cost zero
certifications** (max holdout-z among the 15 refused = 0.79); and two refused candidates hit z≈2.2 on
exactly 1-of-19 window phases despite full-holdout z=0.79 — certifying them would have repeated the
project's documented **phase-selection false-GREEN** failure mode. The rotating window + escalating bar is
precisely what refused it.

## Bottom line

The v4 gate transfers to the real substrate intact — every mechanism binds on real replay data, verified
adversarially (LEAKAGE/STATS/TEETH confirmed; HONESTY wording corrected). The honest headline: **R6c's +30%
replay lift halves out-of-window (456→173 bp/yr) and lands below the single-fold detectability floor; nothing
is locally certifiable, the alpha-wealth design provably suppressed a phase-selection false-GREEN that a naive
scan would have shipped, and the cloud-budget recommendation is narrowed to `trigger_thresholds` /
`confidence_coeffs` (z≈1.6) with an expected residual edge of ~170 bp/yr — not +30% Sharpe.** The v1 flat
refusal is upgraded to a priced, tension-aware diagnosis; the promotion gate (multi-fold cloud PPO truth)
stands exactly where it stood.
