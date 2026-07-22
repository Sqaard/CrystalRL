# PREREGISTRATION — HL-MEM A/B v5 (registered 2026-07-22, BEFORE any v5 run)

Registered on the day the v4 review closed (CONFIRMED_NARROWED, `exp_hlmem_ab_v4_review.json`);
no v5 code has been run. This file locks the design so the budget-80 inversion found post-hoc in
v4 cannot be "confirmed" by the same scope error in the opposite direction.

## What v4 established (the baseline for v5)

- Preregistered read at budget 40: **well-powered NULL** (R1 D_live−B = −1.1, power 0.87) — no
  conditioning component shipped.
- One-wire harm: REPAIR step_mult 1.6 on frontier bars blocks step annealing into the accept band
  (~0.03–0.06); C−B = −1.7, p=0.0004. Selection channel harmless-to-helpful (C_prefer_only 4.75).
- Cards NEUTRAL (v4 R6 confound retracted); islands directional-only; linter unexercised.
- Post-hoc, unreplicated: at budget 80 C−B inverts to **+1.95** (p=0.018) because the baseline 0.6
  shrink self-traps below the band — **neither fixed step policy is adaptive**.

## v5 changes (all fixed now, none tuned after results)

1. **Vocabulary split (hygiene-preserving):** `REJECTED_NO_FRONTIER_GAIN` → `no_gain_too_bold` /
   `no_gain_too_timid`, disclosing ONLY the binary side of which frontier inequality failed (the
   gate already returns this in `info`; no raw values, no window identities — Thresholdout intact).
   REPAIR: too_bold → step_mult 0.6, too_timid → step_mult 1.4 (fixed here, before any run).
2. **Baseline step floor + re-expansion (ALL arms):** step floor = 0.02×range; after 5 consecutive
   rejects at the floor, re-expand ×2 (fixes the A/B self-trap; note this shifts every arm's mean —
   v5 results are NOT comparable to v4 numbers, only arm contrasts within v5 count).
3. **Islands patience on PRICED queries only** (per the original docstring; patience=8).
4. **Horizons preregistered at BOTH budgets: 40 and 80.** Primary endpoint unchanged (certified
   accepts per budget); 20 seeds; paired sign-flip permutation.

## Preregistered v5 reads

- **R1′ (per horizon):** C′−B′ at budget 40 and at budget 80, two-sided p<0.05 with ×2 horizon
  correction. Prediction registered: split-vocabulary conditioning ≥ B′ at both horizons; if C′<B′
  at both, hand-coded repair is dead regardless of vocabulary and only the LLM-writer lever remains.
- **R2′ (inversion replication):** does the v4 budget-80 pattern (conditioning > amnesiac at long
  horizons) replicate with the fixed baseline? If the step floor alone erases the gap, the v4
  inversion was a baseline artifact, not memory value.
- **R3′ (tail guard):** any shipped component must ALSO pass a tail read — no more than 2/20 seeds
  below 2 accepts (v4 C's catastrophic-variance tail, sd 3.34 at budget 80, is the standing risk).
- **Islands:** E′−B′ at both horizons with the priced-patience fix; ship only on p<0.05 AND tail
  guard.
- Gate UNCHANGED (v12 lineage, shared core — business/SCOPE_AND_BRANCHES.md); dev 2019-21 /
  hold 2022-23 only; OOS never loaded; all attempts ledgered (N for Bailey deflation reporting).

## Not in scope for v5

The LLM hypothesis-writer layer (separate prereg when it comes); any REPAIR value re-tuning beyond
the two fixed multipliers above; any substrate change (same de-tuned incumbent — one-substrate
scoping is a known limitation, recorded, not silently widened).
