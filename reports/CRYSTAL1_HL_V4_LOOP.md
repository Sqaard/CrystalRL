# HL v4 over CRYSTAL-1 — the full governed loop on real panels (Dow-29 + csi500)

**Task:** "используй текущие наработки по R6c и создай и прогони необходимые тесты и полноценный loop для
CRYSTAL-1." Reuse the R6c-certified v4 machinery on CRYSTAL-1's native substrate; run the deciding VoI
tests; run the full loop; verify adversarially.

## The VoI decision (which mode the loop is allowed to run in)

The belief-on-capital fence (`voi_gate.py`) stands: **CLOSED on every accessible real substrate** —
BTCUSDT 2023-03 (VoI=0), SOL/AVAX/GALA 2023-08 wide-spread true-BBO (VoI=0, regime-flat negative maker
edges; the config-D law extends to real execution economics). CN A-shares remain the one untested refuge:
the L5 recorder had **died on 2026-07-04 before ever seeing a session** (0 data files) — **restarted
2026-07-06** (caught the Monday afternoon session; needs weeks of accumulation before a VoI test).
Consequence: the loop below certifies **TRANSPARENCY/RISK-mode** edits only — exactly what the gate's own
rule prescribes.

## What was built ([hl_v4_over_crystal1.py](../interpretability/hl_v4_over_crystal1.py))

CRYSTAL-1-native substrate (from the v1 loops, frozen conventions): K=2 self-supervised
`NeuralBayesFilter` on the panel's EW-return stream (train-window-only, frozen) → 4-knob exposure policy
`{t1, t2, lvl_reduced, lvl_defensive}`; anchor = static full exposure. Two real panels: Dow-29
(train 2010–16, dev 2017–18, hold 2019–20 incl. COVID, OOS 2021–23) and csi500 (train 2018–20,
dev 2021-01..08, hold 2021-09..2022-06, OOS 2022-07..2023-03).

The v4 machinery, transplanted from the R6c-certified gate:

| Mechanism | Instantiation here |
|---|---|
| Pareto axes (F1/F10) | **(ann ↑, maxDD ↑-toward-0)** — replaces v1's hand-mixed scalar `ann−&#124;maxDD&#124;` |
| **Typed moves** (new) | RETURN-type (dev ann gain) must prove mean daily delta > 0 at block-z > z_crit on the rotating holdout; **RISK-type (dev DD cut at ann non-inferiority) must prove the DD improvement at z_dd > z_crit (paired block-bootstrap of the path statistic) AND return non-inferiority** — the honest version of the B3 "risk-VoI" claim |
| Rotating holdout (F3) | 120-day date windows, stride 7, inside the hold window; alpha-investing wealth prices the storm |
| Stress adversary (F2) | anchor-defined dev slices: top-quartile trailing-vol days + deepest-drawdown-quartile days |
| Authority (F4/F11) | off-anchor knob count, cumulative budget 6 |
| Canaries (F12) | behavior-identical **config bloat** (thresholds moved, both levels 1.0 → exposure ≡ anchor) + **measured-harmful defensive lockdown** (dev-precondition, the live-provide analog); sandboxed; escape ⇒ `compromised` freeze |
| Bandit (F8/F9) | UCB1 over {t1, t2, lvl_reduced, lvl_defensive, joint_defend}; **joint_defend = the B3-shaped atomic bundle** (defend earlier AND harder, one review) — F5; Dow prior warm-starts csi500 vs a paired cold arm |
| Positive control | a sentinel candidate with a **noisy** +3bp/day edge injected via `strat_fn`, walked through the REAL `review()` path |
| Honest OOS | frozen OOS touched exactly once, at the end |

## The run

- **Dow: 0 certified** (36 no-frontier-gain, 4 holdout rejections). **csi500: 0 certified** in both arms
  (25 no-gain, 4 dominated, 8 holdout, 3 wealth-exhausted). Final coeffs = anchor on all runs; OOS
  single-shot = anchor performance (nothing was changed, nothing to overfit).
- **Canaries 4/4 caught per panel**, gate never compromised. **Positive control: ACCEPTED_RISKMODE,
  z=6.56 > 2.15 through the real review path** — the gate is alive; 0-certified is about the data.
- **F9 transfer honestly inert:** the Dow prior was all-zeros (nothing certified → nothing to transfer),
  so warm ≡ cold on csi500 identically. A zero prior carries zero information — reported as such, not as
  a transfer win.
- **Axis A (MDL) of the exposure decision = 0.000** at anchor and final — the born-legible surface keeps
  the legibility axis at the floor (R6c's production controller measured 0.286 on its 11-feature decision;
  see the wording caveat in the verification section).

## The decisive focused test: B3-config through the typed gate

The v1 loop refused the hand-tuned defensive B3-config via a mixed scalar; the standing objection was
"maybe the scalar hid a real DD benefit." The v4 typed-RISK branch was built to be *maximally favorable*
to an honest DD cut. Result:

- **Dow** (dev: B3 ann +0.166 vs anchor +0.163, DD −0.102 vs −0.160 — dominates on dev): typed RETURN,
  **REJECTED_HOLDOUT z = −1.5/−1.19/−1.22** — the dev advantage *reverses* out-of-window.
- **Forced RISK branch on Dow** (the typing-corridor check): also fails on all 5 rotating windows
  (z_dd ≈ 0.7–0.8, ni_z < bar) — **the typing rule is not load-bearing; both branches refuse B3.**
- **csi500**: typed RISK, z_dd = 0.78/0.80/0.16 (holdout DD gain only 0.1–0.3pp), NI passes — refused on
  the DD leg specifically.

So the strongest available defensive config fails even the certification lane built for it: B3's DD
benefit does not replicate out-of-window at an autocorrelation-honest bar. The v1 conclusion is confirmed
through a strictly fairer lens.

## Adversarial verification round 1 (4 skeptics) → 3 design defects found and FIXED

**TEETH → CONFIRMED.** Full-run reproduction bit-exact (trails, audit, OOS numbers). The positive control
provably traverses every gate stage (call-log instrumentation: Pareto → holdout → adversary → budget; no
early-accept path exists); a zero-mean noise sentinel is REJECTED — the accept is for the edge, not the
shape. Panel data SHA-verified read-only across an accepting review. A forced canary escape freezes the
gate persistently.

**DISCIPLINE → COSMETIC.** Zero leakage: train-only filter params/threshold (probe: train-thr ≠ full-thr
on both panels), disjoint ordered windows, no same-day lookahead, OOS consumed only at the final report
line, stress slices dev-only anchor-defined, every holdout query priced (wealth arithmetic matches the
trail exactly). Cosmetics fixed: the JSON now records `input_prior` (the Dow→csi500 transfer was ALL-ZERO
— mechanically live, informationally vacuous, warm ≡ cold structurally); canary reporting now per-type
with UNARMED entries; wording of "OOS touched once" and the csi300 file naming stated.

**TYPEDGATE + NULLHONESTY → COSMETIC, with the load-bearing find:** the Dow **RISK lane was
unwinnable-by-construction** — three independent proofs: (a) the maxDD bootstrap tests the *spread* of a
localized drawdown cut across block resamples, not its *size* (a perfect-foresight oracle passed z_dd on
2/55 windows and BOTH legs on **0/55**); (b) the COVID crash window was *unreachable*: stride-7 rotation ×
≤8 affordable tests reaches day 56, the crash starts at day 279 — every Dow test ran on calm 2019 windows;
(c) 2 of the 4 Dow holdout tests were *vacuous* (candidate inactive on the window, delta ≡ 0) yet charged
wealth. csi500's null, by contrast, was genuine (the oracle passes there 7/12). RETURN-branch MDE on calm
windows: ~3.7–12.7%/yr — high, stated.

## The three fixes (round 2)

1. **RISK statistic → downside deviation** (`risk_boot_z`): a size-sensitive tail-mean whose bootstrap SE
   shrinks with evidence instead of scaling with the gain. **Winnability restored and verified: the oracle
   now passes both legs on the crash window (z_dsd=+2.98, ni=+3.26 > 2.15).**
2. **Inert-window guard**: `REFUSED_INERT_ON_WINDOW`, no wealth charge (an uninformative test must not
   burn certification budget). Effect visible: csi500 wealth-exhausted refusals 3→1 — one more real test
   afforded.
3. **Span-covering rotation**: stride = span/7, so the ≤8 affordable tests cover the whole hold window —
   Dow starts [0, 54, …, 324, 378] now include the crash.

**The sharpened B3 verdict (the payoff of the typed lane):** on the Dow crash window B3-config now
**passes the downside leg (z_dsd=+3.32 — its tail-risk cut is REAL and detectable)** but **fails return
non-inferiority (ni=−0.25)**. For the first time the B3 story is split into its two legs: the drawdown
benefit exists, and it is bought at a return cost outside the certifiable region. The refusal is now a
*priced trade-off statement*, not a flat null.

**Re-run after fixes:** still **0 certified** on all three runs (Dow 36 no-gain + 4 holdout; csi500
25+4+8+1 with 2 inert refusals) — now as a *genuine data null on a winnable lane*: the gate accepts the
oracle and the noisy positive control, refuses every expressible knob move. Harm canaries honestly
UNARMED (8 audit entries): in this lever space (exposure ≤ 1, cost 10 bp) no expressible config measures
harmful on the calm dev windows — a safety property of the C-6 lever design made visible.

## Final verification (red-team round 2) → CONFIRMED_FIXED, then its findings hardened in-code

The focused red-teamer reproduced all three fixes exactly (oracle winnability robust across seeds; inert
guard charges nothing yet does not swallow active-but-tiny candidates; rotation covers the crash) and then
ran **11,126 full-gate mining runs** (1,260-config knob grid × every reachable window at cheapest pricing):

- **6 accepts total (0.05%)**; five form a **window-33 cluster on csi500** that passes a single window
  while violating its certified non-inferiority off-window by 2–4× the margin — the B3 crash-timing
  pathology relocated. The red-team then ran the exploit end-to-end through the live gate and found it
  **blocked only by coincidence** (belief peaks happen to exceed the t2 cap, so inert "window-shopping"
  spins get priced and the escalated bar refuses the payload).
- Cost-shifting attacks on the new statistic failed (0/8 seeds for injected single-day losses; the only
  passes also genuinely improve maxDD): downside deviation catches what maxDD caught.
- The z_dsd null is anti-conservative (empirical sd≈1.38); the two-leg requirement contains it (harmless
  random de-risking passes both legs at ~nominal alpha).

**Both of the red-team's named hardenings are now in-code, and its own exploit was re-run against them:**

1. **Inert refusals no longer advance the rotation** — 5 inert spins leave the epoch at 0
   (window-shopping dead by design).
2. **RISK accepts must re-confirm the non-inferiority claim on the maximally-distant disjoint window**
   (same priced query; also contains the anti-conservative null). The mined payload, forced onto its best
   window at the cheapest bar, passes leg one (z_dsd=2.22, ni=4.02) and **dies at the confirm
   (ni2=0.97 on the disjoint window) → REJECTED_HOLDOUT**.
3. The canary honesty cap now states the empirical truth: on calm dev windows *no* expressible
   bounded-exposure config measures harmful on dev (both harm canaries UNARMED, logged); harm-rejection
   teeth on such panels rest on the escape⇒freeze wiring + the positive/negative controls, which were
   verified live.

## Bottom line

The full v4 loop now runs governed over CRYSTAL-1 on two real panels with every R6c-proven mechanism
transplanted and re-verified (5 adversarial agents across 2 rounds; the red-team's own end-to-end exploit
is dead by design). The honest result: **0 risk-mode edits certifiable on Dow/csi500 — now as a genuine
data null on a provably winnable lane** (the oracle certifies, the positive control certifies, B3's
tail-risk cut is real but return-costly: z_dsd=+3.32 / ni=−0.25). CRYSTAL-1 keeps Axis A at 0.0 while
governed (vs R6c's 0.286, directional comparison); belief-on-capital stays fenced by the VoI gate
(CLOSED on BTC/SOL/AVAX/GALA; **CN A-shares is the open refuge — the L5 recorder was found dead and
restarted 2026-07-06, accumulating toward the first CN VoI test**). When any substrate opens the VoI
gate, this loop — gate, canaries, typed lanes, priced queries, transfer machinery — points at it as-is.
