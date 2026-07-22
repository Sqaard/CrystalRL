# Memory between attempts — the literature analysis and the HL-MEM design (2026-07-20)

**Ivan's diagnosis:** the heuristic learning loop has a huge hole — no memory between attempts;
the agent must extract a HYPOTHESIS from the deviation, not just try again.
**Sources digested:** Weng's "Learning Beyond Gradients" essay (the HL origin), the
Lightweight_Quant_Agent_Project_Plan (which already *specifies* the card system), Khashabi's
group read closely per instruction, and a **55-paper multi-agent sweep** across 8 families with
**5/5 load-bearing claims CONFIRMED by adversarial verification** — the first fully-confirmed
verification pass in this project's history. Raw layer:
[memory_between_attempts_lit_20260720_raw.json](memory_between_attempts_lit_20260720_raw.json).

## 0. The diagnosis is confirmed, with mechanism

The 0/20 converged-quiet autonomous run is not an anomaly — it is the *predicted* signature of a
memoryless verdict-only loop:

- **Feedback Friction** (Jiang, …, **Khashabi**; NeurIPS 2025, arXiv:2506.11930 — the exact paper
  Ivan cited, verified): even with near-ground-truth targeted feedback over 10 iterations, models
  plateau in 2-4 iterations below ceiling; of the errors still unsolved at iteration 10,
  **62.8–100% are feedback-resistant** (correct feedback given, still not fixed). Binary
  "wrong!" is the measured WORST feedback class — moving to targeted reflective feedback is worth
  **+26.7–33.3pp**; the single best mitigation is **explicit exclusion of previously-attempted
  answers** — Weng's "written-down failed directions", measured.
- **Self-Debug's Spider ablation**: verdict-only retry = **+0.0** — literally our current loop.
- **AutoMat** (…, **Khashabi**; arXiv:2605.00803, the materials-science reproduction paper,
  verified): best coding agents 54.1% overall, **near-zero when the procedure must be
  reconstructed from prose** — a rejected attempt whose procedure isn't ledgered as a runnable
  artifact is a "from-paper" reconstruction next time. Key failure signature: **methodological
  deviation** — hitting the metric via an off-spec route (the Goodhart pattern our gate audits).
- **SELF-[IN]CORRECT** (…, **Khashabi**; AAAI 2025): the proposer's own discrimination adds ~0
  over its generation (54/56 nulls) — "ask the model what went wrong" is not a memory design.

Khashabi is confirmed as directly on point — and, per Ivan's caveat, not the last word: the
critical self-correction line (Huang ICLR 2024: intrinsic self-correction *degrades* GSM8K
95.5→89.0 while the same loop with an oracle gains +8-10pp) and the archive-evolution line
(FunSearch/AlphaEvolve) matter as much.

## 1. The four established facts (all verification-CONFIRMED)

1. **Memory between attempts is load-bearing, not decoration.** FunSearch's SI ablation
   (no-evolution is the worst arm even at millions of samples; the cap-set result hit in only
   4/140 seeds), AlphaEvolve, Trace's NoMem arms, Algorithm Distillation's context-span law
   (in-context improvement emerges only when 2-4 full (attempt, outcome) episodes fit in
   context — degenerates to policy-maintenance otherwise: the exact 0/20 signature), and
   500-queries-with-memory ≳ 100k-without (Zhang et al.).
2. **The sign of reflective memory flips on evaluator trustworthiness.** Reflection helps ONLY
   with an external verifier; with self-evaluation it hurts (Huang; Self-[In]Correct; Reflexion's
   16%-corrupt-verifier inversion). **Our frozen gate is why we are in the positive regime — gate
   integrity and memory value are one coupled requirement.**
3. **Naive accumulation hurts; admission-gated, bounded, frozen-schema memory is the fix.**
   Xiong et al. (ACL 2026): add-all memory underperforms FROZEN memory on all 4 agents (32.3 vs
   40.1); only strict-evaluator-gated admission wins — **an independent replication of our own
   KT-D result**, with the remedy quantified.
4. **Structured deviation content carries the largest measurable share of feedback value.**
   Stechly (+27pp from first-error content in Blocksworld), Eureka (−28.6% without decomposed
   reward reflection), Tyen (LLMs locate the first error ~50% even on clean traces, but
   correction given oracle location gains +18-44pp) — **the locator must be deterministic; the
   LLM only writes the hypothesis.**

Also confirmed: **Dwork's reusable holdout / Thresholdout** gives the concrete query-hygiene
mechanism (coarse, thresholded feedback preserves validity of a reused evaluator), and
**QuantAgent + Microsoft RD-Agent(Q)** are validated (not demoware) knowledge-base loops in quant
specifically.

## 2. The 12 design principles (each grounded in papers + our scars)

1. Reflection only from the frozen gate, never from self-judgment.
2. Verdict upgraded from 1 bit to a structured deviation report — through a **Dwork-hygienic
   channel** (quantized bands, never raw z).
3. **Split location from hypothesis**: deterministic locator (the gate's instrumentation IS the
   oracle), LLM writes the hypothesis card only.
4. Frozen schema, bounded size, admission-gated writes — the KT-D-compatible regime.
5. **Failed directions are first-class; exclusion enforced in code (linter), not in the prompt.**
6. Index memory by behavioral signature (per-input score tuples à la FunSearch), not code text —
   frozen semantics by construction; kills the experience-following pollution mode.
7. A HYP card is admissible only citing executed evidence + a falsifiable prediction;
   **credibility = hit-rate, not eloquence** (CRITIC; anti-sycophancy).
8. Portfolio, not a greedy lineage, with a diversity alarm for converged-quiet (FunSearch 4/140).
9. Prompt-as-memory-interface: last 2-4 FULL cycles, ascending by score, scores visible (AD; OPRO).
10. Success-side memory is procedural and certified-only; failures never become retrievable
    skills (Voyager; AWM's self-labeled variant is the Goodhart channel).
11. **The ledger is the gate's statistics**: every attempt (incl. linter refusals) counts toward
    N for threshold deflation (Bailey–López de Prado: ~45 unreported trials make SR≈1 free).
12. The gate, its config and the card-store write path sit outside the proposer's writable
    surface (the AI Scientist edited its own timeout; our E1 exploit — same threat class).

## 3. HL-MEM v0.1 — seven components around the unchanged v12 gate

1. **Deviation Reporter** (deterministic, disclosure-limited): frozen vocabulary
   {first_failed_bar, margin_band (tercile, never raw), killing_twin_family, regime_tag,
   replay_ptr} — ~4-5 bits/attempt instead of 1, Thresholdout-coarse.
2. **Card Store** — the plan's HYP→EXP→RUN→RESULT→AUDIT chain, finally implemented: frozen JSON
   schemas, append-only, cross-linked IDs; HYP requires mechanism_class (the HS/ vocabulary),
   cited DEV/RUN evidence, a falsifiable prediction + kill condition; AUDIT resolves predictions
   deterministically and maintains per-card hit-rates; only AUDIT-validated cards are
   retrievable; tombstoning by retrieval-utility; caps: 3 active HYP per lineage, depth 3.
3. **Negative-Direction Registry** — falsified (mechanism, knob, regime) triples with killing RUN
   ids + the standing in-house negative knowledge as a frozen context block.
4. **Proposal Linter** (pre-gate, code-level): frozen-embedding diff vs the registry and vs all
   prior proposals; refusals cite the killer and are ledgered (they count toward N).
5. **Skill Registry** (certified-only): E-15, backwardation, future accepts as named composable
   primitives with provenance.
6. **Portfolio Scheduler**: islands over mechanism classes; converged-quiet triggers migration,
   not silence.
7. **Threshold Deflation Hook**: the gate's acceptance thresholds read attempt-count N from the
   ledger (Bailey deflation) — the loop pays for its own multiplicity.

## 4. The pre-registered evaluation (the honest A/B)

Four arms at a **matched budget of 40 gate submissions × ≥5 seeds**, dev/hold windows only (the
2027 prereg and untouched windows are never spent on loop-architecture questions):
**A** strengthened amnesiac control (best prompt + frozen negative-knowledge block — memory must
beat a *good* memoryless proposer, per Huang's weak-baseline critique); **B** linter-only
(exclusion null — if FULL can't beat B, hypothesis extraction is theater and only the linter
ships); **C** B + Deviation Reporter (isolates the structured-location term — predicted to carry
most of the lift); **D** FULL (cards + AUDIT + portfolio), with a FROZEN-store vs LIVE-store
sub-split (the direct KT-D replication on this memory). Primary endpoint: certified accepts per
budget; the standing placebo/twin battery applies to any resulting rule as always.

## 5. Honest verdict

The literature genuinely supports: memory is load-bearing (verified ablations), our frozen gate
is the precondition that puts us in the positive regime, naive accumulation would hurt (our own
KT-D, independently replicated), and the structured deviation report is the highest-value single
component. What remains speculative: the size of the gain on OUR substrate (all numbers come from
reasoning/code/game domains; the finance-specific loops — QuantAgent, RD-Agent(Q) — are validated
but report generous evaluation regimes), and whether the friction floor (Feedback Friction's
residual) leaves enough headroom at a 40-submission budget. That is exactly what the four-arm A/B
answers, and nothing ships without it.
