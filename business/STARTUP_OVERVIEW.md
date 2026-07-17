# X-Lab — Startup Overview (the strategic anchor)

> **Role of this file:** the single summary of the venture — vision, stack, research split, business
> frame, honest state, milestones. Strategic decisions in this project should be checked against
> this document (and it, in turn, defers to [`NORTH_STAR.md`](NORTH_STAR.md) →
> [`WORK_ORDER.md`](WORK_ORDER.md) on anything product-claim-related). Repository map:
> [`MOTHERSHIP.md`](MOTHERSHIP.md). Founders: **Ivan Pavliuk** (research/engineering, model & data)
> and **Joseph Lynch** (interpretability research; business development). Status: draft for the
> founders' Sunday discussion — terms below are proposals until both confirm.

## 1. Vision (one paragraph)

We are building the first personal-investment product whose core competitive asset is **trust made
auditable**: a client states what they can lose, their horizon and their goal; the system returns a
conservative return floor it can defend at a stated probability — and *refuses* when its own
evidence is insufficient. Underneath sits a research bet that is bigger than one product: **a
self-evolving agent is safe and useful exactly to the degree it can be read.** The company
industrializes that sentence: an interpretable agent (CrystalRL) + an LLM coding-agent improvement
loop under experimental controls (Hello Crystal), wrapped in calibration, pre-registration and
fail-closed governance.

## 2. The stack (four parts, and which are frozen)

```
1) FinIR   — information retrieval: the right to READ (point-in-time evidence, provenance)   [frozen]
2) FinGPT  — text processing: the right to MEANING (LLM as constrained feature generator)    [frozen]
3) CrystalRL — interpretability: what the agent believes, plans, optimizes                   [ACTIVE — Joseph]
4) Hello Crystal — self-development: use (3) to gate an LLM loop that improves OOS metrics   [ACTIVE — Ivan]
```

Layers 1–2 are the basis of a future live-trading system but **stay fixed** until the 3↔4 loop is
stable on historical data. The near-term product object is the certified
`profile × horizon × return-quantile` frontier (see the north star), not live text-driven alpha.

## 3. The two research questions (the division of labor)

- **CrystalRL (Joseph leads, Ivan co-authors).** *Can we construct a causal, predictive and
  scalable account of what an agent believes, plans and optimizes?* This is the cutting edge of
  Explainable RL. Our agent already has interventions (`SET_BELIEF` do-writes), teachers
  (pretraining/BC), and a named belief mechanism — but **its causality is currently weak**: the
  controls showed the RL prototype's belief→action link is not faithful (CrystalScore 0.03 vs the
  readable champion rule's 1.0). Closing that gap — making the causal account real, testable and
  scalable — *is* the track.
- **Hello Crystal (Ivan leads, Joseph co-authors).** *Can that account be used to predict failures
  under distribution shift and improve out-of-sample metrics?* The loop machinery (proposals →
  controls → gates → certification), calibration, and the OOS discipline (pre-registered reads
  2027-07-12 / 2029-07).

The two questions interlock: (3) supplies the readable object and the causal account; (4) supplies
the evidence standard that keeps (3) honest. Both papers are co-authored by both founders, with
lead/driver as above.

## 4. Business frame (proposal — to be confirmed together)

- **Co-founding, 50/50 equity, with standard founder vesting** (proposed: 4 years, 1-year cliff,
  for both founders — protects each of us and the company if life intervenes; exact mechanics to be
  agreed before incorporation).
- Rationale: Ivan brings the idea and ~3 months of significant scientific progress (the working
  stack, two paper drafts, one certified result, the governance machinery); Joseph brings the
  interpretability research track and business development. What the startup lacks is not capital
  but **intellectual investment** — both founders' contribution is research that compounds.
- **Strategic decisions are taken jointly.** Open mechanics to agree early (a short founders'
  memo, not lawyers yet): vesting/cliff, a deadlock-breaking rule for a 50/50 split, IP assignment
  (research papers are co-authored academic IP; code/product IP assigns to the company),
  incorporation jurisdiction and its constraints, and what happens to equity if either founder
  steps back.
- **Publishing is part of the moat**, not a leak: the papers (CrystalRL, Hello Crystal) build the
  credibility that the product's "auditable trust" claim needs; the certified machinery, data
  contracts and governance stack stay proprietary.

## 5. What is honestly true today (the investor-grade summary)

| Claim | Status |
|---|---|
| Certified real-data policy (defensive Dow rule) on 629 untouched days | ✅ Sharpe 1.46 vs 1.39; maxDD −12.9% vs −15.5% |
| Transparent DP champion beats best static (+10.5pp P(goal)) and every RL challenger | ✅ |
| 80%-promises hold on US (5/5 profiles, 576 PIT cells; risk 100%/936) | ✅ development-tier |
| China portability | ❌ 1/5 — separate verdict, named fixes queued |
| Alpha on daily data | ❌ triple-confirmed null (the reason for the pivot) |
| RL prototype as faithful command surface | ❌ CrystalScore 0.03 (the CrystalRL track's target) |
| Client certification | ⛔ REJECTED_HONEST by design; unlocks are dated (2027-07-12, 2029-07) |
| Self-evolving loop | ✅ machinery proven (autonomous lifetime run: 40 proposals → 4 certified, self-funding evidence budget); value on real daily data still modest — honest |

## 6. Milestones (the shared clock)

1. **Now → autumn 2026:** Joseph onboards, owns CrystalRL; close the binding evidence gaps of the
   work order (§6: W3 calibration 10/10, W6 residual bounds, W7 Crystal inputs, 10 replay→retrain
   pairs). Two papers to preprint quality.
2. **2027-07-12:** first pre-registered read of the locked one-year forecast ledger — the first
   *external* evidence point for the calibration claim.
3. **2027–2029:** paper-track accumulation, the certified frontier grows via the HL loop; product
   surface (W9) stays fail-closed until W8 certifies.
4. **July 2029:** the untouched 3-year policy fold matures — the strongest single validation event
   on the calendar.

## 7. Standing principles (the culture, in five lines)

1. **Feynman first:** you must not fool yourself — and you are the easiest person to fool.
2. **Controls decide, not scores** (placebo / wrong-direction / dose / out-of-time / one-shot holdout).
3. **Refusal is a feature**; every probability wears its evidence tier.
4. **Verdicts are never averaged across universes**; used OOS is never fresh confirmation.
5. **Write the paper first; log every run** — an unlogged result does not exist.
