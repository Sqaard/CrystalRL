# CRYSTAL-WORLD W0–W4 — the results section (program complete, 2026-07-19)

*The paper-grade summary of the first full pass through the CRYSTAL-WORLD methodology
([theory](../business/CRYSTAL_WORLD_THEORY.md) · [methodology](../business/CRYSTAL_WORLD_METHODOLOGY.md)).
Every phase was adversarially reviewed; six harness versions were invalidated along the way and
every invalidation is part of the record. Written for direct porting into the CrystalRL paper.*

## Headline results

| Phase | Read | Verdict (referee-corrected) |
|---|---|---|
| **W0 / KT-A** | Can L1 latent prediction beat trivial dynamics? | **SPLIT** — real, modest, LINEAR structure (tuned ridge +0.22 hold z 2.2 / +0.35 OOS z 2.6 over fair nulls); a small nonlinear JEPA adds nothing over PCA at this scale. *"L1 is linear until proven otherwise."* |
| **W1 / KT-B** | Does abstraction buy horizon? (the T5 centerpiece) | **Ordering PASS**: reach(L1-week)=40d < reach(L2-month)=84d under the identified protocol (strong nulls, placebo guards, OOS confirms all cells z 6.5–10.7). The representational dividend (pooled > flat twin) is **directional, not significant** (paired z −0.34, closed in W4) — so the honest reading sits between FULL and MECHANICAL: *target aggregation certainly, representation hierarchy plausibly.* |
| **W2** | Probe loop, detours, action channel, KT-C | Episodic store live (822 records). Detour trigger PASS ≥2/3 (COVID 117×, rates 4.4×, tariff marginal) — **generic novelty**, valid as an enrichment trigger, not model-specific detection. Action channel inert as pre-registered. KT-C: hold-bar pass but **OOS-degrading** → led to W3's discard. |
| **W3** | L2 improve-or-discard; semantic store; KT-D | **L2-as-substitute DISCARDED (genuine)** — mechanism named: causal-filter OOD lock-in on the 10Y-yield *level* vs the frozen scaler (posterior pinned at 1.0). Semantic store v0 live. KT-D recompute-operator KILL was **manufactured** (channel permutation) — caught by review; semantics-stable operator emerged as the live lead. |
| **W4** | Interpretability battery + frontier + prereg closes | Below. |

## W4 — the battery on the surviving representation (L1 = PCA-10 of 5d-block features)

- **Naming**: every dim auto-named from loadings (dim1 «+max_dd −VIX +trend», dim2 «+dVIX −ebp
  −ret_sum», …) — the born-legible discipline holds for a linear stack by construction.
- **Simulatability S = 0.70**: 3-feature sparse stories reproduce the dims at R² 0.32–0.97,
  measured out of the naming sample.
- **Stability**: top-8 subspace overlap across train halves = 0.98, **but** the referee showed the
  chance floor is 0.80 (8-of-10-dim ambient) — chance-rescaled ≈ 0.90, and the stricter k=4 read
  is ≈ 0.71. **Honest CrystalScore(L1) = 0.50–0.69** depending on the stability convention
  (vs 0.03 for the cold RL head, 0.92 for the certified champion). Dims 3–8 are individually
  unstable — only the top-2 dims are individually trustworthy named objects.
- **Faithfulness F = 1.0, born-linear** — a property of choosing linearity, not an achievement.
- **The quality–parsimony frontier** (identified protocol, same cell machinery): D=2 → 0.16,
  D=4 → 0.22, D=8 → 0.22, D=10 → 0.24. **The knee is at D≈4**: four named dims buy ~90% of the
  full representation's skill. (Frontier cells carry no placebo guard — cite the shape, not the
  z values.)

## The pre-registered closes

**KT-B third window (cross-market, csi500 — the only data no KT-B analysis ever touched):
UNDERPOWERED / UNREPLICATED.** No significant cell survives the guards on csi500 hold. The
referee sharpened the attribution: the L1@5d candidate cell was not underpowered but
**correctly placebo-killed — its entire margin is generic shrinkage mechanics** (plain train-mean
beats the ridge: +0.150 vs +0.149); the genuinely underpowered cells are L2's (n=17, z 1.4–1.8).
Status of T5-on-markets after this close: **measured on Dow (ordering PASS, twice-confirmed
in-market), unreplicated cross-market; the certified regime rule remains the strongest
independent evidence.** The Dow result stands; its generality is an open question for bigger
cross-market data, not for more Dow analysis.

**KT-D v2 (semantics-stable enrichment, preregistered on csi500): MIXED / HORIZON-CONDITIONAL.**
PASS at k=1 per the prereg letter — and it survives the referee's purge fix (a real year-boundary
look-ahead in the growing stats; purged z 1.43, still above the 1.28 bar; positive all 3 years) —
but **significantly negative at k=4** (z −2.4, driven by 2025). Enrichment helps 5-day forward
statistics and hurts 20-day ones on the clean market. **Consolidation may be wired only with a
horizon gate (k=1 statistics only)**, and the cumulative evidence is one post-hoc positive (Dow),
one preregistered short-horizon positive and one significant long-horizon negative (csi500).

## The methodology's own report card

Two of the two pre-registered "deaths" (§8) were tested and neither killed the program: the
hierarchy signature passed (Dow) and enrichment survived in horizon-conditional form. The
program's real product beyond the verdicts is the **artifact-class register** accumulated by
refute-first review of every phase — eight classes this week: predictor collapse under
encoder-only regularization; window-overlap mechanics; feature-level overlap (trailing
aggregates); untuned-baseline flattery; frozen-mean-null inflation under level shift; capacity
starvation; channel-permutation manufactured kills; chance-floor-inflated stability. Each is now
a rule in the semantic store or the methodology. **The week's meta-result: on this substrate, at
these scales, a NULL is exactly as manufacturable as a PASS — and a review process that attacks
both directions catches both.**

## What Joseph inherits (all in Sqaard/CrystalRL)

The theory file with evidence-tagged theses; the completed methodology with per-phase status;
five experiment families with reports and review files; the episodic + semantic stores; the
frontier (D≈4 named dims ≈ 90% of skill) as the starting point for the interpretability paper
section; and the two open forward items — a cross-market hierarchy read at adequate power, and
the horizon-gated consolidation experiment.
