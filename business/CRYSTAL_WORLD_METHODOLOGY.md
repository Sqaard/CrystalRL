# CRYSTAL-WORLD — Methodology (build spec v0.1)

*The engineering companion to [CRYSTAL_WORLD_THEORY.md](CRYSTAL_WORLD_THEORY.md). Defines the
CRYSTAL-WORLD model — a JEPA-style hierarchical world model for markets that predicts
representations, compares them via an energy head, and enriches a persistent economic world
picture — plus the pre-registered kill tests that keep it honest. 2026-07-18.*

---

## 1. Objective

Build a model that (a) encodes the market situation into representations at several abstraction
levels, (b) predicts its OWN future representations per level (not prices), (c) compares predicted
vs realized representations with an energy head, and (d) routes the comparison errors into a
gated, audited enrichment of the economic world picture. Trading remains the probe: any
deployable claim still passes the v12 frozen gate; the VoI fence and all house rules
(exposure-matched twins, capacity-fair placebos, pre-registration) apply unchanged.

## 2. Architecture

```
            x_t  (daily panel window: 41 engineered cols + vrp/ebp + book state)
              │
        ┌─────▼─────┐    L1 "market micro-state"   s1_t ∈ R^16      horizon target: 1-5d
        │  Encoder  │──► L2 "regime"               s2_t ∈ Δ^K (K≤4, crisp, NAMED)  10-20d
        └───────────┘    L3 "macro-cycle"          s3_t ∈ R^4 (named dims)         1-6m
              │
   per level ℓ: JEPA predictor  P_ℓ(ŝℓ_{t+h} | sℓ_t, a_t, z)   (z = latent uncertainty var)
   energy head: E_ℓ(ŝℓ_{t+h}, sℓ_{t+h})  — "clever compare" (T4)
   cross-level consistency: s2 ≈ slow-pool(s1), s3 ≈ slow-pool(s2)  (Simon near-decomposability)
```

Design commitments, each traceable to the theory file:
- **Levels 2-3 are crisp and born-named** (K-simplex, the CRYSTAL-1 lesson; T6-reformulated:
  representations must be auditable). L2's prototype already exists — P(bear) is a 1-D crisp L2.
- **Per-level horizons grow with level** (T5); each level trains on ITS horizon, not a shared one.
- **Anti-collapse is architectural, not optional** (G1): VICReg/SIGReg variance-covariance floors
  on every level + EMA target encoder; plus the behavioral check — a collapsed level fails its
  variance floor AND its twin test.
- **Predictor is action-conditioned** (a_t = the book's exposure decision) so the model can learn
  its own footprint — the performativity amendment (G5) made concrete.
- Start small: the encoder is a shallow MLP/TCN over 60-90d windows, ~10⁵ params (Fin-JEPA scale,
  which proved SIGReg-stabilized training on equities feasible at 367K params).

## 3. Memory — the economic world picture (CLS-structured)

Two stores, per Complementary Learning Systems (theory §2), formalizing what the project already
half-built:

- **Episodic store** (fast, append-only, dated): per-day records {representations s1-s3, predicted
  ŝ, energies, actions, outcomes, market context}. The EXPERIMENT_LOGBOOK and ledgers are the
  human-facing shadow of this store.
- **Semantic store** (slow, gated): the world picture itself — named regimes and their transition
  structure, validated mechanisms with Evidence-Status tags, **each regularity's crowdedness and
  decay clock** (amended T1), and the negative-knowledge register as a first-class section.
  The DeepSearch world-picture corpus seeds it.

**Enrichment protocol** (T3, with G10 guardrails):
1. Small, schema-consistent errors → parameter-level updates (episodic only; fast path per Tse).
2. Persistent/structured error clusters (schema violations) → the HL loop drafts a semantic-store
   update proposal (new named state, amended mechanism, new decay-clock entry).
3. Proposals pass a consolidation gate: pre-registered, placebo-controlled, and REVERSIBLE
   (every semantic write carries provenance + a rollback path). No raw appends, ever.
4. Consolidation is slow and interleaved (CLS) — the frozen-champion discipline stays; CL-1c
   proved naive continual updating underperforms, and the theory explains why.

## 4. The learning loop

```
perceive(x_t) → encode(s1,s2,s3) → predict(ŝℓ_{t+hℓ} per level) → act (paper-trade probe; real
deployment only via the ratified lane) → realize(sℓ_{t+hℓ}) → compare (energies Eℓ) →
route errors (§3) → log (episodic) → [gated] consolidate (semantic)
```

- The **act** step runs on the existing live_executor paper track — the loop needs no new broker
  machinery and no governance change.
- The compare step is **adversarially protected** (G4): energies are evaluated on held-out
  segments the predictor never trains on, and periodic "detour tests" (G3) inject known structural
  breaks (2020-03, 2022) to check the picture's coherence off the trained manifold.

## 5. Pre-registered evaluation battery (the kill tests)

**KT-A — representation-prediction skill (W0 gate).** Per level ℓ and horizon h: predictor must
beat (a) persistence (ŝ=s_t), (b) a linear/PCA baseline, (c) a shuffled-target placebo, on
held-out energy margin / retrieval accuracy. Kill: no level beats persistence + linear → the
substrate does not support latent prediction beyond trivial dynamics.

**KT-B — THE HIERARCHY SIGNATURE (the falsifiable centerpiece).** Build the FSLE-style curve:
skill half-life h½(ℓ) per level (the horizon where the predictor's edge over persistence halves).
PASS iff h½(L1) < h½(L2) < h½(L3) with non-overlapping windows (G8: Boudoukh) and a flat
capacity-matched twin failing to reproduce the ordering (Nachum control). Kill: flat or inverted
curve → T5 fails on markets and the theory file gets a major amendment. *This is the single most
important scientific read of the program.*

**KT-C — downstream compatibility.** Swapping L2 for the HMM P(bear) inside the certified rule
must not degrade it (twin z within −0.5); any claimed IMPROVEMENT goes through the v12 gate with
the exposure-matched dial twin (house rule, CL-1 lesson). L2 must also pass Joseph's battery:
CrystalScore on representations, contrast-write probes (boundary-aware — the BH1 stage-3 lesson).

**KT-D — enrichment value.** Walk-forward: semantic-store updates consolidated at year Y must
improve representation skill in year Y+1 vs a frozen-memory twin. Kill: enrichment ≤ frozen (the
CL-1c risk, now at the world-picture level). This is the empirical content of "the bot ENRICHES
the picture" — without KT-D, enrichment is decoration.

**KT-E — sanity/collapse battery.** Variance/covariance floors per level; a placebo market
(block-shuffled returns) must yield NO skill anywhere; energy calibration checked before any
probability-like claim (G2 — contracts stay on the bootstrap machinery).

## 6. Phased build plan (cheap-first, each phase gated by the previous)

- **W0 (probe): EXECUTED 2026-07-18 — KT-A = SPLIT** (`interpretability/exp_w0_jepa_kta_v4.py`,
  after a four-iteration harness arc that caught three new artifact classes — see the logbook
  entry). The L1 level has real, modest, LINEAR predictable structure in representation space
  (tuned PCA+ridge margin +0.22 hold z 2.2 / +0.35 OOS z 2.6 over the fair null); a small
  nonlinear JEPA adds nothing over PCA at this data scale (placebo clean). Consequence: **L1 is
  linear until proven otherwise.**
- **W1 (amended per W0): EXECUTED 2026-07-18 — KT-B = FULL PASS** under the identified protocol
  (`interpretability/exp_w1_ktb_v2.py`; the v1 run was invalidated by review on BOTH sides — the
  register's first two-sided invalidation — and the v2 protocol fixed per the referee's
  prescription): **reach(L1)=40d < reach(L2)=84d**, pooled-L2 beats its flat twin, placebo guard
  auto-excluded the mechanical mean-null cells, and OOS confirms every significant cell (z 6.5-10.7).
  The hierarchy-horizon principle has direct market evidence beyond the regime data point. Caveats
  in the logbook (protocol-after-peek → the clean confirmation is a pre-registered third-window
  read; flat-twin point comparison without a z). L3 has no statistical content at n=8.
- **W2: EXECUTED 2026-07-19** (`exp_w2_probe_loop.py`, `exp_w2_ktc_l2_swap.py`; review
  MINOR_ISSUES — the first phase to survive without an invalidation). Episodic store v0 live
  (822 records). Detour tests PASS (≥2/3 robust; correction: the surprise elevation is generic
  novelty — a valid enrichment TRIGGER, not model-specific detection). Action channel inert as
  pre-registered (interface delivered; footprint untestable on history). **KT-C = hold-bar pass
  but OOS-DEGRADING and PROVISIONAL: L2 is largely a noisy re-encode of the HMM decision
  (R²=0.63, 89% agreement) whose independent 11% degrades OOS — W3 must improve or discard L2
  before building memory on it.**
- **W3: EXECUTED 2026-07-19** (`exp_w3_l2_improve.py`, `exp_w3_ktd_memory.py`; review caught a
  manufactured kill). **L2-as-substitute DISCARDED, genuine** (mechanism named: causal-filter OOD
  lock-in on the 10Y-yield level vs the frozen scaler; rule: future L2 rebuilds exclude raw
  levels). Semantic store v0 live with curated + mechanical entries. **KT-D = UNINFORMATIVE at
  the recompute-all operator** (channel-permutation handicap under frozen coefficients — a
  manufactured kill, caught); the referee's probe found a **semantics-stable operator (frozen
  schemas, growing per-schema statistics) that PASSES the bar post-hoc (z +2.99, all 5 years
  positive) → KT-D v2 with that operator is the pre-registered follow-up** at the next data
  refresh / third window. New store rule: enrichment operators must preserve schema-channel
  identity.
- **W4:** Joseph's interpretability battery on the learned representations; CrystalScore
  frontier (representation quality vs legibility); paper section.

## 7. Relation to the existing stack

- CRYSTAL-WORLD is the evolution of layer 3 (CrystalRL) in the 4-layer stack; layers 1-2 stay
  frozen. The HL loop becomes the semantic-store update mechanism (its ledger discipline is
  reused verbatim). The personalization product consumes L2/L3 states and keeps its own
  calibrated-contract machinery (energies never emit client probabilities, G2).
- Joseph's research question ("a causal, predictive, scalable account of what the agent believes")
  becomes: *the L2/L3 representations ARE the account* — named, crisp, probed. Ivan's question
  ("use the account to predict failures under shift") becomes KT-B/KT-D plus the detour tests.
- Governance unchanged: quarantined OOS stays locked; user objectives are never loop knobs;
  deployable claims only through the v12 gate.

## 8. Honest priors and the two ways this dies

1. **KT-B inverts** (no hierarchy dividend on markets): then T5-on-markets is falsified beyond our
   one regime data point, the theory file is amended, and the program's value contracts to the
   world-picture bookkeeping (still real, per amended T1 — but a much smaller claim).
2. **KT-D nulls** (enrichment ≤ frozen memory): then the "self-enriching" half of the vision fails
   on this substrate and CRYSTAL-WORLD reduces to a static representation layer + the existing
   HL loop. Given CL-1c, this is a live risk, not a formality.

Both deaths are informative and pre-registered. The program is built so that its failure modes
produce theory amendments, not silent drift — which is the project's definition of doing science.
