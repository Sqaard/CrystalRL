# External literature on controllable / interpretable RL & control — critique + creative hypotheses

**Plan Б of the CRYSTAL-1 controllability set.** Scope is **controllability** — steering, legibility, authority,
reversibility, composition, command certification — **not** alpha/returns. Anchored to
`reports/_crystal_ctrl/ANCHORS.md`; every transfer maps onto CRYSTAL-1's **K-simplex belief as the writable
command surface** (`src/crystal/belief_filter.py:26` `NeuralBayesFilter`; belief `b` set into obs via
`interpretability/certified_battery_v2.py:120-121` `obs_vec(b,t,iv,burst) → [2*b-1, …]`), tested on the
Series-G polygon (`src/series_g/regime_pomdp.py`) with the 5-gate battery
(`interpretability/certified_battery_v2.py`).

**Source discipline.** External literature is self-run from knowledge, each tagged **Tier-A** (named
author+year / named method) or **Tier-B** ([method class]). Evidence status on every transfer claim: **[E]**
Established / **[SbC]** Strong-but-Contextual / **[P]** Plausible / **[Spec]** Speculative. Where a local report
already owns a family I cross-link it rather than re-derive (chiefly `HL5_FINAL_IV_Command_Certification.md`,
`HL5_IV10_EPISTASIS_DERISK.md`, `_hl5_digest/IV_digest.md`).

**Settled facts I do not contradict** (ANCHORS §"Settled facts"): R6c > CRYSTAL-1 on controllability *today*;
regime is PRICED on daily/most-intraday (Glosten-Milgrom, VoI=0, `B4_REAL_INTRADAY_CLOSURE.md`); B2 retraction
of uniqueness-tracks-fidelity; B5 reward-shaping-for-legibility FALSIFIED → a **structural** mechanism is
required; the corner is real only on the polygon. Every hypothesis below is therefore a **controllability**
claim on the polygon/battery, never a returns claim on daily panels.

---

## 0. The one-paragraph orientation

The unique object CRYSTAL-1 owns that none of the external families own is a **named, low-dimensional,
normalized, generatively-grounded command surface**: the belief is a point on the **K-simplex**, each vertex is
a *named regime* recovered up to permutation by a learned HMM (`belief_filter.py:37-41` `mats()`), it is written
by a **scalar per named axis** (`2*b-1`), and the write is **envelope-scoped to visited states**. Almost every
external steering method pays a tax CRYSTAL-1 does not: it steers an **unnamed, high-dimensional, un-normalized**
latent (a 4096-d residual direction; R6c's own 64-d penultimate, `src/ppo/dirichlet_policy.py`), so it must
*discover* a direction, *name* it post-hoc, and *bound* its dose empirically. The literature's real gift to
CRYSTAL-1 is therefore **not another way to steer** — it is the **discipline** (dose windows, placebo arms,
causal-sufficiency gates, envelope protection, composition arithmetic) that the belief-write surface makes
*cheap to enforce* because the surface is already legible. The critique section's throughline: most external
"controllability" wins are **human-rate-subsidized** (they assume a slow attentive operator reads a natural-
language name) — the honestly-transferable core is what survives mechanization onto a scalar simplex write.

---

## 1. FAMILY: Belief-state / POMDP RL with an EXPLICIT belief

This is CRYSTAL-1's own family, so the transfer bar is highest here.

### 1.1 Belief-MDP / exact belief-state RL — **Tier-A** (Kaelbling, Littman & Cassandra 1998, *Planning and acting in POMDPs*) **[E]**
- **Idea.** Recast a POMDP as an MDP over the belief simplex; the belief is a sufficient statistic of history, so
  the optimal policy is a function of `b` alone.
- **Controllability buys.** The belief simplex is *the* interpretable state and *the* intervention surface — this
  is exactly CRYSTAL-1's thesis, and it is the theoretical warrant that **writing `b` is a complete command**
  (no hidden recurrent side-channel can matter if `b` is truly sufficient). It licenses the blueprint's WH1 law
  ("all memory routes through the K-named belief", `CRYSTAL_AGENT_BLUEPRINT.md` L1) as a *soundness* property,
  not a taste.
- **Documented failure mode.** Exact belief-MDP is intractable beyond tiny K/A_obs (the "curse of dimensionality
  ⊗ curse of history"); PBVI/point-based methods approximate by sampling reachable belief points — which is
  *precisely* the "envelope = visited states" restriction the battery already enforces
  (`certified_battery_v2.py:122`).
- **Transfer to CRYSTAL-1.** Direct and load-bearing: it says the K-simplex write is a **certified-complete**
  command iff belief-sufficiency holds. **Falsifiable gate:** the battery's "reactive ≫ autoregressive" gate is
  the empirical sufficiency test — if an autoregressive-on-own-past-actions policy matches the belief-reactive
  one, `b` is *not* sufficient and the write surface is incomplete. **[E]**

### 1.2 Predictive State Representations (PSRs) — **Tier-A** (Littman, Sutton & Singh 2001; Boots, Siddiqi & Gordon 2011) **[SbC]**
- **Idea.** Replace the latent belief with a vector of **predictions of future observable tests** — state is
  defined by what it predicts about observables, not by a hidden label.
- **Controllability buys.** A *grounded* command surface: every coordinate is an observable-anchored quantity, so
  a write has an operational meaning ("raise P(next obs = BURST)") that needs no post-hoc naming — the anti-
  hallucination property CRYSTAL-1 wants.
- **Failure mode.** Choosing the "core tests" is fragile; learned PSRs are less interpretable-by-vertex than an
  HMM's named states; spectral learning is sensitive to rank misspecification (the C\*≈K bend, `b2_multiseed.py`
  C4, is the CRYSTAL analog of picking the rank).
- **Transfer.** A PSR-style **read-out head** on the K-simplex would make each belief write **certifiable at C0
  by construction** (`IV_digest.md` C0 addressability): the emission matrix `E` (`belief_filter.py:39`) *already*
  maps each vertex to an observable-likelihood signature, so CRYSTAL-1 can express "what does forcing `b→vertex k`
  predict about the next observation" as a native contract. **Hypothesis H4 below operationalizes this.** **[SbC]**

### 1.3 DVRL / deep variational RL & particle-belief RL — **Tier-A** (Igl, Zintgraf et al. 2018, *Deep Variational RL for POMDPs*; Ma, Karkus et al. 2020, particle filter RL) **[SbC]**
- **Idea.** Carry an explicit *approximate* belief as a set of weighted particles or a variational posterior,
  updated by a learned filter, and feed it to the policy.
- **Controllability buys.** A **belief-quality signal for free** — particle-set entropy / ESS / posterior
  variance is exactly the "authority scales with belief QUALITY not point value" precondition the command
  certificate demands (`IV_digest.md` G-SENSOR/BELIEF-QUALITY-GATE; `HL5_FINAL_IV_Command_Certification.md` §4C).
- **Failure mode.** Particle collapse / posterior over-confidence gives a *falsely* high-authority signal — the
  QF72 "bad ADIRU drove protection to command harm" pattern, ported into RL.
- **Transfer.** CRYSTAL-1's filter emits a *categorical* posterior, so its **entropy `H(b)` is a native belief-
  quality meter** with no particle machinery — but the DVRL failure warns that a *sharp* `b` is not a *correct*
  `b`. **This motivates Hypothesis H2 (belief-quality-gated authority): demote the write when the filter's own
  held-out predictive likelihood degrades, not when `H(b)` is merely low.** **[SbC]**

---

## 2. FAMILY: Interpretable-by-design policies

### 2.1 Concept Bottleneck Models — **Tier-A** (Koh, Nguyen et al. 2020, *Concept Bottleneck Models*) **[E for the pattern; SbC as RL policy]**
- **Idea.** Force the network to predict human-named **concepts** as an intermediate layer, then predict the
  output *only* from concepts; you can **intervene** by editing a concept value at test time.
- **Controllability buys.** The canonical *test-time intervention* semantics — and CRYSTAL-1's belief IS a
  concept bottleneck where the concepts are the K regime-probabilities. The write `2*b-1` **is** a concept
  intervention; the battery's `interv_fidelity` (`certified_battery_v2.py:138`) **is** CBM's intervention-
  accuracy metric.
- **Documented failure mode — the load-bearing one.** **Concept leakage** (Mahinpei et al. 2021; Margeloiu et
  al. 2021): the bottleneck's *soft* concept activations smuggle extra task information past the named concepts,
  so intervening on the named concept does **not** fully control the output — the concept is legible but not
  *causally sufficient*. This is the same defect as "localization ≠ editability" (Hase 2023, cited in
  `IV_digest.md` C0→C1 hard gate).
- **Transfer.** CRYSTAL-1 is *structurally* leak-resistant because the belief is **normalized and low-K** (a
  point on Δ^{K-1}, not a free soft vector), and the blueprint FORBIDs any other recurrence
  (`CRYSTAL_AGENT_BLUEPRINT.md` FORBID #2). **But the leak test is exactly the "reactive ≫ autoregressive" +
  "HC-1 ablation must hurt" battery gates** — if ablating the belief channel barely hurts, the policy leaked
  around it. **Hypothesis H1 (concept-leakage stress test) makes this a first-class certificate field.** **[E]**

### 2.2 Prototype / ProtoPNet-style policies — **Tier-A** (Chen, Li et al. 2019, *This Looks Like That*) **[P as RL policy]**
- **Idea.** Classify by similarity to learned **prototypes** ("this looks like that stored case"); the
  explanation is a distance to a named exemplar.
- **Controllability buys.** A *case-based* command: "behave as in prototype-P". R6c already has the analog —
  the K-means codebook centroids (`reports/firewall_upgrade/r6c_code_layer/r6c_codebook.npz`), steered by forcing
  the penultimate toward a centroid (ANCHORS §R6c+).
- **Failure mode.** Prototypes drift off-manifold and become uninterpretable; the similarity metric is itself
  unnamed; "this looks like that" can be a **spurious** match (the dormant-pathway illusion, `IV_digest.md`
  G-DORMANT-PATHWAY).
- **Transfer.** CRYSTAL-1's **vertices are the prototypes**, and they are *generatively* grounded (each vertex
  has a real emission signature `E[:,·]`), so the ProtoPNet drift failure is bounded by the filter's likelihood.
  The transfer is: **a belief-write is a "move toward prototype-vertex-k" command with a built-in on-manifold
  check** (does the forced `b` remain a plausible filter state given recent observations?). **[P]**

### 2.3 Neurosymbolic / program-synthesis policies — **Tier-A** (Verma, Murali et al. 2018, *PIRL: Programmatically Interpretable RL*; Bastani, Pu & Solar-Lezama 2018, VIPER) **[E for distillation; SbC as native policy]**
- **Idea.** Represent the policy as a **program** (PIRL) or distill a neural policy into a **decision tree**
  (VIPER, via DAgger with a Q-weighted loss) that is verifiable and human-readable.
- **Controllability buys.** *Verifiability* — you can model-check a tree/program for safety envelopes, prove
  reachability/no-deadlock (the arbitration selector requirement, `IV_digest.md` G-PROVE-NO-DEADLOCK). This is
  the strongest external anchor for **command certification by proof**, not just by test.
- **Failure mode.** Distillation fidelity gap: the tree is a *lossy* shadow; VIPER's guarantee is relative to the
  teacher, and a high-fidelity tree can still be huge (parsimony tax). Programs are brittle to distribution
  shift.
- **Transfer.** CRYSTAL-1 **already** distills its policy into an interpretable **story tree** over `[b, iv, t]`
  and certifies the *policy-follows-story* fidelity (`certified_battery_v2.py:136` `story.predict([[b,iv,t]])`,
  gate `interv_fidelity ≥ 0.67`). So VIPER is *native*, not aspirational. The novel transfer is the *converse*
  direction — **use the tree as the command language**: a leaf of the story tree is a certified region, and a
  command is "move `b` across the leaf boundary". **Hypothesis H3 (leaf-boundary command grammar) below.** **[E]**

### 2.4 Soft Decision Trees as a policy head — **Tier-A** (Frosst & Hinton 2017, *Distilling a Neural Network Into a Soft Decision Tree*) **[P]**
- **Idea.** A differentiable tree whose internal nodes are learned linear gates; trainable end-to-end, reads as a
  hierarchy of soft decisions.
- **Controllability buys.** A *differentiable* interpretable head — you keep gradient training and get a tree you
  can read and edit node-by-node.
- **Failure mode.** Soft gates blur the "which leaf" attribution; the tree is interpretable *structurally* but a
  given decision is a *mixture* over leaves (the concept-leakage cousin).
- **Transfer.** CRYSTAL-1's manager head could be a soft tree over the K-simplex, making each split a **named
  belief threshold** = a certifiable `command_threshold` (`certified_battery_v2.py:154` already *measures* such a
  threshold). Low priority: adds a head where the story-tree distillation already delivers the readable object.
  **[P]**

---

## 3. FAMILY: Steering / representation editing (the RepE family) — **locally owned by HL5-IV; cross-link, don't re-derive**

> The whole activation-steering / RepE / causal-mediation literature is already mined unit-by-unit in
> `_hl5_digest/IV_digest.md` (units 01 repe_steering, 02 causal_mediation) and synthesized in
> `HL5_FINAL_IV_Command_Certification.md`. I summarize the external anchors and add only the transfer that is
> *specific to a K-simplex write* and not already in IV.

### 3.1 Activation steering / RepE / ITI — **Tier-A** (Zou, Wang et al. 2023, *Representation Engineering*; Turner et al. 2023, ActAdd; Li, Patel et al. 2023, *Inference-Time Intervention*; Templeton et al. 2024, Golden-Gate Claude) **[E]**
- **Idea.** Add a scalar-dosed direction to activations at inference: `h ← h + α·v` (the **BIAS/ADD** verb,
  `IV_digest.md` §2A — "the workhorse and closest analog to CrystalRL's belief-write").
- **Controllability buys.** Inference-time, stateless, **mechanism-reversible** (remove the hook → bitwise-
  identical forward pass, `IV_digest.md` §2C) — the cleanest reversibility asset in the whole survey.
- **Failure mode.** Golden-Gate over-dose (CLAMP past natural range captures the persona); mid-stack sweet spot
  and per-class heterogeneity (some classes anti-steer); dose window `[MEC,MTD]` is empirical and drifts with
  retrain (story decay 0.94→0.67).
- **Transfer specific to CRYSTAL-1 (the added value).** RepE steers an **un-normalized 4096-d direction** and must
  *discover, name, and dose-bound* it. CRYSTAL-1's write is **already** named (vertex k), **already** normalized
  (Δ^{K-1}), and **already** dose-parameterized (the `2*b-1` scalar per axis). So the RepE *failure taxonomy*
  transfers wholesale but the RepE *discovery cost* is zero. **Consequence:** the belief-write is the rare case
  where the C1 causal-sufficiency gate (`IV_digest.md` C1) can be *proved on the polygon* rather than probed —
  because the filter defines the counterfactual (Hypothesis H4). **Cross-link:** `IV_digest.md` §2A BIAS row;
  the R6c 64-d centroid-forcing demo (`reports/firewall_upgrade/r6c_code_control_demo.py`) is the R6c analog.
  **[E]**

### 3.2 Causal mediation / activation & path patching / DAS — **Tier-A** (Vig et al. 2020; Meng et al. 2022, ROME; Geiger, Wu et al. 2023, *Distributed Alignment Search*) **[E]**
- **Idea.** Interchange-intervention: patch an activation from a counterfactual run and measure whether the output
  moves as the causal story predicts (**IIA**); DAS learns the low-rank *subspace* that is the true causal unit.
- **Controllability buys.** The **causal-sufficiency** certificate itself — IIA is the C1 metric
  (`IV_digest.md` C1; the project pin IIA=0.744 vs chance 0.33).
- **Failure mode.** Localization ≠ editability (Hase 2023); DAS is *an optimizer over subspaces* → over-fits a
  direction unless evaluated on held-out counterfactual families → needs a **search ledger = deflated-IIA**
  (`IV_digest.md` §8 misuse: certificate laundering).
- **Transfer.** On CRYSTAL-1 the "subspace" is not searched — it is the **named simplex axis**, so DAS's over-
  fitting risk is structurally removed. The *counterfactual bank* DAS needs is **generated by the filter** (roll
  the HMM forward from a forced vertex), removing the "corruption choice flips the conclusion" fragility
  (Zhang & Nanda, `IV_digest.md` §2B reference-distribution axis). **This is the single cleanest external→
  CRYSTAL transfer and it grounds Hypothesis H4.** **Cross-link:** `IV_digest.md` unit 02. **[E]**

---

## 4. FAMILY: World-model control & command envelopes

### 4.1 Latent world models / Dreamer — **Tier-A** (Hafner, Lillicrap et al. 2020-2023, Dreamer v1-v3) **[SbC]**
- **Idea.** Learn a latent dynamics model; train the policy *in imagination*; the latent is a compact predictive
  state.
- **Controllability buys.** A **forward model to predict a command's effect before executing it** — the C1/C4
  "predict the effect before the write" requirement (`HL5_FINAL_IV_Command_Certification.md` §0) becomes a
  *rollout in the world model*.
- **Failure mode.** The Dreamer latent is **unnamed and entangled**; imagination compounds model error; steering
  the latent has no vertex semantics.
- **Transfer.** CRYSTAL-1's filter **is** a (small, named, generative) world model — `T` and `E`
  (`belief_filter.py:37-41`) *are* the latent dynamics, but **discrete, named, and recoverable to ground truth on
  the polygon**. So "predict a belief-write's consequence by rolling the filter forward" is a native, *cheap,
  certifiable* operation where Dreamer's is expensive and opaque. **This is the mechanism behind H4's
  pre-execution effect prediction.** **[SbC]**

### 4.2 Model-Predictive Control on learned latents / reference governors — **Tier-B** [MPC-on-latents]; reference-governor **Tier-A** (Gilbert & Kolmanovsky 2002) **[SbC]**
- **Idea.** MPC optimizes a short-horizon action plan against the learned model subject to constraints; a
  **reference governor** sits *upstream* of a controller and minimally modifies the *command* so constraints are
  never violated.
- **Controllability buys.** The reference-governor pattern is the exact shape of a **certified command envelope**:
  it takes a desired belief-write and *projects it onto the feasible (visited/safe) set* before it reaches the
  policy — a constructive implementation of C3 flight-envelope protection (`IV_digest.md` C3, §4C).
- **Failure mode.** Needs an accurate constraint model; conservative governors over-clip (lost authority); MPC
  horizon error.
- **Transfer.** Wrap the belief-write API in a **belief reference-governor**: given a requested `b*`, return the
  closest `b` that is (i) on the visited-state envelope and (ii) a plausible filter posterior. This is a
  *structural* command-safety mechanism — exactly what B5 concluded is required after reward-shaping-for-
  legibility failed (`B5` FALSIFIED → structural mechanism, ANCHORS). **Grounds Hypothesis H5.** **[SbC]**

### 4.3 Control-Barrier Functions as command envelopes — **Tier-A** (Ames, Coogan et al. 2019, *Control Barrier Functions: Theory and Applications*) **[SbC]**
- **Idea.** A CBF certifies a **forward-invariant safe set**: any command is minimally modified so the system
  provably never leaves the safe set (a QP filter on the command).
- **Controllability buys.** *Provable* envelope invariance with a **minimal, reversible** intervention — the
  strongest formal version of "the write auto-refuses outside its envelope" (`IV_digest.md` C3 auto-refuse).
- **Failure mode.** Requires a known safe-set and relative-degree assumptions; the barrier can be conservative;
  composing multiple CBFs is non-trivial (the composition problem again).
- **Transfer.** Define a **belief-simplex barrier**: the safe set = {`b` reachable by the filter within envelope
  ∧ belief-quality ≥ threshold}; the barrier-QP projects any requested write onto it. This upgrades the current
  *soft* envelope discipline (probes live on `iv≤2`, `certified_battery_v2.py:122`) into a *hard, minimal-edit*
  guarantee. **Composes with H5 (reference governor) as the enforcement layer.** **[SbC]**

---

## 5. FAMILY: Skill / option discovery with NAMED skills

### 5.1 Options / HRL with symbolic subgoals — **Tier-A** (Sutton, Precup & Singh 1999, *Options framework*; Kulkarni et al. 2016, h-DQN with symbolic subgoals) **[E]**
- **Idea.** Temporally-extended **options** (initiation set, policy, termination) as named macro-actions; a
  manager selects options, a worker executes.
- **Controllability buys.** A **named macro-command vocabulary** with explicit *initiation sets* (= the option's
  envelope) — the option framework is the cleanest external match to CRYSTAL-1's discrete native modes
  {enter/hold/reduce/exit | provide/abstain/aggress} (`CRYSTAL_AGENT_BLUEPRINT.md` L2).
- **Failure mode.** Learned options collapse to degenerate one-step or single-option solutions; termination is
  hard to certify; the "option" name is often post-hoc.
- **Transfer.** CRYSTAL-1's manager→worker already *is* an options hierarchy with **belief as the initiation
  context**. The option's *initiation set* is exactly the C3 envelope; the transfer is to **register each native
  mode as a certified option with a declared initiation set on the simplex**, so "command = force belief into
  option-k's initiation region" is well-typed. **[E]**

### 5.2 DIAYN / unsupervised skill discovery — **Tier-A** (Eysenbach, Gupta et al. 2019, *Diversity Is All You Need*) **[P]**
- **Idea.** Learn a set of **distinguishable** skills by maximizing mutual information between a latent skill code
  `z` and visited states, *without rewards*.
- **Controllability buys.** A **discrete skill code you can set** — `z` is a control token; and the MI objective
  guarantees skills are *behaviorally distinguishable* (the "uniqueness" property).
- **Failure mode — directly relevant.** DIAYN skills are **unnamed** and often **not aligned to task-meaningful
  regimes**; MI can be maximized by trivial state-coverage. Crucially, **CRYSTAL-1's B2 already RETRACTED
  "uniqueness-tracks-fidelity"** (`b2_multiseed.py` C3, ANCHORS) — so DIAYN's core promise (distinguishable ⇒
  useful/faithful) is a claim the project has *falsified locally*. Cite DIAYN as a **cautionary Tier-A anchor**,
  not a recipe: a steerable, distinguishable skill code is **not** thereby a faithful or legible one.
- **Transfer.** Negative/guardrail: any "grow K by discovering a new skill" proposal
  (`CRYSTAL_AGENT_BLUEPRINT.md`: "grow K on C\*-bend") must pass a **fidelity** gate, not a **distinguishability**
  gate, precisely because B2 severed the two. **[P — as a guardrail, not a build]**

---

## 6. FAMILY: Controllable-generation analogs mapped onto a policy

### 6.1 Classifier(-free) guidance — **Tier-A** (Dhariwal & Nichol 2021, classifier guidance; Ho & Salimans 2022, classifier-free guidance) **[P]**
- **Idea.** Steer a generative process at inference by a **guidance scale** toward a class/attribute; a single
  scalar `w` trades fidelity for control strength.
- **Controllability buys.** A **continuous, monotone control knob with a documented over-dose regime** — the
  guidance scale `w` behaves exactly like the belief-write dose `α`: too small = no control, too large =
  incoherence (the "monotone-then-collapse" law, `IV_digest.md` §2B).
- **Failure mode.** High guidance → mode collapse / artifacts (the CFG analog of Golden-Gate over-dose).
- **Transfer.** The belief-write **is** a guidance signal on the policy's action distribution; the battery's
  `P_provide_vs_forced_belief` dose-response curve (`certified_battery_v2.py:155`) is *literally* a guidance-
  scale sweep. The transfer is conceptual confirmation: **CRYSTAL-1's dose window is a classifier-free-guidance
  scale over a named regime**, and its collapse regime is expected, not a bug. **[P]**

### 6.2 Control tokens / prefix & prompt control — **Tier-A** (Keskar et al. 2019, CTRL; Li & Liang 2021, Prefix-Tuning) **[P]**
- **Idea.** Prepend a **control code** that conditions generation on a named attribute.
- **Controllability buys.** A discrete, human-named command channel that is *part of the input*, not a
  post-hoc edit — which is **exactly** how CRYSTAL-1 writes belief: the command lives in the obs vector
  (`obs_vec`, `certified_battery_v2.py:120`), not in a hooked activation.
- **Failure mode.** Control tokens can be *ignored* by a policy that finds the attribute elsewhere in context
  (the input-side concept-leakage failure).
- **Transfer.** This is the **cleanest architectural analogy to CRYSTAL-1's write mechanism** and it names the
  right failure to test: **can the policy ignore the written belief because it re-derives regime from `last_obs`
  in the obs vector?** The `obs_vec` includes `burst` (`certified_battery_v2.py:121`), an observable the policy
  could use to bypass the written `b`. **This is the exact attack Hypothesis H1 tests.** **[P]**

---

## 7. CRITIQUE — which of these are human-rate-subsidized, which are honestly transferable

The load-bearing distinction from the IV survey (`IV_digest.md` §2F, converged 10/10): the surveyed fields
**silently assume a slow, attentive human operator**. A family is *honestly transferable* to a machine-operated
CRYSTAL-1 only to the extent its control survives mechanization onto a scalar simplex write with no human in the
loop.

**Human-rate-subsidized (control is real but leans on a human reading a name):**
- **Concept Bottleneck Models (2.1)** — the "intervene on the concept" story assumes a human *chooses* the
  concept edit and *reads* the result; the concept-leakage failure means the machine-automated version needs the
  causal-sufficiency gate CBM papers rarely enforce. Subsidized.
- **ProtoPNet (2.2)** — "this looks like that" is an *explanation for a human*; as a machine command it collapses
  to centroid-forcing (which R6c already does opaquely). Subsidized at the explanation layer.
- **Activation steering / RepE (3.1)** — the whole value proposition ("a natural-language-named direction") is a
  *human legibility* claim; the *machine* part (scalar dose on a direction) is honestly transferable, but the
  **naming** is the subsidy. CRYSTAL-1 repays it structurally by naming via the generative filter, not by human
  annotation.
- **Control tokens (6.2)** — a control code is human-authored and human-meaningful; the machine can ignore it.
  Subsidized at authoring.

**Honestly transferable (control survives full mechanization):**
- **Belief-MDP sufficiency (1.1)** and **causal mediation / IIA (3.2)** — these are *properties measurable by a
  machine* (sufficiency test, interchange accuracy). No human-rate subsidy. **Highest-value imports.**
- **Reference governors (4.2) and CBFs (4.3)** — *pure runtime* command-projection with formal guarantees; the
  canonical anti-human-rate machinery (the runtime enforces the envelope, `IV_digest.md` §2F point 1). Honestly
  transferable and currently **missing** from CRYSTAL-1 (the envelope is a *soft* probe restriction, not a hard
  projector).
- **VIPER / program distillation (2.3)** — the *verification* (model-check the tree) is machine-run; the
  fidelity gap is measured, not asserted. Honestly transferable; **already native** via the story tree.
- **DVRL belief-quality (1.3)** — posterior entropy/ESS is a machine signal; honestly transferable as the
  authority gate.
- **DIAYN (5.2)** — honestly transferable *as a warning* (B2 already falsified its promise), not as a build.

**The synthesis critique.** The families split cleanly: **naming/legibility** methods (CBM, ProtoP, RepE,
control tokens) are human-rate-subsidized and CRYSTAL-1's contribution is that the **K-simplex + generative
filter repays the naming subsidy structurally** (each vertex is named by ground-truth recovery, not annotation);
**enforcement/certification** methods (belief-MDP sufficiency, IIA, reference governors, CBFs, VIPER) are
honestly transferable and are exactly where CRYSTAL-1 is currently **weakest** — it has a legible surface but the
envelope is soft and the command is not yet projected/certified at runtime. **The literature's real lesson: stop
importing more ways to steer (CRYSTAL-1's surface is already the best-named in the survey) and import the runtime
enforcement + causal-sufficiency machinery it lacks.**

---

## 8. FIVE+ CREATIVE, FALSIFIABLE HYPOTHESES for CRYSTAL-1 controllability

Each: claim → external anchor → **falsifiable test on the polygon/battery** → null → evidence rung. All run on
`src/series_g/regime_pomdp.py` + `interpretability/certified_battery_v2.py` with no retrain unless noted.
None asserts alpha; each is a controllability property.

### H1 — Concept-leakage bypass test: the belief write is only a command if the policy *cannot* re-derive regime from the observable channel
- **Claim.** The belief write `2*b-1` is a *sufficient* command **iff** the policy does not bypass it using the
  observable `burst`/`last_obs` fields in `obs_vec` (`certified_battery_v2.py:121`). If it bypasses, the surface
  is legible-but-not-causal (the CBM concept-leakage failure, 2.1; control-token-ignored failure, 6.2).
- **Anchor.** Koh 2020 (CBM leakage) + Keskar 2019 (control-token ignore) + Hase 2023 (localization ≠ editability,
  `IV_digest.md` C0→C1).
- **Test.** Two arms on the frozen corner PPO: (A) intervene on `b` with `last_obs`/`burst` held *consistent*
  with the forced `b` (the current battery, `interv_fidelity`); (B) intervene on `b` with `last_obs`/`burst`
  held to *contradict* `b`. Report `interv_fidelity_B` and the per-class compliance delta. **Also**: ablate the
  `burst` obs coordinate and re-measure — HC-1-style (`certified_battery_v2.py` HC-1 gate).
- **Null.** `interv_fidelity_B ≈ interv_fidelity_A` and ablating `burst` does not raise compliance ⇒ **no leakage,
  the write is causally sufficient** (a *good* null that certifies the surface). Leakage ⇒
  `interv_fidelity_B ≪ A` and the policy tracks the contradicting observable.
- **Rung.** [E] — direct port of an Established external failure onto existing battery machinery; cheap, no
  retrain.

### H2 — Belief-quality-gated authority: demote the write on filter mis-calibration, not on low `H(b)`
- **Claim.** The correct authority gate is the filter's **held-out predictive likelihood** (calibration), not the
  belief's sharpness. A sharp-but-wrong `b` (DVRL particle-collapse analog, 1.3; QF72 single-source failure,
  `IV_digest.md` G-SENSOR/BELIEF-QUALITY) should carry *less* command authority than a diffuse-but-calibrated
  `b`.
- **Anchor.** Igl/DVRL 2018 (belief quality) + `HL5_FINAL_IV_Command_Certification.md` §4C (authority scales with
  belief QUALITY not value).
- **Test.** Construct polygon states where (i) `H(b)` is low but the filter's 1-step predictive NLL is high
  (feed observations that violate the learned `E`), and (ii) `H(b)` is high but NLL is low. Measure the
  battery's `interv_fidelity` and `monotone_frac` in each. If write-compliance is *higher* in the sharp-but-
  miscalibrated case, the current surface keys authority on **value/sharpness** (the failure); a quality gate
  keyed on **NLL** should invert that.
- **Null.** Compliance already tracks NLL, not `H(b)` ⇒ the surface is already quality-gated (good). Otherwise ⇒
  add `min_predictive_ll` to the command certificate's `belief_preconditions`
  (`HL5_FINAL_IV_Command_Certification.md` §6 schema).
- **Rung.** [SbC] — constructs adversarial states; uses the existing filter + battery; the *direction* is
  Established, the CRYSTAL measurement is new.

### H3 — Leaf-boundary command grammar: the story-tree leaves ARE the certified command alphabet
- **Claim.** The VIPER-distilled story tree over `[b, iv, t]` (`certified_battery_v2.py:136`) partitions the
  simplex×inventory×time space into leaves; a *certifiable* command is exactly "move `b` across a leaf boundary
  to change the prescribed native mode", and the crossing point is the measured `command_threshold`
  (`certified_battery_v2.py:154`).
- **Anchor.** Bastani/VIPER 2018 + Verma/PIRL 2018 (program-as-policy) + Sutton options initiation sets (5.1).
- **Test.** Enumerate the story-tree leaf boundaries in `b`; for each boundary, sweep `b` across it and confirm
  the policy's native mode flips at the *predicted* threshold with a monotone dose-response (`monotone_frac`).
  A boundary where the policy does **not** flip is an **uncertifiable command** (the tree over-claims control
  there). Report the fraction of leaf-boundaries that are certifiable commands = a new battery scalar
  **`command_alphabet_coverage`**.
- **Null.** All leaf boundaries flip the policy at the predicted `b` ⇒ the tree is a *complete, faithful* command
  language (the strong result). Partial coverage localizes exactly where the legible story is not a command.
- **Rung.** [E→SbC] — the tree, thresholds, and monotonicity are already computed; the hypothesis is a new
  *read-out* (coverage scalar) over existing outputs, no retrain.

### H4 — Filter-grounded C1 proof: on the polygon, causal sufficiency of a belief-write can be *proved*, not merely probed
- **Claim.** Because the filter defines the generative model (`T,E`, `belief_filter.py:37-41`) and the polygon's
  ground truth is known, the counterfactual for an interchange intervention (DAS/IIA, 3.2) is **computable**, not
  sampled — so the C1 causal-sufficiency certificate (`IV_digest.md` C1) can be issued with a *closed-form*
  expected-action-shift, and the empirical `interv_fidelity` must match it.
- **Anchor.** Geiger/DAS 2023 + Meng/ROME 2022 (interchange intervention) + Kaelbling 1998 belief sufficiency
  (1.1). Removes the "corruption choice flips the conclusion" fragility (Zhang & Nanda, `IV_digest.md` §2B).
- **Test.** For each vertex-forcing write, roll the *filter's own* generative model forward from the forced `b`
  to get the model-predicted optimal action distribution; compare to the policy's actual action under the write.
  Define **`filter_policy_agreement`** = fraction where they match. A high value = the belief-write is provably
  causal (C1 by construction). A low value = the policy and its own belief model disagree ⇒ the write moves the
  *obs* but the policy ignores the *belief semantics* (the deepest possible leakage).
- **Null.** `filter_policy_agreement ≈ interv_fidelity ≈` the analytic Bayes-optimal action ⇒ the K-simplex is a
  **certified-complete command surface on the polygon** — the strongest controllability result CRYSTAL-1 could
  claim. Disagreement localizes the gap.
- **Rung.** [SbC] — needs a small script rolling the filter forward (the filter runtime exists,
  `belief_filter.py:59-63` `numpy_params`); highest scientific value because it converts a *probed* certificate
  into a *proved* one on the one substrate where ground truth exists (the polygon; ANCHORS: corner is real only
  here).

### H5 — Belief reference-governor: a *structural* command-safety layer (the B5-mandated mechanism)
- **Claim.** Wrapping the belief-write API in a **reference governor / CBF-QP** (4.2, 4.3) that projects any
  requested `b*` onto {visited-envelope ∧ calibrated-filter-state} gives the *structural* command-safety
  mechanism B5 concluded is required after reward-shaping-for-legibility was FALSIFIED (ANCHORS; `b5_crystallize.py`).
- **Anchor.** Gilbert & Kolmanovsky 2002 (reference governor) + Ames 2019 (CBF) + `IV_digest.md` C3 flight-
  envelope protection.
- **Test.** Implement `govern(b*) → b` = nearest point to `b*` inside the safe set (visited `iv`/`t` occupancy +
  `predictive_NLL ≤ τ`). Compare battery outcomes with vs without the governor on **off-envelope** probes (the
  `iv>2` set currently footnoted, `certified_battery_v2.py:131`): the governor must (i) leave on-envelope
  compliance **byte-identical** (small-signal invariance, `IV_digest.md` §2C — a no-op when no limit is active)
  and (ii) convert off-envelope collapse/sign-flip into a bounded, annunciated demotion (emit a `GUARANTEE_DELTA`,
  `HL5_FINAL_IV_Command_Certification.md` §4C).
- **Null.** The governor changes on-envelope behavior (violates small-signal invariance) ⇒ it is mis-specified,
  not a structural safety layer. Success = on-envelope no-op + off-envelope bounded ⇒ the first *structural*
  (not reward-shaped) command-safety mechanism, directly answering B5.
- **Rung.** [SbC] — new runtime wrapper; strongest because it is the *constructive* answer to a project
  falsification, and it is the honestly-transferable machinery the critique flagged as missing.

### H6 (bonus) — Epistasis on a ≥2-belief-dim simplex: the open [P] the IV survey explicitly owes
- **Claim.** Grow K≥3 so the simplex is ≥2-dimensional, then paired belief-writes on two *different* named axes
  can exhibit **sign epistasis** (effect reversal, `IV_digest.md` §5 class 3) — the exact test the epistasis unit
  says is still owed on a "≥2-belief-dim surface (CRYSTAL-1 K-simplex or R6c latent)"
  (`HL5_FINAL_IV_Command_Certification.md` §5B empirical-status note; `HL5_IV10_EPISTASIS_DERISK.md`).
- **Anchor.** Fisher 1918 (interaction variance) + Kuzmin trigenic + the local `iv10_epistasis_pairedwrite.py`
  (which measured non-additivity but **not** sign-reversal on the 1-belief-dim polygon).
- **Test.** On a K=3 filter, run the 2-factor paired-write factorial (write vertex-A axis × write vertex-B axis)
  on `P(PROVIDE)`; measure `sign(ε_ij)` stability across regimes and on the frozen held-out corner. Sign-reversal
  observed ⇒ the **non-commutative `D_total` / forbidden-pair machinery leaves [P] for [SbC]**; sign-stable ⇒ the
  co-activation-cap machinery can stay a blunt structural limiter (the current safe default).
- **Null.** Sign-stable and cross-term < single-lever curvature (as on the 1-D polygon) ⇒ composition remains
  additively-boundable; the forbidden-pair registry stays [P].
- **Rung.** [P] — needs a K=3 retrain (T1 cost) + the existing paired-write harness; it is the *named open
  problem* the IV artifact flags as its single weakest part, so closing it has the highest de-risking value for
  the whole command-certification track.

---

## 9. SELF-GRADE + single weakest part

**Self-grade (controllability focus, per the brief):** **B+/A−.** Strengths: every family is Tier-A/B tagged and
carries a documented failure mode and an *evidence-status'd* transfer that maps onto the real K-simplex write
surface (`belief_filter.py`, `certified_battery_v2.py:120`) rather than a generic latent; the critique cleanly
separates human-rate-subsidized *naming* families from honestly-transferable *enforcement* families and lands the
non-obvious synthesis (CRYSTAL-1 needs *enforcement*, not *more steering*); the six hypotheses are falsifiable on
existing tooling, each with a stated null, and H4/H5/H6 attack the exact open gaps the local IV artifact names
(proved-vs-probed C1; the B5-mandated structural mechanism; the owed ≥2-belief-dim sign-epistasis run). All
external citations are real named works held from knowledge and tagged; no local file ref is invented (all cite
paths read this session).

**Single weakest part:** **H4's "proof, not probe" claim leans on the filter being a *faithful* generative model
of the polygon, and I did not re-verify the filter's param-recovery selftest this session** — I cite it from the
module docstring (`belief_filter.py:15-17`) and the blueprint, not from a fresh run. If the trained filter's `T,E`
have drifted from ground truth (the story-decay phenomenon, `IV_digest.md` §9, applies to the *filter* too), then
the "closed-form counterfactual" in H4 is only as good as a *stale* model, and `filter_policy_agreement` would
conflate policy-leakage with filter-miscalibration — the very confound H2 raises. H4 and H2 should therefore be
run as a *pair* (calibrate the filter first, then test sufficiency), and until the filter's held-out likelihood
is re-confirmed, H4's rung is honestly [SbC], not [E]. Secondarily, the external-lit failure modes are Established
in their home domains but their *transfer* to a scalar simplex write is mostly [SbC]/[P] — the survey imports
discipline, and only H1 and H3 (which reuse already-computed battery outputs) are genuinely [E] on CRYSTAL-1
today.
