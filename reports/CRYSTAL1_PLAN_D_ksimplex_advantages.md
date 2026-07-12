# K-simplex belief + CRYSTAL-1 brain structure — advantages & NEW mechanisms unreachable by R6c

**Plan Д — the generative core of the CRYSTAL-1 controllability set.** Scope is CONTROLLABILITY only
(steering, legibility, authority, reversibility, composition, command certification) — not alpha. Settled facts
from `reports/_crystal_ctrl/ANCHORS.md` are taken as given: R6c > CRYSTAL-1 on controllability *today* (near-term
substrate stays HL-over-R6c); this document is the design case for the *born-legible successor's* control surface,
not a claim it is deployable now.

**Grounding.** L1 filter = `src/crystal/belief_filter.py`; battery = `interpretability/certified_battery_v2.py`;
architecture axioms = `reports/CRYSTAL_AGENT_BLUEPRINT.md`. R6c control surface = `src/ppo/w1_budget_trader_policy.py`
+ `src/ppo/dirichlet_policy.py` (net `pi:[128,64]` → 64-d penultimate → `action_net`), its steering demo
`reports/firewall_upgrade/r6c_code_control_demo.py`, and its OOD machinery `src/evaluation/firewall.py`. Writ ladder
= `reports/HL5_FINAL_IV_Command_Certification.md` (C0–C6). Evidence tags: **[E]** Established, **[SbC]**
Strong-but-Contextual, **[P]** Plausible, **[Spec]** Speculative. External literature tagged Tier-A (named
method/author) or Tier-B ([class]).

---

## 0. The architectural fork in one paragraph

R6c's memory and "concept" surface is a **64-d unnamed real vector** — `latent64 = policy_net.5.ReLU`, the
penultimate that feeds `action_net` (`r6c_code_control_demo.py:3`). It has **no coordinate semantics**, **no
normalization to any manifold**, and it is **not the only place state lives** (the 64-d penultimate is one layer of
a multi-layer feed-forward stack, plus a parallel value tower — forcing it leaves the earlier layers and the value
tower untouched). To steer it you must (a) build a *post-hoc* K-means codebook over natural latents to invent
pseudo-coordinates (`r6c_codebook.npz`), then (b) **force** the vector toward a centroid by linear interpolation
`natural + α·(centroid − natural)`, α∈[0,1] — which promptly leaves the natural manifold, which is exactly why
`safe_alpha`/`OODGate` (`firewall.py:73–110`) exist to shrink α back toward 0. CRYSTAL-1 replaces that vector with
a **K-simplex belief `b ∈ Δ^{K−1}`** produced by a small named generative model (`NeuralBayesFilter`,
`belief_filter.py:26`): every coordinate is a **named regime posterior**, the vector is **a probability
distribution by construction**, and — the load-bearing axiom (WH1) — it is the **SOLE memory channel**
(`CRYSTAL_AGENT_BLUEPRINT.md:68–74,120`). Every advantage below is a consequence of those three properties:
**named, simplex-constrained, sole-channel.**

---

## 1. NAMED belief write vs latent forcing — the C0 read-certificate exists by construction

**Mechanism.** A CRYSTAL-1 command addresses a *named coordinate*: "raise P(TOXIC)". The belief is set directly in
the obs — `obs_vec(b,…) → 2*b−1` (`certified_battery_v2.py:120`, mirrored in `LearnedBeliefEnv`,
`crystal1_b1.py:49–51`) — and each coordinate `k` has a fixed, human-declared regime meaning
(BENIGN/TOXIC, QUIET/BURST from `regime_pomdp.py`).

**Why the K-simplex/brain enables it.** The filter is a *named generative model* whose emission and transition
rows are per-state (`mats()`, `belief_filter.py:37–41`); after training its selftest recovers the true `T`/`E` up
to a state permutation and resolves the permutation by emission signature (`align_to_truth`,
`belief_filter.py:84–93`). So coordinate `k` is not a free label — it is *anchored to a measured world-model
signature*. That is a **C0 read certificate in the sense of the writ ladder** (`HL5_FINAL_IV_Command_Certification.md`
C0: "target site resolves to a codebook id … with probe AUC above floor AND control-task selectivity"): the
address is intrinsic, not fitted after the fact.

**Why R6c's 64-d latent cannot.** R6c has no per-coordinate meaning. Its "codes" are K-means centroids fit
*post-hoc* over natural latents (`r6c_code_control_demo.py:61–65`); the map from code→behavior is *discovered by
running the policy*, and the well-known caveat applies — the layer where a fact **localizes** is not where an
**edit** lands (Tier-A: Hase et al. 2023, cited in the ladder's hard C0→C1 gate). R6c's C0 is therefore a *fitted
correlational* certificate that must be re-earned by a causal patch; CRYSTAL-1's C0 is *structural*.

**NEW control primitive.** `SET_BELIEF(regime=TOXIC, p=0.8)` — a command whose target is a public name, not an
opaque cluster id. It is a legitimate `SET`/`BIAS` verb in the five-verb grammar
(`HL5_FINAL_IV_Command_Certification.md §1`) with a *self-documenting* target_site.

**Acceptance test.** C0 read-certificate: `belief_filter._selftest` gates param-recovery `max|ΔT|,max|ΔE| ≤ 0.10`
and belief-MAE ≤ 0.05 vs the analytic filter (`belief_filter.py:147–150`) — i.e. the named axis provably tracks
the true regime. Battery axis: HC-1 ablation must hurt (below). Status: **[E]** — recovery passes on the polygon;
on real Dow streams L1 recovered p_stay [0.982, 0.954] with the correct filtering signature
(`CRYSTAL1_B3_B4_B5_RESULTS.md`), so the naming survives contact with real data (**[SbC]** off-polygon).

---

## 2. The belief bottleneck as the ONLY memory path ⇒ control COMPLETENESS

**Mechanism.** WH1 law: **all** memory routes through the K-named belief; unstructured recurrence is *forbidden*
as the complexity source (`CRYSTAL_AGENT_BLUEPRINT.md:68–74`, FORBID #2 line 120). Therefore if you own the belief
write, you own **everything the policy can remember**. The policy L2 reads *only* the belief (+ instantaneous
book-state/time); it has no private memory to route around your command.

**Why the K-simplex/brain enables it.** This is the *completeness* property no reward tweak can buy: the belief is
a **structural chokepoint**, not a summary. The battery certifies the chokepoint two ways: (i) **HC-1 ablation** —
replacing the learned belief with noise must materially hurt (verdict axis `HC-1 ablation-hurts`,
`CRYSTAL1_B2_MULTISEED.md`; on Dow "noise belief destroys everything", B3); (ii) **reactive ≫ autoregressive** —
the policy's action is explained by the *observable belief state* far better than by its own action lags
(`y_react` vs `autoreg_sim`, `certified_battery_v2.py:103–104`; 0.909±0.042 ≫ 0.632±0.119 across 3 seeds, B2). A
policy whose behavior is *not* self-predictable from its own lags but *is* predictable from the belief is one whose
memory genuinely lives in the belief — the completeness claim is measured, not asserted.

**Why R6c's 64-d latent cannot.** R6c has **no single isolated memory channel** — the 64-d penultimate is one
layer of a multi-layer feed-forward stack (`pi:[128,64]`, plus the value tower `vf:[256,128]`, plus the
`GraphHierarchicalAssetEncoder`, `w1_budget_trader_policy.py`; the net is feed-forward, no LSTM/GRU/recurrent
state). Forcing `latent64` toward a centroid steers *that layer's contribution*, but the earlier layers and the
value tower are untouched — so a code-force is a *partial* intervention with an uncertifiable residual. There is no architectural guarantee that "I set the concept" equals "I set what the policy
remembers", because R6c never had a single concept channel to begin with.

**NEW control primitive.** **Memory kill-switch / total belief clamp.** `CLAMP_BELIEF(b*)` across an episode
provably fixes the policy's *entire* memory to `b*` — the strongest-authority verb (`CLAMP`,
`HL5_FINAL_IV_Command_Certification.md §1`) becomes *tractable* because its blast radius is bounded by the one
channel. On R6c a full-memory clamp is not expressible: you would have to clamp an unknown set of activations.

**Acceptance test.** C1 causal-patch + the completeness pair: HC-1 ablation collapse (all seeds, B2) **AND**
reactive≫autoregressive (all seeds, B2). Kill-switch time-to-safe (`HL5 §4B`): after "belief-write off", trajectory
distribution returns to baseline within *k* steps — measurable here because there is one hook to remove. Status:
**[E]** on the polygon (B1/B2), **[SbC]** as a general architectural claim (single env family, n=1 real-panel L1).

---

## 3. K-simplex GEOMETRY — interpolatable/mixable commands, distribution-valued writes, on-simplex projection

**Mechanism.** The command surface is a *convex set* `Δ^{K−1}`, not `ℝ^64`. Four distinct capabilities fall out:

- **Convex-combination commands are valid commands.** Any `b = λ·b₁ + (1−λ)·b₂` with `λ∈[0,1]` is itself a legal
  belief. So two certified commands can be **mixed** and the mixture is on-surface — a true `SCHEDULE`/blend verb.
- **You write a full DISTRIBUTION, not a point.** "40% TOXIC, 35% BURST, 25% BENIGN" is one atomic command
  expressing *calibrated uncertainty*. R6c's action is a Dirichlet over assets; its *concept* write is a point in
  ℝ^64 with no notion of "I am unsure between two regimes".
- **Barycentric dose control.** The dose axis is the *simplex coordinate itself* — moving along an edge from vertex
  `i` toward vertex `j` is a bounded, interpretable, unit-ful dose (fraction of probability mass), which is exactly
  the `[MEC, MTD]` dose window C2 asks for (`HL5 §2`, "interpolation coefficient" axis). The dose is intrinsically
  normalized: `α=1` = "fully regime j", no runaway magnitude.
- **Projection keeps writes on-simplex — no OOD-manifold problem.** Any write is renormalized onto `Δ^{K−1}`
  (softmax/normalize), so *every* command lands on the natural manifold **by construction**.

**Why the K-simplex/brain enables it.** Convexity + the sum-to-one constraint *are* the manifold. There is nothing
to project *toward* because the constraint is the geometry.

**Why R6c's 64-d latent cannot — the specific architectural reason.** R6c's steering is linear interpolation in
an **unconstrained** ℝ^64: `natural + α·(centroid − natural)` (`r6c_code_control_demo.py:101–113`). The straight
line between two on-manifold points passes through **off-manifold** interior — the natural-latent cloud is a thin
curved sheet, not a convex blob. That is *precisely* why `OODGate` (Mahalanobis on natural hiddens) and
`safe_alpha` exist (`firewall.py:73–110`): `safe_alpha` **shrinks the requested α until the edited hidden passes
the gate, returning 0.0 "if even a tiny edit is off-manifold"** (`firewall.py:110`). So on R6c the dose you *ask
for* is not the dose you *get*; large regions of the command space are simply refused, and the refusal boundary is
a fitted Mahalanobis ellipsoid, not a semantic limit. The K-simplex deletes this entire failure mode: there is no
`safe_alpha` in CRYSTAL-1 because there is no off-manifold to fall into.

**NEW control primitives.**
1. **`MIX(b₁, b₂, λ)`** — a certifiably-on-surface blend of two commands (composition without leaving the
   manifold).
2. **`DOSE_ALONG_EDGE(i→j, α)`** — barycentric dose with a native `[α_min,α_max]` window and guaranteed
   monotone geometry.
3. **`WRITE_UNCERTAINTY(b)`** — command a *belief distribution* (e.g. deliberately flat b to force the
   demote-to-safe path of §4C when the world is genuinely ambiguous).

**Acceptance test.** WH6 steerability dose-response (`certified_battery_v2.py:143–156`): forced-belief sweep →
`P(action)` must be **monotone** (`monotone_frac ≥ 0.8`) with a recorded `command_threshold`, and per-class
compliance floor ≥ 0.5/0.67. The R6c contrast is directly testable: run the same dose sweep with vs without
`safe_alpha` clamping and count refused doses — CRYSTAL-1's refusal rate is 0 by construction. Status: dose-response
monotonicity **[E]** on the polygon (battery); the *mixability/OOD-free* claims are **[SbC]** — geometric certainties
but the paired-write epistasis run (`iv10_epistasis_pairedwrite.py`) shows mixtures still interact non-additively
(median |ε_logit|=1.84, §5), so "on-surface" ≠ "additive".

---

## 4. Self-supervised GROUNDED belief ⇒ a lie-detector on commands (evidence-support certificate)

**Mechanism.** The belief filter is trained self-supervised to maximize observation log-likelihood with **no regime
labels** (`train_filter`, `belief_filter.py:66–81`; loss `−ll.mean()`). It therefore carries an explicit generative
world model `(T, E, p0)`. Given a commanded belief `b*` and the actual observation stream, you can ask a question
R6c *cannot even phrase*: **does the evidence support `b*`?** — i.e. compute the filter's own posterior `b̂` from
the data and compare to the commanded `b*`. A command that pushes the belief *away* from what the world model
implies is a **detectable lie**.

**Why the K-simplex/brain enables it.** Because L1 is a *checkable* Bayes filter (it beats the memoryless baseline
in held-out log-likelihood, gate (iii) `belief_filter.py:139–150`), it defines a likelihood `P(o_{≤t} | b*)`. The
belief write is thus **falsifiable against the environment**. This is the missing rung the ladder flags as
"belief-quality precondition / authority scales with belief QUALITY not value" (`HL5 §4C`, QF72/AF447 anchors,
Tier-A aviation): CRYSTAL-1 can *compute* belief quality as the evidence-likelihood of the commanded state.

**Why R6c's 64-d latent cannot.** R6c's latent is **not generative** — it is a discriminative feature with no
`P(observation | latent)`. There is no world model to check a forced code against, so a code-force is
*unfalsifiable*: you can command any centroid and the net will produce *some* action, with no notion of whether the
data supported that concept. R6c's only guard is `OODGate` — "is this vector statistically weird?" — which is a
*density* check, not an *evidence* check. Density says "this latent is unusual"; evidence says "the observations
contradict this claim". Only the second is a lie-detector.

**NEW control primitive.** **`CERTIFY_AGAINST_WORLD(b*)`** — a pre-execution gate that computes the filter's
evidence-likelihood of the commanded belief and **refuses / demotes** the write when the world model contradicts it
(a machine-readable `GUARANTEE_DELTA` per `HL5 §4C`). This is the **command lie-detector**: it defeats the
"self-steering agentic" misuse archetype (`HL5 §9` — a policy writing its own belief to fool a monitor), because
the monitor reads the *evidence-grounded* posterior, not the written value.

**Acceptance test.** Fault-injection audit (`HL5 §9`): inject known-wrong belief writes; the evidence-likelihood
gate must flag them before promotion (low detection ⇒ theater). A/A calibration: identical arms, false-alarm rate =
nominal α. Status: the *machinery* (filter with a likelihood) is **[E]**; the lie-detector *gate as a control
primitive* is **[P]** — designed here, not yet run (no `CERTIFY_AGAINST_WORLD` script exists). Honest caveat: on
markets "regime is priced" (config-D Glosten-Milgrom equilibrium, VoI≈0 on daily/most intraday,
`B4_REAL_INTRADAY_CLOSURE.md`), so the evidence-check has *teeth* only where VoI>0 (the polygon / a real
execution-economics task, the program's open bet).

---

## 5. Story-tree-as-POLICY-HEAD — structural legibility that reward-shaping (B5) proved you cannot buy

**Mechanism.** Because the policy reads *only* the low-dim named belief (+ book-state/time), its decision function
is a small function of legible inputs — it **admits a ≤K-leaf decision-tree "story"** as a *faithful* surrogate
(fit in the battery: `DecisionTreeClassifier(max_depth=4)` over `[belief, inv, t]`,
`certified_battery_v2.py:101–103`; balanced-accuracy `y_react` 0.888–0.909). The blueprint's constructive turn is
to make that tree **the actual policy head** (hard architectural parsimony), not a post-hoc explainer.

**Why the K-simplex/brain enables it — and why this is the pivotal advantage.** B5 **falsified** the cheap
alternative: reward-shaping "become the policy your story says you are" (self-distillation agreement bonus) bought
**no reliable simulatability gain** (gains +0.01…+0.07, under margins, one seed null) because **policies OUTGROW
their stories during training** — within-round story agreement *decays* 0.94→0.67, "the story chases, never leads"
(`CRYSTAL1_B3_B4_B5_RESULTS.md` B5). The blueprint's verdict: legibility needs a **STRUCTURAL mechanism, not reward
pressure** (`CRYSTAL_AGENT_BLUEPRINT.md:54, 156–157`). The K-simplex is what *makes the structural mechanism
possible*: a tree head is only faithful if its inputs are few, named, and complete — which is exactly the belief
bottleneck. You cannot bolt a faithful tree head onto a 64-d unnamed latent.

**Why R6c's 64-d latent cannot.** A depth-4 tree over 64 unnamed continuous coordinates is neither legible (splits
on `latent[37] ≤ 0.2` mean nothing) nor faithful (64-d needs a deep tree ⇒ not a *story*). R6c's C\*≈K vocabulary
law (below) says transparency is *vocabulary-bounded*; a 64-d latent has an enormous effective vocabulary, so its
minimal faithful surrogate is large — the opposite of a story. Reward-shaping cannot shrink it (B5), and the latent
is not an architectural quantity you can cap.

**NEW control primitive.** **Editable policy = editable tree.** With the tree AS the head, a command is a **leaf
edit** ("in region {low belief, low inv, t≤13} → PROVIDE"), which is *inspectable, diffable, and version-pinnable*
as source. This is the cleanest possible `SCHEDULE` verb (regime-conditional writes) and it makes the HL coding-agent
update layer concrete — it edits *named leaves*, not weights.

**Acceptance test.** J perturb-and-predict (`certified_battery_v2.py:132–160`): pre-register the tree's predictions,
INTERVENE on the belief input, require intervention-fidelity ≥ 0.67 with per-class compliance ≥ 0.5 and monotone
dose (`j_pass`). The story-head passes iff editing a leaf changes behavior *as the leaf says*. Status: the
**diagnostic** tree is **[E]** (battery); the **story-tree-as-head** is **[Spec]** — B5-v2 design note only, the
structural mechanism is explicitly *unbuilt* (`CRYSTAL_AGENT_BLUEPRINT.md:54`). This is the single highest-value
unbuilt mechanism (§8).

---

## 6. The belief-N7 asymmetry — the arrow of information lives in an auditable place

**Mechanism.** Bayes filtering of an asymmetric-dwell world is **irreversible in time**: the belief stream carries
a measurable *arrow* (permutation-entropy irreversibility). The battery measures it three ways — action-stream N7
(**symmetric**, contemporaneous reactivity carries no arrow), belief-stream N7 (**ASYMMETRIC**), belief-increments
N7 (`n7_grouped`, `certified_battery_v2.py:62–74, 113–116`). Within-episode windows only; the increments variant
kills the trend confound.

**Why the K-simplex/brain enables it.** The arrow is a *property of the belief object*, and the belief object is a
first-class exported channel. So "where does this policy's information-integration happen, and is it real Bayesian
filtering or a lookup?" becomes an **audit query on one named stream** — belief-N7 asymmetric = genuine filtering
signature; symmetric = the policy is not actually integrating over time. On Dow this fired 100/100 (levels and
increments, B3), certifying that L1 read the *real* market's filtering structure.

**Why R6c's 64-d latent cannot.** R6c's arrow-of-information is smeared across the net and *conflated with the
action stream*. FORBID #7 (`CRYSTAL_AGENT_BLUEPRINT.md:124`) is explicit and load-bearing: **N7-action is NOT a
reactivity certificate (refuted); belief-N7 replaces it.** R6c has an action stream but no isolated belief stream,
so it can only measure the *refuted* action-N7 — it literally lacks the auditable object. You cannot ask R6c "is
your information-integration irreversible?" because there is no single place its information lives.

**NEW control primitive.** **Filtering-integrity monitor.** A live `belief-N7 asymmetric?` predicate as a
promotion/kill precondition: if the belief stream goes symmetric (the filter degenerated to a lookup / the world
went memoryless), *demote all belief-write authority* (feeds the C3 flight-envelope belief-quality precondition,
`HL5 §4C`). An information-integrity interlock R6c cannot host.

**Acceptance test.** Battery axis `belief_arrow_asym`: `b_pct ≥ 95 OR i_pct ≥ 95` (`certified_battery_v2.py:180`).
Status: **[E]** — passes all seeds on the polygon (B2) and 100/100 on real Dow streams (B3).

---

## 7. K as a tunable vocabulary (C\*≈K) — a controllability/expressivity dial

**Mechanism.** The number of simplex vertices K **is a first-class architectural quantity** you can grow. The
C\*≈K law: the minimal faithful-explanation size C\* tracks K (analytic + survives learning — gap(G=4)=[0.069,
0.077, 0.088] vs gap(G=12)=[0.314, 0.33, 0.356], complete separation every seed, C4 in `CRYSTAL1_B2_MULTISEED.md`).
So K is a **dial that trades expressivity against transparency**: small K = coarse but maximally legible vocabulary;
grow K (with an LLM proposing names for new regimes) *only when* the C\*-bend shows the world needs it
(`CRYSTAL_AGENT_BLUEPRINT.md:74`, FORBID #9: never grow G before K, line 126).

**Why the K-simplex/brain enables it.** Transparency is *vocabulary-bounded*, and K *is* the vocabulary size — a
knob with a measured law behind it. Adding a regime = adding a vertex = adding a named, addressable command target.

**Why R6c's 64-d latent cannot.** R6c's expressivity/transparency trade is **frozen at 64 and unnamed**. Its
"vocabulary" is the K-means codebook size, but that is a *post-hoc descriptive* choice over a fixed latent — growing
it does not give the policy new *legible capacity*, it just re-partitions the same opaque space more finely. There
is no C\*≈K dial because there is no K in the architecture; 64 is a width hyperparameter, not a semantic budget.

**NEW control primitive.** **`GROW_K(name)`** — a governed vocabulary-expansion operation (add a named vertex when
the C\*-bend demands it), gated through HCS. Each new vertex is immediately an addressable `SET_BELIEF` target with
a fresh C0 read-certificate. This turns "the model needs a new concept" from a retrain-and-hope into a *typed
architectural edit with an acceptance test*.

**Acceptance test.** C4 C\*≈K bend must reproduce after a `GROW_K` (the new vertex must be *load-bearing*: its
HC-1 ablation hurts and it earns a belief-N7 contribution). The C\*≈K law itself is **[E]** (multi-seed ironclad,
B2); **`GROW_K` as an online governed op is [P]** — the law is measured, the LLM-namer + grow-loop is designed not
built (`CRYSTAL_AGENT_BLUEPRINT.md:114`).

---

## 8. Honest scorecard + the 3 highest-value NEW mechanisms to build first

**Evidence-status per advantage (no inflation):**

| # | Advantage | New primitive | Status |
|---|---|---|---|
| 1 | Named write ⇒ structural C0 | `SET_BELIEF(regime,p)` | **[E]** polygon / **[SbC]** real data |
| 2 | Sole memory channel ⇒ completeness | memory kill-switch / total `CLAMP` | **[E]** polygon / **[SbC]** general |
| 3 | Simplex geometry ⇒ mixable, on-manifold | `MIX`, `DOSE_ALONG_EDGE`, `WRITE_UNCERTAINTY` | dose-response **[E]** / OOD-free **[SbC]** |
| 4 | Grounded belief ⇒ lie-detector | `CERTIFY_AGAINST_WORLD(b*)` | machinery **[E]** / gate **[P]** |
| 5 | Story-tree-as-head ⇒ structural legibility | editable-tree (leaf edits) | diagnostic **[E]** / head **[Spec]** |
| 6 | Belief-N7 ⇒ auditable info-arrow | filtering-integrity monitor | **[E]** |
| 7 | K vocabulary dial (C\*≈K) | `GROW_K(name)` | law **[E]** / op **[P]** |

**What is demonstrated vs speculative (blunt):** #1/#2/#6 are *measured* (B1/B2/B3 mechanism axes pass, multi-seed).
#3's dose-response is measured but its *composition* is not free — the paired-write epistasis run confirms writes
interact non-additively (median |ε_logit|=1.84, `HL5 §5B`), so "on-simplex" buys manifold-validity, **not**
additivity; the combination certificate is still mandatory. #4/#5/#7-op are *designed, not built*. And the honest
ceiling from the settled facts: on real *daily* economics **VoI≈0** and the transparent persister is already optimal
(config-D), so several of these advantages only pay rent where **VoI>0** — the polygon today, a real
execution-economics task tomorrow (the program's unbuilt bet, `B4_REAL_INTRADAY_CLOSURE.md`).

**The 3 to build first (highest control-value per unit effort):**

1. **Story-tree-as-policy-head (#5).** *Rationale:* B5 proved reward-shaping can't buy legibility → the structural
   route is the *only* open path, and it is the one that turns commands into diffable source. Highest value, and
   B5-v2 already scoped it. *First test:* J-fidelity ≥ 0.67 with the tree AS head (not surrogate) at return parity.
2. **`CERTIFY_AGAINST_WORLD` lie-detector (#4).** *Rationale:* it is the C3 belief-quality precondition the writ
   ladder demands (`HL5 §4C`) and defeats the self-steering misuse archetype; the filter's likelihood already
   exists (`belief_filter.py`), so it is cheap. *First test:* fault-injection detection ≥ target with A/A-calibrated
   false-alarm rate.
3. **Filtering-integrity monitor / belief-N7 interlock (#6→live).** *Rationale:* the arrow-of-information is already
   **[E]**; wiring it as a *live demote-authority predicate* is low-risk, high-leverage, and gives the whole
   command surface a runtime integrity check R6c structurally cannot have. *First test:* on a synthetically
   degraded belief stream, the monitor demotes write authority before the story breaks.

*(Ordering note: #3's `safe_alpha`-free geometry and #7's `GROW_K` are real advantages but ride on #5 — a legible
head is what makes mixed/grown commands *worth* certifying. Build the head first.)*

---

## Self-grade

**Strengths.** Every advantage is tied to a specific architectural property (named / simplex / sole-channel), each
R6c-cannot claim cites the exact mechanism (`latent64` = `policy_net.5.ReLU`; `safe_alpha` shrinking off-manifold
α; N7-action refuted → belief-N7; no generative likelihood on a discriminative latent), each carries a battery /
C0–C6 acceptance test, and evidence status is honest (only #1/#2/#6 measured; #4/#5/#7-op flagged designed-not-built;
the VoI≈0 ceiling stated). Grade: **B+**.

**Single weakest part.** Advantage **#5 (story-tree-as-policy-head)** is doing the most rhetorical work — it is
pitched as *the* pivotal structural advantage and the #1 build — yet it is the **least evidenced (`[Spec]`, B5-v2
design note only)**. Its load-bearing premise (a tree over the K-belief is *faithful enough to BE the head at return
parity*) is untested; B5 showed the naive constructive turn *failed*, and there is a real risk the structural
version trades away too much return to be a policy rather than an explainer. If one claim in this document breaks,
it is most likely this one — and it is the one I recommend building first precisely to resolve that risk fast.
