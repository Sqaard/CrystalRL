# CRYSTAL-1 CONTROLLABILITY IMPLEMENTATION PLAN (synthesis Д)

**The single actionable, staged plan to develop CRYSTAL-1's controllability.** This is the synthesis of the four
analyses — A (R6c control-surface weaknesses), B (external literature), C (experience corpus), D (K-simplex
advantages) — read against the shared anchors (`reports/_crystal_ctrl/ANCHORS.md`) and the HL5 control grammar
(`reports/HL5_FINAL_*`, `reports/_hl5_digest/*`). Scope is **controllability** — steering, legibility, authority,
reversibility, composition, command-certification — **not alpha/returns**. Settled facts are taken as given and not
re-litigated: **R6c > CRYSTAL-1 on controllability today** (near-term substrate stays HL-over-R6c); **regime is
priced** (Glosten-Milgrom config-D equilibrium, VoI≈0 on daily/most-intraday); the **B2 retraction** of
uniqueness-tracks-fidelity; **B5 falsified** reward-shaping-for-legibility → a *structural* mechanism is required;
**the corner is real only on the polygon**. Every claim below is a controllability claim on the polygon/battery,
never a returns claim.

---

## 1. VISION + the controllability GAP

**Vision.** CRYSTAL-1 is the born-legible successor to R6c in which the **K-simplex belief IS the writable command
surface**. Where R6c's control is a *paradox* — the legible part is a hand-written ~214-knob execution controller
and the learned part is an *unnamed 64-d penultimate latent* (`src/ppo/dirichlet_policy.py`, Plan A §0) — CRYSTAL-1
collapses that split: the policy reads *only* a **named, normalized, generatively-grounded** belief
`b ∈ Δ^{K-1}` (`src/crystal/belief_filter.py` `NeuralBayesFilter`), and a command is a scalar write to a **named
regime axis** (`obs_vec(b,…) → 2*b-1`, `interpretability/certified_battery_v2.py:120`). The end state is a policy
whose entire command surface is (a) addressable by public name, (b) on-manifold by construction, (c) grounded in a
checkable world model, and (d) governed by the same HL5 C0–C6 writ ladder and cumulative-authority budget as R6c —
but far *cheaper* to certify because the write is a legible scalar, not an entangled latent force.

**The gap (distilled from Plan A).** R6c's control surface lacks five things, and they share one root cause —
*there is no factored, named, on-manifold state to write to*:

- **W1/W2/W5 — no named, on-manifold write target.** The only steering surface is the 64-d penultimate, which is
  *unnamed* (codes are post-hoc K-means, `r6c_crisp_primitives_summary.json:49` confirms the latent is genuinely
  continuous), *implicit* (belief is smeared across the whole net — no factored state), and *off-manifold* (steering
  is linear forcing `natural + α·(centroid − natural)` that needs `safe_alpha`/`OODGate` just to be safe,
  `firewall.py:73-110`; `safe_alpha` returns 0.0 "if even a tiny edit is off-manifold"). Plan A ranks fixing this as
  **Fix 1** and its verdict is *migrate, do not retrofit*.
- **W6/W11 — no live per-command certificate and no cumulative-authority budget.** The firewall has forensic pieces
  (ghost ledger, deflated Sharpe, OODGate) but no writ object and no `D_total` ledger wired into rollout →
  salami-slicing and read→write certificate reuse are unguarded (the Hase-2023 localization≠editability trap).
- **W3/W8 — an attention-subsidized ~214-knob surface.** Safe only because a slow human self-rate-limits; a coding
  agent removes the subsidy. Group caps are *soft priorities, not hard budgets* — you cannot promise a
  post-condition.
- **W4/W7 — un-named authority transfer.** `correction_penalty` is an anti-windup tax with no named time-constant
  `Tt`; risk-mode is enable-table conditional so the *guarantee set silently changes* on a gate flip (the
  AF447/Asiana "changed protection set behind an unchanged label" failure).
- **W10 — the one native named primitive commands only cash** (R6c+ prototypes are ~equal-weight on Dow-29): "no
  stock-selection language."

**Why the born-legible K-simplex path closes it (Plan D).** Each gap is a consequence of R6c having *no single named
concept channel*, and each is dissolved by the three structural properties of the belief: **named**
(structural C0 read-certificate by construction — coordinate `k` is anchored to a measured emission signature via
`align_to_truth`, not a fitted label), **simplex-constrained** (writes are on-manifold by construction — there is no
off-manifold to fall into, so `safe_alpha` disappears; and the dose axis is the barycentric coordinate itself, a
native `[MEC,MTD]` window), and **sole memory channel** (WH1: all memory routes through the belief → owning the
write owns *everything the policy can remember* → control *completeness*, measured by HC-1 ablation + reactive≫
autoregressive). Two further advantages have no R6c analog at all: the belief is a **checkable generative world
model** (a *lie-detector* on commands — R6c's latent is discriminative, so a code-force is unfalsifiable), and the
policy-over-belief admits a **≤K-leaf story tree as a faithful head** (the *structural* legibility mechanism B5
proved you cannot buy with reward pressure).

---

## 2. DESIGN PRINCIPLES (converged from B/C/D + the HL5 grammar)

1. **Named writes, not latent forcing.** Every command addresses a public regime name (`SET_BELIEF(regime=TOXIC)`),
   never an opaque cluster id. The name is *earned structurally* (emission-signature recovery, `align_to_truth`,
   `belief_filter.py:84-93`), repaying the naming subsidy the RepE/CBM/control-token families pay with a human
   (Plan B §7). Writes live in the **obs vector** (control-token analogy, Keskar 2019 CTRL, Plan B §6.2), not a
   hooked activation.
2. **Control completeness via the belief bottleneck.** All memory routes through the K-named belief (WH1,
   `CRYSTAL_AGENT_BLUEPRINT.md:68-74,120`); unstructured recurrence is FORBIDden. Belief-MDP sufficiency (Kaelbling
   1998, Plan B §1.1) is the *theoretical warrant* that a belief write is a **complete** command; the empirical test
   is **reactive ≫ autoregressive** — if a policy self-predictable from its own action lags matches the
   belief-reactive one, `b` is not sufficient and the surface leaks (the CBM concept-leakage / control-token-ignore
   failure, Plan B §2.1/§6.2).
3. **On-simplex geometry.** The command surface is a convex set `Δ^{K-1}`, so mixtures of certified commands are
   themselves certified commands (`MIX`), you can write a *distribution* (calibrated uncertainty), and the dose is
   the intrinsically-normalized barycentric coordinate (classifier-free-guidance scale over a *named* regime,
   Plan B §6.1). No `safe_alpha`, because there is no off-manifold.
4. **Grounded-belief command-checking (the lie-detector).** Because L1 is a checkable Bayes filter trained
   self-supervised (no regime labels, `belief_filter.py:66-81`), a commanded `b*` is **falsifiable against the
   evidence**: compute the filter's own posterior and refuse/demote a write the world model contradicts. Authority
   scales with **belief QUALITY (held-out predictive likelihood), not point value or sharpness** (DVRL
   particle-collapse warning, Plan B §1.3; QF72/AF447 anchors, HL5-IV §4C). This defeats the self-steering misuse
   archetype (a policy writing its own belief to fool a monitor).
5. **Structural legibility, not reward-shaped.** B5 falsified the cheap route ("become the policy your story says
   you are" self-distillation bought no simulatability — policies *outgrow* their stories, agreement decays
   0.94→0.67). Legibility must be an **architectural constraint on the state** (Plan C A12: determinism ≠
   interpretability; you cannot shape your way to concept-nameability). The constructive form is the **story tree AS
   the policy head** (VIPER/PIRL made native, Plan B §2.3), which only becomes faithful *because* the inputs are few,
   named, and complete (the belief bottleneck).
6. **Cumulative-authority budget + reversibility.** Per-step legality ≠ trajectory safety (HL5-K §01; El Farol
   switch-count↑⇒payoff↓, Plan C A9). Many legal in-envelope writes can **salami-slice** the belief off-manifold.
   Every write debits a per-principal signed `D` ledger; a co-activation cap `N_live ≤ K` is the blunt load-bearing
   limiter until `D_total` is calibrated (HL5-IV §4). Writes are mechanism-reversible (remove the hook → identical
   forward pass, RepE §2C).
7. **The C0–C6 writ ladder + envelope enforcement.** Reuse HL5-IV verbatim: C0 addressability → C1 causal
   sufficiency → C2 dose window + placebo → C3 flight-envelope protection → C4 side-effect budget → C5 reversibility
   → C6 single-writ certificate. The honestly-transferable machinery CRYSTAL-1 currently *lacks* (Plan B §7) is the
   **runtime enforcement** layer: a **belief reference-governor / CBF-QP** that projects any requested `b*` onto
   {visited-envelope ∧ calibrated-filter-state} *before* it reaches the policy (Gilbert-Kolmanovsky 2002; Ames 2019)
   — the *structural* command-safety mechanism B5 mandates.
8. **Vocabulary is a dial (C\*≈K).** K is a first-class semantic budget with a measured law (C\*≈K ironclad,
   multi-seed, `CRYSTAL1_B2_MULTISEED.md` C4). Grow K only on the C\*-bend and only through a **fidelity** gate, never
   a distinguishability gate (DIAYN falsified locally by B2, Plan B §5.2).
9. **Writes sit INSIDE a fixed skeleton.** The policy may move inside the accounting/logging/permission skeleton but
   cannot rewrite it (CHRL invariant, Plan C A13; = GV "proposer never self-classifies"; = IV "unregistered write
   path is a certification failure"). Every write path is registered; the L0 role-contract loud-fails on a missing
   role (`src/crystal/universe.py`).
10. **Verbs beyond directional: ABSTAIN/WITHDRAW is first-class.** ~19% of a limit order's value is in the
    cancel/hold option (Kwan-Philip 2025 [project-corpus-sourced; unverified against public literature], Plan C A11);
    abstention is a distinct action-equivalence class an operator
    must be able to command, with its own per-class compliance floor and `no_trade` ghost ledger.

---

## 3. THE MECHANISMS TO BUILD (prioritized)

Effort tiers by **blast radius** (HL5-IV rule: tier by blast radius, not edit size): **T0** = config/runtime,
no retrain; **T1** = retrain; **T2** = architecture. Acceptance tests tie to the 5-gate battery
(`interpretability/certified_battery_v2.py`), the C0–C6 ladder, or a dose-response sweep. Evidence:
**designed** (specified, unbuilt) / **validated** (measured on polygon) / **speculative** (direction only).

| # | Mechanism | Source (LOCAL + EXTERNAL) | NEW control primitive | Acceptance test | Effort | Evidence |
|---|---|---|---|---|---|---|
| **M1** | **Named belief-write with structural C0 read-certificate** | D §1 (`belief_filter.py:84-93` `align_to_truth`); A W1 (`r6c_code_control_demo.py`); B §1.1 Kaelbling 1998; B §6.2 Keskar 2019 CTRL | `SET_BELIEF(regime, p)` — target is a public name | `_selftest` param-recovery `max|ΔT|,|ΔE|≤0.10`, belief-MAE ≤0.05 (`belief_filter.py:147-150`) + HC-1 ablation hurts | T0 | **validated** (polygon) / SbC real data |
| **M2** | **Concept-leakage bypass certificate** (write is a *command* only if the policy cannot re-derive regime from the observable channel) | B H1 (`certified_battery_v2.py:121` `burst`/`last_obs`); B §2.1 Koh 2020 CBM leakage; Hase 2023 localization≠editability | `interv_fidelity_B` (belief forced to *contradict* the observable) + `burst`-ablation delta | Two arms: (A) obs consistent w/ `b`, (B) obs contradicts `b`; leakage ⇒ `fid_B ≪ fid_A`. Good null certifies sufficiency | T0 | **designed** (reuses battery) |
| **M3** | **Filter-grounded C1 proof** (closed-form counterfactual: causal sufficiency *proved*, not probed) | B H4 (`belief_filter.py:37-41,59-63`); D §4; B §3.2 Geiger DAS 2023 / Meng ROME 2022 | `filter_policy_agreement` (roll filter forward from forced `b`, compare to policy action) | Agreement ≈ `interv_fidelity` ≈ analytic Bayes-optimal ⇒ certified-complete surface on the polygon | T0 | **designed** (needs small fwd-roll script) |
| **M4** | **Belief reference-governor / CBF-QP** (project `b*` onto {visited-envelope ∧ calibrated filter state}) — the B5-mandated *structural* safety layer | B H5, D §3/§4; B §4.2 Gilbert-Kolmanovsky 2002; B §4.3 Ames 2019 CBF; C3 flight-envelope | `govern(b*) → b` (nearest safe on-manifold write) | On-envelope: **byte-identical** (small-signal invariance); off-envelope: bounded, annunciated demotion (`GUARANTEE_DELTA`) | T0/T1 | **designed** |
| **M5** | **`CERTIFY_AGAINST_WORLD` lie-detector** (evidence-likelihood gate; authority ∝ belief QUALITY not value) | D §4; C H3 (proper-scoring elicitation, Gneiting-Raftery); B H2 (Igl DVRL 2018; QF72/AF447, HL5-IV §4C) | `CERTIFY_AGAINST_WORLD(b*)` — refuse/demote when evidence contradicts | Fault-injection: inject known-wrong writes, gate flags before promotion; A/A false-alarm = nominal α | T0 | **designed** (filter likelihood exists) |
| **M6** | **Story-tree AS policy head** (structural legibility; commands become diffable leaf edits) | D §5 (`certified_battery_v2.py:101-103`); B §2.3 Bastani VIPER 2018 / Verma PIRL 2018; B §2.4 Frosst-Hinton 2017 | editable-tree: leaf edit = `SCHEDULE` command in source | J-fidelity ≥0.67 with tree AS head (not surrogate) at **return parity**; per-class compliance ≥0.5, monotone dose | T2 | **speculative** (B5-v2 note; highest-value unbuilt) |
| **M7** | **Filtering-integrity monitor** (belief-N7 asymmetry as a live demote-authority interlock) | D §6 (`certified_battery_v2.py:62-74,113-116`); FORBID #7 (N7-action refuted → belief-N7) | live `belief-N7 asymmetric?` predicate → demote write authority if symmetric | On a synthetically degraded belief stream, monitor demotes *before* the story breaks | T0 | **validated** (axis) / live wiring designed |
| **M8** | **HL5-K registry on the C1 write surface: ~214→~20 typed levers** | A W3/W8 (`HL5_FINAL_K_KnobRegistry_v2.md:446-473`); C A13 fixed skeleton | ~18–20 typed levers, each carrying the K tuple; hard `group_concentration_cap` post-condition | Each lever passes the 4-property exposure test; MERGE/FENCE/EXPOSE dispositions hold | T0 | **designed** (grammar reused verbatim) |
| **M9** | **C6 single-writ certificate + cumulative-`D` ledger wired live** | A W6/W11 (`ghost_portfolios.py`, `firewall.py`); HL5-IV C0–C6 §150-226,262-287 | signed per-principal `D` ledger; co-activation cap `N_live ≤ K`; `version_pin` = checkpoint+cfbank hash | Salami-slice attack blocked; read-cert cannot license a write; any retrain lapses live writs to C1 | T0/T1 | **designed** (shared with R6c) |
| **M10** | **`MIX` / `DOSE_ALONG_EDGE` / `WRITE_UNCERTAINTY`** (simplex-geometry composition primitives) | D §3 (`r6c_code_control_demo.py:101-113` contrast); B §6.1 Ho-Salimans 2022 CFG | 3 verbs: on-manifold blend, barycentric dose, distribution-valued write | Dose-response `monotone_frac ≥0.8` + recorded `command_threshold`; **paired-write epistasis** logged (not assumed additive) | T0/T1 | **validated** (dose) / SbC (composition non-additive) |
| **M11** | **`GROW_K(name)` governed vocabulary expansion** | D §7 (C\*≈K law, `CRYSTAL1_B2_MULTISEED.md` C4); B §5.2 DIAYN as *guardrail* | `GROW_K(name)` — add a named vertex on the C\*-bend, gated through HCS | New vertex reproduces the C\*≈K bend AND is load-bearing (HC-1 hurts + earns belief-N7); passes a **fidelity** not distinguishability gate | T1 | **designed** (law validated, op unbuilt) |
| **M12** | **ABSTAIN/WITHDRAW as a first-class writable verb** | C H6/A11 Kwan-Philip 2025 [project-corpus-sourced; unverified vs public literature] (~19% cancel value); C A7 minority-game first-class no-trade | `ABSTAIN` write with its own compliance floor | Per-class compliance ≥0.67 for ABSTAIN + `no_trade` ghost ledger shows drawdown-control gain | T0/T1 | **designed** |
| **M13** | **Anti-windup `Tt` + annunciated GUARANTEE-DELTA** (name `correction_penalty`'s time-constant; every mode-flip emits a machine-readable guarantee delta) | A W4/W7 (`configs/stage0_1_active_r_pipeline.yaml:424-431`; `crystal1_b3_riskmode.py`); MEL degraded-knob table HL5-K §360-402 | first-class anti-windup channel + latched, ack-required `{guarantees_now_void[], hazard_rank}` | Alarm fires on the *gap-integral*, not instantaneous gap; each demotion annunciates + is monotone | T0 | **designed** |

**Prioritization rationale.** The **three to build first** (D §8, converging with B's honestly-transferable
verdict): **M6 story-tree-as-head** (the only open path after B5 killed reward-shaping; turns commands into diffable
source; also the *riskiest* claim so resolve it fast), **M5 lie-detector** (C3 belief-quality precondition + defeats
self-steering; filter likelihood already exists → cheap), **M7 filtering-integrity monitor live** (arrow-of-info is
already `[E]`; wiring it as a runtime demote predicate is low-risk high-leverage). M1/M2/M3 are the cheap
*certification* wins that make the surface trustworthy at all; M4/M9 are the runtime enforcement spine B flags as
missing; M8/M13 are the R6c-inherited governance hygiene.

---

## 4. STAGED MILESTONES (the C-series ladder)

Each milestone is a small, gated, falsifiable step. Format: **goal → script/file to create-or-extend →
pass/fail gate → risk**. Ordered so each de-risks the next; C-1..C-3 are cheap `[E]/[SbC]` certification wins,
C-4 is the pivotal structural build, C-5/C-6 are the expansion + migration.

### C-1 — Named-write read-certificate on the polygon (M1 + M2)
- **Goal.** Establish the write surface exists and is *causal, not leaked*: a `SET_BELIEF` command is a command,
  not a legible-but-ignorable label.
- **Create/extend.** Extend `interpretability/certified_battery_v2.py` with a **concept-leakage arm**: run
  `interv_fidelity` twice — (A) `burst`/`last_obs` consistent with forced `b`, (B) contradicting — plus a
  `burst`-ablation re-measure (HC-1 style). New script `interpretability/c1_leakage_certificate.py` driving the
  frozen corner PPO (`src/series_g/corner_ppo_n1.zip`).
- **Gate.** PASS iff `interv_fidelity_A ≥ 0.67` **and** `interv_fidelity_B` does not collapse relative to A
  (no leakage) **and** `burst`-ablation does not *raise* compliance. FAIL (leakage) ⇒ the surface is
  legible-but-not-causal → the write is not yet a command.
- **Risk.** The frozen corner may already leak on `burst` (obs contains an observable the policy can bypass) — a
  *good* falsification to surface early; if so, C-4's tree head must consume belief *only*, forcing the closure.

### C-2 — On-simplex barycentric dose command + filter-grounded C1 proof (M10 + M3)
- **Goal.** Show the dose axis is the barycentric coordinate (no `safe_alpha`, refusal-rate 0) **and** upgrade the
  C1 causal certificate from *probed* to *proved* on the polygon.
- **Create/extend.** Extend the battery's steerability diagnostic (`certified_battery_v2.py:145-155`) with a
  side-by-side **R6c contrast** (run the same dose sweep with vs without `safe_alpha` and count refused doses — 0 for
  C1 by construction). New script `interpretability/c2_filter_grounded_c1.py` that rolls
  `belief_filter.numpy_params` forward from each forced vertex to the model-predicted optimal action and computes
  `filter_policy_agreement`.
- **Gate.** PASS iff `monotone_frac ≥ 0.8` with a recorded `command_threshold`, refusal-rate = 0, **and**
  `filter_policy_agreement ≈ interv_fidelity ≈` analytic Bayes-optimal action. Disagreement localizes the gap.
- **Risk (load-bearing — Plan B §9 weakest part).** The proof leans on the filter being a *faithful* generative
  model; a stale/drifted `T,E` conflates policy-leakage with filter-miscalibration. **Mitigation: run C-3 first (or
  paired) — re-confirm the filter's held-out likelihood before trusting the closed-form counterfactual.**

### C-3 — Grounded-belief command-checker / lie-detector (M5)
- **Goal.** Make authority scale with *belief quality* (evidence-likelihood), not sharpness; defeat self-steering.
- **Create/extend.** New script `interpretability/c3_certify_against_world.py`: given a commanded `b*` and the obs
  stream, compute the filter posterior `b̂` and the 1-step predictive NLL under `E`; gate = refuse/demote when
  NLL exceeds τ. Adversarial states: `H(b)` low but NLL high (sharp-but-wrong) vs `H(b)` high but NLL low.
  Fault-injection harness injects known-wrong writes.
- **Gate.** PASS iff detection of injected wrong-writes ≥ target with A/A false-alarm = nominal α, **and**
  compliance tracks NLL rather than `H(b)`. If compliance already tracks NLL ⇒ surface is already quality-gated
  (good null). Else ⇒ add `min_predictive_ll` to the certificate's `belief_preconditions`.
- **Risk.** "Regime is priced" means the evidence-check has *teeth only where VoI > 0* (the polygon; a real
  execution-economics task) — honest ceiling, not a bug. On daily panels the gate is near-vacuous.

### C-4 — Story-tree policy-head for structural legibility (M6) — **the pivot**
- **Goal.** Deliver the B5-mandated *structural* legibility: make the ≤K-leaf tree over `[b, iv, t]` the **actual
  policy head**, so commands are diffable leaf edits — at return parity.
- **Create/extend.** New driver `interpretability/crystal1_c4_treehead.py` (sibling to `crystal1_b1.py`): replace
  the MLP action head with a differentiable/distilled tree head reading belief-only (+ book-state/time); train and
  evaluate against the frozen belief-reactive baseline. Reuse `series_g_corner_test.py` helpers.
- **Gate.** PASS iff J-fidelity ≥ 0.67 with the tree **as head** (not surrogate), per-class compliance ≥ 0.5,
  monotone dose, **and** return within a pre-registered parity band of the MLP-head baseline. A leaf edit must
  change behavior *as the leaf says*.
- **Risk (Plan D §9 weakest part — the single most likely claim to break).** B5 showed the naive constructive turn
  *failed*; the structural version may trade away too much return to be a policy rather than an explainer. This is
  built first-among-the-hard precisely to resolve that risk fast. If it fails at parity, fall back to
  tree-as-certified-surrogate + M2 leakage closure (legible command language without a tree head).

### C-5 — K-vocabulary dial + composition (M11 + M10 composition + M4 governor)
- **Goal.** Turn "the world needs a new concept" into a typed architectural edit with an acceptance test; and
  install the runtime enforcement layer.
- **Create/extend.** New script `interpretability/c5_grow_k.py` (K=3 retrain on `MultiAssetRegimePOMDP` /
  `regime_pomdp`, LLM-namer stub + HCS gate); extend `iv10_epistasis_pairedwrite.py` to the K=3 2-factor
  paired-write factorial (sign-epistasis test the IV survey explicitly owes, B H6). Add `src/crystal/governor.py`
  implementing `govern(b*)` (reference-governor / CBF-QP projector).
- **Gate.** PASS iff (a) the new vertex reproduces the C\*≈K bend and is load-bearing (HC-1 hurts + belief-N7
  contribution), passing a *fidelity* gate not a distinguishability gate; (b) the governor leaves on-envelope
  compliance byte-identical and converts off-envelope collapse into a bounded annunciated demotion; (c) paired-write
  sign-epistasis is measured (sign-stable ⇒ co-activation cap stays a blunt limiter; sign-reversal ⇒ non-commutative
  `D_total` machinery moves `[P]→[SbC]`).
- **Risk.** K=3 is a T1 retrain (cost); the governor may over-clip (lost authority) if the visited-envelope estimate
  is too tight. Composition non-additivity is *already confirmed* (median |ε_logit|=1.84) so "on-simplex ≠ additive"
  — the certificate must not assume additivity.

### C-6 — Migrate the best R6c mechanism into born-legible form + wire the writ ladder (M8 + M9 + M13)
- **Goal.** Port R6c's genuine controllability IP (the fixed skeleton + the ~20-lever grammar + the C6 certificate)
  onto CRYSTAL-1, and retire the R6c mechanisms that were only reachable by a T2 VQ-retrain (W10 native-selection —
  *migrate, do not retrofit*, Plan A Fix 1/5).
- **Create/extend.** New `heuristic_agent_r6c/contracts/crystal1_knob_registry.yaml` (the K-registry ported to the
  belief-write surface, ~20 typed levers, hard `group_concentration_cap` analog); extend `src/evaluation/firewall.py`
  + `src/evaluation/ghost_portfolios.py` to emit the C6 single-writ certificate and debit a per-principal signed `D`
  ledger with `version_pin` (checkpoint+cfbank hash) and `N_live ≤ K` co-activation cap; add the anti-windup `Tt`
  channel + `GUARANTEE_DELTA` annunciation to the mode/law-reversion path.
- **Gate.** PASS iff each ported lever passes the 4-property exposure test; the salami-slice attack is blocked by the
  `D` ledger; a read-certificate cannot license a write; any retrain lapses every live writ to C1; every mode-flip
  annunciates a machine-readable guarantee delta.
- **Risk.** The `D_total` cross-term is calibrated only on the 1-belief-dim polygon (Plan A self-grade); until C-5's
  K=3 epistasis run lands, `N_live ≤ K` is the *only* trustworthy limiter. The port must not silently inherit R6c's
  soft-cap-that-doesn't-bind pathology (W8).

---

## 5. HONEST LEDGER

**Designed vs validated vs speculative** (no inflation; mirrors D §8 scorecard):

- **Validated on the polygon (`[E]`, multi-seed):** M1 named write + structural C0 (param-recovery,
  belief-MAE); M7 belief-N7 asymmetry (100/100 on real Dow); the **completeness pair** (HC-1 ablation hurts +
  reactive≫autoregressive, 0.909 vs 0.632 across 3 seeds); M10 *dose-response monotonicity*; C\*≈K bend (C-5's law).
- **Designed but unbuilt (`[SbC]/[P]`):** M2 leakage certificate, M3 filter-grounded C1 proof, M4 reference-governor,
  M5 lie-detector, M8 registry-port, M9 writ ladder, M11 `GROW_K` op, M12 ABSTAIN verb, M13 anti-windup — the
  *machinery* exists (filter likelihood, ghost ledger, battery) but no script issues these as control primitives yet.
- **Speculative (`[Spec]`, highest-risk):** M6 story-tree-as-head — pitched as *the* pivotal advantage and #1 build,
  yet least evidenced (B5-v2 design note only); its load-bearing premise (a tree over the belief is faithful enough
  to BE the head at return parity) is untested and B5 showed the naive constructive turn failed.

**Top open risks:**

1. **α_machine (HL5-MASTER top open risk).** The whole ~20-lever + writ-ladder governance assumes a per-principal
   authority budget whose machine-rate constant `α_machine` is uncalibrated. Until then, `N_live ≤ K` co-activation
   cap is the blunt load-bearing limiter (M9). A coding agent that removes the human-rate subsidy (W3) can
   brute-force a 20-lever surface faster than the budget is calibrated.
2. **The polygon→market gap.** Every controllability win is measured on the Series-G polygon where the corner is
   real; **on competitive markets the rents are competed into quotes** (Glosten-Milgrom config-D, VoI≈0 daily). So
   M3/M5 have *teeth only where VoI > 0*; the born-legible advantages may not survive contact with a real book any
   better than R6c's do (Plan A self-grade; Plan C critique §2). The program's open bet is a real
   execution-economics task with VoI > 0.
3. **The B5 lesson — legibility needs a STRUCTURAL mechanism.** Reward-shaping-for-legibility is falsified; the
   entire plan therefore rests on **M6 (structural tree head)**, the least-evidenced mechanism. If M6 fails at return
   parity, the "structural legibility" pillar degrades to a certified *surrogate* (still legible, but the policy is
   not the story).
4. **Composition is not additive even on-simplex.** The paired-write epistasis run confirms non-additivity (median
   |ε_logit|=1.84); "on-manifold" buys validity, not additivity. The combination certificate is mandatory and the
   sign-epistasis run (C-5) is owed before `D_total` cross-terms can be trusted.
5. **Filter faithfulness confound (Plan B §9).** M3's closed-form counterfactual is only as good as the filter's
   `T,E`; run C-3 (recalibrate) before/paired with C-2 (proof) or `filter_policy_agreement` conflates policy-leakage
   with filter drift.
6. **Capacity-matching discipline (memory: text-twin lesson).** Any "posterior-input beats latent-input" claim (the
   A1 replication, Plan C H1) must be obs-dim + noise-placebo matched, or the advantage is capacity, not naming — a
   RED that was overturned to UNINFORMATIVE once already.

**What would FALSIFY the whole controllability thesis:**
- **C-1 shows irreducible leakage** *and* **C-4's belief-only tree head cannot close it at return parity** — i.e.
  the policy can always re-derive regime from the observable channel, so the named write is decorative, not causal.
- **M3 returns `filter_policy_agreement` uncorrelated with `interv_fidelity`** even after C-3 recalibration — the
  policy ignores the *belief semantics* (the deepest possible leakage), so the K-simplex is legible but not a command
  surface.
- **The polygon→market gap is total:** no VoI > 0 substrate is ever found, so every certificate is
  polygon-local and the born-legible surface never earns rent over R6c's HL-over-R6c substrate on any deployable
  task. (This would demote CRYSTAL-1 from "successor" to "interpretability research object" — consistent with the
  pivoted north star being CrystalScore, not alpha.)

---

## 6. SOURCE LINKS (consolidated, traceable)

**LOCAL — repo files & reports:**
- Anchors: `reports/_crystal_ctrl/ANCHORS.md`
- The four analyses: `reports/CRYSTAL1_PLAN_A_R6c_control_analysis.md`,
  `reports/CRYSTAL1_PLAN_B_literature_external.md`, `reports/CRYSTAL1_PLAN_C_experience_corpus.md`,
  `reports/CRYSTAL1_PLAN_D_ksimplex_advantages.md`
- CRYSTAL-1 core: `src/crystal/belief_filter.py` (`NeuralBayesFilter`, `mats`, `align_to_truth`, `train_filter`,
  `_selftest`, `numpy_params`), `src/crystal/universe.py` (L0 role-contract), `src/series_g/regime_pomdp.py`,
  `src/series_g/multiasset_env.py`, `src/series_g/corner_ppo_n1.zip`
- Battery + drivers: `interpretability/certified_battery_v2.py`, `interpretability/crystal1_b1.py`,
  `interpretability/b2_multiseed.py`, `interpretability/crystal1_b3_riskmode.py`,
  `interpretability/crystal1_b4_bridge.py`, `interpretability/b5_crystallize.py`,
  `interpretability/iv10_epistasis_pairedwrite.py`, `interpretability/cross_policy_crystal.py`,
  `interpretability/series_g_corner_test.py`
- CRYSTAL-1 reports: `reports/CRYSTAL_AGENT_BLUEPRINT.md` (L0–L3 axioms, WH1, FORBIDs), `reports/CRYSTAL1_B0_B1_RESULTS.md`,
  `reports/CRYSTAL1_B2_MULTISEED.md`, `reports/CRYSTAL1_B3_B4_B5_RESULTS.md`, `reports/B4_REAL_INTRADAY_CLOSURE.md`
- R6c control surface: `src/ppo/dirichlet_policy.py`, `src/ppo/w1_budget_trader_policy.py`,
  `src/ppo/stage0_1_weight_env.py`, `configs/stage0_1_active_r_pipeline.yaml`, `configs/_r6c_latent_ab.yaml`,
  `reports/firewall_upgrade/r6c_code_control_demo.py`,
  `reports/firewall_upgrade/r6c_code_layer/{r6c_codebook.npz, r6c_code_dictionary.csv, r6c_crisp_primitives_summary.json}`,
  `reports/r_k_window_analysis/R6C_GROUP_RISKAWARE_TOPK_IMPLEMENTATION_NOTE.md`, `reports/MODEL_REGISTRY.md`
- Firewall/HCS/ghost: `src/evaluation/firewall.py` (`deflated_sharpe_ratio`, `OODGate`, `safe_alpha`,
  `block_bootstrap_ci`), `src/evaluation/ghost_portfolios.py` (`GhostLedger`),
  `scripts/hcs_policy_forward_search_loop.py`
- HL5 control grammar: `reports/HL5_FINAL_K_KnobRegistry_v2.md`, `reports/HL5_FINAL_IV_Command_Certification.md`
  (C0–C6 writ ladder), `reports/HL5_FINAL_GV_HL_Constitution.md`, `reports/HL5_FINAL_HLX_Operator_Grammar.md`,
  `reports/HL5_FINAL_TB_TeacherBank_Protocol.md`, `reports/HL5_MASTER_SYNTHESIS.md` (α_machine),
  `reports/HL5_IV10_EPISTASIS_DERISK.md`, `reports/_hl5_digest/{K,IV,GV,HLX,TB}_digest.md`
- Experience corpus (via Plan C): `experience/…/POMDP-рынок…`, `…/Режимно-переключающийся POMDP`,
  `…/Brock–Hommes…`, `…/Kyle 1985…`, `…/Minority Game…`, `…/Santa Fe…`, `…/El Farol…`, `…/Taylor-rule CB…`,
  `…/market-maker LOB…`, `…/Glosten–Milgrom…`, `…/Lux–Marchesi…`,
  `experience/WORLD OF BEST TRADING BOT/CHRL_C_INNOVATIVE_CHRL_APPROACH.md`

**EXTERNAL — named sources (Tier-A unless noted):**
- Belief-state RL: Kaelbling, Littman & Cassandra 1998 (belief-MDP sufficiency); Littman-Sutton-Singh 2001 /
  Boots-Siddiqi-Gordon 2011 (PSRs); Igl-Zintgraf et al. 2018 (DVRL); Ma-Karkus et al. 2020 (particle-filter RL)
- Interpretable-by-design: Koh-Nguyen et al. 2020 (Concept Bottleneck Models); Mahinpei/Margeloiu et al. 2021
  (concept leakage); Chen-Li et al. 2019 (ProtoPNet); Verma-Murali et al. 2018 (PIRL); Bastani-Pu-Solar-Lezama 2018
  (VIPER); Frosst-Hinton 2017 (soft decision trees)
- Steering / causal mediation: Zou-Wang et al. 2023 (RepE); Turner et al. 2023 (ActAdd); Li-Patel et al. 2023 (ITI);
  Templeton et al. 2024 (Golden-Gate Claude); Vig et al. 2020; Meng et al. 2022 (ROME); Geiger-Wu et al. 2023 (DAS);
  Hase et al. 2023 (localization ≠ editability)
- World-model / envelope control: Hafner-Lillicrap et al. 2020-23 (Dreamer v1-v3); Gilbert & Kolmanovsky 2002
  (reference governor); Ames-Coogan et al. 2019 (Control Barrier Functions)
- Skills / generation: Sutton-Precup-Singh 1999 (options); Kulkarni et al. 2016 (h-DQN); Eysenbach-Gupta et al. 2019
  (DIAYN — cautionary); Dhariwal-Nichol 2021 / Ho-Salimans 2022 (classifier-free guidance); Keskar et al. 2019
  (CTRL); Li & Liang 2021 (Prefix-Tuning)
- Market microstructure / ABM (from the corpus): Kyle 1985; Glosten-Milgrom 1985; Brock-Hommes 1998; Anufriev-
  Panchenko; Arthur-Holland-LeBaron-Palmer-Tayler (Santa Fe); Marsili-Challet-Zecchina 1999 (minority game);
  Rand-Stonedahl 2010 / Chmura-Pitz 2006 (El Farol); Taylor 1993 / Orphanides 2001 / Rudebusch 2002 (Taylor rule);
  Macrì et al.; Kinathil et al. 2016; Kwan-Philip 2025 [project-corpus-sourced; unverified against public literature];
  Brodu (decisional states); Lux-Marchesi 1999; Gneiting-Raftery
  (proper scoring rules); Crutchfield/Barnett (ε-machines); Liu et al. (shortcuts to automata)
