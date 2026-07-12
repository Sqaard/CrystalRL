# CRYSTAL-1 controllability — M6 (real soft-tree head) + C-2 + C-3 executed

Follows [CRYSTAL1_C1_C4_RESULTS.md](CRYSTAL1_C1_C4_RESULTS.md). Promotes the C-4 distill pilot to the real M6
(jointly-trained soft-tree head) and executes C-2 (barycentric dose + filter-grounded C1 proof) and C-3
(grounded-belief lie-detector). All on the Series-G corner polygon (K=2, deterministic + distributional). New code:
`src/crystal/soft_tree_policy.py`, `interpretability/{crystal1_m6_softtree,c2_filter_grounded_c1,c3_certify_against_world}.py`.

## M6 — jointly-trained belief-only SOFT-TREE head: **structural leak closure ✓, near-parity return (−3.5%)**

The C-4 pilot distilled a hard sklearn tree post-hoc; M6 is the real thing — a differentiable soft decision tree
(Frosst–Hinton style, `SoftDecisionTree`) that IS the PPO actor, trained end-to-end, reading ONLY `[belief, inv,
time]` (obs idx `[0,2,1]`) and **structurally blind to the raw `burst` observable** (idx 3). The critic sees full obs
(training-only). 8 leaves, gate sharpness β=3, 500k steps.

| metric | value | reading |
|---|---|---|
| **burst-leak in the ACTION DISTRIBUTION** | **TV_mean = TV_max = 0.000** | **CLOSED by construction — verified real** (adversarial check: flipping burst → **0/225 action changes**; the action-dist is byte-identical, not a rounding artifact). The C-1 residual leak is gone from the action distribution, not just from return |
| return (soft-tree vs MLP) | 7.437 vs 7.710 | gap **−0.272 (−3.5%)**; paired diff SEM 0.099 |
| strict parity band (mean − 0.5·SEM = 7.515) | 7.437 < 7.515 → **PARITY = false** | **just-missed / borderline — parity NOT achieved by the pre-registered band, and the verdict is SEED-FRAGILE** (a 120-seed rerun flipped to PARITY=true; the shipped 300-seed run is false). Report as "borderline," never as a clean pass |
| dose (P(PROVIDE) vs belief) | 0.863 → 0.010, monotone, threshold **0.26** | clean GM-threshold command curve, matching the corner MLP's 0.26 |
| β=1 → β=3 crisping | gap −2.57 → **−0.27** | crisper gates closed 89% of the return gap |

**Honest verdict:** M6 delivers the *primary* controllability win — the burst leak is **structurally closed in the
action distribution** (TV=0, adversarially verified), on a legible 8-leaf soft tree, with a monotone command dose.
On return it is **borderline — it JUST MISSES the pre-registered parity band (M6_PASS=false) and the pass/miss is
seed-fragile**; the jointly-trained soft tree costs ~3.5% return vs the MLP (the B5 head-level cost), whereas the C-4
distilled *hard* tree hit full parity (−0.14). The residual gap is soft-mixture optimization cost → the clear next
step is a **warm-start from the distilled hard tree** (or a β-anneal to hard) to reach robust parity.

## C-2 — barycentric dose (no refusals) + filter-grounded C1 **PROOF** (probed → proved)

C-1 probed causality against a story tree the policy was *distilled to fit* (circular). C-2 proves it against an
**exogenous** reference: the exact finite-horizon belief-MDP optimum `pol[t,belief,inv]` from
`src/series_g/phase0_gate.solve_belief_aware(env.m)`.

| policy | filter_policy_agreement vs belief-MDP optimum | refusal-rate | dose | verdict |
|---|---|---|---|---|
| **corner MLP** | **0.80** (exactly on the ≥0.80 gate — a **zero-margin boundary pass**) | 0 | monotone, thr 0.26 | **C1 PROVED** (fragile margin) |
| M6 soft-tree | 0.355 | 0 | monotone, thr 0.26 | correct belief-dose, but substitutes AGGRESS for ABSTAIN (see below) |

- **C1 PROVED for the corner policy:** under a forced belief, its action matches the world-model-optimal action 80% of
  the grid — the named write drives the policy to the *exogenous* dynamic optimum, not merely to a self-fit story. Caveat:
  0.80 sits *exactly* on the ≥0.80 threshold (zero margin) — a real but boundary pass. (The optimum uses ABSTAIN heavily,
  221/375 probes; the MLP genuinely matches that, so the agreement is real.)
- **Refusal-rate = 0 by construction:** for K=2 the belief IS the barycentric coordinate; every `b∈[0,1]` is an
  on-simplex write, so no OOD projection is needed. Structural contrast: R6c steers by forcing the 64-d latent toward a
  code centroid and needs `safe_alpha` (`src/evaluation/firewall.py`) to shrink off-manifold steps → non-zero refusals.
  (CRYSTAL-1's 0 is measured; the R6c contrast is *cited, not re-run* — an asymmetric, disclosed comparison.)
- **The M6 0.355 puzzle — the honest mechanism (confirmed):** M6 emits ABSTAIN **0 times / 375 probes**; the belief-MDP
  optimum uses ABSTAIN 221 times. M6 substitutes **AGGRESS**. At **inv=0**, AGGRESS is a *literal no-op* — identical
  transition (0→0) AND identical reward (0.000) to ABSTAIN (verified in `regime_pomdp.py`) — and trajectories mostly sit
  at inv=0 (M6 rarely PROVIDEs, so rarely accumulates inventory), so the substitution is **return-free there** → a genuine
  Rashomon/behavioral tie that craters action-agreement without hurting return. BUT at **inv>0** AGGRESS≠ABSTAIN and costs
  (e.g. −0.35 vs −0.05 benign, −1.55 vs −0.05 toxic at inv=1), so the substitution is **also real sub-optimality** that
  contributes to the −0.272 return gap. So 0.355 is *both* a Rashomon tie (at inv=0) *and* evidence of genuine inv>0
  sub-optimality — not a probe bug, and not glossed as merely "a different near-optimal policy."

## C-3 — grounded-belief COMMAND-CHECKER (lie-detector): **PASS** — authority tracks evidence, not sharpness

CRYSTAL-1's belief is a self-supervised *generative* world model, so a commanded belief can be certified against the
evidence via predictive NLL (`RegimePOMDP.predict/obs_prob/update`, forward window). R6c's discriminative latent has
no likelihood and structurally cannot build this. The crux is a decoupling of *sharpness* from *groundedness*:

| command class | commanded entropy H | excess-NLL (ungrounded signal) | detector @ 5% FA |
|---|---|---|---|
| **sharp-but-wrong** (b*=0.95 where evidence says benign) | **0.199 (low)** | **0.43** | flagged **0.71** |
| **uncertain-but-right** (b*=0.5 where evidence is ambiguous) | **0.693 (high)** | **≈0** | cleared **0.81** |

- **The load-bearing evidence is the entropy-orthogonality, not an AUC gain.** Verified robust: two commands at
  near-identical LOW entropy but opposite groundedness (b*=0.02 vs 0.95 against a low-toxicity truth) get excess-NLL
  **0.01 vs 0.40** — the detector separates them while entropy cannot. The confident lie carries ~0.43 excess-NLL vs ~0
  for the honest hedge, and **a sharpness/entropy gate points backwards** (H 0.199 < 0.693 → it would *trust* the
  confident lie, *distrust* the honest hedge). So authority must track **evidence (predictive NLL)**, not sharpness.
- Honest framing note: `AUC_decouple=0.722` is only marginally above the generic `AUC_lie_vs_honest=0.687` (Δ0.035) and
  contrasts a *maximal* lie (|b*−b̂|≈0.86) against a *near-honest* command (|b*−b̂|≈0.08) — divergence-mismatched classes,
  so that AUC is an easy separation and is NOT the headline. Generic flip-lie detection is moderate (~0.69) because an
  uncertain belief's flip is often *partially* evidence-supported — an honest limit of the belief's own ambiguity. The
  claim rests on the entropy-orthogonality above, not on 0.722 > 0.69.

## Cross-milestone reading + caveats
- **The born-legible advantage is now demonstrated on three axes:** (M6) a legible tree can BE the actor with the raw
  observable structurally excluded → leak closed in the action distribution; (C-2) the named write is causal against an
  *exogenous* optimum with zero refusals; (C-3) commands are checkable against the world model — a lie-detector R6c
  cannot build. Together these are the "named, on-simplex, grounded, structurally-legible command surface" the plan bet on.
- **Honest limits (all hold):** K=2 polygon; M6 is **near-parity, not full parity** (−3.5%; warm-start is the owed next
  step); the M6 0.355 optimal-agreement shows near-optimal-value ≠ optimal-policy; C-2 proves C1 on the *corner* policy
  (M6's own C1 is dose-correct but not full-grid); C-3 is a forward-window (w-step-late) certification; and per the
  plan's top risk #2, all of this has **teeth only where VoI>0** — the born-legible surface still must survive a real
  VoI>0 execution task, not just the polygon where the corner is real.

## Verdict (adversarially verified; two framing corrections applied)
- **M6:** the burst leak is **structurally closed in the action distribution** (TV=0 — verified real, 0/225 flips). On
  return it is **borderline: it just-misses the pre-registered parity band (M6_PASS=false) and the pass/miss is
  seed-fragile** — state it as "near/borderline parity," not achieved. Warm-start from the C-4 hard tree is the owed step.
- **C-2:** C1 is **PROVED for the corner policy** against the *exogenous* belief-MDP optimum (0.80, a zero-margin
  boundary pass), refusal-rate 0 on-simplex. M6's 0.355 is the honest **ABSTAIN→AGGRESS substitution** — a Rashomon tie
  at inv=0 *and* genuine inv>0 sub-optimality.
- **C-3:** the grounded-belief lie-detector **works** — authority tracks **evidence (predictive NLL), not sharpness**
  (the entropy gate provably points backwards); the load-bearing evidence is the entropy-orthogonality, not the AUC.

All three rest on the **K=2 polygon** and have **teeth only where VoI>0** — the born-legible surface still must survive
a real VoI>0 execution task. Next on the ladder: warm-start M6 to robust parity, then C-5 (K-vocabulary dial + governor
+ the owed ≥2-belief-dim sign-epistasis run) and C-6 (migrate R6c IP + wire the C6 writ ladder + cumulative-authority ledger).
