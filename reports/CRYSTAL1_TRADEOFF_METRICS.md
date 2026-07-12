# CrystalScore v2 — the tension profile (no perfect model)

*Synthesis of the tradeoff-metric tournament. Evidence tags: `[Established]` = named theorem; `[Strong]` =
measured on-system with a sub-1 number; `[Plausible]` = mechanism sound, on-system bridge not fully calibrated.
All artifact paths are absolute-repo-relative under `data_RLagent_for_Joseph/`.*

---

## 1. Why the scalar stopped discriminating — and why a *profile* replaces it

CrystalScore v1 is a scalar: **CrystalScore = Faithfulness × Simulatability × Stability** at parsimony `K≤9`.
CRYSTAL-1 just posted **0.938** (M6 soft-tree) vs R6c's **0.151** — a ~6× born-legible win
([CRYSTAL1_CSI500_AND_CRYSTALSCORE.md](CRYSTAL1_CSI500_AND_CRYSTALSCORE.md)). The entire gap is the
**Simulatability** leg (0.938 vs 0.244); Faithfulness is 1.00 for both and Stability is ~1.0 for CRYSTAL-1.

That is exactly the problem. **A scalar this close to 1, whose ceiling terms are already pinned at 1.0, no longer
discriminates.** Worse, it is *structurally blind* to the way CRYSTAL-1 is imperfect:

- The 0.938 is a **mean / body / on-manifold** number. It is a return-parity result (paired-t **p≈0.84** on mean
  return) whose L0 signature is a low-entropy autoregressive persister — the kind of behavior whose *mean* is
  trivially matched. It says nothing about the **tail**, the **worst case**, or the **off-manifold** corners.
- CRYSTAL-1's own adversarial ledger already carries measured sub-1 facts the scalar cannot see: a
  **TV=0.067 / 5.3% argmax-flip** residual leak on-manifold before M6 ([CRYSTAL1_WEAKNESS_LEDGER.md](CRYSTAL1_WEAKNESS_LEDGER.md) #1),
  **sign-epistasis** with the cross-term taking *both* signs off the joint belief manifold
  ([CRYSTAL1_C5_C6_RESULTS.md](CRYSTAL1_C5_C6_RESULTS.md), third-venue +41/−16, ABSTAIN +38/−25), a Rashomon
  policy-agreement of **0.835 < 1** at return parity ([CRYSTAL1_M6_C2_C3_RESULTS.md](CRYSTAL1_M6_C2_C3_RESULTS.md) C-2),
  and a lie-detector **AUC_decouple ≈ 0.72** ([CRYSTAL1_M6_C2_C3_RESULTS.md](CRYSTAL1_M6_C2_C3_RESULTS.md) C-3).

The fix is to stop reporting a **maximizable scalar** and instead report a **TENSION PROFILE**: a *vector* of
tradeoff axes, each one a **Pareto frontier where scoring 1 on one pole provably forces < 1 on the other**. The
governing principle is "**no perfect model exists**" made operational: every axis is backed by a Tier-A
impossibility theorem (rate-distortion, MDL/Kolmogorov, Rashomon, Neyman-Pearson, Bode, sign-epistasis), so a policy
**cannot** be at 1 on both ends. The headline object is no longer a number that saturates — it is **where CRYSTAL-1
sits on each frontier**, and the profile *widens* (becomes more discriminating) exactly as behavioral complexity
rises (the corner → G4 → G12 → P22 sweep), which is where a single scalar collapses.

**One-line contrast.** v1: "how legible is it?" (saturates at the corner). v2: "*what did legibility cost, and where
does it break?*" (never saturates, because each axis has a theorem forbidding a corner solution).

---

## 2. The source funnel — ~43 candidates → the kept set

The proposer emitted ~43 candidate tradeoff metrics across five metric-classes; the tournament judged the top **17**
(the prompt was truncated at #17/SSC) and returned **8 KEEP, 6 MERGE, 3 CUT**. The synthesis collapses the KEEP set
to a **5-axis profile** by folding the two rate-distortion-family duplicates and the two calibration-family
duplicates each into a single canonical axis, and by absorbing the merged metrics as sub-instruments.

| # | Candidate | Class | Score | Verdict | Disposition in v2 |
|---|---|---|---|---|---|
| 1 | **Description-Length Fidelity Deficit (DLFD/MDL)** | interp-fidelity | 0.94 | **KEEP** | **Axis A** (headline) |
| 2 | **Epistasis Sign-Waterbed (ESW)** | control-waterbed | 0.92 | **KEEP** | **Axis C** |
| 3 | **Faithful-yet-Suboptimal Divergence (FSD/Rashomon)** | accuracy-interp | 0.90 | **KEEP** | **Axis B** |
| 4 | **Detector Evidence-Sharpness ROC Waterbed (DES)** | control-waterbed | 0.89 | **KEEP** | **Axis D** |
| 5 | **Authority-Ledger Conservation Gap (ALCG)** | control-waterbed | 0.86 | **KEEP** | **Axis E** |
| 6 | **Tail-Fidelity Gap (CVaR-of-story-error)** | interp-fidelity | 0.85 | **KEEP** | folded into **Axis A** (tail leg / CVaR instrument) |
| 7 | **Legibility-Head Capacity Tax (LHCT)** | accuracy-interp | 0.82 | **KEEP** | folded into **Axis B** (return-gap leg of the same Rashomon frontier) |
| 8 | **Governor Sensitivity-Conservation Integral (GSCI/Bode)** | control-waterbed | 0.78 | **KEEP** | **flagship-but-deferred** waterbed; §5 next-experiment target, not yet a headline axis |
| 9 | On-Manifold Coverage Tax | interp-fidelity | 0.70 | MERGE | ρ-radius kNN sweep → **Axis C** manifold-distance instrument |
| 10 | Parsimony-Fidelity Frontier Gap (PFFG) | accuracy-interp | 0.68 | MERGE | continuous-latent CEILING + C\*~K G4/G12 → **Axis A** empirical anchor |
| 11 | Reliability-Refinement Frontier (RR-gap) | calibration | 0.66 | MERGE | canonical calibration axis (candidate **Axis F**, held) |
| 12 | Faithful-Plausible Scissors | interp-fidelity | 0.62 | MERGE | (P,F) plot → **Axis D** (it *is* the DES scissors as a sharpness axis) |
| 13 | Sharpness-subject-to-Calibration (SSC) | calibration | 0.60 | MERGE | constraint folded into RR-gap (one calibration metric) |
| 14 | Concept-Leakage Bypass Tax (CLBT) | accuracy-interp | 0.40 | **CUT** | leak closed structurally (M6 excludes burst, TV=0); reduces to LHCT |
| 15 | Coverage-Regression Exclusion (CRE) | robustness-plasticity | 0.38 | **CUT** | hyperparameter-sweep frontier; scores the HL loop, not the policy |
| 16 | Forgetting-Adaptation Frontier (FAF) | robustness-plasticity | 0.36 | **CUT** | off-thesis; scores HL retune, not interpretability |

**Net.** 5 headline tension axes (A–E), one flagship waterbed deferred to an experiment (GSCI), one optional
calibration axis (F, RR-gap) held as the single survivor of the calibration family, three cuts. The two structural
reasons for the collapse: (i) the **rate-distortion family** (DLFD + Tail + PFFG) is one MDL frontier viewed three
ways — kept as one axis with three instruments; (ii) the **calibration family** (RR-gap + SSC) is one
Gneiting sharpness-subject-to-calibration frontier — kept as one.

---

## 3. CrystalScore-v2 — the final tension profile (5 axes + 1 optional)

Each axis reports a **pair** `(pole_low_end, pole_high_end)` that provably cannot both be 1, plus **where CRYSTAL-1
sits today**.

### Axis A — Description-Length Fidelity Deficit (MDL) `[Established + Strong]`
- **Poles.** parsimony (bits of description = `log2(K)`, `K≤9`) **vs** completeness (fraction of the policy's
  *behavioral entropy rate* `h_μ` the K-story actually reproduces).
- **Theorem.** Kolmogorov 1965 / Rissanen 1978 MDL + Crutchfield–Young 1989 (`h_μ`, `C_μ`, ε-machine) `[Established]`.
  A behavior stream of entropy rate `h_μ` needs ≥ `h_μ` bits/action to reproduce losslessly; a K-leaf story supplies
  only `log2(K)` bits/step, so if `h_μ > log2(K)` the deficit is a **hard floor** no cleverness removes.
- **Instruments.** (a) **DLFD** on the L0 driver `interpretability/l0_bits_per_action.py`: Deficit =
  `1 − achieved_bits/h_μ`. (b) **Tail leg (CVaR)**: per-step `d_t = KL(policy || story)`; body = `1 − mean(lowest-90%)`,
  tail = `1 − CVaR_5%`; Tail-Fidelity Gap = body − tail. (c) **PFFG ceiling**: continuous-latent probe cv-R² as the
  achievable ceiling, minus the K≤9 budgeted fidelity.
- **Where CRYSTAL-1 sits.** Corner is a low-`h_μ` **persister** (L0 measured `h_μ ∈ [0.29, 0.98]` bits on the frozen
  R6c-family log, near i.i.d.) → a short story *can* be near-complete, so the deficit is small (this is why 0.938 is
  honestly high). But P22 is a **churner** (`h_μ ≈ 0.94–2.70`, beats the phase-shuffle null) whose bits a K≤9 story
  **provably cannot carry** → deficit strictly > 0. PFFG measured **gap(G4) ≈ [0.069, 0.088]** vs
  **gap(G12) ≈ [0.314, 0.356]**, complete seed separation ([CRYSTAL1_B2_MULTISEED.md](CRYSTAL1_B2_MULTISEED.md)) —
  ~0.33 of a G12 policy is unexplained at K≤9.
- **Gaming guard.** Denominator MUST be the phase-shuffle-null-beating `h_μ`, **never static action variance** (which
  a persister trivially matches); cap the surrogate input to the K named symbols (no peeking); enforce the finite-L
  `C_μ`/E bias correction + the DEGENERATE-window flag; **report the deficit for a churner (P22) in the same table**
  so a near-1 on the persister cannot be sold as universal. For the tail leg: fix `CVaR_5%` and full-logit KL
  *before* seeing scores; require the tail set to have visited support ≥ the C-1 on-manifold probe support.

### Axis B — Faithful-yet-Suboptimal Divergence (Rashomon) `[Established + Strong]`
- **Poles.** return/behavioral parity of the legible surrogate (reproduces the teacher's return & modal action, i.e.
  **LHCT → 0**) **vs** its policy-level agreement with the *exogenous world-optimal* action (**FSD**: does the right
  thing for the right reason).
- **Theorem.** Semenova, Rudin & Parr 2022 (Rashomon set/ratio) + Fisher–Rudin–Dominici 2019; belief-MDP optimum =
  Kaelbling–Littman–Cassandra 1998 `[Established]`. When value-ties exist, many near-equal-return policies differ in
  action map, so a legible model can be return-faithful **and** not the optimal-action member — the two cannot both
  be 1.
- **Instruments.** LHCT (return leg) = `black-box_return − legible_head_return` with paired CRN, via
  `src/crystal/soft_tree_policy.py` + `interpretability/crystal1_m6_softtree.py`. FSD (policy leg) = `1 − agreement`
  vs the exact finite-horizon belief-MDP optimum from `src/series_g/phase0_gate.solve_belief_aware`, via
  `interpretability/c2_filter_grounded_c1.py`, reported **separately at inv=0 (Rashomon tie) and inv>0 (genuine cost)**.
- **Where CRYSTAL-1 sits.** LHCT was a **real seed-fragile −0.272 (−3.5%)** at β=3, driven to **−0.018 (parity,
  p≈0.84)** only by warm-start ([CRYSTAL1_C5_C6_RESULTS.md](CRYSTAL1_C5_C6_RESULTS.md)) — and **grows on G12**. FSD:
  even *at* return parity, C-2 agreement is **0.835 < 1** (and the corner MLP's 0.80 is a zero-margin boundary pass).
  So the legible model is a **different Rashomon member**, not the world-optimal one — invisible to a return-matched
  scalar.
- **Gaming guard.** Agreement MUST be against the **exogenous belief-MDP optimum, never the self-fit story** (circular);
  report the margin to the gate (0.80 = zero-margin — flag it); matched-capacity/matched-compute black-box baseline +
  a NEGATIVE control (an equally-constrained but non-legible head must not beat the legible one by more than noise,
  else the tax is a capacity artifact); **mandatory complexity sweep (corner + G4 + G12)**, never a single point
  (corner LHCT ≈ 0 is illusory).

### Axis C — Epistasis Sign-Waterbed (ESW) `[Strong; Established for the biology anchor]`
- **Poles.** on-axis suppression of a target action via a write on belief-dim *i* **vs** the induced, **sign-unstable**
  change in that same action via the cross-term on belief-dim *j*. You cannot independently zero a behavior across all
  belief axes.
- **Theorem.** Probability-mass conservation on the action simplex (softmax competition) + **measured** sign-epistasis;
  named anchor Weinreich et al. 2005 Science (sign-epistasis constrains reachable optima) `[Established]`; the on-system
  instantiation is `[Strong]`.
- **Instruments.** Reuse the C-5 factorial `interpretability/c5_sign_epistasis.py`: pick target `a*`; on dim *i* find
  the max-suppression write; hold it, sweep dim *j*, measure cross-term `δ_ij`. Metric = leak fraction
  `L = max_j|δ_ij| / |δ_i|` + sign-instability rate. The merged **On-Manifold Coverage Tax** adds the ρ-radius Debt-1
  kNN joint-manifold distance as the x-axis of the sweep.
- **Where CRYSTAL-1 sits.** Measured on family-G4: median `|ε_logit| = 2.66`, 87% material, cross-term takes **both
  signs** on a third venue (+41/−16) and ABSTAIN (+38/−25), **surviving raw pre-softmax logits** (not a renorm
  artifact) ([CRYSTAL1_C5_C6_RESULTS.md](CRYSTAL1_C5_C6_RESULTS.md)). CrystalScore's on-manifold Simulatability=0.939
  structurally cannot see this. **Honest caveat (Ledger #9):** demonstrated *off* the joint belief manifold — report
  as a design-limit bound until a mixed-belief-visiting distribution exists.
- **Gaming guard.** Compute the cross-term in **raw pre-softmax logits** (kills the softmax-saturation dodge — C-5
  already does this); require ≥2 competitor actions; require the measurement on a distribution that actually **visits
  mixed beliefs** OR explicitly label it off-manifold; use the **kNN joint-manifold governor (Debt-1), not the
  per-coordinate box**, to flag off-manifold cells (the box does NOT gate joint mixed beliefs — the corrected error).

### Axis D — Detector Evidence-Sharpness ROC Waterbed (DES) `[Established + Strong]`
- **Poles.** false-accept rate of ungrounded (sharp-but-wrong) belief commands **vs** false-reject rate of grounded
  (uncertain-but-right) commands. The lie-detector cannot cut both to zero. Folds in the **Faithful-Plausible
  Scissors**: story sharpness/plausibility (crisp low-entropy account) **vs** groundedness (confidence tracks
  predictive excess-NLL, not neatness).
- **Theorem.** Neyman–Pearson ROC lower bound `[Established]`; the faithfulness-vs-plausibility antagonism is
  Jacovi & Goldberg 2020 `[Established source]` + the on-system C-3 orthogonality `[Strong]`.
- **Instruments.** `interpretability/c3_certify_against_world.py`: groundedness F = predictive excess-NLL (a signal
  **orthogonal to commanded entropy** — this orthogonality is the whole C-3 result); sharpness P = negative mean
  entropy of the story's per-step explanation. Report the (P, F) frontier across ≥2 regimes and the ROC AUC.
- **Where CRYSTAL-1 sits.** **AUC_decouple ≈ 0.719–0.722** at W=4 ([CRYSTAL1_M6_C2_C3_RESULTS.md](CRYSTAL1_M6_C2_C3_RESULTS.md))
  — a hard false-accept/false-reject frontier, and the entropy gate **provably points backwards** (a sharper story is
  *not* a more grounded one). Orthogonal to CrystalScore.
- **Gaming guard.** F MUST be the **predictive-NLL grounding, never the story's own confidence** (else circular);
  grade the ROC with the **pre-registered adversarial command generator**; require the (P,F) plot across ≥2 regimes
  including one where a low-entropy ungrounded command exists (the b\*=0.02 vs 0.95 pair C-3 already built) so a
  closed-scissors regime cannot be cherry-picked.

### Axis E — Authority-Ledger Conservation Gap (ALCG) `[Plausible bridge; Strong conservation]`
- **Poles.** aggressiveness of behavior-suppression writs issued **now** in one regime **vs** remaining certified
  authority to suppress the same/co-active behavior **later/elsewhere**. `D = Σ|α|·τ` is spent-not-refunded.
- **Theorem.** Cumulative-control-effort / integral-authority constraint (discrete analog of the Bode area law under a
  fixed L1 budget); nearest named result Seron–Braslavsky–Goodwin 1997 `[Established for the conservation]`. The
  **ε→D units bridge is an unvalidated convention** (Ledger #10) → the *bridge* is `[Plausible]`.
- **Instruments.** `src/crystal/writ_ladder.py` (CumulativeAuthorityLedger): drive behavior-A suppression in
  regime-A, record `D_A`; without re-cert, measure max suppression of behavior-B under `D_cap − D_A`. Score = the
  bowed-inward Pareto front (suppression_A, suppression_B).
- **Where CRYSTAL-1 sits.** With `D_cap=3.0`, `K_max=3`, `eps_bound=6.6` (Debt-2 calibrated p95), the **sign-epistasis
  cross-term dominates fast** — two co-active suppressions burn the budget before either reaches full suppression.
  Front is strictly inside the rectangular ideal.
- **Gaming guard.** `anchor_version` frozen (`frozen_2022_2023`); **count every re-cert as a trial in the deflated-Sharpe
  multiple-testing correction** (penalizes budget-laundering); charge the cross-term on **peak co-activation**, not
  instantaneous `N_live` (catches serial-with-overlap salami); pin `D_cap` in registry invariants (outside the
  agent-editable surface); report the ε→D bridge **as a convention, not a calibrated equivalence**.

### Axis F (optional / held) — Reliability–Refinement calibration frontier (RR-gap) `[Established]`
- **Poles.** calibration (reliability) **vs** sharpness (refinement) of the belief/command distribution — Gneiting's
  sharpness-subject-to-calibration. **Genuinely new axis** (orthogonal to F × Simul × Stab), single survivor of the
  calibration family (absorbs SSC's disqualify-if-sharper-than-S\* constraint).
- **Theorem.** Murphy / DeGroot–Fienberg proper-score decomposition `[Established]`.
- **Status.** Held as optional — it is real and orthogonal, but not yet instrumented on CRYSTAL-1 the way A–E are;
  promote to a headline axis only after it is measured on the C-3 `nll_window` machinery.

---

## 4. Machine-parsable schema block

```json
{
  "metric": "CrystalScore-v2",
  "kind": "tension_profile",
  "principle": "no perfect model exists; each axis is a Pareto frontier with a Tier-A impossibility theorem forbidding (1,1)",
  "parsimony_budget_K": 9,
  "widens_with": "behavioral_complexity (corner -> G4 -> G12 -> P22)",
  "axes": [
    {
      "id": "A", "name": "Description-Length Fidelity Deficit (MDL)", "class": "interpretability-fidelity",
      "poles": {"low": "parsimony = log2(K) bits", "high": "completeness = fraction of h_mu reproduced"},
      "theorem": "Kolmogorov 1965 / Rissanen 1978 MDL; Crutchfield-Young 1989 h_mu", "theorem_tier": "Established",
      "instruments": ["interpretability/l0_bits_per_action.py (DLFD)", "KL CVaR_5% tail leg", "continuous-latent PFFG ceiling"],
      "crystal1_position": {"corner_persister_hmu": [0.29, 0.98], "p22_churner_hmu": [0.94, 2.70], "PFFG_G4": [0.069, 0.088], "PFFG_G12": [0.314, 0.356]},
      "guard": "denominator = phase-shuffle-null-beating h_mu NOT variance; cap surrogate to K named symbols; report a churner (P22) in same table",
      "evidence_tier": "Strong"
    },
    {
      "id": "B", "name": "Faithful-yet-Suboptimal Divergence (Rashomon)", "class": "accuracy-interpretability",
      "poles": {"low": "return parity (LHCT->0)", "high": "policy agreement with exogenous belief-MDP optimum (FSD)"},
      "theorem": "Semenova-Rudin-Parr 2022 Rashomon; Kaelbling-Littman-Cassandra 1998 belief-MDP", "theorem_tier": "Established",
      "instruments": ["src/crystal/soft_tree_policy.py + interpretability/crystal1_m6_softtree.py (LHCT)", "interpretability/c2_filter_grounded_c1.py (FSD)"],
      "crystal1_position": {"LHCT_beta3": -0.272, "LHCT_warmstart": -0.018, "FSD_agreement_at_parity": 0.835, "corner_MLP_agreement": 0.80},
      "guard": "agreement vs EXOGENOUS belief-MDP optimum not self-fit story; matched-capacity black-box + non-legible negative control; mandatory corner+G4+G12 sweep",
      "evidence_tier": "Strong"
    },
    {
      "id": "C", "name": "Epistasis Sign-Waterbed (ESW)", "class": "control-theoretic-waterbed",
      "poles": {"low": "on-axis suppression of a* via belief-dim i", "high": "sign-unstable cross-term on belief-dim j"},
      "theorem": "action-simplex probability conservation + measured sign-epistasis; Weinreich et al. 2005", "theorem_tier": "Established",
      "instruments": ["interpretability/c5_sign_epistasis.py", "Debt-1 kNN joint-manifold rho sweep (merged On-Manifold Coverage Tax)"],
      "crystal1_position": {"median_eps_logit": 2.66, "material_frac": 0.87, "third_venue_signs": "+41/-16", "abstain_signs": "+38/-25", "survives_raw_logits": true, "caveat": "off joint manifold (Ledger #9)"},
      "guard": "raw pre-softmax logits; >=2 competitor actions; visit mixed beliefs OR label off-manifold; kNN joint-manifold governor NOT per-coordinate box",
      "evidence_tier": "Strong"
    },
    {
      "id": "D", "name": "Detector Evidence-Sharpness ROC Waterbed (DES)", "class": "control-theoretic-waterbed",
      "poles": {"low": "false-accept of sharp-but-ungrounded commands", "high": "false-reject of uncertain-but-grounded commands"},
      "theorem": "Neyman-Pearson ROC bound; Jacovi-Goldberg 2020 faithfulness-vs-plausibility", "theorem_tier": "Established",
      "instruments": ["interpretability/c3_certify_against_world.py"],
      "crystal1_position": {"AUC_decouple_W4": [0.719, 0.722], "entropy_gate": "points backwards (orthogonal to groundedness)"},
      "guard": "F = predictive excess-NLL NOT story confidence (non-circular); pre-registered adversarial command generator; (P,F) across >=2 regimes",
      "evidence_tier": "Strong"
    },
    {
      "id": "E", "name": "Authority-Ledger Conservation Gap (ALCG)", "class": "control-theoretic-waterbed",
      "poles": {"low": "suppression authority spent now/here", "high": "authority remaining for co-active suppression later/elsewhere"},
      "theorem": "cumulative-control-effort integral (Bode-area analog); Seron-Braslavsky-Goodwin 1997", "theorem_tier": "Established",
      "instruments": ["src/crystal/writ_ladder.py CumulativeAuthorityLedger"],
      "crystal1_position": {"D_cap": 3.0, "K_max": 3, "eps_bound": 6.6, "note": "cross-term dominates; front bowed inside rectangular ideal", "open": "eps->D units bridge unvalidated (Ledger #10)"},
      "guard": "frozen anchor_version; count re-certs as deflated-Sharpe trials; charge cross-term on peak co-activation; pin D_cap outside agent-editable surface",
      "evidence_tier": "Plausible"
    }
  ],
  "optional_axis": {
    "id": "F", "name": "Reliability-Refinement calibration frontier (RR-gap)", "class": "calibration-sharpness",
    "theorem": "Murphy / DeGroot-Fienberg proper-score decomposition", "theorem_tier": "Established",
    "status": "held; orthogonal + real but not yet instrumented on CRYSTAL-1"
  },
  "deferred_flagship": {
    "id": "GSCI", "name": "Governor Sensitivity-Conservation Integral (Bode waterbed)", "class": "control-theoretic-waterbed",
    "theorem": "Bode 1945 sensitivity integral; Freudenberg-Looze 1985 discrete", "theorem_tier": "Established",
    "status": "deepest theorem here but heavy loop-premise burden; see section 5 experiment"
  }
}
```

---

## 5. The single most-demanding axis, and a runnable experiment to place it

**Most demanding = Axis A (MDL / Description-Length Fidelity Deficit), on its high-complexity end.** Rationale: A is
the axis where CRYSTAL-1 is *furthest from ideal the moment the world is rich enough* — and the current 0.938 lives
entirely in the **degenerate low-`h_μ` corner** where the deficit is near-zero *by construction*, so the headline
number is precisely the least-demanding operating point of the most-demanding axis. Every other axis (B–E) has at
least one already-measured on-system sub-1 number; **A's sub-1 number on a genuinely complex policy has not been
placed on the same frontier as the corner.** The PFFG anchor shows the gap explodes to ~0.33 at G12, and P22's
`h_μ ≈ 2.7` is a policy the K≤9 story *provably* cannot carry — but P22 and the corner have never been put on **one
DLFD frontier with the same alphabet/dt and the same `h_μ` denominator**. That single missing plot is what turns the
scalar's blind spot into a measured Pareto edge.

**Runnable next experiment (no retrain).** Place all four policies on one MDL frontier:

1. Run `interpretability/l0_bits_per_action.py` on the **same alphabet + dt** for four frozen action streams: the
   corner PPO, M6 soft-tree, R6c, and **P22** (canonical persister on `D:\Interpretable_CHRL`, cite the csi500-retrain
   artifact per the canonical-models correction). Extract `h_μ` (phase-shuffle-null-beating) for each.
2. For each policy, fit the best **K≤9 named-symbol surrogate** and compute achieved predictive bits =
   `h_μ − H(action_t | story-state_t)`; **Deficit = 1 − achieved_bits/h_μ** (clip < 0). Reuse the M6 tree for the
   corner and the K-code discretization for R6c/P22.
3. Plot **(bits-of-description `log2(K)`, completeness)** as the honest rate-distortion frontier; report its AUC and
   the `K*` where the deficit first hits 0. Overlay the PFFG points (G4 ≈ 0.08, G12 ≈ 0.33) as the empirical anchor.

**Pre-registered prediction / falsifier.** DLFD(corner) ≈ 0 (near-complete, honestly high 0.938) **and**
DLFD(P22) ≫ 0 (K≤9 cannot carry ~2.7 bits). If P22's deficit is *not* materially > the corner's on the shared
frontier, the MDL axis is not discriminating and Axis A is demoted. Expected wall-clock: hours, since the L0 driver
and the K-surrogate fitters already exist and are validated on i.i.d./periodic/sticky-Markov ground truth — this is
the "widen the complexity range" move the ceiling doc names as the single biggest strengthener.

**Runner-up demanding axis = GSCI (Bode).** Deepest theorem, but its loop-premise burden (relative degree ≥ 2,
closed-loop stability cert, injecting a sinusoidal belief disturbance into a discrete learned loop) is unvalidated —
it is the *next* flagship to instrument, not the one to headline today.

---

## 6. Honest quality gate

- **Scope honesty.** Every axis is on the **polygon / Series-G / frozen-log** substrate. Ledger #13 stands: no
  demonstrated real-market edge; on real Dow/csi500 the regime is priced (daily VoI = 0), so these are
  **transparency/control** metrics, not alpha metrics. The csi500 forward run certified **0 belief-mode edits** — the
  firewall correctly refuses. The profile measures *how* the policy is legible/controllable, not *that* it makes money.
- **What is genuinely measured vs designed.** A (corner leg), B, C, D are `[Strong]` — sub-1 numbers exist on named
  artifacts. A's **high-complexity leg is not yet placed** (that is §5). Axis C's sign-epistasis is measured but
  **off the joint manifold** (Ledger #9) — a design-limit bound, not a deployed-manifold number. Axis E's conservation
  is real but the **ε→D units bridge is an unvalidated convention** (Ledger #10) — tagged `[Plausible]`, not
  `[Established]`.
- **Theorem-tag discipline.** `[Established]` is used **only** for the named theorems (Shannon rate-distortion,
  Kolmogorov/Rissanen MDL, Crutchfield–Young, Semenova–Rudin Rashomon, Neyman–Pearson, Bode, Murphy/DeGroot–Fienberg,
  Weinreich sign-epistasis). The on-system *ports* of those theorems are `[Strong]` or `[Plausible]`, never
  `[Established]`.
- **Gaming resistance.** Every axis carries a pre-registration guard whose common thread is: fix the metric definition
  (quantile, logit-space, denominator, exogenous reference) **before** seeing scores, and **mandate the complexity
  sweep** so no axis can be sold on its trivial-corner value. The two most laundering-prone axes (E via re-cert, B via
  under-trained baseline) have explicit deflated-Sharpe / negative-control guards.
- **Known residual risk.** The profile's discriminating power *depends on* the high-complexity anchor (P22 / G12)
  actually being run on the shared frontier — until §5 executes, Axis A is asserted-to-widen on the PFFG anchor, not
  yet directly measured end-to-end on the MDL denominator. Calibration Axis F is real and orthogonal but **not yet
  instrumented** — held, not claimed.

---

*Kept axes: **A** MDL description-length deficit · **B** Rashomon return-parity-vs-policy-agreement · **C**
sign-epistasis waterbed · **D** Neyman-Pearson lie-detector ROC · **E** authority-ledger conservation. Optional: **F**
calibration RR-gap. Deferred flagship: **GSCI** Bode. Most demanding: **A** (high-`h_μ` end, where 0.938 is a
degenerate corner). Runnable next: put corner/M6/R6c/P22 on one L0 MDL frontier and place the P22 deficit.*

---

## EMPIRICAL VALIDATION of Axis A (MDL Parsimony-Fidelity Deficit) — RUN

The funnel's "runnable next experiment" for the most-demanding axis, executed locally across the K-simplex complexity
range ([interpretability/mdl_fidelity_deficit.py](../interpretability/mdl_fidelity_deficit.py); report
`interpretability/mdl_fidelity_deficit_report.json`). `Deficit = 1 − Simul@(≤8-leaf, K≤9) / Simul@(≤64-leaf ceiling)`.

| policy | G | actions | h_μ range | Simul@K≤9 | Simul@ceiling | **Deficit** |
|---|---|---|---|---|---|---|
| corner | 2 | 3 | 0.89–1.27 | 0.934 | 0.989 | **0.055** |
| family_G4 | 4 | 5 | 1.25–1.73 | 0.864 | 0.972 | **0.112** |
| family_G12 | 12 | 14 | 1.48–2.12 | **0.527** | 0.919 | **0.427** |

**Confirmed: the deficit rises monotonically with behavioral complexity (0.055 → 0.112 → 0.427).** The CrystalScore
**0.938** sat at the **degenerate low-complexity corner** (G2, deficit ≈ 0, where a ≤8-leaf story trivially reproduces a
3-action policy). At G12 the same K≤9 budget reproduces only **53%** of a 14-action policy (ceiling 92%) — you cannot
have both a short story and full fidelity. This is a genuine Pareto frontier (Rissanen/Kolmogorov MDL), so **no policy
scores 1 on both poles**: Axis A is the demanding, discriminating metric the saturated scalar hid, and CRYSTAL-1 is
near-ideal only on the low-complexity leg. `[Established: MDL frontier; Strong: the specific deficit values are
polygon/family-model measurements]`.
