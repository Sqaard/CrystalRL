# CRYSTAL-1 CONTROLLABILITY — FINAL CONSOLIDATED REPORT (the C-ladder + the four debts)

**Scope.** The complete execution of [CRYSTAL1_CONTROLLABILITY_PLAN.md](CRYSTAL1_CONTROLLABILITY_PLAN.md): milestones
C-1 → C-6 plus the four honest debts. Every stage adversarially verified by an independent pass; every over-claim the
verifiers caught is folded back in (this report states the corrected numbers only). Detailed stage reports:
[C1+C4](CRYSTAL1_C1_C4_RESULTS.md) · [M6+C2+C3](CRYSTAL1_M6_C2_C3_RESULTS.md) · [C5+C6](CRYSTAL1_C5_C6_RESULTS.md) ·
[Debts](CRYSTAL1_DEBTS_RESULTS.md). Plan inputs: [A/B/C/D analyses](CRYSTAL1_PLAN_A_R6c_control_analysis.md) +
[ANCHORS](_crystal_ctrl/ANCHORS.md).

**One-paragraph verdict.** On the Series-G polygon, CRYSTAL-1's K-simplex belief is now a **named, causal, certified,
enforced command surface**: writes land and are proved causal against an exogenous optimum; the policy head itself is a
legible 8-leaf tree at full return parity with the raw observable structurally excluded; commands are checkable against
the world model (a lie-detector R6c's discriminative latent cannot build); composition is measured (non-additive,
sign-flipping) and therefore governed by a calibrated ledger, a co-activation cap, a joint-manifold governor, and a
C0–C6 writ ladder with a first α_machine estimate. All of it is **polygon-proven**; on real markets the surface is
correctly **fenced by the VoI gate** (the regime is priced ⇒ transparency mode), and the single standing open bet is a
real VoI>0 substrate.

---

## 1. The ladder — what was run and what it established

| Stage | Question | Result (corrected numbers) | Verdict |
|---|---|---|---|
| **C-1** read-certificate (`c1_leakage_certificate.py`) | Is a belief write a *command* or an ignorable label? | fid_A=0.769 (≥0.67), no collapse under contradicting obs, ablation-no-raise; honest **on-manifold** residual burst leak **TV 0.067 / 5.3%** (naive 0.115 was 1.7× off-manifold-inflated); belief governs 83% of contexts | **PASS** — causal, with a small honest leak |
| **C-4 pilot** distill-tree-as-head (`crystal1_c4_treehead.py`) | Can a small tree BE the policy at parity? | 6-leaf tree over [belief,inv,t]: 7.562 vs MLP 7.710 (gap −0.147, parity); **burst adds exactly zero return**; belief-ONLY insufficient (4.67 — needs legible book-state, not a leak) | **PASS** — parity representable |
| **M6** jointly-trained soft-tree head (`soft_tree_policy.py`, `crystal1_m6_softtree.py`) | The B5-mandated *structural* legibility, trained end-to-end | Frosst-Hinton tree actor, burst-**blind by construction**: **TV=0.0** (0/225 flips, verified); β=1 gap −2.57 → β=3 −0.27 → **BC warm-start −0.018 = FULL PARITY** (paired p≈0.84); sharp monotone dose, thr 0.26 | **PASS** — legible head at parity, leak closed in the action distribution |
| **C-2** barycentric dose + C1 proof (`c2_filter_grounded_c1.py`) | Upgrade C1 from probed to **proved**; dose with no refusals | Agreement vs the **exogenous** belief-MDP optimum: corner MLP **0.80** (zero-margin boundary pass), warm-started M6 **0.835**; **refusal-rate 0** (barycentric writes on-simplex by construction; R6c's `safe_alpha` contrast cited, not re-run) | **PASS** — C1 PROVED for both heads |
| **C-3** lie-detector (`c3_certify_against_world.py`) | Authority = evidence, not sharpness | sharp-but-wrong excess-NLL **0.43** vs uncertain-but-right **≈0**; detect 0.71 @ 5% FA, clear 0.81; **an entropy gate points backwards** (H 0.199<0.693 — it would trust the confident lie). Load-bearing = entropy-orthogonality, not the AUC (0.722 vs generic 0.687) | **PASS** — grounded-command checking works; structurally impossible for R6c (no likelihood) |
| **C-5** sign-epistasis on ≥2 belief dims (`c5_sign_epistasis.py`) | Does the cross-term SIGN-flip (the IV-10 debt)? | family-G4 belief×belief factorial, 216 cells: non-additivity large (median\|ε\| 2.66, 87% material); **both signs** on third-venue (+41/−16) and ABSTAIN (+38/−25); **survives raw pre-softmax logits** (not a renorm artifact); own-venue −68/0 = softmax competition | **CONFIRMED** — updates the corner's "sign-stable"; caveat: demonstrated **off the joint manifold** (0 cells within L1≤0.30 of visited) |
| **C-5** governor (`governor.py`) | Hard envelope enforcement | box governor: on-envelope byte-identical, off-envelope projected + GUARANTEE_DELTA + surfing meter | **PASS**, superseded by the debt-1 joint version below |
| **C-6** writ ladder + ledger (`writ_ladder.py`) | Runnable C0–C6 governance | certificate walks C0→C6, deployable only when full, **any version bump voids to C1**; ledger `D=Σ\|α\|τ + worst-case Σ\|ε\|τ` per-principal, **no refund on release**; salami-slicing caught (12×0.25 = exactly cap, 13th breaches), **N_live≤K** blocks over-co-activation | **PASS** (selftests) |
| **C-6** registry (`crystal1_knob_registry.yaml`) | R6c's ~214 knobs → born-legible | **5 agent-facing levers** (SET_BELIEF, EXPOSURE_MODE, hard GROUP_CONCENTRATION_CAP — fixes R6c's soft-cap W8, ABSTAIN_FLOOR, DRAWDOWN_BUDGET) + K_VOCABULARY fenced (T1) + all else engineering-fenced; 6 invariants outside the surface | **SHIPPED** (parses; migrate-not-retrofit) |

## 2. The four debts — closed

| Debt | Fix | Number | Honest scope |
|---|---|---|---|
| **1. Joint-manifold governor** | single-Gaussian Mahalanobis FAILED (multimodal cloud → mean = uniform point); fixed with **kNN novelty** (`ManifoldGovernor`) | flags **24/24** C-5 corners (box: 0/24) at **1.2% held-out FPR** | 24/24 near-inevitable by construction (corners diffuse by design); threshold aggressive (tunable) |
| **2. Calibrated \|ε\|** | per-pair table from C-5 (`c5_debts.py`); ledger default 0.5 → **6.6** | per-pair p95 ∈ [5.5, 7.2], max 8.6 (0.5 was ~13× small — N_live≤K was the real limiter) | logit-space ε charged into α·τ units = an unvalidated units bridge (bookkeeping convention) |
| **3. α_machine** | corrected after adversarial review (first pass conflated effect-wrongness with optimality-wrongness) | **behavioral α_machine = 0.0** (0/48; ≤~6% by rule of three), power 70.8%; optimality-claim false-accept 0.0208 (separate, wider) | K=2, C1/C2-core, n=48 **proxy** for the full-firewall α_machine — a first estimate, not the quantity |
| **4. VoI>0 substrate** | cannot be manufactured (the regime is priced — B4-REAL); converted to the **VoI gate** (`voi_gate.py`) | polygon VoI **5.92** (+282% of blind) → OPEN; real crypto VoI **0** → CLOSED; CN A-shares untested (no L5 data yet) | n=1 engineered-positive vs n=1 real-negative — shows the gate *can* open and *does* close on a real book; separating real candidates needs the owed VoI>0 hunt |

## 3. Is CRYSTAL-1's controllability now greater than R6c's? — a split verdict

**On the command-surface axes, measured on the polygon: YES, on every axis.**

| Controllability axis | R6c (measured/known) | CRYSTAL-1 (measured here) |
|---|---|---|
| Steering primitive | force the unnamed 64-d latent toward a post-hoc K-means code centroid | **named** belief write, K-simplex coordinate |
| Causality of a command | correlation demo (code→behavior consistency) | **proved** vs an exogenous optimum (0.80 / **0.835**) |
| Refusals / OOD | needs `OODGate`+`safe_alpha` (shrinks off-manifold steps; can refuse to 0 authority) | **0 refusals by construction** (barycentric, on-simplex) + kNN joint-manifold governor with annunciated demotion |
| Legibility of the policy itself | post-hoc codes over a continuous latent (genuinely continuous — crisp codes need a T2 VQ retrain) | the **head IS an 8-leaf tree at full return parity**; commands are leaf edits |
| Memory containment | belief implicit, memory smeared through the net → unboundable blast radius | belief bottleneck is the **sole memory channel** (HC-1 + reactive≫autoreg + M6 burst-blind TV=0) |
| Command validation | none (no likelihood in a discriminative latent) | **lie-detector**: authority tracks predictive NLL, not sharpness |
| Composition governance | none live (ghost forensics only) | measured sign-flipping cross-term → **calibrated ledger + N_live≤K + writ ladder + α_machine budget** |

**On deployed-system controllability: R6c still leads.** R6c runs on real panels with a live firewall/HCS path,
operational history, and the ~20-knob T0 surface the HCS harness actually searches. CRYSTAL-1's demonstrations are
polygon-bound (its only real-panel use is the B3 risk-mode: transparent belief-driven exposure that halves Dow drawdown
but is enable-table-conditional). So the settled fact is **superseded in refined form**: *CRYSTAL-1 now beats R6c on
command-surface quality (polygon-proven); R6c still beats CRYSTAL-1 as a deployed, real-market controllable system.*
The gap between the two is exactly one port (see §5).

**Does CRYSTAL-1 already use its advantages?** Of the seven Plan-D advantages: **actively used — 5**
(named-write C0/C1 ✓ C-1/C-2; bottleneck completeness ✓ M6/HC-1; on-simplex geometry ✓ dose/0-refusals;
grounded lie-detector ✓ C-3; story-tree-as-head ✓ M6 at parity). **Partially — 2**: the K-vocabulary dial exists as
family models + the C*≈K law, but `GROW_K` has not been run as a *certified architectural operation*; and
distribution-writes/barycentric *mixing* beyond a scalar dose (write a full simplex point as a blended command) is
exercised only in the C-5 factorial, not as a certified verb. Those two are the natural next polygon steps.

## 4. Honest ledger (whole program)

- **Established on the polygon, multi-seed or adversarially reproduced:** named-write causality; structural head at
  parity with TV=0; evidence-not-sharpness command checking; non-additive, sign-flipping composition; enforcement
  (governor/ladder/ledger) with measured operating points (FPR 1.2%, α_machine ≤~6%, ε p95 6.6).
- **Known limits:** everything is K=2/G-polygon; sign-epistasis demonstrated off the joint manifold (now gated, but not
  a deployed-manifold measurement); α_machine is a C1/C2-core proxy; the ε→D units bridge is a convention; the VoI-gate
  discrimination is n=1 vs n=1.
- **The one open bet:** a real VoI>0 execution substrate (queue/rebate/latency microstructure, or CN A-shares L5 after
  ~4–8 weeks of recorder accumulation). Until then CRYSTAL-1 on real markets = transparency/monitoring object —
  consistent with the pivoted north star (CrystalScore, not alpha).

## 5. What remains until the first Heuristic-Learning loop

The HL loop (per [HL5_MASTER_SYNTHESIS](HL5_MASTER_SYNTHESIS.md)): coding-agent **proposes** (HLX schema) → draws
**teachers** (TB) → edit compiles to **certified writes** (IV C0–C6) on **levers** (K registry) → **gated** by the
Constitution (GV) → outcome becomes a teacher. What already exists vs what's missing:

| # | Piece | Status | What's left (effort) |
|---|---|---|---|
| HL-1 | **Proposal executor** — HLX proposal JSON → sandboxed config apply → evaluator | HCS harness already does mutate→replay→score for R6c T0 knobs (`scripts/hcs_policy_forward_search_loop.py`); HLX schema designed | wire the schema as the HCS candidate format; emit a `change_dossier` per trial (**small — days**) |
| HL-2 | **GV gate v0** — tier-by-blast-radius + eval-budget ledger + independent checker | frozen probes exist (the battery + HCS replay); deflated-Sharpe trial-counting exists (`firewall.py`); dossier schema designed (TB_00) | compute tier from ghost-run behavioral delta (not proposer claims); charge every proposer query to the ledger (**small-medium**) |
| HL-3 | **Teacher Bank v0** — match-first-rank-second store incl. negative teachers | protocol designed (TB FINAL); HCS `trials.jsonl` is proto-TB data | implement `teacher_record` store over past trials + failures with trigger indices; lexicographic match gate (**medium**) |
| HL-4 | **The proposer** | HCS mutation generator = a scripted proposer today | v0: reuse HCS mutations under the schema; v1: coding-agent (LLM) emits schema proposals — **never self-classifies/approves** (**small; v1 medium**) |
| HL-5 | **Close the loop** — accept → apply → re-certify → dossier → TB update → iterate | ladder + ledger + α_machine budget + registry all coded | glue + the anchor discipline (any accepted change re-baselines the cumulative budget only via re-cert) (**medium**) |
| HL-6 | **Substrate decision** | settled: near-term = **HL-over-R6c** (real panels, live harness); CRYSTAL-1 polygon = the certified-writs testbed | run the first loop on R6c T0 knobs; port CRYSTAL-1 via the B3 risk-mode seed when a VoI>0 (or transparency-mandated) real task exists |

**Bottom line:** no research blocker remains before a first closed HL loop — HL-1/HL-2/HL-4-v0 are wiring over the
existing HCS harness plus the new governance objects; HL-3/HL-5 are moderate builds. The first loop should be
**HL-over-R6c on T0 knobs** (days-to-weeks of engineering), with the polygon CRYSTAL-1 loop running in parallel as the
place where writes are *certified*, not just scored. The research risks that HAD to be de-risked first — structural
legibility (M6), command causality (C-2), composition (C-5), machine-rate certification (α_machine) — are done.
