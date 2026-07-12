# CRYSTAL-1 controllability — C-1 + C-4 (pilot) executed

First two milestones of [CRYSTAL1_CONTROLLABILITY_PLAN.md](CRYSTAL1_CONTROLLABILITY_PLAN.md), run on the frozen
Series-G corner PPO (`src/series_g/corner_ppo_n1.zip`, K=2 polygon, deterministic-argmax). Both adversarially
verified; the C-1 leakage magnitude was corrected after verification (see §caveats). Scripts + JSON:
`interpretability/c1_leakage_certificate.py`, `interpretability/crystal1_c4_treehead.py` (+ `*_report.json`).

## C-1 — Named-write READ-CERTIFICATE: **PASS** (the belief write is a command, with a small honest residual leak)

Question: is the K-simplex belief write *causal*, or a legible-but-ignorable label the policy bypasses by re-deriving
the regime from the raw `burst` observable? The corner obs is `[belief, time, inv, burst]` — belief is the only
named channel, burst the only raw observable.

| measure | value | reading |
|---|---|---|
| interv_fidelity_A (burst consistent with belief) | **0.769** | ≥0.67 — the write LANDS |
| interv_fidelity_B (burst contradicting) | 0.672 | *stress test only* — Arm B is 62% off-manifold |
| neutral (burst ablated) | 0.749 | ablating burst does NOT raise compliance |
| burst leakage — all probes (inflated) | TV 0.115 / 11.8% argmax-flip | ~2× overstated |
| **burst leakage — ON-MANIFOLD (honest)** | **TV 0.067 / 5.3% argmax-flip** (75 mixed probes) | the number to trust |
| belief_governs_frac | 0.833 | belief changes the action in 83% of contexts |

**Gate PASS:** fid_A≥0.67 ✓, no collapse under contradicting obs ✓, burst-ablation no-raise ✓.
**Honest finding:** the named write is causal and dominant, but the raw `burst` retains a **small residual authority
(~0.067 TV, 5.3% of on-manifold contexts flip the argmax action)**. So the belief write is a command, not the
*complete* surface — a belief-based policy head should not need burst at all. That sets up C-4.

## C-4 (pilot) — STORY-TREE AS POLICY HEAD: **PASS** at return parity, burst-free

Question (the plan's pivot): can a ≤K-leaf tree over the belief BE the policy head at no return cost? (B5 falsified
reward-shaping-for-legibility → legibility must be *structural*.) Pilot method: distill a leaf-budgeted tree on the
frozen MLP's `state→action`, then **roll the tree out in the env as the policy** (tree AS head, not a surrogate).
Baseline MLP return **7.710** (SEM 0.390); parity floor = mean−0.5·SEM = **7.515**. Eval on 300 held-out episodes.

| feature set | leaves | return | gap vs MLP | parity |
|---|---|---|---|---|
| belief_only | 8 | 4.67 | −3.04 | ✗ (belief alone insufficient) |
| belief+book | 6 | 7.06 | −0.65 | ✗ (1.7 SEM short) |
| **belief+book+time (burst-free)** | **6** | **7.562** | **−0.147** | **✓ PASS** |
| belief+book+time | 8 | 7.571 | −0.139 | ✓ |
| all_incl_burst | 6 / 8 | 7.562 / 7.571 | −0.147 / −0.139 | ✓ (identical to burst-free) |

**Three findings:**
1. **Structural legibility works:** a 6-leaf tree over `[belief, inv, t]` reaches MLP return parity (gap −0.147, 0.36
   SEM; paired-t p=0.147 — indistinguishable). The story tree can BE the policy head. Command = a diffable leaf edit.
   Dose is monotone: PROVIDE at belief≤0.18, ABSTAIN at belief≥0.26 — a clean GM-threshold command curve.
2. **The C-1 leak is closed *in return*:** `all_incl_burst` returns *exactly* what `belief+book+time` returns at every
   leaf budget — **burst adds zero return**. A burst-free head matches the MLP, so the residual burst authority C-1
   found is not needed for performance.
3. **Belief-ALONE is not enough (honest limit):** a belief-only tree caps at 4.67 (gap −3.0). The head genuinely needs
   the **inventory/book state + horizon** — which is expected (the GM-optimal action is belief-dependent *modulated by
   inventory & time*, `regime_pomdp.py`) and is a *legible observable*, not a hidden leak.

## Cross-result + adversarial caveats (verified — narrow the claim honestly)

An independent adversarial verify pass confirmed both PASS verdicts and forced two corrections, both applied above:
- **C-1 Arm B is majority off-manifold** (belief is a deterministic Bayes function of the burst history, so
  high-belief-quiet / low-belief-burst rarely occur). The naive leakage (0.115) was ~2× inflated; the honest
  on-manifold number is **0.067**. fid_B (0.672) *understates* causality and is reported as a stress test, not a
  like-for-like measure.
- **"Leakage closed" is two different senses:** C-1 measures residual burst authority in the **action distribution**;
  C-4 shows burst adds nothing to **return**. These coexist — burst's residual authority lives on off-manifold or
  return-irrelevant states. The defensible claim is **"burst leakage is closed for return-relevant behavior"**, not
  "closed" unconditionally.
- **Scope caveats (all hold):** K=2 polygon only; deterministic-argmax (leakage in *sampled* behavior unmeasured); C-4
  is a **distill-and-rollout pilot**, not the full T2 jointly-trained differentiable tree head (M6) — a positive pilot
  de-risks M6 but does not replace it; and per the plan's top risk #2, all of this has teeth only where **VoI>0** (the
  polygon; the born-legible advantage still must survive a real VoI>0 execution task).
- Verifier also confirmed: no MultiCategorical extraction bug; distill/eval/C-1 seeds fully disjoint (40000+/20000+/
  10000+); no tree-driven distribution shift (tree-driven vs MLP-driven belief occupancy identical, 0.193/0.193).

## Verdict
**C-1 PASS + C-4 PASS.** On the polygon, CRYSTAL-1's named belief write is a causal command (small honest residual
leak, 0.067 TV) and a 6-leaf belief+book+time tree can BE the policy head at return parity with the leak closed in
return — delivering the B5-mandated *structural* legibility that reward-shaping could not buy. **Next:** promote the
pilot to the real M6 (a jointly-trained differentiable/soft tree head that reads belief-only, forcing the closure in
the action distribution too), then C-2 (barycentric dose + filter-grounded C1 proof) and C-3 (grounded-belief
command-checker) — all still gated by the polygon→market VoI>0 caveat.
