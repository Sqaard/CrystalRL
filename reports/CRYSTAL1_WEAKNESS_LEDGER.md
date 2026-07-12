# CRYSTAL-1 — weakness ledger (every flaw the adversarial verifiers found, and its fix status)

Compiled from the independent adversarial-verification pass run after each milestone (C-1..C-6, the 4 debts, both HL
loops). Status key: **FIXED** (corrected in code + re-verified) · **PARTIAL** (mitigated, residual caveat) · **OPEN**
(honest standing limitation). The discipline throughout: fix the code, not the wording.

## A. Method/measurement flaws — mostly FIXED

| # | Weakness (who caught it) | Fix status |
|---|---|---|
| 1 | **C-1 leakage inflated ~1.7×** — Arm B ("flip burst at fixed belief") is 61.5% off-manifold (belief is a deterministic function of burst history), so the naive TV=0.115 overstated the leak | **FIXED** — added an on-manifold measure; honest residual TV=**0.067 / 5.3%**; Arm B relabeled an off-manifold stress test |
| 2 | **M6 borderline parity, seed-fragile** — the jointly-trained soft-tree head first landed at gap −0.27, `M6_PASS=false` | **FIXED** — BC warm-start → gap **−0.018, parity true**, C1-agreement 0.835; the whole loop later hit 96% |
| 3 | **C-2 corner MLP 0.80 = zero-margin boundary pass**; M6 0.355 agreement was a no-op AGGRESS-for-ABSTAIN substitution at inv=0 | **PARTIAL** — the warm-started M6 fixed the 0.355→0.835; the corner MLP's exact-0.80 boundary fragility is a noted property of the reference, not "fixed" |
| 4 | **C-3 AUC goalpost** — leaned on AUC_decouple 0.722 vs generic 0.687 (a 0.035 gap on divergence-mismatched classes) | **FIXED (framing)** — report now states the load-bearing evidence is the entropy-orthogonality, not the AUC |
| 5 | **C-6 salami off-by-one** (12th vs 13th writ) + eps_bound=0.5 stand-in ≪ observed max (~8) + registry miscount (6 vs 5 agent-facing) | **FIXED** — 13th-writ wording; **Debt 2** calibrated eps_bound to **6.6** (per-pair p95); registry corrected to **5** |
| 6 | **HL loop governance not load-bearing** — anti-salami not cumulative (`cum_anchor_delta` was dead code), tier was audit-only, "every law fired live" overclaim | **FIXED** — anti-salami now genuinely cumulative (path length, catches oscillating drift); tier now enforces via a tier-scaled acceptance margin; honest enforcing-vs-wired split written |
| 7 | **α_machine conflated two things** — first pass measured "wrong about optimality" (belief-MDP) not "wrong about the policy's effect"; its single false-accept was a mislabeled true behavioral claim | **FIXED** — reframed to the behavioral ground truth (policy's own modal action); behavioral α_machine = **0.0** (0/48), optimality-claim rate reported separately |
| 8 | **B2 "uniqueness-tracks-fidelity" law** — an earlier n=1 claim | **FIXED (retracted)** — B2 multi-seed retraction; Rashomon demoted to diagnostic |

## B. Structural / scope limitations — the ones that are genuinely OPEN

| # | Weakness (who caught it) | Status & why |
|---|---|---|
| 9 | **Sign-epistasis (C-5) demonstrated OFF the joint belief manifold** — natural family beliefs are near-one-hot; the 4 factorial corners are jointly off-manifold (0 cells within L1≤0.30 of a visited belief) | **PARTIAL** — the box governor gap is closed (**Debt 1** kNN joint-manifold governor now flags 24/24 corners at 1.2% FPR), but the cross-term is *design-justified*, NOT "measured on the deployed manifold". Owed: a run where forced corners are on-manifold (needs a belief distribution that actually visits mixed beliefs) |
| 10 | **ε→D units bridge unvalidated** — the ledger charges a dimensionless logit-space ε into the same additive authority units as \|α\|·τ (a worst-case bookkeeping convention, not a calibrated equivalence) | **OPEN** — flagged; safe only because `N_live ≤ K` caps the pair count. Owed: a calibrated per-pair \|ε\| → authority conversion |
| 11 | **α_machine is a C1/C2-core, K=2, n=48 PROXY** for HL5's full-firewall (C1..C6) machine-rate false-accept | **OPEN (scope)** — honestly a lower-bound estimate, not the whole quantity; the full-firewall α_machine needs C4/C5 automation + a bigger population |
| 12 | **VoI-gate discrimination is n=1 vs n=1** — the polygon side is positive *by construction* (PRIMARY_ENRICHED hand-tuned so the Phase-0 gate opens); crypto side is a real measured 0 | **OPEN — the standing bet** — shows the gate *can* open and *does* close on a real book, not that it discriminates among comparable real candidates |
| 13 | **Everything is the K=2 (or G-family) POLYGON** — no demonstrated real-market edge; teeth only where VoI>0; on real markets CRYSTAL-1 = transparency layer (B4: the regime is priced, daily VoI=0) | **OPEN — the #1 program gate** — this is the single fact that decides CRYSTAL-1's ceiling vs R6c. CN L5 recorder is currently empty (no data). Needs queue/rebate/latency microstructure or CN A-shares L5 |
| 14 | **Tiny knob surface** — 3 knobs on the polygon rule-policy / 5 agent-facing registry levers | **BY DESIGN, but expandable** — this is exactly Q1 (GROW_K belief vocabulary, soft-tree leaves, reward/constraint knobs). Small surface is a safety feature, but a richer one is needed to test the loop at scale |

## C. What is NOT a weakness (verified clean)
- The coding-agent HL loop: proposer blindness confirmed behaviorally (gradient descent overshooting the myopic GM cutoff, finding the horizon-adjusted true optimum — impossible from answer-reading); the gate's ground-truth state file exactly matches the workflow history (executor faithful, no fabrication).
- CRN pairing, disjoint dev/holdout/eval seeds, deny-by-default validation — all verified sound.

## Headline
The **measurement/method flaws are fixed** (items 1–8). What remains is a small set of **honest structural limitations**
(items 9–14), and they cluster on one thing: **CRYSTAL-1 is proven on the polygon, not on a real VoI>0 market.** That
single open gate (item 13) subsumes most of the rest — it is the program's standing bet and the target of Q2. The
knob-surface (item 14) is the Q1 expansion axis.
