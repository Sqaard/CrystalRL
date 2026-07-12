# CRYSTAL-1 HL v4 — the full redesign (all 12 architecture flaws)

**Task:** "сделай redesign, исправь полноценно все изъяны." Fix, in code, every one of the 12
architecture flaws surfaced by the loop meta-critique (`CRYSTAL1_HL_LOOP_META_CRITIQUE.md`) — not the
wording, the mechanism. Then run it and **adversarially verify each fix is real, not cosmetic.**

## The one-sentence change

The old loop maximized a **scalar** (gated return). The redesign makes the objective a **frontier**:
the loop walks the `return × legibility` Pareto front under a *no-uncertified-regression* gate whose
authority is priced in the legibility it spends. Every flaw below is a consequence of that shift plus
the online/portable credit machinery around it.

## What was built

| Module | Fixes | What it does |
|---|---|---|
| `src/hl/tension.py` | F1, F10 | Tension **vector** `(return↑, description_len↓)` + Pareto `dominates`/`non_dominated` + a periodically-measured true `mdl_deficit` (Simul@≤8-leaf,K≤9 ÷ Simul@≤64-leaf ceiling) as the honest legibility axis. |
| `src/hl/pareto_gate.py` | F1,F2,F3,F4,F10,F11,F12 | `ParetoGate.review`: Pareto-non-dominated + strict-forward admission (F1/F10); rotating-holdout + alpha-investing wealth (F3); adversarial validator over **disjoint stressor regimes** (F2); authority priced in **tension-harm** = legibility spent, cumulative (F4/F11). `canary_check` generates two must-reject canaries (F12). |
| `src/hl/mechanism_bandit.py` | F8, F9 | UCB1 over operator **mechanism classes**: online credit = a frontier-expanding accept (F8); `prior()`/`seed_from()` make the credit **portable** to a new substrate (F9). |
| `interpretability/hl_v4_loop.py` | F5,F6,F7 | Integrated loop: `joint_move` atomic multi-clause bundle (F5); population over seeds → a distribution + aggregate frontier (F6); QD archive binned by a **tension coordinate** `(coverage × description_len)` (F7). |

## Flaw-by-flaw fix map

| # | Old flaw (from the meta-critique) | Fix mechanism | Run evidence |
|---|---|---|---|
| F1 | Gate ranked on **scalar return**, blind to the tension vector | Admit only Pareto-non-dominated `(return, −desc_len)` moves | Returns a 3-point frontier; **29** REJECTED_DOMINATED |
| F2 | Validator i.i.d. with dev (corr 0.988, 0 binding catches) | Adversarial validator: search **disjoint stressor regimes** (different env physics — verified NOT reseeds) for a >0.05 regression → veto; the dev-gain precondition was removed after the verifier exposed it as a corridor | **1** VETOED_BY_ADVERSARY; counterfactual: same candidate with no stressors = ACCEPTED |
| F3 | Fixed holdout mineable by a query storm (Goodhart) | **Rotating** holdout slice/query + **alpha-investing** wealth (test level shrinks as wealth drains) | wealth drained to 0.03–0.32; **6** REJECTED_HOLDOUT |
| F4 | Authority ledger priced in raw behavioral-delta magnitude | Price authority in **tension-harm** = Δlegibility spent; deny past budget | **36** REFUSED_TENSION_BUDGET; spent = 3.0/3.0 |
| F5 | Moves atomic single-clause; no correlated bundle | `joint_move` = two clauses proposed and scored as **one** gate decision | joint_move selected + accepted (bandit stat > 0) |
| F6 | Single-seed run, no distribution | Population over **3 seeds** (distinct dev_seeds) → distribution + aggregate | best_return = [4.96, 5.42, 5.06] |
| F7 | QD archive keyed on **coverage only** (capability coord) | Archive binned by `(coverage, binned TRUE MDL deficit)` — v1 used `desc_len`, which the verifier proved degenerate (`desc_len ≡ coverage+1` under dedup) | 4/5/3 niches/seed; keys `(3,0),(3,1),(3,2)` = 3 distinct legibility niches at the *same* coverage |
| F8 | Operator budget a hardcoded schedule | **UCB1** online credit = frontier-expanding accept | prior: retune 0.5 ≫ recombine 0.0 (differentiated online) |
| F9 | Mechanism credit not portable across substrates | Arms keyed on **mechanism class** → prior **transfers** G6→G12; warm-vs-cold **paired on the same seed** (only the prior differs) | warm best 3.32 / frontier 4 vs cold 3.07 / frontier 3 |
| F10 | Objective reported as a scalar best-return | Loop **returns the frontier** of non-dominated policies | 3 mutually non-dominated points reported |
| F11 | Anti-salami accrued raw probe-flip motion | Cumulative budget accrues **tension-harm** (small legibility moves still sum to the cap) | many +1-len accepts sum to exactly 3.0 then block |
| F12 | No adversarial canary that must be rejected | `canary_check` v2: three **empirically harmful** canaries (redundant-duplicate legibility-only; unconditional fixed-venue provide = **−8.59** vs incumbent +3.75; strip-unwind = +1.77, length-*reducing* to attack the waived-holdout path), run **sandboxed** (state snapshot/restore), and an escape **freezes the gate** (`compromised=True` → `REFUSED_GATE_COMPROMISED` on every subsequent move) | **27 caught / 0 escaped**, gate never compromised |

## Run (G6 × 3 seeds + paired G6→G12 transfer; final hardened gate)

- **Frontier (seed 0):** `(ret 0.0, len 1, mdl 0.0) → (1.35, 2, 0.0) → (4.03, 4, 0.198)` — return climbs only
  when certified on unseen seeds; legibility is *paid for* and *capped*, not spent silently.
- **Gate ledger (union over seeds):** 27 dominated/no-gain rejections · 24 REFUSED_TENSION_BUDGET ·
  **29 REFUSED_ALPHA_WEALTH_EXHAUSTED** · rejected-holdout retunes · canaries 27/27 caught, gate never frozen.
- **The strictness is visible in the numbers.** Before the corridor fix the run accepted 30 moves with 0
  wealth refusals; after it, dev-lucky retunes fail the rotating holdout, each failure drains alpha-wealth,
  and the query storm gets *priced to exhaustion* (29 refusals). Best-return dropped 4.96→4.03 — that gap
  is exactly the dev-mined optimism the old gate was crediting.
- **The headline contrast:** the old return-only gate would have accepted the dominated / no-gain / mined
  moves this gate refuses, walking straight into the tension-blind corner (`hl_tension_blind.py`: return
  gated 0.86→6.09 while the MDL deficit blew 0→0.41 unseen). The v4 gate stops at the legibility budget
  and at the certification wall.

## Adversarial verification (three rounds)

**Round 1 — 12 independent skeptics, one per flaw** (each reproduced + counterfactual-probed its fix):
**10× CONFIRMED_FIXED, 2× COSMETIC** (F7, F12), 0 broken. Highlights: F1's fuzz (90 trials, 0 invariant
violations); F2's decisive counterfactual (same candidate: VETOED with stressors, ACCEPTED without);
F4's decoupling probe (harm invariant to return from −100 to 10⁶); F5's instrumentation (exactly one
`review()` per bundle).

**Round 2 — repair + 3 fresh skeptics** (F7, F12, widened F2):
- **F7 → CONFIRMED_FIXED.** The MDL-deficit key is genuinely 2-D: same-coverage policies land in different
  bins for real behavioral reasons (tight-retuned vs loose thresholds, paired-window effect +0.096 ≈ 2.4
  bin-widths, ordering preserved 10/12 windows), and in-loop measurement is deterministic (bins cannot flip).
  Caveat adopted: 0.04-wide bins over-resolved the noise floor → coarsened to 0.10.
- **F2 → CONFIRMED_FIXED.** The widening is load-bearing: an equal-dev-return, stressor-harmful strip that
  the OLD gate demonstrably accepted is now VETOED_BY_ADVERSARY (−1.384 on the persistence stressor); no
  over-vetoing (ordinary accepts don't regress on stressors).
- **F12 → still COSMETIC.** Holes 1+3 closed (canaries genuinely harmful; sandbox restores bit-exactly;
  escape really freezes the gate). But a **new corridor**: the equal-length holdout waiver let a retune
  *mined against the fixed dev seeds* through (dev "+0.065", true −0.075, z=−4.16 on 4000 independent
  seeds; 2-step salami → z=−7.00). Plus canary (a) could false-positive-brick the gate on degenerate incumbents.

**Round 3 — corridor closed + red-team rematch → CONFIRMED_FIXED.** Equal-length moves now require the
rotating-holdout z-test; strictly-shorter moves get a non-inferiority test (genuine simplification still
ACCEPTED); canary (a) copies an incumbent clause verbatim. The fresh red-teamer then failed on every front:
- **(A)** the exact mined retune → REJECTED_HOLDOUT (z=−0.49); the 2-step salami dies (step 1 holdout,
  step 2 dominated). The predecessor's z=−7.00 compounding is dead.
- **(B)** mining at scale: every truly-harmful *reachable* retune (3 found, harm −0.109..−0.117) fails the
  holdout on **all 400 rotation slices** even at the easiest z_crit. A 416-perturbation grid found 33
  harmful moves that could slip a lucky slice — **every one requires an unreachable cap/thr the operator
  set cannot produce.** Why it holds: holdout seeds are i.i.d. with the truth set, so dev-overfits don't
  transfer. The salami-refill storm is self-defeating: wealth collapses 0.25→0.0005 in ~6 steps (genuine
  refills are finite, ≤G−1 venues; at high wealth one failed probe drains the entire budget).
- **(C)** non-inferiority branch: 0 material-harm escapes (harmful strips die at the dev-forward check;
  flukes are caught by the stressor adversary as a live third gate).
- **(D)** canary brick: found **only on unreachable dead-clause incumbents** (cap=0/thr>1) — see closure below.
- **(E)** sandbox restores all five state fields bit-exactly even on an ACCEPTED probe.

**Residual-risk closure (post-verdict):** the red-team's worst residual — `strip_unwind` as a self-DoS
primitive one operator-change away from live — was closed by asserting the canary **only where its harm
precondition holds** (some provide clause with cap>0, thr≤1 can actually fire). Verified: on the exact
brick state the canary is not emitted and the gate does not freeze; reachable incumbents unchanged (3/3 caught).

**Honest load-bearing assumption (kept, not fixed):** the whole holdout defense assumes holdout seeds are
drawn from the same distribution as deployment. Under genuine regime shift the z-test loses its teeth and
dev-overfits could transfer. On this static substrate that is out of scope; on a real panel it is exactly
the walk-forward/frozen-test discipline the HCS ladder already imposes — the gate does not replace it.

## Bottom line

**All 12 architecture flaws are fixed in mechanism, and every fix survived independent adversarial
verification (12 + 3 + 1 red-team agents, ~1.1M tokens of refutation attempts).** Two fixes required a
second repair round because the first version was provably cosmetic (F7's degenerate archive key, F12's
non-toxic canary + minable holdout waiver) — which is itself the strongest demonstration of the redesign's
core thesis: *a gate is only as real as the adversary that failed to break it.* The loop's objective is now
the return×legibility frontier under certified-no-regression, with authority priced in legibility, credit
assigned online and portable across substrates, and a canary tripwire that freezes the gate on escape.
