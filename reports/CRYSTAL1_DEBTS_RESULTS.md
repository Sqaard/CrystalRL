# CRYSTAL-1 controllability — the four remaining debts, closed

The honest debts from [C5/C6](CRYSTAL1_C5_C6_RESULTS.md): (1) joint-manifold governor, (2) calibrated per-pair |ε|,
(3) α_machine calibration, (4) a VoI>0 substrate. All addressed; #4 honestly scoped. New code:
`src/crystal/governor.py` (ManifoldGovernor), `interpretability/{c5_debts,c_alpha_machine,voi_gate}.py`.

## Debt 1 — joint-manifold governor: **CLOSED**
The box `BeliefGovernor` gates each coordinate's marginal but passes JOINT off-manifold beliefs (the diffuse corners
C-5's factorial forces). First attempt (single-Gaussian Mahalanobis) failed by construction — the visited belief cloud
is **multimodal** (near-one-hot clusters around vertices) whose mean IS the uniform point, so Mahalanobis-to-mean makes
the uniform (diffuse) belief the *most* central. Fixed with **kNN novelty detection** (`ManifoldGovernor`): a command is
on-manifold iff its L1 distance to the k-th nearest visited belief is within a threshold calibrated from the visited
set's own kNN distances. Result (`interpretability/c5_debts.py`, natural family-G4 beliefs, entropy 0.367 = near-one-hot):

| governor | C-5 factorial corners flagged off-manifold |
|---|---|
| box (marginal) | **0 / 24** (the gap) |
| **kNN manifold** | **24 / 24** (closed) |

A natural near-one-hot belief still passes (knn-dist 0.05 < thr 0.13). So the sign-epistasis is now *actually gated* in
deployment — it stands as design justification for the signed cross-term, and the beliefs it was measured on are flagged
off-manifold + demoted, exactly as the honesty of the C-5 caveat requires. **Honest disclosure (verifier):** held-out
false-positive rate on natural beliefs = **1.2%** (matches the q=0.99 design and is tighter than the box governor's
7.5%); "24/24 caught" is *near-inevitable* by construction (the C-5 corners have entropy 0.76–1.39 vs natural max ~0.84,
i.e. diffuse by design), and the gate is fairly aggressive — it trips at `[0.85,0.05,0.05,0.05]` (only ~20% toward
uniform). Defensible given the 1.2% FPR, but the aggressiveness is a tunable, not a free lunch.

## Debt 2 — calibrated per-pair |ε| bound: **CLOSED**
The ledger's `eps_bound=0.5` was C-5's *materiality threshold*, not the observed magnitude. Calibrated from the C-5
cells (`c5_debts.py`): global |ε_logit| **p95 = 6.61, max = 8.62**; per-venue-pair p95 in **[5.5, 7.2]**. So the 0.5
stand-in was ~13× too small — `N_live ≤ K` was doing the real limiting. `writ_ladder.py`'s default is updated to the
calibrated **6.6** (global p95), with the per-pair table (`c5_debts_report.json`) as the recommended charged bound and
`max` as the hard-refuse trip. Honest note: the *magnitude* is now conservative, not a placeholder. **Caveat (verifier):**
`|ε_logit|` is a dimensionless **logit-space** cross-term, but the ledger charges it (`eps_bound·τ·N_live`) into the same
additive authority units as `|α|·τ` — an **unvalidated units bridge** (a worst-case bookkeeping convention, not a
calibrated equivalence). The number is right; its use as a `D`-charge inherits that unstated assumption.

## Debt 3 — α_machine calibration: **CLOSED (corrected after adversarial review)**
HL5 named α_machine — the machine-rate false-accept of the firewall — as the loop's top open risk. Measured on the
corner (`c_alpha_machine.py`): a writ claim = "in belief region R the write makes the policy take action a_R"; the
automatable C1/C2 certifier accepts iff the policy takes a_R in ≥τ=0.67 of the region's probes.

**Correction (verifier caught a real conflation):** the first pass scored claims against the *belief-MDP optimum*, so its
single "false-accept" was actually a **true behavioral claim** (the PPO policy genuinely takes that action in 80% of
probes) that merely disagreed with the optimum — conflating "wrong about the policy's EFFECT" with "wrong about
OPTIMALITY." Refixed to score against the policy's OWN modal action:

- **α_machine (behavioral, correctly framed) = 0.0** — the certifier accepts a wrong claim about *what the write
  actually does* (a non-modal action described as the write's effect) **0 / 48** times. By the rule of three, that's a
  95% upper bound of ≈ **6%**, not a proven 0.
- behavioral **power on true modal claims = 70.8%** (some regions are genuinely multi-modal so even the modal action
  misses the 0.67 floor — independent conservativeness, not entangled with optimality).
- **Secondary** — the *optimality*-claim false-accept (null = non-belief-MDP-optimal, which ALSO fires when the policy
  is merely sub-optimal) = **0.0208**; a wider, different quantity, reported separately, **not** α_machine.

**Honest scope:** this is a **K=2, C1/C2-core-only, n=48 PROXY** for HL5's machine-rate false-accept (which ultimately
spans the full C1..C6 firewall) — a first estimate / upper bound, **not** "the number HL5 left unquantified." Implication
stands: size `D_cap` / `N_live ≤ K` to `proposal_rate × α_machine` against the human-rate anchor (the two same-session
retractions).

## Debt 4 — the VoI>0 substrate: **honestly scoped — the gate, not a fake edge**
The program's B4-REAL closure established the regime is **priced** on competitive markets (Glosten-Milgrom zero-profit):
belief-VoI = 0 on daily and on the accessible intraday crypto. We cannot manufacture a VoI>0 substrate. The constructive
deliverable is the **VoI gate** (`voi_gate.py`) — the blueprint's "environment-selected objective by measured
belief-VoI": deploy the belief-writing surface on **capital only where VoI > ε**; else transparency/monitoring only. The
gate **discriminates correctly**:

| substrate | belief-VoI | gate |
|---|---|---|
| Series-G polygon | **5.92** (+282% of the belief-blind value; aware vs blind belief-MDP optimum) | **OPEN** |
| real crypto intraday (B4-REAL) | **0.0** (regime priced) | **CLOSED** |
| CN A-shares | untested — no usable L5 data yet (recorder started post-close Fri; Sat closed) | UNTESTED (owed) |

So the born-legible surface is **validated AND correctly fenced**: it acts on capital only where the regime is not
already priced. **No accessible real substrate currently clears the gate** — the standing open bet is a genuine VoI>0
execution task (queue/rebate/latency microstructure, or CN A-shares once the L5 recorder has accumulated ~4–8 weeks).
Until one does, CRYSTAL-1 is a **transparency/interpretability object on real markets** — consistent with the pivoted
north star (CrystalScore, not alpha). This is the honest statement of the program's central open question, now
operationalized as a runnable gate rather than left as a caveat. **Caveat (verifier):** the polygon VoI is genuinely
apples-to-apples (both are optimal-policy values at the identical start state), and crypto VoI=0 is the *measured* B4
number — but the polygon side is **positive by construction** (PRIMARY_ENRICHED was hand-tuned so the Phase-0 gate's
`material` margin passes). So "the gate discriminates" is validated across **n=1 engineered-positive vs n=1 real-negative**
— it shows the gate *can* open and *does* close on the real book, not that it separates comparable real candidates. That
stronger claim needs a population of real substrates, which is exactly the owed VoI>0 hunt.

## Net
Three of four debts are closed with running code + measured numbers (joint governor gates 24/24 corners at a 1.2%
held-out FPR; per-pair |ε| calibrated to p95≈6.6; behavioral α_machine = 0.0 / ≤~6% CI, with the optimality-claim
false-accept 0.0208 reported separately) — all as **K=2/G-polygon, C1/C2-core proxies**, honestly scoped, not
whole-firewall quantities. The fourth is honestly converted from "find a VoI>0 substrate" (which the program shows is not
accessible now) into the **decision rule** that fences the surface to where it would pay — validated to *open* on the
polygon and *close* on the real market (n=1 vs n=1). The controllability program's remaining question is singular and
explicit: a real VoI>0 substrate. Everything else in the C-1..C-6 ladder is built, enforced, and adversarially checked —
with every over-claim caught by the adversarial passes folded back in.
