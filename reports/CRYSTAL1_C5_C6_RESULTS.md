# CRYSTAL-1 controllability — M6 warm-start + C-5 + C-6 executed (the rest of the ladder)

Completes the [CRYSTAL1_CONTROLLABILITY_PLAN.md](CRYSTAL1_CONTROLLABILITY_PLAN.md) ladder after
[C-1/C-4](CRYSTAL1_C1_C4_RESULTS.md) and [M6/C-2/C-3](CRYSTAL1_M6_C2_C3_RESULTS.md). New code:
`src/crystal/{governor,writ_ladder}.py`, `heuristic_agent_r6c/contracts/crystal1_knob_registry.yaml`,
`interpretability/c5_sign_epistasis.py`. All on the Series-G polygon (K=2 corner + G-regime family); teeth only where VoI>0.

## M6 warm-start → robust parity (BC + crisp PPO) — **PASS (full parity)**
The borderline M6 (jointly-trained soft tree, gap −0.27, `M6_PASS=false`) is resolved by a **BC warm-start** (the
TB_gen_08 lesson): behavior-clone the soft tree to the corner MLP's actions on `[belief,inv,time]` first, then crisp
PPO fine-tune (β=3, 500k). Result (`interpretability/crystal1_m6_softtree.py ... warmstart`):

| | return | gap vs MLP | parity | burst-leak (action dist) | C1 (C-2, vs belief-MDP optimum) | dose |
|---|---|---|---|---|---|---|
| **warm-started M6** | **7.692** (SEM 0.39) | **−0.018** | **PARITY = TRUE** (paired p≈0.84) | **TV = 0.0** (structural) | **0.835 → C1 PROVED** | monotone, sharp, thr 0.26 |

Progression: β=1 gap −2.57 → β=3 −0.27 → **warm-start −0.018 (parity)**. So the born-legible soft-tree head now
**simultaneously** (i) reaches MLP return parity, (ii) closes the C-1 burst leak structurally in the action distribution
(TV=0, burst excluded by construction), (iii) is **provably causal** against the exogenous belief-MDP optimum (0.835,
*higher* than the corner MLP's 0.80), and (iv) is a legible 8-leaf tree with a near-hard monotone command dose
(threshold 0.26). This supersedes the "borderline / 0.355-agreement" M6 in [M6/C-2/C-3](CRYSTAL1_M6_C2_C3_RESULTS.md):
warm-start closes both the return gap and the optimal-action gap. **The C-4 pivot's structural-legibility bet is met.**

## C-5 — K-vocabulary dial + governor + the owed ≥2-belief-dim SIGN-EPISTASIS run

### (a) SIGN-EPISTASIS confirmed on a genuine ≥2-belief-dim surface — updates the IV-10 debt
The corner test (belief × **inventory**, 1-belief-dim) found the cross-term non-additive but **sign-stable** (+68/−0),
so the scary OOS-reversal machinery stayed `[PLAUS]`. C-5 runs the real test on **belief × belief**: the G-regime
family env (`RegimeRotationEnv(G=4)`, a 4-simplex belief), model `src/crystal/_b2/family_G4_s1.zip`. 2-factor factorial
(mass on venue *i* × mass on venue *j*), 6 venue-pairs × 12 contexts × 3 outcomes = 216 interaction cells.

| outcome | median \|ε_logit\| | material (>0.5) | signs (+/−) | both signs? |
|---|---|---|---|---|
| own venue *i* (direct competition) | 4.18 | 94% | +0 / −68 | no (sign-stable, softmax competition) |
| a **third** venue *k* | 1.48 | 79% | **+41 / −16** | **YES** |
| **ABSTAIN** | 2.44 | 88% | **+38 / −25** | **YES** |

- **NON-ADDITIVITY: confirmed and large** (median \|ε_logit\| 2.66 overall, 87% material) — bigger than the corner's 1.84.
- **SIGN-EPISTASIS: CONFIRMED** — on outcomes not mechanically pinned by direct competition (a third venue, ABSTAIN),
  the cross-term takes **both signs**. So two belief-writes CAN interact with a sign that reverses across pairs/context
  — the OOS-reversal IV-10 warned about is **real on genuine multi-belief-dim writes**; the corner's belief×inventory
  simply could not surface it (1-belief-dim).
- **Not a softmax artifact (adversarially verified):** the sign-flip **survives on the raw pre-softmax logits**
  (third-venue +47/−17, abstain +38/−25) — it lives in the network's own command→logit map, not in mass-reshuffling.
  The own-venue −68/0 sign-stability is the softmax-competition artifact (raw logits are mixed +24/−43), correctly
  separated out.
- **Design update (scoped honestly):** IV-10's heavier machinery — the **co-activation cap, forbidden-pair registry,
  and worst-case signed cross-term** — is now justified as **design** by a *measured* sign-flip, so it moves from
  `[PLAUS]` toward **`[SBC — measured, but see the manifold caveat]`**. Per-command certificates provably do not compose;
  the combination certificate must carry a **signed, non-commutative** cross-term (the ledger uses a worst-case \|ε\|
  bound because the sign is unpredictable). *This updates the earlier corner-only "sign-stable" conclusion.*
- **REQUIRED honest caveat (verifier):** the four factorial corners are **jointly OFF the natural belief manifold** —
  natural family beliefs are near-one-hot (entropy p95≈0.70) whereas the forced "both"/"neither" corners are diffuse;
  requiring all four corners within L1≤0.30 of a visited belief leaves **zero cells**. So the sign-epistasis is
  demonstrated in a belief region the filter essentially never produces. It is a valid **design justification** for a
  signed cross-term; it is **not** "measured on the deployed manifold." And the box `BeliefGovernor` is **per-coordinate**
  (marginal box ≈ [0,1]⁴), so it does **not** gate these *joint* off-manifold beliefs — the earlier "the governor handles
  that" is corrected: a *joint*-manifold projector (e.g. hull/Mahalanobis) is the owed next step. Single trained family policy.

### (b) Governor — the runtime envelope-enforcement layer (`src/crystal/governor.py`)
The literature review (Plan B) flagged that CRYSTAL-1's envelope was a *soft probe restriction*, not a hard projector.
`BeliefGovernor` fixes that: a command **inside** the visited envelope passes **byte-identical** (no authority lost); a
command **outside** is **projected** to the nearest on-envelope belief (closed-form box-CBF clamp-and-renormalize on the
simplex) and returns a **GUARANTEE_DELTA** annunciation (authority *demotion*, the AF447 anti-pattern of annunciating the
lost guarantee); chronic boundary contact is **metered** into a cumulative budget (envelope-surfing tripwire). Selftest
passes: on-envelope identical, off-envelope projected+demoted+annunciated, surfing metered. **Caveat (verifier):** it is
a **per-coordinate** box projector (marginal envelope), so it does **not** enforce the *joint* belief manifold — exactly
the joint off-manifold region C-5's sign-epistasis lives in. A joint projector (convex-hull / Mahalanobis on visited
beliefs) is the owed upgrade before that evidence can gate deployed writes.

### (c) K-vocabulary dial (GROW_K)
The belief vocabulary size K (= the family env's G) is the typed architectural dial. That growing K reproduces the
**C\*≈K bend** and that a new vertex must be *load-bearing* (HC-1 ablation hurts + belief-N7) before exposure is the
**ironclad multi-seed B2 result (C4)** — reused, not re-run. In the registry `K_VOCABULARY` is engineering-fenced (a T1
retrain), gated on the load-bearing test.

## C-6 — R6c IP migration + C0..C6 writ ladder + cumulative-authority ledger

### (a) Born-legible knob registry (`heuristic_agent_r6c/contracts/crystal1_knob_registry.yaml`)
R6c's ~214-knob surface migrated onto the belief-write command surface: **6 agent-facing levers** (SET_BELIEF,
EXPOSURE_MODE, GROUP_CONCENTRATION_CAP made a **hard** cap — fixing R6c's soft-cap-that-didn't-bind W8, ABSTAIN_FLOOR as
a first-class verb, K_VOCABULARY dial, DRAWDOWN_BUDGET), each with a machine-checkable **guarantee** (post-condition, not
a mode name), authority tier, rate/dwell limit, trajectory budget, min writ-rung, reversibility class R0–R4, and a
version pin; the rest **engineering-fenced** (reward/PPO/filter/formation knobs training-time only; turnover_cap /
lambda_cash / de-conc fenced as near-no-ops on the near-EW book). Six cross-cutting invariants live *outside* the
modifiable surface. Parses clean.

### (b) C0..C6 writ ladder + signed cumulative-authority ledger (`src/crystal/writ_ladder.py`)
`WritCertificate` walks C0 addressable → C1 causal (wired to C-2's 0.80 proof) → C2 dose (C-2, refusal-rate 0) → C3
envelope (the governor) → C4 side-effect (ghost + the C-5 worst-case cross-term) → C5 frozen gate → C6 version-pin; only
a **fully** certified writ may act on capital, and any **policy/cfbank bump voids it back to C1**. `CumulativeAuthority
Ledger` is signed per-principal: `D = Σ|α|·τ + worst-case Σ|ε_ij|·τ` (the C-5-confirmed cross-term), reset only by
re-cert vs the anchor. Selftest demonstrates the two attacks are caught: **salami-slicing** (twelve individually-tiny
α=0.25 writs sum to *exactly* `D_cap=3.0`; the **13th** breaches at 3.25 — cumulative, no refund on release) and
**co-activation** (`N_live ≤ K` blocks the (K+1)-th simultaneous belief-write, because C-5 showed the cross-term is
unbounded beyond direct competition). Honest note (verifier): the salami demo exercises only the **main term** (writs are
released before the next issues, so the cross-term is 0 there — it is charged only in the co-activation demo); and
`eps_bound=0.5` is C-5's *materiality threshold*, not the observed max \|ε\| (which reaches ~8), so it is a per-unit
worst-case stand-in that is safe **only because `N_live ≤ K` caps the pair count** — not a bound conservative w.r.t.
observed magnitudes. A calibrated per-pair \|ε\| bound is the owed refinement.

## What the full ladder establishes (honest ledger)
On the polygon, CRYSTAL-1's born-legible controllability is now demonstrated **and enforced** end-to-end: named causal
writes (C-1/C-2) → on-simplex dose with zero refusals (C-2) → a structural legible head (M6) → grounded-command
lie-detector (C-3) → hard envelope projection + demotion (governor) → measured sign-epistasis forcing a signed
combination certificate (C-5) → the C0–C6 writ ladder + per-principal cumulative-authority ledger that catch
salami-slicing and over-co-activation (C-6). **Top open risks unchanged:** α_machine (the machine-rate false-accept
constant) is still uncalibrated — `N_live ≤ K` and `D_cap` are the load-bearing limiters meanwhile; everything is on the
**K=2/G-regime polygon** where the corner is real, and **has teeth only where VoI>0** — the born-legible surface still
must survive a real VoI>0 execution task, the program's standing open bet.
