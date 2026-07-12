# The experience/ DeepSearch corpus — critical mining for CRYSTAL-1 controllability + hypotheses

**Plan C of the CRYSTAL-1 controllability set (A–Д).** Lens = CONTROLLABILITY (steering, legibility, authority,
reversibility, composition, command certification), NOT alpha. Grounded in `reports/_crystal_ctrl/ANCHORS.md`; do
not re-litigate its settled facts (R6c > CRYSTAL-1 on controllability today; "regime is priced" / Glosten-Milgrom
config-D equilibrium; B2 uniqueness-retraction; B5 reward-shaping-for-legibility falsified → needs a STRUCTURAL
mechanism; corner competed away off-polygon). Evidence tags: **[Est]/[Strong-Ctx]/[Plaus]/[Spec]**. External
literature tagged **Tier-A** (named author+year/method) vs **Tier-B** ([class]); nothing fabricated.

---

## 0. What I sampled vs skipped (honest funnel over 617 files)

**READ IN FULL (the load-bearing set, 8 docx + all 5 HL5 digests + 1 CHRL file):**
- *Designing the Counterfactual Market* — `DCM: Brock–Hommes`, `Kyle 1985`, `Режимно-переключающийся POMDP`
  (regime-switch LOB), `Minority/Majority Game`, `Santa Fe`, `El Farol`, `POMDP-рынок`. These are the environment
  candidates that expose a **named, writable belief/command surface**.
- *Behavioral-Complexity Zoo* — `термостат/PID`, `LLM next-token`, `правило Тейлора (Taylor-rule CB)`,
  `market-maker`. These fix the two ends of the **controllability ceiling**.
- HL5 already-digested: `reports/_hl5_digest/{K,IV,GV,HLX,TB}_digest.md` (reused, not reinvented).
- `experience/WORLD OF BEST TRADING BOT/CHRL_C_INNOVATIVE_CHRL_APPROACH.md` (whole-system control skeleton).

**SAMPLED (coordinate + vocab + ceiling lines only, via grep):** `DCM: Glosten–Milgrom` (already settled →
config-D), `Chiarella`, `Lux–Marchesi`, `ABIDES`, `Gode–Sunder ZI`; `ZOO: boids, ant_colony, foraging,
gene/immune, chess/go, poker, Atari/MuJoCo, deep-RL/StarCraft, delta-hedge, optimal-control(LQR/MPC)`. I pulled
their (behavioral_complexity, interpretability) coordinates and Puzzle-Piece verdicts; that is sufficient to place
the ceiling, and re-reading each 40k-char report would not move the argument.

**SKIPPED ENTIRELY:** `experience/{Economic World Picture, Technical Trading Bot Picture, Best Interpretable
model}` and the non-CHRL half of *WORLD OF BEST TRADING BOT* (FinGPT/IR world-pictures) — these are alpha/economic
world-pictures, orthogonal to the controllability question; the *Series K / GV / HLX / TB* folders under
`experience/` are the raw inputs already fully digested in `reports/_hl5_digest/` (I used the digests, not the raw
docx, per the brief's "reuse, don't reinvent"). The two duplicate DCM docx (second Glosten–Milgrom, second
regime-switch) were byte-identical to their twins.

**Why this sample is strategic, not lazy:** the controllability question has two axes — (a) *which polygon exposes
a named latent an operator can write* (answered by the DCM environment reports) and (b) *where legible per-decision
control stops being possible* (answered by the ZOO ceiling). Everything read maps to one of these; everything
skipped maps to neither.

---

## 1. MINED-ATOMS TABLE (source file → controllability claim → how it informs CRYSTAL-1)

Each row cites an exact local file under `experience/`. CRYSTAL-1 code refs are real
(`src/crystal/belief_filter.py`, `interpretability/certified_battery_v2.py`, `src/series_g/regime_pomdp.py`).

| # | Source file (experience/…) | Controllability-relevant claim (evidence) | How it informs CRYSTAL-1 |
|---|---|---|---|
| **A1** | `POMDP-рынок как frontier-бенчмарк…` | **Macrì et al.: a POMDP market-maker whose input is an EXPLICIT posterior over latent regimes beats — AND is more interpretable than — the same policy fed a raw recurrent embedding.** "interpretability ≈ 4.0/5 IF the benchmark forces explicit belief-state variables, not embeddings." **[Est, Tier-A — toy-latent scope; extrapolation to markets is [SbC] until H1 runs; see §3.2/§5]** | This is the strongest *external* validation of CRYSTAL-1's core bet: `NeuralBayesFilter`'s K-simplex belief `b` (`src/crystal/belief_filter.py`) is not just legible, it is *performance-competitive*. The choice "posterior-as-obs vs RNN-latent" is the R6c-vs-CRYSTAL-1 controllability gap, and the literature says posterior wins on BOTH axes. |
| **A2** | `POMDP-рынок…` | **Kinathil et al. 2016: a market-making POMDP with closed-form symbolic-DP optimal policy** ("Stage-0 gold policy"); optimum is strongly inventory-dependent. **[Est, Tier-A]** | Gives the battery a **known-optimal reference** the current polygon lacks. `interpretability/certified_battery_v2.py` gate-5 (J-intervention fidelity ≥0.67) currently compares to *learned* behavior; a symbolic-DP oracle lets us score belief-write fidelity against ground truth, not against a co-trained proxy. |
| **A3** | `Режимно-переключающийся POMDP` (regime-switch LOB) | **5-concept writable belief vocabulary: {toxicity-belief, liquidity/resilience-belief, queue-advantage, inventory-pressure, time-pressure}; strong policy = 4–6 macro-rules with HYSTERETIC thresholds (entry≠exit).** Puzzle title "Belief-Driven Toxicity LOB", coord (4.2, 4.0). **[Strong-Ctx, Tier-B synthesis]** | Names the successor polygon beyond the 2-regime BENIGN/TOXIC toy (`src/series_g/regime_pomdp.py`). Each concept is a *named scalar an operator could write* — the K-simplex generalized to a small typed panel. Hysteresis (entry≠exit thresholds) is a control primitive the belief-write API must expose (maps to IV-digest dwell/hysteresis §2D). |
| **A4** | `Режимно-переключающийся POMDP`; `Brock–Hommes…` | **Proper-scoring-rule belief-elicitation side channel**: agent emits its posterior-over-regimes for a tiny *strictly proper* bonus that does not materially change the objective → inference becomes *auditable*, and you can test whether an interp method recovers the *right hidden concept* vs merely the right action. **[Strong-Ctx / Spec, Tier-A method (Gneiting–Raftery scoring rules)]** | A NEW battery gate: force CRYSTAL-1 to *report* `b`, score its calibration on the polygon's true hidden regime. Converts "faithfulness" from a post-hoc probe into a *built-in, certifiable* signal. Directly reusable — `NeuralBayesFilter.forward` already produces `b`. |
| **A5** | `Brock–Hommes adaptive beliefs…` | **Named ecological command surface: {fundamental-gap, trend-strength, belief-ecology (who wins: stabilizers vs extrapolators), switch-hazard, friction-budget}; strong policy = {ride, fade, abstain}, action SIGN flips endogenously at the SAME price gap depending on ecology.** coord (4.2, 4.0). **[Est model, Tier-A Brock–Hommes 1998]** | The richest *human-named* multi-latent surface in the corpus. "Sign flips at same observable given ecology" is exactly regime-conditioned control — an operator writing `ecology=extrapolators-winning` must flip the policy's action sign. Candidate CRYSTAL-2 polygon where K>2 named regimes each carry a writable semantic. |
| **A6** | `Kyle 1985…` | **Insider optimum = ~1-D feedback rule over {mispricing, residual-private-info, depth/λ, concealment, urgency}; pure Kyle is UNDER-complex (2.4/5) — high interpretability partly BECAUSE under-complex.** **[Est theorem, Tier-A Kyle 1985]** | A caution: a maximally-legible belief surface can be *trivially* legible because the task is too easy. CRYSTAL-1's B4 "regime is priced / VoI=0 on daily" (`interpretability/crystal1_b4_bridge.py`) is the same shape — legibility that is real but not *load-bearing*. Use Kyle as the "right-anchor" calibration: CRYSTAL-1 must be legible HERE, but passing Kyle proves nothing about the hard end. |
| **A7** | `Minority Game и Majority Game…` | **Counterintuitive [Est]: adding a little agent REFLEXIVITY (accounting for own market impact) DECREASES behavioral chaos and IMPROVES both individual and collective efficiency (Marsili–Challet–Zecchina 1999).** Also: abstention/no-trade must be a **first-class action**. **[Est, Tier-A]** | Against the naive "more strategic reasoning ⇒ less interpretable." A CRYSTAL-1 that models its own impact (capacity-aware belief) may become *simpler and more steerable*, not harder. Argues for an explicit `own-impact` belief coordinate + a first-class ABSTAIN action in the writable grammar. |
| **A8** | `Santa Fe Artificial Stock Market…` | **Explicit timing makes strict no-look-ahead NATURAL not a patch** (dividend→expectations→clearing→next-dividend); "rich psychological regime" only appears at realistic learning speed; interpretable state-bits already ~ ready-made concept features. **[Est, Tier-A Arthur–Holland–LeBaron–Palmer–Tayler]** | Design constraint for the CRYSTAL-1 polygon: bake leakage-safety into the *event ordering*, not into a downstream guard. SFI descriptor-bits are the archetype for a **named-bit observation contract** (cf. L0 `RoleContractError` in `src/crystal/universe.py`). |
| **A9** | `El Farol…` (Rand–Stonedahl 2010; Chmura–Pitz 2006) | **"Smarter" is not always better: more computation can LOWER welfare + raise oscillation; a human-legible marker of a GOOD policy is "fewer needless switches" (switch-count ↓ correlates with payoff ↑).** **[Est, Tier-A]** | A directly measurable **legibility-of-good-behavior** signal: excessive belief-write churn is a *pathology*, not sophistication. Feeds a battery diagnostic "steering with fewer, larger, hysteretic writes beats high-frequency writes" — ties to IV-digest chatter budget `D_hf`. |
| **A10** | `правило Тейлора (Taylor-rule CB)…` | **Named sparse legible controller {inflation-gap, output-gap, neutral-rate, lagged-rate}; BUT Orphanides: algebraic simplicity ≠ real-time behavioral simplicity (revised-data fit is pseudo-simplicity); AND smoother policy = MORE behavioral complexity (lower entropy, longer memory).** coord (low complexity, high interp). **[Est, Tier-A Taylor 1993 / Orphanides 2001 / Rudebusch 2002]** | Two warnings for the belief-write claim: (1) a legible-looking scalar write can hide real-time non-identifiability — CRYSTAL-1 must certify writes on *point-in-time* belief, not smoothed (mirrors IV C0 addressability on the *current checkpoint hash*). (2) Rate-smoothing the belief-write RAISES memory/complexity — the "envelope-scoped, visited-states-only" write must be metered for cumulative memory it induces. |
| **A11** | `market-maker LOB…` (Kwan–Philip 2025 [project-corpus-sourced; unverified against public literature]; Brodu decisional-states) | **The CANCEL/HOLD option carries ~19% of a limit order's value — much control-value lives in "when to withdraw," not "where to quote." "Decisional states" = causal states merged by utility-equivalence → named human concepts like "front-of-queue-under-adverse-pressure."** **[Strong-Ctx, Tier-A]** | (1) CRYSTAL-1's writable grammar must include a WITHDRAW/abstain verb with its own certificate, not just a directional write. (2) **Decisional states are the principled bridge from behavioral complexity to a small named vocabulary** — a concrete recipe for CRYSTAL-2's codebook: cluster belief-states by *action-equivalence*, not by geometry (fixes the B2-retracted "uniqueness-tracks-fidelity"). |
| **A12** | `LLM next-token…`; `термостат/PID…` | **CEILING atoms: (i) determinism does NOT imply interpretability (transformers realize automata via non-human "shortcuts"); (ii) ε-transducer/causal states are minimal-predictive states, NOT human-readable concepts; (iii) thermostat = 1 bit of hysteresis, ~(0.04, 0.97).** **[Est, Tier-A Crutchfield/Barnett; Liu et al. "shortcuts to automata"]** | Hard bound on the belief-write program: *a low-entropy / near-deterministic CRYSTAL-1 is NOT automatically legible*, and its learned latent codes are not automatically operator-nameable. Legibility must be *engineered as a structural constraint on the state* (the born-legible thesis), which is precisely why B5 reward-shaping-for-legibility FAILED — you cannot shape your way to concept-nameability; you must architect it. |
| **A13** | `CHRL_C_INNOVATIVE_CHRL_APPROACH.md` | **Whole-system control skeleton: action grammar `how much risk → through which groups → into which stocks → at what pace`; invariant — "learned policies may move INSIDE the skeleton but cannot change its accounting, logging, validation, or capital-permission rules."** **[Proj, Est in-house]** | This is R6c's controllability advantage stated crisply, and the *target contract* CRYSTAL-1 must match before it can win: the belief-write must sit INSIDE a fixed skeleton whose accounting the policy cannot rewrite (= GV "proposer never self-classifies"; = IV "unregistered write path is a certification failure"). |
| **A14** | `Glosten–Milgrom…` (sampled) | Bid–ask + adverse selection arise **endogenously** as the equilibrium response to information asymmetry → "regime is priced." **[Est, Tier-A]** | Confirms the ANCHORS settled fact (config-D equilibrium, VoI=0 daily). Bounds the *claim* CRYSTAL-1 may make: on a competitive book the belief-write steers *cash-timing/legibility*, not un-priced alpha (consistent with `reports/B4_REAL_INTRADAY_CLOSURE.md`). |
| **A15** | `Lux–Marchesi…`; `Santa Fe…` (sampled) | **Named "high-complexity / LOW-policy-interpretability" negative anchor**: herding/transition dynamics reproduce stylized facts but the strong policy has an impoverished, non-distillable vocabulary. **[Est, Tier-A Lux–Marchesi 1999]** | The polygon a CRYSTAL-1 controllability claim must AVOID — high behavioral complexity with no compact named surface is exactly where the born-legible thesis has no purchase. Use as the negative control in any "belief-write is legible" test. |

---

## 2. THE CONTROLLABILITY CEILING (what the ZOO ruler says about where legible control stops)

Placing every ZOO agent on the (behavioral_complexity, interpretability) plane (coordinates lifted verbatim from
each report's Structured Summary; all authors' own **[Plaus]/[Strong-Ctx]** self-estimates, one shared 0–1 ruler).
The convergence count below is **14 Zoo reports + DCM anchors** (the plot mixes in DCM-sourced points such as
market-maker and Taylor-CB); the "14-report" figure in §3.1/§5 refers to the Zoo reports alone.

```
 interp
 1.0 | thermostat(0.04,0.97)
     |   PID(0.16,0.88)      boids(0.30,0.80)
 0.8 |     LQR/MPC(low,hi)   Taylor-CB(low,hi)
     |          foraging(0.40,0.70)
 0.6 |               GRN(0.55,0.63)   market-maker(0.60,~0.55)
     | ------------- CONTROLLABILITY CEILING (interp ≈ 0.40) ---------------
 0.4 |   Atari/MuJoCo(0.38,0.40)  chess(0.72,0.38)  go(0.90,0.35)
     |        immune(0.68,0.39)
 0.2 |          poker(0.78,0.22)  StarCraft/OpenAI-Five(0.78,0.22)  LLM(0.75,0.25)
 0.0 +--------------------------------------------------------------------→ behavioral complexity
       0.0        0.2        0.4        0.6        0.8        1.0
```

**Reading (the load-bearing finding):**
1. **There is a hard anti-correlation, and a knee.** Below behavioral-complexity ≈ 0.45 (thermostat→PID→LQR/MPC→
   Taylor-CB→boids→foraging), interpretability stays ≥0.7: **per-decision legible control is possible** — you can
   name the state and write a scalar with a predictable effect. Above ≈ 0.65 (poker, chess, go, StarCraft, LLM,
   immune), interpretability collapses below 0.40: **legible per-decision control STOPS**; only aggregate /
   statistical / envelope control survives. **[Strong-Ctx]**
2. **The ceiling is not entropy.** `LLM next-token` (0.75, 0.25) and `термостат` (0.04, 0.97) both make the same
   point from opposite ends: *determinism ≠ interpretability*, and the killer is **long predictive memory + many
   regimes that no small concept-set compresses** (excess entropy / statistical complexity), NOT action randomness.
   A policy can be near-deterministic and still uncontrollable at the decision level. **[Est]**
3. **Where CRYSTAL-1 and R6c actually sit.** R6c's L0 log is "low-but-STRUCTURED" (memory index: L0 ruler); its
   64-d latent steering is a code-forcing intervention (`reports/firewall_upgrade/r6c_code_control_demo.py`) — it
   lives *near the market-maker band* (~0.55–0.6 complexity, ~0.5 interp) where control is possible but only
   through un-named codes. CRYSTAL-1's bet is to move LEFT-AND-UP of R6c on interpretability at equal complexity by
   making the state a **named posterior** (A1) rather than a 64-d latent — i.e. to sit with the market-maker/GRN
   band on complexity but with Taylor-CB-grade interpretability.
4. **The ceiling caps the CLAIM, not just the method.** The corpus says: any CRYSTAL-1 controllability result is
   only meaningful *below the knee*. On a Lux–Marchesi/StarCraft-class polygon (A15) the born-legible thesis has no
   room. So the controllability program should deliberately target the **market-maker / GRN / regime-switch-LOB
   band (complexity 0.5–0.65, interp target ≥0.6)** — high enough that legibility is non-trivial (unlike Kyle/
   thermostat), low enough that a named surface still exists (unlike poker/LLM). **[Strong-Ctx]**

---

## 3. CRITIQUE — what the corpus OVER-CLAIMS or leaves UNPROVEN for controllability

1. **The (complexity, interpretability) coordinates are self-graded expert inferences, not measurements.** Every
   ZOO report tags its exact numbers **[Plaus]** and admits "structured estimates, not a benchmark on one shared
   dataset" (`термостат/PID` Quality Gate; `LLM` interpretability_point **[Plaus]**). The *ordering* is robust and
   converges across 14 independent reports; the *cardinal coordinates and the knee at 0.40* are soft. **Do not port
   these as calibrated CrystalScore anchors** — port the ordering and the ceiling *shape*. **[critique: Strong-Ctx]**

2. **"Belief posterior is interpretable" is proven on TOY latents, extrapolated to markets.** A1 (Macrì et al.) is
   real and strong, but the finite-regime toy where posterior-input wins is far simpler than a live book. The
   corpus itself hedges: `POMDP-рынок` marks the market coordinate **[moderate]** and warns the belief-variable
   advantage assumes a *well-specified* finite latent. CRYSTAL-1's own B4 already shows the daily-proxy latent
   carries VoI=0. **Unproven:** that a *learned* `NeuralBayesFilter` posterior over a *mis-specified* market regime
   stays both faithful and writable. This is the single biggest gap. **[critique: Established gap]**

3. **The "controllable named surface" is asserted from the MODEL's vocabulary, not the OPTIMAL POLICY's.**
   Brock–Hommes/regime-LOB list human words (gap, trend, ecology, toxicity) that name the *environment's* state.
   The corpus repeatedly cautions (`market-maker`: Brodu; `Kyle`: "high interp partly from under-complexity") that
   the *policy's* minimal states (ε-transducer / causal states) are **NOT** guaranteed to align with those words.
   A12 is explicit: minimal-predictive states are not human concepts. **Over-claim:** that because the polygon has
   named regimes, a policy trained on it is operator-writable in those names. The regime-LOB report's own
   proper-scoring side-channel (A4) exists *precisely because* this alignment is not free — it has to be forced and
   certified. **[critique: Established]**

4. **Costs/impact/leakage are treated as "honesty add-ons," but they change the control surface itself.**
   Anufriev–Panchenko (in `Brock–Hommes`) show the trading protocol *moves the bifurcation thresholds*: the same
   belief-write has a *different* effect under Walrasian vs order-book clearing. The corpus flags this as the
   "architecture-invariance" falsifier but never resolves it. **Unproven:** that a certified belief-write's
   dose–response (`interpretability/certified_battery_v2.py` steerability) is *invariant* to the execution wrapper.
   If it is not, every write certificate is protocol-local — a composition hazard the IV-digest §5 predicts but the
   DCM corpus does not test. **[critique: Established gap]**

5. **The corpus offers no CUMULATIVE-authority story for belief writes.** It gives per-decision legibility and
   per-episode complexity, but the HL5 digests (IV §4, TB §6, GV/K/HLX) all converge that *per-step safety ≠
   trajectory safety* and that the field ships NO trajectory-budget. The DCM/ZOO reports never ask "what does a
   thousand small legible belief-writes integrate to?" — El Farol's switch-count (A9) is the closest hint. **This
   is not the corpus's job, but the plan must not let the corpus's per-step legibility masquerade as
   controllability**; controllability = per-step legibility × trajectory budget × reversibility × composition, and
   the corpus only supplies the first factor. **[critique: Strong-Ctx]**

6. **Reflexivity-simplifies (A7) is real but narrow.** Marsili–Challet–Zecchina is a minority-game exact result;
   whether "modeling own impact simplifies the policy" survives on a competitive book (where impact is priced into
   quotes, A14) is untested. Treat as **[Plaus]**, not a design guarantee.

---

## 4. CREATIVE HYPOTHESES (≥5, each falsifiable on the polygon/battery)

Each: claim → mechanism → cheapest test on existing tooling → null → evidence rung → firewall/battery route. All
routes reuse `interpretability/certified_battery_v2.py`, `src/crystal/belief_filter.py`,
`src/series_g/regime_pomdp.py`, and the IV C0–C6 ladder (`reports/_hl5_digest/IV_digest.md`).

### H1 — Posterior-input dominates latent-input on BOTH steerability and fidelity (the A1 replication)
- **Claim:** a CRYSTAL-1 whose obs carries the *explicit K-simplex posterior* `b` beats an otherwise-identical twin
  fed a same-dimension *learned recurrent latent* on (i) return AND (ii) belief-write dose–response monotonicity —
  reproducing Macrì et al. (A1) in-house. **[Plaus → would become Strong-Ctx if it holds]**
- **Mechanism:** posterior is a sufficient statistic *named* to the operator; the RNN latent is sufficient but
  un-named, so writes to it are code-forcing (R6c-style), not semantic.
- **Cheapest test:** on `src/series_g/regime_pomdp.py` (2-regime), train the posterior-obs policy (already exists,
  `interpretability/crystal1_b1.py` v3) vs a capacity-/noise-matched RNN-latent twin (obs-dim + noise-placebo
  matched — the text-twin lesson, memory index). Compare gate-5 J-intervention fidelity and steerability
  dose–response. **No retrain of R6c needed.**
- **Null:** the RNN twin matches on fidelity → the posterior's advantage is capacity, not naming.
- **Route:** battery gate-5 + steerability diagnostic; **must** capacity/noise-match (else UNINFORMATIVE, per the
  overturned text-twin RED).

### H2 — Decisional-state codebook fixes the retracted "uniqueness-tracks-fidelity"
- **Claim:** clustering CRYSTAL-1 belief-states by **action-equivalence** (Brodu decisional states, A11) yields a
  codebook whose code-count C\* tracks write-fidelity, where the *geometric* VQ codebook did NOT (B2 retraction).
  **[Spec]**
- **Mechanism:** B2 failed because geometric uniqueness ≠ behavioral distinctness; utility-merged states collapse
  exactly the belief differences the policy ignores, so remaining codes are the ones a write can actually move.
- **Cheapest test:** re-run `interpretability/b2_multiseed.py` C4 (C\*≈K bend) but replace the VQ/KMeans codebook
  with a decisional-state partition (merge belief cells with identical argmax-action under the frozen policy).
  Check whether C\* now correlates with per-code steerability.
- **Null:** decisional codebook shows the same flat fidelity as geometric → naming problem is deeper than
  clustering (supports A12 ceiling).
- **Route:** battery diagnostic sim@K + IV C1 (causal sufficiency at the actual write site).

### H3 — Proper-scoring belief-elicitation is a certifiable faithfulness gate ("write-what-you-report")
- **Claim:** forcing CRYSTAL-1 to *emit* its posterior `b` under a tiny strictly-proper scoring bonus makes
  belief-write fidelity *certifiable*: an operator write to `b` produces the action the *reported* belief predicts,
  closing the IV "self-steering" hole (policy writes its own belief channel to fool a pre-action monitor).
  **[Strong-Ctx, Tier-A scoring rules]**
- **Mechanism:** the bonus makes truthful `b` incentive-compatible; the monitor then reads *post-intervention* `b`
  AND realized action and alarms on belief–action decoupling (IV §8 self-steering defense).
- **Cheapest test:** add a strictly-proper (log/Brier) side-reward on the regime-POMDP's *known* hidden regime;
  measure calibration of reported `b` vs true regime, then measure whether writing `b:=e_k` yields the
  regime-k-conditioned action. Cheap: the true regime label exists in `src/series_g/regime_pomdp.py`.
- **Null:** the bonus materially shifts the objective (policy games the score) → not "tiny/proper"; abandon.
- **Route:** NEW battery gate wired into `certified_battery_v2.py`; IV C1+C4 (side-effect vector on return while the
  elicitation bonus is on).

### H4 — Belief-write dose–response is NOT protocol-invariant (composition hazard from Anufriev–Panchenko)
- **Claim:** the *same* certified belief-write has a *different* signed effect under a market-maker clearing wrapper
  vs an order-driven wrapper on the same belief ecology → write certificates are **protocol-local**, an instance of
  IV §5 sign-epistasis. **[Plaus → Established if it flips]**
- **Mechanism:** Anufriev–Panchenko (A5 source): trading protocol moves the bifurcation thresholds; a write that
  rides a self-validating trend under one clearing rule fades it under another.
- **Cheapest test:** build the Brock–Hommes ecology once, wrap it in two clearing rules (Walrasian / market-maker),
  apply the *identical* belief-write, and compare dose–response sign via the steerability diagnostic. This is a
  synthetic-env test (Series-G class), safe for archive-return.
- **Null:** dose–response sign is stable across wrappers → write certificates are protocol-portable (good news;
  widens IV C3 envelope).
- **Route:** IV C3 (envelope certificate must carry a `protocol_index`) + battery steerability under two wrappers.

### H5 — Cumulative belief-write budget: legible writes SALAMI-SLICE into an illegible trajectory
- **Claim:** many individually-legible, in-envelope belief-writes integrate to a policy whose *trajectory* leaves
  the visited-state envelope and whose behavior no longer matches any single write's story — El Farol's
  "switch-count↑ ⇒ payoff↓" (A9) at the belief-write level. **[Plaus, Tier-A El Farol + HL5 IV/TB/GV convergence]**
- **Mechanism:** IV §4 / TB §6 trajectory ratchet: per-write envelope-scoping (visited states only) is not
  compositional; a sequence of legal writes can walk the belief off-manifold.
- **Cheapest test:** on the regime-POMDP, apply N sub-cap belief-writes per episode at rising N; measure (a) the
  fraction of resulting states OUTSIDE the training envelope (OODGate Mahalanobis, `src/evaluation/firewall.py`)
  and (b) story-decay of the write's predicted vs realized action. Predict both degrade monotonically in N.
- **Null:** envelope-scoping holds under composition (writes never walk off-manifold) → CRYSTAL-1's
  visited-states scoping is compositional-safe (strong positive result).
- **Route:** IV §4 cumulative budget `D_ep = Σ|α_t|` + OODGate; feeds a per-episode write-budget knob.

### H6 — Abstention/withdraw as a first-class writable verb captures disproportionate control-value
- **Claim:** exposing a WITHDRAW/ABSTAIN verb in CRYSTAL-1's writable grammar (not just directional belief-writes)
  captures an outsized share of steerable value, mirroring the market-maker cancel-option (~19%, A11) and El Farol/
  minority-game first-class no-trade (A7, A9). **[Plaus]**
- **Mechanism:** much control-value is in *when to withdraw*, which a purely directional belief-write cannot
  express; ABSTAIN is a distinct action-equivalence class (H2) an operator must be able to command.
- **Cheapest test:** in `crystal1_b3_riskmode.py`'s 3-mode exposure, add a fourth explicit ABSTAIN mode (abstain is not one of the existing three) and measure the
  steerability dose–response of the abstain-write vs directional-write; check drawdown-control gain (cf.
  reward-scope memory: de-risking cut DD not Sharpe — abstain should show the same asymmetric signature).
- **Null:** abstain-write adds no steerable value beyond directional writes → withdraw is redundant on this polygon
  (consistent with Kyle under-complexity, A6).
- **Route:** battery gate-5 per-class compliance (ABSTAIN as its own class with its own 0.67 floor) + ghost-portfolio
  `no_trade` ledger (`src/evaluation/ghost_portfolios.py`).

---

## 5. Self-grade + single weakest part

**Self-grade: 4.1 / 5.** Source auditability 4.5 (every atom cites a real local file; CRYSTAL-1 code refs
verified on disk). Controllability focus 4.5 (held the line on steering/legibility/authority/composition, not
alpha). Ceiling synthesis 4.3 (14-report convergence gives the knee real weight). Hypothesis falsifiability 4.0
(all six route to existing tooling with named nulls). Critique sharpness 4.0.

**Single weakest part:** the **quantitative ceiling knee (interp ≈ 0.40) and every ZOO coordinate are self-graded
[Plaus] expert inferences, not measurements on a shared dataset** — so the whole §2 frontier is ordinally solid but
cardinally soft, and any downstream CrystalScore anchoring off these numbers would be borrowing unearned
precision. The runner-up weakness is that H1's foundational atom (A1, posterior-beats-latent-and-is-more-
interpretable) is proven on toy finite latents and **extrapolated** to markets, where CRYSTAL-1's own B4 already
shows the daily-regime latent carries VoI=0 — so the plan's headline mechanism rests on an unreplicated toy result
until H1 runs.
