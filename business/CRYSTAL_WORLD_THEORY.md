# CRYSTAL-WORLD — The Theory File

*The intellectual foundation for the project's reframed end goal. Written 2026-07-18 from Ivan's
theses (inspired by Yann LeCun's Harvard CMSA Ding Shum Lecture, 2026-03-28 video: "Objective-Driven
AI: Towards AI systems that can learn, remember, reason, and plan" — youtu.be/yUmDRxV0krg) plus an
8-thread, 63-atom adversarial literature sweep
([raw atoms](../reports/crystal_world_lit_atoms_20260718.json)). This document is the reference for
all CRYSTAL-WORLD decisions; the companion build spec is
[CRYSTAL_WORLD_METHODOLOGY.md](CRYSTAL_WORLD_METHODOLOGY.md).*

---

## 0. The reframed end goal (Ivan, 2026-07-18)

> The project's end goal is NOT a self-evolving trading bot for its own sake. It is a bot that
> **accurately represents and enriches the economic world picture** that underlies its knowledge
> (its world model). Trading is the bot's way of testing and enriching that picture — the probe,
> not the product.

The six theses this file examines:

- **T1** — the product is the (audited) economic world picture, enriched by the bot.
- **T2** — the bot predicts the **representation** of the market situation, not the raw price (JEPA).
- **T3** — the bot has **memory/experience** holding the world picture; enrichment = analyze →
  predict own representation → act/trade → compare predicted vs actual representation → learn from
  the difference. *"This is how humans learn and how science works": science does not store raw
  data; planetary motion is six numbers plus abstractions.*
- **T4** — **energy-based models**: compare representations cleverly via a scalar compatibility.
- **T5** — **multi-hierarchical representation**: notation → formulas → theories; quantum theory →
  thermodynamics. Key property: **the higher the abstraction level, the longer the prediction
  horizon** (quantum: seconds; thermodynamics: hours).
- **T6** — philosophy: AI should be maximally similar to humans.

Verdict up front, in our house style: **T5 is the best-supported thesis — with a theorem-grade
backbone in physics, a replicated cortical instantiation, and (strikingly) our own project as its
cleanest finance evidence. T2/T3/T4 are well-grounded with named, load-bearing caveats. T1 needs a
reflexivity amendment. T6 survives only in a reformulated form.** Details and citations below;
every claim carries an Evidence-Status tag from the sweep.

---

## 1. T2 — Predict the representation, not the price

**Support.**
- I-JEPA (Assran et al., CVPR 2023) and V-JEPA (Bardes et al., TMLR 2024): predicting in learned
  representation space beats generative pixel reconstruction for semantic structure because the
  encoder may *discard unpredictable, irrelevant detail* — V-JEPA: latent prediction has "the
  flexibility to eliminate irrelevant or unpredictable pixel-level details", +3-4pp over VideoMAE
  under frozen eval at ~2× less pretraining compute. *(Strong-but-Contextual: with full fine-tuning
  pixel models close much of the gap — the honest claim is efficiency + abstraction, not dominance.)*
- V-JEPA 2 / V-JEPA 2-AC (2025): the full predict-in-latent → plan-by-comparing-representations
  loop works at deployment scale (zero-shot robot manipulation). *(Established, in robotics.)*

**The economics analogy (and it is more than an analogy).**
- **Hayek 1945**: the price system is a decentralized *compression device* — prices aggregate
  dispersed tacit knowledge into a low-dimensional statistic; **Grossman-Stiglitz 1980** prove the
  compression stays lossy in equilibrium. The price series is an *already-compressed* representation
  of the economy; residual edge lives only in the noise wedge. Re-decoding raw prices ever harder
  has structurally bounded value — which is precisely our triple-confirmed daily-price VoI≈0.
- **Factor models** (Ross 1976 APT; Kozak-Nagel-Santosh 2018): no-arbitrage itself forces the
  cross-section into a few latent components — low-dimensional representation as an equilibrium
  *consequence*, not a modeling convenience.
- **Sims 2003 rational inattention**: mainstream economics already models the human agent as a
  finite-capacity lossy compressor. Compression is the correct model of the market AND of the agent.

**Project-internal evidence.** Our certified rule's input is P(bear) — a 1-dimensional predicted
representation — and it works (twin z ≈ 2.6-3.0 hold) while price-level prediction is null. The
price level is the "unpredictable pixel detail" JEPA throws away; the regime is the structural
latent worth predicting.

**Load-bearing caveats.**
- **Collapse** (LeJEPA, Balestriero-LeCun 2025; VICReg, ICLR 2022): the JEPA objective has a
  trivial degenerate optimum — constant representations, zero loss, zero information. We have met
  its cousins twice: PPO heads collapsing to constant dials (E-27/28) and the degenerate z_dsd
  statistic (CL-1 review). *Representation prediction is not self-certifying.*
- **SNR floor**: TS foundation models at scale still fail to beat the random walk on daily returns
  (Noguer i Alonso & Franklin 2026: gains "small and sparse"); Fin-JEPA (SSRN 2026) — the first
  JEPA on equities — learns representations but **reproduces the alpha null**. Predicting
  representations does not repeal the noise floor at the price level.

## 2. T3 — Memory, enrichment, and the science analogy

**Support.**
- **Rao-Ballard 1999 predictive coding**: feedback carries top-down predictions, feedforward
  carries only the residual error — the literal cognitive-science instantiation of "predict →
  compare → learn from the difference". *(Established as an algorithmic model.)*
- **Schultz-Dayan-Montague 1997**: dopamine RPE is a physically measured prediction-error learning
  signal — among the best-validated results in computational neuroscience. **Dabney et al. 2020**:
  the code is *distributional* — the brain learns a distribution over outcomes, exactly the form of
  our certified value (a regime distribution acted on at the pessimistic tail).
- **Complementary Learning Systems** (McClelland et al. 1995): agents need TWO memories — fast
  episodic (hippocampus) + slow schema-extracting (neocortex) — as the mathematically forced
  solution to catastrophic interference. **This retro-justifies our architecture**: EXPERIMENT_LOGBOOK
  + ledgers = episodic store; the world-picture corpus + certified rules + negative-knowledge
  register = semantic store; and our CL-1c finding that *continual learning underperforms the
  frozen champion* is a textbook CLS prediction, not an embarrassment.
- **Tse et al. 2007 schemas**: schema-consistent facts consolidate dramatically faster;
  schema-*violating* input triggers structural relearning. Enrichment is not uniform error
  minimization — regime breaks are the expensive, salient events.

**Caveats.**
- The strong framing "this is how humans learn" leans on the free-energy principle, which leading
  critics (incl. a review co-authored by Andy Clark) tag as unfalsifiable as a grand theory. Use
  the *specific, validated* mechanisms (RPE, predictive coding, CLS, schemas) — not FEP-as-theorem.
- Classical predictive coding is *generative* (it reconstructs sensations); JEPA deliberately is
  not. T2's "predict the representation" is better justified by ML engineering than by cognitive
  science — the human-likeness here is partial, and we say so.
- **Memory pollution is real** (experience-following in LLM agents; sequential model-editing
  degradation): enrichment must be gated, audited, and reversible — never raw appends.

## 3. T4 — Energy-based "clever compare"

**Support** *(Established)*: LeCun et al. 2006 EBM tutorial; Dawid-LeCun 2024. Energies natively
represent multimodal futures and free the objective from likelihood; inference = pick the
lowest-energy configuration. The finance mapping is *relative-value ranking vs density
forecasting* — a desk needs the correct ordering under current conditions.

**Caveats** *(Established)*: energies are **uncalibrated** — no P(DD≤5%)≥90% contract can be read
off an energy; a separate calibration layer (our block-bootstrap contract machinery) remains
mandatory. And uncalibrated scores from separately trained models cannot be naively combined — the
same reason our methodology now demands exposure-matched twins instead of cross-comparing raw
statistics.

## 4. T5 — The abstraction-horizon ladder (the centerpiece)

**The physics backbone** *(Established)*:
- **Wilson RG / effective field theory**: integrating out fast, fine degrees of freedom leaves a
  coarse theory depending on a handful of relevant parameters — *when scales separate*.
- **Israeli-Goldenfeld 2004**: coarse-graining renders even computationally irreducible systems
  (Rule 110!) predictable at the coarse level — but the coarse rule is not unique and must be
  *searched*; 16 of 256 CA had none.
- **FSLE** (Aurell et al. 1996): predictability time is a quantitative *function of scale* —
  λ(δ) decreases with feature size in multiscale turbulence. This gives T5 its sharpest form: a
  measurable horizon-vs-scale curve.
- **Simon 1962 near-decomposability + Simon-Ando aggregation theorem**: in nearly-decomposable
  hierarchies, fast dynamics live within subsystems, slow dynamics between them — T5 stated
  formally sixty years before the lecture.

**The cortical instantiation** *(Established)*: temporal receptive windows lengthen from sensory
(tens of ms) to prefrontal cortex (seconds+) — a hierarchy of timescales (Kiebel-Daunizeau-Friston
2008; Murray; Hasson). "Higher abstraction = longer horizon" is a replicated property of cortex.

**The markets ladder** (the practical analogy Ivan asked for):

| Physics | Markets | Horizon | Our evidence |
|---|---|---|---|
| quantum / molecular | tick microstructure, L5 order book | seconds-minutes | CN recorder lead; execution economics |
| molecular chaos | daily prices | ~0 usable | **VoI≈0, triple-confirmed** |
| mesoscopic / kinetic | volatility state (EWMA/HAR cascade — Müller's heterogeneous-market hypothesis, Corsi HAR) | days-weeks | vol is the load-bearing variable (PK-3: belief ≤ vol in tails) |
| **thermodynamics** | **regime (Hamilton 1989 2-state)** | **~10-20 days** | **the certified rule; twin z 2.6-3.0** |
| climate | macro cycle (VRP/EBP at 1-6m; business cycle) | months-quarters | literature-confirmed state info; adds nothing to OUR belief (CL-1c) |
| geology | structural (valuation, demographics) | years | Boudoukh caveat below |

**The strongest sentence in this file** (and the sweep's own words): *the project's certified
regime rule — an abstract 2-state belief at a 10-day horizon, alive exactly where daily-level
prediction is null — is currently better evidence for the hierarchy-horizon principle on markets
than anything the JEPA literature itself has produced.* H-JEPA remains an undemonstrated proposal
(V-JEPA 2 plans at T=1 receding horizon; hierarchical planning is listed as future work).

**UPDATE 2026-07-18 — KT-B executed: FULL PASS.** The hierarchy signature now has direct
experimental evidence on this substrate (`interpretability/exp_w1_ktb_v2.py`, after a two-sided
harness invalidation): reach(L1-week) = 40d < reach(L2-month) = 84d over a strong dev-chosen null
family with placebo guards, the pooled month-level representation beats a capacity-matched flat
twin (the Nachum control — part of the dividend is representational, not just target smoothing),
and every significant cell confirms out-of-sample (z 6.5–10.7). Caveats: the identified protocol
was fixed after a referee had probed the data (clean confirmation path = a pre-registered
third-window read), and the flat-twin comparison lacks a z on the difference. T5's status on
markets upgrades from "our regime result is the local evidence" to "directly measured, twice
confirmed, pending one clean-window replication".

**Hard qualifications** (these go into every design decision):
1. **Lorenz 1969 / Palmer et al. 2014 — the real butterfly effect**: with a shallow enough
   cross-scale energy spectrum, fine-scale errors cascade *upscale* and impose a finite
   predictability barrier on coarse features. Markets: one microcap liquidity shock can break an
   index-level regime forecast — the register's 2022 stock-bond-hedge lesson. Higher abstraction
   does NOT always buy horizon; it requires benign cross-scale coupling (scale separation).
2. **Boudoukh-Richardson-Whitelaw 2008**: classic long-horizon predictability (DY/CAPE multi-year
   R²) is partly a statistical mirage of overlapping observations. Any "better at the slow level"
   claim needs non-overlapping windows and artifact-matched twins — our degenerate-z_dsd lesson,
   already learned the hard way.
3. **Nachum et al. 2019**: in learned agents, hierarchy's measured gains are mostly *exploration*,
   not long-horizon credit assignment. The abstraction-horizon link must be demonstrated with a
   flat, capacity-matched twin — never assumed from architecture.
4. **Smith-Foley 2008**: the econ↔thermo correspondence is real for statics but breaks on
   dynamics — no conservation laws, no ergodicity. Use coarse variables for *risk description*;
   never expect an equation of motion to transfer.

## 5. T1 — The world picture as the product (with the reflexivity amendment)

The world-picture-as-product frame has a 1945 pedigree (Hayek: the market itself is a collective
world-model compressor) and matches the startup anchor ("trust made auditable" — the audited
economic world model is the auditable object par excellence). But two Established results force an
amendment:

- **Performativity** (Perdomo et al., ICML 2020) + **McLean-Pontiff 2016** (97 predictors: −26%
  OOS, −58% post-publication): representing-and-acting *changes the territory*. Planets do not
  read ephemerides; traders read papers. An economic world picture that omits the observer — the
  agent's own footprint, crowding, and each regularity's decay clock — is not merely incomplete;
  it is systematically wrong in the direction of overconfidence.
- **Amended T1**: the product is a world picture that stores *"this regularity, this crowdedness,
  this decay clock"* — never *"this law"*. Our negative-knowledge register and alpha-decay
  observations (E-14b: csi500 reversal replicating at half magnitude) are early instances.

## 6. T6 — Human-likeness, reformulated

Sutton's bitter lesson (2019) stands against human-likeness as a *performance* principle — and our
own evidence agrees (cold PPO, given compute, rediscovers dials; bigger black boxes would match
the readable rule). The **defensible T6**: human-likeness is an *epistemic governance* principle.
In a domain where the evaluator cannot be trusted (backtests lie, replay is optimistic, statistics
degenerate), every claim must be simulatable, auditable, and falsifiable by humans — which forces
representations that are named, crisp, and low-dimensional. That is CrystalScore, and it is why the
champion is a readable rule rather than a weight blob. Human-likeness earns its place through the
trust channel, not the Sharpe channel.

## 7. Negative-knowledge appendix (guardrails, each Established unless noted)

G1 Collapse is the default failure of compare-representation learning; anti-collapse machinery
   (VICReg variance/covariance floors; SIGReg) is mandatory, plus behavioral twins (our rule).
G2 EBMs/probes have no intrinsic goodness metric; probe wins are protocol-sensitive. Only the
   frozen gate confers value claims.
G3 Next-step prediction skill does not certify a coherent world model (Vafa et al. 2024: the
   transformer's "impossible Manhattan"; fails on detours = structural breaks). Detour tests are
   part of the battery.
G4 Policies exploit learned-model flaws (Ha-Schmidhuber's dream exploits; our +30% replay lift
   halving OOW). The compare step must be adversarially protected.
G5 Performativity/decay (see §5) — the picture includes the observer.
G6 The SNR floor at price level survives representation learning (TSFM nulls; Fin-JEPA null).
G7 T5's dividend is undemonstrated in the source architecture family (V-JEPA 2 plans at T=1);
   our regime result is the local evidence — treat H-JEPA as blueprint, not authority.
G8 Long-horizon R² inflates mechanically on overlapping windows (Boudoukh).
G9 Upscale error cascades bound coarse horizons (Lorenz 1969).
G10 Memory enrichment pollutes without gating (experience-following; ROME/MEMIT drift); and our
    CL-1c: naive continual updating *underperforms* the frozen champion — consolidation must be
    slow, interleaved, and gate-certified (CLS).

---

*Status: living document. Every thesis here is falsifiable and several already carry pre-registered
kill tests in the methodology file. The evidence-status tags are the sweep agents' skeptical grading
verified against primary sources where quoted; the raw atoms with full citations are the appendix.*
