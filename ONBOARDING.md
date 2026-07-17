# Onboarding — Joseph Lynch (start here)

Welcome. You explain patterns of **brain** behavior; here the subject is an artificial trading
agent — and, unlike any brain, you get read–write access to every synapse, seed-reproducible
behavior, and free, ethical interventions. This page is the helicopter view of what you are
joining, in two parts: **A. the business** and **B. the science** — then a study plan and the
hypotheses worth starting with.

Authority order for everything here:
[`business/NORTH_STAR.md`](business/NORTH_STAR.md) →
[`business/WORK_ORDER.md`](business/WORK_ORDER.md) →
[`business/STARTUP_OVERVIEW.md`](business/STARTUP_OVERVIEW.md) (the venture frame) →
this page. The lab notebook is [`EXPERIMENT_LOGBOOK.md`](EXPERIMENT_LOGBOOK.md); the map of this
repository is [`business/MOTHERSHIP.md`](business/MOTHERSHIP.md).

---

## Part A — The business (helicopter view)

**The company (working name X-Lab)** builds a personal-investment product whose competitive asset
is *trust made auditable*: the client states capacity, horizon and goal; the system returns a
conservative return floor at a stated probability — and refuses when its own evidence is thin.

**The stack has four major parts** (data-flow order); two are deliberately frozen:

| # | Layer | Right it owns | Status |
|---|---|---|---|
| 1 | **FinIR** | to READ (point-in-time evidence, provenance) | frozen |
| 2 | **FinGPT** | to MEAN (LLM as constrained feature generator, never a trader) | frozen |
| 3 | **CrystalRL** | to be UNDERSTOOD (beliefs, plans, objectives — causally) | **active — yours** |
| 4 | **Hello Crystal** | to IMPROVE (an LLM coding-agent loop under experimental controls) | active — Ivan's |

Layers 1–2 are the basis of a future live-trading system and stay fixed until the 3↔4 loop is
stable on historical data. The near-term product object is the certified
`profile × horizon × return-quantile` frontier, not live text-driven alpha.

**The division of labor.** Two research questions, two leads, both papers co-authored:

- **Yours (CrystalRL):** *can we construct a **causal, predictive and scalable** account of what an
  agent believes, plans and optimizes?* This is the cutting edge of Explainable RL.
- **Ivan's (Hello Crystal):** *can that account be used to **predict failures under distribution
  shift** and improve out-of-sample metrics?*

They interlock: your track supplies the readable object and the causal account; his supplies the
evidence standard (gates, calibration, pre-registration) that keeps yours honest.

**The honest state of play** (details in [`business/MOTHERSHIP.md`](business/MOTHERSHIP.md)): one
certified real-data policy (the defensive Dow rule, held on 629 untouched days); a transparent DP
champion (+10.5pp over best static; beats every RL challenger); US promises hold 5/5, China
honestly 1/5; the alpha history is a triple-confirmed null (that's *why* the pivot happened);
client certification is REJECTED_HONEST by design with dated unlocks (2027-07-12, 2029-07).

**The clock:** onboard now → papers to preprint quality by autumn 2026 → the first pre-registered
forecast read 2027-07-12 → the untouched 3-year fold matures July 2029.

---

## Part B — The science (what CrystalRL is, and what it lacks)

### What exists

CRYSTAL-1 is an agent built legible **by construction**: its only memory is a *named* regime
posterior `P(bear)` (an explicit Bayes filter you can write to — `SET_BELIEF` is a do-operator);
its policy is a small decision tree at full performance parity; its command surface is five named
levers under a certified write-governance stack. The measurement discipline is
**CrystalScore = Faithfulness × Simulatability × Stability** plus the MDL deficit. Read the paper:
[`paper_crystalrl/main.pdf`](paper_crystalrl/main.pdf) (you drive it).

### What CrystalRL currently LACKS — your problem statement, honestly

1. **Causality is weak where it matters most.** The transparent champion *rule* is a faithful
   command surface (F = 1.0), but the RL prototype head is not: belief-write fidelity **3%**,
   CrystalScore **0.03** — under controls, its "interventions" turned out to be near-uniform /
   global-bias artifacts. We can *read* the agent; we cannot yet prove the belief *causes* the
   action across the model family. The track's core gap.
2. **A literature-grounded explanation exists and is testable.** Richens & Everitt (ICLR 2024)
   prove agents need causal world models only under *distribution-shift pressure*; our substrate's
   optimum is a constant dial, so the agent has no incentive to *use* its belief. Low fidelity may
   be a *training-pressure diagnosis*, not a measurement failure — hypothesis BH1 below.
3. **Ceiling metrics are single-substrate.** Naming faithfulness 1.0 / stability 1.0 / MDL 0.0
   were measured where legibility is cheap; on the hard designed substrate legibility is provably
   *not* free (a certified return↔legibility tension).
4. **No human data.** Simulatability (0.92–1.0) is machine-measured; nobody has tested whether a
   *human* can predict the policy — your psychophysics, literally.
5. **The belief itself is primitive.** A 2-state Gaussian HMM — the regime-detection literature has
   moved to statistical jump models (persistent, less flip-flop, still nameable).
6. **No formal account of objectives.** We say "the dd05 head optimizes its budget" informally;
   goal-directedness is now formally measurable (MEG, NeurIPS 2024) and we've never computed it.

### The literature map (10 load-bearing works — read in this order)

1. **Milani, Topin, Veloso & Fang** (ACM Computing Surveys 2024) — *the* XRL taxonomy; CRYSTAL-1
   occupies the rare "interpretable-by-design at every level" cell.
2. **Richens & Everitt** (ICLR 2024, "Robust agents learn causal world models") — the theoretical
   spine of your track; explains our fidelity gap.
3. **Atrey, Clary & Jensen** (ICLR 2020, "Exploratory, Not Explanatory") — why interventions, not
   plausibility, are the arbiter; our tie-break-artifact finding is its descendant.
4. **Li et al.** (ICLR 2023, Othello-GPT) + **Nanda et al.** (2023) — the probe-then-intervene
   validation standard our belief-write fidelity implements.
5. **Bush et al.** (ICLR 2025 oral, "Interpreting Emergent Planning in Model-Free RL") — the
   evidence standard for a causal account of *plans* (probe + intervene + behavioral convergence).
6. **Everitt et al.** (AAAI 2021, causal incentives) + **MacDermott et al.** (NeurIPS 2024, MEG) —
   the formal language for *objectives*: influence diagrams and measurable goal-directedness.
7. **Marton et al.** (ICLR 2025 spotlight, SYMPOL) — direct tree-in-PPO beats post-hoc
   distillation; defends our SoftTree design and gives the benchmark to compare against.
8. **Shu, Yu & Mulvey** (2024, statistical jump models; `pip install jumpmodels`) +
   **Aydınhan et al.** (Annals of OR 2024, continuous JM) — the belief upgrade path.
9. **Nasvytis et al.** (AAMAS 2024, DEXTER) + **Mallen et al.** (2023–25, mechanistic anomaly
   detection) — internal structure as an OOS-failure predictor, with honest limits; the bridge to
   Ivan's track.
10. **de la Rica Escudero et al.** (PLOS One 2025) + **Verma et al.** (2026, HMM+RL allocation) —
    the finance-XRL frontier: nearly empty, nobody does belief-level do-interventions. Our gap.

**Two citable open niches the lab already occupies** (from the July-2026 sweep): nobody gates an
LLM self-improvement loop with online-FDR-style statistical controls, and nobody uses a *named*
belief state as the OOS-failure predictor in trading. Your work lands in open territory.

### Study plan (4 weeks, reading + doing interleaved)

**Week 1 — the object and the standard.**
Read: this repo's paper (`paper_crystalrl/main.pdf`), Milani survey, Atrey (why interventions).
Do: run the sandbox (`personal_invest.py --menu`), the live demo (`live/CrystalRL_live_testing.html`),
and `interpretability/exp_e21_transparency_audit.py`; read the top-5 logbook entries; load
[`agent_skill.md`](agent_skill.md) into your Claude. Log your first (replication) entry.

**Week 2 — causality of beliefs.**
Read: Richens & Everitt; Othello-GPT + Nanda; Bush et al.
Do: reproduce the E-28 control battery (`exp_e28_all_levers.py`); draw CRYSTAL-1's causal
influence diagram (does the belief node carry a response incentive in the reward-dial head vs the
champion rule?); write it up — this becomes a paper section.

**Week 3 — the belief upgrade and the objectives account.**
Read: Shu/Mulvey jump models; MacDermott (MEG); SYMPOL.
Do: BH2 (jump-model belief swap — the cheapest real experiment, `pip install jumpmodels`) through
the existing gate; compute MEG for the dd05 head (BH4).
**Week 4 — humans and the bridge.**
Read: DEXTER + Mallen (anomaly detection); Kohler (simulatability proxies).
Do: design the human-simulatability protocol (BH6/H1) on the DP tables; sketch BH3 (belief-NLL as
an OOS alarm) with Ivan. Pick YOUR one bold hypothesis for the autumn.

### Hypotheses to start with (each with its kill test; the bold ones are bold on purpose)

- **BH1 — The Pressure Hypothesis** *(bold; the track's centerpiece)*. Belief→action causality
  emerges **iff** the training environment exerts distribution-shift pressure (Richens-Everitt
  applied to markets: a priced regime ⇒ no pressure ⇒ no causal use — which also *explains our VoI
  gate*). Test: train the same head under a shift-pressure curriculum (regime-switching episodes,
  shifted eval); fidelity should rise from 0.03 materially. Kill: fidelity stays ≈0 under genuine
  pressure. Publishable either way.
- **BH2 — Jump-model belief.** Replace the 2-state Gaussian HMM with a (continuous) statistical
  jump model; the belief becomes more persistent and the certified rule's NI improves through the
  frozen gate. Kill: no NI improvement / gate rejects. (Cheapest first experiment; PyPI package.)
- **BH3 — The belief is an OOS alarm** *(the bridge to Ivan's track)*. Drift/NLL of the *named*
  belief predicts strategy failure windows better than raw world-model surprise (DEXTER's negative
  result is the bar to beat). Kill: belief-based alarms no better than volatility baselines.
- **BH4 — Measurable objectives.** MEG-style goal-directedness of each head w.r.t. its stated
  budget; prediction: dd05 scores high, dd08/dd12 low (they breach). Kill: MEG fails to separate
  heads whose budget adherence differs.
- **BH5 — The incentive diagram predicts fidelity.** Draw the causal influence diagram per policy;
  belief-node response incentive present ⇒ high write-fidelity (champion rule), absent ⇒ low
  (reward dial). Kill: an incentive-bearing policy with near-zero fidelity.
- **BH6 — Human simulatability (H1).** Machine simulatability 0.92–1.0 predicts human prediction
  accuracy on unseen DP-table cells. Kill: humans at chance. (Your methods, on humans.)
- **BH7 — e-value gating** *(bold, methodological)*. Upgrade the loop's accept/reject accounting to
  e-value online FDR (e-GAI 2025); no published finance application exists. Kill: e-gating loses
  power vs the current gate on the replayed proposal ledger.
- **BH8 — Direct beats distilled.** Benchmark our SoftTree-in-PPO against SYMPOL and post-hoc
  VIPER-style distillation on the same substrate. Kill: distillation matches direct training (would
  overturn a design premise — worth knowing).

The full H1–H12 program lives in the paper (§ "The research program"); BH1–BH8 are the fresh,
literature-grounded additions from the July-2026 sweep. Reshape freely — the kill-test discipline
is the only non-negotiable.

---

## The two rules that matter most

1. **Be honest — lead with the null.** A clear negative result is a real contribution; a positive
   result with no stated null is not. (Feynman: you must not fool yourself — and you are the
   easiest person to fool.)
2. **Log every run** — the template and protocol are in
   [`.claude/skills/experiment-logger/SKILL.md`](.claude/skills/experiment-logger/SKILL.md); an
   unlogged result does not exist.

## Your first afternoon (three commands)

```bash
python interpretability/personal_invest.py --max-dd 0.10 --confidence 0.9 --target inflation
python interpretability/personal_invest_dp.py        # the readable champion; exports the policy tables
python interpretability/exp_profile_promise_backtest.py --universe US   # the flagship experiment
```

Then open one DP policy CSV (`data/_personal_invest_registry/dp_policies/`) — that state→action
table is the object your metrics live on. Questions → Sunday sync, or a `\joseph{}` note in the
paper; asking early beats a week of guessing.
