# CrystalRL

Interpretability track of the CrystalRL lab. This repo is self-contained: paper, live demo,
code, data, and results. This page is your reading order — nothing else.

## 1 · Try it in 60 seconds

Open **[`live/CrystalRL_live_testing.html`](live/CrystalRL_live_testing.html)** in any browser
(one file, no install). Then do two things:

1. Crank **SET_BELIEF write strength** to 100% — CRYSTAL-1 obeys, R6c cannot even receive the
   command (no named memory).
2. Drag the shared **drawdown budget** — hard cap for CRYSTAL-1, soft nudge for R6c.

That's the thesis in your hands. Everything below explains how it's built and measured.

## 2 · Analyze these files first, in this order

| # | File | Why first |
|---|---|---|
| 1 | [`paper/main.pdf`](paper/main.pdf) | The whole story with equations. Read: abstract → §5 (CrystalScore) → §9 (**H1–H12 — your research program**) → the `Joseph:` margin notes. |
| 2 | [`docs/CODE_MAP.md`](docs/CODE_MAP.md) | The map of everything in this repo: entry points by theme, conventions, what's deliberately not included. |
| 3 | [`agent_skill.md`](agent_skill.md) | Load this into your Claude before any work — the honesty contract, guardrails, and the mandatory logging protocol. |
| 4 | [`src/crystal/belief_filter.py`](src/crystal/belief_filter.py) | The named memory: a Bayes filter over market regimes (paper Eq. 2). ~Small file, read fully. |
| 5 | [`src/crystal/soft_tree_policy.py`](src/crystal/soft_tree_policy.py) | The policy that IS the story (paper Eq. 3). |
| 6 | [`interpretability/crystal_score.py`](interpretability/crystal_score.py) | How Faithfulness × Simulatability × Stability is actually computed. |
| 7 | [`interpretability/exp_e21_transparency_audit.py`](interpretability/exp_e21_transparency_audit.py) + [`exp_e28_all_levers.py`](interpretability/exp_e28_all_levers.py) | The audit and the four influence levers on the real panel — incl. the refused intervention. |
| 8 | [`src/crystal/writ_ladder.py`](src/crystal/writ_ladder.py) + [`contracts/crystal1_knob_registry.yaml`](contracts/crystal1_knob_registry.yaml) | How writes get certified; the 5-lever command surface. |
| 9 | [`artifacts/stage4/`](artifacts/stage4/) → `R6c_*_for_Joseph/README_R6C_FROZEN_TEST_FOR_JOSEPH.md` | The frozen R6c behavior package prepared for you — the black-box side of every comparison. |
| 10 | [`data/_personal_invest_registry/dp_policies/`](data/_personal_invest_registry/dp_policies/) | The readable policy tables — the object your first experiment (H1, simulatability psychophysics) lives on. |
| 11 | [`docs/LOGBOOK_PROTOCOL.md`](docs/LOGBOOK_PROTOCOL.md) | How every run gets logged. An unlogged result does not exist. |
| 12 | [`reports/CRYSTAL1_CONTROLLABILITY_FINAL_REPORT.md`](reports/CRYSTAL1_CONTROLLABILITY_FINAL_REPORT.md) | The provenance of every number — go here when you doubt a claim. |

Context reloaders when needed: [`pitch/pitch.pdf`](pitch/pitch.pdf) (the meeting deck),
[`pitch/SPEECH.md`](pitch/SPEECH.md).

## 3 · Set up the code (2 minutes)

```bash
git clone https://github.com/Sqaard/CrystalRL.git && cd CrystalRL
pip install -r requirements.txt
python -c "import sys; sys.path[:0]=['.','scripts']; import interpretability.mdl_fidelity_deficit"
# ~30 s: recomputes the frozen-log headline -> your setup works
```

## 4 · Then let your Claude teach you the rest

Open your Claude in this repo and paste exactly this:

```
/learn I just cloned the CrystalRL repo — I own its interpretability track. My background is
neuroscience (explaining behavior patterns of brains), so use neuroscience analogies wherever
they fit. First read agent_skill.md and docs/CODE_MAP.md yourself, then teach me the system
one concept at a time, quizzing me before moving on:

1. The named memory: src/crystal/belief_filter.py — the Bayes filter over market regimes, and
   why a NAMED belief makes interventions possible (paper Eq. 2).
2. The policy that IS the story: src/crystal/soft_tree_policy.py and the certified rule inside
   interpretability/crystal_ppo.py (paper Eqs. 3-4).
3. CrystalScore = Faithfulness x Simulatability x Stability: how each axis is computed in
   interpretability/crystal_score.py and cross_policy_crystal.py, and why R6c scores 0.15 while
   CRYSTAL-1 scores 0.92 — using the frozen R6c log in artifacts/stage4/ as the black-box side.
4. Interventions and refusals: SET_BELIEF and the dose nudges in
   interpretability/exp_e28_all_levers.py, and the writ ladder in src/crystal/writ_ladder.py —
   including why the +1-nat promotion was REFUSED (this is the optogenetics analogy).
5. The designed market and its fence: src/series_g/family_env.py and
   interpretability/voi_gate.py — why polygon results are instrument calibration, never market
   evidence.

When I can explain all five back, help me pick ONE hypothesis from H1-H12 in paper/main.tex
(section "The research program"), draft the cheapest experiment that could kill it, and write
my first EXPERIMENT_LOGBOOK.md entry skeleton per docs/LOGBOOK_PROTOCOL.md.
```

## 5 · The one rule

> "The first principle is that you must not fool yourself — and you are the easiest person to
> fool." — Feynman

Every run → one logbook entry (template in [`docs/LOGBOOK_PROTOCOL.md`](docs/LOGBOOK_PROTOCOL.md)):
the null you tested, the exact command + seed, the worst caveat, an honest verdict. Questions →
Sunday sync, or leave a `\joseph{}` note in the paper.
