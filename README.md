# CrystalRL

**Crystal Clear Reinforcement Learning: a trading agent that is legible *by construction* — a
policy you can read like a table, a memory you can write to like a command line — and the
measurement discipline that keeps "legible" from being a vibe.**

This repository is the **interpretability-track starter kit** for Joseph Lynch: the paper you
will drive, an interactive demo you can open in 60 seconds, the skill file for your coding
agent, and the working protocol of the lab. The full research codebase (data, gates, loops,
logbook) lives in the mothership repo:
[Sqaard/self-evolving-trading-bot](https://github.com/Sqaard/self-evolving-trading-bot).

---

## Why this exists

Most "explainable RL" trains a black box and then tells stories about it. We walked the other
road and measured the difference. The journey: **CHRL** — a strong hierarchical PPO trader with
readable *layers* but a 64-dimensional unnamed latent core and ~214 tuning knobs — was rebuilt
into **CRYSTAL-1**, whose only memory is a *named* probability (`P(bear)`, writable:
`SET_BELIEF(bear)=0.8` is a command the policy must obey) and whose policy head *is* an
≤8-leaf decision tree at full return parity.

Scored on the identical scalar — **CrystalScore = Faithfulness × Simulatability × Stability**
at a fixed story budget — the journey moves the agent from **0.15 to 0.94**, and the entire
6× gap is Simulatability: a short named story reproduces CRYSTAL-1's behavior and cannot
compress the latent. The claim is falsifiable, the costs are stated (on hard substrates
legibility is provably *not* free), and the honest split verdict is kept: the old agent still
leads as a *deployed* system.

Why it matters beyond one trading bot: in the companion work (*Hello Crystal*) an LLM coding
agent improves this system autonomously — and **an agent may evolve itself exactly as far as
it can be read.** Interpretability is the load-bearing wall, not documentation. That wall is
your track.

## Try it in 60 seconds

Open **[`live/CrystalRL_live_testing.html`](live/CrystalRL_live_testing.html)** in any browser
(no install, one file). You get both models with their architecture schemas, a 14-knob control
surface, and a chart with toggleable curves. Two things to try first:

1. Crank **SET_BELIEF write strength** to 100% — CRYSTAL-1 obeys (its memory is a named,
   writable belief); R6c does not move, because *there is no named memory to write to*.
2. Drag the shared **drawdown budget** — it is a hard cap for CRYSTAL-1 and only a soft nudge
   for R6c. Same knob, different coupling.

That is the whole thesis, in your hands: *different models respond differently to the same
knobs, and different knobs have different effects.*

## What's in the box

| Path | What it is |
|---|---|
| [`paper/main.pdf`](paper/main.pdf) | **The paper** — *CrystalRL: Crystal Clear Reinforcement Learning* (draft v0.1): the CHRL→CRYSTAL-1 journey, the final 10-axis comparison, the CrystalScore protocol, the four behavior-influence levers, the honest nulls, and **twelve falsifiable hypotheses (H1–H12) — your research program**, with `\joseph{…}` margin notes addressed to you. Sources: `main.tex` + `refs.bib` (Overleaf-ready), figures regenerable via `make_crystalrl_figures.py`. |
| [`live/`](live/) | The interactive demo above (single self-contained HTML + its build script). |
| [`agent_skill.md`](agent_skill.md) | Drop this into your personal Claude: the lab's honesty contract, your track's guardrails, and the **mandatory log-after-every-experiment protocol**. |
| [`docs/LOGBOOK_PROTOCOL.md`](docs/LOGBOOK_PROTOCOL.md) | The logbook template, the verdict enum, the honesty rules, and a real filled example. |
| [`pitch/`](pitch/) | The meeting deck (PDF + PPTX) and its spoken script — the fastest way to reload context. |

## The headline numbers (each traceable to a logged experiment)

| Claim | Number |
|---|---|
| CrystalScore, identical scalar | R6c **0.151** → CRYSTAL-1 **0.938** (gap = Simulatability: 0.24 vs 0.94) |
| The tree head costs nothing | return-parity gap **−0.018** (paired p≈0.84) after a BC warm start |
| Commands are causal, not correlational | agreement with an exogenous optimum **0.80 / 0.835**; write fidelity **1.0** |
| The system can say no | a +1-nat belief promotion **REFUSED** by the non-inferiority bar |
| Honesty has teeth | a famous indicator (CAPE) falsified; the RL challenger loses to the readable table — both reported |
| The open frontier | a **certified** return↔legibility tension on hard substrates — H3 in your program |

## How we work

Weekly Sunday sync (call or recorded video); agenda out 1–2 days ahead — what was done / what's
blocking / what help is needed / goals. Write the paper first. Every run logged, nulls first.
One caveat, the worst one. And the lab's first principle, from Feynman's *Cargo Cult Science*:

> "The first principle is that you must not fool yourself — and you are the easiest person to fool."

---

*Maintainers: Ivan Pavliuk (model & data) · Joseph Lynch (interpretability). Coding agents:
Claude, Codex — they propose, the gates dispose. Research code; nothing here is investment
advice, and every probability in the lab wears its evidence tier on its sleeve.*
