# The Mothership — a Guided Map of the Main Repository

> The "mothership" is the primary research repository,
> [`Sqaard/self-evolving-trading-bot`](https://github.com/Sqaard/self-evolving-trading-bot)
> (branch `research/exec-econ-newhigh-cycle`). This file is the English orientation map for a new
> partner. Governance authority lives in [`NORTH_STAR.md`](NORTH_STAR.md) →
> [`WORK_ORDER.md`](WORK_ORDER.md); the company frame lives in
> [`STARTUP_OVERVIEW.md`](STARTUP_OVERVIEW.md).

## The four-layer stack (what the company runs on)

The startup's technical stack has four major parts, in data-flow order:

| # | Layer | Question it answers | State |
|---|---|---|---|
| 1 | **FinIR** (Information Retrieval) | *what may be read*: point-in-time, source-disciplined evidence with provenance | built in an earlier arc; **frozen** |
| 2 | **FinGPT** (text → features) | *what the text means*: an LLM as a constrained, leakage-safe feature generator (never a trader) | built; honest verdict: current text payload is largely a macro re-encoding, **not a proven edge**; **frozen** |
| 3 | **CrystalRL** (interpretability) | *what the agent believes, plans and optimizes* — a causal, predictive, scalable account | **active — Joseph's research track** |
| 4 | **Hello Crystal** (self-development) | *can that account be used* to gate an LLM coding-agent loop that improves OOS metrics under distribution shift | **active — Ivan's research track** |

Layers 1–2 stay fixed until the layer-3↔4 loop is stable on historical data: the near-term product
is not live text-driven trading but a **certified self-evolving loop** whose every accepted change
is readable (3) and control-gated (4).

## Where things live in the mothership

| Path | What it is |
|---|---|
| `PROJECT_NORTH_STAR.md`, `PERSONAL_INVEST_WORK_ORDER.md` | the governing contracts (Russian originals; English in this folder) |
| `EXPERIMENT_LOGBOOK.md` | the lab notebook — every run, newest first, nulls included; **the project's memory** |
| `business/` | this folder: vision, translations, the startup overview |
| `src/crystal/`, `src/series_g/`, `src/hl/` | the CRYSTAL-1 core, the designed market, the HL-loop library |
| `interpretability/` | all active experiments: the scenario engine, DP champion, PPO heads, gates, backtests |
| `paper_crystalrl/`, `paper_hl_personal_invest/` | the two write-paper-first drafts (CrystalRL = Joseph-led; Hello Crystal = Ivan-led) |
| `reports/` | methodologies, stage verdicts, data-quality reports, pre-registrations (incl. `preregistration_w8_2027.md`) |
| `data/_personal_invest_registry/` | W2-signed data, the locked forecast ledger, DP policy tables |
| `heuristic_agent_r6c/`, `HS/` | the coding-agent scaffold and HCS harness (layer 4 machinery) |
| `tests/` | 116 tests: contract, gates, shield, solver, governance |

**Joseph's starter extract** of the interpretability track (code + data + results, self-contained,
verified to run from a clone) is the separate repo
[`Sqaard/CrystalRL`](https://github.com/Sqaard/CrystalRL); its `docs/CODE_MAP.md` maps every module.

## The honest state of play (read this before believing anything)

- **One certified real-data result**: the defensive Dow rule (`P(bear)>0.66 → exposure 0.74`), held
  on 629 untouched days (Sharpe 1.46 vs 1.39; maxDD −12.9% vs −15.5%); lives on a ratified paper track.
- **The DP champion beats every challenger** (best static +10.5pp; PPO regret 6.6pp) and is the
  transparent production personalizer.
- **US promises hold, CN honestly does not yet**: 5/5 US profiles ≥80% over 576 PIT cells (risk side
  100%/936, both markets); CN 1/5 with named fixes — verdicts never averaged.
- **The alpha history is a triple-confirmed null** — that is *why* the project pivoted to
  interpretability + calibrated promises; text features (layer 2) are not a proven edge.
- **W8 certification = REJECTED_HONEST by design**; the first pre-registered reads are
  **2027-07-12** (locked 1y forecasts) and **July 2029** (untouched 3y policy fold).
- **The audit-remediation arc (2026-07-12)** confirmed with controls: the RL risk-dial prototype's
  belief-command surface is not faithful (CrystalScore 0.03 vs the champion rule's 1.0) — the exact
  gap Joseph's track exists to close.

## Access

The mothership is private. Partner access = GitHub collaborator invite from Ivan (ask for it — the
north star / work order / logbook are meant to be read in full by both founders).
