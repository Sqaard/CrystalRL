# Scope of responsibility and the branch model

*Created 2026-07-19. Defines who owns what, which branch it lives on, and what neither track may
change unilaterally. Defers to [STARTUP_OVERVIEW.md](STARTUP_OVERVIEW.md) on strategy and to
[NORTH_STAR.md](NORTH_STAR.md) / [WORK_ORDER.md](WORK_ORDER.md) on product claims.*

## 1. The two tracks

| | **CrystalRL** (layer 3) | **Hello Crystal** (layer 4) |
|---|---|---|
| Owner | **Joseph** | **Ivan** |
| Research question | a causal, predictive, scalable account of what the agent believes, plans and optimizes | use that account to predict failures under distribution shift and improve OOS metrics |
| Branch | `track/crystalrl` | `track/hello-crystal` |
| Mirror repo | `github.com/Sqaard/CrystalRL` (Joseph's clone-and-run perimeter) | — (mothership only) |

## 2. Ownership map (who commits what, without asking)

**CrystalRL — Joseph's scope**
- `interpretability/` — the interpretability experiment families: the W-series (`exp_w0…exp_w4`),
  CRYSTAL-WORLD (`exp_wp*`, `exp_wb*`, `exp_ktb*`, `exp_ktd*`), probes, batteries, CrystalScore.
- `src/crystal/`, `src/ssl/`, `src/causal/`, `src/diagnostics/` — representation & probe machinery.
- `business/CRYSTAL_WORLD_THEORY.md`, `business/CRYSTAL_WORLD_METHODOLOGY.md`.
- `experience/` — the economic world picture corpus, the V2 atoms, the binding pipeline.
- `reports/CRYSTAL_WORLD_*`, `reports/beyond_daily_bar_*`, the interpretability paper
  (`paper_crystalrl/`).
- `data/_crystal_world/` — the episodic + semantic stores.

**Hello Crystal — Ivan's scope**
- The coding-agent machinery: `interpretability/hl_v10*`, `hl_v11*`, `hl_v12*`,
  `hl_autonomous_loop*`, the proposer/gate evolution in `src/hl/` (`gate.py`, `staged_gate.py`,
  `pareto_gate.py`, `proposer.py`, `mechanism_bandit.py`, `teacher_bank*.py`).
- The personalization product: `interpretability/personal_invest_*`, `contracts/`,
  `paper_hl_personal_invest/`.
- Live/paper trading: `live/`, `scripts/` (incl. the CN L5 recorder), the deployment spec.
- `heuristic_agent_r6c/`, the R6c/CHRL model line, data acquisition and panel builders.

## 3. The shared core — changed only by agreement (both sign off)

These are the objects that make results comparable across tracks. A unilateral change here
invalidates the other track's evidence, so they move only by explicit agreement:

1. **The certification machinery** — the v12 frozen gate, the exposure-matched dial twin, the
   capacity-fair noise twins, the i.i.d. placebo-market guard, purge/embargo rules.
2. **The window discipline** — TRAIN 2010-18 / DEV 2019-21 / HOLD 2022-23 / OOS 2024-26
   read-once-per-claim; the quarantined pre-registrations (2027-01, July 2029) stay locked.
3. **`EXPERIMENT_LOGBOOK.md`** — append-only, both tracks write; nobody edits another's entry.
4. **The artifact register** (9 classes) and the standing rules derived from it.
5. **`business/`** strategy docs: NORTH_STAR, WORK_ORDER, STARTUP_OVERVIEW, MOTHERSHIP, this file.
6. **`data/`** panels — read-only shared substrate; a panel rebuild is announced before it lands.

## 4. Branch protocol

```
main                          ← released/consolidated state (protected; merges only from track/*)
├── track/crystalrl           ← Joseph works here; mirrors to Sqaard/CrystalRL
└── track/hello-crystal       ← Ivan works here
```

- **Work happens on your own track branch.** Neither track commits to `main` directly.
- **Cross-track changes** (anything in §3, or a file in the other track's map) are proposed as a
  PR/merge request to the other branch, not pushed into it.
- **Merge cadence:** each Sunday call — rebase both tracks on `main`, merge what is complete,
  resolve conflicts in the shared core together.
- **The logbook never conflicts:** append a new entry at the top of the entries section; if git
  reports a conflict there, keep BOTH entries.
- **Reproducibility rule:** a merge into `main` must leave the repo clone-and-run (verified by
  running one experiment end-to-end, as of 2026-07-19 the CrystalRL mirror passes this).

## 5. Deadlock rule (mirrors the 50/50 governance)

If the two tracks disagree on a shared-core change: the owner of the affected layer proposes, the
other reviews within the week, and if unresolved it goes to the Sunday call with a written
one-paragraph case from each side. Nothing in the shared core changes silently; nothing in a
track's own scope needs permission.

## 6. Current state (2026-07-19)

Both branches are cut from `research/exec-econ-newhigh-cycle` @ the BD-2 commit — i.e. they start
identical and carry the full CRYSTAL-WORLD program (W0-W4 complete), the world picture V2, the
binding pipeline, the beyond-daily analysis, and all reviews. The older research branch stays as
the historical record and is not deleted.
