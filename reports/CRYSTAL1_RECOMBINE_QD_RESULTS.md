# recombine on a modular POET/QD substrate — the operator bites when the task is modular

The HL v3 verifier showed `recombine` is implemented + correct but **inert on the non-modular corner polygon** (0%
yield) — because that task's optimum is a single structure with no complementary partial-solutions to combine. This runs
recombine on a **genuinely modular** substrate to show its value is a property of task modularity, exactly as predicted.
Code: [modular_rule_policy.py](../src/hl/modular_rule_policy.py), [hl_recombine_qd.py](../interpretability/hl_recombine_qd.py).
Report: `interpretability/hl_recombine_qd_report.json`.

## The modular substrate
`RegimeRotationEnv(G=6)`: 6 venues, and `provide@venue-v` earns the spread only when the hidden regime is `v` — so the
reward **decomposes per venue** and each venue is an **independent sub-strategy (niche)**. A policy is a list of per-venue
provide clauses (+ a shared unwind clause); its **coverage** = which venues it provides at. Coverage is additive: covering
{0,1}=+2.7, {0,1,2,3}=+4.8, all six=+6.0 (measured). This is the modular structure crossover is the yield operator for.

## MAP-Elites / QD loop (frozen-holdout insertion)
Archive keyed by the **set** of covered venues (preserving the diversity crossover needs — count-keyed niches collapse to
nested coverage and kill it). Two operators: **add_provide** (mutation: cover one more venue) and **recombine** (uniform
crossover: union two elites' coverage, keeping the better per-venue threshold). Insertion is gated on a frozen holdout
(propose on dev, confirm on holdout). Run WITH recombine vs a **mutation-only** baseline, 5 seeds each, 45-round budget.

## Result — recombine bites (and the contrast is the point)

| | WITH recombine | mutation-only | non-modular corner (v3) |
|---|---|---|---|
| **recombine yield rate** (child beats BOTH parents) | **21%** (22/104) | — | **0%** |
| recombine-authored MAP-Elites inserts | **23** (vs 77 from mutation) | — | 0 |
| **reached FULL coverage (6 venues)** | **5/5 runs** | **3/5 runs** | n/a |
| max coverage (mean) | **6.0** | 5.6 | n/a |
| best honest return | 5.87 | 5.86 | n/a |
| raw QD-score / niches filled | 103 / 21 | 130 / 27 | n/a |

**The load-bearing findings:** (1) recombine's **yield rate is 21% here vs 0% on the non-modular polygon** — the same
operator, its value determined entirely by task modularity, as the v3 verifier predicted. (2) With recombine, **all 5
runs assemble the full-coverage policy** within budget; mutation-only assembles it in only **3/5** — because crossover
fuses two disjoint partial-coverage elites into complete coverage in one step, which venue-by-venue mutation sometimes
fails to finish. Honest counter-point: mutation-only posts a **higher raw QD-score** (fills more fine-grained low-coverage
niches, 27 vs 21) — a different QD axis; recombine's edge is reliably reaching the **high-value full-coverage** niche, not
niche count.

## What this closes
Every HL5-audit idea is now not just implemented but **demonstrated firing on a substrate that stresses it**: recombine —
the one operator the polygon left inert — bites on a modular POET/QD task (21% yield, 5/5 vs 3/5 full-coverage
assembly). This confirms the v3 honest call: recombine was correct all along; the polygon simply had no modularity to
exploit. The natural home for the full growth stack (add_knob/add_rule/recombine + the ecology/QD archive) is exactly this
kind of modular, multi-niche substrate.
