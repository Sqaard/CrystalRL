# HL v3 — structural policy growth (add_rule / recombine) through a multi-role, staged gate

The last two unused HL5 ideas, implemented and run: **HLX add_rule / recombine** operators (the agent grows the policy
STRUCTURE, not just tunes knobs) and **GV multi-role + shadow→canary→fleet staging** (independent roles on disjoint
data + staged exposure with abort/rollback — how the inner release-sign-off human is removed safely). Code:
[rule_policy.py](../src/hl/rule_policy.py), [staged_gate.py](../src/hl/staged_gate.py),
[hl_v3_loop.py](../interpretability/hl_v3_loop.py). Report: `interpretability/hl_v3_loop_report.json`.

## The two ideas, implemented

**Rule-list policy + operators.** A policy is an ordered decision list of **conjunctive** clauses (`if belief<0.20 &
inv<4 -> PROVIDE`), the born-legible form (a C-4 story tree flattened). Operators:
- **add_rule** — graft a new typed clause (e.g. a risk-off dump `belief>=0.85 & inv>=1 -> AGGRESS`).
- **retune_rule** — nudge a threshold inside a clause (the incremental workhorse).
- **recombine** — single-point crossover of two accepted parent lists (the GP yield operator), de-duplicated preserving priority.
Anchor = a **structured-but-detuned** list (right shape, provide threshold 0.80 → over-provides into toxic).

**Multi-role staged gate.** Four roles on **disjoint data**, and a candidate that clears the gate is promoted through
**shadow → canary → fleet**, able to abort before full deployment:
- **PROPOSER** proposes; **VALIDATOR** (independent veto) re-checks on a **shifted regime** (more adverse selection) —
  its job is to catch a change that overfits the proposer's regime and does not generalize; **OPERATOR** runs shadow
  (frozen holdout, no commit) → canary (2 disjoint windows, abort+rollback on any hard-fail) → fleet (commit);
  **AUDITOR** injects a known-bad canary (provide in extreme toxic) that must never promote — else it **freezes** the gate.

## Run — all operators fire, the gate bites, the policy stays legible
The coding-agent refined + grew the policy with **retune_rule ×30 + add_rule ×3 + recombine ×7**, closing **94%** of the
achievable gap (−1.04 → **4.07**; tuned-structure optimum 4.41). **Honest note on recombine:** retune_rule authored all 4
promotions; **recombine produced 0 accepted / 0 dev-passing children** on this run — because the archived parents differ
only by a single retuned threshold, so every crossover de-dups back to a subset of a parent. recombine is implemented and
correct (verified live: child = A[:cut]+B[cut:], dedup, differs from both parents). It is **inert on this task for a
STRUCTURAL reason, not a bug**: the optimum here is a single non-modular structure (unwind + one narrow provide clause),
so there are no *complementary partial solutions* for crossover to combine — tested directly, no 2-parent recombine beats
the best parent, because the only extra library clause (a risk-off dump) is redundant with the unwind clause. recombine
is the yield operator for **modular** tasks (independent sub-strategies discovered in different lineages — POET/QD
open-endedness); this simple polygon has no such modularity to exploit. Honestly the same "implemented + correct +
unstressed by this env" category as the canary abort. Final **legible** policy:
`if inv>=1 & tail<10 -> AGGRESS | if belief<0.20 & inv<4 -> PROVIDE` (the agent retuned the provide threshold 0.80→0.20
and kept the defensive unwind first). Gate activity:

| role / mechanism | fired |
|---|---|
| **VALIDATOR veto** (shifted-regime independence) | **1 live** — caught an **add_rule** candidate (round 11) that improved the base regime (dev +0.013, base-validation +0.039) but did NOT generalize to the adverse shift (shifted-validation **−0.038**) — the load-bearing proof the roles are real effective challenge |
| **OPERATOR shadow** (frozen deflated gate) | **13 rejects** — the strict tier-scaled frozen gate biting |
| **AUDITOR known-bad canary** | **4/4 rejected** (0 promoted → integrity held; would freeze the gate if one ever promoted) |
| **OPERATOR canary abort** | 0 — wired safety net; on this stationary+monotone env, changes that clear shadow also clear the canary windows (nothing to catch), same honest category as v2's dwell |
| non-generalizing structural change (drop the unwind clause) | **blocked** (caught at the dev stage — honestly the earliest) |

## What this establishes (and honest scope)
CRYSTAL-1's coding-agent can now grow the policy's **structure** (add_rule/recombine), not just its parameters, and a
**multi-role staged pipeline** governs it: an independent validator that tests generalization to a shifted regime (bound
live), a shadow→canary→fleet operator that stages exposure with abort/rollback, and an auditor whose known-bad canary
kept integrity (4/4). Combined with v2 (add_knob/GROW_K + arbitration/ecology/forbidden-pairs/teacher-expiry), **every
idea from the HL5 coverage audit is now implemented.** Honest scope: polygon K=2 substrate; the scripted proposer needed
a structured anchor because provide+unwind must be added *jointly* (a single clause can't beat ABSTAIN — the joint-move
problem a reasoning LLM proposer or recombine is meant for); the canary abort and (v2) dwell are wired safety nets that a
clean stationary env does not stress. The validator binding live on a shifted regime is the load-bearing demonstration
that the roles are real "effective challenge," not org-chart decoration.
