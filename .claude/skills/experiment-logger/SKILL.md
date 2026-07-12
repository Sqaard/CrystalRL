---
name: experiment-logger
description: >
  Fill the shared EXPERIMENT_LOGBOOK.md with an honest, reproducible entry after ANY experiment, run,
  loop, ablation, or measurement in the self-evolving-trading-bot project. Trigger whenever the user says
  "log this", "logbook", "record this run", after running an interpretability / Heuristic-Learning loop or
  a firewall/certification pass, or whenever a result was just produced that another person (Ivan, Joseph, or the
  professor) would need to reproduce or trust. Also owns the periodic self-improvement of its own template
  (skillLens audit + skillOpt update). Use it PROACTIVELY — an unlogged result does not exist.
---

# Experiment logger

Every result in this lab must be reproducible and honest. This skill turns a just-finished experiment into a
LOGBOOK entry so that six months later (or the other collaborator, or the professor) can see *exactly how a
number was produced and how much to trust it.* The governing principle is **be honest: lead with the null.**

## How to log (do this after an experiment)

1. Gather the facts from the actual run — do **not** invent numbers. Read the report JSON / stdout you just
   produced. If a field is unknown, write `unknown`, never a guess.
2. Append **one entry** to `EXPERIMENT_LOGBOOK.md` using the template below. Newest entries go at the **top**
   of the "Entries" section (reverse chronological).
3. Fill the honesty fields for real: **Null tested**, **Honest caveat**, and a **Verdict** from the fixed
   enum. If you cannot name the null you tested against, the experiment is not finished — say so.
4. Cross-link: if the result changes a claim in `PAPER.md`, note the paper section; if it closes/opens a
   TODO, note the `ROADMAP.md` / Issue id.
5. **Propagate in the same turn** — hand off to the **research-writing** skill: update the `PAPER.md` table
   cell + one sentence, tick/strike the `ROADMAP.md` TODO, and add a **figure** for the result. A logbook entry
   whose result never reaches the paper (with a figure and its null) is only half-done. (Also: check the `E-nn`
   id doesn't collide with an existing one — this cycle's E-02/E-04 mix-up is the failure to prevent.)

### Entry template (copy verbatim, then fill)

```markdown
### E-<NN> · <YYYY-MM-DD> · <short title>
- **Who / agent:** <Ivan | Joseph> via <coding-agent>
- **Track:** <model-&-data | interpretability>
- **Question:** <the one question this run answers>
- **Setup:** <substrate / model / data window / what changed vs the anchor>
- **Command:** `<exact command, incl. seed>`   → **artifact:** `<report path>`
- **Result:** <the key numbers, verbatim from the artifact>
- **Null tested:** <what you compared AGAINST (anchor, placebo, best-of-N floor, prior run) and did it fail?>
- **Honest caveat:** <the single most important reason to distrust this — sample size, priced-in, seed-fragile, single-path, saturating metric, off-manifold, etc.>
- **Verdict:** <CONFIRMED | PLAUSIBLE | NULL | REFUSED | INCONCLUSIVE>  — <one clause why>
- **Follow-up:** <next experiment this implies, or ROADMAP/Issue id, or "none">
```

**Verdict enum** — pick the honest one, never "looks good":
- **CONFIRMED** — passed the frozen firewall / adversarial check; reproduced.
- **PLAUSIBLE** — a real signal but not yet certified (e.g. a loop lift that hasn't cleared the hold-out).
- **NULL** — tested against its null and did not beat it (a *contribution*, log it proudly).
- **REFUSED** — the gate rejected it with a reason (record the reason).
- **INCONCLUSIVE** — underpowered / a bug / needs a rerun (say which).

## Rules the entry must obey (the honesty contract)

- **A loop/replay lift is not a result** — only a certified change is. Say PLAUSIBLE, not CONFIRMED, until the
  firewall passes it.
- **State the null or it's not done.** Every CONFIRMED/PLAUSIBLE entry names what it beat.
- **One caveat, the worst one.** Not a disclaimer wall — the single reason a skeptic would push back.
- **Exact command + seed.** If it can't be reproduced from the entry, it didn't happen.
- **Numbers are copied, not remembered.** Read them from the artifact in this turn.

## Self-improvement: skillLens → skillOpt (run every ~10 entries, or when asked)

This template must get better as the lab learns. Do NOT edit it ad-hoc; use the gated loop the project uses
for its own controllers:

1. **skillLens (audit).** Read the last ~10 logbook entries and ask: *what honest information is consistently
   missing or weak?* Examples of real findings: entries that never name a null; "Result" fields that
   paraphrase instead of quoting the artifact; caveats that are generic ("results may vary") instead of the
   binding one; missing seeds. Write the audit as a short note (what's weak, with entry ids as evidence).
2. **skillOpt (gated update).** Propose ONE change to this SKILL (a new required field, a sharper prompt, a
   verdict-enum tweak) that would have fixed the audit finding. Adopt it **only if** it demonstrably improves
   the next entries — i.e. re-log one past experiment under the new template and check the missing information
   now appears. If it doesn't help, revert. Record the skillLens finding + the skillOpt decision as a dated
   note at the **bottom** of `EXPERIMENT_LOGBOOK.md` under "## Skill evolution log", so the template's own
   history is honest and auditable — exactly like the agent's teacher memory.

The point: the logbook that records the experiments is itself an experiment that improves under evidence.
