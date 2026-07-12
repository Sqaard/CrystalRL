# The experiment-logbook protocol

**The rule: update the logbook after EVERY experiment — an unlogged result does not exist.**
The full, living logbook is in the mothership repo
([self-evolving-trading-bot/EXPERIMENT_LOGBOOK.md](https://github.com/Sqaard/self-evolving-trading-bot/blob/research/exec-econ-newhigh-cycle/EXPERIMENT_LOGBOOK.md))
— it is already filled with dozens of entries; copy their tone and granularity. Start your own
`EXPERIMENT_LOGBOOK.md` in this repo with the same template, newest entries at the top.

## The template (copy verbatim, then fill)

```markdown
### E-<NN> · <YYYY-MM-DD> · <short title>
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** <the one question this run answers>
- **Setup:** <substrate / model / data window / what changed vs the anchor>
- **Command:** `<exact command, incl. seed>`   → **artifact:** `<report path>`
- **Result:** <the key numbers, verbatim from the artifact>
- **Null tested:** <what you compared AGAINST (anchor, placebo, best-of-N floor, prior run) and did it fail?>
- **Honest caveat:** <the single most important reason to distrust this>
- **Verdict:** <CONFIRMED | PLAUSIBLE | NULL | REFUSED | INCONCLUSIVE> — <one clause why>
- **Follow-up:** <next experiment this implies, or "none">
```

**Verdict enum** — pick the honest one, never "looks good":
- **CONFIRMED** — passed the frozen gate / adversarial check; reproduced.
- **PLAUSIBLE** — a real signal not yet certified (e.g., a lift that hasn't cleared the hold-out).
- **NULL** — tested against its null and did not beat it (a *contribution*; log it proudly).
- **REFUSED** — a gate rejected it with a reason (record the reason).
- **INCONCLUSIVE** — underpowered / a bug / needs a rerun (say which).

## The honesty contract every entry obeys

- **Numbers are copied, not remembered** — read them from the artifact you just produced;
  unknown fields say `unknown`, never a guess.
- **Exact command + seed** — if it can't be reproduced from the entry, it didn't happen.
- **One caveat, the worst one** — not a disclaimer wall; the one thing a skeptic would attack.
- **A loop/replay lift is not a result** — PLAUSIBLE until the gate confirms it.
- **State the null or it's not done.**

## A real example (from the mothership logbook, 2026-07-12)

> ### CRYSTALRL-PAPER + JOSEPH-HANDOFF · 2026-07-12 · the second write-paper-first draft (CrystalRL) + the handoff kit
> - **What (no new experiments — a synthesis + handoff artifact set):** `paper_crystalrl/` — "CrystalRL —
>   Crystal Clear Reinforcement Learning" (Overleaf-ready main.tex, 11 pp): the CHRL→CRYSTAL-1 journey
>   with the FINAL comparison table (…), the CrystalScore protocol (F×S×St @ K≤9; R6c 0.151 vs
>   CRYSTAL-1 0.938 — the whole 6× gap is Simulatability …), the four influence levers R/T/I/A each
>   with a measured outcome + the NI-bar refusal (E-27/E-28) …
> - **Honest caveat:** the paper synthesizes EXISTING logged results — no new runs; the numbers trace to
>   CRYSTAL1_* reports and E-21/E-27/E-28 entries; three background survey agents stalled, so the key
>   numbers were verified directly against the source reports instead …
> - **Verdict:** n/a (documentation artifact) — claims inherit the verdicts of their source entries.
> - **Follow-up:** Ivan uploads both papers to Overleaf …; Joseph reshapes H1–H12 into his six-month
>   program (first read: ONBOARDING → this paper → the `\joseph{}` todos).

Why this discipline exists, in one sentence — Feynman (*Cargo Cult Science*, 1974):
**"The first principle is that you must not fool yourself — and you are the easiest person to fool."**
