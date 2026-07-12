---
name: research-writing
description: >
  Turn a just-produced result into paper-quality material in the self-evolving-trading-bot project: clear
  scientific prose in the project's voice, a FIGURE or table (matplotlib → PNG, embedded), literature grounding
  with citations + Evidence-Status tags, and — critically — keeping PAPER.md / ROADMAP.md in sync in the SAME
  turn. Trigger when writing or revising PAPER.md, a report in reports/, a figure/table/schema, a citation or
  lit-review, or a progress deck; and after any experiment whose result should reach the paper. Composes with
  the experiment-logger skill (that records the raw honest entry; this makes it publication-grade and propagated).
  Use PROACTIVELY — a result that isn't in the paper with a figure and its null does not yet count as done.
---

# Research writing

The deliverable of this project is a *paper* (write-the-paper-first), read by a collaborator and a professor and
aimed at publication. So every result must become clear prose + a figure + a citation-grounded claim, and the
paper must never go stale. The binding rule is the same as everywhere: **lead with the null; a figure that hides
the null is forbidden.**

## 1. Style — scientific and clear (match the existing voice)

- **Lead with the result, then the method.** One-sentence claim first; setup after.
- **Define every term on first use**; keep notation consistent with `PAPER.md`'s glossary (`b_t`, belief,
  MDL deficit, VoI, firewall). Don't rename a thing mid-document.
- Short declarative sentences. **No hype words** ("powerful", "novel", "state-of-the-art") — state the number.
- Every quantitative claim carries its **null** and its **one binding caveat** (sample size / priced-in /
  single-path / saturating / off-manifold). A claim without a null is not written.
- Verdicts use the fixed enum (CONFIRMED / PLAUSIBLE / NULL / REFUSED / INCONCLUSIVE), never "looks good".

## 2. Figures, tables, schemas (Eisner: every result gets "axes and captions")

- **Make a figure for every result**, saved to `reports/figures/<Enn>_<slug>.png` and embedded in the paper /
  report / logbook. matplotlib only; inline all data (no external assets). Consistent style:
  ```python
  import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
  plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
  # ... one relationship per figure; label axes with units; title = the CLAIM, not "Figure 1"
  fig.savefig("reports/figures/E03_newhigh_null.png", bbox_inches="tight")
  ```
- **The caption states the claim AND the null** ("New-high-arming (orange) tracks buy-and-hold (blue) — no
  drawdown cut; risk-adj Δ CI [−0.04, +0.03] straddles 0 → NULL"). One relationship per figure; never a figure
  that hides a straddling CI or a cherry-picked axis range.
- Tables in GitHub-markdown; a numeric column is monospace-aligned. Schemas as a small ASCII diagram or an
  inline SVG — readable in the repo without rendering.
- **A null result gets a figure too** — the near-identical curves ARE the evidence. Don't skip figures for negatives.

## 3. Literature grounding + citation (before claiming novelty)

- For each nontrivial claim, attach a **mechanism** and a **citation** with an Evidence-Status tag:
  `[Established]` / `[Plausible]` / `[Speculative]`. Keep the running bibliography in `reports/REFERENCES.md`.
- **Lit-check before claiming an effect or novelty** — is this a known result? Project anchors to cite:
  priced-execution / adverse selection = Glosten–Milgrom 1985; vol-managed fragility = Moreira–Muir 2017 +
  Cederburg et al. 2020; regime-switching = Hamilton 1989; belief-state control = Kaelbling, Littman &
  Cassandra 1998; soft trees = Frosst & Hinton 2017; MDL = Rissanen 1978; do-intervention = Pearl 2009.
- Cite the project's own prior result when a claim rests on it (e.g. "VoI=0 on daily, [B4]").

## 4. Keep the paper in sync — the propagation rule (fixes the #1 friction)

**In the same turn a result is produced,** propagate it — do not batch for later:
1. `EXPERIMENT_LOGBOOK.md` — the honest entry (experiment-logger skill).
2. `PAPER.md` — the relevant results **table cell** (fill the ☐) + one sentence in the text; add the figure.
3. `ROADMAP.md` — tick/strike the TODO or Issue this result closes/opens.
4. `REFERENCES.md` — any new citation.
5. **Check numbering + cross-links are consistent** (Enn ids, table refs) before finishing — a stale paper or a
   duplicated id is a defect. (This turn's E-02/E-04 collision is the failure mode to prevent.)

## 5. Self-improvement (skillLens → skillOpt, every ~10 uses)

Audit the last ~10 write-ups: are figures missing? captions omitting the null? claims uncited? the paper stale
vs the logbook? Propose ONE change to this skill that fixes the top finding; adopt only if it demonstrably
improves the next write-up (re-do one under the new rule and check). Record the finding + decision as a dated
note at the bottom of `reports/REFERENCES.md` under "## research-writing skill evolution". Honesty applies to
the skill itself: whether it helped is judged next cycle, not asserted now.
