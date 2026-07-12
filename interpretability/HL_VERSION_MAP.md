# HL loop versions — the map

The `hl_v*.py` files look like version sprawl but are a **dependency lineage, not redundant
copies**: each is a logged experiment (its `E-nn` id below), later versions `import` earlier ones
verbatim, and `crystal_ppo.py` / `personal_invest.py` / `live_executor.py` / dozens of `exp_*`
scripts import specific versions. They are **kept, not collapsed** — merging them would break ~15
import sites each and, worse, make the logbook commands unreproducible. This map is how you
navigate them.

## The lineage (bottom = foundation, top = latest)

| File | Exp | What it added | Imported by | Status |
|---|---|---|---|---|
| `hl_v2_loop.py` | — | first self-expanding governed loop (8 expansion items) | *(nothing)* | **superseded prototype** — read for history only |
| `hl_v3_loop.py` | — | the agent grows/refines POLICY STRUCTURE (add/retune/recombine rule) | *(nothing)* | **superseded prototype** |
| `hl_v4_loop.py` | — | the redesign: walk the return×legibility Pareto frontier | `hl_v4_g12` | live (polygon) |
| `hl_v4_g12.py` | E-04 | G12 non-saturated certified legibility rule — **Joseph's interpretability contribution point** | — | live |
| `hl_v4_over_r6c.py` | — | the Pareto gate over REAL R6c policy-forward replay | — | live |
| `hl_v4_over_crystal1.py` | — | **the foundation of the real-data lineage**: the full Pareto loop on real panels (Dow-29 + csi500) | 15 modules incl. `crystal_ppo`, `exp_e28`, autonomous loops | **load-bearing** |
| `hl_v5_crystal1_upgraded.py` | — | the U1+U2+U3 CRYSTAL-1 upgrades + the v4 gate | `hl_v6` | live |
| `hl_v6_crystal1_features.py` | — | belief built from the PREPROCESSED panel features (GaussianHMM) | `exp_e21`, `exp_p2`, `hl_v7`, `hl_v9`, … | **load-bearing** |
| `hl_v7_macro_belief.py` | E-11 | belief over the LIVE macro block on the verified-clean panel | `hl_v8`, autonomous loop | live |
| `hl_v8_rebalance_lane.py` | E-13 | the 5–20d rebalance lane (test the macro signal at its own horizon) | `exp_e21`, `personal_invest`, `hl_v9`, … | **load-bearing** |
| `hl_v9_fresh_oos.py` | E-15 | the fresh-OOS re-cut — the panel/belief loader most experiments call | `crystal_ppo`, `live_executor`, `personal_invest`, 10+ `exp_*` | **load-bearing (core loader)** |
| `hl_v10_bold.py` | E-23 | the bold hypotheses, each through a fresh frozen-gate lifetime | `hl_v11`, `personal_invest` | live |
| `hl_v11_gate_upgrades.py` | E-25 | the two gate upgrades (re-entry lane, carry-matched anchor) + self-audit | *(terminal)* | live |

## How to read this

- **Start at** `hl_v4_over_crystal1.py` (the real-data foundation) and `hl_v9_fresh_oos.py` (the
  loader everything imports) — those two carry most of the machinery you will touch.
- The `E-nn` column ties each file to its `EXPERIMENT_LOGBOOK.md` entry — that's where the *why*
  and the result live.
- `hl_v2_loop` / `hl_v3_loop` are the only genuine dead versions (imported by nothing). They are
  kept — not deleted — because the logbook's early entries reference them by path; treat them as
  historical record, not live code.

If you find yourself wanting to "clean up" this lineage: don't. Each file is the frozen state of
one experiment, and reproducibility (a logbook command that still runs years later) is worth more
than a tidy tree. The map is the fix; the sprawl is the record.
