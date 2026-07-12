# Code map — clone and continue

This repo carries the **complete interpretability-track code** extracted from the lab's
mothership repo, verified to import and run from a fresh clone. Excluded on purpose: the
heuristic-learning **coding-agent** machinery (proposal executors, autonomous loops, gate
harnesses) and the **Hello Crystal** product line (personal-invest contract/engine/product) —
those stay in the mothership. Everything a metrics/atlas/intervention experiment needs is here.

```bash
pip install -r requirements.txt
python -c "import sys; sys.path[:0]=['.','scripts']; import interpretability.mdl_fidelity_deficit"
# runs the frozen-log headline in ~30 s -> you are set up correctly
```

## Layout

| Path | What lives there |
|---|---|
| `src/crystal/` | **The CRYSTAL-1 core**: `belief_filter.py` (the Hamilton filter of Eq. 2), `soft_tree_policy.py` (the Frosst–Hinton actor, Eq. 3), `governor.py` (the kNN manifold governor), `writ_ladder.py` (C0–C6 certified-write governance + ledger), `universe.py`. |
| `src/series_g/` | **The designed market** (§8 of the paper): `regime_pomdp.py`, `family_env.py`, `generators.py`, phase0–3 gates/falsifiers/anchors — the environment that "rewards a clean mind". |
| `src/hl/` | The legibility-loop *library* (not the coding agent): `modular_rule_policy.py`, `tension.py`, `pareto_gate.py`, `mechanism_bandit.py`, substrates incl. `substrate_hard.py` (G12). |
| `src/ppo/`, `src/data/`, `src/evaluation/` | The CHRL-lineage training/eval stack the exporters and cross-scores lean on (instrumented PPO, weight env, firewall, ghost portfolios, Dow-30 sectors). |
| `interpretability/` | 58 experiment scripts + their `*_report.json` results + result CSVs (see the guide below). |
| `scripts/` | `extract_stage0_1_hidden_state_package.py` — the hidden-activation exporter. |
| `data/_dow_extended/` | The **real US daily panel** (29 Dow names + VIX/trend/credit/short-rate macro columns, 2000–2026) + per-ticker caches. Rebuild/extend anytime: `python interpretability/build_dow_extended_panel.py` (fetches public quotes). |
| `data/_personal_invest_registry/dp_policies/` | The **readable DP policy tables** (state → action CSVs) — the object H1's simulatability psychophysics lives on. |
| `artifacts/stage4/R6c_*_for_Joseph/` | **The frozen R6c package, prepared for you**: `frozen_test_behavior_log_daily.csv` (the 2022–23 behavior log the R6c-side CrystalScore is computed on), `hidden_activations/` (64-d latent exports + unit stats), `test_codes.csv`, rollout npz, manifest, its own README. `feature_scalers_frozen/fold_2021/model_ready.csv` ships **gzipped** (GitHub size limit): `gunzip -k` it before running anything that rebuilds the env features. |
| `artifacts/stage0_1/` | The csi500 P22/R6c score JSONs + the completed-run metrics table used by the frontier/anchor scripts. |
| `contracts/crystal1_knob_registry.yaml` | The 5-lever command surface (the C-6 registry). |
| `reports/CRYSTAL1_*.md` | The full C-ladder build/controllability reports (B0–B5, C1–C6, debts, the final consolidated report) — the provenance of every number in the paper. |
| `reports/W7_HL_NORTHSTAR.*` | The interpretability north-star registry (the metrics you own). |
| `.claude/skills/` | `experiment-logger` + `research-writing` — your Claude auto-loads these; `agent_skill.md` references them. |

## Entry points by theme (all verified to import from a fresh clone)

**CrystalScore & complexity metrics**
- `interpretability/crystal_score.py`, `crystal_score_crystal1.py` — the F×S×St scalar, both agents.
- `interpretability/cross_policy_crystal.py`, `cross_arch_crystal.py` — score sweeps across
  policies/architectures (results: `cross_*_crystalscore.csv`, `*_report.json`).
- `interpretability/mdl_fidelity_deficit.py` — the MDL deficit (Eq. 16).
- `interpretability/l0_bits_per_action.py` (+ `l0_csi500_anchors.py`) — the dynamic-complexity ruler (h_mu, C_mu, E).
- `interpretability/k_rashomon.py`/`_v2`, `k_codelevel.py`, `h1_tiny_surrogate.py`, `h2_mode_entropy.py` — story-budget and surrogate probes.

**The C-ladder (causal writes, lie detector, composition)**
- `c1_leakage_certificate.py` → `c2_filter_grounded_c1.py` → `c3_certify_against_world.py` →
  `c5_sign_epistasis.py` + `c5_debts.py` → `c_alpha_machine.py`; `iv10_epistasis_pairedwrite.py`.
- `crystal1_c4_treehead.py`, `crystal1_m6_softtree.py` — the tree-as-head at parity.
- `voi_gate.py`, `b4_real_voi.py`/`_v2` — the value-of-information gate (the fence of §8).

**The four levers on the real panel (paper §6)**
- `crystal_ppo.py` — the R/T/I/A heads (trained checkpoints in `crystal_ppo_models/`).
- `exp_e28_all_levers.py` — the all-levers battery; `exp_e27c_prob_diagnostics.py` — the
  flat-preference (fake competence) diagnostic; `exp_e21_transparency_audit.py` — the audit
  (naming faithfulness / 10-seed stability / re-discovery).

**The designed market & the legibility frontier**
- `series_g_corner_test.py`, `series_g_family_sweep.py` (+ `src/series_g/`).
- `hl_v4_loop.py`, `hl_v4_g12.py`, `hl_tension_blind.py` — the certified return↔legibility
  tension (E-04); `grow_k.py`, `family_curve_ppo.py`, `ontogeny_corner.py`, `r_crystallization.py`.
- `b2_multiseed.py`, `b5_crystallize.py`, `b5b_k14.py`, `crystal1_b1*.py`, `crystal1_b3_riskmode.py`,
  `crystal1_b4_bridge.py`, `crystal1_oracle_ceiling.py` — the B-series builds.

**Support modules you will rarely call directly** — `hl_v4_over_crystal1.py`,
`hl_v5_crystal1_upgraded.py`, `hl_v6_crystal1_features.py` (the GaussianHMM belief),
`hl_v7_macro_belief.py`, `hl_v8_rebalance_lane.py`, `hl_v9_fresh_oos.py` (panel/belief loaders) —
kept because the experiment scripts import them; they are dependency code, not your API.

## Conventions

- Run everything **from the repo root**; scripts locate the root relative to their own file
  (`ROOT = HERE.parent`) and expect `data/`, `artifacts/`, `src/` beside them.
- Every experiment writes `<name>_report.json` next to itself — the existing reports ARE the
  numbers cited in the paper; re-running should reproduce them (same seeds are in the scripts).
- After every run: one entry in your `EXPERIMENT_LOGBOOK.md` per `docs/LOGBOOK_PROTOCOL.md`.

## Known not-included (and what that blocks)

| Missing piece | Size/where | What it blocks |
|---|---|---|
| Canonical P22/W2 checkpoints (`D:/Interpretable_CHRL/stage0_1/current`) | external drive | re-*scoring* P22 from raw checkpoints in `cross_arch_crystal.py`; its computed outputs are included |
| `artifacts/action_vq/A67_*/ja67_joint_controls_daily.csv` | 183 MB | the A67 branch of the cross-arch sweep (results included) |
| The coding-agent machinery (`heuristic_agent_r6c/`, autonomous loops, HCS harness) | mothership | proposing/certifying *machinery changes* — Ivan's track; ask when you need a loop run |
| The Hello Crystal product line (`personal_invest_*`, gates W2–W9) | mothership | the product paper's experiments; the DP policy *tables* you need are included |

If an experiment you want hits one of these, say so on Sunday — extracting more is a
ten-minute job, we just kept the starter repo lean.
