# EXPERIMENT_LOGBOOK — CrystalRL interpretability track (Joseph)

*Newest entries at top. Rule: an unlogged result does not exist. Numbers are copied from the
artifact, not remembered. Template + verdict enum: [`docs/LOGBOOK_PROTOCOL.md`](docs/LOGBOOK_PROTOCOL.md).*

*Environment for entries E-39 … E-47: run against a fresh clone of the CrystalRL mirror
(`API-Capital/CrystalRL` @ `7bc57df`, itself taken from `Sqaard/CrystalRL` @ `d1dbd66`, 2026-07-23);
Python 3.14 venv with system torch 2.9.0, gymnasium 1.3.0, stable-baselines3 2.9.0, numpy 2.3.4;
`PYTHONPATH=<repo>:<repo>/scripts`. The canonical home for these entries is
`API-Capital/self-evolving-trading-bot` @ `track/crystalrl`; the mirror and the mothership carry an
identical `src/hl/pareto_gate.py` (blob `e10dfddc`), so the gate provenance is unaffected.*

*Numbering: E-01 … E-32 belong to Ivan's July programme. Entries below start at E-39 to continue the
shared sequence; a reference of the form "Ivan's E-04" means his, an unqualified "E-4x" means mine.*

---

### E-48 · 2026-08-18 · BH1-Q2 A1 — transition oversampling does NOT let cold PPO discover belief-use (NULL)
- **Who / agent:** Joseph
- **Track:** interpretability
- **Question:** Does oversampling regime-switch neighborhoods (±W=5, ≥50% of train days near a switch) let cold PPO discover belief-use at c=2.0, measured by contrast-write F?
- **Setup:** Designed 2-regime market (`gen_market`, c=2.0, P_STAY=0.97, N_DAYS=2500). Two arms, same seeds (0,1,2), cold start, 30k steps, `ent_coef=0.005`, unchanged `train()` from `exp_bh1_pressure2`. **Cold:** train on unaltered stream. **A1:** train on concatenated ±5 windows around switches (with replacement). Both evaluated on unaltered `gen_market(c, seed+500)`. Primary = `contrast_write_probe` (not the strict-delta probe). Pre-registered in `interpretability/exp_bh1_q2_a1_PREREG.md` (commit `abedd0e`) **before** any A1 code existed. Locked ceiling = this machine's BC-oracle F **0.833**.
- **Command:** `PYTHONPATH=. python interpretability/exp_bh1_q2_a1.py` → **artifact:** `interpretability/exp_bh1_q2_a1_report.json`
- **Result:** copied from the artifact. Local BC-oracle contrast-write F **0.833** (bc_acc 0.966) matches the locked ceiling. A1 train neighborhood frac **0.999 / 0.999 / 1.000** (source train 0.274 / 0.236 / 0.236; eval 0.290 / 0.259 / 0.294). n_switches source 72 / 59 / 62.
  | seed | F_cold | F_A1 | gap_cold | gap_A1 | ann_cold | ann_A1 | oracle_ann |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0276 | 0.0276 | 3.5893 |
  | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0201 | 0.0201 | 3.4143 |
  | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0886 | 0.1562 | 4.2939 |
  mean F_cold **0.0**, mean F_A1 **0.0**, gap **0.0**. 0/3 seeds F_A1 ≥ 0.50. Versions: python 3.14.6, numpy 2.3.4, torch 2.9.0, stable_baselines3 2.9.0.
- **Null tested:** A1 indistinguishable from cold (pre-registered kill: all three A1 seeds F ≤ 0.20). Kill fired. Cold baseline reproduced ~0.00 (did not move). Oracle still crushes both arms on return (ann 3.41–4.29 vs 0.02–0.16), so the designed pressure is still real — A1 failed at *discovery*, not because the objective was flat.
- **Honest caveat:** A positive would have meant "discovery is possible when switches are common," not "cold RL finds belief-use on the original market." The null is stronger than that caveat: even with ~100% of train days inside a switch neighborhood, contrast-write F stayed 0.0 and regime gap stayed 0.0 (constant dials: seed 0 exposure 0.25/0.25 both arms; seed 1 all-cash 0.0/0.0 both arms; seed 2 cold 0.25 vs A1 0.50, still unconditioned). One contrast, 3 seeds, 30k steps — not a power calculation.
- **Verdict:** **NULL** — transition rarity is not why cold RL fails to discover belief-use. The binding constraint is elsewhere (architecture, optimization, or credit-assignment geometry). A2/A3 are not licensed by this result; they need their own pre-registrations.
- **Follow-up:** none from this file. Curriculum (A2) and meta-pressure (A3) remain untested mechanisms; this null does not promote them.

### E-47 · 2026-08-11 · P(goal) table VERIFIED against its committed artifact (all 10 cells) — but the pipeline is NOT re-runnable from any repo, the published arrow chains dev→hold, and error bars are UNCOMPUTED
- **Who / agent:** Joseph
- **Track:** interpretability (verification of the personal-invest / Hello Crystal deliverable)
- **Question:** Can the published P(goal) table (start→final per investor profile) be independently reproduced, and what error bar does it carry — the prerequisite before any ">80% by 26 Aug" work.
- **Setup:** Source located: logbook entry *HL-ABLATION + PAPER · 2026-07-11*, script `interpretability/exp_hl_agent_ablation.py`, artifact `interpretability/exp_hl_agent_ablation_report.json` (present in `Hello-Crystal-Heuristic-Learning-over-CrystalRL` and mirrored in `CrystalRL-Crystal-Clear-Reinforcement-Learning`). Engine `W3_FORECAST_V1_4`; `status: RESEARCH_ONLY_UNCALIBRATED_SCENARIOS`; `accept_rule: dev J_contract improvement > 0.005; hold re-read on accept`; `total_cycles: 65`; seeds committed in source (`DEV_SEED, HOLD_SEED = 99, 199`). Objective J = P(goal) − λ·E[shortfall].
- **Command:** artifact analysis only — re-execution attempted and blocked (see caveat) → **artifacts read:** `exp_hl_agent_ablation_report.json`, `hl_profile_cycles_report.json`
- **Result:**
  | profile | published | dev base | hold base | dev final | hold final | n_accepts |
  |---|---|---|---|---|---|---|
  | conservative | 0.01→0.52 | 0.014 | 0.016 | 0.531 | **0.518** | 5 |
  | mod_conservative | 0.18→0.46 | 0.182 | 0.188 | 0.432 | **0.457** | 2 |
  | moderate | 0.06→0.48 | 0.062 | 0.068 | 0.456 | **0.476** | 4 |
  | mod_aggressive | 0.00→0.35 | 0.0 | 0.0 | 0.334 | **0.347** | 2 |
  | aggressive | 0.08→0.25 | 0.079 | 0.079 | 0.263 | **0.252** | 2 |
  All ten published cells are present in the artifact. `n_accepts` = **5/2/4/2/2** and `total_cycles` 65 match the published "accepts/13" exactly. **The published starts are DEV values; the published finals are HOLD values** — every final matches the hold column to 2dp and no final matches its dev column. Final dev−hold differences per profile, as read: 0.013 / 0.025 / 0.020 / 0.013 / 0.011 (five paired observations, seeds 99 vs 199). **No error bar is computed from these.** Five pairs cannot support a variance estimate, σ varies with p so profiles cannot be pooled, and n_paths is unverified — error bars are **UNCOMPUTED** pending the ≥30-seed sweep. Contributions probe (`hl_profile_cycles_report.json`, candidate `contrib_3pct_probe`, 3%/yr): P(goal) = **1.0 / 1.0 / 0.998 / 0.967 / 0.747** against baselines 0.014/0.182/0.062/0.0/0.079.
- **Null tested:** dev-overfit divergence. Final dev−hold gaps are **+0.013, −0.025, −0.020, −0.013, +0.011** — three of five NEGATIVE (hold beat dev), no systematic dev-optimism. Vanya's logbook claim *"dev≈hold (no dev-overfit divergence)"* is supported by the artifact. Twin and matched-random controls are present per profile (`final_controls`).
- **Honest caveat:** **this verification is artifact-reading, not re-execution — internal consistency is not validation.** What is established is that the published table faithfully reports its artifact; NOT that the underlying numbers are correct. Independent re-execution is impossible from any committed state: the forecast engine reads `data/_personal_invest_registry/{approved_daily_returns.csv, inflation_annual_worldbank.csv, shiller_cape_monthly.csv}` and **none of the three exists in any of the five cloned repos** — the registry present in `CrystalRL/data/_personal_invest_registry/` contains only `dp_policies/` (5 CSVs), and the code repos have no `data/` dir at all. `approved_daily_returns.csv` is the raw return series the whole ~3,000-path resample rests on. Also: n_paths was NOT verified in code (the "~3000" is from Vanya's prose) and is treated as unknown. No dispersion statistic is reported — the five dev−hold differences are raw observations, not a variance estimate.
- **Verdict:** four claims —
  - *The published table traces to a committed artifact with script and seeds:* **CONFIRMED** (this is NOT the CrystalScore-0.92 pattern).
  - *All ten published cells match that artifact:* **CONFIRMED**, subject to the basis defect below.
  - *The 1%→52% gain is dev-overfit:* **NULL** — tested against the artifact's dev/hold pairs and not supported.
  - *The pipeline reproduces from a clean clone (SOP §1/§2):* **INCONCLUSIVE — BLOCKED**, inputs absent from every repo.
- **Figure:** none (no new run).
- **Follow-up:** (1) Minimal ask to Vanya — commit the three registry inputs (see `BLOCKER_REPORT_pgoal.md`), then re-execute at seeds 99/199 and run the ≥10-seed error-bar sweep. (2) Report the arrow on ONE basis (dev→dev or hold→hold); the current dev→hold chaining violates the never-chain-across-splits rule even though the distortion here is ≤0.025. (3) For the >80% question: no error bar exists yet, so no claim about distinguishability from 0.80 can be made until the ≥30-seed sweep runs. Separately, the contributions probe shows >0.80 is reachable on 4/5 profiles by changing the *client's contract* (3%/yr contributions), not the strategy — a product-claim decision, not a modelling one.

### E-46 · 2026-08-11 · G-family frontier — certified legibility buy-back is NULL across G2–G12; the free→priced transition sits between G4 and G6 but is CONFOUNDED with the gate's fixed tension budget
- **Who / agent:** Joseph
- **Track:** interpretability
- **Question:** Across the designed Series-G substrate family, where does legibility stop being free, and can certified rule-adding buy it back? (Extension of **Ivan's E-04** · 2026-07-07, which tested G12 only and returned NULL.)
- **Setup:** **Ivan's E-04** loop verbatim (constants LEG_MDL 0.02, RET_TOL 0.10, MDL_NS 20, same dev_seeds formula, canary every 10 rounds), G parameterised over {2,4,6,8,10,12} × seeds {0,1,2} × 45 rounds. Substrates from `hl_v4_loop.ctors(G)` → `RegimeRotationEnv(G=G)` — ONE parametric family (the series_g PPO zips "corner G2"/family_G4/G12 are a different env+policy world and were NOT used). Every point judged by the SAME frozen gate: `src/hl/pareto_gate.py`, blob `e10dfddc` = the blob at commit **7b81994** (verified in-run; the driver aborts if it differs). Repo at `7bc57df`.
- **Command:** `python interpretability/hl_v4_family_frontier.py` → **artifact:** `interpretability/hl_v4_family_frontier_report.json`; figure `interpretability/hl_v4_family_frontier.png` via `hl_v4_family_frontier_figure.py`
- **Result:** MDL deficit at best-return point (per seed → mean) and certified legibility-raising moves:
  | G | deficits per seed | mean | certified | budget refusals |
  |---|---|---|---|---|
  | 2 | 0.007, 0.0, 0.007 | 0.005 | 0 | 0 |
  | 4 | 0.0, 0.0, 0.096 | 0.032 | 1 | 0 |
  | 6 | 0.201, 0.204, 0.166 | 0.190 | 0 | 24 |
  | 8 | 0.177, 0.213, 0.178 | 0.189 | 0 | 22 |
  | 10 | 0.075, 0.067, 0.221 | 0.121 | 0 | 17 |
  | 12 | 0.374, 0.093, 0.187 | 0.218 | 0 | 23 |
  Transition (deficit crossing **Ivan's E-04** 0.05 non-saturation threshold) sits **between G4 and G6**, i.e. BELOW the G6→G12 range the codebase narrative assumed. Deficit is **not monotone** in G (G10 mean 0.121 < G8 0.189). Both anchors green: **A1** G12 replicates **Ivan's E-04** NULL (0 certified, non-saturated); **A2** G6 seed0 reproduces **Ivan's E-01** committed artifact — best-return point return 4.03 / desc_len 4, mdl **0.201 vs Ivan's E-01 0.198** (Δ0.003, MDL_NS=20 noise). Run is **bit-identical across two independent executions** (determinism check).
- **Null tested:** (a) the certified-buy-back null — 17 of 18 runs produced zero gate-certified legibility-raising moves; (b) reproduction nulls A1/A2 against **Ivan's E-04** and **Ivan's E-01** committed artifacts, both passed. The single G4 "certified" move (seed 2, round 2, `retune`, mdl 0.085→0.0, ret 5.802→6.24) is **not counted as a positive**: `mdl_after` is exactly the 0.0 floor, it was measured at the noisier trajectory precision (MDL_NS=20) rather than the frontier's default, and it did not persist — seed 2's final best-return point sits at deficit 0.096.
- **Honest caveat:** **the transition is perfectly confounded with the instrument.** The gate's `tension_budget` is G-invariant (3.0) and produces 0 refusals at G2/G4 but 17–24 refusals at G6–G12 — i.e. it starts binding at exactly the G where the deficit rises. Substrate complexity and gate constraint therefore cannot be separated in this design, so "legibility becomes expensive because the substrate is complex" is NOT established. (Pre-registered as confound C1; C2 floor artifact flagged on G2 seed1; C3 operator-menu shrinkage visible as joint_move 0/3 selections at G2/G4 vs 34–36 at G6+.) Designed-substrate family only — instrument calibration, never market evidence.
- **Verdict:** three claims, three verdicts —
  - *Certified rule-adding can raise legibility on this family:* **NULL** (0 reliable instances across 6 substrates × 3 seeds).
  - *Legibility stops being free at a given substrate complexity:* **INCONCLUSIVE** — the transition is real in the data but confounded with the budget onset.
  - *"E-01 showed G6 is saturated / legibility is FREE on G6"* (`hl_v4_g12.py:3-5` docstring — the E-01 it names is **Ivan's**): **REFUSED** — contradicted by the committed artifact of **Ivan's E-01** itself, whose G6 best-return deficit is 0.198, failing **Ivan's E-04** non-saturation test (>0.05). The "~0" deficits in that frontier belong only to the desc_len≤2 points, which are the C2 floor artifact. Same class of error as the title mislabel on **Ivan's E-04** that SOP §4 was written to prevent.
- **Figure:** `interpretability/hl_v4_family_frontier.png` — (a) deficit vs G with the non-saturation threshold; (b) the confound, budget refusals vs G overlaid with certified moves.
- **Follow-up:** (1) To de-confound C1, re-run with `tension_budget` scaled with G (e.g. ∝ G) as a *separate* experiment — the frozen gate stays frozen; this needs Vanya's agreement since it changes the shared instrument. (2) Correct the `hl_v4_g12.py` docstring and the `HL_VERSION_MAP.md` line per SOP §4. (3) Reconcile the verdict enums: SOP §4 lists CONFIRMED|NULL|KILLED|OPEN, `agent_skill.md` and my working rules list CONFIRMED|PLAUSIBLE|NULL|REFUSED|INCONCLUSIVE — this entry uses the latter.

### E-45 · 2026-07-30 · CrystalScore FULLY reproduced — real Faithfulness input (Vanya's commit 48800da)
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** With the steering curve now supplied, does the full CrystalScore (F×S×St) compute on R6c?
- **Setup:** pulled Vanya's `48800da` ("Add CrystalScore Faithfulness inputs: steering curve + K=9 codebook") — `code_steering_curve.csv`, `r6c_codebook.npz`, `r6c_code_dictionary.csv` — into the clone; re-ran with all 5 inputs present.
- **Command:** `git checkout origin/main -- reports/firewall_upgrade/{r6c_code_control/code_steering_curve.csv,r6c_code_layer/r6c_codebook.npz,r6c_code_layer/r6c_code_dictionary.csv}` then `python interpretability/crystal_score.py`
- **Result:** **Faithfulness = 1.000** (2/2 steered codes move cash in their labeled direction monotonically). Full R6c: F 1.000 · Simul 0.244 (disc) / 0.820 (cont) · Completeness 0.244/0.141 · Controllability 0.200/0.000 · Stability 0.619. **CrystalScore STANCE = 0.151** (F×S×St), SELECTION 0.087. Matches the published R6c 0.151 headline exactly — and confirms E-43's degraded-mode inference (R6c F ≈ 1.0, so 0.151 = Simul×Stab).
- **Null tested:** reproduces the paper's R6c headline from a fresh clone with all real inputs; the degraded-mode (E-43) prediction that F≈1.0 is now confirmed by direct measurement.
- **Honest caveat:** `latent_decodability.csv` and `crisp_primitives_search.csv` are still my regenerations (E-42), not Vanya's; provenance is consistent (same Stage-4 package) but worth one cross-check against his versions if he commits them.
- **Verdict:** CONFIRMED — CrystalScore protocol fully reproduces on R6c with all authoritative inputs. The E-41 blocker is closed.
- **Follow-up:** the CRYSTAL-1 side for the apples-to-apples comparison (#5); the certified-G12-rule attempt (planned, not yet run) still needs the tagged G12 snapshot Vanya has NOT yet sent.

### E-44 · 2026-07-30 · MDL-deficit is seed-stable (designed-substrate ladder)
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** Are the MDL parsimony–fidelity deficits (G2/G4/G12) robust to the decision-tree seed, or an artifact of `random_state=0`?
- **Setup:** reused the deterministic PPO rollouts once; swept the `DecisionTreeClassifier` `random_state` 0–9 (max_leaf_nodes 8 vs 64) on the same rollout data. Wrapper: `scratchpad/mdl_seed_stability.py`.
- **Command:** `python scratchpad/mdl_seed_stability.py`
- **Result:** G2 deficit mean **0.055** (std 0.000); G4 **0.113** (std 0.002, range 0.110–0.116); G12 **0.424** (std 0.003, range 0.419–0.428) across 10 tree seeds. Matches the seed-0 headline (0.055/0.112/0.422).
- **Null tested:** the "is it just seed 0?" null — rejected: std ≤ 0.003 everywhere, so the tension G2≪G12 is not a seeding artifact.
- **Honest caveat:** this is the **designed-substrate** ladder, NOT the `R6c-latent 0.29 vs CRYSTAL-1 0.02` comparison (that one needs the R6c-vs-CRYSTAL-1 wiring; still to do). Only the tree seed was swept, not the rollout seeds.
- **Verdict:** CONFIRMED — MDL deficits are seed-stable on the designed ladder.
- **Follow-up:** run the analogous seed-stability on the R6c-vs-CRYSTAL-1 0.29/0.02 comparison once wired.

### E-43 · 2026-07-30 · crystal_score.py reproduces R6c (degraded mode; Faithfulness pending)
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** With 4/5 inputs recovered (E-42) and only the steering curve missing, can CrystalScore run and reproduce the R6c headline?
- **Setup:** local 1-line graceful-degradation fix at `crystal_score.py:539` — dropped the `np.isfinite(F)` requirement from the complexity-curve guard, since `crystal()` already needs only ≥2 finite axes (so it composes Simulatability × Stability when Faithfulness is absent). Marked in-code as a Joseph local fix.
- **Command:** `python interpretability/crystal_score.py`   → **artifacts:** `crystal_score_report.json`, `CRYSTAL_SCORE.md`, `crystal_{pareto,complexity}_curve.csv/.png`
- **Result:** runs clean. Sub-metrics (R6c frozen 2022–23, n=289, K≤9): Faithfulness **N/A** (steering curve missing); Simulatability **0.244** discrete / **0.820** continuous ceiling; Completeness@9 STANCE **0.244**, SELECTION **0.141** (raw-W 0.340 is a ~27.8× exposure artifact); Controllability cash **0.200**, selection **0.000**; Stability **0.619** (cross-seed ARI K9). **CrystalScore STANCE = 0.151** (matches the published R6c headline), SELECTION 0.087.
- **Null tested:** reproduces the paper's R6c 0.151 headline — the composite came out identical even without the Faithfulness axis, consistent with R6c faithfulness ≈ 1.0 (so 0.151 ≈ Simul × Stab).
- **Honest caveat:** the composite is a **PARTIAL CrystalScore (Faithfulness pending)** — do not report it as the certified F×S×St until `code_steering_curve.csv` lands. Also flagged: the script's hardcoded interpretation string still prints "Faithfulness is perfect" even when F=N/A — a narrative bug to fix (it's not driven by the computed F).
- **Verdict:** PLAUSIBLE — R6c sub-metrics + headline reproduce; the composite is Faithfulness-pending, not yet certified.
- **Follow-up:** re-run with the real steering curve when Vanya supplies it; then the CRYSTAL-1 side for the apples-to-apples comparison (#5).

### E-42 · 2026-07-30 · Regenerated 2 of 3 missing CrystalScore inputs; steering-curve blocker isolated
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** Can the missing `reports/firewall_upgrade/` CrystalScore inputs be regenerated locally instead of requested from Vanya — and what exactly does each need?
- **Setup:** ran the mothership generator scripts inside the CrystalRL extract (so repo-root paths resolve to CrystalRL, where the Stage-4 `..._for_Joseph` package lives).
- **Command:**
  - `python reports/firewall_upgrade/probe_r6c_latent.py`   → **artifact:** `reports/firewall_upgrade/r6c_latent_probe/latent_decodability.csv` (+ `latent_probe_report.json`)
  - `python reports/firewall_upgrade/r6c_crisp_primitives_search.py`   → **artifact:** `reports/firewall_upgrade/r6c_code_layer/crisp_primitives_search.csv` (+ summary)
  - `python reports/firewall_upgrade/r6c_code_control_demo.py`   → **FAILED**
- **Result:** ✅ `latent_decodability.csv` (1319 B) and ✅ `crisp_primitives_search.csv` (4604 B, verdict "crisp achievable": behavior+trig HDBSCAN k=2, silhouette 0.534) both regenerate from the Stage-4 package alone — they read only `hidden_activations/r6c_frozen_hidden_activations.npz` + `frozen_test_behavior_log_daily.csv`, which are present. Copied the two JSONs `code_control_report.json` + `r6c_code_layer_manifest.json` from the mothership. ❌ `code_control_demo.py` (→ `code_steering_curve.csv`) fails: needs (a) module `build_r6c_frozen_test_rollout_for_joseph` (present in mothership/scripts — fixable via PYTHONPATH), (b) `r6c_code_layer/r6c_codebook.npz` + `r6c_code_dictionary.csv` — **absent from all 16 repos**, (c) `artifacts/stage6b/cloud_packages_r6c_stage6b_counterfactual_20260605/…` — **only the generator scripts exist** (`run_stage6b_r6c_counterfactual_rollout.py`, `..._huawei_package.py`); the data package is a Huawei-cloud output not in any repo. Re-running `crystal_score.py` with 4/5 inputs present still crashes at `:570` — the missing `code_steering_curve.csv` feeds Faithfulness (`F`), so the same empty-`complexity_pts` path is hit.
- **Null tested:** n/a (tooling/regeneration, not a scientific run).
- **Honest caveat:** the two copied JSONs came from the mothership's `firewall_upgrade/` and may be from a different run than the regenerated CSVs — provenance should be reconciled once the full set is consistent. The regenerated numbers have not been checked against any committed reference (none exists).
- **Verdict:** INCONCLUSIVE — CrystalScore still blocked, but the blocker is now **isolated to one input** (`code_steering_curve.csv`) and its precise upstream needs (codebook files + a compute-generated stage6b package).
- **Follow-up:** minimal ask to Vanya — either commit `code_steering_curve.csv` (+ `r6c_codebook.npz`, `r6c_code_dictionary.csv`), OR provide the stage6b package + codebook + compute so the counterfactual rollout can be run locally. Separately: the 2-line guard fix in `crystal_score.py` so a missing Faithfulness input reports `N/A` instead of crashing.

### E-41 · 2026-07-30 · CrystalScore protocol — BLOCKED (missing faithfulness inputs)
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** Does the CrystalScore protocol (Faithfulness × Simulatability × Stability) reproduce on the R6c frozen 2022–23 rollout from a fresh clone?
- **Setup:** `interpretability/crystal_score.py` on the shipped R6c frozen package.
- **Command:** `python interpretability/crystal_score.py`   → **artifact:** none written (crashed before summary/report)
- **Result:** `ValueError: All arrays must be of the same length` at `crystal_score.py:570` (building `OUT_COMPLEXITY_CSV`). Root cause: the Faithfulness/probe/steering inputs the script expects under `reports/firewall_upgrade/` are **absent from the clone** — all 5 missing: `r6c_latent_probe/latent_decodability.csv`, `r6c_code_layer/crisp_primitives_search.csv`, `r6c_code_layer/r6c_code_layer_manifest.json`, `r6c_code_control/code_steering_curve.csv`, `r6c_code_control/code_control_report.json` (the whole `reports/firewall_upgrade/` dir does not exist). With Faithfulness `F=NaN` / `X_std=None`, the loop at `:539` (`if X_std is not None and np.isfinite(F)`) is skipped so `complexity_pts_stance/sel` stay empty, but the CSV block at `:570` is guarded only on `comp.get("stance")` and crashes on the empty lists. (Stage-4 frozen inputs — `hidden_activations/r6c_frozen_hidden_activations.npz`, `frozen_test_behavior_log_daily.csv` — ARE present.)
- **Null tested:** n/a — did not reach a measurement; this is an environment/handoff failure, not a scientific run.
- **Honest caveat:** two causes are entangled — a genuine **missing-input (handoff) gap** AND a **code-robustness bug** (the complexity-CSV block doesn't guard on the same condition as the loop that fills it, so a missing-input case surfaces as a cryptic pandas error instead of a clean `N/A`). Cannot confirm the 0.151-vs-0.938 headline until the inputs are supplied.
- **Verdict:** INCONCLUSIVE — blocked on a missing handoff input (see `../BLOCKER_REPORT_for_Vanya.md`); needs Vanya to ship `reports/firewall_upgrade/` (or confirm where these live in the mothership).
- **Follow-up:** request the 5 files from Vanya; re-run; separately, propose the one-line guard fix so the missing-input case reports `N/A` instead of crashing.

### E-40 · 2026-07-30 · MDL parsimony–fidelity deficit reproduces (designed complexity ladder)
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** Does the MDL parsimony–fidelity deficit ruler (CrystalScore-v2 Axis A) reproduce from a fresh clone?
- **Setup:** `interpretability/mdl_fidelity_deficit.py` on the designed Series-G substrates (G2/G4/G12).
- **Command:** `python interpretability/mdl_fidelity_deficit.py`   → **artifact:** `interpretability/mdl_fidelity_deficit_report.json`
- **Result:** `corner_G2`: deficit **0.055** (simul@K9 0.934, ceiling64 0.989, 3 actions); `family_G4`: deficit **0.112** (simul@K9 0.864, 5 actions); `family_G12`: deficit **0.422** (simul@K9 0.527, 14 actions). Confirms the parsimony↔fidelity tension: the near-zero-deficit corner (G2) is the degenerate low-complexity end; deficit rises with vocabulary size.
- **Null tested:** the tension itself IS the finding vs the naive "0.938 corner is ideal" reading — the deficit is 0 only on the low-complexity leg, ≫0 on G12. Reproduced the shipped narrative.
- **Honest caveat:** this reproduces the **designed-substrate** ladder (G2/G4/G12); it is **not** the `R6c-latent ≈0.29 vs CRYSTAL-1-soft-tree ≈0.02` comparison from PAPER.md — that comparison needs the same missing `r6c_latent_probe` inputs as E-41, so the specific "0.29 vs 0.02" seed-stability check is also blocked.
- **Verdict:** CONFIRMED (reproduction of the designed-substrate MDL machinery) — matches the shipped report.
- **Follow-up:** seed-stability sweep of the tree fits on G12; and the R6c-vs-CRYSTAL-1 0.29/0.02 check once the probe inputs arrive (E-41).

### E-39 · 2026-07-30 · hl_v4_loop reproduces (good-first-issue / onboarding)
- **Who / agent:** Joseph via Claude
- **Track:** interpretability
- **Question:** Does the certified return×legibility HL loop (`hl_v4_loop.py`) reproduce its Pareto frontier + firewall audit from a fresh clone? (README/ROADMAP good-first-issue.)
- **Setup:** `interpretability/hl_v4_loop.py`, seed 0, substrates G6→G12.
- **Command:** `python interpretability/hl_v4_loop.py`   → **artifact:** `interpretability/hl_v4_loop_report.json`
- **Result:** G6 frontier (seed0): `[{return 0.0, desc_len 1, mdl_deficit 0.0}, {return 1.35, desc_len 2, mdl_deficit 0.0}, {return 4.03, desc_len 4, mdl_deficit 0.201}]`. Adversary vetoes 0 (F2); alpha-wealth refusals 29 (F3); tension-budget refusals 24 (F4/F11); canary caught **27/27**, escaped 0, gate not compromised (F12); mechanism prior transfers G6→G12: warm best return 2.87 vs cold 2.61 (F9).
- **Null tested:** the loop rejected 27 dominated/no-gain moves a return-only gate would have accepted (the gate's refusals ARE the null it enforces); frontier is 3 non-dominated points, not a scalar.
- **Honest caveat:** benign warning `sklearn ... y_pred contains classes not in y_true` (unused-class in a metric); this is a **demo-substrate** run (Series-G polygon), not real-market evidence — polygon results are instrument calibration only. This reproduces **Ivan's E-01** "worked example", previously run by Ivan's agent; it is now reproduced independently.
- **Verdict:** CONFIRMED (reproduction) — frontier + firewall audit reproduce; matches the shipped `hl_v4_loop_report.json` narrative.
- **Follow-up:** attempt a certified legibility rule on the HARD substrate G12 (needs Vanya's tagged G12 snapshot + frozen firewall commit-hash per COLLABORATION.md).
