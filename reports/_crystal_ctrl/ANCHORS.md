# CRYSTAL-1 controllability plan — shared ANCHORS (real file paths; read this first)

Goal of the plan set (A–Д): **develop CRYSTAL-1's controllability** (the "build from the brain" path). Near-term
deployable substrate is still **HL-over-R6c** (R6c currently beats CRYSTAL-1 on controllability); CRYSTAL-1 is the
born-legible successor whose **K-simplex belief IS the writable command surface**. Do not re-litigate that.

## R6c / CHRL (the incumbent whose control surface we critique)
- **Policy architecture:** `src/ppo/dirichlet_policy.py` — `DirichletActorCriticPolicy`, `SharedStockMlpExtractor`,
  `action_net`/`value_net`; `src/ppo/w1_budget_trader_policy.py` — `BudgetPMActorCritic` (Beta cash/risk target +
  review horizon), `GraphHierarchicalAssetEncoder` (group + residual-correlation stock encoder).
  Net: `pi:[128,64]` → **64-d penultimate → action_net (Dirichlet alphas)**; `vf:[256,128]`. Hierarchy = root
  cash/risky split (Beta) + per-stock workers + group tier.
- **Control knobs (~214 raw → ~20 T0 + ~3 T1):** `heuristic_agent_r6c/contracts/knob_registry.yaml` (tiered
  registry); `configs/stage0_1_active_r_pipeline.yaml` (reward weights, `turnover_cap`, `lambda_cash`, `group_caps`,
  `top_k_buy/sell`, `correction_penalty`, recovery/`risk_stress` thresholds, PPO params);
  `configs/_r6c_latent_ab.yaml` (`root_split_latent_action`, `num_codes`, `residual_mix`, `cash_blend`).
- **Latent-action layer / steering (R6c+):** `src/ppo/stage0_1_weight_env.py` (latent codebook load);
  `reports/firewall_upgrade/r6c_code_control_demo.py` + `reports/firewall_upgrade/r6c_code_layer/`
  (`r6c_codebook.npz`, `r6c_code_dictionary.csv`, `r6c_crisp_primitives_summary.json`). Steering = FORCE the 64-d
  penultimate toward a K-means code centroid (code-level intervention), α∈[0,1] interpolation.
- **Firewall / HCS / ghost:** `src/evaluation/firewall.py` (`deflated_sharpe_ratio`, `OODGate` Mahalanobis,
  `safe_alpha` on-manifold step, `block_bootstrap_ci`); `src/evaluation/ghost_portfolios.py` (`GhostLedger`
  original|repaired|opposite|no_trade); `scripts/hcs_policy_forward_search_loop.py` (cheap controller search +
  failure tags).
- **R6c reports:** `reports/HL5_FINAL_K_KnobRegistry_v2.md` (the ~214→~20 audit), `reports/MODEL_REGISTRY.md`
  (canonical on `D:\Interpretable_CHRL\...R6c_root_K20_stock_K5_...`; C: = frozen exports / retrains only; **canonical
  P22 = PERSISTER**, csi500-retrain "churner" is an artifact), `reports/R6C_STAGE1_IMPLEMENTATION_NOTE.md` (64-d
  latent replay), `reports/r_k_window_analysis/R6C_GROUP_RISKAWARE_TOPK_IMPLEMENTATION_NOTE.md`.

## CRYSTAL-1 (the born-legible successor we develop)
- **L0 role-contract:** `src/crystal/universe.py` — `RoleContractError`, `UniverseSpec` (`to_json/from_json`,
  `breadth_counts`), `RoleAdapter` (loud fail on missing role), `RoleGate`/`BoundGate`. Law: no column-name keys,
  no silent absent→0, breadth as fractions of N.
- **L1 learned neural Bayes belief filter (THE K-simplex belief):** `src/crystal/belief_filter.py` —
  `NeuralBayesFilter(K, A_obs)`: learnable `logits_T` (K×K), `logits_E` (K×A_obs), `logits_p0` (K); `mats()` softmax
  → probability matrices; `forward()` recursive Bayes `b_pred=b@T; b=(b_pred*E[:,o])/Σ`; `train_filter()` maximizes
  filtering loglik with **NO regime labels**; param recovery + belief-MAE selftests. Belief `b` ∈ **K-simplex**
  (float32 (K,)). **Writable command surface:** belief is directly SET in the obs (`obs_vec(b,t,iv,burst)` →
  `2*b-1`), envelope-scoped (visited states only). R6c can only steer via 64-d latent forcing; CRYSTAL-1 steers via
  a scalar, K-**named** belief write (readable regime semantics).
- **Env / polygon:** `src/series_g/regime_pomdp.py` (`RegimePOMDP`: BENIGN/TOXIC, QUIET/BURST, PROVIDE/ABSTAIN/
  AGGRESS; `gm_threshold()`), `src/series_g/multiasset_env.py` (`MultiAssetRegimePOMDP`), `src/series_g/corner_ppo_n1.zip`.
- **B0–B5 drivers:** `interpretability/crystal1_b1.py` (`LearnedBeliefEnv` + PPO + 5-gate battery; v3 8k SSL passes
  5/5), `interpretability/b2_multiseed.py` (C1 replicates 3/3; **C3 uniqueness-tracks-fidelity RETRACTED**; C4
  C*≈K bend ironclad; C5 price-of-learning=0), `interpretability/crystal1_b3_riskmode.py` (3-mode exposure; Dow DD
  halved, csi500 fails — enable-table conditional), `interpretability/crystal1_b4_bridge.py` (VoI=0 on daily proxies),
  `interpretability/b5_crystallize.py` (reward-shaping-for-legibility FALSIFIED → structural mechanism needed).
- **Battery (5 hard gates):** `interpretability/certified_battery_v2.py` — (1) L0 bits/action+structure, (2)
  reactive≫autoregressive, (3) belief-N7 asymmetry, (4) HC-1 ablation hurts, (5) J-intervention fidelity ≥0.67 +
  per-class compliance; diagnostics = state-aware Rashomon, sim@K, steerability dose-response. Helpers:
  `cross_policy_crystal.py` (`behavioral_complexity_dynamic`), `series_g_corner_test.py` (`rashomon`, `n7_grouped`,
  `autoreg_sim`). Epistasis de-risk: `interpretability/iv10_epistasis_pairedwrite.py`.
- **CRYSTAL-1 reports:** `reports/CRYSTAL_AGENT_BLUEPRINT.md` (L0–L3 axioms, 10 FORBIDs, build order), `CRYSTAL1_B0_B1_RESULTS.md`,
  `CRYSTAL1_B2_MULTISEED.md`, `CRYSTAL1_B3_B4_B5_RESULTS.md`, `B4_REAL_INTRADAY_CLOSURE.md` (regime is PRICED /
  Glosten-Milgrom equilibrium → config-D).

## The HL5 control grammar (already synthesized — reuse, don't reinvent)
`reports/HL5_FINAL_K_KnobRegistry_v2.md` (lever schema), `..._GV_HL_Constitution.md` (blast-radius tiers),
`..._HLX_Operator_Grammar.md` (operator + proposal schema), `..._TB_TeacherBank_Protocol.md`,
`..._IV_Command_Certification.md` (C0–C6 writ ladder), `HL5_MASTER_SYNTHESIS.md` (unified cumulative-authority
ledger; **α_machine = top open risk**), digests in `reports/_hl5_digest/`.

## Settled facts (do NOT contradict)
R6c > CRYSTAL-1 on controllability *today* → near-term = HL-over-R6c. Canonical P22 = persister (churner was a
csi500-retrain artifact). "Regime is priced" (Glosten-Milgrom, config-D is an equilibrium; VoI=0 on daily/most
intraday). B2 retraction of uniqueness-tracks-fidelity. Reward-shaping-for-legibility (B5) falsified → needs a
STRUCTURAL mechanism. Corner is real only on the polygon; on competitive markets its rents are competed into quotes.
