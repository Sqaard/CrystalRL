# R6c Frozen Test Rollout For Joseph

This is the out-of-sample 2022-2023 frozen rollout requested for Stage 4.

Model:
`R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1`

Window:
`2022-01-03` to `2023-02-28`

Files:

- `r6c_frozen_2022_2023_rollout_package.npz`: same-style NPZ with `test_*` arrays.
- `test_codes.parquet`: `date`, `code_id`, `valid`.
- `frozen_test_behavior_log_daily.parquet`: test codes joined to R6c daily behavior logs.
- `frozen_test_code_summary.csv`: quick per-code frozen-test summary.

Codebook:

- KMeans K=8, window_length=17.
- Codebook refit with the same seed/window on the Stage 1 train+validation hidden states.
- `valid == False` marks centered-window boundary rows.

Important:

This is frozen out-of-sample rollout only. It should be used to test whether
the primitive structure found on train+validation holds on 2022-2023.
