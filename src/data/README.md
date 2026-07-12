# src/data — panels and features

Feature construction and static reference data for the Stage 0.1 / CHRL panels.

| Module | Role |
|---|---|
| `stage0_1_feature_builder.py` | build the weight-based PPO features from raw panels |
| `stage0_1_normalization.py` | fold-local (train-only, then frozen) feature normalization — the leakage guard |
| `dow30_sectors.py` | the static sector map for the 29-ticker Dow-style universe |

Normalization is train-only-then-frozen on purpose: scaling statistics fit on the train window are
reused unchanged on validation/test, so no future information leaks into a feature.
