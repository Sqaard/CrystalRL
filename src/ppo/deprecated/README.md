# Deprecated PPO Experiments

This directory keeps implementations that are useful for audit history but are
not part of the active experiment surface.

- `stage0_1_w2_qcritic_train.py` / `offpolicy_q.py`: off-policy W2
  Q-actor-critic prototype. It ran, but the candidate collapsed into highly
  defensive, low-return behavior and is methodologically separate from the
  active PPO/CTDE line. Keep it for reference; do not package it as an active
  Huawei candidate without a new design review.
