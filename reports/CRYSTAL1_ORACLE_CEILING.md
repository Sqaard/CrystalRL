# Oracle-ceiling decomposition: is the real-data null the model's fault or the signal's?

**Question (user):** «будет ли зарабатывать модель, если ей дать oracle сигналы? Если нет — проблема в модели.»
Script: [crystal1_oracle_ceiling.py](../interpretability/crystal1_oracle_ceiling.py); JSON:
`interpretability/crystal1_oracle_ceiling_report.json`.

## Design

Deliberately future-peeking oracle beliefs fed through the **unchanged** 4-knob threshold policy and the
**real** v4 gate (fresh instance, cheapest bar): O-A = realistic regime oracles (forward-20d vol quartile;
forward-20d cum-return < −3%); O-B = next-day-sign oracle (sanity ceiling); O-C = the policy class's
headroom with the *current* learned K=2 belief (dev-grid best, raw hold evaluation).

## Result: ДА — модель зарабатывает с oracle. Проблема в сигнале/предобработке.

| Test | csi500 | Dow |
|---|---|---|
| O-A oracle-vol → 4 knobs → gate | REJECTED_HOLDOUT (z=1.52) — but raw hold **ann +0.31 vs +0.03, DD −0.10 vs −0.25** | **ACCEPTED (z=2.33)** — hold ann +0.30 vs +0.23, **DD −0.06 vs −0.33** |
| O-A oracle-dd → 4 knobs → gate | REJECTED_HOLDOUT (z=1.30) | REJECTED_HOLDOUT (z=0.96) |
| O-B next-day sign → gate | **ACCEPTED** (ann +166%/yr, DD −0.2%) | **ACCEPTED** (ann +223%/yr) |
| O-C headroom of the class with the CURRENT K=2 signal | dev-best **hurts** on hold (ann −0.12 vs +0.03) | dev-best hold ann −0.00 vs +0.23 |

Decomposition of the null:

1. **Policy class (the "model" shell): SUFFICIENT.** With a good regime signal the same 4 knobs certify
   through the honest gate on Dow and monetize massively raw on both panels. With the sign oracle they
   certify everywhere. The shell is not the bottleneck.
2. **Signal (K=2 binary-|EW-ret| belief): THE bottleneck.** With the current signal the class's headroom
   is *negative* out-of-window (dev-selection actively hurts). The preprocessing throws away the sign of
   returns, their magnitude, all cross-sectional structure, and all 38 panel features — the oracle test
   proves that information, if recovered, is monetizable through the existing shell.
3. **csi500 has a second, separate ceiling: per-window detectability.** Even the oracle tops out at
   z≈1.5–1.9 on 120-day csi500 windows on BOTH lanes (RETURN z=1.52; forced RISK z_dsd=0.35–1.68 — the
   2021–22 bear is a low-vol *grind*, so vol-regime avoidance is thin per window). On csi500 the honest
   single-window certification bar exceeds even oracle-grade effects; on Dow it does not (oracle passes).

## Implications for the upgrade (task 3)

- Upgrade the **belief observation**, not the shell: signed-return × vol emissions, K=3, panel breadth —
  the exact package is the research workflow's synthesis (CRYSTAL1_UPGRADE_RESEARCH).
- Target an **honest certified accept on Dow** (proven reachable); on csi500 report the signal honestly as
  a *fraction of the oracle ceiling* (per-window z capture), since the absolute bar is above oracle there.
- Gate note: for both-axes-forward candidates the RETURN lane shadows the RISK lane; an either-lane rule
  (each lane with its disjoint confirm) is a fair power improvement — the oracle probe shows it wouldn't
  change the csi500 verdict (both lanes fail there) but removes an arbitrary lane assignment.
