# CRYSTAL-1 v6 — belief from the PREPROCESSED panel features (item 0 + "сделай (а)")

**User's correction (item 0):** v5 (and everything before) read only `close` from
`data/adapters/_csi500_wide/csi300_model_ready.csv` → EW return. The 39 engineered columns
(Regime_1_Prob, atr_rel, rsi_30, macd, dx_30, turbulence, GRU forecasts, fundamentals — built per
`Preprocessing/Data_preprocessing.ipynb`) went unused. Item 1(a): rebuild the belief from those features.

## What was built ([hl_v6_crystal1_features.py](../interpretability/hl_v6_crystal1_features.py))

A born-legible **causal Gaussian-emission HMM** (the notebook's own `infer_causal_hmm_states` discipline:
past-only filtering, train-frozen standardizer, K∈{2,3} by held-out LL) over a pre-registered NAMED
market-feature vector `{reg1, atr_rel, xdisp, rsi30, dx30}` — chosen by TRAIN-only forward correlations
(fwd-vol +0.26…+0.42). Dropped: `macd` (dead, +0.06), `turbulence`/`10Y_Yield` (100% zero), `Market_Regime`
(≥82% zero). **GRU quarantined** to a flagged ablation (forecast_std fwd_ret +0.49 is the in-sample-fit
signature; IC t=+10 in-window). Three arms: **core** (the HMM), **direct_Regime_1_Prob** (the panel's own
regime posterior used verbatim as the belief), **core+GRU** (the ablation). Everything downstream — the
belief-gated policy, the v4 typed gate, the U3 rf=0/lag stress, the falsifiers — is v5 verbatim, so
v5(raw close) vs v6(preprocessed features) is a clean A/B on the signal only. One legitimate policy fix:
the vol-targeting branch was inert on a *low-vol* grind (σ̂<σ* ⇒ clip to 1), so `E_GRIND` now **caps
exposure whenever the belief is armed** (`min(vol_target, E_GRIND)`), live on K=2 and K=3.

## Result: preprocessing does NOT unlock certification — 0 accepts on every arm, same as v5

| Arm | Belief | Falsifiers | csi500 accepts |
|---|---|---|---|
| core | causal Gaussian HMM, K=2 (CALM/CRISIS, cleanly separated: CRISIS = high reg1/atr_rel/xdisp) | LL✓ states✓ placebo✓ armed✗ | **0** |
| direct_Regime_1_Prob | the panel's own regime posterior | placebo✓ armed✗ | **0** |
| core+GRU | + quarantined GRU | LL✓ states✓ placebo✓ armed✗ | **0** |

## Two concrete discoveries about the preprocessing (the real content)

**(A) The panel's entire US-macro block is STALE DATA after 2021-06-11 — a preprocessing defect, not a quiet
signal.** *(corrected by the verifier — my first read was wrong.)* The notebook fits the HMM regime on US
macro (`^VIX, ^TNX, ^GSPC`) and appends it to Chinese names. On this file, **`Regime_1_Prob`, `Regime_0_Prob`,
`VIX`, `SP500_Trend`, and `Market_Regime` all FREEZE to a single constant on the same date, 2021-06-11, and
stay frozen through all of hold AND OOS** (reg1 on the 2021-09..2022-06 hold: mean 0, std 0, nunique 1; it
varies normally on train/dev, nunique 532/106). This is a data-ingestion / forward-fill cutoff — the US-macro
features are **non-computable** on the decision windows, not merely uninformative. So the `direct_Regime_1_Prob`
arm ran on frozen dead data on hold/OOS, and the HMM's reg1 channel contributed nothing there. Correct fix:
the macro block must be **dropped** from any hold/OOS-facing belief on this panel — it has no valid post-cutoff
values. (This staleness is also *why* the gate rejects csi500 at the DEV `NO_FRONTIER_GAIN` stage — see (C).)

**(B) The CN-native technical features describe the WRONG axis for a grind, and time it WORSE than raw
signed returns.** A grind-bear is a *low-vol negative-return drift*; the live technicals (atr_rel, xdisp)
describe volatility/dispersion, which is *low* there. For an identical hindsight-best defensive cut
(`b_risk>0.4 → exposure 0.3`) on the full csi500 hold:
- **v6 (preprocessed)**: DD −0.211 vs anchor −0.251, **z_dsd 2.43** — the risk leg PASSES the bar (2.15),
  a real lift over v5-raw's z_dsd 1.51 from the earlier report; but return-NI **ni −0.16** (fails), RETz −0.83.
- **v5 (raw signed return)**: DD **−0.082** vs −0.251, ann **+0.035** (better than anchor on *both*!),
  **z_dsd 6.80**, but return-NI **ni +0.28** (still fails 2.15 — the return delta is net-positive but
  underpowered).

The raw signed-return belief times the csi500 grind *better* than the rich vol/regime features, because the
grind is a sign-of-return event, not a vol event.

**(C) Two failure stages, stated honestly.** *The gate's actual failure leg* on csi500 is DEV
`NO_FRONTIER_GAIN` (30/30): at the literature-default `Q_ON=0.85` the v6 belief **never arms on the dev
window** (2021-01..08), so every candidate ≡ anchor on dev and the return-NI/hold leg never fires. *As a
manual full-hold diagnostic* (forcing an arming config), return non-inferiority is the binding constraint —
an arming v6 config (`Q_ON 0.70/E_GRIND 0.30`) cuts DD (z_dsd 2.29) but costs return (hold ann −0.10,
ni −0.87). Both statements are true at different levels; neither yields a certification. (The v5-raw
"better than anchor on both" number is an **un-certified single-path artifact** — v5's own gate rejected it
too. And the F_U2a placebo PASSes *vacuously* on csi500 because the belief is inert on dev, so real ≡ anchor.)

## Bottom line

**"Проблема в предобработке" is answered NO for this panel.** Using the full preprocessed feature set,
built with the notebook's own causal discipline and kept born-legible, does not change the certification
verdict and, on the risk axis, the engineered vol/regime features are *inferior* to raw signed returns for
the specific csi500 grind — plus the panel's marquee macro-regime feature is inert on the CN event. The
daily-frequency door stays closed with the causal story now complete on the data side too: alpha priced
(B4) + per-window ceilings (oracle) + real-but-return-costly risk timing (v5) + **richer features don't
help because they describe vol not the grind's sign, and the imported macro regime is CN-blind (v6)**.
The certifiable route remains a substrate where the VoI gate can open (intraday CN, recorder accumulating).

## Adversarial verification → CONFIRMED (machinery sound, 0-certified honest), with 2 headline corrections

An independent skeptic reproduced the run byte-for-byte and could not break it: **no leakage** (a shock
injected at a hold date changes zero belief values before it; standardizer/HMM/K-selection all train-only;
t−1 policy convention), **teeth alive** (a +3bp/day sentinel certifies through the v6 gate, RETURN z=3.9),
falsifiers honest. It forced the gate at the cheapest bar on the full hold for the best arming configs —
all still reject. Two corrections, both folded into (A) and (C) above:
1. **reg1 is stale, not "CN-blind"** — the whole US-macro block freezes to constants on 2021-06-11 (std 0
   across hold/OOS). A concrete data-quality defect in the preprocessing, stronger than my first reading.
2. **The gate's failure leg is DEV `NO_FRONTIER_GAIN`, not return-NI** — the belief doesn't arm on dev at
   the default `Q_ON`; return-NI binds only as a manual full-hold diagnostic on a forced-arming config.

Net: the answer to "проблема в предобработке?" is now sharper — **part of the preprocessing is literally
broken on the test window** (the US-macro/regime block is forward-filled dead after 2021-06-11), and the
part that is live (CN technicals) describes volatility, the wrong axis for a low-vol grind. Neither the raw
nor the preprocessed signal certifies; the daily csi500 door stays closed for reasons now pinned down on
both the model side and the data side.
