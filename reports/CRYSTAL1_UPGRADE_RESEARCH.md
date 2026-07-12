# CRYSTAL-1 Upgrade Research — weakness map + literature -> the U1/U2/U3 design

Produced by the crystal1-upgrade-research workflow (5 investigators + synthesis), 2026-07-06.
Companion: CRYSTAL1_ORACLE_CEILING.md (the oracle decomposition that scoped this research).

---
## SYNTHESIS (the design)

# CRYSTAL-1 Upgrade Design — csi500 real-data, v4-gate-certifiable

**Synthesis verdict:** All five investigations converge on the same diagnosis: the binding constraint is the observation channel (sign-blind, train-anchored, ~0 bits on a low-vol grind), and the only honestly certifiable lane on a ~185-day hold is RISK (vol/DD cut) with RETURN as non-inferiority. The package below targets exactly that: rebuild the observation so the grind-bear is representable, make the policy act continuously only where the literature says the effect is mechanical, and fix the two accounting biases that currently rig the non-inferiority leg against the certification we want.

**Implementation order:** U3 first (5 lines, rebaselines everything), then U1, then U2. Ranked below by expected certification value.

---

## (1) Top-3 upgrade package

### U1 — Signal rebuild: signed, vol-normalized, dispersion-augmented alphabet + sticky K=3 filter
*(files: `interpretability/hl_v4_over_crystal1.py::build_belief`, `src/crystal/belief_filter.py`)*

**Observation (replaces `|EW_ret| > train_q80`):**
```
r_t     = EW panel daily return (as now)
sd_t    = std(r_{t-60..t-1})                     # trailing, exclusive of today; PIT-safe
z_t     = r_t / sd_t
s_ret   = DOWN if z_t < -Z_HI;  UP if z_t > +Z_HI;  else MID
          where Z_HI = train-q80 of |z| on 2018-2020 (frozen)
disp_t  = cross-sectional std of daily_return across names at t   # panel-audit winner "xsec_disp"
s_disp  = HI if disp_t > train-q80(disp) else LO                  # frozen train quantile
obs_t   = s_ret x s_disp   →  A_obs = 6 named symbols
```
Named knobs: `VOL_WIN=60` (fixed), `Z_HI` (fixed at train-q80), `DISP_Q=0.80` (fixed).

**Filter:** `NeuralBayesFilter(K=3, A_obs=6)`, same class, two training changes:
- **Sticky MAP prior:** add `KAPPA` pseudo-counts to diag(A) in the M-step; report as "expected dwell = 1/(1−A_kk) days". Fix KAPPA to a prior dwell of ~20 days (not searched).
- **Train on the full 2018-2020 sequence as one block (B=1)** — kills the 60-day belief-reset train/deploy mismatch and the downward-biased stickiness (audit #9).

**State naming — pre-registered rule, applied once at train time, frozen:**
CRISIS = argmax_k P(disp=HI | k); GRIND = remaining state with max P(ret=DOWN | k); CALM = remainder. Printable K×6 emission card; belief is a 3-simplex; fully simulatable.

Note breadth is deliberately absent: the classic literature pick (lit lens 3, audit #4) is empirically DEAD on this panel (fwd correlations ~0, autocorr −0.06). `xsec_disp` is the panel-audit-verified substitute — raw-data, full-history, autocorr 0.69, legible ("how scattered were today's returns").

### U2 — Policy: belief-gated conditional vol targeting with floor, hysteresis, and self-calibrating gate
*(replaces the 3-level `{t1,t2,lvl_reduced,lvl_defensive}` shell)*

```
b_risk,t = b_t[GRIND] + b_t[CRISIS]
g_on,t   = Quantile_{Q_ON}(b_risk over trailing 252d)   # PIT-safe rolling rule; frozen knob, adaptive value
gate ON  when b_risk > g_on;  gate OFF when b_risk < g_on − HYST      # hysteresis

if gate OFF (CALM):                    ex_t = 1                        # never de-risk in calm — Bongaerts 2020
if gate ON and b[CRISIS] >= b[GRIND]:  ex_t = clip(SIGMA_STAR / sigma_hat_t, E_MIN, 1)
if gate ON and b[GRIND]  >  b[CRISIS]: ex_t = E_GRIND                  # fixed reduced weight; grind is LOW-vol,
                                                                       # so vol targeting alone cannot see it
sigma_hat_t = trailing 20d realized vol of EW returns
SIGMA_STAR  = train-median of sigma_hat (frozen constant, not searched)
```
Named knobs: `Q_ON` (searched, ~0.85), `E_GRIND` (searched, ~0.7), `HYST=0.10` (fixed), `E_MIN=0.5` (fixed — the Wang-Li A-share "high vol → high return" result is the direct argument for a high floor; an unfloored version is a re-run of B3's known failure), `VOL_WIN2=20`, `SIGMA_STAR` (fixed). Only **two searched knobs** — preserves gate power. Still one paragraph a human can simulate from a price chart plus the belief card.

### U3 — Execution honesty: cash yield + A-share cost stress slices
*(file: `hl_v4_over_crystal1.py::strat`; stress slices go into the existing adversary plumbing, not a new model)*

```
ret_t = ex_t * ro_t + (1 - ex_t) * RF_DAILY - COST*max(dEx,0) - (COST+STAMP)*max(-dEx,0)
RF_DAILY = 0.02/252 for csi (CNY deposit/repo), T-bill equivalent for Dow    # named per-panel constant
STAMP    = 0.001 sell-side (stamp duty proxy)
Stress slices (gate veto, existing plumbing): (i) one-day execution lag (exposure bl[:-2] applied to ro[2:]);
(ii) rf=0 slice — the RISK-lane vol/DD legs must survive rf=0; rf may only support the non-inferiority leg.
```

---

## (2) What each fixes from the internal audit

| Upgrade | Audit items closed | Mechanism |
|---|---|---|
| U1 | **#1** (sign), **#2** (train-anchored threshold → `REFUSED_INERT_ON_WINDOW`), **#3** (K=2 ceiling: grind-bear unrepresentable), **#4** (cross-sectional collapse — via the verified channel, not the dead one), **#9** (block-reset training) | Vol-normalization restores a stationary firing rate on the 2021-22 low-vol window; sign separates crash/rebound/chop (B3's exact failure mechanism); disp channel gives the crisis state a second witness; K=3 gives the grind a state to live in |
| U2 | **#5** (dead absolute thresholds, `cost_churn` canary, no hysteresis), plus B3's return non-inferiority failure (floor keeps rebound participation; action concentrated in confirmed high-vol states) | Rolling-quantile gate guarantees knob liveness; continuous sizing removes the knife-edge threshold bootstrap variance; hysteresis kills daily flip bleed |
| U3 | **#6** (zero cash yield — biases the gate against the exact certification sought), **#8** (symmetric/frictionless China microstructure; limit-down fiction) | Defensive sleeve earns the deposit rate; sell-side asymmetry and lag priced as stress vetoes |

Audit #7 (own-drawdown guard) is deliberately **not** added: it chases the same effect as the GRIND state through a second knob — keep it as the fallback if U1's grind state fails identifiability (below), never both simultaneously.

## (3) Expected failure modes and the falsifier each must pass

**U1:**
- *K=3 unidentifiable on ~730 train obs / label instability.* Falsifiers: (i) held-out LL must beat both memoryless AND the K=2-signed ablation (existing L1 selftest); (ii) pre-registered occupancy/dwell check on train: every state >5% occupancy, mean dwell >5d, emission cards pairwise-distinct; (iii) permutation-aligned label consistency across the rotating holdout folds. Any failure → fall back to K=3→K=2 with the signed alphabet kept.
- *Observation still inert on hold.* Falsifier: non-MID symbol firing rate on every hold window within [0.5x, 2x] of train rate — directly falsifies the measured inert-audit failure.
- *disp channel is noise.* Falsifier: ablation A_obs=6→3 (drop disp); if held-out LL and gate outcome are unchanged, drop the channel (parsimony budget). Fallback channel if disp fails: VIX column (stronger numbers but 2018-10-30 warm-up hole + upstream-model provenance — second choice for a frozen 2018-2020 train).

**U2:**
- *Belief gate adds nothing over always-on vol targeting.* Falsifier: paired comparison vs the unconditional `clip(σ*/σ̂,E_MIN,1)` baseline — the gated version must match risk legs with materially lower turnover (the Bongaerts claim), AND a block-shuffled-belief placebo must fail the gate (belief load-bearing).
- *Rolling gate de-arms in a persistent crisis (quantile creeps up) or never arms.* Falsifier: armed-canary — gated-days fraction on every 120d window in [5%, 60%]; outside → refuse as inert, don't tune around it.
- *Grind row does nothing (E_GRIND too timid) or fails return non-inferiority (too aggressive).* Falsifier: the standard v4 typed lanes — RISK accept requires bootstrap CI on vol/maxDD delta excluding 0 with the RETURN non-inferiority leg passing on disjoint confirm; a delta==0 result on any window re-triggers the liveness refusal, not a knob widening.
- *Certification is phase-selection.* Falsifier: rotating disjoint-confirmed holdout as-is; no window may be hand-picked (two false-GREEN precedents: N7, company-text).

**U3:**
- *rf flatters defensive configs.* Falsifier: rf=0 stress slice must preserve the RISK-lane legs; rf enters only the non-inferiority margin, at a conservative 2%.
- *Edge is microstructure fiction (trading limit-down closes).* Falsifier: 1-day-lag slice must preserve the certification; if the effect dies at t+1 execution, it was never tradeable in A-shares — honest refuse.

**Package-level:** the perfect-foresight oracle must still pass the gate after all three changes (gate winnability preserved), and total searched knobs stay at 2 (`Q_ON`, `E_GRIND`).

## (4) Explicitly NOT do

1. **GRU-derived features** (`gru_spread`, `gru_uncert`, forecast level) — quarantined; in-window IC t=+10 is the signature of in-sample fit; untouchable until train-window provenance is verified.
2. **Daily-sign breadth** (fraction of decliners) — empirically dead on this panel (no signal, no persistence) despite being the literature's favorite; do not re-add it because a paper says so.
3. **Return-lane vol-managed sizing** (Moreira-Muir c/σ²) — inverted/spanned in A-shares (Wang-Li, Chi et al.) and undetectable at 120-day blocks; never claim the RETURN lane.
4. **TSMOM / trend overlay** — would have "worked" on this exact hold, which is precisely the phase-selection trap; per-asset effect ~0.1-0.3 SR is statistically invisible here.
5. **K=4, HDP-HMM, HSMM, online/adaptive EM, scheduled refits** — identification risk, born-unnamed states, or broken simulatability/frozen-knob contract. Stickiness comes from KAPPA only.
6. **CPPI / equity-floor drawdown control** — path-dependent (block-bootstrap-hostile), de-risks after losses in a reversal-heavy market; it is B3's failure with more knobs.
7. **Feeding the 38 panel features into the filter** — the capacity route breaks the one-named-observation legibility story; the parsimony budget is one return symbol + one dispersion bit.
8. **Simultaneous U1-grind-state + audit-#7 dd-guard** — two knobs, one effect, pure overfit surface; the dd-guard is the pre-registered fallback only.

**Predicted honest outcome:** RISK-lane accept on csi500 (20-35% hold-window vol/maxDD cut is the one effect large and mechanical enough for 120-day block-bootstrap detection), RETURN non-inferiority passing conditional on `E_MIN=0.5` and the U3 cash carry; the flagged escalation if U1+U2 fail the gate is replacing the HMM core with a statistical jump model (Shu-Yu-Mulvey 2024: K centroids + one jump penalty λ, DP-solvable, arguably more legible) — a substitution, not a bolt-on.

---
## Investigation 1: internal weakness audit

RANKED INVENTORY — information discarded / capacity missing in the real-data CRYSTAL-1 stack
(verified against `interpretability/hl_v4_over_crystal1.py` lines 64-101, `src/crystal/belief_filter.py`, `interpretability/hl_real_dow.py` lines 32-47, 69-88, and both panel headers)

---

**#1 — SIGN of returns discarded at observation construction** (stage: observation)
- Code: `obs = (|EW_ret| > train_q80)` — `build_belief()` in both hl files (hl_v4 line 76-80). A -2% day and a +2% day are the same symbol; the "toxic" state is a HIGH-VOLATILITY state, not a bear state.
- Lost: direction. The filter is structurally incapable of distinguishing crash, rebound, and two-sided chop.
- Why it matters on csi500 2021-09..2022-06: it fails BOTH ways. (a) Grind phase: persistent small negative drift under modest vol → almost no exceedances → belief stays benign → full exposure through the slide. (b) Spike/rebound phase (Mar-Apr 2022 crash, late-Apr V-rally): the biggest UP days live in the same high-|r| cluster as the down days, so a defensive config exits the drawdown late AND sits out the rebound — this is precisely the mechanism behind "B3's tail cut fails return non-inferiority."
- Minimal legible fix: 3-symbol alphabet, `A_obs=3`: {down-exceed: r < −q80(neg-train), neutral, up-exceed: r > +q80(pos-train)}. `NeuralBayesFilter` already takes `A_obs` as a parameter; emissions stay a readable K×3 table with named columns. One-line change to obs construction, zero change to the filter class.

**#2 — Threshold frozen at a train-window quantile + magnitude discarded** (stage: observation)
- Code: `thr = quantile(|train_returns|, 0.80)` frozen forever; a 1.01×thr day and a 6×thr day are the same symbol.
- Lost: the exceedance RATE is only meaningful if vol is stationary between train and deployment. csi500 train = 2018-2020 (2018 bear + COVID) → high threshold → 2021-22 grind days almost never cross it → the observation stream is near-constant zeros → the belief is pinned near its prior and carries ~0 bits. This is the direct cause of the measured `REFUSED_INERT_ON_WINDOW` audits (candidate never activates on hold).
- Minimal legible fix: threshold on VOL-NORMALIZED returns — `|r_t| / rolling_std_60(r)` > train-q80 of the same standardized quantity. PIT-safe (trailing window only), one named preprocessing step ("exceedance relative to recent vol"), restores a stationary firing rate. Alternatively a slow rolling quantile; either is one sentence of story.

**#3 — K=2 filter capacity: the grind-bear is a THIRD regime the model cannot represent** (stage: filter/belief)
- Code: `train_filter(..., K=2, A_obs=2)`. With a binary vol alphabet the entire learned object is ~5 free parameters; the belief is functionally an EWMA of exceedance counts.
- Lost: any state space distinguishing {calm-bull, low-vol grind-bear, high-vol crisis}. A low-vol drift-down regime maps onto "calm" by construction, so no knob setting can ever defend against it — the gate's 0-certified result on csi500 is partly a ceiling of the substrate, not of the search.
- Why it matters: csi500 2021-22 IS the third regime. K=2 + sign-blind obs makes the failure unfixable downstream.
- Minimal legible fix: K=3 with the signed alphabet from #1 (K=3 is unidentifiable on a sign-blind binary stream, so #1 is a prerequisite). Three named states with readable emission signatures is still fully simulatable; keep the same self-supervised training and the same L1 selftest discipline (permutation-align, held-out LL vs memoryless).

**#4 — Cross-sectional structure collapsed to a single EW mean** (stage: observation)
- Code: `ew_returns()` reads only `date,tic,close` and does `groupby(date).mean()`. Both panels carry 38-62 unused columns — the Dow panel literally contains `drawdown_20d`, `drawdown_60d`, `realized_vol_20d`, `universe_return_20d`, `turbulence`; the csi panel carries `Market_Regime`, `Regime_0/1_Prob`, GRU forecasts.
- Lost: breadth (fraction of decliners), dispersion, and every engineered feature. Breadth is the classic grind-bear detector: in a grind, daily index moves are small but 60-70% of names fall day after day — high-information, low-|r|.
- Minimal legible fix (without touching the 38 features): ONE extra binary channel, `breadth_t = frac(tic returns > 0) < train-q20`, factorized with the return symbol into a named 4-6 symbol alphabet. Still an HMM with a printable emission table. (Feeding the panel features into the filter would be the capacity route but breaks the "one named observation" story; breadth is the minimal step.)

**#5 — Policy sees only the instantaneous belief; threshold grid vs belief dynamic-range mismatch** (stage: policy class)
- Code: `exposure()` is a 3-level step function of b_t; knobs t1∈[0.10,0.60], t2∈[0.40,0.90] are ABSOLUTE belief levels. With overlapping learned emissions the belief's realized range on hold may never reach t1 — measured: 2/4 Dow tests had delta==0, and the harm-canary bank was UNARMED all run on both panels.
- Lost: duration-in-state, belief velocity, and any guarantee the knobs are live on a given window. Also no hysteresis — a belief hovering at a threshold flips exposure daily and bleeds the 10bp cost (the `cost_churn` canary is exactly this failure).
- Why it matters on the grind: whatever weak belief signal exists sits in a compressed range; absolute thresholds can be structurally unreachable, so the certified-0 conflates "no signal" with "knobs never engaged."
- Minimal legible fix: (a) calibrate t1/t2 as DEV-window belief QUANTILES ("t1 = 85th percentile of last-2-years belief") — still one sentence, guarantees knob liveness; (b) dual enter/exit thresholds (exit = t1−δ, one extra knob) for hysteresis.

**#6 — Zero cash yield in execution** (stage: execution)
- Code: `strat()` returns `ex * ro − costs`; the (1−ex) sleeve earns exactly 0.
- Lost: the risk-free carry of being defensive. CNY deposits/repo ~2%/yr in 2021-22; US T-bills 4-5% by 2023 (Dow OOS window). Every defensive config's return is understated by (1−ex)×rf, and the RISK lane's binding constraint is return NON-INFERIORITY — this omission biases the gate AGAINST the exact certification the project is trying to win.
- Minimal legible fix: one constant per panel, `ret = ex*ro + (1−ex)*rf_daily − costs`. Fully legible ("cash earns the deposit rate"). Cheapest item on this list relative to impact.

**#7 — No conditioning on the book's own drawdown** (stage: policy class)
- Code: exposure is a function of b_t only; the Dow panel even ships `drawdown_20d/60d` unused.
- Lost: a grind-bear is invisible to a vol filter but perfectly visible in the strategy's own trailing equity — cumulative drift is exactly the statistic the daily binary obs destroys.
- Why it matters: it is the only stage that can catch the grind WITHOUT rebuilding the filter (#1-#3). Caveat from project memory: pure momentum is dead on the broad universe, so frame it as a risk-lane guard (cap exposure at lvl_reduced while trailing-60d EW return < 0), not an alpha rule.
- Minimal legible fix: one named predicate, one knob: `if trailing_dd_60 < −X%: ex = min(ex, lvl_reduced)`. A human can simulate it from a price chart; it composes with the belief rule as an AND/MIN, preserving the 3-level story.

**#8 — Cost/market-microstructure model wrong for China; EW book internally frictionless** (stage: execution)
- Code: `COST=0.001` symmetric, charged only on |Δexposure|; the daily re-equal-weighting inside `groupby(date).mean()` is free; dividends unmodeled (raw `close`); no T+1 settlement (A-shares can't sell same-day buys), no sell-side stamp duty (0.05-0.1%), no ±10% price-limit days (you cannot exit at the printed close on limit-down days — which cluster exactly in the Apr-2022 stretch of the hold window).
- Lost: realistic frictions are ASYMMETRIC and worst on the days the defensive policy trades. A certified DD cut that trades through limit-down closes is partially fictional.
- Minimal legible fix: sell-side cost adder (COST_sell = COST + stamp) + a one-day execution-lag variant (`bl[:-2]` vs `ro[2:]`) run as a STRESS SLICE in the existing adversary, not as a new model — the gate already has the stress-veto plumbing.

**#9 — Filter training chops the stream into L=60 blocks with belief reset** (stage: filter)
- Code: `obs_tr[:n*L].reshape(n, 60)` — belief re-initialized to p0 every 60 days during training (deployment runs one continuous recursion, a train/deploy mismatch), tail observations silently dropped, and cross-block dependence (regimes lasting >60d — i.e., grinds) is invisible to the likelihood.
- Lost: long-persistence structure; stickiness estimates biased down, exactly the parameter that governs how long the belief stays defensive.
- Minimal legible fix: train on the single full sequence (B=1) or overlapping windows with carried-in initial belief. Zero model change, zero legibility cost.

**#10 — Panel composition honesty** (stage: observation/data)
- Fixed 2010-23 Dow-29 and "csi500"-lineage csi300 membership → survivorship tilt inflates the buy-and-hold anchor, making de-risking harder to certify (bias direction: against accepts — same direction as #6, so at least conservative). Dividends likely absent from `close` on the csi panel (raw closes) → EW anchor understated there (opposite direction). Not fixable minimally; log it as a caveat in `honesty_caps` alongside the existing csi300-naming note.

---

Cross-cutting note for the caller: #1+#2 are the binding constraint — everything downstream (K, policy knobs, the gate) is starved by an observation channel that delivers <1 bit/day of vol-only, train-anchored information; on a low-vol grind-bear that channel is ~0 bits. Fixes #1/#2/#4 change only the observation alphabet (the filter class, gate, and policy shell are untouched), #5/#6/#7 are one-knob/one-constant edits — all preserve the "named states + printable emission table + threshold story" legibility contract. #6 is the highest fix-value-per-line and directly attacks the known return-non-inferiority failure mode.

---
## Investigation 2: panel audit (csi300_model_ready.csv)

AUDIT: C:/Users/ivanp/RL for Time-Series Forecasting/data_RLagent_for_Joseph/data/adapters/_csi500_wide/csi300_model_ready.csv

=== FILE STRUCTURE ===
- 430,688 rows x 41 cols; 344 tickers x 1,252 dates; 2018-01-02 -> 2023-03-01; perfectly balanced (344 names every day).
- dtypes: date=datetime, tic=object, all 39 others float64.
- NaN rate: 0.00% for every column in every year. Missingness is encoded as ZEROS, not NaN — the zero-rate table is the real coverage audit.

=== COVERAGE (zero-rate) FINDINGS ===
- DEAD columns (100% zero all years): 10Y_Yield, turbulence. Market_Regime is 64-90% zero 2018-2021 and 100% zero 2022-2023 — effectively dead.
- WARM-UP: VIX, Regime_0_Prob, Regime_1_Prob, SP500_Trend are all-zero until 2018-10-30 (81.9% of 2018), nonzero and clean thereafter. Technicals (rsi_30, cci_30, dx_30, atr, atr_rel) zero-filled for the first ~2 months of 2018 (5-12% of 2018).
- MARKET-LEVEL columns (cross-sectional std = 0, one value per day): VIX, Regime_0_Prob/Regime_1_Prob, SP500_Trend, Market_Regime, day_sin/day_cos. Despite the names, VIX/SP500_Trend are the adapter's China-market analogues.
- PER-NAME columns with real cross-sectional dispersion (median relative dispersion): daily_return 2.94, GRU forecasts 2.4-7.8, cci_30 2.96, macd 5.88, eps_growth 10.98 (outlier-heavy), fundamentals ~1.0-2.6. volume_ratio has near-zero relative dispersion (0.048) — it is per-name but nearly market-wide. obv_pct_change has absurd dispersion (1.8e11) — unusable raw.
- cur_ratio/quick_ratio zero for a constant 10.5% of names (36 financials — structurally missing, not time-varying).

=== SANITY SIGNAL, TRAIN WINDOW ONLY ===
Daily-aggregated candidate vs FORWARD 20d EW return and forward 20d realized vol (std of EW daily returns), signal dates 2018-01-02 -> 2020-12-03 so no forward window touches 2021+. n=710 daily obs (511 for VIX/regime after warm-up), but overlapping 20d windows mean only ~35 independent blocks — 2-sigma noise floor on these correlations is roughly |r|=0.3. Per-year Spearman given as 2018/2019/2020.

RANKED SHORTLIST (top 5):

1. VIX (market-level vol column, per-day constant) — RISK lane.
   fwd_vol: Spearman +0.51, Pearson +0.33; per-year +0.85/+0.66/+0.27 (sign-stable). fwd_ret: -0.16 (noise). Autocorr(1)=0.98. Caveat: starts 2018-10-30.

2. Regime_1_Prob (the panel's own regime probability, market-level) — RISK lane, weak RETURN lane.
   fwd_vol: Spearman +0.42, Pearson +0.29; per-year +0.79/+0.57/+0.20. fwd_ret: Pearson -0.34, Spearman -0.32 but sign-unstable per-year (+0.06/-0.30/-0.33) — risk signal only. Autocorr 0.99. Starts 2018-10-30. Largely redundant with VIX (both upstream-model/market-level).

3. xsec_disp = daily cross-sectional std of daily_return — RISK lane. THE best raw-data candidate.
   fwd_vol: Spearman +0.28, Pearson +0.29; per-year -0.00/+0.43/+0.36. fwd_ret: +0.08 (dead). Autocorr 0.69. Fully legible ("how scattered were today's returns"), computable from raw returns over the FULL history including the 2018 warm-up gap, no upstream model dependence.

4. gru_spread = daily cross-sectional std of forecast_mean — RETURN lane, QUARANTINED.
   fwd_ret: Spearman +0.50, Pearson +0.45; per-year +0.42/+0.59/+0.23 (all positive). fwd_vol: ~0. Sister stats: gru_uncert (EW mean of forecast_std) fwd_ret Pearson +0.60/Spearman +0.48; EW forecast level fwd_ret Pearson +0.66. LEAKAGE FLAG: the GRU 1d forecast has daily cross-sectional IC +0.028 with t=+10.0 inside 2018-2020 and +0.15 correlation with next-day EW return — consistent with the GRU having been FIT on this window. Provenance of the GRU training window is unknown; treat all in-window GRU correlations as potentially in-sample-inflated. Only usable if the GRU's train/test split is verified to end before 2018 or is walk-forward.

5. dx_mean = EW mean of dx_30 (trend-strength) — weak RISK lane.
   fwd_vol: Spearman +0.25, Pearson +0.31; per-year -0.07/+0.23/+0.52 (sign flips 2018). fwd_ret: +0.13 (noise). Below the noise floor; keep only as a tie-breaker.

REJECTED (negative knowledge):
- breadth_down (fraction of names with negative return) — the most classic legible observation is DEAD on this panel: fwd_ret Spearman +0.02, fwd_vol +0.05, and autocorr(1) = -0.06 (white noise daily; unlike all other candidates it has no persistence for an HMM to latch onto).
- agg_turnover (median volume_ratio): fwd_ret Pearson +0.20 but Spearman +0.01 (pure outlier artifact), per-year sign flips (-0.23/+0.14/-0.20).
- atr_rel_med: fwd_vol Spearman +0.33 headline but per-year -0.06/+0.64/+0.15 — one-year wonder, ranked below xsec_disp despite the higher pooled number.

BOTTOM LINE: for a richer-but-legible K=2 belief observation, the honest upgrade path is the RISK lane — replace/augment the binary |EW ret|>q80 indicator with xsec_disp (raw-data, full-history, legible) and/or VIX / Regime_1_Prob (stronger numbers but upstream-model, 2018-10-30 start). This matches the known B3 result (belief value = risk/drawdown timing). The only RETURN-lane candidates are the GRU aggregate stats, and they carry an unresolved in-sample-leakage flag that must be cleared before any gate run.

Scripts: C:/Users/ivanp/AppData/Local/Temp/claude/C--Users-ivanp-RL-for-Time-Series-Forecasting-data-RLagent-for-Joseph/dcbf7e23-a1be-45c6-9f0f-982f00a9c753/scratchpad/{audit_panel.py, signal_check.py, gru_probe.py}

---
## Investigation 3: literature — regime models

# Literature lens 1 — Regime models: upgrading the K=2 binary-observation HMM

## 0. Framing

The current filter (K=2, binary emission `|r_EW| > q80`) is a degenerate special case of every model below: it is a 1-bit vol-regime detector. The literature says the cheapest large information gains, in order, are (a) restoring sign + magnitude via continuous emissions, (b) K=3 to separate "volatile-bear" from "crisis", (c) duration/stickiness control, (d) one extra cross-sectional emission channel. Everything fancier (nonparametric state counts, online re-estimation) buys little on daily index risk-timing and costs legibility.

## 1. Hamilton (1989) Markov-switching with Gaussian emissions on SIGNED returns

- **What**: Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle," *Econometrica* 57(2), 1989 — K=2 latent Markov chain, regime-dependent mean (originally AR on quarterly GNP). Applied to equities: Turner, Startz & Nelson (*JFE* 1989) put the switch in mean AND variance of S&P returns; Hamilton & Susmel (*J. Econometrics* 1994, SWARCH) showed regime-switching variance dominates plain ARCH on weekly equity returns; Rydén, Teräsvirta & Åsbrink (*J. Applied Econometrics* 1998) showed a 2-state Gaussian HMM reproduces most stylized facts of **daily** S&P returns.
- **Evidence on daily equities**: strong and old. The key empirical fact: regimes in *variance* are far better identified than regimes in *mean* (mean differences are small relative to daily noise, variance differences are ~2-4x). Bulla, Mergner, Bulla & Sesboüé, "Markov-switching asset allocation: do profitable strategies exist?" (*J. Asset Management* 12, 2011) is the closest template to your shell: 2-state Gaussian HMM on **daily** S&P 500 / DAX, one-day implementation lag, out-of-sample regime-gated exposure → similar return, materially lower vol and max-drawdown, surviving realistic costs because switches are infrequent ([ResearchGate](https://www.researchgate.net/publication/50245257_Markov-switching_asset_allocation_Do_profitable_strategies_exist), [Hamilton's own survey](https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf)). That is exactly your B3 "risk/drawdown timing" lane.
- **Legibility cost**: near zero. Emission parameters are nameable market facts ("calm state: μ=+4bp/d, σ=0.7%/d; stressed state: μ=−8bp/d, σ=1.9%/d"). Belief remains a K-simplex; the forward recursion is the same simulatable NeuralBayesFilter update. Signed Gaussian emissions are MORE legible than the binary indicator, because the current q80 threshold hides an arbitrary quantile choice while a Gaussian mean/var pair is directly auditable.
- **Complexity**: trivial — replace Bernoulli emission likelihood with Gaussian (or Student-t, see below) pdf in `belief_filter.py`; EM on the frozen train window; ~1 day of work.

## 2. Joint vol/return emissions and heavy tails

- **What**: 2-D emission (r_t, vol-proxy_t), e.g. (signed EW return, log of a short realized-vol or |r| EWMA), or 1-D with Student-t emissions. Hardy (*NAAJ* 2001) is the canonical 2-state lognormal on index returns; Bulla (*Quantitative Finance* 2011) shows **t-emissions** on daily returns fix the main pathology of Gaussian HMMs — spurious state flips on single outlier days — and increase regime persistence without touching the transition matrix; a recent systematic comparison of heavy-tail emission families for regime-conditional risk is [arXiv 2606.23492](https://arxiv.org/pdf/2606.23492).
- **Evidence**: robust. The known failure of Gaussian K=2 on daily data is that the ACF of |r| decays too fast (Rydén et al. 1998); t-emissions and duration modeling are the two standard fixes. Joint (return, vol) emissions are what the successful daily-frequency allocation papers actually use (Nystrup et al. line, jump-model line below).
- **Legibility cost**: low. One extra named observation channel; ν (t-dof) is one named knob ("how much a single crash day is allowed to move belief"). Caveat: the vol-proxy must be built from train-frozen normalization to stay PIT.
- **Complexity**: low; diagonal-covariance 2-D emission keeps EM closed-form (Gaussian) or one-step ECM (t).

## 3. K=3-4 canonical regimes

- **What/evidence**: Guidolin & Timmermann (*JEDC* 31, 2007, "Asset allocation under multivariate regime switching", [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=940652)) find **four** regimes — crash / slow-growth / bull / recovery — are needed for the joint stock-bond distribution, with out-of-sample economic value; but that is *monthly, multivariate*. Maheu, McCurdy & Song (*JBES* 2012, "[Bull corrections and bear rallies](https://www.economics.utoronto.ca/public/workingPapers/tecipa-402.pdf)") identify 4 states (bull / bull-correction / bear / bear-rally) on long-history weekly data. Ang & Bekaert (*FAJ* 2004) show even K=2 regime-switching allocation beats static out-of-sample. For **daily single-index risk timing**, the working consensus in the applied literature (QuantStart-style replications, [MDPI 2020 regime-switching factor investing](https://www.mdpi.com/1911-8074/13/12/311), [arXiv 2107.05535](https://arxiv.org/pdf/2107.05535)) is K=3: calm-bull (low vol, +drift), volatile-bear (high vol, −drift), crisis (very high vol). K=4's fourth state (recovery: high vol, +drift) is real but weakly identified on ~3.5k daily obs per train window — expect label instability across retrains.
- **Legibility cost**: low at K=3 (states remain nameable; belief is a 3-simplex, thresholds t1/t2 generalize to per-state exposure weights — still a simulatable table). Rises at K=4+ via label-switching and near-degenerate states — a direct legibility hazard given your policy is a threshold shell on the belief.
- **Complexity**: low (same EM, more components); the real cost is validation that state 3 is stable across your rotating holdout folds.

## 4. Sticky-HDP-HMM and duration modeling

- **Sticky-HDP-HMM**: Fox, Sudderth, Jordan & Willsky (*Annals of Applied Statistics* 2011, "[A sticky HDP-HMM with application to speaker diarization](https://ics.uci.edu/~sudderth/papers/aoas11hdphmm.pdf)"; [tech report](https://people.eecs.berkeley.edu/~jordan/papers/stickyHDPHMM_LIDS_TR.pdf)) — nonparametric state count plus explicit self-transition mass κ. Financial application: Song (*J. Applied Econometrics* 2014, "[Modelling regime switching and structural breaks with an infinite HMM](https://onlinelibrary.wiley.com/doi/10.1002/jae.2337)"). **Evidence** on daily equity risk-timing specifically: thin — mostly fit-quality, not trading value. **Legibility cost: high and disqualifying** — the number of states is a posterior random variable, MCMC inference, label identity is not stable; you cannot name states that are born unnamed. **Verdict: take the κ idea, not the model** — stickiness as an explicit self-transition prior on a fixed-K HMM is one named knob ("minimum expected regime dwell = 1/(1−A_kk)").
- **Duration modeling (HSMM)**: Bulla & Bulla (*Computational Statistics & Data Analysis* 2006, "[Stylized facts of financial time series and hidden semi-Markov models](https://www.sciencedirect.com/science/article/abs/pii/S0167947306002374)") — negative-binomial sojourn times fix the too-fast |r|-ACF decay of geometric-duration HMMs on daily data. **Evidence**: solid on fit; modest documented trading value. **Legibility**: OK (a dwell-time distribution is a nameable object) but **complexity** is the highest here — duration-augmented forward-backward, and your belief state stops being a clean K-simplex (it becomes state x elapsed-duration). Mostly dominated by t-emissions + sticky diagonal, which achieve the same persistence more cheaply.
- **Statistical jump models** (the modern form of duration control): Nystrup, Lindström & Madsen (*Expert Systems with Applications* 2020, "Learning hidden Markov models with persistent states by penalizing jumps"); Shu, Yu & Mulvey 2024, "[Downside risk reduction using regime-switching signals: a statistical jump model approach](https://arxiv.org/html/2402.05272v2)" — daily S&P 500 / DAX / Nikkei, fully out-of-sample, realistic costs: jump-penalty regime signal cuts max drawdown and improves Sharpe vs both buy-and-hold and standard HMM. This is currently the **best-documented daily-equity risk-timing evidence in the whole survey**, and the model is arguably MORE legible than an HMM: K centroids in named feature space + one jump-penalty λ; state sequence solvable by dynamic programming (fully simulatable). Related: [dynamic factor allocation with JM signals, arXiv 2410.14841](https://arxiv.org/html/2410.14841v1).

## 5. Feature-based emissions (breadth, dispersion)

- **What/evidence**: Kritzman, Page & Turkington (*FAJ* 2012, "Regime shifts: implications for dynamic strategies") fit Markov-switching to a financial-turbulence index (a Mahalanobis cross-sectional feature) rather than raw returns, with out-of-sample value — the canonical precedent for "emit a constructed feature, not the return". Market breadth is a robust return/risk predictor across 64 countries ([Herding for profits, *Economic Modelling* 2019](https://www.sciencedirect.com/science/article/pii/S0264999319312982)); cross-sectional dispersion collapse marks stress/synchronization regimes ([S&P DJI dispersion research](https://www.spglobal.com/spdji/en/documents/research/research-dispersion-measuring-market-opportunity.pdf), [NBER w26329](https://www.nber.org/papers/w26329)). Feature-saliency HMMs for allocation: Fons et al., *Expert Systems with Applications* 2021 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417420305443)). Jump-model feature selection: Nystrup et al. 2021.
- **Relevance to you**: your panels already carry 38 unused columns, and breadth/dispersion are computable from the EW panel itself with zero new data. Cross-sectional breadth (fraction of the 29/500 names up today) and dispersion (cross-sectional std of daily returns) are exactly the structure the current EW-|r| observation destroys.
- **Legibility cost**: low **if capped at 1-2 named features** ("crisis = high dispersion + negative breadth" is a sentence a human can check); each added dimension multiplies the audit surface, so treat emission dimensionality as a parsimony budget.
- **Complexity**: low (extra emission dims); the PIT discipline point is that feature standardization constants must be train-frozen.

## 6. Online EM vs frozen-train

- **What/evidence**: Nystrup, Madsen & Lindström (*J. Forecasting* 2017, "[Long memory of financial time series and HMMs with time-varying parameters](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2447)") — forgetting-factor adaptive estimation of a 2-state HMM reproduces long memory and improves daily density forecasts; their *JPM* 2015 allocation paper uses it live. Online EM machinery: Cappé & Moulines (2009). So yes, adaptivity helps *fit*.
- **But for you**: adaptive parameters break the two things your project treats as non-negotiable — **simulatability** (a human can no longer replay the filter from a printed parameter card, because the card changes daily) and the **frozen-train gate** (the v4 gate certifies knob settings; online EM makes every day a new knob setting, and forgetting factors are a classic overfitting channel that your gate's rotating holdout cannot price). The legible middle ground, if train/test drift ever proves fatal, is *scheduled* re-fits at pre-registered dates with the re-fit rule itself frozen — not per-step adaptation. Default: **frozen-train, online belief filtering only** (which is what you already do).

## The 3 upgrades I would implement first (frozen-train PIT discipline)

1. **K=3 Student-t emissions on the 2-D observation (signed EW return, log realized-vol proxy)** — replaces the 1-bit Bernoulli channel. This single change restores sign, magnitude, and the calm-bull / volatile-bear / crisis distinction, with the strongest and oldest evidence base (Hamilton 1989 lineage; Bulla et al. 2011 daily OOS allocation; Bulla 2011 t-robustness). Legibility improves: three nameable states with printable (μ, σ, ν) cards; belief is a 3-simplex; the exposure shell generalizes to a 3-row threshold table. EM fit on the train window, all parameters frozen. This is the direct fix for B3's problem — a sign-blind filter cannot distinguish a melt-up from a meltdown, which is plausibly why the tail cut fails return non-inferiority.
2. **Sticky self-transition prior with a named dwell-time knob** (MAP mass κ on diag(A), reported as "expected dwell days per state"), tuned only on train — the cheap fixed-K distillation of Fox et al. (2011) stickiness and Bulla & Bulla (2006) duration evidence. Directly targets the whipsaw/turnover channel that determines whether risk-timing survives costs (the mechanism behind the Bulla 2011 and Shu-Mulvey 2024 results). One knob, one sentence of meaning, zero new observation data.
3. **One cross-sectional emission channel: breadth (fraction of panel names up) or cross-sectional dispersion — computed from the panel you already have, PIT-clean by construction.** Precedent: Kritzman et al. (2012) turbulence-regime timing; breadth's cross-country robustness (Economic Modelling 2019). This is the only upgrade that adds *new information* rather than un-throwing-away old information, and it is the cheapest test of whether the filter's ceiling is the observation or the regime machinery. Cap at one channel to respect the parsimony budget.

**Explicitly deferred**: HDP-HMM (nonparametric state count = illegible by construction), full HSMM (dominated by t + sticky at far higher belief-state complexity), K=4 (identification risk on your window lengths; revisit only if the K=3 crisis state visibly conflates crash and recovery), online EM (breaks simulatability and the gate's frozen-knob contract). **One flagged alternative**: if upgrades 1-2 fail the v4 gate, the statistical jump model (Shu, Yu & Mulvey 2024) is the strongest-evidence daily risk-timing method in this survey and is itself born-legible (K centroids + one jump penalty, DP-solvable state sequence) — it is a legitimate replacement for the HMM core rather than an upgrade of it.

Sources:
- [Hamilton, Regime-Switching Models survey (UCSD)](https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf)
- [Bulla et al., Markov-switching asset allocation: do profitable strategies exist? (J. Asset Management 2011)](https://www.researchgate.net/publication/50245257_Markov-switching_asset_allocation_Do_profitable_strategies_exist)
- [Rydén et al. / Bulla & Bulla, Stylized facts and hidden semi-Markov models (CSDA 2006)](https://www.sciencedirect.com/science/article/abs/pii/S0167947306002374)
- [Guidolin & Timmermann, Asset Allocation under Multivariate Regime Switching (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=940652)
- [Maheu, McCurdy & Song, Bull corrections and bear rallies (Toronto WP)](https://www.economics.utoronto.ca/public/workingPapers/tecipa-402.pdf)
- [Fox, Sudderth, Jordan & Willsky, A sticky HDP-HMM (AoAS 2011)](https://ics.uci.edu/~sudderth/papers/aoas11hdphmm.pdf) and [LIDS tech report](https://people.eecs.berkeley.edu/~jordan/papers/stickyHDPHMM_LIDS_TR.pdf)
- [Song, Infinite HMM for regime switching and structural breaks (JAE 2014)](https://onlinelibrary.wiley.com/doi/10.1002/jae.2337)
- [Shu, Yu & Mulvey, Downside risk reduction using regime-switching signals: statistical jump model (arXiv 2402.05272)](https://arxiv.org/html/2402.05272v2)
- [Dynamic factor allocation leveraging regime-switching signals (arXiv 2410.14841)](https://arxiv.org/html/2410.14841v1)
- [Nystrup et al., Long memory and HMMs with time-varying parameters (J. Forecasting 2017)](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2447)
- [Nystrup, Dynamic Asset Allocation PhD thesis (DTU)](https://backend.orbit.dtu.dk/ws/files/152802375/phd465_Nystrup_P.pdf)
- [Fons et al., Feature Saliency HMMs for smart beta (ESWA 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0957417420305443)
- [Herding for profits: market breadth and global equity returns (Economic Modelling 2019)](https://www.sciencedirect.com/science/article/pii/S0264999319312982)
- [S&P DJI, Dispersion: Measuring Market Opportunity](https://www.spglobal.com/spdji/en/documents/research/research-dispersion-measuring-market-opportunity.pdf)
- [Cross-Sectional Dispersion of Risk in Trading Time (NBER w26329)](https://www.nber.org/papers/w26329)
- [Heavy-tail emission families for equity-return HMMs (arXiv 2606.23492)](https://arxiv.org/pdf/2606.23492)
- [Regime-switching factor investing with HMMs (MDPI 2020)](https://www.mdpi.com/1911-8074/13/12/311)
- [Predicting risk-adjusted returns using an asset-independent regime-switching model (arXiv 2107.05535)](https://arxiv.org/pdf/2107.05535)

---
## Investigation 4: literature — risk-timing/sizing

# Literature lens 2 — Risk-timing / sizing policies: what certifies honestly, what survives A-shares, and what to try on csi500 2021-22

## 1. Volatility-managed portfolios (Moreira-Muir 2017) — RETURN-lane claim, fragile

- **Claim**: scale exposure by c/σ̂²(t-1) (inverse realized variance). In-sample spanning alphas ~2-5%/yr on market and momentum factors; Sharpe gains up to ~0.2.
- **The critique is decisive for our purposes**: Cederburg, O'Doherty, Wang & Yan (JFE 2020), over 103 equity strategies: the spanning-regression alphas are **not implementable in real time**; honest out-of-sample versions earn *lower* certainty-equivalent returns and Sharpe than unmanaged buy-and-hold for most strategies, due to structural instability of the spanning coefficients. Direct Sharpe comparisons show no systematic outperformance. This is precisely the "dev-grid winner fails the frozen gate" failure mode this project already knows.
- **China A-shares — worse than the US**: Wang & Li (Pacific-Basin Finance Journal 88, 2024) find A-share factor and industry portfolios have **higher returns following high-volatility months**, so the standard cut-exposure-when-vol-is-high portfolio is *spanned* (no opportunity-set expansion); only the **inverse** (scale-up-in-vol) version expands the set. A separate China working paper (Chen, 71 factors) finds 15/71 significant Sharpe improvements and a multifactor OOS SR of 1.50 vs 0.99 — but at monthly factor level, pre-cost, and in tension with the PBFJ spanning result. A China mutual-fund study (PBFJ 2023) finds **downside**-volatility-managed versions beat total-vol-managed (avg SR ~1.02).
- **Legible form**: `e_t = min(cap, c/σ̂_t²)` — one line, fully simulatable.
- **120-day block-bootstrap detectability**: RETURN lane, expected ΔSharpe ≈ 0 to +0.2 (possibly negative in A-shares) — **undetectable** on a ~185-day hold; expect an honest reject.

## 2. Volatility targeting (Harvey et al 2018) — RISK-lane claim, robust

- **Claim**: `e_t = σ*/σ̂_t` (inverse vol, not variance). For risk assets (equities, credit): modest Sharpe gain (~+0.05-0.1, via the leverage effect); the **robust, near-mechanical effect is tail/drawdown reduction** — left-tail events are less severe because crashes occur in high-vol states when exposure is already cut. Vol clustering (1-month realized vol forecasting R² is high everywhere, including CSI indices — confirmed in recent Realized-GARCH work on SSE50/CSI300/CSI500) makes the *realized-vol reduction* close to deterministic.
- **China**: the vol-clustering channel is intact (CSI vol is highly forecastable); the Wang-Li caveat applies only to the RETURN side, not the vol/DD side. The strong Chinese low-volatility anomaly (Blitz et al, J. Asset Mgmt 2021: SR 0.51 low-risk vs 0.00 high-risk) supports vol as the priced dimension in A-shares.
- **Detectability**: a 20-40% realized-vol cut and a proportional max-DD cut in a bear window detect at almost any sample size — **the single most detectable effect available at a 120-day block bootstrap**. The risk is the RETURN non-inferiority leg (see recommendation).

## 3. Conditional volatility targeting (Bongaerts, Kang & van Dijk, FAJ 2020) — the key refinement

- **Finding**: essentially all benefits of vol targeting are **concentrated in high-volatility states**. Acting only in extreme vol states (scale down in confirmed high-vol, unscaled otherwise) beats standard always-on targeting on risk-adjusted return, drawdown, *and* turnover.
- This maps one-to-one onto the existing stack: the K=2 belief from |r|>q80 **is** a high-vol-state detector, and the current failure (B3 tail cut fails return non-inferiority) is exactly the failure conditional targeting fixes — stay at exposure 1 in the calm state, act continuously only in the turbulent state.
- **Legible form**: two named states + one scalar; fully simulatable.

## 4. Trend / TSMOM overlays (Moskowitz-Ooi-Pedersen 2012) — undetectable here

- Diversified 12-month TSMOM SR ~1.0 across ~58 futures, but per-asset SR gain only ~0.1-0.3 gross, and Huang, Li, Wang & Zhou (JFE 2020) show asset-level predictability is statistically unreliable — the strategy ≈ a historical-mean (constant-tilt) strategy.
- **China**: index-level evidence is mixed-to-contrarian at longer lookbacks; technical/MA timing rules on Chinese indices mostly die under data-snooping correction (White reality-check studies). Also note: a TSMOM overlay requires the **sign** of returns, which the current binary |r|>q80 observation deliberately discards — it is a signal change, not just a policy change.
- **Detectability**: single-asset effect of ~0.1-0.3 SR on ~185 days with ~1.5 independent 120-day blocks — **hopeless**. Would likely have *helped* in this particular hold (flat/short through H1-2022), but that is exactly the phase-selection trap already logged twice (N7, company-text).

## 5. Drawdown control / CPPI (Grossman-Zhou 1993; Black-Perold) — legible but never certified

- Legible (`e_t = m·(W_t - floor)/W_t`), but it is convex insurance: it **buys** tail protection with expected return, de-risks *after* losses (toxic in reversal-heavy A-shares), and is path-dependent (bootstrap-hostile: resampled blocks scramble the wealth path, so the policy evaluated in the bootstrap is not the policy run live). No published honest out-of-sample Sharpe certification exists anywhere. DD delta would detect in the grind-bear, but return non-inferiority would likely fail the same way B3 did.

## 6. Continuous vs discrete 3-level sizing

No direct published horse-race, but the smoothing/continuous-targeting literature (e.g., smoothed vol targeting) shows continuous sizing cuts turnover-cliff whipsaw and threshold-boundary variance. For certification this matters indirectly: a continuous rule has no knife-edge {t1,t2} whose bootstrap distribution straddles the boundary, so the same true effect certifies with less variance. Zero alpha claim, low risk, preserves legibility (still one formula).

## Detectability calibration (the honest math)

Hold ≈ 185 trading days; 120-day blocks give ~1.5 non-overlapping draws — RETURN-lane Sharpe deltas below ~0.5-1.0 annualized are statistically invisible even with paired (common-book, ρ>0.9) bootstrap. RISK-lane vol/DD deltas of 20%+ are visible almost immediately because vol clustering makes them near-mechanical. **Only RISK-lane claims are honestly certifiable on this window**; the return leg can only realistically pass as a *non-inferiority* constraint, and the Wang-Li A-share result (returns higher after high vol) says any policy that floors exposure too low will fail it.

## Single recommended policy-class upgrade

**Belief-gated continuous volatility targeting (conditional vol targeting), replacing the discrete 3-level shell** — target the RISK lane with return non-inferiority:

```
e_t = 1                                if b_t < t1          (calm state — never de-risk)
e_t = clip(σ*/σ̂_t, e_min, 1)          if b_t >= t1         (turbulent state)
```

with σ̂_t = 20d realized vol of the EW panel, σ* = train-period target (e.g., train median vol), and a **floor e_min ≈ 0.4-0.5** (the A-share high-vol-then-high-return finding is the direct argument for a high floor — keep rebound participation, don't repeat B3's flat tail cut). This is the only class where (a) the published effect (Harvey 2018 tail/DD channel + Bongaerts 2020 conditioning) is large and mechanical enough for a 120-day block bootstrap to detect on a 9-month grind-bear, (b) the A-share-specific evidence supports the risk channel while warning off the return channel, (c) the existing belief filter is *already* the state detector the literature says to condition on, and (d) the functional form stays one legible line — two named states, one scalar, hand-simulatable, K-simplex-compatible. Optional second-order variant if the base rule certifies: swap σ̂ for downside semi-volatility (the China mutual-fund result says downside beats total vol).

Predicted gate outcome: RISK lane accept (vol/DD cut ~20-35% in the hold window, detectable), RETURN lane non-inferiority pass *conditional on the floor*; an unfloored or always-on version should be expected to fail exactly where B3 failed.

Sources:
- [Moreira-Muir critique — Cederburg, O'Doherty, Wang, Yan 2020, JFE (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3357038) / [full paper PDF](https://www.lehigh.edu/~xuy219/research/COWY.pdf)
- [Wang & Li, "Volatility-managed portfolios in the Chinese equity market", PBFJ 2024](https://ideas.repec.org/a/eee/pacfin/v88y2024ics0927538x24003263.html)
- [Chen, "Do Volatility-Managed Portfolios Work? Evidence from China" (71 factors)](https://yaohanchen.com/PDF/volatility_managed_portfolios02.pdf)
- [China mutual funds vol-managing, PBFJ 2023](https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002998)
- [Volatility-tail-risk-managed portfolios in China, Empirical Economics 2024](https://ideas.repec.org/a/spr/empeco/v66y2024i3d10.1007_s00181-023-02493-9.html)
- [Harvey et al, "The Impact of Volatility Targeting", JPM 2018 (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175538) / [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf)
- [Bongaerts, Kang, van Dijk, "Conditional Volatility Targeting", FAJ 2020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3636727) / [CFA summary](https://rpc.cfainstitute.org/research/financial-analysts-journal/2020/conditional-volatility-targeting)
- [Huang, Li, Wang, Zhou, "Time series momentum: Is it there?", JFE 2020](https://ideas.repec.org/a/eee/jfinec/v135y2020i3p774-794.html)
- [The Volatility Effect in China, J. Asset Management 2021](https://link.springer.com/article/10.1057/s41260-021-00218-0)
- [Technical trading rules in China under data-snooping correction (Alpha Architect summary)](https://alphaarchitect.com/technical-trading-rules-data-snooping-evidence-china/)
- [TSMOM and contrarian effects in Chinese indices (arXiv)](https://arxiv.org/pdf/1702.07374)
- [Realized-GARCH vol forecasting on SSE50/CSI300/CSI500/CSI1000](https://pmc.ncbi.nlm.nih.gov/articles/PMC11488709/)

---
## Investigation 5: literature — CN A-shares specifics

FINDINGS — Literature lens 3: CN A-shares regime/risk timing (csi300/csi500, 2018-2023)

Panel checked: `C:\Users\ivanp\RL for Time-Series Forecasting\data_RLagent_for_Joseph\data\adapters\_csi500_wide\csi300_model_ready.csv` has NO raw volume/OHLC — only `close, daily_return` plus derived `volume_ratio, obv, obv_pct_change, atr, atr_rel, macd, rsi_30, cci_30, dx_30, turbulence` and fundamentals. All candidates below are mapped to these columns.

## 1. Structural facts that shape regime timing on A-shares

- **Retail dominance is quantified, not folklore**: individuals own ~88% of free float; this motivated Liu–Stambaugh–Yuan to add a sentiment factor to their China model, and the sentiment proxy they chose is **abnormal turnover** (PMO factor, CH-4). Reversal and turnover anomalies exceed 1%/month even net of CH-3. Turnover = sentiment is the single most institutionalized China-specific fact. ([Size and Value in China, JFE 2019](https://faculty.wharton.upenn.edu/wp-content/uploads/2018/03/Size-and-Value-in-China.pdf), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304405X19300625))
- **T+1 settlement** produces a unique structure: significantly **negative overnight returns** (unlike T+0 venues), with intraday/overnight momentum that flips sign across the session boundary; daily reversal in China is excess retail liquidity provision, not compensated liquidity supply. This is a mechanism-level explanation for the project's B4 finding (alpha priced at daily) — daily-frequency alpha on A-shares is structurally crowded by retail reversal trading. ([Overnight return puzzle and the T+1 rule, J. Fin. Markets 2020](https://www.sciencedirect.com/science/article/abs/pii/S1386418120300033), [A unique T+1 trading rule in China](https://www.sciencedirect.com/science/article/abs/pii/S0378426611002561))
- **Leverage is causally implicated in crashes**: Bian–He–Shue–Zhou show account-level margin proximity to leverage caps → fire sales → abnormal declines that reverse (2015 crash; shadow-margin worst). Margin balance is a genuine regime state variable — but it is an **external feed, not computable from this panel**. Its panel-visible footprint is vol spikes + cascading consecutive down days. ([NBER w25040](https://www.nber.org/papers/w25040), [BFI](https://bfi.uchicago.edu/working-paper/leverage-induced-fire-sales-and-stock-market-crashes/))
- **The 2021-09..2022-06 window (csi500 hold) was policy-jump driven**: the ~18-month regulatory crackdown (Ant/Alibaba/Meituan fines, data/antitrust rules) plus the Shanghai/Beijing lockdowns of Apr–May 2022. These are exogenous jumps poorly forecastable from price trend; the bear was a low-volatility grind punctuated by policy shocks. ([CNBC](https://www.cnbc.com/2022/05/18/china-signals-easing-of-its-tech-crackdown-but-dont-expect-a-u-turn.html), [china-briefing](https://www.china-briefing.com/news/china-tech-crackdown-recent-developments-signal-easing-regulations/), [asiafundmanagers](https://www.asiafundmanagers.com/int/china-tech-stocks-feel-the-brunt/))

## 2. Ranking of drawdown-regime observables per the literature

1. **Abnormal turnover** — strongest and most-cited. Turnover proxies speculative belief dispersion (Mei–Scheinkman–Xiong A-B premia work: [Princeton](https://www.princeton.edu/~wxiong/papers/china.pdf)); in Greenwood–Shleifer–You, turnover + volatility + run-up path jointly predict **crash probability** (not mean return) and can be timed profitably ([Bubbles for Fama, JFE 2019](https://www.nber.org/papers/w23191)); it is the CH-4 sentiment factor. Crucially for the grind-bear: high abnormal turnover predicts crashes, **collapsing turnover accompanies grind persistence** — a two-sided signal.
2. **Trend / moving averages** — unusually strong in China. The Liu–Zhou–Zhu trend factor (MA prices at short/mid/long horizons + volume signals) earns monthly Sharpe 0.48 vs 0.11 for market — attributed to ~80% individual trading. ([Trend Factor in China](https://wrds-www.wharton.upenn.edu/documents/1119/WARSP_TrendChina.pdf), [SSRN 3402038](https://www.ssrn.com/abstract=3402038)); TSM on mainland indices also documented ([arXiv 1702.07374](https://arxiv.org/pdf/1702.07374)).
3. **Index volatility** — informative for crash *probability*, but its *expected-return* sign is inverted vs the US: factor/industry returns are HIGHER after high vol (see lens on overlays below). Vol alone is a poor defensive trigger; combined with run-up and turnover it is a crash-probability feature (GSY).
4. **Margin debt** — causally strongest (see Bian et al.) but external; **northbound Stock Connect flow** — genuine "smart money" predictive content but conditional (works when global inflation low) and fast-decaying ([Liao–Tang–Xu, IJFE 2024](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2751), [time-varying interaction](https://www.sciencedirect.com/science/article/abs/pii/S1544612324011061)). Both are **not computable from this panel** — exclude.
5. **Breadth** — China-specific direct evidence is thinner (mostly generic: <40% of stocks above 200d-MA near index highs = deterioration; [stockalarm summary](https://pro.stockalarm.io/blog/stock-market-crash-warning-signs)), but market-state-transition early-warning work on Chinese data shows internal-organization/breadth-style shifts lead crash labels by days ([Frontiers in Physics 2025](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1647667/full)). Moderate confidence, cheap to compute.
6. **Lottery/skew (MAX)** — MAX negatively predicts returns in China, concentrated in the overnight component and high-retail stocks; conditional skewness relates to crash risk ([MAX effect in China](https://www.scirp.org/pdf/tel_2023081014211370.pdf), [lottery mindset & IVOL](https://www.sciencedirect.com/science/article/abs/pii/S1062940820301637)). Secondary.

## 3. Do standard vol-managed / trend overlays work on A-shares?

- **Vol-managed (Moreira–Muir style deleverage-when-vol-high): NO at index/factor level.** Chi–Qiao–Yan–Deng: Chinese factor and industry portfolios earn HIGHER returns after high volatility, so vol-managed versions are spanned by the originals; it is the **inverse** (vol-scaled, increase exposure in volatile times) that expands the opportunity set ([IRF 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/irfi.12336)). A 2024 Pacific-Basin study finds vol-management can work, but the gains come from high-arbitrage-risk stocks and price-limit effects — i.e., cross-sectional implementation, not an index exposure dial ([Pacific-Basin 2024](https://www.sciencedirect.com/science/article/abs/pii/S0927538X24003263)). Direct relevance: the current K=2 filter is effectively a vol-gate on an EW book — the A-share literature says that exact overlay family has no documented edge on returns; its defensible product is crash-probability/drawdown timing only (consistent with B3).
- **Trend overlays: YES, best-documented overlay family in China** (retail herding sustains trends; trend factor Sharpe dominates market/size/value). But trend is a slow detector, and the 2021-09..2022-06 hold window was a policy-jump grind where trend fires late on entry and whipsaws on the Apr-2022 lockdown leg — the literature's answer to grind-bears is turnover collapse + breadth, not vol.

## 4. Concrete, citable indicator candidates computable from THIS panel

All cross-sectional (CS) aggregates over the ~300/500 tickers per date; all leak-free from existing columns:

1. **Panel abnormal turnover** = CS-mean of `volume_ratio`, z-scored on trailing 252d. Cite CH-4 PMO (Liu–Stambaugh–Yuan) + GSY. Two-sided read: high-z → crash-prob up (cut tail exposure); low-and-falling during drawdown → grind persistence (stay defensive).
2. **Breadth-200** = fraction of tickers with `close` > their 200d MA (also cheap variants: fraction `rsi_30`>50, fraction `macd`>0, advance-decline share from sign of `daily_return`). Cite breadth deterioration rule + Chinese early-warning work.
3. **GSY crash-probability composite** = trailing 2y run-up of EW panel × CS-mean `atr_rel` z × abnormal-turnover z. This is exactly the "attributes of the run-up" recipe from Bubbles for Fama and is the literature-correct way to use volatility on A-shares (probability feature, not return-sign feature).
4. **Trend state** = EW panel price vs 50d/200d MA + CS-mean sign(`macd`) as a 3-state trend flag. Cite Liu–Zhou–Zhu trend factor (which explicitly mixes MA horizons with volume — `volume_ratio` interaction available).
5. **Signed flow proxy** = CS-mean `obv_pct_change` (crude in-panel stand-in for the un-computable northbound/margin flows).
6. **Herding/dispersion** = CS std of `daily_return` (dispersion collapse while trend is down = herding fragility; standard CSAD herding literature on China — Tan et al. 2008 / Chiang–Zheng, from prior knowledge, not fetched this session).
7. **Lottery share** = CS-mean of trailing-20d max `daily_return` (MAX). Elevated lottery demand → lower forward returns, overnight-concentrated.
8. `turbulence` column is already a Mahalanobis-style regime input — z-score it rather than adding a new vol feature.

## 5. Direct implications for the K=2 belief filter

- The current observation (binary |EW return| > q80) is a pure vol-gate; the A-share literature says vol's return-sign is inverted in China and the 2021-09..2022-06 hold window was a **low-vol grind** — the filter's observable is near-blind exactly where the csi500 hold must live. The two literature-backed observables that DO see grind-bears and are computable here are **abnormal turnover (falling)** and **breadth (persistently low)**.
- Minimal legibility-preserving upgrade consistent with the literature: keep K=2 HMM, replace/augment the binary emission with a small named vector — {signed EW return bucket, abnormal-turnover z, breadth-200} — every element human-simulatable and citable (CH-4 turnover-as-sentiment; GSY crash composite; breadth rule). Sign restoration matters because T+1/retail structure makes down-moves and up-moves informationally asymmetric (short-sale constraints → crashes cluster on the downside).
- Do not expect a return edge from any vol-managed exposure dial on the index (Chi et al. spanning result); the defensible lane remains RISK (drawdown), matching the existing B3 read.

Sources: [Liu–Stambaugh–Yuan JFE 2019](https://www.sciencedirect.com/science/article/pii/S0304405X19300625), [Wharton PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2018/03/Size-and-Value-in-China.pdf), [Bian–He–Shue–Zhou NBER w25040](https://www.nber.org/papers/w25040), [Greenwood–Shleifer–You, Bubbles for Fama](https://www.nber.org/papers/w23191), [Chi–Qiao–Yan–Deng IRF 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/irfi.12336), [Vol-managed portfolios in Chinese equity market, Pacific-Basin 2024](https://www.sciencedirect.com/science/article/abs/pii/S0927538X24003263), [Trend Factor in China (Liu–Zhou–Zhu)](https://wrds-www.wharton.upenn.edu/documents/1119/WARSP_TrendChina.pdf), [TSM in Chinese indices arXiv](https://arxiv.org/pdf/1702.07374), [T+1 overnight puzzle JFM 2020](https://www.sciencedirect.com/science/article/abs/pii/S1386418120300033), [T+1 theory & evidence JBF](https://www.sciencedirect.com/science/article/abs/pii/S0378426611002561), [Liao–Tang–Xu northbound IJFE 2024](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2751), [northbound time-varying FRL 2024](https://www.sciencedirect.com/science/article/abs/pii/S1544612324011061), [Mei–Scheinkman–Xiong](https://www.princeton.edu/~wxiong/papers/china.pdf), [crash early-warning Frontiers 2025](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1647667/full), [MAX effect China](https://www.scirp.org/pdf/tel_2023081014211370.pdf), [lottery mindset IVOL](https://www.sciencedirect.com/science/article/abs/pii/S1062940820301637), [breadth warning signs](https://pro.stockalarm.io/blog/stock-market-crash-warning-signs), [crackdown timeline CNBC](https://www.cnbc.com/2022/05/18/china-signals-easing-of-its-tech-crackdown-but-dont-expect-a-u-turn.html), [china-briefing](https://www.china-briefing.com/news/china-tech-crackdown-recent-developments-signal-easing-regulations/).