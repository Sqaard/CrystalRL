# CRYSTAL-1 v5 — the upgraded model on real data (задачи 1–3)

**User's hypothesis:** «проблема в самой модели и предобработке имеющихся реальных данных».
Three deliverables: (1) the oracle decomposition, (2) the weakness map + literature design,
(3) the HL loop re-run with the upgraded model.

## 1. Oracle verdict (CRYSTAL1_ORACLE_CEILING.md)

**Модель зарабатывает с oracle-сигналами** — the hypothesis is confirmed in its *signal* half:
the unchanged 4-knob shell certifies through the honest v4 gate with a realistic forward-vol oracle on
Dow (z=2.33; hold DD −0.06 vs −0.33) and monetizes hugely raw on both panels; the *current* K=2
binary-|EW-ret| signal's class headroom is **negative** out-of-window. Separately: csi500 has a
*second* ceiling — even oracle-grade regime effects top out at z≈1.5–1.9 per 120-day window on the
2021–22 grind-bear (both lanes), i.e. **the honest per-window bar exceeds oracle effects there**.

## 2. Weakness map + literature (CRYSTAL1_UPGRADE_RESEARCH.md)

Five investigations (internal audit; panel audit; regime-model, sizing, and CN-A-share literature)
converged on the U1/U2/U3 package — with two data-driven surprises: classic **breadth is DEAD on this
panel** (white noise, autocorr −0.06), and the panel's GRU columns are **leakage-quarantined**
(in-window IC t=+10). The design: U1 signed vol-normalized 6-symbol alphabet (× cross-sectional
dispersion) + sticky K-selected filter with pre-registered state naming; U2 belief-gated conditional
vol targeting (Bongaerts 2020) with liveness quantile-gate, hysteresis, floor E_MIN=0.5, and only TWO
searched knobs; U3 cash yield on the sleeve + sell-side stamp asymmetry + rf=0/execution-lag
acceptance stress. Explicit NOT-do list: GRU features, breadth, return-lane vol management (spanned
in A-shares), TSMOM (phase-selection trap), CPPI (B3's failure with more knobs), K≥4/HDP/online-EM.

## 3. The v5 loop ([hl_v5_crystal1_upgraded.py](../interpretability/hl_v5_crystal1_upgraded.py))

**The model is genuinely better — the pre-registered falsifier battery says so, not us:**
held-out LL beats memoryless (and K=3 beats K=2 on equal footing, kept by its own selection rule);
states occupied and named (CRISIS/CALM/GRIND, dwell 74/382/20d on csi500); symbol firing rate
stationary on the hold (the old signal was near-silent there — the direct cause of the inert-window
audits); the block-shuffled-belief placebo fails while the real belief adds dev edge (belief is
load-bearing). Two falsifiers honestly FAIL and are reported: U2b arming is thin on early hold windows,
and Dow's firing rate drifts out of band (U1c).

**Gate result: 0 certified on both panels — but now with a full diagnosis instead of a mute null:**

| Panel | The verified full-hold diagnosis |
|---|---|
| csi500 | **Sign problem under a detectability ceiling.** Full-hold paired delta of the literature default = **−1.7%/yr** (RETURN z ≈ −1.1); downside gain thin (z_dsd = 1.51 — *exactly* the oracle per-window ceiling). 0/10 seeds pass two legs. No window design could certify here. |
| Dow | **Real risk reduction bought at a return cost the certification rightly refuses.** The belief-gated vol targeting buys a genuine, statistically detectable downside-deviation cut: **+3.3 bp/day, z_dsd = 2.31 > 2.15 on the full 504-day hold, surviving rf=0 and 1-day lag** — but pays **−4.6%/yr of return** for it, failing the pre-registered 5%/yr non-inferiority leg ~20× over (ni = +0.18), on the hold AND again on the frozen OOS (z_dsd 3.29, ni −0.08). The binding leg is return non-inferiority, not effect size. |

## Verification (2 adversarial agents) + the confirmatory counterfactual

**DISCIPLINE → CONFIRMED.** PIT clean, verified numerically (a −10% return shock injected mid-hold
changes NO exposure at or before the shock day; truncation reproduces prefixes bit-exactly; the whole
U1→U2 pipeline re-implemented independently matches to 0.0). All 16 priced reviews, filter cards,
falsifiers, and perf rows reproduce byte-exactly. Teeth proven live: a +3bp/day sentinel is ACCEPTED
through the real path on both panels (with rf0/lag stress executing), a zero-mean sentinel REJECTED,
a boosted canary escape freezes the gate.

**THE COUNTERFACTUAL (the load-bearing check):** a pre-registered single confirmatory shot of the
literature default at the cheapest bar (2.15) on the **full 504-day Dow hold** does **NOT** certify —
the risk leg passes (z_dsd 2.31) but the return-non-inferiority leg fails by an order of magnitude.
**The null is design-robust: the rotating-window/alpha-wealth machinery did not cost a certification.**

**Honest defect ledger (verifier-found, disclosed):**
- **D1, the most instructive:** the Dow "anchor" was NOT buy-and-hold — Q_ON=1.01 clamps to the rolling
  252-day *maximum* and strict `>` arms on new b_risk highs, so the baseline itself embedded crisis
  vol-targeting (212 armed days, 75 in the COVID hold) and cut hold maxDD **−32.9% → −21.2% at Sharpe
  0.912 → 1.246** vs true buy-and-hold. PIT-legal and gate-conservative (a harder baseline), but "0
  accepts" on Dow means "nothing beats the accidentally-protected anchor". The effect is COVID-episode-
  concentrated (26 armed OOS days ⇒ anchor ≈ BH there) — the B3 single-episode caveat applies in full;
  treated as a *lead* for a pre-registered new-high-arming mechanism test on fresh data, not a result.
- D2: the two harm canaries were accidentally identical configs (coverage overstated; escape⇒freeze
  wiring separately verified live). D3: the RISK confirm window is only fully disjoint on Dow (~60%
  overlap on csi500; unexercised — no RISK accepts). D4: the U2a placebo shuffles b_risk but not
  b_cri/sigma20 (partial). D5: E_GRIND single-knob moves were behaviorally inert while the gate was
  disarmed (10/30 rounds on a dead arm). K-selection margins are epsilon-thin (Dow a 4-decimal tie) —
  K=3-vs-K=2 is *unresolved*, and not load-bearing (anchor certified regardless).

## Bottom line (все три задачи)

1. **Oracle: модель зарабатывает** — the shell+gate certify with oracle signals; the old K=2 signal was
   the bottleneck (its class headroom is negative). The user's model/preprocessing hypothesis is
   confirmed in its signal half, refuted in its shell half.
2. **The upgrade is real as a model**: falsifier-passing filter (live, informative, placebo-failing),
   literature-grounded legible policy, fair execution — and it *found a real effect*: on Dow the
   belief-gated vol targeting delivers a statistically solid downside cut (z_dsd 2.31–3.29 across
   hold/OOS, robust to rf=0 and execution lag).
3. **The certification verdict is an honest, fully-diagnosed refusal**: that downside cut costs
   −4.6%/yr of return — outside the pre-registered non-inferiority region; csi500 adds nothing
   certifiable at daily granularity (sign wrong under an oracle-level ceiling). The daily-frequency
   door is now closed with a *complete* causal story (B4: alpha priced; oracle: per-window ceilings;
   v5: real-but-costly risk timing). The standing route to a certified accept remains the substrate
   where the VoI gate can open — intraday CN (recorder accumulating since 2026-07-06) — and the
   D1 new-high-arming lead as a pre-registered test on data no eye has touched.
