# Global analysis — information sources beyond the daily bar (2026-07-19)

**Question (Ivan):** the daily-bar door is closed by an honest null (VoI≈0, and the whole
knowledge→model queue refused); find a NEW information source *orthogonal* to the daily OHLCV bar.

**Method.** 8 parallel researchers (one per source family) + a synthesis judge + 8 adversarial
verifiers, all given the project's honest priors (VoI≈0 daily; insider Form-345 small-cap-only;
FINRA short-vol underpowered; text = macro re-encoding; the certification discipline and the
artifact register). 1.18M tokens, 245 web calls. Raw:
[beyond_daily_bar_analysis_20260719_raw.json](beyond_daily_bar_analysis_20260719_raw.json).

## Lead with the null: 8/8 finalists WEAKENED, 0 CONFIRMED

The adversarial verifiers weakened **every** shortlisted candidate. The honest synthesis verdict:
*"mostly another set of nulls with better provenance, with a realistic 1–2 marginal RISK-DIAL
refinements surviving — and the return-VoI wall stays closed, because none of these is a
return-timing source that beats a constant exposure dial."* This is the fourth independent
confirmation of the project's core result, now from the information-source side: **no beyond-daily
source on this large-cap universe is a return-alpha source; the only live question is whether any
sharpens the RISK dial.**

## The candidate ladder (ranked; every one is a RISK-channel refinement, not alpha)

| # | Source | Orthogonality | Access | Honest status |
|---|--------|---------------|--------|---------------|
| 1 | **MOVE−VIX divergence** z(MOVE)−z(VIX) + MOVE level | **HIGH** (swaption vol = different asset class, different participants) | **free-now** (^MOVE on Yahoo) | The single most genuinely-new free source — the bot has never seen rates-vol. Enemy = crisis-clustering (few independent events 2010-26, the FINRA short-vol power trap). |
| 2 | VIX term-structure slope (VIX3M−VIX, backwardation) | medium (term ≠ level) | free-now | Partly already in the certified queue (5th eye); genuinely different surface dimension. |
| 3 | **Yang–Zhang / HAR range-vol** as the risk-dial vol input | low (it IS vol) | free-now | The near-guaranteed modest gain: not orthogonal, but a 5–14× more efficient estimator — can only sharpen drawdown control a little at matched turnover. Costs nothing. |
| 4 | Commodity backwardation (5-eye) | high | feasible | **Already graduated internally**; awaits its pre-registered 2027-01 read. Re-run as an exposure-matched dial twin on 2024-26 before then. |
| 5 | Downside semivariance / signed-jump variation | medium | free-now | "Bad vol" carries asymmetry symmetric vol can't; Patton–Sheppard matches the drawdown lane. |
| 6 | Chicago Fed ANFCI + NFCICREDIT/NFCILEVERAGE | medium (orthogonalized to activity) | free-now (ALFRED vintages) | Leverage subindex carries balance-sheet quantities no price encodes; weekly, ~1-week lag. |
| 7 | Broad USD (DTWEXBGS) + **USDCNH** | medium | free-now | The rare source with a native CN leg — USDCNH is first-order csi500 risk. BIS: dollar carries risk-appetite "over and above VIX". |
| 8 | GZ credit-curve slope (short−long EBP) | medium | feasible | The un-used piece of EBP (level was a PK-survivor that didn't solve the problem); partial out vol first. |
| 9 | SEC 8-K unscheduled-filing severity (EDGAR) | medium | free-now | Cleanest PIT text-event source; feasible on both universes (EDGAR + CNINFO). |

## The two places with the best odds (synthesis)

1. **Genuinely-orthogonal cross-asset regime eyes** — MOVE/VIX divergence (#1) and commodity
   backwardation (#4, already in flight). Free, structurally orthogonal to equity prices, shaped
   for the P(bear)→exposure lane where our one certified policy lives. Their enemy is
   crisis-clustering — any "win" must clear the **exposure-matched dial twin** (the CL-1c
   statistic), not B&H, and the i.i.d. placebo-market guard (the WB-1 lesson).
2. **The csi500 side is the structurally most-open door** — daily door shut, but USDCNH, CN
   funding stress (DR007), CN ETF-option VRP, and the live L5 recorder are under-exploited RISK
   inputs. Data-poor, short-history, PIT-fragile (the mixed-clock scar) → a build-and-wait bet.

The free **Yang–Zhang HAR vol upgrade (#3)** is the most certain but least exciting: a small,
near-guaranteed drawdown-control gain at zero data cost.

## Notable exclusions (the verifiers' catches — the graveyard)

Dealer gamma (GEX): correlation collapses −0.36→−0.03 once VIX+ATM IV controlled. Single-name
option smirk: 2/3 a borrowing-fee proxy, Dow-30 is the worst case. VPIN: mechanical re-encoding of
volume (Andersen–Bondarenko). Auction imbalance: paid, return-timing, turnover >> 30bp. COT: weekly
+ 3-day lag, 70% return-chasing. Short interest: power lives in the microcaps we exclude. Retail
order flow (BJZZ): needs paid TAQ; large-cap post-2016 null. Satellite/card/web alt-data: paid,
~nil coverage of our tickers, PIT-contaminated backfill. Copper/gold & Bitcoin: contemporaneous
not leading, ~zero orthogonal content. PEAD/analyst revisions: net-of-cost dead in liquid
large-caps. FOMC/seasonal effects: textbook post-publication deaths. US→csi500 lead-lag: a lagged
price (violates orthogonality) and priced-out in A-shares. Index/single-name VRP: our own CL-wave
NULL. Full reasons in the raw JSON.

## This-week test plan (zero new data; every one → the RISK dial, under the full discipline)

Seven free US series drop into the frozen-gate P(bear) belief as single causal columns and run the
standard discipline (**exposure-matched dial twin + i.i.d. placebo-market guard + 2022-23 hold /
2024-26 fresh OOS, net 30bp**): (1) VIX3M−VIX slope; (2) z(MOVE)−z(VIX) divergence; (3) Yang–Zhang
HAR vol as a drop-in vs the EWMA twin; (4) ANFCI + credit/leverage subindices with the true lag;
(5) broad USD + USDCNH; (6) GZ credit-curve slope, vol partialled first; (7) 8-K unscheduled-filing
severity. Separately: keep banking the CN L5 recorder; on the weeks collected, run daily-aggregated
OFI vs next-day realized vol under two mandatory controls (regress vs raw volume — the VPIN lesson;
within-year — the FINRA Simpson's lesson), verdict pre-registered as power-DEFERRED.

## Honest bottom line

The information-source door is not wide open, but it is not the flat wall the daily-price door is.
The realistic expectation is 1–2 small orthogonal drawdown-control refinements (best bet: MOVE
divergence + the free vol upgrade) and another stack of well-provenanced nulls. **No beyond-daily
source is expected to manufacture return alpha on this large-cap universe.** The genuinely new
frontier is the CN intraday/funding side — but that is a build-and-wait bet, not a this-quarter
certification. First concrete step, run now: the MOVE eye (the most orthogonal free source) through
the exposure-matched twin + placebo guard.
