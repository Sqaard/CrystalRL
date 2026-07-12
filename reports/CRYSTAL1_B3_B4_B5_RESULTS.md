# CRYSTAL-1 — B3 / B4 / B5 results (the real-world stages: honest, mostly conditional-or-negative — which are results)

Scripts: `interpretability/{crystal1_b3_riskmode, crystal1_b4_bridge, b5_crystallize, b5b_k14}.py` (+ JSON).

## B3 — risk-mode on real daily panels: CONDITIONAL PASS (crash-insurance, not universal)

CRYSTAL-1's L1 trained self-supervised on REAL market observation streams (train-window only, leak-safe);
pre-registered 3-mode exposure policy (FULL/REDUCED/DEFENSIVE) over the EW book; 10bp costs.

| | EW buy&hold | **CRYSTAL-1 belief-mode** | trailing-RV incumbent | HC-1 noise |
|---|---|---|---|---|
| **Dow-29 OOS 2017-2023** (1548d) | Sharpe 0.772, maxDD **−0.329**, Calmar 0.425 | Sharpe 0.725, maxDD **−0.167**, Calmar 0.428 → **PASS 4/4** | Sharpe 0.927, maxDD −0.154, Calmar **0.667** | Sharpe 0.03 (collapse) |
| **csi500 OOS 2021-23** (521d, grind bear) | Sharpe 0.695, maxDD −0.251 | Sharpe 0.322, maxDD −0.251 → **FAIL 4/4** | Sharpe 0.611 | — |

- **L1 read the real market's regime structure without labels**: Dow p_stay [0.982, 0.954], burst signatures
  [0.10, 0.43] — and its belief stream shows the REAL filtering signature (belief-N7 ASYM 100/100, levels and
  increments). The stance is structured (2/2). HC-1 holds: noise belief destroys everything.
- **Honest verdicts:** (1) the DD-halving on Dow replicates the alpha-era law (de-risking cuts DD, not
  Sharpe) with a *certified transparent* mechanism; (2) the **incumbent trailing-RV is stronger** on Dow
  (Calmar 0.667 vs 0.428) — CRYSTAL-1 risk-mode's value is TRANSPARENCY at comparable-not-better performance,
  exactly as the blueprint's claim-ceiling anticipated; (3) **csi500 grind-bear FAIL replicates the old
  regime-conditionality** (vol-de-risking is crash insurance; it hurts in chop) — the enable-table
  (book-state + market-regime conditional) is a necessity, not an option.

## B4 — the bridge: the corner does NOT open on daily execution PROXIES (clean negative with a mechanism)

Corwin-Schultz spread estimates from real csi500 OHLC (344 names, 2018-2023) + regime-conditional
provide-edge + the WH2 VoI machinery, λ-swept:

- adverse selection rises **significantly** in toxic (90 → 120 bp);
- spread widening in toxic is **NOT significant** (CI [−30, +52] bp straddles 0);
- ⇒ provide-edge is regime-FLAT at every λ ∈ [0.05, 0.5] → **no sign flip, VoI = 0.0** — the corner stays
  closed at daily-proxy execution economics.
- **The data requirement is now empirically demonstrated, not asserted:** daily bars cannot resolve execution
  economics (CS "spreads" ~460bp conflate range and volatility on volatile CN mid-caps — flagged). The bridge
  needs fill-level / intraday LOB data. That acquisition IS the program's main open bet, unchanged.

## B5 — the constructive turn: mechanism v1 (self-distillation reward bonus) FALSIFIED as a legibility lever

Round-based "become the policy your story says you are" (train → distill own K-leaf tree → train with
agreement bonus → re-distill), on G=12 where the budget gap is real (sim@K9 baselines 0.583–0.611):

| arm | sim@budget | baselines | return | Goodhart collapse? |
|---|---|---|---|---|
| K=9 bonus, seed 1 | 0.685 | ≤0.611 | 4.65 | none (x 1.15–1.79, 2/2) |
| K=9 bonus, seed 2 | 0.587 | ≤0.611 | **11.75** | none |
| K=14 (=G+2) bonus, seed 1 | 0.869 | ≤0.855 | **21.08** | none (x 1.37–1.91, 2/2) |

- **No reliable legibility gain at either K<C\* or K=C\*** (gains +0.01…+0.07, under the pre-set margins;
  one seed null). The naive reward-shaping mechanism does not buy simulatability. Policies OUTGROW their
  stories during training (within-round story agreement decays 0.94→0.67) — the story chases, never leads.
- **No Goodhart collapse anywhere** (complexity/structure/return held) — the gates work.
- **Unexpected side observation (unconfirmed, flagged):** both bonus arms posted unusually high returns
  (11.75, 21.08 vs baseline range 3.99–9.63) — "self-consistency as a learning regularizer"? Confounded by
  +30% training steps and n=2; queued as a future question, NOT a claim.
- **Blueprint consequence:** the constructive turn needs a STRUCTURAL mechanism (e.g., the story tree as the
  actual policy head / hard architectural parsimony), not reward shaping. B5-v2 design note recorded.

## The B3-B5 net for the blueprint
1. Risk-mode is deployable ONLY as conditional crash-insurance with a transparency (not performance) value
   proposition; the trailing-RV incumbent remains the performance benchmark to beat.
2. The corner genuinely requires intraday data — every cheaper substitute is now empirically exhausted
   (daily returns: WH2; daily execution proxies: B4).
3. Interpretability-by-construction (L1) survived contact with real data (readable world model, filtering
   signature, load-bearing belief); interpretability-by-training-pressure (B5 v1) did not.
