# csi500 forward run + how the CrystalScore metrics are used (and how they change)

Two questions answered with runs. Code: [hl_real_csi500.py](../interpretability/hl_real_csi500.py),
[crystal_score_crystal1.py](../interpretability/crystal_score_crystal1.py). Reports: the matching `*_report.json`.

## 1. Run on the real csi500 (the forward substrate)
The csi500 panel (**344 names, 2018-2023, 41 features** incl. fundamentals + regime probs) is the right substrate to
carry forward. I ran the HL/CRYSTAL-1 governed loop on it exactly like the Dow run: a K=2 self-supervised belief filter
(trained 2018-2020, frozen — it works: belief mean 0.318, std 0.321, 20% toxic days), a belief-conditioned exposure
policy, anchor = static-full (EW), a **block-bootstrap CRN** gate on disjoint DEV (2021H1) / HOLDOUT (2021H2-2022H1)
windows, honest OOS (2022H2-2023Q1).

**Result: the governed loop certified 0 belief-mode edits** (39 no-dev-signal, 1 holdout-reject); OOS static-full and
HL-belief are identical (risk-adj −0.050, Sharpe 0.483, maxDD −0.109). Even the pre-registered B3 config fails the gate
(dev +0.008 but **holdout −0.138**). This is **consistent with and stronger than B3** (which failed csi500 4/4): under a
proper walk-forward firewall, csi500's grind-bear regime yields **no certifiable risk-adjusted belief-VoI**. The honest
read: csi500 is the better *forward* substrate (more names, richer features, the belief filter runs cleanly), but its
*alpha/risk* is not certifiable here — the firewall correctly refuses, the same verdict as Dow. Its value is as the
richer transparency/control substrate, not a certified edge (unchanged program conclusion: on real markets the regime is
priced).

## 2. The CrystalScore metrics — how they're used, and whether they change
**How they're used (honest framing).** CrystalScore is the **north-star target of the controllability/interpretability
track** (not alpha): `CrystalScore = Faithfulness × Simulatability × Stability` at a parsimony budget K≤9, ported from
the LLM-evaluation axes. Through this whole CRYSTAL-1 build I measured its **axes natively under the C-ladder names**
rather than assembling the scalar: **faithfulness = C-1/C-2** (the belief write causally controls the action, proved vs
the belief-MDP), **simulatability = M6** (an ≤8-leaf story tree IS the policy at return parity), **stability = B2**
(the mechanism replicates 3/3 seeds), **controllability = C-1 dose-response / the governor**. This run finally computes
the **same scalar** for CRYSTAL-1 so it is directly comparable to the R6c v1 number.

**Do they change? YES — measurably, and in the predicted direction:**

| policy | Faithfulness | Simulatability | Stability | **CrystalScore** |
|---|---|---|---|---|
| **R6c v1** (stance) | 1.00 | **0.244** | 0.619 | **0.151** |
| **CRYSTAL-1 corner PPO** | 1.00 | **0.939** | 0.991 | **0.931** |
| **CRYSTAL-1 M6 soft-tree** | 1.00 | **0.938** | 1.00 | **0.938** |

CRYSTAL-1's CrystalScore is **~6× R6c's** (0.93 vs 0.15), and the entire gap is **Simulatability** (0.94 vs 0.24): a
K≤9 (≤8-leaf) story reproduces CRYSTAL-1's policy almost exactly, but **cannot** compress R6c's 64-d latent stance
(0.244). Faithfulness is 1.0 for both (steering works on both); stability is near-1 for CRYSTAL-1. So the metric moves
exactly as the born-legible thesis predicts: CRYSTAL-1 sits near the **top of the (behavioral-complexity, CrystalScore)
frontier** where R6c sits low.

**What makes them change:** the scalar rises when the policy becomes more *simulatable at a fixed parsimony budget* —
i.e. when a short named story reproduces behavior. That is precisely what the born-from-the-brain design buys, and what
each growth/certification step protects: M6 keeps simulatability at parity (not a post-hoc probe), the governor/writ
ladder keep faithfulness certified, B2/re-roll keep stability. Alpha does **not** enter CrystalScore — consistent with
the pivot (the csi500 run above yields no certified alpha, yet CRYSTAL-1's CrystalScore is 0.93; the two axes are
orthogonal, by design).

## Bottom line
csi500 is the right forward substrate and the belief filter runs on it, but under the firewall it yields no certifiable
edge (honest, expected). And the CrystalScore metrics are live and discriminating: computing them natively puts
CRYSTAL-1 at **0.93 vs R6c's 0.15** on the identical scalar — the born-legible advantage, quantified on the very metric
the interpretability track was built to optimize.
