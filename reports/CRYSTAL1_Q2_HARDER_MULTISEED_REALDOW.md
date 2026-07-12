# Q2 — harder task + multi-seed proposer distribution + real VoI>0 substrate attempt

Three executed pieces. Code: [substrate_hard.py](../src/hl/substrate_hard.py) (6-knob task),
[hl_multiseed.py](../interpretability/hl_multiseed.py), [hl_real_dow.py](../interpretability/hl_real_dow.py).

## 1. Harder task — a 6-knob interacting-lever substrate
`substrate_hard.py` adds a genuinely harder polygon task and, in doing so, **triples the knob surface** (3 → 6): a
provide threshold, a **provide-hysteresis** sticky band (cuts churn), a **risk-off DUMP** lever (unwind in toxic), a
horizon-unwind lever, an inventory threshold, and a **capacity cap** — levers that *interact*, so a greedy one-knob
search can stall in a local optimum. The full HL gate runs on it unchanged (the substrate is a drop-in module).

## 2. Multi-seed proposer DISTRIBUTION (not one comparison)
Ran the scripted teacher-guided proposer with **10 seed-varied search orders** on both substrates, through the full gate,
scored on a **fresh honest seed set** (20000–20400, never gated):

| substrate | knobs | gap-closed (mean ± std) | min / max | accepts |
|---|---|---|---|---|
| easy | 3 | **0.40 ± 0.17** | 0.15 / 0.66 | 5.5 |
| hard | 6 | **0.46 ± 0.08** | 0.23 / 0.49 | 3.7 |

**Key finding: the scripted proposer is ORDER-SENSITIVE.** The earlier headline "77%" was a *favorable fixed order*; over
random orders the scripted greedy coordinate-descent closes only **40 ± 17%**. This re-frames the LLM coding-agent's
**96%** (single run): a large part of its advantage is **robustness** — it reasons about which lever to move, rather than
following a fixed coordinate sweep, so it does not depend on a lucky order. (Caveat: LLM n=1 vs scripted n=10; the
6-knob random-search optimum 4.47 is a random-search lower bound, so its 46% is mildly optimistic. The clean claim is
*the scripted proposer's competence is order-fragile*, which is itself a reason to prefer a reasoning proposer.)
More knobs → **lower variance** (6-knob 0.46±0.08 vs 3-knob 0.40±0.17): a richer surface gives more paths to improvement,
so the governed loop is more robust, not less.

## 3. Real VoI>0 substrate — the standing program gate, answered NEGATIVELY (rigorously)
The CN L5 recorder is empty and intraday crypto VoI=0 (regime priced), so the strongest *available* real substrate is a
**belief-conditioned risk-mode on the real Dow-29 panel** (2010–2023): a K=2 self-supervised belief filter (trained
2010–2016, frozen) drives a 4-knob exposure policy; the anchor is static-full exposure (= EW buy-and-hold); the gate
uses **block-bootstrap CRN** paired improvement + deflated margin on DEV (2017–18) and a disjoint HOLDOUT (2019–20, incl.
COVID), with honest OOS (2021–23).

**Result: the governed loop certifies ZERO belief-mode edits, and this is CORRECT — not a proposer weakness.**
- The scripted loop refused all 40 proposals (38 no-dev-signal, 2 holdout-reject).
- The **pre-registered B3 config** {0.3, 0.7, 0.6, 0.3} — which "halved Dow drawdown" as a single path (holdout maxDD
  −0.167 vs −0.329) — **also fails to certify**: through the block-bootstrap gate its paired holdout risk-adj mean is
  **−0.0136 (negative under resampling)**. Its drawdown benefit is a **single-path artifact of the specific 2020 crash
  timing** that does not survive block resampling — exactly the failure the finance-safe firewall exists to catch (the
  same discipline as deflated Sharpe).
- And B3-config **fails honest OOS outright**: risk-adj −0.200 vs static −0.160, **Sharpe −0.27 vs +0.39**.

**Verdict on the standing VoI>0 gate:** on the best real substrate locally available, under proper block-bootstrap
rigor, there is **no certifiable belief-VoI — not alpha, and not even the drawdown-VoI B3 appeared to show**. This is
consistent with and *stronger than* B4 (the regime is priced): even the risk-mode's apparent benefit is single-path
luck. **CRYSTAL-1's real-market value stays transparency/control, not a certifiable edge — the firewall correctly
refuses to certify one.** The genuinely-untested refuges are unchanged: queue/rebate/latency microstructure, or CN
A-shares L5 once the recorder accumulates data.

## What Q2 establishes
- The governed loop scales to a **harder, 3× larger knob surface** unchanged, and is *more* robust there (lower variance).
- The scripted proposer's competence is **order-fragile (40±17%)**; the reasoning LLM proposer's 96% is largely a
  robustness win — motivating the coding-agent as the proposer.
- The firewall **works on real data**: it refuses to certify a plausible-looking, single-path risk-mode. The real VoI>0
  gate is answered negatively under rigor — the program's honest standing conclusion, now demonstrated end-to-end on the
  real Dow panel through the full HL governance stack.
