# The next HL loop, and a meta-critique of the loop's ARCHITECTURE

Two parts: (1) the next HL loop run, read against the new CrystalScore-v2 tension profile — which surfaces the loop's
core defect; (2) an adversarial architecture critique (12 structural flaws, ranked, code-cited, one empirically
re-confirmed). Code: [hl_tension_blind.py](../interpretability/hl_tension_blind.py). Companion:
[CRYSTAL1_TRADEOFF_METRICS.md](CRYSTAL1_TRADEOFF_METRICS.md).

## 1. Next loop run → the defect it exposes: TENSION-BLINDNESS
Instead of another return-only loop (which the previous runs already showed closes 77–96% of the return gap), I ran the
loop against the **new tension axes**. On a modular G=10 substrate, growing coverage one venue at a time — every step
**gate-ACCEPTED** because return rises — I tracked what the gate sees (return) vs what it does not (MDL parsimony-fidelity
deficit, Axis A):

| venues | RETURN (gated) | MDL deficit (BLIND) |
|---|---|---|
| 1 | +0.86 | −0.08 |
| 4 | +2.59 | 0.16 |
| 7 | +4.56 | 0.35 |
| 10 | +6.09 | **0.41** |

**The loop certifies a monotone return climb 0.86→6.09 while silently riding the legibility-loss frontier 0→0.41.**
Every step passes the gate; none is refused for the legibility it spends. This is not a tuning miss — the gate has **no
tension axis as a state variable**, so it *cannot* refuse a return gain that pays in legibility. The saturated scalar
CrystalScore (0.938) hid this because it lived at the degenerate low-complexity corner; the tension profile makes it
visible.

## 2. Architecture critique — 12 structural flaws (ranked)

**🔴 Blocking (3, one root cause):**
- **F1 Tension-blindness / single-objective gate** — accept predicate is return-only (`gate.py:130,142`); no tradeoff
  axis is a gate state variable. Evidence: the table above. Fix: **vector-valued gate** — admit only Pareto-non-dominated
  moves over `(return, −MDL_deficit, −ESW_leak, −DES_AUC, ledger_D)`; store the v2 profile in the dossier; deny on any
  axis regression.
- **F4 Governance rations process, not objective** — the writ ladder + authority ledger debit on behavioral-delta
  *magnitude* (`alpha=max(delta,0.05)`), never on *what tradeoff the move buys*; a small-delta legibility-destroying edit
  spends almost no authority. Fix: **price authority in tension units** (α includes the MDL/ESW cost); add a C-rung
  certifying "no uncertified tension-axis regression."
- **F10 Objective is return by definition** — the loop's success metric is `gap_closed` on return (`hl_first_loop.py:99`);
  legibility never enters the loss it optimizes exactly what F1 says it can't see. Fix: redefine the objective as a point
  on the CrystalScore-v2 **frontier** (max return s.t. MDL-deficit ≤ budget).

**🟠 Serious (6):**
- **F2 Gate-signal correlation (roles don't bind)** — dev/holdout/validator/canary sample the same stationary DGP →
  near-collinear. **Empirically re-confirmed this run: corr(dev,holdout)=0.988, corr(dev,valid)=0.988, sign-agreement
  19/19, ZERO dev-pass/holdout-fail cases.** v3's validator fired only via a hand-injected shifted regime; canary/dwell/
  ecology bound 0×. Fix: roles must sample **structurally disjoint stressors** (adversarial / worst-window / off-manifold
  / shifted-regime); make the validator an *adversary* that searches for the candidate's failure regime.
- **F3 Goodharting the frozen holdout at α_machine** — fixed finite seed set, deflation only √(1+ln trials); a machine
  proposer's query rate outruns the log penalty. Fix: rotate/partition holdout per epoch; α-investing / e-value
  sequential budget priced at machine query rate.
- **F5 Joint-move blindness** — one-knob-per-round proposer + dev>0-on-single-edit gate → any two-lever-together
  improvement is unreachable (provide+unwind); recombine 0/0 on non-modular tasks. Fix: structured joint-move operator +
  atomic bundle scoring.
- **F6 Proposer order-sensitivity / myopia** — greedy coordinate descent; reachable optimum depends on knob order+seed.
  Fix: population/archive search + restarts; report a distribution, not one number.
- **F7 Archive descriptor is coverage-only** — MAP-Elites niches keyed by covered-venue set (a return-serving axis);
  legibility is not a niche coordinate. Fix: add tension coordinates (MDL-deficit / h_μ bins) to the behavior descriptor.
- **F9 No cross-substrate teacher transfer** — teachers keyed on raw `(knob,direction)`; nothing transfers to R6c / a new
  surface. Fix: key on mechanism-class / abstract feature; carry a portable prior.

**🟡 Minor (3):** F8 operator budget attributed only post-hoc by ablation (fix: online bandit over operators); F11
anti-salami measures motion magnitude not harm direction (fix: hazard-weighted Σδ); F12 known-bad canary is one
hardcoded failure — security theater against a named attacker (fix: generate canaries adversarially from the policy's
weakest measured axis).

## Headline — the deepest limitation and the one redesign
**The loop is a single-objective (return) optimizer wearing a governance costume:** every guard — writ ladder, authority
ledger, anti-salami, multi-role staging — rations the *process* (how much / how fast the policy moves) while leaving the
*objective* one-dimensional; and on a stationary same-DGP substrate the roles meant to add challenge are collinear
(r≈0.99, 0 binding catches), so they are near-inert decoration. **The single redesign that fixes the most (F1, F4, F7,
F10 directly; F2, F12 indirectly): make the gate and the authority ledger operate over the CrystalScore-v2 tension
VECTOR instead of scalar return** — admit only Pareto-non-dominated moves, price authority in tension units, bin the QD
archive by a tension coordinate. This converts the loop from "maximize return under process limits" into "walk the
return × legibility frontier under a certified no-uncertified-regression rule" — which is exactly what the "no perfect
model exists" framing demands, now made operational.
