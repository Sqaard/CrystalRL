# CRYSTAL-1 B2 — multi-seed replication: core claims IRONCLAD, one fresh claim RETRACTED

**B2 did exactly what it exists for: it killed an n=1 artifact — one WE minted in B1 the day before.**
12 training jobs (crystal-v3 ×3 seeds, privileged ×3, crystal-v1 ×2, family G∈{4,12} ×2), fixed eval
protocol, training-pipeline seeds varied. `interpretability/b2_multiseed.py` + `b2_multiseed_results.jsonl`
+ `b2_multiseed_report.json`.

## Verdicts by claim

| Claim | Result | Verdict |
|---|---|---|
| **C1 corner profile with LEARNED belief** | The 5 MECHANISM axes (complexity, structure, reactive≫autoreg, belief-N7 asym, HC-1 ablation) pass **3/3 seeds**: react 0.909±0.042 ≫ auto 0.632±0.119; x>0.6+structure all seeds; N7 asym all; ablation collapse all. The 6th axis (Rashomon) noisy (see C3). | ✅ **REPLICATED (mechanism)** |
| **C2 privileged reference** | Same 4 mechanism axes pass **3/3** (react 0.904±0.016). Rashomon gate fails **0/3** (0.17–0.81 across rng-seeds!) — the original seed-0 crispness 0.02 was **seed luck**. | ✅ mechanism / ⚠️ Rashomon exposed |
| **C3 uniqueness-tracks-fidelity law (B1's "discovery")** | **Spearman(filter MAE, Rashomon) = 0.0 across all 8 runs.** v1 seeds got MAE 0.009–0.013 (320 episodes sometimes suffice — the SSL-size lever was weaker than assumed) and Rashomon scattered 0.07–0.47; a v3 seed hit 0.676 at MAE 0.008. The B1 4-point dose-response was **seed coincidence**. | ❌ **RETRACTED** |
| **C4 C\*≈K learned bend** | gap(G=4) = [0.069, 0.077, 0.088] vs gap(G=12) = [0.314, 0.33, 0.356] — **complete separation, every seed**. | ✅ **REPLICATED (now multi-seed ironclad)** |
| **C5 price of learned beliefs ≈ 0** | crystal [6.88–7.23] vs privileged [7.02–7.23] — parity in every seed. | ✅ **REPLICATED** |

## The honest re-readings
1. **The B1 core claim STANDS, stronger than before:** CRYSTAL-1's learned-belief agent reproduces the corner
   MECHANISM profile in every seed, at return parity with the privileged agent. L1 (self-supervised structured
   filter) is validated as the architecture's foundation.
2. **The B1 "new design law" is RETRACTED.** Uniqueness does not track belief fidelity in this env at these
   fidelity levels; the Rashomon-of-stance metric is dominated by **PPO-seed idiosyncrasy** (which stance
   program a particular run happens to converge to), with within-instrument spread up to [0.17, 0.81] on one
   policy. Both the v1 "failure" and the privileged "crispness" that anchored the law were seed artifacts.
   (Memory + B0/B1 report corrected.)
3. **Instrument re-scope (Rashomon):** reliable at the EXTREMES — the churner-vs-persister gross separation
   stands (canonical P22 folds 0.04–0.10 on fixed behavior logs vs csi500-churner 0.79–1.00; measured on
   4 independent folds each, far outside the noise band) — but **fine gradations (0.07–0.7) on small-env PPO
   policies are uninformative**. Demoted from hard per-seed gate to a distributional diagnostic (report the
   across-seed range; gate only gross separations). The corner acceptance battery is therefore **5 mechanism
   axes + Rashomon-as-diagnostic**, not 6 hard gates.
4. **A cheerful C5 note:** learning your own beliefs costs nothing in return, in every seed — the strongest
   version of the B1 competence result.

## Battery-gate set after B2 (the operational change)
Hard per-policy gates: high-complexity, structure-vs-shuffle, reactive≫autoregressive, belief-N7 asymmetry,
HC-1 ablation-hurts. Diagnostics (reported, not gated per-seed): Rashomon ε-curve (extremes only),
simulatability@K trend, J intervention fidelity.

## Next (per blueprint)
**B3** — risk-mode on real daily panels (belief-driven drawdown budget over ~EW book; DD/Calmar objective,
Sharpe-no-degrade, book-state-conditional, frozen-window gates). Then **B4** (the real execution-economics
task — the program's main bet) and **B5** (the constructive turn).
