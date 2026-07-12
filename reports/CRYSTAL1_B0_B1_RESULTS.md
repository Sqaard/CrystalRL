# CRYSTAL-1 — B0 + B1 results (the first two blueprint stages, both PASSED)

> ⚠️ **B2 UPDATE (`reports/CRYSTAL1_B2_MULTISEED.md`):** the "uniqueness tracks belief fidelity" law claimed
> below is **RETRACTED** — multi-seed replication found Spearman(MAE, Rashomon) = 0.0 across 8 runs; the
> 4-point dose-response was seed coincidence (Rashomon-of-stance is dominated by PPO-seed idiosyncrasy; the
> privileged 0.02 was seed luck). The CORE B1 claims survived replication in every seed: the corner mechanism
> profile with a learned belief, and return parity with the privileged agent.

**Status: B0 complete (role-contract layer, 6/6 selftests). B1 complete (CRYSTAL-1 v3 passes ALL SIX battery
gates with a LEARNED belief filter) — plus a new measured design law: explanation uniqueness is a smooth
function of L1 belief fidelity.** Code: `src/crystal/{universe,belief_filter}.py`,
`interpretability/crystal1_b1*.py` (+ reports JSON).

## B0 — the role-contract layer (`src/crystal/universe.py`)
The portability fix the DNA audit demanded, now structural:
- **RoleAdapter fails LOUD** — a role bound to an absent column raises `RoleContractError` at build; the
  legacy silent absent→0 default (which blocked all investment on foreign universes) is impossible by
  construction. **RoleGate fails at CONSTRUCTION**, not mid-episode.
- **UniverseSpec is serializable** (JSON round-trip) — no runtime-registered maps; subprocess-safe by design.
- **Breadth as fractions of N** with deterministic half-up derivation (caught Python's banker's rounding:
  `round(14.5)=14` — config derivation must never surprise): 29→10/15, 344→115/172.
- **One contract, two worlds:** the same adapter binds the real csi500 panel (VIX/Regime_1_Prob/SP500_Trend/
  turbulence roles) and the synthetic polygon spec. 6/6 selftests.

## B1 — CRYSTAL-1 with a LEARNED belief (the first real test of L1)
The certified corner model received the env-computed (privileged) Bayes belief. CRYSTAL-1 must EARN it:

**L1 (`src/crystal/belief_filter.py`)** — a structured neural Bayes filter (learnable transition + emission;
the belief lives on the K-simplex; memory routes ONLY through it — the WH1 law by construction), trained
**self-supervised on raw observation streams (no regime labels)** by HMM filtering likelihood. Module gate:
- **Parameter recovery**: max|ΔT| = 0.012, max|ΔE| = 0.025 vs the true hidden world model — the filter
  literally learned a READABLE world model (you can print its stickiness and burst signatures).
- Belief MAE vs the analytic filter 0.024 (held-out); beats the memoryless baseline in held-out likelihood.

**L2**: PPO on `[learned_belief, t, inv, last_obs]` — the identical obs layout as the privileged corner model,
so the ONLY difference is the belief's source.

### The B1 battery (v3 = filter trained on 8000 episodes)
| gate | v3 result | verdict |
|---|---|---|
| high complexity | x = [0.92, 1.15] bits/action | ✅ |
| structured (phase-shuffle) | 2/2 | ✅ |
| reactive ≫ autoregressive | 0.851 vs 0.647 | ✅ |
| state-aware Rashomon crisp (median of 3 seeds) | **0.145** ≤ 0.3 | ✅ |
| belief-N7 asym (on the LEARNED stream) | **100.0 ASYMMETRIC** | ✅ |
| HC-1 ablation hurts | 6.90 → **−0.54** (total collapse) | ✅ |

**Competence:** return 6.90–7.10 across v2/v3 vs privileged 7.11 — **the price of learning your own beliefs
converged to ≈ 0.** The analytic optimum is 8.58 (the shared learning gap of both).

### The new design law (the discovery of B1): uniqueness tracks belief fidelity
The v1 run failed exactly one gate — state-aware Rashomon (0.395) — and the failure was REAL (stable across
3 seeds × N=200), not instrument noise. Dose-response across four L1 fidelity levels:

| L1 filter | belief MAE vs analytic | Rashomon ratio (median, 3 seeds) |
|---|---|---|
| privileged (analytic, MAE 0) | 0 | **0.02** |
| **v3** (8000 SSL episodes) | **0.0116** | **0.145** |
| v2 (2000 episodes) | 0.0132 | 0.305 |
| v1 (320 episodes) | 0.024 | 0.395 |

Monotone in both directions, converging to the privileged limit. **A noisier belief costs almost no return
(≈4% at v1, ≈0 at v2/v3) but costs explanation UNIQUENESS smoothly.** For the blueprint this is a measurable
lever: *investment in L1 fidelity buys interpretability-uniqueness, not just competence* — and the Rashomon
ε-curve doubles as an L1-quality meter. (We did NOT tune the 0.30 threshold to pass; v2 missed it by 0.005
and the answer was more data, i.e. better L1 — the honest route.)

## What B0+B1 establish for the blueprint
1. **L1 works as designed**: a structured filter learns the world model self-supervised, is readable
   (parameters comparable to truth), exhibits the filtering signature (belief-N7 asym), is load-bearing
   (HC-1 collapse), and supports the full corner profile — with return parity to the privileged agent.
2. **The battery works as a training-stage gate** (caught the v1 uniqueness deficit; diagnosed its cause).
3. **B0's contract held** throughout (polygon + real panel on one adapter).

## Honest limitations
- Single seed per PPO run (B2 multi-seed is the designed next stage); K=G=2 (the smallest case — the
  rotation family K>2 filters are the next L1 scale test); polygon-only (B3/B4 are the real-data stages);
  the SSL data is cheap here (env sampling) — on real data L1 fidelity will be data-limited, making the
  uniqueness law operationally important rather than academic.

## Next (per blueprint)
**B2** multi-seed replication (corner + family + B1) → **B3** risk-mode on real daily panels (belief-driven
drawdown budget over ~EW book) → **B4** the real execution-economics task (the program's main bet) → **B5**
the constructive turn (battery as training objective).
