# PREREGISTRATION — BH1-Q2 arm A1 (transition oversampling), registered 2026-08-18, BEFORE any A1 run

Registered before any `exp_bh1_q2_a1.py` code exists. This file locks the design so a null
cannot be "rescued" by a post-hoc metric swap, a new W, or extra seeds, and a positive cannot be
over-read as "cold RL finds belief-use on the original market."

## What stages 1–5 established (the baseline for A1)

- **Stage 1** (`exp_bh1_pressure.py`): fidelity 0.00 at EVERY contrast, including c=2 where the
  regimes differ by 2σ and the belief is a near-perfect Bayes filter. Every head collapsed to a
  constant dial — the E-27 cold-PPO signature.
- **Stage 2** (`exp_bh1_pressure2`): the designed pressure is REAL (oracle edge over the best
  dial at c=2) and belief-use SURVIVES when a teacher installs it (warm-start fidelity high),
  but cold PPO cannot discover it even at 90k / high-entropy. **Discovery, not pressure, is
  the binding constraint.**
- **Stage 3** (`exp_bh1_pressure3`): the strict fixed-delta E-28 probe is PROVABLY BLIND on
  saturated-belief threshold policies (BC oracle acc 0.97 scores 0.00 on strict, 0.85 on
  contrast-write; cold control 0.00 on both). Fidelity must be measured with the
  **contrast-write** probe, not the strict probe.
- **Stage 4** (`exp_bh1_pressure4`): retention rises with contrast under warm-start, but the
  c=0 teacher itself cannot encode the rule (ceiling 0.00), so "erosion under no pressure" was
  never actually tested.
- **Stage 5** (`exp_bh1_pressure5`): the transplant test — a c=2 teacher fine-tuned on c=0
  stays at 0.92. **INERT OPTIMIZER.** The honest BH1 claim is INSTALL-ONLY: pressure governs
  what a teacher can encode; 30k PPO fine-tune is behaviorally conservative either way.

**The open question A1 attacks** (ONBOARDING, BH1): *what makes cold RL **discover**
belief-use — curriculum, transition oversampling, meta-pressure?* Stages 1–5 left
discovery as the binding constraint but never tested a mechanism. A1 is the cheapest of the
three candidate mechanisms: a **sampling** change, not an architecture change.

## Design (all fixed now, none tuned after results)

1. **Substrate.** Designed 2-regime Markov market from `exp_bh1_pressure.gen_market`,
  **c = 2.0 only** — the only contrast where stage 1's kill has teeth and the oracle beats
  every constant dial. Constants unchanged: `SIGMA=0.012`, `P_STAY=0.97`, `RF_D=0.02/252`,
  `N_DAYS=2500`. The belief fed to the agent is the TRUE Bayes-filter posterior (the
  agent's belief input is not changed by A1).
2. **Intervention (sampling only).** Build the TRAIN stream by oversampling
   regime-switch neighborhoods: concatenate `±W` windows around every switch (with
   replacement) until the stream length is `N_DAYS=2500`. **W = 5.** Target: **≥ 50%** of
   train days sit within W of a switch (the natural rate is ~3%, since `P_STAY=0.97` ⇒
   ~3% of timesteps are switches). The EVAL stream is the unaltered `gen_market(c, seed+500)`
   in both arms — A1 is never evaluated on the oversampled distribution.
3. **Architecture.** Unchanged `train()` from `exp_bh1_pressure2` — SoftTree PPO,
   `device="cpu"`, `n_steps=2048`, `batch_size=256`, `n_epochs=6`, `learning_rate=3e-4`,
   `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.005`, `tree_depth=3`,
   `beta=1.0`, `critic_arch=(32,32)`. **Cold start** (`init_from=None`) in both arms. 30k
   steps. No warm-start in A1 — that is the point.
4. **Arms.** Two arms, same seeds: **cold** (train on the original `gen_market` stream,
   no oversampling) vs **A1** (train on the oversampled stream). Both evaluated on the
   original eval stream.
5. **Seeds.** 0, 1, 2. Stages 1–5 used seeds 0–1; **seed 2 is the held-out seed** and is
   part of the pre-registered set, not an add-on.

## Measures (all pre-registered)

- **Primary outcome:** `contrast_write_probe` from `exp_bh1_pressure3` (60 probes,
  seed 28). **NOT** `fidelity_probe` — paper §6 / stage 3 establish the strict fixed-delta
  probe is blind on saturated-belief threshold policies.
- **Ceiling:** the BC-oracle contrast-write fidelity **on this machine = 0.833**
  (locally reproduced 2026-08-17; the committed artifact reads 0.85, a platform difference,
  not a bug). A1's threshold is anchored to **0.833**, never 0.85.
- **Secondary (not used to rescue the primary):** behavioural regime gap
  (`behavioral_voi`: mean exposure in bull vs bear true states on a fresh rollout); return
  vs oracle (`run_policy` annualised return / maxDD / mean shaped reward).

## Preregistered verdict (locked thresholds, applied in code from the JSON)

- **SUPPORTED:** mean(F_A1) − mean(F_cold) **≥ 0.40** **AND** F_A1 **≥ 0.50 in ≥ 2 of 3
  seeds. Read: transition rarity is (part of) why cold RL fails to discover belief-use, and
  oversampling switches lets cold PPO find it.
- **NULL / KILL:** all three A1 seeds **F ≤ 0.20**. Read: transition rarity is NOT
  why cold RL fails — the binding constraint is elsewhere (architecture, optimization,
  or the credit-assignment geometry), and no sampling fix will rescue it. A null is a
  result and is logged as such.
- **INCONCLUSIVE:** anything between (e.g. one seed clears 0.50 but the mean gap is
  < 0.40). **No extra seeds, no new W, no post-hoc metric swap.** Re-run only as a new
  pre-registered experiment.

## Pre-logged honest caveats (state now, before the run)

- **Distribution shift.** Oversampling changes the TRAIN distribution. A SUPPORTED
  result means "discovery is possible when switches are common," **not** "cold RL finds
  belief-use on the original market." The eval stream is unaltered, so the fidelity is
  measured on the original distribution — but the policy was trained on a different one.
- **One mechanism, one contrast.** A1 tests transition oversampling at c=2 only. It does
  not test curriculum (A2) or meta-pressure (A3), and says nothing about other contrasts.
- **Laptop-scale, 3 seeds.** 30k steps × 3 seeds × 2 arms is a laptop run; it is not a
  power calculation. The ≥2-of-3-seeds rule is a discovery bar, not a significance test.
- **Cold baseline expected ≈ 0.00.** Stage 3 established the cold head scores 0.00 on
  contrast-write. If the cold arm here does not reproduce ~0.00, the run is INCONCLUSIVE
  (the baseline moved) and is reported as such, not as a positive A1.

## Not in scope for A1

- Arms A2 (curriculum) and A3 (meta-pressure) — separate pre-registrations if A1 lands.
- The 1M-step erosion ladder (BH1-Q1) — needs a confirmed compute allocation.
- Any change to `exp_bh1_pressure*.py`, `gen_market`, `train`, or `contrast_write_probe`.
  A1 reuses them unchanged.
- Any claim about the real panel. A1 is a designed-market instrument calibration,
  VoI-fenced, never market evidence.
