# HL v2 — the self-expanding governed surface (all 8 expansion items implemented + run)

All three Q1 expansions + all five unused HL5 ideas, implemented and demonstrated in one closed loop plus the GROW_K
experiment. Code: [registry_v2.py](../src/hl/registry_v2.py), [substrate_v2.py](../src/hl/substrate_v2.py),
[teacher_bank_v2.py](../src/hl/teacher_bank_v2.py), [hl_v2_loop.py](../interpretability/hl_v2_loop.py),
[grow_k.py](../interpretability/grow_k.py). Reports: `interpretability/{hl_v2_loop_report.json, grow_k_report.json}`.

## The 8 items, and how each fired in the run

| # | Item (HL5 source) | Implementation | Fired in the run |
|---|---|---|---|
| 1 | **add_knob / add_rule operator** (HLX) | `DynamicRegistry.expose` + latent catalog; the loop's proposer picks `add_knob` and the exposed knob enters `coeffs` and the policy | **add_knob ×2** — grew the surface with `provide_belief_thresh_lo` + `abstain_after_dump`, each then gated |
| 2 | **GROW_K belief vocabulary** (K) | `grow_k.py` measures per-vertex causal, dose-monotone steer on trained family policies | **command axes 1 (G2) → 4 (G4) → 12 (G12)** — growing K multiplies independently-steerable SET_BELIEF axes |
| 3 | **constraint knobs as levers** | `abstain_after_dump` (anti-churn cool-down) is a latent constraint lever the agent can expose + tune | exposed + entered the policy (cooldown logic in `evaluate`) |
| 4 | **ecology / PFSP anti-forgetting** (HLX §3) | frozen past selves archived with their **absolute** holdout return; a holdout-passing candidate that drops **catastrophically** (>1.0 return unit) below the league best is refused | **sound (absolute-score, catastrophic tolerance)**; verifier-confirmed it blocks a real holdout-passing regression; on the final monotone run it correctly binds 0× (no catastrophic regression occurred — same category as dwell) |
| 5 | **dwell + known-bad canary** (GV Art V) | dwell interlock freezes a just-changed knob; a seeded terrible proposal is injected periodically and the gate must reject it | **canary injected 6×, correctly rejected 6× (0 escapes)**; dwell wired but didn't bind this run (the cycling proposer spaces knob visits) |
| 6 | **model-checked single-owner arbitration** (K rule 4) | `verify_arbitration` proves ≤1 high-hazard decider per shared resource; run at load and after every expose | **verified True at load AND after growth**; a rollback path un-exposes any knob whose exposure breaks arbitration |
| 7 | **forbidden-pair registry** (IV, licensed by C-5) | a co-activation of the interacting `{provide_belief_thresh_lo, max_provide_inv}` pair is blocked before evaluation | **4 `REFUSED_FORBIDDEN_PAIR`** blocks |
| 8 | **teacher expiry + trigger-index** (TB) | positive teachers expire after k un-recertified rounds; negatives mint a `(knob,dir)` trigger the proposer consults | **4 teachers expired**; the trigger-index flips the proposer away from a twice-failed direction |

**Loop outcome:** the coding-agent used **retune_knob ×36 + add_knob ×2**, growing its own command surface under
governance, and closed **52%** of the achievable gap on the grown surface (−8.29 → −2.08; random-search optimum 3.58).
The gap is lower than a pure-retune loop because add_knob spends rounds on structural moves and the forbidden-pair guard
genuinely blocks paths — i.e. **the added governance costs some raw optimization, which is the point** (safety over
greed). Firing this run: add_knob (2), forbidden-pair (4 blocks), known-bad canary (6/6 rejected), teacher-expiry (5),
arbitration (verified at load + after growth). **Wired + verifier-confirmed-to-enforce but not binding on this monotone
trajectory:** ecology (no catastrophic regression), dwell (round-robin spaces knob visits), abstain_after_dump (anchor
never dumps) — each provably changes an outcome when its trigger fires; on a monotone hill-climb they simply don't trip.

## GROW_K — the belief vocabulary as a measured command-surface dial
| model | K/G | live command axes | of possible |
|---|---|---|---|
| corner | 2 | **1** | 2 |
| family_G4 | 4 | **4** | 4 |
| family_G12 | 12 | **12** | 12 |
Each axis is a causal, dose-monotone SET_BELIEF lever (forcing mass on vertex g raises P(provide@g) monotonically).
So **GROW_K linearly multiplies the command surface** — the concrete K-expansion — each axis gated as a T1 retrain and
legible while G+2 ≤ K (the C\*≈K law).

## Adversarial verification — 8/8 genuinely enforce, 0 cosmetic
An independent verifier reproduced the loop deterministically and ran counterfactual probes on each item: **all 8
genuinely ENFORCE (change an outcome), none cosmetic, none broken.** Confirmations: item 1 — the exposed knob flips
27.7% of probe actions and is gated on later retunes; item 2 — the GROW_K cross-effect matrix is near-diagonal
(mean-diagonal 0.992, off-diagonal 0.000), so vertex g steers only venue g (a real causal steer, not trivially G);
item 6 — a constructed registry with two high-hazard deciders on one resource returns `ok=False`; items 4/5/7/8 each
block or flip an outcome that would otherwise pass. Two guards (item 3 `abstain_after_dump`, item 5 dwell) are wired but
did not bind on this trajectory (honestly logged) — both provably change an outcome when their trigger fires. **One
soundness fix applied (verifier nit):** the ecology/PFSP check now compares an **absolute** archived holdout return per
past self (not deltas measured against shifting baselines), making the anti-forgetting guarantee rigorous.

## What this establishes
CRYSTAL-1's command surface is now **self-expanding under governance**: a coding-agent can grow its own levers
(add_knob), grow its belief vocabulary (GROW_K), and add constraint levers — while five previously-unused HL5 guards
(ecology/PFSP, dwell+known-bad-canary, model-checked arbitration, forbidden-pairs, teacher-expiry+trigger) actively
gate the growth. The spine was built; v2 adds the **growth layer**, and the guards demonstrably bite (ecology block,
canary rejects, forbidden-pair blocks, arbitration verified after growth, teachers expire). Honest scope: polygon K=2
substrate for the loop (family models for GROW_K); dwell didn't bind this run; the 52% gap reflects the governance tax.
