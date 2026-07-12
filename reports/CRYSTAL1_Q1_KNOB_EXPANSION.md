# Q1 — how to expand CRYSTAL-1's knob/command surface (+ HL5 idea-coverage audit)

## The framing answer (why Stage0-9 is different for CRYSTAL-1)
[METHODOLOGY_MARKET_MECHANISM_GROUNDED_INTERPRETABLE_PPO.md](../METHODOLOGY_MARKET_MECHANISM_GROUNDED_INTERPRETABLE_PPO.md)
is the **R6c interpretability path**: train a black-box PPO, then *discover* primitives (Stage 1 VQ-VAE), *label* them
(Stage 3), and *test interventions* (Stage 5/5.5). It is "read the MRI after the fact." CRYSTAL-1 is built the opposite
way — **born legible** — so the Stage ladder collapses:
- **Stage 1 (discover primitives) is FREE** — the K-simplex belief IS the codebook; no VQ-VAE, no utilization/perplexity
  gates. **Stage 3 (label) is FREE** — the roles are named at construction (UniverseSpec / KNOWN_ROLES).
- **Stage 5 / 5.5 (interventions) is ALREADY BUILT** as the C-ladder: C-1 (causal write), C-2 (proved vs belief-MDP),
  C-3 (grounded lie-detector), C-5 (sign-epistasis), the governor (OOD gates = C3 envelope), the writ ladder.
- So **running Stage0-9 on CRYSTAL-1 mostly VALIDATES that the born-legible design short-circuits the discovery
  pipeline** — its value is a same-framework head-to-head vs R6c, not new controllability. The intervention-control
  question the methodology builds toward (Stage 5.5) is the one CRYSTAL-1 already answered on the polygon.

## Three concrete ways to EXPAND the surface (in priority order)
1. **Let the coding-agent GROW the surface (HLX operators `add_knob` / `add_rule`).** Today only `retune_knob`
   (SET/DELTA) is implemented — the agent can only tune existing knobs, not add new ones. Implementing add_knob/add_rule
   (each new knob is born as a typed registry row, tier T1+, gated) makes the surface *self-expanding under governance* —
   the single highest-leverage move, and the truest form of Heuristic Learning. (Demonstrated as feasible: the 6-knob
   `substrate_hard.py` is exactly the kind of surface add_knob would build.)
2. **GROW_K — the belief vocabulary dial.** K=2 today. K≥3 adds named belief coordinates (the experience corpus gives a
   ready 5-concept vocab: toxicity / liquidity / queue / inventory / time). Each new coordinate is a new SET_BELIEF
   command axis. Gated as a T1 retrain; the C*≈K law says legibility holds while G+2 ≤ K. (`c5_sign_epistasis.py`
   already runs on the K≥3 family_G4 belief.)
3. **Reward / constraint knobs as first-class levers.** The Stage-0.1 reward (λ_turnover, λ_drawdown, λ_concentration,
   λ_action_change) and constraints (governor envelope width, N_live≤K, D_cap, ABSTAIN_FLOOR) become proposer-editable
   levers — directly additive to the command surface, each with a guarantee post-condition. (`substrate_hard.py` adds 3
   such interacting levers and the loop handles them unchanged.)
- **Pretrain / BC**: already used — the M6 head is BC-warm-started from the MLP (the TB_gen_08 lesson); a stronger
  version is BC from R6c's behavior into CRYSTAL-1 (teacher transfer, a TB idea). Pretraining the belief filter on real
  data is the B3 path (done). BC is a *training* technique here, not a knob — it expands *competence*, not the surface.

## HL5 idea-coverage audit (what's built vs unused)
Independent audit of the 5 FINAL artifacts vs the code. **The command/certification SPINE is fully built** (K registry,
C0–C6 writ ladder, cumulative-authority ledger with cross-term, blast-radius gate, teacher bank, closed loop) — and the
code is **ahead of the artifacts on epistasis** (`c5_sign_epistasis.py` CONFIRMED sign-epistasis on ≥2 belief dims; the
artifacts still tag it [Plausible]). Highest-value **UNUSED** ideas, each a concrete expansion:

| Series | Unused / partial idea | What a minimal build adds |
|---|---|---|
| **HLX §3** | Open-endedness **ECOLOGY / PFSP anti-forgetting** + `add_knob`/`add_rule`/`recombine` operators (only `retune_knob` exists) | a frozen-past-selves archive the candidate must not regress against; self-expanding surface — the biggest missing HLX surface |
| **GV Art III/V** | **Staged exposure** shadow→canary→**dwell** + a seeded **known-bad canary** + multi-role independence (loop is single-process) | a dwell counter + a known-bad the gate must reject before promotion — hardens the gate against a clever proposer |
| **K rule 4** | **Model-checked single-owner arbitration / no-deadlock** over the lever graph (only `interaction_owner` strings) | a load-time verifier: each shared resource has one owner + a total-order selector (the artifact's own thinnest section) |
| **IV** | **Forbidden-pair registry** (now *licensed* by the confirmed sign-epistasis, but unbuilt) + anti-windup `Tt` | a forbidden-pair set consulted in `ledger.can_issue` — turns the measured sign-flip into an enforced constraint |
| **TB** | Teachers never **expire/recertify**; negatives lack a **trigger-index / causal diagnosis**; match on 1 axis not 4 | expiry-on-non-positive-recert + a CHEF-style trigger index → the teacher bank becomes a proposal-time oracle |

**Headline:** the spine is done; the unused ideas are the *growth* layer — self-expanding operators (add_knob/ECOLOGY),
a harder gate (dwell + known-bad canary), and turning the confirmed sign-epistasis into an enforced forbidden-pair
registry. See [CRYSTAL1_Q2_HARDER_MULTISEED_REALDOW.md](CRYSTAL1_Q2_HARDER_MULTISEED_REALDOW.md) for the executed Q2 work.
