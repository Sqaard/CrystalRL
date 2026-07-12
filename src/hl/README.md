# src/hl — the legibility-loop library

The reusable machinery of the Heuristic-Learning loop: rule-policy substrates, the gate, and the
tension/Pareto layer. **This is the loop's LIBRARY, not the coding agent** — the LLM proposer and
the autonomous loops live in the mothership; this package is the substrate they act on and the
gate that judges them, so the certified experiments here are self-contained and reproducible.

| Group | Modules | Role |
|---|---|---|
| Substrates | `substrate.py`, `substrate_hard.py` (G12), `substrate_v2.py`, `polygon_substrate.py` | rule-based policies on the Series-G polygon (the harder ones make legibility *not* free) |
| Rule policies | `rule_policy.py`, `modular_rule_policy.py` | conjunctive / per-venue rule-list policies the agent grows structurally |
| The gate | `gate.py`, `staged_gate.py`, `pareto_gate.py`, `tension.py` | deny-by-default proposer-blind gate; the return↔legibility tension vector |
| Registry / proposer | `registry_v2.py`, `schema.py`, `proposer.py`, `mechanism_bandit.py` | the growable knob registry, the HLX proposal schema, the teacher-guided search |
| Teachers | `teacher_bank.py`, `teacher_bank_v2.py` | memory-as-teachers (match-first-rank-second; v2 teachers expire) |
| Adapter | `r6c_tension_adapter.py` | real HCS policy-forward daily loop for the R6c side |

Note: `registry.py` is a superseded stub — the real registry is `registry_v2.py`.

**Entry point:** `substrate_hard.py` (the G12 substrate where the certified return↔legibility
tension lives) + `pareto_gate.py`/`tension.py` (how that tension is measured and gated).
