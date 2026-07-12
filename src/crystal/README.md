# src/crystal — the CRYSTAL-1 core

The born-legible agent (paper §3, §6). Everything here is meant to be *read*, not just run.

| Module | Role | Paper |
|---|---|---|
| `belief_filter.py` | the NAMED memory — a Bayes filter over market regimes; `P(bear)` as the sole memory channel, writable | Eq. 2 |
| `soft_tree_policy.py` | the policy that IS the story — a jointly-trained Frosst–Hinton soft decision tree over [belief, inventory, time] | Eq. 3 |
| `universe.py` | the role-contract layer (the portability fix: assets bind by role, not by ticker index) | §3 |
| `governor.py` | runtime envelope enforcement for belief writes (the kNN manifold governor) | §8 |
| `writ_ladder.py` | the C0–C6 certified-write ladder + the per-principal signed ledger | §8 |

**Entry point for reading:** `belief_filter.py` → `soft_tree_policy.py`. These two are the whole
thesis. `governor.py` + `writ_ladder.py` are how a *write* to the belief gets certified before it
persists (the intervention/refusal machinery).
