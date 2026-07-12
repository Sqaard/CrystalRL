# src/evaluation — the finance-safe firewall

The gate that decides whether an apparent win is real, and the shadow-P&L bookkeeping behind it.

| Module | Role |
|---|---|
| `firewall.py` | the finance-safe evaluation firewall: walk-forward / purged CV, deflated-Sharpe trial counting, matched & negative controls |
| `ghost_portfolios.py` | shadow P&L for repairs/interventions (measure a change's effect without deploying it) |

The firewall is proposer-blind by design: it does not care who (Ivan, Codex, an LLM loop) proposed
a change; it refuses with a reason. Trust it — it exists to catch your own mistakes.
