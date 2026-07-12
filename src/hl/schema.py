"""HL-1 — the HLX PROPOSAL schema + the GV change dossier (the data objects of the heuristic-learning loop).

Design laws honored (HL5 FINALs):
  - DENY-BY-DEFAULT: a proposal missing any required field is INVALID (no silent defaults; the hcs_metadata law).
  - The proposer NEVER assigns its own tier: `predicted_blast_radius` is a CLAIM the gate re-measures; the dossier
    stores both `tier_claimed` and `tier_assigned` so laundering (small-looking edit, big delta) is auditable.
  - Every change carries a ROLLBACK ref executable without the changed system (the AF447 anti-pattern).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

REQUIRED = ("proposal_id", "target", "operator", "value", "claimed_direction", "predicted_blast_radius",
            "teacher_provenance", "rollback_ref", "requested_eval_budget", "expiry_round")


@dataclass
class Proposal:
    proposal_id: str
    target: str                      # knob name (must exist in the registry)
    operator: str                    # "SET" (absolute) | "DELTA" (relative step)
    value: float
    claimed_direction: str           # what the proposer claims the edit does (e.g. "increase_return")
    predicted_blast_radius: str      # CLAIM: "small" | "medium" | "large" — the gate re-measures
    teacher_provenance: list         # teacher ids consulted (empty list allowed but must be present)
    rollback_ref: dict               # {knob: previous value} — executable rollback without the changed system
    requested_eval_budget: int       # dev-eval episodes requested
    expiry_round: int                # proposal invalid after this loop round
    proposer_id: str = "proposer_v0"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def validate(p: Proposal, registry: dict, current_round: int):
    """Deny-by-default validation. Returns (ok: bool, reason: str)."""
    d = asdict(p)
    for f_ in REQUIRED:
        if d.get(f_) is None:
            return False, f"DENY missing field {f_}"
    if p.target not in registry:
        return False, f"DENY unknown knob {p.target} (not in registry — no hidden write path, C0)"
    spec = registry[p.target]
    lo, hi = spec["range"]
    v = p.value if p.operator == "SET" else spec.get("_current", spec["default"]) + p.value
    if not (lo <= v <= hi):
        return False, f"DENY value {v} outside registry range [{lo},{hi}] (C2 dose window)"
    if spec.get("type") == "int" and abs(v - round(v)) > 1e-9:
        return False, f"DENY non-integer value for int knob {p.target}"
    if p.expiry_round < current_round:
        return False, "DENY expired proposal"
    if p.predicted_blast_radius not in ("small", "medium", "large"):
        return False, "DENY malformed blast-radius claim"
    return True, "ok"


@dataclass
class ChangeDossier:
    """The immutable audit record per proposal (the TB_00 dossier, minimal fields for the polygon loop)."""
    change_id: str
    parent_change_id: str
    proposer_id: str
    round: int
    target: str
    operator: str
    value: float
    tier_claimed: str
    tier_assigned: str               # measured by the gate, never by the proposer
    behavioral_delta: float          # measured fraction of frozen probes whose action changed
    capital_scope_delta: float
    dev_improvement: float
    holdout_improvement: float
    eval_queries_charged: int
    verdict: str                     # ACCEPTED | REJECTED_* | REFUSED_*
    writ_status: str
    rollback_ref: dict
    registry_version: str
    policy_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def coeffs_hash(coeffs: dict) -> str:
    return hashlib.sha256(json.dumps(coeffs, sort_keys=True).encode()).hexdigest()[:12]
