"""HL-3 — the TEACHER BANK: memory-as-teachers, match-first-rank-second, with NEGATIVE teachers (the TB FINAL protocol).

Every gate outcome becomes a teacher:
  - ACCEPTED -> a POSITIVE teacher (this knob+direction improved the holdout; provenance-hashed).
  - REJECTED/REFUSED -> a NEGATIVE teacher = a COMPILED failure (knob+direction+reason) with a TRIGGER (the context that
    produced it), so the proposer treats it as a proposal-time oracle (the CHEF-style predictive trigger index), NOT as
    wreckage. A lone raw negative is never a gradient; it is a constraint the proposer must respect.
Selection is MATCH-FIRST (same knob = the hard match axis here), then RANK by holdout improvement among matches.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Teacher:
    """One compiled gate outcome as a teacher: a positive (accepted) or negative (rejected) knob+direction record
    with dev/hold/delta stats and provenance."""
    teacher_id: str
    kind: str                    # "positive" | "negative"
    knob: str
    direction: int               # sign of the edit (+1/-1/0)
    from_value: float
    to_value: float
    dev: float
    hold: float
    delta: float
    round: int
    reason: str
    provenance_hash: str


@dataclass
class TeacherBank:
    """Memory-as-teachers: match-first (same knob) then rank-by-holdout, with negative teachers as a
    proposal-time oracle."""
    teachers: list = field(default_factory=list)

    def _hash(self, *parts):
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:10]

    def record(self, dossier, accepted: bool):
        """Compile a gate outcome into a positive or negative teacher and append it to the bank."""
        knob, val, op = dossier.target, dossier.value, dossier.operator
        direction = int((1 if val > 0 else -1) if op == "DELTA" else 0)
        t = Teacher(
            teacher_id=self._hash(dossier.change_id, knob, val),
            kind="positive" if accepted else "negative",
            knob=knob, direction=direction, from_value=None, to_value=val,
            dev=dossier.dev_improvement, hold=dossier.holdout_improvement, delta=dossier.behavioral_delta,
            round=dossier.round, reason=dossier.verdict,
            provenance_hash=dossier.policy_hash)
        self.teachers.append(t)
        return t

    def matched(self, knob: str):
        """MATCH-FIRST: teachers on the same knob (the hard match axis on this substrate)."""
        return [t for t in self.teachers if t.knob == knob]

    def guidance(self, knob: str):
        """Aggregate teacher guidance for a knob: best positive direction (rank by holdout) + forbidden/penalized
        directions from negatives. Returns {prefer_direction, avoid_directions, n_pos, n_neg}."""
        ms = self.matched(knob)
        pos = [t for t in ms if t.kind == "positive"]
        neg = [t for t in ms if t.kind == "negative"]
        prefer = None
        if pos:
            best = max(pos, key=lambda t: t.hold)
            prefer = best.direction
        # a direction is penalized if a negative teacher for it exists AND no positive overrides it
        avoid = set(t.direction for t in neg if t.direction != 0)
        if prefer in avoid:
            avoid.discard(prefer)
        return {"prefer_direction": prefer, "avoid_directions": sorted(avoid), "n_pos": len(pos), "n_neg": len(neg)}
