"""HL v2 Teacher Bank — teachers EXPIRE (no tenure) and negatives carry a TRIGGER-INDEX (a CHEF-style proposal-time
oracle). Superset interface of the v1 bank: record(dossier, accepted, round) + matched(knob) + guidance(knob), plus
trigger(knob, direction) -> failure evidence and prune(round).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Teacher:
    """A v2 teacher (expires without recert): a knob+direction outcome carrying a failure trigger for negatives."""
    teacher_id: str
    knob: str
    direction: int
    accepted: bool
    born: int
    last_recert: int
    reason: str = ""
    trigger: tuple = ()          # (knob, direction) region that predicts this failure (negatives only)


class TeacherBankV2:
    """v2 teacher bank: teachers expire without recertification and negatives feed a (knob, direction) trigger
    index used as a proposal-time failure oracle."""
    def __init__(self, expiry_k: int = 8):
        self.teachers: list[Teacher] = []
        self.expiry_k = expiry_k
        self.trigger_index: dict[tuple, int] = {}    # (knob, sign) -> failure count (the oracle)
        self._n = 0

    def record(self, dossier, accepted: bool, round: int = 0):
        """Append a teacher for this outcome; on rejection also bump its (knob, direction) trigger count."""
        self._n += 1
        knob = dossier.target
        direction = 1 if dossier.value > 0 else -1
        t = Teacher(teacher_id=f"t{self._n}", knob=knob, direction=direction, accepted=accepted,
                    born=round, last_recert=round, reason=dossier.verdict)
        if not accepted:
            t.trigger = (knob, direction)
            self.trigger_index[t.trigger] = self.trigger_index.get(t.trigger, 0) + 1
        self.teachers.append(t)

    def prune(self, round: int):
        """Expiry: positive teachers not recertified within expiry_k rounds are retired (no tenure)."""
        keep = []
        for t in self.teachers:
            if t.accepted and (round - t.last_recert) > self.expiry_k:
                continue                                  # expired positive teacher
            keep.append(t)
        expired = len(self.teachers) - len(keep)
        self.teachers = keep
        return expired

    def matched(self, knob: str):
        """Return all teachers on the given knob."""
        return [t for t in self.teachers if t.knob == knob]

    def guidance(self, knob: str):
        """Aggregate a knob's guidance: last positive direction to prefer + rejected directions to avoid."""
        pos = [t for t in self.teachers if t.knob == knob and t.accepted]
        neg = [t for t in self.teachers if t.knob == knob and not t.accepted]
        prefer = pos[-1].direction if pos else 0
        avoid = {t.direction for t in neg}
        return {"prefer_direction": prefer, "avoid_directions": avoid}

    def recertify(self, knob: str, direction: int, round: int):
        """A repeated success refreshes the matching positive teacher's clock (delays expiry)."""
        for t in self.teachers:
            if t.accepted and t.knob == knob and t.direction == direction:
                t.last_recert = round

    def trigger(self, knob: str, direction: int):
        """Proposal-time oracle: how many times has (knob, direction) failed? High => the proposer should avoid it."""
        return self.trigger_index.get((knob, direction), 0)
