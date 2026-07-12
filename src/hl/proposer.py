"""HL-4 — the PROPOSER (v0): a teacher-guided coordinate search over the heuristic knobs.

This is the coding-agent's role, scripted for v0 (the loop ARCHITECTURE is the deliverable; v1 swaps in an LLM that
emits the same Proposal schema). It PROPOSES only — it NEVER assigns its own tier (it claims a blast-radius by step
size; the gate re-measures and auto-escalates laundering). It consults the Teacher Bank: prefer a positive teacher's
direction, AVOID directions compiled by negative teachers (the proposal-time oracle), shrink the step after a rejection.

The proposer does NOT know the optimum; it discovers it through the gate's accept/reject feedback mediated by teachers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hl.schema import Proposal  # noqa: E402
from src.hl import substrate as S  # noqa: E402


class Proposer:
    def __init__(self, seed: int = 0, sub=S):
        self.S = sub
        self.knobs = list(sub.KNOBS)
        rng = np.random.default_rng(seed)
        rng.shuffle(self.knobs)                         # seed-varied visitation order (for a multi-seed distribution)
        self.k_idx = 0
        # per-knob search state: current step (signed, seed-jittered), shrink on reject
        self.step = {k: float((0.15 * rng.uniform(0.8, 1.2)) * (sub.KNOBS[k]["range"][1] - sub.KNOBS[k]["range"][0]))
                     for k in self.knobs}
        self.direction = {k: -1 for k in self.knobs}   # start by trying to DECREASE (detuned defaults are too high)
        self._n = 0

    def _blast_claim(self, knob, step):
        span = self.S.KNOBS[knob]["range"][1] - self.S.KNOBS[knob]["range"][0]
        frac = abs(step) / span
        return "small" if frac < 0.1 else ("medium" if frac < 0.3 else "large")

    def propose(self, current_coeffs, teacher_bank, current_round):
        knob = self.knobs[self.k_idx % len(self.knobs)]
        self.k_idx += 1
        g = teacher_bank.guidance(knob)
        # teacher guidance: prefer a positive direction; flip away from an avoided direction
        if g["prefer_direction"] in (-1, 1):
            self.direction[knob] = g["prefer_direction"]
        elif self.direction[knob] in g["avoid_directions"]:
            self.direction[knob] = -self.direction[knob]
        step = self.step[knob] * self.direction[knob]
        if self.S.KNOBS[knob]["type"] == "int":
            step = float(int(round(step)) or self.direction[knob])   # at least +-1 for int knobs
        # CLIP so the resulting value stays inside the registry range (a well-behaved proposer respects bounds;
        # the gate still deny-by-defaults, but we don't spam it with out-of-range edits)
        cur = current_coeffs[knob]
        lo, hi = self.S.KNOBS[knob]["range"]
        step = float(np.clip(cur + step, lo, hi) - cur)
        if abs(step) < 1e-9:                                          # already at the bound in this direction -> flip
            self.direction[knob] = -self.direction[knob]
            step = float(np.clip(cur + self.step[knob] * self.direction[knob], lo, hi) - cur)
            if self.S.KNOBS[knob]["type"] == "int":
                step = float(int(round(step)))
        self._n += 1
        return Proposal(
            proposal_id=f"p{self._n}", target=knob, operator="DELTA", value=step,
            claimed_direction="increase_return",
            predicted_blast_radius=self._blast_claim(knob, step),
            teacher_provenance=[t.teacher_id for t in teacher_bank.matched(knob)],
            rollback_ref={knob: cur}, requested_eval_budget=200, expiry_round=current_round + 1)

    def on_result(self, knob, accepted):
        """After the gate: on reject, shrink the step and flip direction (explore); on accept, keep momentum."""
        if not accepted:
            self.step[knob] *= 0.6
            self.direction[knob] = -self.direction[knob]
        # on accept: leave step/direction (keep pushing the winning direction)
