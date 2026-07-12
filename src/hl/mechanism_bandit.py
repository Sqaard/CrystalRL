"""HL v4 — an online UCB bandit over OPERATOR MECHANISM-CLASSES. Fixes:
  F8 budget attribution : credit is assigned ONLINE (reward = a frontier-expanding accept per query), so round budget
                          flows to the operator that is load-bearing FOR THIS task, instead of a hardcoded schedule.
  F9 cross-substrate     : arms are keyed on the MECHANISM CLASS (operator name), NOT a raw knob id, so the learned
                          priors (arm stats) are PORTABLE — a new substrate can be warm-started from a prior bandit.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field


@dataclass
class MechanismBandit:
    arms: list
    counts: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)
    t: int = 0

    def __post_init__(self):
        for a in self.arms:
            self.counts.setdefault(a, 0); self.reward.setdefault(a, 0.0)

    def select(self, available):
        self.t += 1
        # try each available arm once, then UCB1
        for a in available:
            if self.counts[a] == 0:
                return a
        return max(available, key=lambda a: self.reward[a] / self.counts[a] + math.sqrt(2 * math.log(self.t) / self.counts[a]))

    def update(self, arm, reward):
        self.counts[arm] += 1; self.reward[arm] += float(reward)

    def prior(self):
        """Portable prior (F9): per-mechanism success rate, transferable to a new substrate's bandit."""
        return {a: (round(self.reward[a] / self.counts[a], 3) if self.counts[a] else None) for a in self.arms}

    def seed_from(self, prior, pseudo=2.0):
        """Warm-start a new substrate from a prior mechanism success table (F9 transfer)."""
        for a, sr in prior.items():
            if a in self.counts and sr is not None:
                self.counts[a] += pseudo; self.reward[a] += pseudo * float(sr)
