"""HL-MEM v0.1 — hypothesis-extracting memory between attempts, around the UNCHANGED v12 gate.

Per reports/MEMORY_BETWEEN_ATTEMPTS_LIT_20260720.md (55-paper analysis, 5/5 verified).
Components (procedural skeleton; the LLM hypothesis-writer rides on top later — this layer is
what the preregistered 4-arm A/B measures):

  DeviationReporter  — deterministic gate-verdict -> frozen-vocabulary DEV record. Dwork/
                       Thresholdout hygiene: margins are TERCILE BANDS of z/z_crit, never raw z;
                       no window identities are disclosed (window-shopping guard).
  NegativeRegistry   — Weng's failed-directions, mechanical: (arm, direction) pairs falsified by
                       repeated far-miss rejections; consulted by the linter.
  ProposalLinter     — code-level exclusion BEFORE the gate: near-duplicates and falsified
                       directions are refused WITHOUT spending a gate query (Feedback-Friction's
                       best-measured lever). Refusals are ledgered and count toward N.
  CardStore          — (arm, direction, last_DEV_band) -> Beta(accepts, rejects) hit-rate
                       records; retrievable set = cards with >=1 resolution (AUDIT-gated);
                       FROZEN vs LIVE admission modes (the KT-D replication switch).
  PortfolioScheduler — islands over mechanism families; converged-quiet triggers migration,
                       not silence (FunSearch 4/140-seeds lesson).
  Deflation hook     — REPORTING ONLY: attempt count N (incl. linter refusals) is recorded for
                       Bailey-style threshold deflation, but the gate's bars are UNCHANGED —
                       gate modifications are shared core (business/SCOPE_AND_BRANCHES.md §3)
                       and require both tracks' agreement.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------- Deviation Reporter
FIRST_FAILED = ("dominated", "no_frontier_gain", "inert", "holdout_dsd", "holdout_ni",
                "holdout_return_z", "confirm_fail", "adversary_high_vol", "adversary_deep_dd",
                "tension", "wealth", "absurdity", "none")
BANDS = ("far_miss", "near_miss", "pass")


def _band(ratio):
    """Tercile-quantized z/z_crit ratio; never expose the raw statistic."""
    if ratio is None or not np.isfinite(ratio):
        return "far_miss"
    return "pass" if ratio >= 1.0 else ("near_miss" if ratio >= 0.5 else "far_miss")


def deviation_report(verdict, info):
    """Gate (verdict, info) -> frozen-vocabulary DEV record. Deterministic, no LLM."""
    if verdict.startswith("ACCEPTED"):
        return {"first_failed_bar": "none", "margin_band": "pass"}
    if verdict == "REJECTED_DOMINATED":
        return {"first_failed_bar": "dominated", "margin_band": "far_miss"}
    if verdict == "REJECTED_NO_FRONTIER_GAIN":
        return {"first_failed_bar": "no_frontier_gain", "margin_band": "far_miss"}
    if verdict == "REFUSED_INERT_ON_WINDOW":
        return {"first_failed_bar": "inert", "margin_band": "far_miss"}
    if verdict == "REFUSED_ABSURDITY_ALARM":
        return {"first_failed_bar": "absurdity", "margin_band": "far_miss"}
    if verdict == "REFUSED_ALPHA_WEALTH_EXHAUSTED":
        return {"first_failed_bar": "wealth", "margin_band": "far_miss"}
    if verdict == "REFUSED_TENSION_BUDGET":
        return {"first_failed_bar": "tension", "margin_band": "far_miss"}
    if verdict == "VETOED_BY_ADVERSARY":
        s = info.get("stressor", "high_vol")
        return {"first_failed_bar": f"adversary_{s}", "margin_band": "far_miss"}
    if verdict == "REJECTED_HOLDOUT":
        zc = info.get("z_crit") or 1.0
        if info.get("type") == "RETURN":
            return {"first_failed_bar": "holdout_return_z",
                    "margin_band": _band((info.get("z") or 0.0) / zc)}
        z_dsd, ni = info.get("z_dsd"), info.get("ni_z")
        if isinstance(info.get("confirm"), (dict, str)) and info.get("confirm") != "":
            # primary passed, confirm window failed
            if (z_dsd or 0) > zc and (ni or 0) > zc:
                return {"first_failed_bar": "confirm_fail", "margin_band": "near_miss"}
        if (z_dsd or 0) <= zc:
            return {"first_failed_bar": "holdout_dsd", "margin_band": _band((z_dsd or 0.0) / zc)}
        return {"first_failed_bar": "holdout_ni", "margin_band": _band((ni or 0.0) / zc)}
    return {"first_failed_bar": "dominated", "margin_band": "far_miss"}


# The mechanical repair map (arm C): DEV -> proposal-conditioning directives.
# Grounded in the gate's own semantics: dsd failure = defense too shallow; ni failure = defense
# too expensive (costs/return give-up); inert = the move never activates; dominated = too timid;
# adversary = too aggressive on stress slices; tension = too many off-anchor knobs.
REPAIR = {
    "dominated":         {"step_mult": 1.6, "prefer": None, "flip": False},
    "no_frontier_gain":  {"step_mult": 1.6, "prefer": None, "flip": False},
    "inert":             {"step_mult": 1.8, "prefer": ["lvl_defensive", "lvl_reduced"], "flip": False},
    "holdout_dsd":       {"step_mult": 1.0, "prefer": ["joint_defend", "lvl_defensive"], "flip": False},
    "holdout_ni":        {"step_mult": 0.7, "prefer": ["H", "t2"], "flip": True},
    "holdout_return_z":  {"step_mult": 0.7, "prefer": ["t2", "H"], "flip": False},
    "confirm_fail":      {"step_mult": 0.5, "prefer": None, "flip": False},
    "adversary_high_vol": {"step_mult": 0.5, "prefer": ["t1"], "flip": True},
    "adversary_deep_dd": {"step_mult": 0.5, "prefer": ["lvl_defensive"], "flip": False},
    "tension":           {"step_mult": 0.6, "prefer": None, "flip": False},
    "wealth":            {"step_mult": 1.0, "prefer": None, "flip": False},
    "absurdity":         {"step_mult": 0.4, "prefer": None, "flip": True},
    "none":              {"step_mult": 1.0, "prefer": None, "flip": False},
}


# ---------------------------------------------------------------- Negative registry + linter
class NegativeRegistry:
    """(arm, direction_sign) pairs falsified by >= k far-miss holdout rejections at the current
    step scale. Cleared for a pair when its step is re-expanded materially (a genuinely new try)."""
    def __init__(self, k=2):
        self.k = k
        self.counts = {}
        self.falsified = set()

    def record(self, arm, sign, dev_record):
        if dev_record["first_failed_bar"].startswith("holdout") and dev_record["margin_band"] == "far_miss":
            key = (arm, sign)
            self.counts[key] = self.counts.get(key, 0) + 1
            if self.counts[key] >= self.k:
                self.falsified.add(key)

    def clear(self, arm, sign):
        self.counts.pop((arm, sign), None)
        self.falsified.discard((arm, sign))

    def is_falsified(self, arm, sign):
        return (arm, sign) in self.falsified


class ProposalLinter:
    """Pre-gate exclusion: exact/near-duplicate configs and falsified directions are refused
    WITHOUT a gate query. Every refusal is ledgered (counts toward N)."""
    def __init__(self, registry, round_dp=3):
        self.registry = registry
        self.seen = set()
        self.round_dp = round_dp
        self.refusals = {"duplicate": 0, "falsified_direction": 0}

    def key(self, cand):
        return tuple(round(float(v), self.round_dp) for _, v in sorted(cand.items()))

    def check(self, arm, sign, cand):
        k = self.key(cand)
        if k in self.seen:
            self.refusals["duplicate"] += 1
            return "LINT_DUPLICATE"
        if arm != "joint_defend" and self.registry.is_falsified(arm, sign):
            self.refusals["falsified_direction"] += 1
            return "LINT_FALSIFIED_DIRECTION"
        return None

    def commit(self, cand):
        self.seen.add(self.key(cand))


# ---------------------------------------------------------------- Card store (arm D)
class CardStore:
    """HYP-card skeleton: key = (arm, direction_sign, last_DEV_first_failed_bar).
    value = Beta(accepts+1, rejects+1) hit-rate. AUDIT = the gate verdict resolves each card's
    implicit prediction ('this mechanism helps from this deviation state'). Retrieval = Thompson
    sampling over RESOLVED cards only. FROZEN mode stops updates after the warm-up (KT-D switch)."""
    def __init__(self, rng, frozen_after=None):
        self.rng = rng
        self.cards = {}
        self.frozen_after = frozen_after
        self.n_updates = 0

    def update(self, arm, sign, last_bar, accepted):
        if self.frozen_after is not None and self.n_updates >= self.frozen_after:
            return
        key = (arm, int(sign), last_bar)
        a, b = self.cards.get(key, (1.0, 1.0))
        self.cards[key] = (a + (1.0 if accepted else 0.0), b + (0.0 if accepted else 1.0))
        self.n_updates += 1

    def thompson_pick(self, arms, sign_of, last_bar):
        """Sample a hit-rate for each arm from its card (falling back to the bar-agnostic card,
        then to the uniform prior); return the argmax arm."""
        best, best_arm = -1.0, None
        for arm in arms:
            key = (arm, int(sign_of.get(arm, -1)), last_bar)
            a, b = self.cards.get(key, None) or self.cards.get((arm, int(sign_of.get(arm, -1)), "any"), (1.0, 1.0))
            draw = self.rng.beta(a, b)
            if draw > best:
                best, best_arm = draw, arm
        return best_arm


class PortfolioScheduler:
    """Islands over mechanism families; quiet (no accept in `patience` priced queries) rotates
    the active island instead of going silent."""
    ISLANDS = (("t1", "t2"), ("lvl_reduced", "lvl_defensive"), ("H", "joint_defend"))

    def __init__(self, patience=8):
        self.active = 0
        self.patience = patience
        self.quiet = 0

    def arms(self):
        return list(self.ISLANDS[self.active])

    def note(self, accepted):
        if accepted:
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet >= self.patience:
                self.active = (self.active + 1) % len(self.ISLANDS)
                self.quiet = 0
                return True                          # migrated
        return False
