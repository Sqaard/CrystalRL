"""HL v4 — the PARETO / TENSION-VECTOR gate. Fixes the gate-side architecture flaws:

  F1  vector gate      : admit only Pareto-NON-DOMINATED moves over (return, -description_len); no scalar return.
  F10 objective        : the accepted set IS a frontier (kept as self.frontier), reported as such.
  F3  Goodhart holdout : ROTATING holdout slice per query + ALPHA-INVESTING wealth priced at query rate (not sqrt-ln):
                         the significance level shrinks as wealth depletes, so a machine query storm cannot mine it.
  F2  role collinearity: the validator is an ADVERSARY — it searches a set of DISJOINT STRESSOR regimes for one where
                         the candidate's return advantage REVERSES, and vetoes if found (not a second i.i.d. holdout).
  F4  authority units  : the authority ledger is priced in TENSION-HARM (how much legibility a move spends), not raw
                         behavioral-delta magnitude; deny once the tension budget is exhausted.
  F11 anti-salami      : the cumulative budget accrues TENSION-HARM (hazard-weighted), not raw probe-flip motion.
  F12 adversarial canary: a canary is GENERATED from the policy's current weakest axis (add a legibility-only-cost rule
                         and a provide-in-toxic rule); the gate must reject both, else it is compromised.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from src.hl import modular_rule_policy as M
from src.hl import tension as T


@dataclass
class ParetoGate:
    """The Pareto / tension-vector gate: admits only non-dominated forward moves over (return, -description_len),
    priced by alpha-investing wealth and a tension-harm budget, with an adversarial validator and canary teeth."""
    dev_ctor: object                          # base regime
    stressor_ctors: list                      # DISJOINT stressor regimes for the adversarial validator (F2)
    G: int
    dev_seeds: list = field(default_factory=lambda: list(range(200, 400)))
    holdout_pool: list = field(default_factory=lambda: list(range(1000, 1600)))   # rotated in slices (F3)
    tension_budget: float = 3.0               # cumulative tension-harm cap (F4/F11)
    wealth: float = 0.20                       # alpha-investing wealth (F3)
    frontier: list = field(default_factory=list)   # list of (policy, vec) — the accepted Pareto set (F1/F10)
    spent_tension: float = 0.0
    epoch: int = 0
    compromised: bool = False                 # F12 teeth: a canary escape FREEZES the gate (deny-all) until manual reset
    audit: dict = field(default_factory=lambda: {"dominated": 0, "adversary_vetoes": 0, "budget_refusals": 0,
                                                 "wealth_refusals": 0, "canary_caught": 0, "canary_escaped": 0})

    def _vec(self, pol, seeds): return T.tension_vector(pol, self.dev_ctor, seeds)

    def _rotating_holdout(self):
        # F3: a DIFFERENT holdout slice each query (never reuse a seed set across the query storm)
        n = 200; start = (self.epoch * 61) % (len(self.holdout_pool) - n)
        self.epoch += 1
        return self.holdout_pool[start:start + n]

    def _tension_harm(self, cand_vec, cur_vec):
        # F4/F11: authority spent = the legibility a move burns (description_len increase), never return.
        return max(0.0, (cand_vec["description_len"] - cur_vec["description_len"]))

    def review(self, cand, current):
        """Admit `cand` only if it is a Pareto-non-dominated forward move that clears the alpha-investing holdout,
        survives every stressor regime, and fits the tension budget; returns (verdict, info, resulting policy)."""
        # F12 teeth: once a canary has escaped, the gate is COMPROMISED and refuses every move until manual reset.
        if self.compromised:
            return "REFUSED_GATE_COMPROMISED", {}, current
        cur_vec = self._vec(current, self.dev_seeds)
        cand_vec = self._vec(cand, self.dev_seeds)

        # F1: must be Pareto-non-dominated vs the accepted frontier (a genuine frontier point, not behind it)
        front_vecs = [v for _, v in self.frontier] or [cur_vec]
        if not T.non_dominated(cand_vec, front_vecs):
            self.audit["dominated"] += 1
            return "REJECTED_DOMINATED", {"cand": cand_vec}, current
        # ...and a strict FORWARD move vs the incumbent: it must add return, OR cut length at no return loss.
        # A longer-but-higher-return policy IS an admissible new frontier point (that is what a frontier is for);
        # what stops runaway length is the tension budget (F4/F11), not this check. A lateral/backward
        # Pareto-incomparable jump (loses return to buy nothing, or same return same length) is refused here.
        forward = (cand_vec["return"] > cur_vec["return"] + 1e-6) or \
                  (cand_vec["description_len"] < cur_vec["description_len"] - 1e-9 and cand_vec["return"] >= cur_vec["return"] - 1e-6)
        if not forward:
            self.audit["dominated"] += 1
            return "REJECTED_NO_FRONTIER_GAIN", {"cand": cand_vec}, current

        # F3: alpha-investing — level shrinks as wealth depletes; a query storm cannot keep testing
        if self.wealth < 1e-3:
            self.audit["wealth_refusals"] += 1
            return "REFUSED_ALPHA_WEALTH_EXHAUSTED", {"wealth": round(self.wealth, 4)}, current
        alpha_t = min(0.5, self.wealth / 2.0)          # test level ∈ (0, 0.5]; never >=1 (log domain) or compounding
        z_crit = math.sqrt(2.0 * math.log(1.0 / alpha_t))
        hs = self._rotating_holdout()
        rb = M.evaluate(current, self.dev_ctor, hs); rc = M.evaluate(cand, self.dev_ctor, hs)
        d = rc - rb; hmean, hsem = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)) + 1e-9)
        z = hmean / hsem
        # Equal-length moves (retunes) MUST prove their gain on unseen rotating seeds — the re-verifier mined the
        # fixed dev seeds through the old equal-length waiver (dev "+0.065" that was truly -0.075, z=-4.16 on 4000
        # independent seeds, salami-compounding to z=-7.00). Strictly-shorter moves buy legibility, so they get a
        # NON-INFERIORITY test instead: no material return regression (0.05 = the gate's materiality bar) on the
        # holdout — a genuine simplification (identical behavior) passes, a mined harmful strip does not.
        shorter = cand_vec["description_len"] < cur_vec["description_len"] - 1e-9
        if shorter:
            passes_holdout = (hmean + 0.05) / hsem > z_crit
        else:
            passes_holdout = hmean > 0 and z > z_crit
        if not passes_holdout:
            self.wealth -= alpha_t / (1 - alpha_t)          # alpha-investing drain on a failed test
            return "REJECTED_HOLDOUT", {"z": round(z, 2), "z_crit": round(z_crit, 2), "wealth": round(self.wealth, 4)}, current

        # F2: adversarial validator — the candidate must not REGRESS vs the incumbent on any disjoint stressor regime.
        # No dev-gain precondition: a length-reducing move that quietly loses on a stressor is still a regression
        # (that precondition was the corridor the F12 verifier exploited to walk a harmful move past the adversary).
        for sc in self.stressor_ctors:
            adv = float((M.evaluate(cand, sc, self.dev_seeds[:80]) - M.evaluate(current, sc, self.dev_seeds[:80])).mean())
            if adv < -0.05:
                self.audit["adversary_vetoes"] += 1
                self.wealth -= alpha_t / (1 - alpha_t)
                return "VETOED_BY_ADVERSARY", {"stressor_advantage": round(adv, 3)}, current

        # F4/F11: price authority in TENSION-HARM; deny once the tension budget is exhausted
        harm = self._tension_harm(cand_vec, cur_vec)
        if self.spent_tension + harm > self.tension_budget:
            self.audit["budget_refusals"] += 1
            return "REFUSED_TENSION_BUDGET", {"spent": round(self.spent_tension, 2), "harm": round(harm, 2)}, current

        # ACCEPT: pay a small bounded alpha-investing discovery payout ω (not the full alpha_t — that compounds and
        # would drive the test level past 1), debit tension, expand + prune the frontier
        self.wealth = min(1.0, self.wealth + 0.05)
        self.spent_tension += harm
        self.frontier = [(p, v) for (p, v) in self.frontier if not T.dominates(cand_vec, v)]
        self.frontier.append((cand, cand_vec))
        return "ACCEPTED_PARETO", {"cand": cand_vec, "harm": round(harm, 3), "wealth": round(self.wealth, 4)}, cand

    def _sandboxed_review(self, cand, current):
        """Run review() without letting the probe touch live state: a canary must never spend wealth/tension,
        advance the holdout rotation, or (worst case) be ACCEPTED into the real frontier."""
        snap = (self.wealth, self.spent_tension, self.epoch, list(self.frontier), dict(self.audit))
        try:
            verdict, _, _ = self.review(cand, current)
        finally:
            self.wealth, self.spent_tension, self.epoch, self.frontier, self.audit = \
                snap[0], snap[1], snap[2], snap[3], snap[4]
        return verdict

    # F12: adversarial canaries — each EMPIRICALLY harmful on this substrate (verified: incumbent +3.75 vs
    # unconditional-provide -8.59 and unwind-strip +1.77); ALL must be rejected. An escape is not a statistic:
    # it sets compromised=True and the gate refuses every subsequent move until manually reset.
    def canary_check(self, current):
        """Sandbox-review generated harmful canaries (legibility-only-cost, unconditional provide, unwind strip);
        any ACCEPTED escape sets compromised=True and freezes the gate."""
        canaries = []
        # (a) legibility-only-cost: an EXACT duplicate of an existing provide clause — guaranteed behavior-identical
        # (a default-thr/cap duplicate could be a live improvement over a dead clause and false-positive-brick the gate)
        covered = sorted(T.M.covered_venues(current))
        dup = next((c for c in current if c[0] == "provide"), None)
        if dup is not None:
            canaries.append(("legibility_only_cost", current + [dup]))
        # (b) return-toxic: provide at one fixed venue UNCONDITIONALLY (thr=0) — adverse ~ (G-1)/G of the time
        canaries.append(("unconditional_fixed_venue", [T.M.provide_clause(0, thr=0.0)] +
                         [c for c in current if c[0] == "unwind"]))
        # (c) return-toxic + length-REDUCING (attacks the waived-holdout path): strip the unwind clause.
        # PRECONDITION (kills the self-DoS the red-team flagged): stripping unwind is a harm probe only if some
        # provide clause can actually fire and accumulate inventory (cap>0, thr<=1); on a dead-clause incumbent
        # the strip is a LEGITIMATE simplification and asserting it as a canary would false-positive-freeze the gate.
        live_provide = any(c[0] == "provide" and c[3] > 0 and c[2] <= 1.0 for c in current)
        if any(c[0] == "unwind" for c in current) and live_provide:
            canaries.append(("strip_unwind", [c for c in current if c[0] != "unwind"]))
        results = []
        for name, cand in canaries:
            v = self._sandboxed_review(cand, current)
            results.append((name, v))
            if v.startswith("ACCEPTED"):
                self.audit["canary_escaped"] += 1
                self.compromised = True                       # the teeth: freeze the gate
            else:
                self.audit["canary_caught"] += 1
        return results
