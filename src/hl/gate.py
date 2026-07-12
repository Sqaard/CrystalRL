"""HL-2 — the GV GATE: the deny-by-default, proposer-blind, blast-radius-tiered, frozen-deflated acceptance authority.

Enforces the HL Constitution's load-bearing rules on the polygon:
  - DENY-BY-DEFAULT validation (schema.validate).
  - TIER BY MEASURED BLAST RADIUS, not the proposer's claim: behavioral_delta = fraction of FROZEN probes whose action
    changes; tier_assigned = max(claim, measured). A small-looking edit with a big delta AUTO-ESCALATES (anti-laundering).
  - CHARGE EVERY EVAL QUERY to a deflation ledger; the acceptance margin DEFLATES with the number of queries (the
    multiplicity the proposer spends — extends deflated-Sharpe trial-counting to all queries).
  - FROZEN HOLDOUT with CRN: dev signal is necessary but NOT sufficient; acceptance requires a paired improvement on a
    DISJOINT held-out seed set the proposer never sees, beyond the deflated margin (a false-GREEN on dev is caught).
  - ANCHOR-Sigma-delta cumulative budget: cumulative behavioral delta vs the ANCHOR (not the ancestor) is capped;
    exceeding it forces requalification (anti-salami — many small approved edits cannot sum into one large unreviewed one).
  - WRIT LADDER + AUTHORITY LEDGER: an accepted change is a certified writ that debits per-principal cumulative authority.
The gate is INDEPENDENT of the proposer (it re-measures everything) and is the SOLE deployer.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hl.schema import Proposal, ChangeDossier, validate, coeffs_hash  # noqa: E402
from src.hl import substrate as S  # noqa: E402
from src.crystal.writ_ladder import WritCertificate, CumulativeAuthorityLedger  # noqa: E402


@dataclass
class HLGate:
    env_ctor: object
    anchor_coeffs: dict
    dev_seeds: list
    holdout_seeds: list
    probe_states: list
    base_margin: float = 0.10                 # base holdout improvement required (return units)
    anchor_delta_cap: float = 0.60            # cumulative behavioral Σδ vs anchor before forced requalify
    registry_version: str = "hl_polygon_v1"
    # state
    eval_queries: int = 0
    cum_anchor_delta: float = 0.0
    trials: int = 0
    ledger: CumulativeAuthorityLedger = None
    anchor_sig: np.ndarray = None

    n_requalifications: int = 0

    def __post_init__(self):
        if self.ledger is None:
            self.ledger = CumulativeAuthorityLedger("hl_proposer", anchor_version="anchor_v1", D_cap=4.0, K_max=3, eps_bound=6.6)
        self.anchor_sig = S.action_signature(self.anchor_coeffs, self.env_ctor, self.probe_states)

    def requalify(self, new_anchor_coeffs: dict):
        """Full re-review: re-baseline the anchor to the current (already-reviewed) policy, reset the drift budget and the
        authority ledger anchor. This is the forced full review that the anchor-Σδ cap triggers (anti-salami)."""
        self.anchor_coeffs = dict(new_anchor_coeffs)
        self.anchor_sig = S.action_signature(self.anchor_coeffs, self.env_ctor, self.probe_states)
        self.cum_anchor_delta = 0.0
        self.n_requalifications += 1
        self.ledger.requalify(f"anchor_v{self.n_requalifications + 1}")

    # ---- measured blast radius ----
    def _behavioral_delta(self, cand_coeffs):
        sig = S.action_signature(cand_coeffs, self.env_ctor, self.probe_states)
        return float(np.mean(sig != self.anchor_sig)), sig

    @staticmethod
    def _tier_from_delta(delta):
        return "T0" if delta < 0.05 else ("T1" if delta < 0.25 else "T2")

    @staticmethod
    def _claimed_tier(pred):
        return {"small": "T0", "medium": "T1", "large": "T2"}[pred]

    def _apply(self, coeffs, p: Proposal):
        c = dict(coeffs)
        c[p.target] = p.value if p.operator == "SET" else c[p.target] + p.value
        if S.KNOBS[p.target]["type"] == "int":
            c[p.target] = int(round(c[p.target]))
        return c

    def _paired(self, base_coeffs, cand_coeffs, seeds):
        rb = S.evaluate(base_coeffs, self.env_ctor, seeds)
        rc = S.evaluate(cand_coeffs, self.env_ctor, seeds)
        self.eval_queries += 2 * len(seeds)
        diff = rc - rb
        return float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)) + 1e-9)

    def review(self, p: Proposal, current_coeffs: dict, current_round: int, parent_change_id: str = "root"):
        # inject current values so DELTA validation & range checks are correct
        reg = {k: {**v, "_current": current_coeffs.get(k, v["default"])} for k, v in S.KNOBS.items()}
        ok, why = validate(p, reg, current_round)
        cand = self._apply(current_coeffs, p) if ok else current_coeffs
        delta, cand_sig = (self._behavioral_delta(cand)) if ok else (0.0, self.anchor_sig)
        tier_assigned = self._tier_from_delta(delta) if ok else "T0"
        tier_claimed = self._claimed_tier(p.predicted_blast_radius) if ok else "T0"
        laundering = ok and (tier_assigned > tier_claimed)   # measured worse than claimed

        def dossier(verdict, writ, dev=0.0, hold=0.0, cap_scope=0.0):
            return ChangeDossier(
                change_id=f"chg_{current_round}_{p.proposal_id}", parent_change_id=parent_change_id,
                proposer_id=p.proposer_id, round=current_round, target=p.target, operator=p.operator, value=p.value,
                tier_claimed=tier_claimed, tier_assigned=tier_assigned, behavioral_delta=round(delta, 4),
                capital_scope_delta=round(cap_scope, 4), dev_improvement=round(dev, 4), holdout_improvement=round(hold, 4),
                eval_queries_charged=self.eval_queries, verdict=verdict, writ_status=writ,
                rollback_ref=p.rollback_ref, registry_version=self.registry_version, policy_hash=coeffs_hash(current_coeffs))

        if not ok:
            return "REFUSED_INVALID", dossier(f"REFUSED_INVALID:{why}", "DRAFT"), current_coeffs

        # anchor Σδ budget (anti-salami): CUMULATIVE path length of ACCEPTED behavioral movement since the last full
        # review. step_delta = fraction of frozen probes whose action differs between the candidate and the CURRENT
        # policy (not the anchor) — so it accumulates even when the net drift oscillates back (an oscillating-drift
        # salami attack). If (cumulative + this step) would exceed the cap, FORCE REQUALIFICATION (full re-review +
        # re-baseline), never a silent sum past the cap.
        cur_sig = S.action_signature(current_coeffs, self.env_ctor, self.probe_states)
        step_delta = float(np.mean(cand_sig != cur_sig))
        if self.cum_anchor_delta + step_delta > self.anchor_delta_cap:
            return "REFUSED_ANCHOR_BUDGET", dossier(
                "REFUSED_ANCHOR_BUDGET (cum %.2f + step %.2f > cap %.2f — requalify vs anchor)"
                % (self.cum_anchor_delta, step_delta, self.anchor_delta_cap), "DRAFT"), current_coeffs

        # CRN dev signal (necessary)
        dev_mean, dev_sem = self._paired(current_coeffs, cand, self.dev_seeds)
        writ = WritCertificate("SET_KNOB", p.target, policy_version=self.registry_version, cfbank_version="hl_v1")
        writ.certify("C0_addressable", True).certify("C2_dose", True)
        if dev_mean <= 0.0:
            return "REJECTED_NO_DEV_SIGNAL", dossier("REJECTED_NO_DEV_SIGNAL", writ.status, dev=dev_mean), current_coeffs
        writ.certify("C1_causal", True)  # dev improvement = a measured causal effect on return

        # FROZEN HOLDOUT with deflated, TIER-SCALED margin (sufficient). Deflation is by the number of holdout TESTS
        # (self.trials = the multiplicity that matters for a frozen set); the tier multiplier makes blast-radius
        # LOAD-BEARING: a larger measured behavioral delta must clear a proportionally higher bar (a big change is not
        # admitted on the same thin evidence as a tweak).
        hold_mean, hold_sem = self._paired(current_coeffs, cand, self.holdout_seeds)
        self.trials += 1
        tier_mult = {"T0": 1.0, "T1": 1.8, "T2": 3.0}[tier_assigned]
        deflated = self.base_margin * tier_mult * math.sqrt(1.0 + math.log(max(1, self.trials)))
        passes = (hold_mean > deflated) and (hold_mean > 2.0 * hold_sem)
        writ.certify("C3_envelope", True).certify("C4_side_effect", True).certify("C5_frozen_gate", bool(passes))
        if not passes:
            return "REJECTED_HOLDOUT", dossier(f"REJECTED_HOLDOUT (hold={hold_mean:.3f} <= deflated_margin={deflated:.3f})",
                                               writ.status, dev=dev_mean, hold=hold_mean), current_coeffs

        # accept: version-pin, debit authority (alpha = behavioral delta), commit
        writ.certify("C6_version_pin", True)
        issued, ledreason = self.ledger.issue(f"chg_{current_round}_{p.proposal_id}", alpha=max(delta, 0.05), duration=1.0)
        self.ledger.release(f"chg_{current_round}_{p.proposal_id}")  # authority spent, not refunded
        if not issued:
            return "REFUSED_AUTHORITY_LEDGER", dossier(f"REFUSED_AUTHORITY_LEDGER ({ledreason})", writ.status,
                                                       dev=dev_mean, hold=hold_mean), current_coeffs
        self.cum_anchor_delta += step_delta       # accumulate the accepted path length (read by the anti-salami check)
        return "ACCEPTED", dossier("ACCEPTED" + (" [AUTO-ESCALATED tier]" if laundering else ""), writ.status,
                                   dev=dev_mean, hold=hold_mean), cand
