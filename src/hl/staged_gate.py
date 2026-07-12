"""HL v3 — the MULTI-ROLE, STAGED gate (GV Art III/V). Four roles on DISJOINT seed sets (so a change that overfits one
role's data is caught by another — real 'effective challenge', not org-chart independence), and a candidate that clears
the gate is not deployed directly: it is promoted shadow -> canary -> fleet, and can ABORT at canary (rolled back)
before full deployment. This is how the 'inner human' (release sign-off) is removed safely.

  PROPOSER (external): proposes a candidate.
  VALIDATOR (veto): re-measures on a VALIDATION seed set disjoint from the proposer's dev; vetoes if not an improvement.
  OPERATOR (staged exposure): shadow (frozen holdout, no commit) -> canary (2 disjoint windows, no hard-fail) -> fleet.
  AUDITOR (regime check): a known-bad canary is injected periodically; if it is ever PROMOTED, the auditor FREEZES the
    gate (the regime is broken). Tracks the canary-abort and false-accept rates.
"""
from __future__ import annotations
import math
import numpy as np
from src.hl import rule_policy as RP
from src.crystal.writ_ladder import CumulativeAuthorityLedger

DEV = list(range(200, 400))          # proposer signal
VALID = list(range(600, 800))        # VALIDATOR — independent veto (disjoint from dev)
SHADOW = list(range(1000, 1200))     # OPERATOR shadow / frozen holdout
CANARY = [list(range(1400, 1500)), list(range(1500, 1600))]  # OPERATOR canary windows (disjoint)


class StagedGate:
    """The multi-role staged gate: proposer/validator/operator/auditor on disjoint seed sets, promoting a candidate
    shadow -> canary -> fleet with abort-at-canary, and freezing if a known-bad is ever promoted."""
    def __init__(self, env_ctor, anchor_rules, probes, base_margin=0.10, val_env_ctor=None):
        # val_env_ctor = a SHIFTED regime for the VALIDATOR + CANARY roles: their job is to catch a change that overfits
        # the proposer's regime and does not generalize under a shift (real 'effective challenge', not more i.i.d. seeds).
        self.env_ctor = env_ctor; self.val_ctor = val_env_ctor or env_ctor
        self.probes = probes; self.base_margin = base_margin
        self.anchor_sig = RP.action_signature(anchor_rules, env_ctor, probes)
        self.trials = 0
        self.ledger = CumulativeAuthorityLedger("hl_v3", anchor_version="v1", D_cap=6.0, K_max=4, eps_bound=6.6)
        self.frozen = False
        self.audit = {"validator_vetoes": 0, "shadow_rejects": 0, "canary_aborts": 0,
                      "known_bad_promoted": 0, "canary_windows_checked": 0}

    def _paired(self, base, cand, seeds, ctor=None):
        c = ctor or self.env_ctor
        d = RP.evaluate(cand, c, seeds) - RP.evaluate(base, c, seeds)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)) + 1e-9)

    def _tier(self, cand):
        delta = float(np.mean(RP.action_signature(cand, self.env_ctor, self.probes) != self.anchor_sig))
        return "T0" if delta < 0.05 else ("T1" if delta < 0.25 else "T2"), delta

    def review(self, cand, current, rnd, is_known_bad=False):
        """Returns (verdict, stagelog, new_current). new_current changes ONLY on FLEET promotion."""
        stg = {}
        if self.frozen:
            return "GATE_FROZEN", stg, current
        tier, delta = self._tier(cand); stg["tier"] = tier

        # --- PROPOSER dev signal (necessary) ---
        dev_m, _ = self._paired(current, cand, DEV); stg["dev"] = round(dev_m, 3)
        if dev_m <= 0:
            return "REJECTED_NO_DEV_SIGNAL", stg, current

        # --- VALIDATOR: independent veto on disjoint validation seeds UNDER THE SHIFTED REGIME ---
        val_m, _ = self._paired(current, cand, VALID, ctor=self.val_ctor); stg["validation"] = round(val_m, 3)
        if val_m <= 0:
            self.audit["validator_vetoes"] += 1
            return "VETOED_BY_VALIDATOR", stg, current

        # --- OPERATOR shadow: frozen holdout beyond a deflated, tier-scaled margin (no commit) ---
        self.trials += 1
        sh_m, sh_s = self._paired(current, cand, SHADOW)
        margin = self.base_margin * {"T0": 1.0, "T1": 1.8, "T2": 3.0}[tier] * math.sqrt(1 + math.log(max(1, self.trials)))
        stg["shadow"] = round(sh_m, 3); stg["margin"] = round(margin, 3)
        if not (sh_m > margin and sh_m > 2 * sh_s):
            self.audit["shadow_rejects"] += 1
            return "REJECTED_SHADOW", stg, current

        # --- OPERATOR canary: staged exposure across disjoint windows; ABORT (no commit) on any hard-fail ---
        cur_windows = []
        for w in CANARY:
            self.audit["canary_windows_checked"] += 1
            rc = RP.evaluate(cand, self.val_ctor, w).mean(); rb = RP.evaluate(current, self.val_ctor, w).mean()
            cur_windows.append(round(float(rc - rb), 3))
            if rc < rb - 0.05:                                   # hard-fail: candidate materially worse on a canary slice
                self.audit["canary_aborts"] += 1
                stg["canary"] = cur_windows
                return "ABORTED_CANARY", stg, current            # rolled back — NOT deployed
        stg["canary"] = cur_windows

        # --- AUDITOR: a known-bad that reaches here means the regime is broken -> FREEZE ---
        if is_known_bad:
            self.audit["known_bad_promoted"] += 1; self.frozen = True
            stg["auditor"] = "KNOWN_BAD_PROMOTED_GATE_FROZEN"
            return "GATE_FROZEN", stg, current

        # --- OPERATOR fleet: commit (writ + authority ledger) ---
        issued, _ = self.ledger.issue(f"chg{rnd}", alpha=max(delta, 0.05), duration=1.0); self.ledger.release(f"chg{rnd}")
        if not issued:
            return "REFUSED_AUTHORITY_LEDGER", stg, current
        return "PROMOTED_FLEET", stg, cand
