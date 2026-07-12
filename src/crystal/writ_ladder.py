"""C-6 — the C0..C6 WRIT-CERTIFICATION LADDER + the per-principal signed CUMULATIVE-AUTHORITY LEDGER, wired to the
measured CRYSTAL-1 experiments. This is the capstone that turns the HL5 IV FINAL (design) + the C-1..C-5 evidence
into runnable governance code for belief writes.

C0 addressability     — the write target is a named, registered belief coordinate (K-simplex), no hidden path.
C1 causal sufficiency — the write moves the policy to the world-model optimum (C-2 filter_policy_agreement vs the
                        exogenous belief-MDP optimum: corner=0.80 PROVED).
C2 dose window+placebo— monotone dose over the barycentric coordinate, refusal-rate 0 (C-2).
C3 envelope+demotion  — valid only on the visited envelope; off-envelope -> BeliefGovernor projects + annunciates a
                        GUARANTEE_DELTA (src/crystal/governor.py).
C4 side-effect budget — bounded collateral; because C-5 CONFIRMED sign-epistasis on >=2-belief-dim writes, per-command
                        budgets DO NOT SUM and the cross-term is SIGN-UNSTABLE -> the ledger charges the WORST-CASE
                        |epsilon_ij| for co-active writs and caps N_live <= K.
C5 frozen gate        — survives the pre-registered frozen deflated evaluation (hook; the program's firewall).
C6 version-pinned     — bound to (policy_version, cfbank_version); ANY bump voids the writ back to C1.

The ledger is the cross-series first-class object (K per-knob trajectory + GV anchor-Sigma-delta + IV D=Sum|a|*tau +
epistasis cross-term + TB ratchet), signed per-principal, reset only by re-certification vs the anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, field


C_RUNGS = ["C0_addressable", "C1_causal", "C2_dose", "C3_envelope", "C4_side_effect", "C5_frozen_gate", "C6_version_pin"]


@dataclass
class WritCertificate:
    verb: str                       # e.g. "SET_BELIEF"
    site: str                       # named belief coordinate, e.g. "b_toxic" / "venue_2"
    policy_version: str
    cfbank_version: str
    rungs: dict = field(default_factory=lambda: {r: False for r in C_RUNGS})
    dose_window: tuple = (0.0, 1.0)
    envelope: tuple = None          # (lo,hi) on the site coordinate
    status: str = "DRAFT"

    def certify(self, rung: str, ok: bool, evidence: str = ""):
        assert rung in self.rungs, f"unknown rung {rung}"
        self.rungs[rung] = bool(ok)
        self._recompute()
        return self

    def _recompute(self):
        # highest contiguous rung passed from C0
        n = 0
        for r in C_RUNGS:
            if self.rungs[r]:
                n += 1
            else:
                break
        self.status = C_RUNGS[n - 1] if n > 0 else "DRAFT"
        self.level = n - 1

    def void_on_version_bump(self, policy_version: str, cfbank_version: str):
        if policy_version != self.policy_version or cfbank_version != self.cfbank_version:
            # C6 and above void; the write lapses back to C1 (causal claim survives, deployment authority does not)
            for r in C_RUNGS[2:]:
                self.rungs[r] = False
            self.policy_version, self.cfbank_version = policy_version, cfbank_version
            self._recompute()
            return True
        return False

    def deployable(self) -> bool:
        return all(self.rungs[r] for r in C_RUNGS)     # only a fully C0..C6 writ may act on capital


@dataclass
class CumulativeAuthorityLedger:
    """Per-principal signed trajectory ledger. D = Sum |alpha|*duration (+ worst-case Sum |eps_ij|*co_duration for
    co-active pairs, because C-5 showed the cross-term sign-flips and cannot be predicted). Reset only by
    re-certification vs the anchor. Enforces N_live <= K_max and catches salami-slicing via the cumulative cap."""
    principal: str
    anchor_version: str
    D_cap: float
    K_max: int
    eps_bound: float = 6.6          # calibrated worst-case |eps_ij| per co-active pair per unit duration: C-5 global p95
                                    # of |eps_logit| (interpretability/c5_debts.py; max ~8.6). The old 0.5 was the
                                    # MATERIALITY threshold, ~13x too small — N_live<=K was doing the real limiting.
                                    # Prefer a PER-PAIR table (c5_debts_report.json per_pair_bounds) over this scalar.
    D: float = 0.0
    live: dict = field(default_factory=dict)     # writ_id -> alpha
    log: list = field(default_factory=list)

    def can_issue(self, writ_id: str):
        if len(self.live) >= self.K_max:
            return False, f"N_live cap ({self.K_max}) — co-activation blocked (epistasis unbounded beyond K)"
        return True, None

    def issue(self, writ_id: str, alpha: float, duration: float):
        ok, why = self.can_issue(writ_id)
        if not ok:
            self.log.append(("BLOCK", writ_id, why)); return False, why
        main = abs(alpha) * duration
        # worst-case epistasis cross-term vs every currently-live writ (signed -> abs bound)
        cross = self.eps_bound * duration * len(self.live)
        self.D += main + cross
        self.live[writ_id] = alpha
        breach = self.D > self.D_cap
        self.log.append(("ISSUE", writ_id, round(main, 3), round(cross, 3), round(self.D, 3), "BREACH" if breach else "ok"))
        if breach:
            return False, f"D={self.D:.3f} > D_cap={self.D_cap} — FORCE REQUALIFY vs anchor {self.anchor_version} (anti-salami)"
        return True, None

    def release(self, writ_id: str):
        # authority is not refunded on release (irreversible-consequence discipline): D stays spent
        self.live.pop(writ_id, None)

    def requalify(self, new_anchor_version: str):
        self.D = 0.0; self.anchor_version = new_anchor_version; self.log.append(("REQUALIFY", new_anchor_version))


def _selftest():
    # ---- build a belief-write certificate wired to the measured experiments ----
    cert = WritCertificate("SET_BELIEF", "b_toxic", policy_version="corner_ppo_n1", cfbank_version="cfbank_v1",
                           dose_window=(0.0, 1.0), envelope=(0.02, 0.62))
    cert.certify("C0_addressable", True, "named K-simplex coord; RoleContract registered")
    cert.certify("C1_causal", True, "C-2 filter_policy_agreement=0.80 vs exogenous belief-MDP optimum (PROVED)")
    cert.certify("C2_dose", True, "C-2 dose monotone, refusal-rate 0 (barycentric, on-simplex)")
    cert.certify("C3_envelope", True, "BeliefGovernor projects off-envelope + annunciates GUARANTEE_DELTA")
    cert.certify("C4_side_effect", True, "ghost-portfolio bounded; C-5 sign-epistasis -> worst-case cross-term in ledger")
    cert.certify("C5_frozen_gate", False, "not yet run through the frozen deflated gate")
    assert cert.status == "C4_side_effect" and not cert.deployable(), "must not be deployable without C5/C6"
    cert.certify("C5_frozen_gate", True).certify("C6_version_pin", True)
    assert cert.deployable(), "fully certified writ should be deployable"
    # version bump voids back to C1
    voided = cert.void_on_version_bump("corner_ppo_n1_RETRAIN", "cfbank_v1")
    assert voided and not cert.deployable() and cert.rungs["C1_causal"], "retrain lapses the writ to C1"

    # ---- salami-slicing: many small writs each under a per-write cap, cumulative D catches it ----
    led = CumulativeAuthorityLedger("agent_alpha", anchor_version="frozen_2022_2023", D_cap=3.0, K_max=3, eps_bound=0.5)
    breach_at = None
    for i in range(20):
        ok, why = led.issue(f"w{i}", alpha=0.25, duration=1.0)   # each tiny (0.25) — "harmless" alone
        led.release(f"w{i}")                                      # released, but D stays spent (no refund)
        if not ok:
            breach_at = i; break
    assert breach_at is not None, "cumulative cap must catch salami-slicing of individually-small writs"

    # ---- co-activation cap: N_live <= K blocks the K+1'th simultaneous writ ----
    led2 = CumulativeAuthorityLedger("agent_beta", "frozen_2022_2023", D_cap=100.0, K_max=2, eps_bound=0.5)
    r1 = led2.issue("a", 0.3, 1.0); r2 = led2.issue("b", 0.3, 1.0); r3 = led2.issue("c", 0.3, 1.0)
    assert r1[0] and r2[0] and not r3[0], "N_live cap must block the 3rd co-active writ"

    print("writ_ladder selftest OK:")
    print(f"  certificate status walked C0->C6; deployable only when fully certified; retrain voided it to C1.")
    print(f"  salami-slicing: twelve 0.25-alpha writs sum to exactly 3.0; the {breach_at + 1}th breached D_cap=3.0 "
          f"(cumulative caught it; no refund on release).")
    print(f"  co-activation: N_live cap K=2 blocked the 3rd simultaneous writ ('{r3[1]}').")


if __name__ == "__main__":
    _selftest()
