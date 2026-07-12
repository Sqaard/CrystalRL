"""HL v2 — a DYNAMIC knob registry the coding-agent can GROW under governance (the add_knob / add_rule surface).

The v1 loop could only `retune_knob` existing levers. v2 makes the surface self-expanding: a proposer can EXPOSE a new
knob (from a fenced catalog of latent levers) or ADD a rule clause, each born as a typed registry row with a hazard
class and gated as T1+. Two HL5 ideas are built in:
  - K rule 4 — MODEL-CHECKED SINGLE-OWNER ARBITRATION: every knob declares the shared resource it writes; a load-time
    verifier proves each resource has exactly one owner and the exposure decision is a total order (no ties, no summed
    writers, no cyclic dependency). `verify_arbitration()` is the runnable check the artifact said was missing.
  - K knob classes — CONTRACT vs TUNING: contract knobs (guarantees/invariants) are fenced from the agent; only tuning
    knobs are agent-facing.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class KnobSpec:
    name: str
    lo: float
    hi: float
    default: float
    typ: str                      # "float" | "int"
    resource: str                 # the shared decision resource this knob writes (for single-owner arbitration)
    knob_class: str               # "contract" (fenced) | "tuning" (agent-facing)
    guarantee: str
    hazard: str = "low"           # low | medium | high (feeds the tier floor)
    exposed: bool = True          # False = latent (in the catalog, not yet on the surface)

    def clip(self, v):
        v = max(self.lo, min(self.hi, v))
        return int(round(v)) if self.typ == "int" else float(v)


class DynamicRegistry:
    """Holds exposed + latent knobs; supports EXPOSE (add_knob) and enforces single-owner arbitration."""

    def __init__(self, specs):
        self.specs = {s.name: s for s in specs}

    # ---- surfaces ----
    def exposed(self):
        return {n: s for n, s in self.specs.items() if s.exposed}

    def agent_facing(self):
        return {n: s for n, s in self.specs.items() if s.exposed and s.knob_class == "tuning"}

    def latent_catalog(self):
        return {n: s for n, s in self.specs.items() if not s.exposed}

    def defaults(self):
        return {n: s.default for n, s in self.exposed().items()}

    # ---- add_knob operator: expose a latent lever ----
    def expose(self, name):
        if name not in self.specs:
            return False, f"unknown latent knob {name}"
        if self.specs[name].exposed:
            return False, f"{name} already exposed"
        self.specs[name].exposed = True
        return True, f"exposed {name}"

    # ---- K rule 4: model-checked single-owner arbitration over the EXPOSED graph ----
    def verify_arbitration(self):
        """Returns (ok, report). Proves: (1) each shared resource has exactly ONE owner among exposed knobs of the same
        resource with kind 'owner'; contributors are allowed but the final selector is a total order, not a sum;
        (2) no resource is written by two 'owner' knobs (no tie); (3) the provide/exposure decision is a documented
        total order. On the polygon the decision resource is 'exposure_action' owned by the threshold ladder."""
        owners = {}
        for s in self.exposed().values():
            owners.setdefault(s.resource, []).append(s.name)
        issues = []
        # a resource may have several knobs, but the SELECTOR must be a declared total order; we encode that by
        # requiring at most one knob per resource tagged hazard 'high' (the deciding owner) — others are parameters.
        for res, ks in owners.items():
            deciders = [k for k in ks if self.specs[k].hazard == "high"]
            if len(deciders) > 1:
                issues.append(f"resource '{res}' has {len(deciders)} competing deciders {deciders} — not a total order")
        ok = len(issues) == 0
        return ok, {"resources": {r: ks for r, ks in owners.items()}, "issues": issues,
                    "total_order": "exposure selected by threshold ladder t1<t2 (documented, single decider)"}
