"""HL v2 substrate — registry-DRIVEN rule policy with EXPOSED + LATENT knobs (targets of the add_knob operator) and
CONSTRAINT knobs as first-class levers. The policy reads whatever the DynamicRegistry currently exposes; a latent knob
sits at its default (no-op) until the coding-agent EXPOSES it. This is the self-expanding command surface.
"""
from __future__ import annotations
import numpy as np
from src.hl.registry_v2 import KnobSpec, DynamicRegistry

ACTIONS = {"PROVIDE": 0, "ABSTAIN": 1, "AGGRESS": 2}


def build_registry():
    """Construct the v2 DynamicRegistry: exposed tuning knobs, latent add_knob targets, and a fenced contract knob."""
    specs = [
        # --- exposed TUNING knobs (agent-facing at start) ---
        KnobSpec("provide_belief_thresh", 0.05, 0.95, 0.80, "float", "exposure_action", "tuning",
                 "PROVIDE only when P(toxic) < this", hazard="high", exposed=True),
        KnobSpec("provide_hysteresis", 0.0, 0.40, 0.0, "float", "exposure_action", "tuning",
                 "sticky provide band (cuts churn)", hazard="low", exposed=True),
        KnobSpec("unwind_belief_thresh", 0.50, 0.99, 0.99, "float", "exposure_action", "tuning",
                 "AGGRESS-dump when P(toxic) >= this", hazard="medium", exposed=True),
        KnobSpec("unwind_inv_thresh", 1, 4, 4, "int", "exposure_action", "tuning",
                 "unwind when inventory >= this near horizon", hazard="low", exposed=True),
        KnobSpec("unwind_horizon", 0, 10, 0, "int", "exposure_action", "tuning",
                 "last-N steps are the unwind tail", hazard="low", exposed=True),
        KnobSpec("max_provide_inv", 1, 4, 4, "int", "exposure_action", "tuning",
                 "capacity cap: stop providing at this inventory", hazard="low", exposed=True),
        # --- LATENT knobs (in the catalog; targets of add_knob) ---
        KnobSpec("provide_belief_thresh_lo", 0.02, 0.50, 0.02, "float", "exposure_action", "tuning",
                 "aggressive-provide tier: below this belief, provide even at capacity (a second decision tier)",
                 hazard="medium", exposed=False),
        KnobSpec("abstain_after_dump", 0, 5, 0, "int", "exposure_action", "tuning",
                 "CONSTRAINT lever: after an AGGRESS-dump, ABSTAIN for N steps (anti-churn cool-down)",
                 hazard="low", exposed=False),
        # --- CONTRACT knob (fenced; never agent-facing) ---
        KnobSpec("simplex_invariant", 1.0, 1.0, 1.0, "float", "belief_state", "contract",
                 "belief remains a valid simplex (structural invariant; not editable)", hazard="high", exposed=True),
    ]
    return DynamicRegistry(specs)


def default_coeffs(reg):
    """Return the registry's exposed-knob defaults."""
    return reg.defaults()


def policy_action(coeffs, belief, inv, t, T, I_max, providing_prev=False, cooldown=0):
    """Registry-driven rule reading whatever knobs are exposed (latent ones sit at no-op defaults):
    returns 0 PROVIDE / 1 ABSTAIN / 2 AGGRESS."""
    if cooldown > 0:
        return ACTIONS["ABSTAIN"]                                       # abstain_after_dump cool-down (if exposed)
    if inv > 0 and belief >= coeffs.get("unwind_belief_thresh", 0.99):
        return ACTIONS["AGGRESS"]
    if inv >= coeffs.get("unwind_inv_thresh", 4) and inv > 0 and (T - t) <= coeffs.get("unwind_horizon", 0):
        return ACTIONS["AGGRESS"]
    eff = coeffs["provide_belief_thresh"] + (coeffs.get("provide_hysteresis", 0.0) if providing_prev else 0.0)
    cap = min(int(coeffs.get("max_provide_inv", 4)), I_max)
    lo = coeffs.get("provide_belief_thresh_lo", None)                   # aggressive-provide tier (if exposed)
    if lo is not None and belief < lo and inv < I_max:                 # below lo: provide even past the soft cap
        return ACTIONS["PROVIDE"]
    if belief < eff and inv < cap:
        return ACTIONS["PROVIDE"]
    return ACTIONS["ABSTAIN"]


def evaluate(coeffs, env_ctor, seeds):
    """Return per-seed episodic returns, threading the hysteresis state and the post-dump cool-down."""
    env = env_ctor(); rets = []
    ad = int(coeffs.get("abstain_after_dump", 0))
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False; R = 0.0; prev = False; cd = 0
        while not done:
            b = float(env.belief); inv = int(env.inv[0]); t = env.t
            a = policy_action(coeffs, b, inv, t, env.T, env.m.I_max, providing_prev=prev, cooldown=cd)
            prev = (a == ACTIONS["PROVIDE"])
            cd = ad if a == ACTIONS["AGGRESS"] else max(0, cd - 1)
            obs, r, term, trunc, _ = env.step([a]); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


def action_signature(coeffs, env_ctor, probe_states):
    """The policy's action on a fixed probe grid = its behavioral fingerprint (for blast-radius measurement)."""
    env = env_ctor(); T, I_max = env.T, env.m.I_max
    return np.array([policy_action(coeffs, b, inv, t, T, I_max) for (b, inv, t) in probe_states], dtype=int)


def frozen_probe_states(env_ctor, n_belief=13):
    """Build the fixed (belief, inventory, tail) probe grid used to fingerprint a policy."""
    env = env_ctor(); states = []
    for b in np.linspace(0.02, 0.98, n_belief):
        for inv in range(env.m.I_max + 1):
            for t in (2, 6, 10, 14, 18):
                states.append((float(b), int(inv), int(t)))
    return states
