"""HL-6 (polygon substrate) — a RULE-BASED heuristic policy on the Series-G corner env, with a small typed knob set.

This is the "Heuristic System" the coding-agent edits (the HL paradigm: hierarchy + rules + coefficients). It is
deliberately DETUNED at the start so the loop has real headroom to improve — the loop's job is to move the knobs toward
the Glosten-Milgrom-optimal thresholds through the full governance stack (schema -> gate -> writ -> ledger -> teacher).

The policy is a transparent belief-threshold rule (the GM structure): PROVIDE when toxic-belief is low enough and not
capacity-capped; UNWIND (AGGRESS) when inventory is high near the horizon; else ABSTAIN. Knobs = the thresholds.
"""
from __future__ import annotations

import numpy as np

# The tunable knob registry for the heuristic policy (typed; ranges are the legal edit space).
KNOBS = {
    "provide_belief_thresh": {"range": [0.05, 0.95], "default": 0.80, "type": "float",
                              "guarantee": "PROVIDE only when P(toxic) < this (GM cutoff is s/(s+alpha))"},
    "unwind_inv_thresh":     {"range": [1, 4], "default": 4, "type": "int",
                              "guarantee": "AGGRESS-unwind when inventory >= this near the horizon"},
    "unwind_horizon":        {"range": [0, 10], "default": 0, "type": "int",
                              "guarantee": "treat the last-N steps as the unwind tail"},
}
ACTIONS = {"PROVIDE": 0, "ABSTAIN": 1, "AGGRESS": 2}


def default_coeffs():
    """Return the detuned starting coeffs (each knob at its default)."""
    return {k: v["default"] for k, v in KNOBS.items()}


def policy_action(coeffs, belief, inv, t, T, I_max):
    """Transparent rule: returns 0 PROVIDE / 1 ABSTAIN / 2 AGGRESS."""
    tail = (T - t) <= coeffs["unwind_horizon"]
    if inv >= coeffs["unwind_inv_thresh"] and inv > 0 and tail:
        return ACTIONS["AGGRESS"]
    if belief < coeffs["provide_belief_thresh"] and inv < I_max:
        return ACTIONS["PROVIDE"]
    return ACTIONS["ABSTAIN"]


def evaluate(coeffs, env_ctor, seeds):
    """Mean episodic return of the rule policy under `coeffs` over `seeds` (deterministic env resets)."""
    env = env_ctor()
    rets = []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False; R = 0.0
        while not done:
            b = float(env.belief); inv = int(env.inv[0]); t = env.t
            a = policy_action(coeffs, b, inv, t, env.T, env.m.I_max)
            obs, r, term, trunc, _ = env.step([a]); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


def action_signature(coeffs, env_ctor, probe_states):
    """The policy's action on a fixed frozen probe grid = its behavioral fingerprint (for blast-radius measurement)."""
    env = env_ctor()
    T, I_max = env.T, env.m.I_max
    return np.array([policy_action(coeffs, b, inv, t, T, I_max) for (b, inv, t) in probe_states], dtype=int)


def frozen_probe_states(env_ctor, n_belief=13):
    """Build the fixed (belief, inventory, tail) probe grid used to fingerprint a policy."""
    env = env_ctor()
    states = []
    for b in np.linspace(0.02, 0.98, n_belief):
        for inv in range(env.m.I_max + 1):
            for t in (2, 6, 10, 14, 18):
                states.append((float(b), int(inv), int(t)))
    return states
