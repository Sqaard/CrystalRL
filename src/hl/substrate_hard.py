"""HL-6 (HARDER substrate) — a 6-knob rule policy with INTERACTING levers (Q2 "harder task" + Q1 "bigger knob surface").

Drop-in replacement for src/hl/substrate.py (same interface: KNOBS, default_coeffs, policy_action, evaluate,
action_signature, frozen_probe_states) so the HL gate runs on it unchanged (rebind gate.S). Harder than the 3-knob
substrate because the optimum requires COORDINATING levers that interact:
  - a risk-off DUMP lever (unwind when toxic) that trades off against the horizon-unwind lever,
  - a PROVIDE HYSTERESIS band (sticky provide, cuts churn) that interacts with the provide threshold,
  - a capacity cap that interacts with both.
A greedy one-knob-at-a-time search can get stuck in a local optimum, so the proposer's search quality matters more.
"""
from __future__ import annotations

import numpy as np

KNOBS = {
    "provide_belief_thresh": {"range": [0.05, 0.95], "default": 0.80, "type": "float",
                              "guarantee": "PROVIDE only when P(toxic) < this"},
    "provide_hysteresis":    {"range": [0.0, 0.40], "default": 0.0, "type": "float",
                              "guarantee": "once providing, keep providing until P(toxic) > thresh+this (sticky, cuts churn)"},
    "unwind_belief_thresh":  {"range": [0.50, 0.99], "default": 0.99, "type": "float",
                              "guarantee": "AGGRESS-dump inventory when P(toxic) >= this (risk-off; 0.99 = never)"},
    "unwind_inv_thresh":     {"range": [1, 4], "default": 4, "type": "int",
                              "guarantee": "AGGRESS-unwind when inventory >= this AND near the horizon"},
    "unwind_horizon":        {"range": [0, 10], "default": 0, "type": "int",
                              "guarantee": "treat the last-N steps as the unwind tail"},
    "max_provide_inv":       {"range": [1, 4], "default": 4, "type": "int",
                              "guarantee": "stop PROVIDING once inventory >= this (capacity cap)"},
}
ACTIONS = {"PROVIDE": 0, "ABSTAIN": 1, "AGGRESS": 2}


def default_coeffs():
    return {k: v["default"] for k, v in KNOBS.items()}


def policy_action(coeffs, belief, inv, t, T, I_max, providing_prev=False):
    if inv > 0 and belief >= coeffs["unwind_belief_thresh"]:
        return ACTIONS["AGGRESS"]                                   # risk-off dump in toxic
    if inv >= coeffs["unwind_inv_thresh"] and inv > 0 and (T - t) <= coeffs["unwind_horizon"]:
        return ACTIONS["AGGRESS"]                                   # horizon unwind
    eff = coeffs["provide_belief_thresh"] + (coeffs["provide_hysteresis"] if providing_prev else 0.0)
    cap = min(int(coeffs["max_provide_inv"]), I_max)
    if belief < eff and inv < cap:
        return ACTIONS["PROVIDE"]
    return ACTIONS["ABSTAIN"]


def evaluate(coeffs, env_ctor, seeds):
    env = env_ctor()
    rets = []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False; R = 0.0; prev = False
        while not done:
            b = float(env.belief); inv = int(env.inv[0]); t = env.t
            a = policy_action(coeffs, b, inv, t, env.T, env.m.I_max, providing_prev=prev)
            prev = (a == ACTIONS["PROVIDE"])
            obs, r, term, trunc, _ = env.step([a]); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


def action_signature(coeffs, env_ctor, probe_states):
    env = env_ctor(); T, I_max = env.T, env.m.I_max
    return np.array([policy_action(coeffs, b, inv, t, T, I_max, providing_prev=False)
                     for (b, inv, t) in probe_states], dtype=int)


def frozen_probe_states(env_ctor, n_belief=13):
    env = env_ctor(); states = []
    for b in np.linspace(0.02, 0.98, n_belief):
        for inv in range(env.m.I_max + 1):
            for t in (2, 6, 10, 14, 18):
                states.append((float(b), int(inv), int(t)))
    return states
