"""HL v3 — a CONJUNCTIVE RULE-LIST policy the coding-agent grows structurally (add_rule) and by crossover (recombine).

A policy is an ordered decision list; each rule = a CONJUNCTION of typed predicates (feat op thr) + an action. The first
rule whose predicates ALL hold fires; else the default action (ABSTAIN). Conjunctions are needed because a profitable
policy on this market is inherently conjunctive ("provide when benign AND under capacity"; "unwind when inventory high
AND near the horizon") — a single-predicate list cannot express it. This is the born-legible form (a decision list =
the C-4 story tree flattened). The anchor is always-ABSTAIN, so the policy CANNOT improve without add_rule.
"""
from __future__ import annotations
import numpy as np

ACTIONS = {"PROVIDE": 0, "ABSTAIN": 1, "AGGRESS": 2}
_ANAMES = ("PROV", "ABST", "AGGR")


def rule(conds, action):
    """conds = list of (feat, op, thr); feat in {belief,inv,tail}; op in {<,>=}. action int."""
    return {"conds": [(f, o, float(t)) for (f, o, t) in conds], "action": int(action)}


def _match(r, belief, inv, tail):
    vals = {"belief": belief, "inv": inv, "tail": tail}
    for f, o, t in r["conds"]:
        x = vals[f]
        if (o == "<" and not x < t) or (o == ">=" and not x >= t):
            return False
    return True


def act(rules, belief, inv, t, T, I_max):
    """Fire the action of the first rule whose conjunction all holds; else ABSTAIN."""
    tail = (T - t)
    for r in rules:
        if _match(r, belief, inv, tail):
            return r["action"]
    return ACTIONS["ABSTAIN"]


def default_policy():
    """Return the detuned anchor: a never-matching rule, so the policy always ABSTAINs until add_rule grows it."""
    return [rule([("belief", ">=", 999.0)], ACTIONS["ABSTAIN"])]   # never matches -> always ABSTAIN (detuned anchor)


def evaluate(rules, env_ctor, seeds):
    """Return per-seed episodic returns of the rule-list policy over `seeds`."""
    env = env_ctor(); rets = []
    for s in seeds:
        obs, _ = env.reset(seed=s); done = False; R = 0.0
        while not done:
            a = act(rules, float(env.belief), int(env.inv[0]), env.t, env.T, env.m.I_max)
            obs, r, term, trunc, _ = env.step([a]); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


def action_signature(rules, env_ctor, probe_states):
    """The policy's action on a fixed probe grid = its behavioral fingerprint (for blast-radius measurement)."""
    env = env_ctor(); T, I_max = env.T, env.m.I_max
    return np.array([act(rules, b, inv, t, T, I_max) for (b, inv, t) in probe_states], dtype=int)


def frozen_probe_states(env_ctor, n_belief=13):
    """Build the fixed (belief, inventory, tail) probe grid used to fingerprint a policy."""
    env = env_ctor(); states = []
    for b in np.linspace(0.02, 0.98, n_belief):
        for inv in range(env.m.I_max + 1):
            for t in (2, 6, 10, 14, 18):
                states.append((float(b), int(inv), int(t)))
    return states


def is_real(r):
    """True unless r is the detuned never-matching anchor rule (i.e. a real, non-anchor clause)."""
    return not (len(r["conds"]) == 1 and r["conds"][0][2] >= 900)


# ---- structural operators ----
def add_rule(rules, r, pos=0):
    """Structural operator: insert a copy of rule r into the decision list at position pos."""
    out = [dict(x) for x in rules]; out.insert(max(0, min(pos, len(out))), dict(r)); return out


def retune_rule(rules, idx, cond_i, new_thr):
    """Retune operator: set the threshold of condition cond_i in rule idx to new_thr (copy-on-write)."""
    out = [dict(x) for x in rules]
    if 0 <= idx < len(out) and 0 <= cond_i < len(out[idx]["conds"]):
        c = list(out[idx]["conds"]); f, o, _ = c[cond_i]; c[cond_i] = (f, o, float(new_thr)); out[idx]["conds"] = c
    return out


def recombine(parent_a, parent_b, cut=None):
    """Single-point crossover on the ordered decision lists (A[:cut] + B[cut:]), de-duplicated preserving priority."""
    ca = cut if cut is not None else len(parent_a) // 2
    child = list(parent_a[:ca]) + list(parent_b[ca:])
    seen, out = set(), []
    for r in child:
        k = (tuple(r["conds"]), r["action"])
        if k in seen:
            continue
        seen.add(k); out.append(dict(r))
    return out if out else default_policy()


def describe(rules):
    """Render the real (non-anchor) rules as a compact legible decision-list string."""
    def one(r):
        c = " & ".join(f"{f}{o}{t:.2f}" for f, o, t in r["conds"])
        return f"if {c}->{_ANAMES[r['action']]}"
    return " | ".join(one(r) for r in rules if is_real(r)) or "ABSTAIN"
