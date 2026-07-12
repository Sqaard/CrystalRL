"""HL v3 (modular substrate) — a per-VENUE rule policy on RegimeRotationEnv, where the reward DECOMPOSES per venue, so
each venue is an INDEPENDENT sub-strategy (a niche). This is the MODULAR structure recombine (crossover) is the yield
operator for: a parent that covers venues {0,1} and a parent that covers {2,3} recombine into a child covering all four
— strictly better than either. (On the non-modular corner polygon recombine was inert; here it should bite.)

Policy = a list of clauses. Clause types:
  ("unwind",)                      -> UNWIND inventory near the horizon (shared inventory management)
  ("provide", v, thr, cap)         -> if belief[v] >= thr and inv < cap, PROVIDE @ venue v (independent per-venue edge)
act() unwinds first if due, else provides at the highest-belief COVERED venue whose clause matches, else ABSTAINs.
"""
from __future__ import annotations
import numpy as np
from src.series_g.family_env import RegimeRotationEnv


def env_ctor(G=6):
    """Return a zero-arg constructor for a RegimeRotationEnv with G venues (fixed seed)."""
    def _c():
        return RegimeRotationEnv(G=G, seed=0)
    return _c


def unwind_clause():
    """Build the shared inventory-management clause: UNWIND near the horizon."""
    return ("unwind",)


def provide_clause(v, thr=0.556, cap=4):
    """Build a per-venue PROVIDE clause: provide at venue v when belief>=thr and inventory<cap."""
    return ("provide", int(v), float(thr), int(cap))


def covered_venues(policy):
    """Return the set of venues the policy has a provide clause for (its niche coverage)."""
    return frozenset(c[1] for c in policy if c[0] == "provide")


def act(policy, env):
    """Choose the env action: UNWIND if inventory is due near the horizon, else PROVIDE at the highest-belief
    covered venue whose clause matches, else ABSTAIN."""
    belief, inv, t, T, I_max = env.belief, env.inv, env.t, env.T, env.I_max
    has_unwind = any(c[0] == "unwind" for c in policy)
    if has_unwind and inv > 0 and (T - t) <= inv:
        return env.UNWIND
    best_v, best_b = None, -1.0
    for c in policy:
        if c[0] != "provide":
            continue
        _, v, thr, cap = c
        if belief[v] >= thr and inv < cap and belief[v] > best_b:
            best_v, best_b = v, float(belief[v])
    return best_v if best_v is not None else env.ABSTAIN


def evaluate(policy, ctor, seeds):
    """Return per-seed episodic returns of the modular policy over `seeds`."""
    env = ctor(); rets = []
    for s in seeds:
        env.reset(seed=s); done = False; R = 0.0
        while not done:
            a = act(policy, env)
            _, r, term, trunc, _ = env.step(a); R += float(r); done = term or trunc
        rets.append(R)
    return np.array(rets)


# ---- structural operators ----
def add_provide(policy, v, thr=0.556, cap=4):
    """Structural operator: add a provide clause for venue v unless one already covers it."""
    if any(c[0] == "provide" and c[1] == v for c in policy):
        return policy
    return list(policy) + [provide_clause(v, thr, cap)]


def recombine(a, b):
    """UNIFORM crossover over order-independent modular clauses: the child inherits the clause set of BOTH parents,
    de-duplicated per venue (keeping the more permissive / lower-threshold clause on a venue collision). For modular
    sub-strategies this is the correct recombination — it UNIONS the two parents' venue coverage, which is exactly the
    yield single-point crossover could not produce on equal-length lists. (One unwind clause is kept if either has it.)"""
    best = {}                                    # venue -> best provide clause; plus an 'unwind' sentinel
    has_unwind = False
    for c in list(a) + list(b):
        if c[0] == "unwind":
            has_unwind = True
        else:
            _, v, thr, cap = c
            if v not in best or thr < best[v][2]:    # keep the lower (more permissive) threshold per venue
                best[v] = c
    out = ([unwind_clause()] if has_unwind else []) + [best[v] for v in sorted(best)]
    return out


def describe(policy):
    """Render the policy as a compact legible string (unwind flag + covered provide venues)."""
    u = "UNWIND" if any(c[0] == "unwind" for c in policy) else ""
    vs = ",".join(str(c[1]) for c in policy if c[0] == "provide")
    return f"[{u}{' | ' if u and vs else ''}provide@{{{vs}}}]"
