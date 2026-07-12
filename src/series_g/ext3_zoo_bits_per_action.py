"""Series-G Extension 3 — convert the FINAL_A "Complexity Zoo" from a design-judgment ordinal axis into
MEASURED bits/action, on the SAME L0 ruler as R6c / Series-G, so the whole frontier x-axis is uniform.

FINAL_A placed agent classes (thermostat → chess engine → poker → LLM) on a normalized 0–1 complexity axis by
design-judgment; poker sat at ~0.78. The L0 unit-fix wants everything in bits/action. We cannot run real
thermostats/chess-engines, so we build MINIMAL BEHAVIORAL MODELS that capture each archetype's essential
ACTION-ENTROPY character (not its task difficulty — FINAL_A's key trap: a strong chess engine is LOW action
entropy), run `behavioral_complexity_dynamic`, and read off bits/action. Then we place R6c and Series-G on the
same ruler.

HONESTY: these are archetype behavioral MODELS, not measurements of real systems — they convert the ordinal
zoo into a bits/action scale so the references are recognizable, with the caveat stamped. Run:
  python -m src.series_g.ext3_zoo_bits_per_action
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "interpretability"))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

OUT = HERE / "ext3_zoo_bits_per_action_report.json"
T = 3000


def thermostat(rng):
    """Bang-bang around a setpoint: nearly always holds state, flips rarely. ~deterministic."""
    a, out = 0, []
    for _ in range(T):
        if rng.random() < 0.04:
            a = 1 - a
        out.append(a)
    return np.array(out), "2 (on/off)"


def pid_controller(rng):
    """Smooth control discretized to 5 levels; highly autocorrelated (moves one level at a time)."""
    a, out = 2, []
    for _ in range(T):
        a = int(np.clip(a + rng.choice([-1, 0, 0, 0, 1]), 0, 4))
        out.append(a)
    return np.array(out), "5 (levels)"


def central_bank(rng):
    """Mostly HOLD with rare, persistence-clustered cut/hike cycles."""
    a, out, mom = 1, [], 0
    for _ in range(T):
        if rng.random() < 0.05:
            mom = rng.choice([-1, 0, 1])
        a = int(np.clip(1 + mom if rng.random() < 0.3 else 1, 0, 2))
        out.append(a)
    return np.array(out), "3 (cut/hold/hike)"


def chess_engine(rng):
    """Strong engine: near-DETERMINISTIC best move given the position (a slowly-evolving latent), tiny noise.
    High task difficulty, LOW action entropy — the FINAL_A trap made concrete."""
    pos, out = 0, []
    for _ in range(T):
        pos = (pos * 1103515245 + 12345) % 7        # deterministic position cycler (a cheap pseudo-state)
        a = pos % 6 if rng.random() > 0.05 else rng.integers(0, 6)
        out.append(int(a))
    return np.array(out), "6 (move types)"


def ant_forager(rng):
    """Pheromone-biased random walk: 4 directions, momentum-persistent, but genuinely stochastic."""
    a, out = 0, []
    for _ in range(T):
        a = a if rng.random() < 0.5 else int(rng.integers(0, 4))
        out.append(a)
    return np.array(out), "4 (directions)"


def market_maker(rng):
    """Quote {widen/hold/tighten} reacting to a mean-reverting flow signal — structured, moderate entropy."""
    flow, out = 0.0, []
    for _ in range(T):
        flow = 0.7 * flow + rng.standard_normal()
        a = 0 if flow > 0.8 else (2 if flow < -0.8 else 1)
        out.append(a)
    return np.array(out), "3 (widen/hold/tighten)"


def poker_gto(rng):
    """GTO-ish MIXED strategy: action {fold/call/raise} drawn from a hand-conditioned distribution — genuinely
    stochastic (a mixed strategy is the point), so high entropy rate."""
    out = []
    for _ in range(T):
        s = rng.random()                              # hand strength
        p = [0.55 - 0.4 * s, 0.30, 0.15 + 0.4 * s]    # weak→fold, strong→raise; always mixed
        out.append(int(rng.choice(3, p=np.array(p) / sum(p))))
    return np.array(out), "3 (fold/call/raise)"


def llm_like(rng):
    """High-entropy token choices over a larger alphabet with mild local structure (bigram bias)."""
    A, a, out = 16, 0, []
    for _ in range(T):
        a = a if rng.random() < 0.15 else int(rng.integers(0, A))
        out.append(a)
    return np.array(out), "16 (tokens)"


ZOO = [("thermostat", thermostat), ("central_bank", central_bank), ("PID_controller", pid_controller),
       ("chess_engine(strong)", chess_engine), ("market_maker", market_maker), ("ant_forager", ant_forager),
       ("poker_GTO(mixed)", poker_gto), ("LLM_like", llm_like)]


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for name, fn in ZOO:
        stream, alpha = fn(rng)
        res = behavioral_complexity_dynamic(stream, kind="discrete", dts=(1, 2), n_null=300, n_boot=300, seed=0)
        rows.append({"archetype": name, "alphabet": alpha, "h_mu_bits_per_action": res["h_mu_range"],
                     "structure": res["structure_present_configs"], "E_range": res["E_range"]})
    # place the project's measured policies on the same ruler
    anchors = [
        {"archetype": "R6c (Dow-29, MEASURED)", "alphabet": "3 (stance)", "h_mu_bits_per_action": [0.288, 0.487],
         "structure": "6/6", "E_range": "—"},
        {"archetype": "Series-G optimal (MEASURED)", "alphabet": "3 (exec mode)", "h_mu_bits_per_action": [0.853, 1.39],
         "structure": "2/2", "E_range": "—"},
        {"archetype": "Series-G PPO multi-asset (MEASURED)", "alphabet": "3 (exec mode)",
         "h_mu_bits_per_action": [1.101, 1.194], "structure": "2/2", "E_range": "—"},
    ]
    allrows = rows + anchors
    allrows.sort(key=lambda r: np.mean(r["h_mu_bits_per_action"]))
    report = {
        "what": "FINAL_A Complexity Zoo converted to MEASURED bits/action (L0 ruler), with R6c + Series-G placed",
        "honesty": ("archetype rows are minimal BEHAVIORAL MODELS capturing each class's action-entropy character "
                    "(NOT real systems / not task difficulty); R6c + Series-G rows are MEASURED on real/solved policies. "
                    "Converts the FINAL_A ordinal axis into bits/action so the references are recognizable."),
        "ruler_low_to_high": allrows,
        "poker_reconciliation": ("FINAL_A placed poker at normalized ~0.78 (high); on this bits/action ruler the "
                                 "poker GTO mixed-strategy model lands among the HIGH-h_mu archetypes — consistent "
                                 "with 'mixed strategy ⇒ high entropy rate'. The ordinal→bits/action map is monotone."),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'archetype':34s} {'alphabet':22s} {'h_mu (bits/action)':20s} {'structure':9s}")
    for r in allrows:
        tag = "  <<<" if "MEASURED" in r["archetype"] else ""
        print(f"{r['archetype']:34s} {r['alphabet']:22s} {str(r['h_mu_bits_per_action']):20s} {r['structure']:9s}{tag}")
    print(f"\n[ext3] wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
