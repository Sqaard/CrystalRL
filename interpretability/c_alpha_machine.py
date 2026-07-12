"""Debt 3 — CALIBRATE alpha_machine: the machine-rate FALSE-ACCEPT rate of the writ-ladder's automatable certifier.

HL5 named alpha_machine (the machine-rate false-accept of the firewall) as the whole loop's top open risk, unquantified;
its only prior estimate was "the two same-session firewall retractions" (a human-rate anecdote). This measures it on the
polygon.

A WRIT CLAIM = "in belief region R the commanded write makes the policy take action a_R." The automatable certifier
(C1/C2 core) ACCEPTS a claim iff, on that region's probe set, the policy takes a_R in >= tau of probes (tau=0.67, the
C-2 compliance floor).

CRITICAL FRAMING (fixed after adversarial review): alpha_machine must measure the certifier being WRONG ABOUT THE
POLICY'S EFFECT, NOT wrong about optimality. So the ground truth is the policy's OWN modal action per region, not the
belief-MDP optimum (mixing those conflates "certifier fooled" with "policy sub-optimal"). Two DISTINCT numbers:
  [PRIMARY] alpha_machine_behavioral: null claim = a NON-MODAL action for R (a genuine mis-description of what the write
     does); alpha_machine = fraction of these mis-descriptions the certifier ACCEPTS. Genuine = the policy's modal
     action (power = how often a true behavioral claim is certified). This is the certifier's true false-accept.
  [SECONDARY] false-accept of OPTIMALITY claims (null=non-belief-MDP-optimal action): a different, wider quantity that
     ALSO fires when the policy is merely sub-optimal — reported separately, NOT called alpha_machine.
Both are C1/C2-core-only, K=2 proxies (NOT the full C1..C6 firewall false-accept HL5 ultimately means); stated as such.
Run: python interpretability/c_alpha_machine.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402
from src.series_g.phase0_gate import solve_belief_aware  # noqa: E402

OUT = HERE / "c_alpha_machine_report.json"
TAU = 0.67
N_ACTIONS = 3


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from stable_baselines3 import PPO
    env = MultiAssetRegimePOMDP(n_assets=1, seed=0)
    corner = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")
    g, V, pol = solve_belief_aware(env.m, n_bins=121)

    def obs_vec(b, t, iv, burst):
        return np.array([2 * b - 1, 2 * t / env.T - 1, 2 * (iv / env.m.I_max) - 1, 1.0 if burst else -1.0], dtype=np.float32)

    def act(model, b, t, iv, burst):
        return int(np.asarray(model.predict(obs_vec(b, t, iv, burst), deterministic=True)[0]).reshape(-1)[0])

    # regions = belief bins x inventory (the certifier certifies a claim over each region's probe set)
    belief_bins = [(lo, lo + 0.125) for lo in np.arange(0.0, 1.0, 0.125)]
    ctx_t, ctx_burst = (2, 6, 10, 14, 18), (False, True)

    def region_probe_actions(model, blo, bhi, iv):
        acts = []
        for b in np.linspace(blo + 0.01, bhi - 0.01, 5):
            for t in ctx_t:
                for burst in ctx_burst:
                    acts.append(act(model, b, t, iv, burst))
        return np.array(acts)

    def certifier_accepts(acts, claimed_action):
        return float(np.mean(acts == claimed_action)) >= TAU

    def belief_mdp_action(blo, bhi, iv):
        bmid = 0.5 * (blo + bhi)
        return int(pol[8, int(round(bmid * (len(g) - 1))), iv])     # optimum at mid-belief, mid-time

    # PRIMARY (behavioral): ground truth = the policy's OWN modal action per region.
    beh_genuine_pass, beh_genuine_tot = 0, 0
    beh_null_pass, beh_null_tot = 0, 0
    # SECONDARY (optimality): ground truth = the belief-MDP optimum (wider quantity; also fires on sub-optimality).
    opt_null_pass, opt_null_tot = 0, 0
    beh_null_accepts = []   # record which mis-descriptions slipped through (for honest inspection)
    for (blo, bhi) in belief_bins:
        for iv in (0, 1, 2):
            acts = region_probe_actions(corner, blo, bhi, iv)
            counts = np.bincount(acts, minlength=N_ACTIONS)
            a_modal = int(counts.argmax())
            modal_frac = float(counts[a_modal] / counts.sum())
            a_opt = belief_mdp_action(blo, bhi, iv)
            # --- behavioral: genuine = modal action; null = any NON-modal action (a true mis-description) ---
            beh_genuine_tot += 1; beh_genuine_pass += int(certifier_accepts(acts, a_modal))
            for a_false in range(N_ACTIONS):
                if a_false == a_modal:
                    continue
                beh_null_tot += 1
                acc = certifier_accepts(acts, a_false)
                beh_null_pass += int(acc)
                if acc:
                    beh_null_accepts.append({"region_belief": [round(blo, 3), round(bhi, 3)], "iv": iv,
                                             "claimed": a_false, "modal": a_modal, "modal_frac": round(modal_frac, 3)})
            # --- optimality (secondary): null = any non-belief-MDP-optimal action ---
            for a_false in range(N_ACTIONS):
                if a_false == a_opt:
                    continue
                opt_null_tot += 1; opt_null_pass += int(certifier_accepts(acts, a_false))

    alpha_machine = round(beh_null_pass / max(1, beh_null_tot), 4)          # PRIMARY, correctly framed
    beh_power = round(beh_genuine_pass / max(1, beh_genuine_tot), 4)
    alpha_optimality = round(opt_null_pass / max(1, opt_null_tot), 4)        # SECONDARY, wider quantity
    chance = round(1.0 / N_ACTIONS, 4)

    report = {
        "substrate": "Series-G corner PPO; certifier = C1/C2 core (policy takes claimed action in >=tau=%.2f of a region's probes)" % TAU,
        "n_regions": beh_genuine_tot,
        "alpha_machine_behavioral": alpha_machine,
        "n_behavioral_null_claims": beh_null_tot,
        "behavioral_power_on_true_modal_claims": beh_power,
        "behavioral_null_accepts": beh_null_accepts,
        "alpha_optimality_SECONDARY": alpha_optimality,
        "n_optimality_null_claims": opt_null_tot,
        "random_claim_chance": chance,
        "reading": (f"alpha_machine (behavioral, correctly framed) = {alpha_machine} = the automatable C1/C2 certifier's "
                    f"FALSE-ACCEPT rate for WRONG claims about what the write actually does (a non-modal action described "
                    f"as the write's effect passes {alpha_machine:.1%} of the time); behavioral POWER on true modal "
                    f"claims = {beh_power:.1%}. The SEPARATE, wider optimality-claim false-accept "
                    f"(null=non-belief-MDP-optimal, which ALSO fires when the policy is merely sub-optimal) = "
                    f"{alpha_optimality}. This is a K=2, C1/C2-CORE-ONLY, n={beh_null_tot} PROXY for HL5's machine-rate "
                    "false-accept (which ultimately spans the full C1..C6 firewall) — a lower bound / first estimate, "
                    "not the whole quantity. Implication unchanged: size D_cap / N_live<=K to (proposal_rate * "
                    "alpha_machine) against the human-rate anchor (the program's two same-session retractions)."),
        "caveats": ["FIXED after adversarial review: alpha_machine now measures being wrong about the POLICY'S EFFECT "
                    "(vs its own modal action), NOT wrong about optimality — the earlier 0.0208 conflated the two (its "
                    "single 'false-accept' was a TRUE behavioral claim that merely disagreed with the belief-MDP).",
                    "C1/C2 core only (C4 side-effect + C5 frozen gate would lower it further, not automated here); "
                    "K=2 polygon; concentrated deterministic policy; a PROXY for, not equal to, the full-firewall alpha_machine."],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== Debt 3 — alpha_machine calibration (behavioral, fixed) ===")
    print(f"regions={beh_genuine_tot}  behavioral null_claims={beh_null_tot}")
    print(f"alpha_machine (BEHAVIORAL false-accept: wrong about the policy's EFFECT) = {alpha_machine}")
    print(f"  behavioral power on true modal claims = {beh_power}  (chance={chance})")
    print(f"  [secondary] optimality-claim false-accept (also fires on sub-optimality) = {alpha_optimality}")
    if beh_null_accepts:
        print(f"  behavioral null-accepts ({len(beh_null_accepts)}): {beh_null_accepts}")
    print(report["reading"])
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
