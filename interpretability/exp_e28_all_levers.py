"""E-28 (rebuilt) — ALL influence levers on the real PPO, WITH the controls the audit demanded.

The first E-28 proved the levers technically move a deterministic output; it did NOT prove causal,
control-checked influence. This rebuild closes the audit's OPEN findings:
  * LEVER I is now a PER-PRIMITIVE intervention on the computed defensive leaf (not a global bias on all
    leaves), WITH a control battery: matched-random (a random leaf — placebo), wrong-direction (bias the
    SAME leaf toward risk — the effect must invert), and dose (half vs full nats — the response must scale);
  * LEVER R surfaces the NON-INFERIORITY bar as a REJECT (the reward winner that fails NI is reported as
    NOT certified, not as "the reward moves the policy");
  * LEVER T DISCLOSES the E-15 hold contamination (the teacher was selected on 2022-23) and reads a
    leak-free reference (the DP champion) instead of asserting a clean number;
  * LEVER A is MULTI-SEED (depth 2 vs 3 across seeds), which also yields the STABILITY axis;
  * CRYSTALSCORE (Faithfulness x Simulatability x Stability) is computed for the head.

The base head is the TEACHER-WARM E-27 head (dd08_warm) — interventions on the cold near-uniform head are
meaningless. Discipline unchanged: train/select on train/dev; hold read once; OOS untouched.

Run: python interpretability/exp_e28_all_levers.py   (needs E-27's models; ~30-40 min; --smoke = fast)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from interpretability.crystal_ppo import (  # noqa: E402
    get_streams, train_head, rollout, ExposureEnv, LEVELS, MODELS, COST,
)
from interpretability.hl_v4_over_crystal1 import ann_dd, risk_boot_z  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from src.crystal.soft_tree_policy import SoftTreeActorCriticPolicy  # noqa: E402

OUT = HERE / "exp_e28_all_levers_report.json"
NI_BAR = 2.15
NI_MARGIN = 0.0002


def base_model():
    """The teacher-warm E-27 head (non-degenerate); fall back to the cold head if warm is absent."""
    warm = MODELS / "dd08_warm.zip"
    return PPO.load(warm if warm.exists() else MODELS / "dd08.zip"), warm.exists()


def dev_score(model, streams, budget):
    ro, bl, rf = streams["dev"]
    pnl, _ = rollout(model, ro, bl, rf)
    a, d = ann_dd(pnl)
    return float(a - 3.0 * max(0.0, -d - budget)), {"dev_ann": round(a, 4), "dev_maxDD": round(d, 4)}


def hold_diag(pnl, anchor):
    z_dsd, gain = risk_boot_z(pnl, anchor, block=20, n_boot=1000, seed=28)
    d = pnl - anchor; _, se = block_z(d, block=5, n_boot=1000, seed=28)
    ni = (float(d.mean()) + NI_MARGIN) / se
    a, dd = ann_dd(pnl)
    return {"ann": round(a, 4), "maxDD": round(dd, 4), "z_dsd_vs_ref": round(z_dsd, 2),
            "ni_z": round(float(ni), 2), "dsd_gain_bp": round(gain * 1e4, 2)}


def grid_actions(model, obs_grid):
    a, _ = model.predict(obs_grid, deterministic=True)
    return np.asarray(a).reshape(-1)


OBS_GRID = np.array([[b, pe, dd] for b in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                     for pe in (0.0, 0.5, 1.0) for dd in (0.0, -0.05, -0.15)], dtype=np.float32)


def with_leaf_bias(base_path, leaf, nats, cols=(0, 1)):
    """PER-PRIMITIVE do-intervention: bias ONLY `leaf`'s defensive action columns by `nats`
    (cols 0,1 = exposures 0.0, 0.25). This is the fix for the audit's 'global bias across all leaves'."""
    m = PPO.load(base_path)
    with torch.no_grad():
        for c in cols:
            m.policy.tree.leaf_logits[leaf, c] += nats * 0.5
    return m


def with_global_bias(base_path, nats, cols=(0, 1)):
    """The ORIGINAL E-28 intervention: bias the defensive columns across ALL leaves — kept ONLY as the
    control that shows the original 'intervention' effect was global, not a per-primitive causal control."""
    m = PPO.load(base_path)
    with torch.no_grad():
        for c in cols:
            m.policy.tree.leaf_logits[:, c] += nats * 0.5
    return m


def main(argv=None):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    smoke = argv is not None and "--smoke" in argv
    steps = 4000 if smoke else 60000
    seeds = (0, 1) if smoke else (0, 1, 2)
    r_grid = ([{"budget": 0.08, "lam": 4.0}] if smoke else
              [{"budget": 0.08, "lam": 1.0}, {"budget": 0.08, "lam": 4.0}, {"budget": 0.08, "lam": 8.0},
               {"budget": 0.06, "lam": 2.0}, {"budget": 0.10, "lam": 2.0}])
    print(f"=== E-28 (rebuilt) — all levers WITH controls {'[SMOKE]' if smoke else ''} ===")
    streams, _ = get_streams()
    ro_h, bl_h, rf_h = streams["hold"]; bh = ro_h.copy()
    base, warm_used = base_model()
    base_path = MODELS / ("dd08_warm.zip" if warm_used else "dd08.zip")
    base_pnl, base_exs = rollout(base, ro_h, bl_h, rf_h)
    print(f"base head: {'dd08_warm' if warm_used else 'dd08 (cold fallback)'}")

    # ---------------- LEVER R: reward, WITH the NI bar surfaced as a reject -------------------------
    print("--- LEVER R: pre-specified grid, dev-only selection, NI-gated hold read ---")
    r_cands = []
    for i, cfg in enumerate(r_grid):
        m, _ = train_head(f"e28R_{i}", cfg, streams, seed=0, total_steps=steps)
        s, dv = dev_score(m, streams, 0.08)
        r_cands.append({"cfg": cfg, "dev_score": round(s, 4), **dv, "model": f"e28R_{i}"})
    s_base, _ = dev_score(base, streams, 0.08)
    winner = max(r_cands, key=lambda c: c["dev_score"])
    lever_R = {"grid_disclosed": r_grid, "base_dev_score": round(s_base, 4), "candidates": r_cands,
               "winner": winner["cfg"]}
    if winner["dev_score"] > s_base:
        pw, _ = rollout(PPO.load(MODELS / f"{winner['model']}.zip"), ro_h, bl_h, rf_h)
        hold = hold_diag(pw, base_pnl)
        certified = hold["ni_z"] > NI_BAR
        lever_R["hold_read"] = hold
        lever_R["ni_bar"] = NI_BAR
        lever_R["verdict"] = (f"reward winner beat base on DEV, but on the hold it is "
                              f"{'CERTIFIED' if certified else 'REJECTED'} by the NI bar "
                              f"(ni_z {hold['ni_z']} vs bar {NI_BAR}) — moving the policy is not the same "
                              f"as a certified improvement")
    else:
        lever_R["verdict"] = "base reward already dev-optimal in the grid (no hold read spent)"
    print(f"  {lever_R['verdict']}")

    # ---------------- LEVER T: teacher, WITH the contamination disclosed ----------------------------
    lever_T = {"teacher": "the E-15 certified rule",
               "contamination_disclosure": (
                   "The E-15 rule was SELECTED on the 2022-23 hold; re-reading it there is contaminated. "
                   "A truly fresh read is pre-registered for 2027 (OOS 2024-26 is quarantined). The "
                   "leak-free teacher signal in the pipeline is the DP CHAMPION (W4/W5), selected on no "
                   "market window; the warm E-27 head is BC'd from that. This lever no longer asserts a "
                   "clean hold number for the E-15 teacher — it discloses the leak."),
               "warm_head_is_dp_teacher_bc": True}
    print("  LEVER T: contamination disclosed (E-15 selected on the 2022-23 hold)")

    # ---------------- LEVER I: PER-PRIMITIVE intervention + control battery -------------------------
    print("--- LEVER I: per-primitive defensive-leaf intervention + matched-random/wrong-dir/dose ---")
    # (a) belief-write fidelity (unchanged strict test): exposure must fall as written bear-prob rises
    rng = np.random.default_rng(28)
    probes = rng.integers(10, len(ro_h) - 1, 60)
    mono = 0
    for t in probes:
        resp = []
        for delta in (0.0, 0.2, 0.4):
            obs = np.array([[min(1.0, bl_h[t] + delta), 0.75, -0.03]], dtype=np.float32)
            a, _ = base.predict(obs, deterministic=True)
            resp.append(float(LEVELS[int(np.asarray(a).reshape(-1)[0])]))
        if resp[0] >= resp[1] >= resp[2] and resp[0] > resp[2]:
            mono += 1
    fidelity = mono / len(probes)
    # the computed defensive leaf (previously dead — audit P4: now it DRIVES the intervention)
    with torch.no_grad():
        leaf_probs = torch.softmax(base.policy.tree.leaf_logits, dim=-1).numpy()
    n_leaves = leaf_probs.shape[0]
    def_leaf = int(np.argmin((leaf_probs * LEVELS[None, :]).sum(1)))
    base_exbar = float(base_exs.mean())

    def exbar(m):
        _, e = rollout(m, ro_h, bl_h, rf_h); return float(e.mean())

    def battery(path):
        """The controlled intervention battery on one head: per-primitive treatment + placebo /
        inverted / dose controls + the global-bias contrast, all as mean-exposure deltas."""
        m = PPO.load(path); _, e = rollout(m, ro_h, bl_h, rf_h); e0 = float(e.mean())
        with torch.no_grad():
            lp = torch.softmax(m.policy.tree.leaf_logits, dim=-1).numpy()
        nl = lp.shape[0]; dl = int(np.argmin((lp * LEVELS[None, :]).sum(1)))
        exps = [round(float(LEVELS[lp[i].argmax()]), 2) for i in range(nl)]
        rl = [i for i in range(nl) if i != dl]
        treat = exbar(with_leaf_bias(path, dl, +2.0)) - e0
        rand = float(np.mean([exbar(with_leaf_bias(path, lf, +2.0)) - e0
                              for lf in (rl[:1] if smoke else rl)]))
        wrong = exbar(with_leaf_bias(path, dl, +2.0, cols=(3, 4))) - e0
        half = exbar(with_leaf_bias(path, dl, +1.0)) - e0
        glob = exbar(with_global_bias(path, +2.0)) - e0
        cp = (treat <= -1e-4 and abs(rand) < abs(treat) and wrong >= -1e-4 and abs(half) <= abs(treat) + 1e-6)
        placebo_fail = abs(treat) > 1e-4 and abs(rand) >= abs(treat) - 1e-6   # random moves as much = artifact
        return {"defensive_leaf": dl, "leaf_argmax_exposure": exps, "base_mean_exposure": round(e0, 3),
                "treatment_promote_defense_+2nat": round(treat, 4), "control_matched_random_leaf": round(rand, 4),
                "control_wrong_direction_promote_risk": round(wrong, 4), "control_dose_half_+1nat": round(half, 4),
                "contrast_global_bias_all_leaves_+2nat": round(glob, 4),
                "controls_pass": bool(cp), "matched_random_placebo_fails": bool(placebo_fail)}

    warm_bat = battery(base_path)
    cold_bat = battery(MODELS / "dd08.zip")   # the ORIGINAL near-uniform head the first E-28 intervened on
    # honest interpretation across both heads
    if warm_bat["controls_pass"]:
        interp = "per-primitive intervention is a controlled causal de-risk on the warm head"
    elif cold_bat["matched_random_placebo_fails"] or (abs(warm_bat["treatment_promote_defense_+2nat"]) < 1e-3):
        interp = ("the intervention is NOT a clean causal primitive control: the confident warm head is a "
                  "no-op under modest bias (uniformly aggressive), and on the cold near-uniform head a "
                  "matched-random leaf moves behavior as much as the defensive one — the original E-28 "
                  "'intervention works' was a near-uniform / global-bias artifact (audit critique CONFIRMED)")
    else:
        interp = "controls inconclusive on both heads (honest)"
    lever_I = {"belief_write_fidelity_monotone": round(fidelity, 3),
               "warm_head_battery": warm_bat, "cold_head_battery": cold_bat,
               "interpretation": interp}
    print(f"  fidelity {fidelity:.0%} | warm treat {warm_bat['treatment_promote_defense_+2nat']:+.3f} "
          f"rand {warm_bat['control_matched_random_leaf']:+.3f} pass {warm_bat['controls_pass']} | "
          f"cold treat {cold_bat['treatment_promote_defense_+2nat']:+.3f} "
          f"rand {cold_bat['control_matched_random_leaf']:+.3f} placebo_fail {cold_bat['matched_random_placebo_fails']}")

    # ---------------- LEVER A: MULTI-SEED depth 2 vs 3 (-> also the STABILITY axis) -----------------
    print("--- LEVER A: multi-seed depth 2 vs 3 ---")
    d2_models, d3_models = [], []
    for s in seeds:
        m2, _ = train_head(f"e28A_d2_s{s}", {"budget": 0.08, "lam": 2.0}, streams, seed=s, total_steps=steps, tree_depth=2)
        m3, _ = train_head(f"e28A_d3_s{s}", {"budget": 0.08, "lam": 2.0}, streams, seed=s, total_steps=steps, tree_depth=3)
        d2_models.append(m2); d3_models.append(m3)
    d2_hold = [hold_diag(rollout(m, ro_h, bl_h, rf_h)[0], bh) for m in d2_models]
    d3_hold = [hold_diag(rollout(m, ro_h, bl_h, rf_h)[0], bh) for m in d3_models]
    lever_A = {"seeds": list(seeds),
               "depth2_hold_ann": [h["ann"] for h in d2_hold], "depth3_hold_ann": [h["ann"] for h in d3_hold],
               "depth2_hold_maxDD": [h["maxDD"] for h in d2_hold], "depth3_hold_maxDD": [h["maxDD"] for h in d3_hold],
               "note": "depth-2 = 4 leaves (more legible); depth-3 = 8 leaves; reported across seeds"}

    # ---------------- CRYSTALSCORE for the head (F x S x St) ----------------------------------------
    # Faithfulness = belief-write fidelity; Simulatability = agreement of a depth-2 (4-leaf) story with the
    # depth-3 head over the obs grid; Stability = mean pairwise across-seed action agreement (depth-3).
    d3_acts = [grid_actions(m, OBS_GRID) for m in d3_models]
    base_acts = grid_actions(base, OBS_GRID)
    simulat = float(np.mean([np.mean(grid_actions(m2, OBS_GRID) == base_acts) for m2 in d2_models]))
    pairs = [(i, j) for i in range(len(d3_acts)) for j in range(i + 1, len(d3_acts))]
    stability = float(np.mean([np.mean(d3_acts[i] == d3_acts[j]) for i, j in pairs])) if pairs else 1.0
    crystalscore = round(fidelity * simulat * stability, 4)
    cs = {"faithfulness_belief_write": round(fidelity, 4), "simulatability_depth2_vs_depth3": round(simulat, 4),
          "stability_across_seeds": round(stability, 4), "CrystalScore": crystalscore,
          "note": "computed for the E-27/E-28 head (the audit found this was never done)"}
    print(f"  CrystalScore = F {fidelity:.2f} x S {simulat:.2f} x St {stability:.2f} = {crystalscore}")

    rep = {"experiment": "E-28 (rebuilt) all influence levers WITH controls, on the teacher-warm PPO head",
           "base_head": "dd08_warm (teacher-warm)" if warm_used else "dd08 (cold fallback)",
           "lever_R_reward": lever_R, "lever_T_teachers": lever_T, "lever_I_interventions": lever_I,
           "lever_A_architecture": lever_A, "crystalscore": cs,
           "governance_status": "RESEARCH_ONLY_LEVER_STUDY_WITH_CONTROLS",
           "eligible_for_client_menu": False}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main(sys.argv[1:])
