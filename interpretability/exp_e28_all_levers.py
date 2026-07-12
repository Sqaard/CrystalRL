"""E-28 — HL loops with ALL influence levers on the REAL PPO model (now that one exists).

The three levers Ivan named — now runnable because E-27 built a genuine PPO head:
  LEVER R (reward):        penalty coefficients / budgets — the model learns to price behaviors differently.
                           Pre-specified candidate grid (multiplicity disclosed), retrain per proposal,
                           SELECTION ON DEV ONLY, the winner + base read hold ONCE with full diagnostics.
  LEVER T (pretrain/teachers): warm-start from teacher trajectories — the teacher is OUR OWN E-15 certified
                           rule (leak-free by construction). Arms: BC-only / BC->PPO fine-tune / cold PPO.
  LEVER I (interventions): promote / suppress / replace a latent primitive — the CRYSTAL-1 command surface:
                           (a) belief-WRITE fidelity sweep (the do-intervention on the K-simplex);
                           (b) leaf-logit promote/suppress of the most-defensive leaf (the behavior-code
                               shift, ±2 nats); (c) a "+1 nat promote-defense" intervention as a CANDIDATE
                               through full-hold diagnostics — can an intervention be CERTIFIED as a move?
  LEVER A (architecture):  tree depth 2 vs 3 — legibility (4 vs 8 leaves) against performance.

Discipline: training/selection on train/dev; hold read once per pre-specified comparison; OOS untouched.

Run: python interpretability/exp_e28_all_levers.py     (needs E-27's models; ~30-40 min CPU)
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
    get_streams, train_head, rollout, evaluate_head, ExposureEnv, LEVELS, MODELS, COST,
)
from interpretability.hl_v4_over_crystal1 import ann_dd, risk_boot_z  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402

OUT = HERE / "exp_e28_all_levers_report.json"
CERT = {"t1": 0.30, "t2": 0.657, "lvl_reduced": 1.0, "lvl_defensive": 0.738, "H": 10}
NI_MARGIN = 0.0002


def dev_score(model, streams, budget):
    ro, bl, rf = streams["dev"]
    pnl, exs = rollout(model, ro, bl, rf)
    a, d = ann_dd(pnl)
    return float(a - 3.0 * max(0.0, -d - budget)), {"dev_ann": round(a, 4), "dev_maxDD": round(d, 4)}


def hold_diag(pnl, anchor):
    z_dsd, gain = risk_boot_z(pnl, anchor, block=20, n_boot=1000, seed=28)
    d = pnl - anchor; _, se = block_z(d, block=5, n_boot=1000, seed=28)
    ni = (float(d.mean()) + NI_MARGIN) / se
    a, dd = ann_dd(pnl)
    return {"ann": round(a, 4), "maxDD": round(dd, 4), "z_dsd_vs_ref": round(z_dsd, 2), "ni_z": round(float(ni), 2),
            "dsd_gain_bp": round(gain * 1e4, 2)}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-28 — ALL levers on the real PPO ===")
    streams, _ = get_streams()
    ro_h, bl_h, rf_h = streams["hold"]
    bh = ro_h.copy()
    base = PPO.load(MODELS / "dd08.zip")
    base_pnl, base_exs = rollout(base, ro_h, bl_h, rf_h)

    # ---------------- LEVER R: reward coefficients --------------------------------------------------
    print("--- LEVER R (reward): pre-specified grid, retrain per proposal, dev-only selection ---")
    grid = [{"budget": 0.08, "lam": 1.0}, {"budget": 0.08, "lam": 4.0}, {"budget": 0.08, "lam": 8.0},
            {"budget": 0.06, "lam": 2.0}, {"budget": 0.10, "lam": 2.0}]
    r_cands = []
    for i, cfg in enumerate(grid):
        m, secs = train_head(f"e28R_{i}", cfg, streams, seed=0)
        s, dv = dev_score(m, streams, 0.08)
        r_cands.append({"cfg": cfg, "dev_score": round(s, 4), **dv, "model": f"e28R_{i}"})
        print(f"  R{i} {cfg} -> dev score {s:+.4f} ({dv})")
    s_base, dv_base = dev_score(base, streams, 0.08)
    print(f"  base(lam2,b08) dev score {s_base:+.4f} ({dv_base})")
    winner = max(r_cands, key=lambda c: c["dev_score"])
    lever_R = {"grid_disclosed": grid, "base_dev_score": round(s_base, 4), "candidates": r_cands,
               "winner": winner["cfg"], "hold_read": None,
               "verdict": None}
    if winner["dev_score"] > s_base:
        mw = PPO.load(MODELS / f"{winner['model']}.zip")
        pw, _ = rollout(mw, ro_h, bl_h, rf_h)
        lever_R["hold_read"] = {"winner_vs_base_hold": hold_diag(pw, base_pnl),
                                "winner_vs_bh": hold_diag(pw, bh), "base_vs_bh": hold_diag(base_pnl, bh)}
        lever_R["verdict"] = "reward shaping MOVES the policy (winner beat base on dev; hold read shown)"
    else:
        lever_R["verdict"] = "base reward already dev-optimal in the grid (no hold read spent)"
    print(f"  LEVER R verdict: {lever_R['verdict']}")

    # ---------------- LEVER T: teacher warm-start ---------------------------------------------------
    print("--- LEVER T (pretrain/teachers): BC from the certified rule -> PPO fine-tune vs cold ---")
    ro_t, bl_t, rf_t = streams["train"]
    # teacher trajectory: simulate the certified rule on train, record (obs, action_level)
    ex, eq, peak = 1.0, 1.0, 1.0
    obs_l, act_l = [], []
    for t in range(len(ro_t)):
        dd = eq / peak - 1.0
        if t % CERT["H"] == 0:
            b = bl_t[t]
            tgt = 1.0 if b < CERT["t1"] else (CERT["lvl_reduced"] if b < CERT["t2"] else CERT["lvl_defensive"])
        obs_l.append([bl_t[t], ex, dd]); act_l.append(int(np.argmin(np.abs(LEVELS - tgt))))
        p = tgt * ro_t[t] + (1 - tgt) * rf_t[t] - abs(tgt - ex) * COST
        ex = tgt; eq *= (1 + p); peak = max(peak, eq)
    X = torch.tensor(np.array(obs_l), dtype=torch.float32)
    y = torch.tensor(np.array(act_l), dtype=torch.long)
    # BC: train a fresh policy's tree supervised
    env = ExposureEnv(ro_t, bl_t, rf_t, budget=0.08, lam=2.0, seed=0)
    from src.crystal.soft_tree_policy import SoftTreeActorCriticPolicy
    bc_model = PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=0,
                   policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0, "critic_arch": (32, 32)})
    opt = torch.optim.Adam(bc_model.policy.tree.parameters(), lr=1e-2)
    for ep in range(400):
        logits = bc_model.policy.tree(X)
        loss = torch.nn.functional.nll_loss(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    bc_acc = float((bc_model.policy.tree(X).argmax(1) == y).float().mean())
    pnl_bc, _ = rollout(bc_model, ro_h, bl_h, rf_h)
    warm, _ = train_head("e28T_warm", {"budget": 0.08, "lam": 2.0}, streams, seed=0,
                         init_from=bc_model.policy.state_dict())
    pnl_warm, _ = rollout(warm, ro_h, bl_h, rf_h)
    lever_T = {"teacher": "the E-15 certified rule (leak-free: our own certified artifact)",
               "bc_train_accuracy": round(bc_acc, 3),
               "hold": {"BC_only": hold_diag(pnl_bc, bh), "BC_then_PPO": hold_diag(pnl_warm, bh),
                         "cold_PPO(base)": hold_diag(base_pnl, bh)}}
    dsd = lambda p: float(np.sqrt((np.minimum(p, 0) ** 2).mean())) * 1e4
    lever_T["verdict"] = (f"BC acc {bc_acc:.0%}; hold dsd: BC {dsd(pnl_bc):.1f} / warm {dsd(pnl_warm):.1f} / "
                          f"cold {dsd(base_pnl):.1f} bp — "
                          + ("warm-start retains teacher structure" if dsd(pnl_warm) < dsd(base_pnl)
                             else "PPO washes the teacher out (reward dominates init)"))
    print(f"  LEVER T: {lever_T['verdict']}")

    # ---------------- LEVER I: interventions --------------------------------------------------------
    print("--- LEVER I (interventions): belief-writes + promote/suppress the defensive leaf ---")
    # (a) belief-write fidelity: monotone exposure response to written bear-prob
    rng = np.random.default_rng(28)
    probes = rng.integers(10, len(ro_h) - 1, 60)
    mono = 0
    for t in probes:
        exs_resp = []
        for delta in (0.0, 0.2, 0.4):
            obs = np.array([[min(1.0, bl_h[t] + delta), 0.75, -0.03]], dtype=np.float32)
            a, _ = base.predict(obs, deterministic=True)
            exs_resp.append(float(LEVELS[int(np.asarray(a).reshape(-1)[0])]))
        # a flat response is NOT fidelity: the exposure must actually move down as the written
        # bear-prob rises (plain >= counted 0.75,0.75,0.75 as a "monotone response" — E-27c fix)
        if exs_resp[0] >= exs_resp[1] >= exs_resp[2] and exs_resp[0] > exs_resp[2]:
            mono += 1
    fidelity = mono / len(probes)
    # (b) promote / suppress the most-defensive leaf (behavior-code shift)
    tree = base.policy.tree
    with torch.no_grad():
        leaf_probs = torch.softmax(tree.leaf_logits, dim=-1).numpy()
    def_leaf = int(np.argmin((leaf_probs * LEVELS[None, :]).sum(1)))   # leaf with lowest expected exposure
    # promote/suppress the DEFENSIVE PRIMITIVE = bias the defensive ACTION columns across all leaves
    # ("the model softly shifts toward another behavior-code"):
    def with_action_bias(nats):
        m2 = PPO.load(MODELS / "dd08.zip")
        with torch.no_grad():
            m2.policy.tree.leaf_logits[:, 0] += nats * 0.5             # action 0 = exposure 0.0
            m2.policy.tree.leaf_logits[:, 1] += nats * 0.5             # action 1 = exposure 0.25
        return m2
    shifts = {}
    for nm, nats in (("promote_defense_+2", +2.0), ("suppress_defense_-2", -2.0)):
        m2 = with_action_bias(+2.0 if "+2" in nm else -2.0)
        p2, e2 = rollout(m2, ro_h, bl_h, rf_h)
        shifts[nm] = {"mean_exposure": round(float(e2.mean()), 3), "hold": hold_diag(p2, bh)}
        print(f"  {nm}: ex_bar {shifts[nm]['mean_exposure']} ann {shifts[nm]['hold']['ann']:+.2%} "
              f"maxDD {shifts[nm]['hold']['maxDD']:.2%}")
    # (c) a certified intervention move: +1 nat promote vs the unmodified head
    m1 = with_action_bias(+1.0)
    p1, _ = rollout(m1, ro_h, bl_h, rf_h)
    interv_move = hold_diag(p1, base_pnl)
    certifies = interv_move["z_dsd_vs_ref"] > 2.15 and interv_move["ni_z"] > 2.15
    lever_I = {"belief_write_fidelity_monotone": round(fidelity, 3), "defensive_leaf": def_leaf,
               "behavior_code_shifts": shifts,
               "certified_intervention_candidate(+1nat_promote)": {**interv_move, "certifies_full_hold": bool(certifies)}}
    print(f"  belief-write fidelity {fidelity:.0%}; +1nat promote vs base: z_dsd {interv_move['z_dsd_vs_ref']:+.2f} "
          f"ni {interv_move['ni_z']:+.2f} -> certifies={certifies}")

    # ---------------- LEVER A: architecture ---------------------------------------------------------
    print("--- LEVER A (architecture): tree depth 2 vs 3 ---")
    m_d2, _ = train_head("e28A_depth2", {"budget": 0.08, "lam": 2.0}, streams, seed=0, tree_depth=2)
    p_d2, _ = rollout(m_d2, ro_h, bl_h, rf_h)
    lever_A = {"depth2_hold": hold_diag(p_d2, bh), "depth3_hold": hold_diag(base_pnl, bh),
               "note": "depth-2 = 4 leaves (more legible); depth-3 = 8 leaves"}
    print(f"  depth2 ann {lever_A['depth2_hold']['ann']:+.2%} DD {lever_A['depth2_hold']['maxDD']:.2%} | "
          f"depth3 ann {lever_A['depth3_hold']['ann']:+.2%} DD {lever_A['depth3_hold']['maxDD']:.2%}")

    rep = {"experiment": "E-28 all influence levers on the real PPO (reward / teachers / interventions / architecture)",
           "lever_R_reward": lever_R, "lever_T_teachers": lever_T, "lever_I_interventions": lever_I,
           "lever_A_architecture": lever_A}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
