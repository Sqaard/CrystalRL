"""BH1 stage 2 — is the stage-1 KILL a true pressure result or a cold-PPO optimization collapse?

Stage 1 (exp_bh1_pressure.py) read fidelity 0.00 at EVERY contrast, including c=2 where the regimes
differ by 2 sigma of daily mean and the belief is a near-perfect Bayes filter. Every head converged
to a CONSTANT dial (one seed to all-cash IN A BULL-HEAVY MARKET), which is the E-27 signature of
cold-PPO collapse, not evidence about pressure. Before logging a verdict we must separate:

  (H-opt)  the environment DOES reward belief-use but cold PPO cannot find it in 30k steps
           -> an optimization failure; stage 1 says nothing about the Pressure Hypothesis itself;
  (H-null) even the oracle policy barely beats the best constant dial under this reward
           -> the designed pressure was too weak; stage 1's KILL was reading a flat objective.

Arms (all at the TOP contrast c=2.0, the only level where stage 1's kill has teeth):
  A. VALIDITY (numpy, seconds): belief AUC vs the true state per contrast; the oracle policy
     (bear->0, bull->1, executed on the belief) vs every constant dial under the SAME dd08 reward
     and under raw return/maxDD. If oracle >> best dial, pressure is real -> H-opt.
  B. TEACHER WARM-START (the E-27c cure, 2 seeds): BC the oracle rule into the SoftTree, PPO
     fine-tune 30k, then the SAME strict fidelity probe + placebo + regime gap as stage 1.
     If fidelity stays high, belief-use SURVIVES optimization when the initialization finds it.
  C. COLD BUT BIGGER (2 seeds): 90k steps + ent_coef 0.02 (more exploration). Does compute alone
     rescue the cold head?

Run: python interpretability/exp_bh1_pressure2.py     (~10 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO  # noqa: E402
from interpretability.crystal_ppo import (ExposureEnv, SoftTreeActorCriticPolicy, LEVELS, COST)  # noqa: E402
from interpretability.exp_bh1_pressure import gen_market, fidelity_probe, behavioral_voi, SIGMA, RF_D, CONTRASTS  # noqa: E402

OUT = HERE / "exp_bh1_pressure2_report.json"
C_TOP = 2.0
BUDGET, LAM = 0.08, 2.0


def run_policy(ro, rf, ex_seq):
    """Equity + dd08-style reward stream for a given exposure sequence."""
    eq, peak, ex_prev = 1.0, 1.0, 1.0
    rews = np.empty(len(ro))
    for t in range(len(ro)):
        ex = ex_seq[t]
        p = ex * ro[t] + (1 - ex) * rf[t] - abs(ex - ex_prev) * COST
        eq *= (1 + p); peak = max(peak, eq)
        dd = eq / peak - 1.0
        rews[t] = p - LAM * max(0.0, -dd - BUDGET)
        ex_prev = ex
    n = len(ro)
    ann = eq ** (252 / n) - 1
    # max drawdown
    eqs, e, pk, mdd = [], 1.0, 1.0, 0.0
    ex_prev = 1.0
    for t in range(n):
        p = ex_seq[t] * ro[t] + (1 - ex_seq[t]) * rf[t] - abs(ex_seq[t] - ex_prev) * COST
        e *= (1 + p); pk = max(pk, e); mdd = min(mdd, e / pk - 1); ex_prev = ex_seq[t]
    return {"ann": round(ann, 4), "maxDD": round(mdd, 4), "mean_rew": round(float(rews.mean() * 1e4), 3)}


def oracle_seq(bl):
    return np.where(bl > 0.5, 0.0, 1.0)


def bc_oracle_teacher(streams, seed=0, epochs=400):
    """BC the oracle rule (P(bear)>0.5 -> 0.0 else 1.0) into a fresh SoftTree; mirrors build_teacher."""
    ro, bl, rf = streams["train"]
    ex, eq, peak, obs_l, act_l = 1.0, 1.0, 1.0, [], []
    for t in range(len(ro)):
        dd = eq / peak - 1.0
        tgt = 0.0 if bl[t] > 0.5 else 1.0
        obs_l.append([bl[t], ex, dd]); act_l.append(int(np.argmin(np.abs(LEVELS - tgt))))
        p = tgt * ro[t] + (1 - tgt) * rf[t] - abs(tgt - ex) * COST
        ex = tgt; eq *= (1 + p); peak = max(peak, eq)
    X = torch.tensor(np.array(obs_l), dtype=torch.float32)
    y = torch.tensor(np.array(act_l), dtype=torch.long)
    env = ExposureEnv(ro, bl, rf, budget=BUDGET, lam=LAM, seed=seed)
    m = PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=seed,
            policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0, "critic_arch": (32, 32)})
    opt = torch.optim.Adam(m.policy.tree.parameters(), lr=1e-2)
    for _ in range(epochs):
        loss = torch.nn.functional.nll_loss(m.policy.tree(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((m.policy.tree(X).argmax(1) == y).float().mean())
    return m.policy.state_dict(), acc


def train(streams, seed, steps, ent, init_from=None):
    ro, bl, rf = streams["train"]
    env = ExposureEnv(ro, bl, rf, budget=BUDGET, lam=LAM, seed=seed)
    model = PPO(SoftTreeActorCriticPolicy, env, n_steps=2048, batch_size=256, n_epochs=6,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=ent,
                seed=seed, device="cpu", verbose=0,
                policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0, "critic_arch": (32, 32)})
    if init_from is not None:
        model.policy.load_state_dict(init_from)
    model.learn(total_timesteps=steps, progress_bar=False)
    return model


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== BH1 stage 2 — collapse vs pressure: validity, warm-start, bigger-cold ===")

    # ---- Arm A: validity ----
    arm_a = {}
    for c in CONTRASTS:
        (ro, bl, rf), z = gen_market(c, 700)
        if z.min() == z.max():
            auc = float("nan")
        else:
            order = np.argsort(bl); ranks = np.empty(len(bl)); ranks[order] = np.arange(len(bl))
            pos = ranks[z == 1]; auc = float((pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * (z == 0).sum()))
        dials = {str(d): run_policy(ro, rf, np.full(len(ro), d)) for d in LEVELS}
        orc = run_policy(ro, rf, oracle_seq(bl))
        best_dial = max(dials.items(), key=lambda kv: kv[1]["mean_rew"])
        arm_a[str(c)] = {"belief_auc": round(auc, 3), "oracle": orc,
                         "best_dial": {"level": best_dial[0], **best_dial[1]},
                         "oracle_edge_rew": round(orc["mean_rew"] - best_dial[1]["mean_rew"], 3)}
        print(f"  A c={c}: AUC {auc:.3f} | oracle rew {orc['mean_rew']} ann {orc['ann']:+.1%} DD {orc['maxDD']:.1%}"
              f" | best dial {best_dial[0]} rew {best_dial[1]['mean_rew']} ann {best_dial[1]['ann']:+.1%}"
              f" | edge {arm_a[str(c)]['oracle_edge_rew']}")

    # ---- Arms B & C at c = C_TOP ----
    arm_b, arm_c = [], []
    for seed in (0, 1):
        tr, _ = gen_market(C_TOP, seed)
        ev, z_ev = gen_market(C_TOP, seed + 500)
        streams = {"train": tr, "dev": ev, "hold": ev}
        t0 = time.time()
        sd, acc = bc_oracle_teacher(streams, seed=seed)
        mw = train(streams, seed, 30_000, 0.005, init_from=sd)
        fid = fidelity_probe(mw, ev[1]); plc = fidelity_probe(mw, ev[1], placebo=True)
        eb, er = behavioral_voi(mw, ev, z_ev)
        arm_b.append({"seed": seed, "bc_acc": round(acc, 3), "fidelity": round(fid, 3),
                      "placebo": round(plc, 3), "regime_gap": round(eb - er, 3),
                      "perf": run_policy(ev[0], ev[2], _rollout_ex(mw, ev)), "s": int(time.time() - t0)})
        print(f"  B seed {seed}: bc_acc {acc:.2f} fidelity {fid:.2f} placebo {plc:.2f} gap {eb-er:+.2f} ({arm_b[-1]['s']}s)")
        t0 = time.time()
        mc = train(streams, seed, 90_000, 0.02)
        fid = fidelity_probe(mc, ev[1]); plc = fidelity_probe(mc, ev[1], placebo=True)
        eb, er = behavioral_voi(mc, ev, z_ev)
        arm_c.append({"seed": seed, "fidelity": round(fid, 3), "placebo": round(plc, 3),
                      "regime_gap": round(eb - er, 3),
                      "perf": run_policy(ev[0], ev[2], _rollout_ex(mc, ev)), "s": int(time.time() - t0)})
        print(f"  C seed {seed}: fidelity {fid:.2f} placebo {plc:.2f} gap {eb-er:+.2f} ({arm_c[-1]['s']}s)")

    edge_top = arm_a[str(C_TOP)]["oracle_edge_rew"]
    fb = float(np.mean([s["fidelity"] for s in arm_b])); fc = float(np.mean([s["fidelity"] for s in arm_c]))
    gb = float(np.mean([s["regime_gap"] for s in arm_b]))
    pressure_real = edge_top > 0.5              # oracle beats the best dial by >0.05bp/step of shaped reward
    if pressure_real and fb >= 0.5 and fc < 0.3:
        verdict = ("OPTIMIZATION FAILURE, NOT A PRESSURE NULL: the designed pressure is real "
                   f"(oracle edge {edge_top}bp shaped reward/step) and belief-use SURVIVES when the teacher "
                   f"initialization finds it (warm fidelity {fb:.2f}, gap {gb:+.2f}), but cold PPO cannot "
                   f"discover it even at 90k/high-entropy (fidelity {fc:.2f}). Stage-1's flat curve measured "
                   "cold-PPO's discovery ceiling, NOT the Pressure Hypothesis. BH1 needs re-framing: "
                   "pressure is necessary but discovery is the binding constraint (the E-27c lesson again).")
    elif pressure_real and fc >= 0.5:
        verdict = ("PRESSURE WORKS WITH COMPUTE: bigger cold training discovers belief-use "
                   f"(fidelity {fc:.2f}) — stage 1 was undertrained; rerun the full curve at 90k.")
    elif not pressure_real:
        verdict = (f"DESIGN TOO WEAK: even the oracle barely beats the best dial (edge {edge_top}) — "
                   "the reward geometry, not PPO, flattened stage 1; redesign the pressure dial.")
    else:
        verdict = f"MIXED: edge {edge_top}, warm fidelity {fb:.2f}, cold-90k {fc:.2f} — needs more seeds."
    rep = {"experiment": "BH1 stage 2 — separating optimization collapse from a true pressure null",
           "arm_a_validity": arm_a, "arm_b_warm": arm_b, "arm_c_cold90k": arm_c, "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


def _rollout_ex(model, streams_ev):
    from interpretability.crystal_ppo import rollout
    ro, bl, rf = streams_ev
    _, exs = rollout(model, ro, bl, rf)
    return exs


if __name__ == "__main__":
    main()
