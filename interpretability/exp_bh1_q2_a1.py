"""BH1-Q2 arm A1 — does oversampling regime-switch neighborhoods let cold PPO discover belief-use?

PRE-REGISTERED in interpretability/exp_bh1_q2_a1_PREREG.md (committed 2026-08-18, BEFORE this
file existed). Do not change W, seeds, steps, probe, or the Supported/Kill thresholds after a run.
If this report JSON exists without that pre-reg commit as an ancestor, the run is void.

Question: cold PPO never used the belief (stages 1–2). Warm-start installs it. ~3% of days are
regime switches (P_STAY=0.97) — the only days a belief pays. A1 oversamples ±W neighborhoods of
those switches in TRAIN only. Eval stays the unaltered Markov stream.

Primary: contrast_write_probe (NOT the strict fixed-delta probe).
Ceiling: this machine's BC-oracle contrast-write F = 0.833 (not the committed 0.85).

Run: python interpretability/exp_bh1_q2_a1.py     (~10-20 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.crystal_ppo import rollout  # noqa: E402
from interpretability.exp_bh1_pressure import gen_market, behavioral_voi, N_DAYS  # noqa: E402
from interpretability.exp_bh1_pressure2 import (  # noqa: E402
    train, run_policy, oracle_seq, bc_oracle_teacher, BUDGET, LAM,
)
from interpretability.exp_bh1_pressure3 import contrast_write_probe  # noqa: E402
from interpretability.crystal_ppo import ExposureEnv, SoftTreeActorCriticPolicy  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

OUT = HERE / "exp_bh1_q2_a1_report.json"
W = 5
STEPS = 30_000
ENT = 0.005
SEEDS = (0, 1, 2)
C = 2.0
LOCKED_ORACLE_F = 0.833
SUPPORTED_GAP = 0.40
SUPPORTED_F = 0.50
SUPPORTED_N_SEEDS = 2
NULL_F = 0.20
COLD_BASELINE_MAX = 0.20  # if mean F_cold exceeds this, baseline moved → INCONCLUSIVE


def switch_indices(z):
    return np.where(np.diff(z) != 0)[0] + 1


def switch_neighborhood_frac(z, W=W):
    n = len(z)
    sw = switch_indices(z)
    if len(sw) == 0:
        return 0.0
    mask = np.zeros(n, dtype=bool)
    for t in sw:
        mask[max(0, int(t) - W): min(n, int(t) + W + 1)] = True
    return float(mask.mean())


def oversample_transitions(ro, bl, rf, z, W=W, n=N_DAYS, rng=None):
    """Concatenate ±W windows around switches, with replacement, until length n."""
    if rng is None:
        rng = np.random.default_rng(0)
    sw = switch_indices(z)
    if len(sw) == 0:
        raise RuntimeError("oversample_transitions: no regime switches in the source stream")
    pieces_ro, pieces_bl, pieces_rf, pieces_z = [], [], [], []
    length = 0
    while length < n:
        t = int(rng.choice(sw))
        lo, hi = max(0, t - W), min(len(z), t + W + 1)
        pieces_ro.append(ro[lo:hi]); pieces_bl.append(bl[lo:hi])
        pieces_rf.append(rf[lo:hi]); pieces_z.append(z[lo:hi])
        length += hi - lo
    ro_o = np.concatenate(pieces_ro)[:n]
    bl_o = np.concatenate(pieces_bl)[:n]
    rf_o = np.concatenate(pieces_rf)[:n]
    z_o = np.concatenate(pieces_z)[:n]
    return (ro_o, bl_o, rf_o), z_o


def local_oracle_f(seed=0):
    """Reproduce the stage-3 BC-oracle contrast-write number on this machine."""
    tr, _ = gen_market(C, seed)
    ev, _ = gen_market(C, seed + 500)
    streams = {"train": tr, "dev": ev, "hold": ev}
    sd, acc = bc_oracle_teacher(streams, seed=seed)
    env = ExposureEnv(*streams["train"], budget=BUDGET, lam=LAM, seed=seed)
    m_bc = PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=seed,
               policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": 3, "beta": 1.0,
                              "critic_arch": (32, 32)})
    m_bc.policy.load_state_dict(sd)
    return round(float(contrast_write_probe(m_bc)), 3), round(float(acc), 3)


def eval_head(model, ev, z_ev):
    f = float(contrast_write_probe(model))
    e_bull, e_bear = behavioral_voi(model, ev, z_ev)
    _, exs = rollout(model, *ev)
    perf = run_policy(ev[0], ev[2], exs)
    orc = run_policy(ev[0], ev[2], oracle_seq(ev[1]))
    return {
        "f": round(f, 3),
        "e_bull": round(float(e_bull), 3),
        "e_bear": round(float(e_bear), 3),
        "regime_gap": round(float(e_bull - e_bear), 3),
        "ann": perf["ann"],
        "maxDD": perf["maxDD"],
        "mean_rew": perf["mean_rew"],
        "oracle_ann": orc["ann"],
        "oracle_maxDD": orc["maxDD"],
        "oracle_mean_rew": orc["mean_rew"],
    }


def apply_prereg(f_cold, f_a1):
    """Locked thresholds from exp_bh1_q2_a1_PREREG.md. No post-hoc rescue."""
    mean_cold = float(np.mean(f_cold))
    mean_a1 = float(np.mean(f_a1))
    gap = mean_a1 - mean_cold
    n_clear = int(sum(f >= SUPPORTED_F for f in f_a1))
    all_null = all(f <= NULL_F for f in f_a1)
    if mean_cold > COLD_BASELINE_MAX:
        return "INCONCLUSIVE", (
            f"INCONCLUSIVE: cold baseline moved (mean F_cold {mean_cold:.3f} > {COLD_BASELINE_MAX}) "
            "— A1 is not interpretable against a shifted null"
        )
    if all_null:
        return "NULL", (
            f"NULL: all three A1 seeds F ≤ {NULL_F:.2f} "
            f"(F_A1={list(f_a1)}, mean {mean_a1:.3f}; F_cold mean {mean_cold:.3f}). "
            "Transition rarity is NOT why cold RL fails."
        )
    if gap >= SUPPORTED_GAP and n_clear >= SUPPORTED_N_SEEDS:
        return "SUPPORTED", (
            f"SUPPORTED: mean(F_A1)−mean(F_cold)={gap:.3f} ≥ {SUPPORTED_GAP} and "
            f"{n_clear}/3 seeds F_A1 ≥ {SUPPORTED_F} "
            f"(F_A1={list(f_a1)}, F_cold={list(f_cold)}). "
            "Discovery is possible when switches are common — not that cold RL finds "
            "belief-use on the original market."
        )
    return "INCONCLUSIVE", (
        f"INCONCLUSIVE: gap {gap:.3f} (need ≥ {SUPPORTED_GAP}) and "
        f"{n_clear}/3 seeds ≥ {SUPPORTED_F} (need ≥ {SUPPORTED_N_SEEDS}). "
        f"F_A1={list(f_a1)}, F_cold={list(f_cold)}. No extra seeds, no new W, no metric swap."
    )


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    import stable_baselines3 as sb3
    print("=== BH1-Q2 A1 — transition oversampling, cold PPO, c=2, contrast-write F ===")
    print(f"  pre-reg: interpretability/exp_bh1_q2_a1_PREREG.md | W={W} steps={STEPS} seeds={list(SEEDS)}")

    oracle_f, oracle_acc = local_oracle_f(0)
    print(f"  local BC-oracle contrast-write F {oracle_f} (locked ceiling {LOCKED_ORACLE_F}; acc {oracle_acc})")

    per_seed = []
    for seed in SEEDS:
        tr, z_tr = gen_market(C, seed)
        ev, z_ev = gen_market(C, seed + 500)
        rng = np.random.default_rng(seed * 100003 + 17)
        tr_a1, z_a1 = oversample_transitions(*tr, z_tr, W=W, n=N_DAYS, rng=rng)
        frac_tr = switch_neighborhood_frac(z_tr, W)
        frac_a1 = switch_neighborhood_frac(z_a1, W)
        frac_ev = switch_neighborhood_frac(z_ev, W)
        if frac_a1 < 0.50:
            raise RuntimeError(
                f"A1 train switch-neighborhood fraction {frac_a1:.3f} < 0.50 at seed {seed}; abort"
            )
        print(f"  seed {seed}: neighborhood frac train {frac_tr:.3f} | A1-train {frac_a1:.3f} | eval {frac_ev:.3f}")

        streams_cold = {"train": tr, "dev": ev, "hold": ev}
        streams_a1 = {"train": tr_a1, "dev": ev, "hold": ev}

        t0 = time.time()
        m_cold = train(streams_cold, seed, STEPS, ENT, init_from=None)
        cold = eval_head(m_cold, ev, z_ev)
        s_cold = int(time.time() - t0)

        t0 = time.time()
        m_a1 = train(streams_a1, seed, STEPS, ENT, init_from=None)
        a1 = eval_head(m_a1, ev, z_ev)
        s_a1 = int(time.time() - t0)

        row = {
            "seed": seed,
            "neighborhood_frac_train": round(frac_tr, 3),
            "neighborhood_frac_a1_train": round(frac_a1, 3),
            "neighborhood_frac_eval": round(frac_ev, 3),
            "n_switches_source": int(len(switch_indices(z_tr))),
            "cold": cold,
            "a1": a1,
            "s_cold": s_cold,
            "s_a1": s_a1,
        }
        per_seed.append(row)
        print(f"  seed {seed}: F_cold {cold['f']:.3f} gap_cold {cold['regime_gap']:+.3f} ({s_cold}s) | "
              f"F_A1 {a1['f']:.3f} gap_A1 {a1['regime_gap']:+.3f} ({s_a1}s)")

    f_cold = [s["cold"]["f"] for s in per_seed]
    f_a1 = [s["a1"]["f"] for s in per_seed]
    verdict_tag, verdict = apply_prereg(f_cold, f_a1)
    logbook_verdict = "CONFIRMED" if verdict_tag == "SUPPORTED" else verdict_tag

    rep = {
        "experiment": "BH1-Q2 arm A1 — transition oversampling vs cold PPO at c=2",
        "preregistration": "interpretability/exp_bh1_q2_a1_PREREG.md",
        "framing": "DESIGNED-market instrument calibration (VoI-fenced); not market evidence",
        "design": {
            "c": C, "W": W, "n_days": N_DAYS, "steps": STEPS, "ent_coef": ENT,
            "seeds": list(SEEDS), "init": "cold (init_from=None)",
            "primary": "contrast_write_probe (not fidelity_probe)",
            "eval": "unaltered gen_market(c, seed+500) for both arms",
            "locked_oracle_f": LOCKED_ORACLE_F,
            "thresholds": {
                "supported_gap": SUPPORTED_GAP,
                "supported_f": SUPPORTED_F,
                "supported_n_seeds": SUPPORTED_N_SEEDS,
                "null_f": NULL_F,
                "cold_baseline_max": COLD_BASELINE_MAX,
            },
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "stable_baselines3": sb3.__version__,
        },
        "local_oracle": {"contrast_write_f": oracle_f, "bc_acc": oracle_acc,
                         "locked_ceiling": LOCKED_ORACLE_F},
        "per_seed": per_seed,
        "summary": {
            "f_cold": f_cold,
            "f_a1": f_a1,
            "mean_f_cold": round(float(np.mean(f_cold)), 3),
            "mean_f_a1": round(float(np.mean(f_a1)), 3),
            "gap": round(float(np.mean(f_a1) - np.mean(f_cold)), 3),
            "n_a1_ge_050": int(sum(f >= SUPPORTED_F for f in f_a1)),
        },
        "prereg_verdict": verdict_tag,
        "logbook_verdict": logbook_verdict,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
