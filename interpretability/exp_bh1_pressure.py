"""BH1 — the Pressure Hypothesis: does belief->action causality emerge under distribution-shift pressure?

Theory (Richens & Everitt, ICLR 2024): agents need (and therefore learn to USE) a causal world model
only when the environment exerts shift pressure; with none, competent-looking policies may ignore
their beliefs. Our honest null — belief-write fidelity 3% on the real-panel reward-dial head — is
predicted by this: the daily substrate's optimum is a constant dial (the regime is priced), so
nothing forces the head to read P(bear).

Design (a DESIGNED-market instrument calibration, per the north-star VoI fence — none of this is
market evidence): a 2-regime Markov return process with KNOWN parameters; the belief fed to the
agent is the TRUE Bayes-filter posterior. The pressure dial is the regime CONTRAST
c = (mu_bull - mu_bear) / sigma: at c=0 the regimes are identical (belief worthless by
construction); as c grows, ignoring the belief costs return/drawdown. Same architecture and
training as E-27 (SoftTreeActorCriticPolicy depth-3 on ExposureEnv, dd08-style reward), 2 seeds per
pressure level.

Measures per (level, seed), all pre-registered:
  * belief-write FIDELITY — the strict monotone test from E-28 (exposure must FALL as the written
    bear-prob rises; flat responses fail);
  * a PLACEBO control — the same test with writes to a non-belief obs coordinate (prev-exposure):
    fidelity must NOT rise there (else the test itself is confounded);
  * behavioral VoI — mean-exposure gap between bear and bull states (does behavior condition on
    regime at all).

PREREGISTERED READ: BH1 is SUPPORTED if fidelity increases materially and monotonically-ish with c
(low at c=0, high at the top contrast, Spearman > 0 across levels) while the placebo stays flat.
KILL: fidelity stays ~ at its c=0 level at the highest contrast.

Run: python interpretability/exp_bh1_pressure.py     (~10-15 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.crystal_ppo import train_head, LEVELS  # noqa: E402

OUT = HERE / "exp_bh1_pressure_report.json"
SIGMA = 0.012                      # daily vol, both regimes (pressure = mean contrast only)
P_STAY = 0.97                      # sticky regimes
RF_D = 0.02 / 252
N_DAYS = 2500
CONTRASTS = (0.0, 0.5, 1.0, 2.0)   # c = (mu_bull - mu_bear)/sigma; mu_bull = +c/2*sigma, mu_bear = -c/2*sigma
SEEDS = (0, 1)
STEPS = 30_000


def gen_market(c, seed, n=N_DAYS):
    """2-regime Markov returns + the TRUE Bayes-filter posterior P(bear) (the agent's belief feed)."""
    rng = np.random.default_rng(seed * 1009 + int(c * 100))
    mu_b, mu_g = -c / 2 * SIGMA, +c / 2 * SIGMA
    z = np.empty(n, dtype=int); z[0] = 0
    for t in range(1, n):
        z[t] = z[t - 1] if rng.random() < P_STAY else 1 - z[t - 1]
    ro = np.where(z == 1, mu_b, mu_g) + SIGMA * rng.standard_normal(n)
    # exact Bayes filter under the true parameters
    b = np.empty(n); bel = 0.5
    for t in range(n):
        pr = P_STAY * bel + (1 - P_STAY) * (1 - bel)                     # prior of bear
        lb = np.exp(-0.5 * ((ro[t] - mu_b) / SIGMA) ** 2)
        lg = np.exp(-0.5 * ((ro[t] - mu_g) / SIGMA) ** 2)
        bel = lb * pr / (lb * pr + lg * (1 - pr) + 1e-300)
        b[t] = bel
    rf = np.full(n, RF_D)
    return (ro, b, rf), z


def fidelity_probe(model, bl, strict=True, n_probes=60, seed=28, placebo=False):
    """The strict E-28 monotone belief-write test; placebo=True writes to prev-exposure instead."""
    rng = np.random.default_rng(seed)
    probes = rng.integers(10, len(bl) - 1, n_probes)
    mono = 0
    for t in probes:
        resp = []
        for delta in (0.0, 0.2, 0.4):
            if placebo:
                obs = np.array([[bl[t], min(1.0, 0.75 + delta), -0.03]], dtype=np.float32)
            else:
                obs = np.array([[min(1.0, bl[t] + delta), 0.75, -0.03]], dtype=np.float32)
            a, _ = model.predict(obs, deterministic=True)
            resp.append(float(LEVELS[int(np.asarray(a).reshape(-1)[0])]))
        ok = resp[0] >= resp[1] >= resp[2] and resp[0] > resp[2]
        mono += int(ok)
    return mono / n_probes


def behavioral_voi(model, streams, z):
    """Mean exposure in bull vs bear TRUE states on a fresh rollout — does behavior condition on regime?"""
    from interpretability.crystal_ppo import rollout
    ro, bl, rf = streams
    _, exs = rollout(model, ro, bl, rf)
    zz = z[: len(exs)]
    e_bull = float(exs[zz == 0].mean()) if (zz == 0).any() else float("nan")
    e_bear = float(exs[zz == 1].mean()) if (zz == 1).any() else float("nan")
    return e_bull, e_bear


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== BH1 — the Pressure Hypothesis: fidelity(contrast) on the designed regime market ===")
    levels = {}
    for c in CONTRASTS:
        per_seed = []
        for seed in SEEDS:
            train, z_tr = gen_market(c, seed)
            evalm, z_ev = gen_market(c, seed + 500)
            streams = {"train": train, "dev": evalm, "hold": evalm}
            t0 = time.time()
            model, _ = train_head(f"bh1_c{int(c*10):02d}_s{seed}",
                                  {"budget": 0.08, "lam": 2.0}, streams, seed=seed,
                                  total_steps=STEPS)
            fid = fidelity_probe(model, evalm[1])
            plc = fidelity_probe(model, evalm[1], placebo=True)
            e_bull, e_bear = behavioral_voi(model, evalm, z_ev)
            per_seed.append({"seed": seed, "fidelity": round(fid, 3), "placebo": round(plc, 3),
                             "e_bull": round(e_bull, 3), "e_bear": round(e_bear, 3),
                             "regime_gap": round(e_bull - e_bear, 3),
                             "train_s": int(time.time() - t0)})
            print(f"  c={c:3.1f} seed {seed}: fidelity {fid:.2f} placebo {plc:.2f} "
                  f"exposure bull/bear {e_bull:.2f}/{e_bear:.2f} gap {e_bull-e_bear:+.2f} ({per_seed[-1]['train_s']}s)")
        levels[str(c)] = {"per_seed": per_seed,
                          "fidelity_mean": round(float(np.mean([s["fidelity"] for s in per_seed])), 3),
                          "placebo_mean": round(float(np.mean([s["placebo"] for s in per_seed])), 3),
                          "regime_gap_mean": round(float(np.mean([s["regime_gap"] for s in per_seed])), 3)}

    f_by_c = [levels[str(c)]["fidelity_mean"] for c in CONTRASTS]
    p_by_c = [levels[str(c)]["placebo_mean"] for c in CONTRASTS]
    # Spearman across the 4 levels (small n — descriptive, pre-registered as the direction check)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(CONTRASTS, f_by_c)
    supported = (f_by_c[-1] >= f_by_c[0] + 0.30) and (rho > 0) and (max(p_by_c) <= f_by_c[-1] - 0.20)
    killed = f_by_c[-1] <= f_by_c[0] + 0.10
    verdict = ("SUPPORTED: belief->action causality emerges with shift pressure "
               f"(fidelity {f_by_c[0]:.2f}@c=0 -> {f_by_c[-1]:.2f}@c={CONTRASTS[-1]}, placebo flat) — "
               "the real-panel 3% fidelity is a PRESSURE diagnosis, not an architecture defect"
               if supported else
               ("KILLED: fidelity does not rise with pressure — the weak causality is NOT explained "
                "by missing shift pressure (architecture/training must be suspected)" if killed else
                "INCONCLUSIVE: partial rise — needs more seeds/levels before a verdict"))
    rep = {"experiment": "BH1 Pressure Hypothesis — fidelity(contrast) on a designed 2-regime market",
           "framing": "DESIGNED-market instrument calibration (VoI-fenced); not market evidence",
           "theory": "Richens & Everitt (ICLR 2024): causal world-model use requires shift pressure",
           "design": {"sigma": SIGMA, "p_stay": P_STAY, "contrasts": list(CONTRASTS), "seeds": list(SEEDS),
                       "steps": STEPS, "belief": "TRUE Bayes posterior under known params",
                       "fidelity_test": "strict E-28 monotone write test", "placebo": "write to prev-exposure"},
           "levels": levels, "spearman_fidelity_vs_contrast": round(float(rho), 3),
           "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
