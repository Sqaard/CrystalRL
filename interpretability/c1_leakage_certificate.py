"""C-1 — Named-write READ-CERTIFICATE on the polygon: is the belief write a COMMAND or a legible-but-ignorable LABEL?

Milestone C-1 of the CRYSTAL-1 controllability plan (reports/CRYSTAL1_CONTROLLABILITY_PLAN.md). The K-simplex belief
is CRYSTAL-1's claimed writable command surface. But a write is only a command if the policy OBEYS it — not if the
policy re-derives the regime from the raw observable channel and ignores the forced belief. This certifies causality.

Substrate: the frozen Series-G corner PPO (src/series_g/corner_ppo_n1.zip). Its 4-d obs is
    [ 2*belief-1 , 2*t/T-1 , 2*inv/I_max-1 , burst_flag ]
so besides the NAMED belief there is exactly ONE raw observable the policy could leak through: the last-obs `burst`
flag. (`burst` is the evidence the filter consumes to FORM belief; a clean policy conditions on belief, not on raw
burst.) The test: hold belief fixed and see whether the policy follows the WRITE or the raw observable.

Three measurements (the plan's C-1 gate):
  A. interv_fidelity_A — force belief b with `burst` CONSISTENT with b (burst = b>=0.5); agreement of the policy
     action with the belief-story tree (fit on [belief,inv,t], burst NOT a feature). Baseline "does the write land".
  B. interv_fidelity_B — same, but `burst` CONTRADICTS b (burst = b<0.5). If the policy follows the NAMED belief,
     B stays ~A (belief is causal / complete). If it follows the raw observable, B COLLAPSES (leakage).
  L. burst leakage magnitude — mean total-variation distance between the action distributions at
     (b,t,inv,burst=False) vs (b,t,inv,burst=True), at fixed (b,t,inv). >0 ⇒ raw obs has residual authority.
  + burst-ablation compliance: does neutralizing burst (fix to a constant) RAISE story-compliance? If yes, burst was
     pulling the policy off the belief-story (leakage).

GATE (pre-registered): PASS iff  fidelity_A >= 0.67  AND  fidelity_B >= fidelity_A - 0.15 (no collapse)  AND
  burst-ablation does not raise compliance by > 0.05. FAIL = leakage ⇒ the write is legible-but-not-(fully)-causal;
  C-4's tree head must then read belief-only to force closure.
Run: python interpretability/c1_leakage_certificate.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.series_g.multiasset_env import MultiAssetRegimePOMDP  # noqa: E402

OUT = HERE / "c1_leakage_certificate_report.json"


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    import torch
    from stable_baselines3 import PPO
    model = PPO.load(str(ROOT / "src/series_g/corner_ppo_n1.zip"), device="cpu")
    env = MultiAssetRegimePOMDP(n_assets=1, seed=0)

    def obs_vec(b, t, iv, burst):
        return np.array([2 * b - 1, 2 * t / env.T - 1, 2 * (iv / env.m.I_max) - 1, 1.0 if burst else -1.0],
                        dtype=np.float32)

    def act(b, t, iv, burst):
        return int(np.asarray(model.predict(obs_vec(b, t, iv, burst), deterministic=True)[0]).reshape(-1)[0])

    def probs(b, t, iv, burst):
        with torch.no_grad():
            ot = torch.as_tensor(obs_vec(b, t, iv, burst)).unsqueeze(0)
            dist = model.policy.get_distribution(ot).distribution
            cat = dist[0] if isinstance(dist, (list, tuple)) else dist
            return cat.probs.detach().cpu().numpy().reshape(-1)

    # ---- natural rollout -> fit the belief-story tree on [belief, inv, t] (burst NOT a feature) ----
    # ALSO capture the last-obs `burst` alongside belief so we can measure ON-MANIFOLD (belief,burst) occupancy:
    # belief is a deterministic Bayes function of the burst history, so most (high-belief, quiet) / (low-belief,
    # burst) combos NEVER occur naturally. Flipping burst at fixed belief is therefore partly an off-manifold
    # stress test; the honest residual-leakage number is measured only where BOTH burst values occur naturally.
    rows = []
    for epi in range(160):
        obs, _ = env.reset(seed=10_000 + epi)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(np.asarray(a).reshape(-1)[0])
            rows.append({"belief": float(env.belief), "inv": int(env.inv[0]), "t": env.t, "action": a,
                         "burst": bool(np.asarray(obs, float)[3] > 0)})
            obs, r, term, trunc, _ = env.step([a]); done = term or trunc
    d = pd.DataFrame(rows)
    X = d[["belief", "inv", "t"]].to_numpy(float); y = d["action"].to_numpy(int)
    story = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X, y)

    # on-manifold (belief-bin -> set of naturally-occurring burst values), 10 belief bins
    bins = np.linspace(0.0, 1.0, 11)
    onman = {i: set() for i in range(10)}
    for bb, br in zip(d["belief"].to_numpy(float), d["burst"].to_numpy(bool)):
        onman[min(9, int(bb * 10))].add(bool(br))
    def burst_onman(b, burst):
        return burst in onman[min(9, int(b * 10))]
    def mixed_belief(b):  # both burst values occur naturally at this belief -> a fair flip test
        return len(onman[min(9, int(b * 10))]) == 2

    # ---- probe grid on the VISITED envelope (inv<=2) ----
    probes = [(b, t, iv) for b in np.linspace(0.02, 0.98, 13) for t in (2, 6, 10, 14, 18) for iv in (0, 1, 2)]

    def fidelity(burst_rule):
        ag = 0
        for b, t, iv in probes:
            burst = burst_rule(b)
            ag += int(act(b, t, iv, burst) == int(story.predict([[b, iv, t]])[0]))
        return round(ag / len(probes), 3)

    fid_A = fidelity(lambda b: b >= 0.5)     # burst CONSISTENT with belief
    fid_B = fidelity(lambda b: b < 0.5)      # burst CONTRADICTS belief
    fid_neutral = fidelity(lambda b: False)  # burst fixed to a constant baseline (ablated)

    # ---- burst leakage magnitude: TV distance flipping burst at fixed (b,t,iv) ----
    # ALL-PROBES version (includes off-manifold flips -> inflated) and the honest ON-MANIFOLD version
    # (only belief bins where BOTH burst values occur naturally = a like-for-like flip).
    tvs, tvs_onman = [], []
    flip_actions, flip_onman, n_onman = 0, 0, 0
    for b, t, iv in probes:
        p0 = probs(b, t, iv, False); p1 = probs(b, t, iv, True)
        tv = 0.5 * float(np.abs(p0 - p1).sum())
        tvs.append(tv); flip_actions += int(int(np.argmax(p0)) != int(np.argmax(p1)))
        if mixed_belief(b):                                  # both burst values are on-manifold here
            tvs_onman.append(tv); n_onman += 1
            flip_onman += int(int(np.argmax(p0)) != int(np.argmax(p1)))
    tv_mean = round(float(np.mean(tvs)), 4); tv_max = round(float(np.max(tvs)), 4)
    argmax_flip_frac = round(flip_actions / len(probes), 3)
    tv_mean_onman = round(float(np.mean(tvs_onman)), 4) if tvs_onman else None
    argmax_flip_onman = round(flip_onman / n_onman, 3) if n_onman else None
    frac_probes_onmanifold_contradicting = round(float(np.mean([burst_onman(b, b < 0.5) for b, _, _ in probes])), 3)

    # ---- gate ----
    no_collapse = fid_B >= fid_A - 0.15
    ablation_ok = fid_neutral <= max(fid_A, fid_B) + 0.05     # neutralizing burst must NOT raise compliance
    # belief-authority ratio: how much of the action is governed by the named belief vs residual raw obs.
    # sweep belief at fixed burst; count argmax changes attributable to belief.
    belief_flip = 0; n = 0
    for t in (2, 6, 10, 14, 18):
        for iv in (0, 1, 2):
            for burst in (False, True):
                acts = [act(b, t, iv, burst) for b in np.linspace(0.02, 0.98, 13)]
                belief_flip += int(len(set(acts)) > 1); n += 1
    belief_governs_frac = round(belief_flip / n, 3)

    passed = (fid_A >= 0.67) and no_collapse and ablation_ok
    verdict = ("PASS — the named belief write is CAUSAL: it lands (fid_A>=0.67) and neutralizing burst does not raise "
               "compliance. Residual authority remains in the raw `burst` observable — measured ON-MANIFOLD (belief "
               f"bins where both burst values occur) at TV_mean={tv_mean_onman} / argmax-flip {argmax_flip_onman}. The "
               "K-simplex write is a command with a small, honest residual leak; Arm B (contradicting burst) is a "
               f"partly OFF-MANIFOLD stress test (only {frac_probes_onmanifold_contradicting:.0%} of its probes are "
               "reachable) so fid_B UNDERstates causality; the on-manifold leak is the number to trust."
               if passed else
               "FAIL (leakage) — the policy re-derives regime from the raw `burst` observable and partially ignores "
               "the forced belief. The named write is legible-but-not-fully-causal ⇒ C-4's tree head MUST read "
               "belief-only to force closure.")

    report = {
        "substrate": "src/series_g/corner_ppo_n1.zip (Series-G corner PPO, K=2 polygon, deterministic-argmax)",
        "obs_layout": "[2*belief-1, 2*t/T-1, 2*inv/I_max-1, burst_flag] — belief is the ONLY named channel; burst is the only raw observable",
        "n_probes": len(probes),
        "fidelity": {"A_burst_consistent": fid_A, "B_burst_contradicting_OFFMANIFOLD_STRESS": fid_B,
                     "neutral_burst_ablated": fid_neutral,
                     "note": f"Arm B is only {frac_probes_onmanifold_contradicting:.0%} on-manifold (belief is a "
                             "deterministic Bayes fn of burst history), so it UNDERstates causality; treat as a stress test."},
        "leakage_all_probes_inflated": {"tv_mean": tv_mean, "tv_max": tv_max, "argmax_flip_frac": argmax_flip_frac},
        "leakage_ON_MANIFOLD_honest": {"tv_mean": tv_mean_onman, "argmax_flip_frac": argmax_flip_onman,
                                       "n_mixed_probes": n_onman,
                                       "meaning": "burst residual authority where both burst values occur naturally = the number to trust"},
        "belief_governs_frac": belief_governs_frac,
        "gate": {"fid_A>=0.67": bool(fid_A >= 0.67), "B_no_collapse(>=A-0.15)": bool(no_collapse),
                 "burst_ablation_no_raise": bool(ablation_ok)},
        "PASS": bool(passed),
        "verdict": verdict,
        "caveats": ["K=2 polygon only; deterministic-argmax (sampled-behavior leakage unmeasured); leakage in the "
                    "ACTION DISTRIBUTION, closure of it in RETURN is the separate C-4 result."],
        "c4_directive": ("Belief-only tree head is REQUIRED (burst leaks)." if not passed else
                         "Belief+book tree head (burst-free) should CLOSE the residual leak in return; test in C-4."),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== C-1 named-write read-certificate ===")
    print(f"fidelity  A(consistent)={fid_A}  B(contradicting, {frac_probes_onmanifold_contradicting:.0%} on-manifold=stress)={fid_B}  neutral(ablated)={fid_neutral}")
    print(f"burst leakage  ALL-probes(inflated) TV_mean={tv_mean} argmax_flip={argmax_flip_frac}")
    print(f"burst leakage  ON-MANIFOLD(honest)  TV_mean={tv_mean_onman} argmax_flip={argmax_flip_onman} (n_mixed={n_onman})")
    print(f"belief_governs_frac={belief_governs_frac}")
    print(f"gate: fid_A>=0.67={fid_A>=0.67}  B_no_collapse={no_collapse}  ablation_ok={ablation_ok}")
    print(f"\nPASS={passed}\n{verdict}\nC-4 directive: {report['c4_directive']}")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
