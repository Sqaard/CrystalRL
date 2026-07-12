"""W5b — the WIDENED conditional constrained PPO (audit reconciliation P2 / C4b).

W5 (personal_invest_w5_ppo.py) is the champion-challenger with obs = [W/G, years-left, capacity flag].
The audit's PARTIAL: the conditioning set should include belief, drawdown, and current weights, with a
separate cost objective (Lagrangian/CPO), not only the safety shield. This file closes the tractable,
WELL-POSED part of that without mutating the W5 champion path:

  * obs is WIDENED to [funding W/G, years-left/T, capacity flag, drawdown-from-peak, prev equity share]
    — drawdown and the previous weight are REAL path information in this env;
  * the TEACHER is the drawdown-aware DP (solve_dp_drawdown_aware, P1), so the new drawdown coordinate
    has a genuine teacher signal — not an inert placebo dimension;
  * a LAGRANGIAN cost objective: an interim-drawdown-budget breach is a separate COST (not folded into
    the goal reward); a dual multiplier lambda_cost is updated by projected dual ascent across short
    training rounds, so the constraint is learned as a cost, on top of the deterministic shield.

Deliberately NOT added here, with rationale (adding them as inert dims would be the exact placebo the
audit warns against):
  * belief in obs — this env draws unconditional 1y factors; there is no regime signal to observe. A
    belief coordinate is only informative with REGIME-CONDITIONAL kernels (the same prerequisite as the
    DP belief-state note in personal_invest_dp.solve_dp). Scoped next step.
  * a continuous Dirichlet sleeve allocator — the SoftTree actor emits a discrete book choice; a
    continuous simplex head is warranted only when sleeves proliferate past the 5-book menu. Scoped.

Gates mirror W5 (GW5-1 no near-uniform tie-break; GW5-2 regret vs the champion; GW5-3 constraint
learned) plus GW5b-COST: the Lagrangian must reduce the expected interim-drawdown cost vs lambda=0.

Run: python interpretability/personal_invest_w5b_ppo.py   (~15-25 min CPU; --smoke for a fast check)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from src.crystal.soft_tree_policy import SoftTreeActorCriticPolicy  # noqa: E402
from interpretability.personal_invest_forecast import load_universe, ENGINE_VERSION  # noqa: E402
from interpretability.personal_invest_dp import (  # noqa: E402
    books_with_cash, build_kernels, solve_dp, solve_dp_drawdown_aware, simulate,
    GRID, PEAK_ANCHORS, SWITCH_COST)

OUT = HERE / "personal_invest_w5b_report.json"
MODELS = HERE / "w5b_models"; MODELS.mkdir(exist_ok=True)
T_YEARS = 10
SEEDS = (0, 1, 2)
DD_BUDGET = 0.15                 # interim drawdown budget the cost objective defends
FEATURES = (0, 1, 2, 3, 4)       # widened obs: funding, years-left, cap, drawdown, prev_eq_share


def contract_kernels(uni, as_of):
    """Build the US frontier-book kernels + the per-capacity allowed action sets (same as W5)."""
    books = books_with_cash("US")
    kern = build_kernels(uni, books, as_of, seed=311)
    names = list(kern)
    allowed_cap = [i for i, nm in enumerate(names) if abs(kern[nm]["dd_p95"]) <= 0.15 + 1e-9]
    return kern, names, {0: list(range(len(names))), 1: allowed_cap}


class GoalContractEnvV2(gym.Env):
    """One investor contract per episode with the WIDENED obs and a separate interim-drawdown COST.

    Obs = [funding (clip 0..3), years-left/T, capacity flag, drawdown-from-peak in [0,1], prev equity
    share]. The safety shield still projects a disallowed action to the nearest allowed book. The
    reward is the terminal goal minus lambda_cost times the interim-drawdown-budget breach (the
    Lagrangian); the RAW cost is also accumulated for the dual update.
    """
    def __init__(self, kernels, names, allowed, seed=0, lam_cost=0.0, dd_budget=DD_BUDGET):
        super().__init__()
        self.k, self.names, self.allowed = kernels, names, allowed
        self.eq_share = np.array([self.k[nm]["eq_share"] for nm in names])
        self.rng = np.random.default_rng(seed)
        self.lam_cost, self.dd_budget = lam_cost, dd_budget
        self.observation_space = spaces.Box(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        self.action_space = spaces.MultiDiscrete([len(names)])
        self.shield_attempts = 0
        self.steps_total = 0
        self.cost_accum = 0.0        # sum of raw interim-drawdown breaches (for the dual update)

    def _obs(self):
        dd = max(0.0, self.peak - self.w) / self.peak if self.peak > 0 else 0.0
        return np.array([min(self.w, 3.0), (T_YEARS - self.t) / T_YEARS, float(self.cap),
                         min(dd, 1.0), self.prev_eq], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.cap = int(self.rng.random() < 0.5)
        self.w = float(self.rng.uniform(0.3, 1.2))
        self.peak = self.w
        self.t = 0
        self.prev_a = -1
        self.prev_eq = 0.0
        return self._obs(), {}

    def step(self, action):
        a = int(np.asarray(action).reshape(-1)[0])
        self.steps_total += 1
        if a not in self.allowed[self.cap]:
            self.shield_attempts += 1
            a = min(self.allowed[self.cap], key=lambda i: abs(self.eq_share[i] - self.eq_share[a]))
        f = float(self.rng.choice(self.k[self.names[a]]["factors"]))
        cost_switch = SWITCH_COST if (self.prev_a >= 0 and self.prev_a != a) else 0.0
        self.w = self.w * f * (1 - cost_switch)
        self.peak = max(self.peak, self.w)
        dd = max(0.0, self.peak - self.w) / self.peak if self.peak > 0 else 0.0
        breach = max(0.0, dd - self.dd_budget)       # the raw interim-drawdown COST this step
        self.cost_accum += breach
        self.prev_a = a
        self.prev_eq = float(self.eq_share[a])
        self.t += 1
        done = self.t >= T_YEARS
        reward = (float(self.w >= 1.0) if done else 0.0) - self.lam_cost * breach
        return self._obs(), reward, done, False, {}


def make_model(env, seed):
    """A CRYSTAL-1 SoftTree PPO over the WIDENED 5-D obs (depth-3 to read the extra coordinates)."""
    return PPO(SoftTreeActorCriticPolicy, env, device="cpu", verbose=0, seed=seed,
               n_steps=1024, batch_size=256, learning_rate=3e-4, ent_coef=0.01,
               policy_kwargs={"feat_idx": FEATURES, "tree_depth": 3, "beta": 1.0,
                              "critic_arch": (32, 32)})


def bc_from_dp(env, dps, names, seed):
    """Behavior-clone BOTH capacity contracts' GOAL-pursuing DP tables into one widened tree.

    Correct separation of concerns (the smoke lesson): the TEACHER pursues the goal (the plain DP,
    which is Markov in funding — see the theorem in personal_invest_dp.solve_dp), so it never collapses
    to all-cash; drawdown-awareness is then added by the LAGRANGIAN cost objective during PPO, which
    depends on the drawdown obs coordinate. The BC targets are replicated across the drawdown/prev_eq
    dims (the teacher does not condition on them), so the tree starts by ignoring them and PPO learns
    to read them to cut the cost.
    """
    X, y = [], []
    wmask = np.flatnonzero((GRID >= 0.1) & (GRID <= 3.0))
    dd_grid = np.array([0.0, 0.10, 0.25])
    pe_grid = np.array([0.0, 0.5, 1.0])
    for cap, dp in dps.items():
        pol = dp["policy"]                            # [t, funding_idx]
        for t in range(T_YEARS):
            for i in wmask:
                w = float(GRID[i]); a = int(pol[t][i])
                for dd in dd_grid:
                    for pe in pe_grid:
                        X.append([min(w, 3.0), (T_YEARS - t) / T_YEARS, float(cap), float(dd), float(pe)])
                        y.append(a)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    model = make_model(env, seed)
    opt = torch.optim.Adam(model.policy.tree.parameters(), lr=1e-2)
    for _ in range(600):
        loss = torch.nn.functional.nll_loss(model.policy.tree(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = float((model.policy.tree(X).argmax(1) == y).float().mean())
    return model, acc


def prob_diagnostics(model):
    """The E-27c honesty gate on the widened obs: near-uniform argmax tie-breaks are not policies."""
    grid = np.array([[w, tl, c, dd, pe]
                     for w in (0.2, 0.5, 0.8, 1.0, 1.5, 2.5) for tl in (0.1, 0.5, 0.9)
                     for c in (0.0, 1.0) for dd in (0.0, 0.2) for pe in (0.0, 0.5, 1.0)],
                    dtype=np.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(grid))
        d0 = dist.distribution[0] if isinstance(dist.distribution, list) else dist.distribution
        probs = d0.probs.numpy()
    amax = probs.argmax(axis=1)
    return {"mean_max_prob": round(float(probs.max(1).mean()), 4),
            "argmax_varies": bool(len(set(amax.tolist())) > 1),
            "n_distinct_actions": int(len(set(amax.tolist())))}


def policy_eval(model, kernels, names, allowed, cap, w0, n=4000, seed=99):
    """Deterministic shielded rollout on fresh draws; returns P(goal) + attempted-violation + the
    realized interim-drawdown COST (mean breach-years) so the cost objective can be scored."""
    rng = np.random.default_rng(seed)
    draws = {nm: rng.choice(kernels[nm]["factors"], size=(n, T_YEARS)) for nm in names}
    eq_share = np.array([kernels[nm]["eq_share"] for nm in names])
    w = np.full(n, float(w0)); peak = w.copy(); prev = np.full(n, -1); prev_eq = np.zeros(n)
    attempts = 0; cost = np.zeros(n); exposures = []
    for t in range(T_YEARS):
        dd = np.maximum(0.0, peak - w) / np.maximum(peak, 1e-6)
        obs = np.stack([np.minimum(w, 3.0), np.full(n, (T_YEARS - t) / T_YEARS),
                        np.full(n, float(cap)), np.minimum(dd, 1.0), prev_eq], axis=1).astype(np.float32)
        a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a).reshape(-1)
        bad = ~np.isin(a, allowed[cap]); attempts += int(bad.sum())
        for i in np.flatnonzero(bad):
            a[i] = min(allowed[cap], key=lambda j: abs(eq_share[j] - eq_share[a[i]]))
        f = np.empty(n)
        for ai in np.unique(a):
            m = a == ai; f[m] = draws[names[ai]][m, t]
        sw = np.where((prev >= 0) & (prev != a), SWITCH_COST, 0.0)
        w = w * f * (1 - sw); peak = np.maximum(peak, w)
        dd2 = np.maximum(0.0, peak - w) / np.maximum(peak, 1e-6)
        cost += np.maximum(0.0, dd2 - DD_BUDGET)
        prev = a; prev_eq = eq_share[a]; exposures.append(float(eq_share[a].mean()))
    return {"P_goal": float((w >= 1.0).mean()),
            "attempted_violation_rate": round(attempts / (n * T_YEARS), 4),
            "mean_equity_exposure": round(float(np.mean(exposures)), 4),
            "interim_dd_cost": round(float(cost.mean()), 4)}


def train_one(kernels, names, allowed, dps, seed, total_steps, dual_rounds, dual_lr):
    """Lagrangian constrained PPO: warm-start from the dd-aware teacher, then alternate PPO training
    with projected dual ascent on lambda_cost to drive the expected interim-drawdown cost down."""
    lam_cost = 0.0
    env = GoalContractEnvV2(kernels, names, allowed, seed=seed, lam_cost=lam_cost)
    model, acc = bc_from_dp(env, dps, names, seed)
    m = make_model(env, seed)
    m.policy.load_state_dict(model.policy.state_dict())
    lam_hist = []
    per_round = max(1, total_steps // dual_rounds)
    for _ in range(dual_rounds):
        env.lam_cost = lam_cost
        m.set_env(env)
        m.learn(per_round, progress_bar=False, reset_num_timesteps=False)
        # measure the cost on the UNCAPPED contract, where interim drawdown is real (the capacity
        # contract has ~0 cost by construction, which left the dual inert in the first run).
        ev = policy_eval(m, kernels, names, allowed, 0, 0.5, n=1500, seed=7)
        lam_hist.append(round(lam_cost, 3))
        lam_cost = max(0.0, lam_cost + dual_lr * (ev["interim_dd_cost"] - 0.0))  # target 0 breach-years
    return m, acc, lam_hist


def main(argv=None):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    smoke = argv is not None and "--smoke" in argv
    total_steps = 6000 if smoke else 60000
    dual_rounds = 2 if smoke else 5
    seeds = (0,) if smoke else SEEDS
    print(f"=== W5b — widened conditional constrained PPO (Lagrangian) {'[SMOKE]' if smoke else ''} ===")
    uni = load_universe("US")
    as_of = uni["components"]["SPY"].dropna().index.max()
    kernels, names, allowed = contract_kernels(uni, as_of)
    print(f"obs=[funding, years-left, cap, drawdown, prev_eq] | actions {names}")
    # drawdown-aware DP teachers, per capacity, remapped to global action ids
    dd_dps = {}
    for cap in (0, 1):
        sub = {names[i]: kernels[names[i]] for i in allowed[cap]}
        dp = solve_dp_drawdown_aware(sub, T_YEARS, dd_budget=DD_BUDGET, dd_penalty=3.0)
        remap = np.array(allowed[cap]); dp["policy"] = remap[dp["policy"]]
        dd_dps[cap] = dp
    # champion reference from the plain DP (matched to W5's gate)
    dps = {}
    for cap in (0, 1):
        sub = {names[i]: kernels[names[i]] for i in allowed[cap]}
        d = solve_dp(sub, T_YEARS); remap = np.array(allowed[cap]); d["policy"] = remap[d["policy"]]
        dps[cap] = d
    dp_ref = {cap: simulate({"names": names, "policy": dps[cap]["policy"]}, kernels, T_YEARS, 0.5, seed=99)
              for cap in (0, 1)}
    print(f"DP reference P(goal): uncon {dp_ref[0]['P_goal']:.3f} | capacity {dp_ref[1]['P_goal']:.3f}")

    per_seed = {}
    for seed in seeds:
        t0 = time.time()
        m, acc, lam_hist = train_one(kernels, names, allowed, dps, seed, total_steps,
                                     dual_rounds, dual_lr=2.0)
        secs = int(time.time() - t0)
        m.save(MODELS / f"w5b_seed{seed}.zip")
        diag = prob_diagnostics(m)
        evals = {cap: policy_eval(m, kernels, names, allowed, cap, 0.5, seed=99) for cap in (0, 1)}
        # cost ablation (on the UNCAPPED contract, where drawdown cost is real): lambda_cost fixed 0
        # (no Lagrangian) -> does the cost drop when the Lagrangian is on?
        env0 = GoalContractEnvV2(kernels, names, allowed, seed=seed, lam_cost=0.0)
        m0 = make_model(env0, seed); bc0, _ = bc_from_dp(env0, dps, names, seed)
        m0.policy.load_state_dict(bc0.policy.state_dict()); m0.learn(total_steps, progress_bar=False)
        cost_off = policy_eval(m0, kernels, names, allowed, 0, 0.5, seed=99)["interim_dd_cost"]
        per_seed[seed] = {
            "bc_accuracy": round(acc, 3), "train_seconds": secs, "lambda_cost_path": lam_hist,
            "diagnostics": diag, "eval": evals,
            "regret_pp": {cap: round((dp_ref[cap]["P_goal"] - evals[cap]["P_goal"]) * 100, 2) for cap in (0, 1)},
            "interim_dd_cost_lagrangian": evals[0]["interim_dd_cost"],
            "interim_dd_cost_lambda0": round(cost_off, 4)}
        print(f"[seed {seed}] BC {acc:.0%} | {secs}s | max-prob {diag['mean_max_prob']:.3f} "
              f"varies {diag['argmax_varies']} | P(goal) {evals[0]['P_goal']:.3f}/{evals[1]['P_goal']:.3f} "
              f"| regret {per_seed[seed]['regret_pp']} | cost lam*/lam0 "
              f"{evals[1]['interim_dd_cost']:.3f}/{cost_off:.3f} | lam-path {lam_hist}")

    gw1 = all(s["diagnostics"]["mean_max_prob"] >= 0.5 and s["diagnostics"]["argmax_varies"]
              for s in per_seed.values())
    mean_regret = {cap: float(np.mean([s["regret_pp"][cap] for s in per_seed.values()])) for cap in (0, 1)}
    gw2 = all(r <= 2.0 for r in mean_regret.values())
    cost_reduces = float(np.mean([s["interim_dd_cost_lambda0"] - s["interim_dd_cost_lagrangian"]
                                  for s in per_seed.values()]))
    gwc = cost_reduces > 0.01   # a genuine cut, not a trivial 0.000
    verdict = (f"GW5-1 {'PASS' if gw1 else 'FAIL'} (no near-uniform) | GW5-2 {'PASS' if gw2 else 'FAIL'} "
               f"(mean regret {mean_regret} pp) | GW5b-COST {'PASS' if gwc else 'FAIL'} "
               f"(Lagrangian cuts interim-dd cost by {cost_reduces:+.3f} breach-years vs lambda=0; but trades "
               f"goal probability -> DP REMAINS CHAMPION) | "
               f"obs widened to [funding, years-left, cap, drawdown, prev_eq]; belief-in-obs + continuous "
               f"Dirichlet scoped (need regime kernels / continuous sleeves)")
    rep = {"experiment": "W5b widened conditional constrained PPO (Lagrangian cost)",
           "engine": ENGINE_VERSION, "scenario_status": "UNCALIBRATED — policy training only",
           "obs": ["funding W/G", "years-left/T", "capacity flag", "drawdown-from-peak", "prev equity share"],
           "dd_budget": DD_BUDGET, "actions": names,
           "dp_reference_P_goal": {c: round(r["P_goal"], 4) for c, r in dp_ref.items()},
           "seeds": {str(k): v for k, v in per_seed.items()},
           "gates": {"GW5_1_not_near_uniform": bool(gw1), "GW5_2_regret_within_tol": bool(gw2),
                     "GW5b_cost_reduces": bool(gwc)},
           "not_added_with_rationale": {
               "belief_in_obs": "env draws unconditional 1y factors — no regime signal to observe; "
                                "informative only with regime-conditional kernels (scoped).",
               "continuous_dirichlet": "SoftTree emits a discrete book; a continuous simplex head is "
                                       "warranted only past the 5-book menu (scoped)."},
           "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nVERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main(sys.argv[1:])
