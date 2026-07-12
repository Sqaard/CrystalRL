"""E-27 — the REAL CRYSTAL-1 PPO on the real substrate, with a RISK-TARGETED reward.

The concept (Ivan): the RL layer's intelligence comes from its reward — so point the reward at the USER'S
objective class. Each PPO head is trained to maximize return SUBJECT TO a drawdown budget baked into its
reward: r_t = pnl_t − λ · max(0, DD_t − budget). A family of heads (budgets 5%/8%/12% + a pure-return head)
grows the personalization frontier into the conservative gap the rule-book menu couldn't reach.

CRYSTAL-1 identity preserved:
  * the ACTOR is the genuine `SoftTreeActorCriticPolicy` (src/crystal/soft_tree_policy.py) — a depth-3 soft
    decision tree whose every action is a soft path to a named leaf (leaf_report() ships with each head);
  * the belief (the frozen v9 macro HMM posterior) is the ONLY market memory in the observation;
  * the observation is 3 NAMED coordinates: [P(bear), previous exposure, current drawdown-vs-peak] —
    the dd-state is part of the USER-objective, so the policy may see it (named, causal, self-computed);
  * actions are 5 legible exposure levels {0, .25, .5, .75, 1.0}; cash earns T-bills; 10bp per change.

Discipline: training data = 2010-2018 ONLY (the belief's own freeze window); dev 2019-21 / hold 2022-23 for
evaluation; the frozen OOS 2024-26 is NEVER touched (heads enter the menu as PENDING; their read = the next
pre-registered evaluation). Each head's evaluation: full-hold diagnostics vs B&H (z_dsd, NI), a carry-matched
constant twin, and an INFERENCE-PLACEBO (block-shuffled belief at inference — the head's edge must collapse
if the belief is load-bearing).

Run: python interpretability/crystal_ppo.py            (trains + evaluates all heads; ~10-20 min CPU)
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
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, DEV, HOLD, TRAIN  # noqa: E402
from interpretability.hl_v4_over_crystal1 import ann_dd, risk_boot_z  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from interpretability.build_dow_extended_panel import fetch  # noqa: E402

OUT = HERE / "crystal_ppo_report.json"
MODELS = HERE / "crystal_ppo_models"; MODELS.mkdir(exist_ok=True)
LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
COST = 0.001
EP_LEN = 252
TOTAL_STEPS = 60_000
HEADS = {"pure_return": {"budget": None, "lam": 0.0},
         "dd12": {"budget": 0.12, "lam": 2.0},
         "dd08": {"budget": 0.08, "lam": 2.0},
         "dd05": {"budget": 0.05, "lam": 2.0}}
NI_MARGIN = 0.0002


# ---------------- data ---------------------------------------------------------------------------
def get_streams():
    r, macro = load_extended()
    bel, meta = build_belief(r, macro)
    irx = fetch("^IRX").set_index("date")["close"].reindex(r.index).ffill()
    rf = (irx / 100 / 252).fillna(0.0)
    def win(a, b):
        m = (r.index >= np.datetime64(a)) & (r.index <= np.datetime64(b))
        return (r[m].to_numpy()[1:], bel[m].to_numpy()[:-1], rf[m].to_numpy()[1:])
    return {"train": win(TRAIN[0], TRAIN[1]), "dev": win(*DEV), "hold": win(*HOLD)}, meta


# ---------------- env -----------------------------------------------------------------------------
class ExposureEnv(gym.Env):
    """Belief-gated exposure control with a drawdown-budget reward. Obs = [P(bear), prev_ex, dd_state]."""
    def __init__(self, ro, bl, rf, budget=None, lam=0.0, ep_len=EP_LEN, seed=0):
        super().__init__()
        self.ro, self.bl, self.rf = ro, bl, rf
        self.budget, self.lam, self.ep_len = budget, lam, ep_len
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=np.array([0, 0, -1], dtype=np.float32),
                                            high=np.array([1, 1, 0], dtype=np.float32))
        self.action_space = spaces.MultiDiscrete([len(LEVELS)])

    def _obs(self):
        return np.array([self.bl[self.t], self.ex, self.dd], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.t0 = int(self.rng.integers(0, len(self.ro) - self.ep_len - 1))
        self.t = self.t0
        self.ex, self.eq, self.peak, self.dd = 1.0, 1.0, 1.0, 0.0
        return self._obs(), {}

    def step(self, action):
        new_ex = float(LEVELS[int(np.asarray(action).reshape(-1)[0])])
        pnl = new_ex * self.ro[self.t] + (1 - new_ex) * self.rf[self.t] - abs(new_ex - self.ex) * COST
        self.ex = new_ex
        self.eq *= (1 + pnl); self.peak = max(self.peak, self.eq)
        self.dd = self.eq / self.peak - 1.0
        r = pnl
        if self.budget is not None:
            r -= self.lam * max(0.0, -self.dd - self.budget)      # per-day penalty while beyond budget
        self.t += 1
        done = (self.t - self.t0) >= self.ep_len
        return self._obs(), float(r * 100.0), done, False, {}


# ---------------- training + rollout ---------------------------------------------------------------
def train_head(name, cfg, streams, seed=0, total_steps=TOTAL_STEPS, tree_depth=3, init_from=None):
    ro, bl, rf = streams["train"]
    env = ExposureEnv(ro, bl, rf, budget=cfg["budget"], lam=cfg["lam"], seed=seed)
    model = PPO(SoftTreeActorCriticPolicy, env, n_steps=2048, batch_size=256, n_epochs=6,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                seed=seed, device="cpu", verbose=0,
                policy_kwargs={"feat_idx": (0, 1, 2), "tree_depth": tree_depth, "beta": 1.0,
                               "critic_arch": (32, 32)})
    if init_from is not None:                                    # teacher warm-start (E-28 lever T)
        model.policy.load_state_dict(init_from, strict=False)
    t0 = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    model.save(MODELS / f"{name}.zip")
    return model, round(time.time() - t0, 1)


def rollout(model, ro, bl, rf, bl_override=None):
    """Deterministic policy-forward; returns (pnl, exposures)."""
    b = bl if bl_override is None else bl_override
    ex, eq, peak = 1.0, 1.0, 1.0
    pnl = np.empty(len(ro)); exs = np.empty(len(ro))
    for t in range(len(ro)):
        dd = eq / peak - 1.0
        obs = np.array([[b[t], ex, dd]], dtype=np.float32)
        a, _ = model.predict(obs, deterministic=True)
        new_ex = float(LEVELS[int(np.asarray(a).reshape(-1)[0])])
        p = new_ex * ro[t] + (1 - new_ex) * rf[t] - abs(new_ex - ex) * COST
        ex = new_ex; eq *= (1 + p); peak = max(peak, eq)
        pnl[t] = p; exs[t] = ex
    return pnl, exs


def evaluate_head(name, model, streams, seed=0):
    ro_h, bl_h, rf_h = streams["hold"]
    pnl, exs = rollout(model, ro_h, bl_h, rf_h)
    bh = ro_h.copy()
    a, d = ann_dd(pnl)
    z_dsd, gain = risk_boot_z(pnl, bh, block=20, n_boot=1000, seed=27)
    dd_ = pnl - bh; _, se = block_z(dd_, block=5, n_boot=1000, seed=27)
    ni = (float(dd_.mean()) + NI_MARGIN) / se
    # carry-matched twin
    ebar = float(exs.mean())
    twin = ebar * ro_h + (1 - ebar) * rf_h
    z_tw, _ = risk_boot_z(pnl, twin, block=20, n_boot=1000, seed=27)
    dt = pnl - twin; _, set_ = block_z(dt, block=5, n_boot=1000, seed=27)
    ni_tw = (float(dt.mean()) + NI_MARGIN) / set_
    # inference placebo: block-shuffled belief
    rng = np.random.default_rng(seed)
    blocks = [bl_h[i:i + 60].copy() for i in range(0, len(bl_h), 60)]
    rng.shuffle(blocks)
    bl_pl = np.concatenate(blocks)[:len(bl_h)]
    pnl_pl, _ = rollout(model, ro_h, bl_h, rf_h, bl_override=bl_pl)
    z_pl, _ = risk_boot_z(pnl_pl, bh, block=20, n_boot=1000, seed=27)
    return {"hold_ann": round(a, 4), "hold_maxDD": round(d, 4),
            "hold_dsd_bp": round(float(np.sqrt((np.minimum(pnl, 0) ** 2).mean())) * 1e4, 2),
            "mean_exposure": round(ebar, 3), "exposure_changes": int((np.abs(np.diff(exs)) > 1e-9).sum()),
            "vs_bh": {"z_dsd": round(z_dsd, 2), "ni_z": round(float(ni), 2), "dsd_gain_bp": round(gain * 1e4, 2)},
            "vs_carry_twin": {"z_dsd": round(z_tw, 2), "ni_z": round(float(ni_tw), 2)},
            "inference_placebo_z_dsd": round(z_pl, 2),
            "belief_load_bearing": bool(z_dsd - z_pl > 1.0)}


def policy_prob_diagnostics(model, n_levels=len(LEVELS)):
    """Full action-probability diagnostics (E-27c): argmax alone hides an untrained near-uniform policy —
    a constant deterministic dial can be a numerical tie-break, not a learned preference. Report the
    distribution itself: per-leaf probs and entropy/max-prob over a fixed obs grid."""
    grid = np.array([[b, pe, dd] for b in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                     for pe in (0.0, 0.5, 1.0) for dd in (0.0, -0.05, -0.15)], dtype=np.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(grid))
        d0 = dist.distribution[0] if isinstance(dist.distribution, list) else dist.distribution
        probs = d0.probs.numpy()
        leaf_probs = torch.softmax(model.policy.tree.leaf_logits, dim=-1).numpy()
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    amax = probs.argmax(axis=1)
    return {"mean_max_prob": round(float(probs.max(1).mean()), 4),
            "mean_entropy": round(float(ent.mean()), 4),
            "max_entropy": round(float(np.log(n_levels)), 4),
            "argmax_varies_over_grid": bool(len(set(amax.tolist())) > 1),
            "leaf_action_probs": [[round(float(p), 4) for p in row] for row in leaf_probs],
            "note": "mean_max_prob ~ 1/n_levels + entropy ~ log(n_levels) = near-uniform policy; "
                    "treat its deterministic dial as a tie-break artifact, not a learned dial"}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-27 — the real CRYSTAL-1 PPO, risk-targeted heads ===")
    streams, bel_meta = get_streams()
    print(f"belief: K={bel_meta['K']} | train {len(streams['train'][0])}d, hold {len(streams['hold'][0])}d")
    results = {}
    for name, cfg in HEADS.items():
        model, secs = train_head(name, cfg, streams)
        ev = evaluate_head(name, model, streams)
        leaf = model.policy.tree.leaf_report(len(LEVELS))
        results[name] = {"cfg": cfg, "train_seconds": secs, **ev,
                         "leaf_argmax_actions": [l["argmax_action"] for l in leaf["leaves"]],
                         "action_prob_diagnostics": policy_prob_diagnostics(model)}
        print(f"[{name}] {secs}s | hold ann {ev['hold_ann']:+.2%} maxDD {ev['hold_maxDD']:.2%} "
              f"dsd {ev['hold_dsd_bp']}bp ex_bar {ev['mean_exposure']} | vs B&H z_dsd {ev['vs_bh']['z_dsd']:+.2f} "
              f"ni {ev['vs_bh']['ni_z']:+.2f} | twin z {ev['vs_carry_twin']['z_dsd']:+.2f} | "
              f"placebo z {ev['inference_placebo_z_dsd']:+.2f} load_bearing={ev['belief_load_bearing']}")
    rep = {"experiment": "E-27 real CRYSTAL-1 PPO (soft-tree actor) with risk-targeted reward heads",
           "identity": "SoftTreeActorCriticPolicy depth-3; obs=[P(bear), prev_ex, dd_state]; 5 exposure levels; "
                       "belief = frozen v9 macro HMM; train 2010-2018 only; OOS quarantined (heads = PENDING)",
           "heads": results}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
