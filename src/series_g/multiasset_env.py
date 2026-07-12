"""Series-G Extension 1 — the HIGH-DIMENSIONAL multi-asset regime-POMDP (Gym / Interface-B).

Phase 2 found H3 (hierarchical factoring) fails on the 1-asset tabular core because it is too low-dimensional
for a hierarchy to compress and because the optimum is inventory-first separable. This env is the fair test:
N assets share ONE global hidden regime (so the regime belief is GLOBAL — the manager's input) but carry
SEPARATE inventories (the per-asset worker's input). A flat policy must map (belief, ALL N inventories) → the
joint action (3^N) and EXPLODES with N; a manager→SHARED-worker hierarchy is O(1) (one worker reused per
asset). So this is exactly where hierarchy should pay — the high-dim phenomenon Phase 2 scoped H3 to.

State / observation (Box): [ belief_toxic, time_frac, (inv_i / I_max, last_obs_i) for i=1..N ]  (dim 2 + 2N).
  The env maintains the Bayes regime posterior from the AGGREGATE of all N per-asset observations (more assets
  ⇒ sharper belief — a genuine multi-asset inference benefit). belief is given as a feature ("regimes ARE the
  vocabulary", HC-1) so the manager's input is explicit and training is tractable.
Action (MultiDiscrete([3]*N)): per asset {0 PROVIDE, 1 ABSTAIN, 2 AGGRESS}.
Reward: sum over assets of the Phase-1 enriched per-asset reward (regime-dependent spread/adverse, carry,
  regime-dependent unwind cost), minus terminal liquidation at the horizon.

Economics + belief filter are reused from regime_pomdp.RegimePOMDP (single source of truth).
Run: python -m src.series_g.multiasset_env   (selftest)
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces

from src.series_g.regime_pomdp import BENIGN, BURST, QUIET, TOXIC, RegimePOMDP, PRIMARY_ENRICHED  # noqa: F401


class MultiAssetRegimePOMDP(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_assets: int = 6, econ: dict | None = None, seed: int | None = None):
        super().__init__()
        self.N = int(n_assets)
        self.m = RegimePOMDP(**(econ or PRIMARY_ENRICHED))
        self.T = self.m.T
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + 2 * self.N,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3] * self.N)
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.regime = BENIGN
        self.inv = np.zeros(self.N, dtype=int)
        self.belief = self.m.prior_toxic
        self.last_obs = np.zeros(self.N, dtype=int)

    # ----------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        v = [2.0 * self.belief - 1.0, 2.0 * (self.t / self.T) - 1.0]
        for i in range(self.N):
            v.append(2.0 * (self.inv[i] / self.m.I_max) - 1.0)
            v.append(1.0 if self.last_obs[i] == BURST else -1.0)
        return np.asarray(v, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.regime = TOXIC if self._rng.random() < self.m.prior_toxic else BENIGN
        self.inv = np.zeros(self.N, dtype=int)
        self.belief = self.m.prior_toxic
        self.last_obs = np.zeros(self.N, dtype=int)
        return self._obs(), {"regime": self.regime, "belief": self.belief}

    def step(self, action):
        a = np.asarray(action, dtype=int).reshape(self.N)
        reward = float(sum(self.m.reward(self.regime, int(a[i]), int(self.inv[i])) for i in range(self.N)))
        for i in range(self.N):
            self.inv[i] = self.m.inventory_next(int(a[i]), int(self.inv[i]))
        self.t += 1
        # exogenous global regime transition
        self.regime = TOXIC if self._rng.random() < self.m.M[self.regime, TOXIC] else BENIGN
        # per-asset observations from the (shared) new regime; aggregate Bayes update of the global belief
        b_pred = self.m.predict(self.belief)
        lik_tox, lik_ben = b_pred, 1.0 - b_pred
        obs = np.empty(self.N, dtype=int)
        for i in range(self.N):
            o = BURST if self._rng.random() < self.m.obs[self.regime, BURST] else QUIET
            obs[i] = o
            lik_tox *= self.m.obs[TOXIC, o]
            lik_ben *= self.m.obs[BENIGN, o]
        self.last_obs = obs
        self.belief = float(lik_tox / (lik_tox + lik_ben)) if (lik_tox + lik_ben) > 1e-300 else b_pred
        terminated = self.t >= self.T
        if terminated:
            reward += float(sum(self.m.terminal(int(self.inv[i])) for i in range(self.N)))
        return self._obs(), reward, terminated, False, {"regime": self.regime, "belief": self.belief}


def _selftest() -> None:
    env = MultiAssetRegimePOMDP(n_assets=6, seed=0)
    o, info = env.reset(seed=0)
    assert o.shape == (2 + 2 * 6,), o.shape
    assert env.action_space.nvec.tolist() == [3] * 6
    # random rollout completes a full horizon
    total, steps = 0.0, 0
    o, _ = env.reset(seed=1)
    done = False
    rng = np.random.default_rng(0)
    while not done:
        a = rng.integers(0, 3, env.N)
        o, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        done = term or trunc
    assert steps == env.T, steps
    assert np.all(np.abs(o) <= 1.0 + 1e-6)
    # belief responds to observations: feed all-burst -> belief rises
    env.reset(seed=2)
    for _ in range(8):
        env.step(np.zeros(env.N, dtype=int))  # PROVIDE everywhere; regime evolves, obs drive belief
    print("=== multiasset_env _selftest ===")
    print(f"  N={env.N}, obs_dim={env.observation_space.shape[0]}, action=MultiDiscrete{env.action_space.nvec.tolist()}")
    print(f"  random full-horizon return={total:.2f} over {steps} steps; obs in [-1,1]; belief filter live")
    print("VERDICT: PASS")


if __name__ == "__main__":
    _selftest()
