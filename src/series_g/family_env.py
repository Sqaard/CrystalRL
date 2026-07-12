"""Series-G FAMILY — the G-regime ROTATION env (the polygon becomes a parameterized universe family).

Design (the honest way to scale complexity): a hidden birth-death Markov chain over G regimes; regime g has
severity s_g = g/(G-1) and — the rotation — selects WHICH of G venues currently pays the spread. Providing at
the regime's venue earns +spread; at any other venue it costs −adverse (adverse selection). The agent observes
only a noisy severity signal y = s_g + σ·ε (σ scaled ∝ the regime gap so per-step inference difficulty is
G-invariant) and must track a G-dim Bayes belief.

Why this is NOT manufactured complexity: the optimal policy has one genuinely distinct behavior PER REGIME
("provide at venue g when confident it's regime g"), plus uncertainty-driven ABSTAIN and inventory UNWIND —
i.e., regime-conditional selection, the exact structure a dead market lacks. The number of named modes grows
linearly with G, so a K-concept explanation can cover it iff K ≳ G — the C*≈K law's test bed (the other
agent's analytic sweep found the bend; this env tests it on LEARNED policies).

Action space: Discrete(G+2) = {provide@venue_0..G-1, ABSTAIN, UNWIND}. Obs: [belief (G), t/T, inv/I_max, y].
Run: python -m src.series_g.family_env   (selftest: belief tracks, oracle >> random)
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces


class RegimeRotationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, G: int = 4, T: int = 20, I_max: int = 4, p_stay: float = 0.90,
                 spread: float = 3.0, adverse: float = 2.5, unwind_cost: float = 0.2,
                 hold_cost: float = 0.02, liq_penalty: float = 1.0, seed: int | None = None):
        super().__init__()
        self.G, self.T, self.I_max, self.p_stay = int(G), int(T), int(I_max), float(p_stay)
        self.spread, self.adverse = float(spread), float(adverse)
        self.unwind_cost, self.hold_cost, self.liq_penalty = float(unwind_cost), float(hold_cost), float(liq_penalty)
        self.sev = np.linspace(0.0, 1.0, self.G) if self.G > 1 else np.array([0.5])
        gap = 1.0 / max(1, self.G - 1)
        self.sigma = 0.55 * gap                      # per-step SNR (gap/σ ≈ 1.8) G-invariant
        self.observation_space = spaces.Box(-10.0, 10.0, (self.G + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.G + 2)
        self.ABSTAIN, self.UNWIND = self.G, self.G + 1
        self._rng = np.random.default_rng(seed)
        self.t = 0; self.g = 0; self.inv = 0
        self.belief = np.full(self.G, 1.0 / self.G)
        self.y = 0.0

    # ------------------------------------------------------------------
    def _transition(self, g: int) -> int:
        if self._rng.random() < self.p_stay or self.G == 1:
            return g
        if g == 0:
            return 1
        if g == self.G - 1:
            return self.G - 2
        return g + (1 if self._rng.random() < 0.5 else -1)

    def _emit(self) -> float:
        return float(self.sev[self.g] + self.sigma * self._rng.standard_normal())

    def _belief_update(self, y: float) -> None:
        # predict through the birth-death chain, then Gaussian-likelihood measurement update
        b = self.belief
        bp = np.empty(self.G)
        for g in range(self.G):
            stay = b[g] * self.p_stay
            inflow = 0.0
            if g - 1 >= 0:
                inflow += b[g - 1] * (1 - self.p_stay) * (1.0 if g - 1 == 0 else 0.5)
            if g + 1 <= self.G - 1:
                inflow += b[g + 1] * (1 - self.p_stay) * (1.0 if g + 1 == self.G - 1 else 0.5)
            bp[g] = stay + inflow
        lik = np.exp(-0.5 * ((y - self.sev) / self.sigma) ** 2)
        post = bp * lik
        s = post.sum()
        self.belief = post / s if s > 1e-300 else bp / bp.sum()

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.belief, [2.0 * self.t / self.T - 1.0, self.inv / self.I_max, self.y]]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0; self.inv = 0
        self.g = int(self._rng.integers(self.G))
        self.belief = np.full(self.G, 1.0 / self.G)
        self.y = self._emit()
        self._belief_update(self.y)
        return self._obs(), {"regime": self.g}

    def step(self, action):
        a = int(action)
        r = -self.hold_cost * self.inv
        if a < self.G:                                   # provide @ venue a
            if self.inv < self.I_max:
                r += self.spread if a == self.g else -self.adverse
                self.inv += 1
        elif a == self.UNWIND and self.inv > 0:
            r -= self.unwind_cost
            self.inv -= 1
        self.t += 1
        self.g = self._transition(self.g)
        self.y = self._emit()
        self._belief_update(self.y)
        term = self.t >= self.T
        if term:
            r -= self.liq_penalty * self.inv
        return self._obs(), float(r), term, False, {"regime": self.g}


def oracle_policy(env: RegimeRotationEnv, p_min: float = 0.556):
    """MAP heuristic: provide at the argmax-belief venue when confident enough & not capped; unwind at the
    horizon tail; else abstain. (Myopic provide threshold: p*spread-(1-p)*adverse>0 ⟺ p>adverse/(spread+adverse).)"""
    b = env.belief
    g_hat = int(np.argmax(b))
    if env.inv > 0 and (env.T - env.t) <= env.inv:
        return env.UNWIND
    if b[g_hat] > p_min and env.inv < env.I_max:
        return g_hat
    return env.ABSTAIN


def _selftest():
    rng = np.random.default_rng(0)
    for G in (2, 6, 12):
        env = RegimeRotationEnv(G=G, seed=0)
        # belief tracks the hidden regime (MAP accuracy >> 1/G)
        hits, n = 0, 0
        env.reset(seed=1)
        for _ in range(400):
            _, _, term, _, info = env.step(env.ABSTAIN)
            hits += int(np.argmax(env.belief) == env.g); n += 1
            if term:
                env.reset(seed=int(rng.integers(1 << 30)))
        map_acc = hits / n
        # oracle >> random
        def run(policy_fn, eps=60):
            tot = 0.0
            for ep in range(eps):
                env.reset(seed=1000 + ep); done = False
                while not done:
                    a = policy_fn(env)
                    _, r, term, trunc, _ = env.step(a); tot += r; done = term or trunc
            return tot / eps
        r_orc = run(oracle_policy)
        r_rnd = run(lambda e: int(rng.integers(e.G + 2)))
        assert map_acc > min(0.75, 2.5 / G), (G, map_acc)
        assert r_orc > r_rnd + 5, (G, r_orc, r_rnd)
        print(f"  G={G:2d}: MAP-acc={map_acc:.2f} (chance {1/G:.2f})  oracle={r_orc:+.1f}  random={r_rnd:+.1f}")
    print("VERDICT: PASS (belief tracks; oracle >> random; scales in G)")


if __name__ == "__main__":
    _selftest()
