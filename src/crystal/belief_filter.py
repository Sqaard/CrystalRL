"""CRYSTAL-1 L1 — the LEARNED neural Bayes filter (the belief bottleneck, learned self-supervised).

The blueprint's L1 law (WH1): all memory routes through an explicit K-named belief; unstructured recurrence is
forbidden as the complexity source. This module implements L1 as a STRUCTURED differentiable filter — a
learnable K-state HMM run as a Bayes recursion:

    b_pred = T^softmax  @ b_{t-1}          (learned transition, rows on the simplex)
    b_t    ∝ b_pred * E[:, o_t]            (learned emission likelihood)

trained SELF-SUPERVISED by maximizing the observation log-likelihood  Σ log P(o_t | o_{<t}) — NO regime labels
(the env's hidden state is never seen). Because the filter IS a small named generative model, it is
interpretable by construction: after training you can READ the learned world model (transition stickiness,
emission signatures) and compare it to ground truth where truth exists (the polygon).

Selftest = the L1 gate: on corner-env observation streams the filter must (i) recover the true transition and
emission parameters up to state permutation, (ii) track the analytic Bayes belief (low MAE), (iii) beat the
memoryless baseline in held-out log-likelihood. Run: python -m src.crystal.belief_filter
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NeuralBayesFilter(nn.Module):
    """Learnable K-state discrete-observation Bayes filter (an HMM trained by filtering likelihood)."""

    def __init__(self, K: int = 2, A_obs: int = 2, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.K, self.A = K, A_obs
        self.logits_T = nn.Parameter(0.3 * torch.randn(K, K, generator=g))
        self.logits_E = nn.Parameter(0.3 * torch.randn(K, A_obs, generator=g))
        self.logits_p0 = nn.Parameter(torch.zeros(K))

    def mats(self):
        """Return the softmax-normalized (transition T, emission E, prior p0) probability matrices."""
        T = torch.softmax(self.logits_T, dim=1)     # T[i,j] = P(j | i), rows sum 1
        E = torch.softmax(self.logits_E, dim=1)     # E[g,o] = P(o | g)
        p0 = torch.softmax(self.logits_p0, dim=0)
        return T, E, p0

    def forward(self, obs: torch.Tensor):
        """obs: (B, L) int tensor. Returns (beliefs (B, L, K) AFTER each update, per-seq loglik (B,))."""
        T, E, p0 = self.mats()
        B, L = obs.shape
        b = p0.expand(B, self.K)
        beliefs, ll = [], torch.zeros(B)
        for t in range(L):
            b_pred = b @ T                                       # (B,K) predict
            lik = E[:, obs[:, t]].t()                            # (B,K) emission likelihood
            joint = b_pred * lik
            p_o = joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
            b = joint / p_o
            ll = ll + torch.log(p_o.squeeze(1))
            beliefs.append(b)
        return torch.stack(beliefs, dim=1), ll

    # ---------------- numpy runtime (frozen filter inside an env wrapper) ----------------
    def numpy_params(self):
        """Return (T, E, p0) as detached numpy arrays for the frozen numpy-runtime filter."""
        with torch.no_grad():
            T, E, p0 = self.mats()
        return T.numpy(), E.numpy(), p0.numpy()


def train_filter(sequences: np.ndarray, K: int = 2, A_obs: int = 2, epochs: int = 300,
                 lr: float = 5e-2, seed: int = 0, verbose: bool = True) -> NeuralBayesFilter:
    """sequences: (N, L) int array of raw observation streams (no labels)."""
    torch.manual_seed(seed)
    f = NeuralBayesFilter(K, A_obs, seed=seed)
    obs = torch.as_tensor(sequences, dtype=torch.long)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        _, ll = f(obs)
        loss = -ll.mean()
        loss.backward()
        opt.step()
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"  [filter] epoch {ep:4d}  nll/step = {loss.item() / obs.shape[1]:.4f}")
    return f


def align_to_truth(T_learn, E_learn, E_true):
    """Resolve the HMM label permutation by matching emission signatures; returns permuted (T, E)."""
    from itertools import permutations
    K = E_true.shape[0]
    best, bperm = None, None
    for perm in permutations(range(K)):
        err = np.abs(E_learn[list(perm)] - E_true).sum()
        if best is None or err < best:
            best, bperm = err, list(perm)
    return T_learn[np.ix_(bperm, bperm)], E_learn[bperm], bperm


def _selftest() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
    from src.series_g.multiasset_env import MultiAssetRegimePOMDP
    from src.series_g.regime_pomdp import RegimePOMDP, PRIMARY_ENRICHED, BENIGN, TOXIC

    # ---- collect raw observation streams (random actions; obs are exogenous to actions) ----
    env = MultiAssetRegimePOMDP(n_assets=1, seed=0)
    rng = np.random.default_rng(0)
    n_ep, L = 400, env.T
    seqs = np.empty((n_ep, L), dtype=int)
    for e in range(n_ep):
        env.reset(seed=100_000 + e)
        for t in range(L):
            env.step([1])                     # ABSTAIN; observation stream unaffected by actions
            seqs[e, t] = int(env.last_obs[0])
    train, held = seqs[:320], seqs[320:]

    f = train_filter(train, K=2, A_obs=2, epochs=300, seed=0)
    T_l, E_l, p0_l = f.numpy_params()
    m = RegimePOMDP(**PRIMARY_ENRICHED)
    T_true = m.M; E_true = m.obs
    T_a, E_a, perm = align_to_truth(T_l, E_l, E_true)

    # (i) parameter recovery
    errT = float(np.abs(T_a - T_true).max()); errE = float(np.abs(E_a - E_true).max())
    # (ii) belief tracking vs the analytic filter on held-out streams
    with torch.no_grad():
        bel_l, ll_held = f(torch.as_tensor(held, dtype=torch.long))
    bel_l = bel_l.numpy()[:, :, perm.index(TOXIC) if TOXIC in perm else 1]
    # analytic beliefs
    mae = []
    for e in range(held.shape[0]):
        b = m.prior_toxic; errs = []
        for t in range(held.shape[1]):
            b = m.update(m.predict(b), int(held[e, t]))
            errs.append(abs(b - bel_l[e, t]))
        mae.append(np.mean(errs))
    mae = float(np.mean(mae))
    # (iii) held-out likelihood vs memoryless baseline
    p_marg = train.mean()
    ll_base = float((held * np.log(max(p_marg, 1e-9)) + (1 - held) * np.log(max(1 - p_marg, 1e-9))).sum(axis=1).mean())
    ll_f = float(ll_held.mean())

    print("=== crystal.belief_filter selftest (L1 gate) ===")
    print(f"  learned T (aligned):\n{np.round(T_a,3)}\n  true T:\n{np.round(T_true,3)}")
    print(f"  learned E (aligned):\n{np.round(E_a,3)}\n  true E:\n{np.round(E_true,3)}")
    print(f"  (i) param recovery: max|dT|={errT:.3f}, max|dE|={errE:.3f}  (gate <= 0.10)")
    print(f"  (ii) belief MAE vs analytic filter (held-out): {mae:.4f}  (gate <= 0.05)")
    print(f"  (iii) held-out loglik/seq: filter {ll_f:.2f} vs memoryless {ll_base:.2f}  (gate: filter wins)")
    ok = errT <= 0.10 and errE <= 0.10 and mae <= 0.05 and ll_f > ll_base
    print(f"VERDICT: {'PASS — L1 learns the world model self-supervised (no labels)' if ok else 'FAIL'}")


if __name__ == "__main__":
    _selftest()
