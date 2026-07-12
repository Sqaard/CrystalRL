"""C-5 governor — the runtime ENVELOPE-ENFORCEMENT layer for belief writes (the reference-governor / CBF the
literature review B said CRYSTAL-1 lacks: its envelope was a soft probe restriction, not a hard projector).

A belief write is only certified on the VISITED envelope (C3 of the writ ladder). The BeliefGovernor turns that from a
convention into an enforced projection:
  - a command INSIDE the envelope passes through BYTE-IDENTICAL (no authority lost);
  - a command OUTSIDE is PROJECTED to the nearest on-envelope belief (clamp-to-box on the simplex) and returns a
    GUARANTEE_DELTA annunciation = an explicit authority DEMOTION (the AF447 anti-pattern is annunciating the LOST
    guarantee, not silently reverting);
  - chronic boundary contact is METERED into a cumulative budget (envelope-surfing: a protection ridden continuously
    becomes a de-facto controller — the K/IV human-rate-subsidy re-grounding).

Envelope = per-coordinate [lo,hi] box fit from naturally-visited beliefs (quantiles), intersected with the simplex.
For K=2 this is an interval clamp on b[toxic]; the projection is the closed-form of the CBF-QP for a box safe-set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeliefGovernor:
    """C-5 box governor: clamps belief writes to the visited per-coordinate envelope, annunciating demotions."""
    lo: np.ndarray                      # per-coordinate lower bound of the visited envelope (K,)
    hi: np.ndarray                      # per-coordinate upper bound (K,)
    eps: float = 1e-6                   # "changed" tolerance
    boundary_contact_budget: float = 0.0   # cumulative sum of projection magnitudes (envelope-surfing meter)
    n_projections: int = 0
    n_calls: int = 0
    history: list = field(default_factory=list)

    @classmethod
    def from_visited(cls, beliefs: np.ndarray, q: float = 0.02):
        """Fit the envelope box from visited belief vectors using [q, 1-q] quantiles per coordinate."""
        B = np.atleast_2d(np.asarray(beliefs, float))
        lo = np.quantile(B, q, axis=0); hi = np.quantile(B, 1 - q, axis=0)
        return cls(lo=lo, hi=hi)

    def _project(self, b: np.ndarray) -> np.ndarray:
        """Clamp to the box then renormalize onto the simplex (closed-form box-CBF projection)."""
        c = np.clip(b, self.lo, self.hi)
        s = c.sum()
        return c / s if s > 1e-12 else np.full_like(b, 1.0 / len(b))

    def govern(self, b_command: np.ndarray, guarantees=None):
        """Return (b_out, annunciation). b_out == b_command byte-identical if on-envelope; else projected + demoted."""
        self.n_calls += 1
        b = np.asarray(b_command, float)
        on = bool(np.all(b >= self.lo - self.eps) and np.all(b <= self.hi + self.eps))
        if on:
            self.history.append(("PASS", 0.0))
            return b.copy(), {"status": "ON_ENVELOPE", "authority": "full", "guarantee_delta": None, "projection_l1": 0.0}
        b_out = self._project(b)
        mag = float(np.abs(b_out - b).sum())
        self.boundary_contact_budget += mag; self.n_projections += 1
        self.history.append(("PROJECT", mag))
        lost = (list(guarantees) if guarantees else ["command_on_visited_envelope"])
        ann = {"status": "OFF_ENVELOPE_PROJECTED", "authority": "DEMOTED",
               "guarantee_delta": {"lost": lost, "from": [round(float(x), 4) for x in b],
                                   "to": [round(float(x), 4) for x in b_out], "projection_l1": round(mag, 4)},
               "cumulative_boundary_contact": round(self.boundary_contact_budget, 4)}
        return b_out, ann

    def surfing_alarm(self, cap: float) -> bool:
        """Envelope-surfing tripwire: chronic boundary contact turns a protection into a controller."""
        return self.boundary_contact_budget > cap


@dataclass
class ManifoldGovernor:
    """JOINT-manifold governor (debt-1 fix): the box BeliefGovernor gates each coordinate's marginal but PASSES JOINT
    off-manifold beliefs (the diffuse corners C-5's sign-epistasis factorial forces). The visited-belief cloud is
    MULTIMODAL (near-one-hot clusters around vertices), so a single-Gaussian Mahalanobis is exactly wrong — its mean is
    the uniform point. This uses kNN novelty detection: a command is on-manifold iff its distance to the k-th nearest
    visited belief is within a threshold calibrated from the visited set's own kNN distances. A diffuse 'both' belief is
    far from every near-vertex cluster -> caught; a near-one-hot belief sits in its cluster -> passes. Off-manifold ->
    project toward the nearest visited belief until inside the envelope, renormalize, GUARANTEE_DELTA demotion + meter."""
    ref: np.ndarray                  # subsample of visited beliefs (M,K)
    k: int
    thr: float                       # kNN-distance (L1) acceptance threshold
    boundary_contact_budget: float = 0.0
    n_projections: int = 0
    n_calls: int = 0

    @classmethod
    def from_visited(cls, beliefs: np.ndarray, k: int = 8, q: float = 0.99, max_ref: int = 1500, seed: int = 0):
        """Fit the kNN reference set and acceptance threshold from the visited belief cloud's own kNN distances."""
        B = np.atleast_2d(np.asarray(beliefs, float))
        rng = np.random.default_rng(seed)
        if len(B) > max_ref:
            B = B[rng.choice(len(B), max_ref, replace=False)]
        # calibrate the threshold from the visited set's own k-th NN distances (leave-one-out via the (k+1)-th)
        D = np.abs(B[:, None, :] - B[None, :, :]).sum(-1)          # pairwise L1 (M,M)
        knn_self = np.sort(D, axis=1)[:, k]                        # k-th neighbor (excludes self at index 0)
        thr = float(np.quantile(knn_self, q))
        return cls(ref=B, k=k, thr=thr)

    def knn_dist(self, b: np.ndarray) -> float:
        """Return the L1 distance from b to its k-th nearest visited belief (the novelty score)."""
        d = np.abs(self.ref - np.asarray(b, float)).sum(-1)
        return float(np.sort(d)[self.k - 1])

    def govern(self, b_command: np.ndarray, guarantees=None):
        """Pass on-manifold commands identically; project off-manifold ones toward the nearest visited belief + demote."""
        self.n_calls += 1
        b = np.asarray(b_command, float)
        dist = self.knn_dist(b)
        if dist <= self.thr:
            return b.copy(), {"status": "ON_MANIFOLD", "authority": "full", "knn_dist": round(dist, 3), "guarantee_delta": None}
        # project toward the NEAREST visited belief until inside the envelope
        nn = self.ref[int(np.argmin(np.abs(self.ref - b).sum(-1)))]
        c = b.copy()
        for s in np.linspace(0.0, 1.0, 21):
            c = (1 - s) * b + s * nn
            if self.knn_dist(c) <= self.thr:
                break
        c = np.clip(c, 0, None); c = c / c.sum()
        mag = float(np.abs(c - b).sum()); self.boundary_contact_budget += mag; self.n_projections += 1
        ann = {"status": "OFF_MANIFOLD_PROJECTED", "authority": "DEMOTED", "knn_dist": round(dist, 3), "thr": round(self.thr, 3),
               "guarantee_delta": {"lost": list(guarantees) if guarantees else ["command_on_joint_belief_manifold"],
                                   "from": [round(float(x), 4) for x in b], "to": [round(float(x), 4) for x in c],
                                   "projection_l1": round(mag, 4)},
               "cumulative_boundary_contact": round(self.boundary_contact_budget, 4)}
        return c, ann


def _selftest():
    # K=2 envelope fit from a benign-skewed visited distribution
    rng = np.random.default_rng(0)
    vb = np.clip(rng.beta(1.5, 5.0, size=4000), 0, 1)               # toxic-belief mostly low
    beliefs = np.stack([1 - vb, vb], axis=1)                         # (N,2) simplex
    gov = BeliefGovernor.from_visited(beliefs, q=0.02)
    # on-envelope command -> byte-identical
    b_in = np.array([1 - 0.2, 0.2]); b_out, ann = gov.govern(b_in)
    assert np.allclose(b_out, b_in) and ann["status"] == "ON_ENVELOPE", "on-envelope must pass through identically"
    # off-envelope command (toxic=0.99, far above the visited hi) -> projected + demoted + annunciated
    b_hi = np.array([0.01, 0.99]); b_out2, ann2 = gov.govern(b_hi)
    assert ann2["status"] == "OFF_ENVELOPE_PROJECTED" and ann2["authority"] == "DEMOTED"
    assert b_out2[1] < 0.99 and ann2["guarantee_delta"]["projection_l1"] > 0
    # cumulative metering + surfing alarm
    for _ in range(50):
        gov.govern(np.array([0.0, 1.0]))
    assert gov.boundary_contact_budget > 0 and gov.surfing_alarm(cap=1.0)
    print("box governor selftest OK: on-envelope identical; off-envelope projected+demoted+annunciated; surfing metered.")
    print(f"  envelope toxic in [{gov.lo[1]:.3f}, {gov.hi[1]:.3f}]; cum boundary contact after 51 off-env cmds="
          f"{gov.boundary_contact_budget:.3f}; projections={gov.n_projections}/{gov.n_calls}")

    # ---- debt-1: JOINT-manifold governor catches diffuse beliefs the box governor passes (K=4, near-one-hot world) ----
    G = 4
    # natural beliefs: near-one-hot around random vertices (like the family env's filter output)
    nat = []
    for _ in range(3000):
        v = int(rng.integers(G)); b = rng.dirichlet(np.where(np.arange(G) == v, 12.0, 0.4))
        nat.append(b)
    nat = np.array(nat)
    box = BeliefGovernor.from_visited(nat, q=0.02)
    man = ManifoldGovernor.from_visited(nat, k=8, q=0.99)
    diffuse = np.full(G, 1.0 / G)                       # a C-5-style 'both' corner: uniform belief
    _, ann_box = box.govern(diffuse)
    _, ann_man = man.govern(diffuse)
    onehot = np.zeros(G); onehot[1] = 0.9; onehot[0] = onehot[2] = onehot[3] = 0.1 / 3     # natural-shaped belief
    _, ann_man_ok = man.govern(onehot)
    assert ann_box["status"] == "ON_ENVELOPE", "box governor PASSES the diffuse joint belief (the debt-1 gap)"
    assert ann_man["status"] == "OFF_MANIFOLD_PROJECTED", "manifold governor must CATCH the diffuse joint belief"
    assert ann_man_ok["status"] == "ON_MANIFOLD", "manifold governor must PASS a natural near-one-hot belief"
    print("manifold governor (debt-1) OK: box PASSES the diffuse uniform belief, manifold CATCHES it "
          f"(knn_dist={ann_man['knn_dist']} > thr={ann_man['thr']}); a natural near-one-hot belief passes "
          f"(knn_dist={ann_man_ok['knn_dist']}).")


if __name__ == "__main__":
    _selftest()
