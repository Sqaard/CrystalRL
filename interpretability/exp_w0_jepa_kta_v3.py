"""W0 v3 — the CORRECT JEPA formulation for KT-A: disjoint future-block targets.

ITERATION RECORD (the honest chain, all preserved):
  v1: predictor collapsed to the global mean (VICReg on encoder only) + retrieval saturated by
      window overlap. Not a substrate verdict.
  v2: residual predictor exposed TWO more artifacts: (a) train/eval target mismatch (EMA targets
      in training, online targets in eval) made the JEPA margin meaninglessly negative; (b) the
      PCA+ridge margin +0.84 was MECHANICAL — overlapping 60d windows make s_{t+1} a nearly
      deterministic linear shift of s_t: "predictability" with zero market content (the G8 /
      Boudoukh overlap artifact in ML clothing). Not a substrate verdict either.
  v3 (this file): the I-JEPA-faithful design — context and target are DISJOINT. The market is
      chopped into non-overlapping 5-day blocks; one block encoder EncB (trained through the
      context path; EMA copy for targets) maps each block (5d x 12 feats) -> R^16; the context
      aggregator maps the last 12 block representations -> s_ctx; predictors P_k forecast the
      representation of the k-th FUTURE block (k=1: days t+1..t+5; k=2: days t+6..t+10).
      Persistence null = the CURRENT block's (EMA) representation. Linear baseline = PCA-16 of
      raw block vectors + ridge from the stacked last-12-block PCA reps, same-space persistence.
      Placebo = year-block-shuffled targets. All evaluation in the EMA-encoder space (consistent
      with training targets).

KT-A BAR (unchanged in spirit, pre-registered): PASS iff on HOLD for some k in {1,2}:
  (a) energy margin (relative error reduction vs persistence) > 0 with paired block-boot
      (4 blocks ~ 20d, 1000) z >= 1.28;
  (b) JEPA margin >= PCA+ridge margin (each vs its own-space persistence);
  (c) placebo margin < half the real margin.
KILL otherwise — and after v1/v2's artifacts, a v3 KILL is a real substrate verdict.

Run: python interpretability/exp_w0_jepa_kta_v3.py     (~10 min CPU)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
from interpretability.exp_w0_jepa_kta import build_features, vicreg  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402

OUT = HERE / "exp_w0_jepa_kta_v3_report.json"
B = 5                       # block length, days
CTX = 12                    # context = last 12 blocks (60 days)
DIM = 16
KS = (1, 2)                 # predict the k-th future block
EPOCHS = 200
BATCH = 256
EMA_TAU = 0.996
SEEDS = (0, 1)


def make_blocks(F):
    idx = F.index
    m_tr = np.asarray((idx >= pd.Timestamp(TRAIN[0])) & (idx <= pd.Timestamp(TRAIN[1])))
    mu = F[m_tr].mean(); sd = F[m_tr].std() + 1e-9
    Z = ((F - mu) / sd).to_numpy(dtype=np.float32)
    ok = np.isfinite(Z).all(axis=1)
    blocks, b_end = [], []
    t = len(idx) - 1
    ends = list(range(len(idx) - 1, B - 2, -B))[::-1]          # non-overlapping 5d blocks
    for e in ends:
        s = e - B + 1
        if s >= 0 and ok[s:e + 1].all():
            blocks.append(Z[s:e + 1].ravel()); b_end.append(e)
        else:
            blocks.append(None); b_end.append(e)
    return blocks, np.array(b_end), idx


class EncB(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, DIM))
    def forward(self, x):
        return self.f(x)


class Agg(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(CTX * DIM, 128), nn.ReLU(), nn.Linear(128, DIM))
    def forward(self, s):
        return self.f(s)


class PredK(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, DIM))
    def forward(self, s):
        return self.f(s)


def build_samples(blocks, b_end, idx):
    """Sample i = context blocks [i-CTX+1..i], targets blocks i+k. Valid iff all present."""
    n = len(blocks)
    samples = []
    for i in range(CTX - 1, n - max(KS)):
        ctx_ids = list(range(i - CTX + 1, i + 1))
        if any(blocks[j] is None for j in ctx_ids):
            continue
        if any(blocks[i + k] is None for k in KS):
            continue
        samples.append(i)
    return samples


def window_mask(samples, b_end, idx, a, b):
    d = idx[b_end[samples]]
    return np.asarray((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b)))


def train_v3(Xb, samples, ctx_of, tgt_of, tr_m, dev_m, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = Xb.shape[1]
    enc = EncB(d_in); enc_t = EncB(d_in); enc_t.load_state_dict(enc.state_dict())
    for p in enc_t.parameters():
        p.requires_grad_(False)
    agg = Agg(); preds = {k: PredK() for k in KS}
    params = (list(enc.parameters()) + list(agg.parameters())
              + [p for m in preds.values() for p in m.parameters()])
    opt = torch.optim.Adam(params, lr=1e-3)
    Xt = torch.tensor(Xb)
    tr_idx = np.where(tr_m)[0]
    best = (np.inf, None, 0)
    for ep in range(EPOCHS):
        perm = np.random.permutation(tr_idx)
        for b0 in range(0, len(perm), BATCH):
            bi = perm[b0:b0 + BATCH]
            if len(bi) < 32:
                continue
            sb = enc(Xt[ctx_of[bi].reshape(-1)]).reshape(len(bi), CTX, DIM)
            s_ctx = agg(sb.reshape(len(bi), -1))
            loss = 0.0
            for k in KS:
                with torch.no_grad():
                    s_tgt = enc_t(Xt[tgt_of[k][bi]])
                loss = loss + nn.functional.mse_loss(preds[k](s_ctx), s_tgt)
            v, c = vicreg(sb.reshape(-1, DIM))
            v2, c2 = vicreg(s_ctx)
            loss = loss + 5.0 * (v + v2) + 1.0 * (c + c2)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, pt in zip(enc.parameters(), enc_t.parameters()):
                    pt.mul_(EMA_TAU).add_(p, alpha=1 - EMA_TAU)
        with torch.no_grad():
            di = np.where(dev_m)[0]
            sb = enc(Xt[ctx_of[di].reshape(-1)]).reshape(len(di), CTX, DIM)
            s_ctx = agg(sb.reshape(len(di), -1))
            dl = float(sum(nn.functional.mse_loss(preds[k](s_ctx), enc_t(Xt[tgt_of[k][di]]))
                           for k in KS))
        if dl < best[0] - 1e-5:
            best = (dl, ({k2: v2.clone() for k2, v2 in enc.state_dict().items()},
                          {k2: v2.clone() for k2, v2 in agg.state_dict().items()},
                          {k: {k2: v2.clone() for k2, v2 in preds[k].state_dict().items()} for k in KS}), 0)
        else:
            best = (best[0], best[1], best[2] + 1)
            if best[2] >= 20:
                break
    if best[1] is not None:
        enc.load_state_dict(best[1][0]); agg.load_state_dict(best[1][1])
        for k in KS:
            preds[k].load_state_dict(best[1][2][k])
    return enc, enc_t, agg, preds


def eval_v3(enc, enc_t, agg, preds, Xb, ctx_of, tgt_of, m):
    with torch.no_grad():
        S_t = enc_t(torch.tensor(Xb)).numpy()                    # EMA space — consistent with training
        ii = np.where(m)[0]
        sb = enc(torch.tensor(Xb[ctx_of[ii].reshape(-1)])).reshape(len(ii), CTX, DIM)
        s_ctx = agg(sb.reshape(len(ii), -1))
        out = {}
        for k in KS:
            s_hat = preds[k](s_ctx).numpy()
            tgt = S_t[tgt_of[k][ii]]
            cur = S_t[ctx_of[ii][:, -1]]                          # persistence: current block's EMA rep
            e_pred = ((s_hat - tgt) ** 2).sum(1)
            e_pers = ((cur - tgt) ** 2).sum(1)
            out[k] = (e_pred, e_pers)
    return out


def pca_arm(Xb, ctx_of, tgt_of, tr_m, m_eval):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    tri = np.where(tr_m)[0]
    pca = PCA(n_components=DIM, random_state=0).fit(Xb[np.unique(ctx_of[tri].reshape(-1))])
    S = pca.transform(Xb).astype(np.float32)
    out = {}
    for k in KS:
        Xtr = S[ctx_of[tri]].reshape(len(tri), -1)
        reg = Ridge(alpha=1.0).fit(Xtr, S[tgt_of[k][tri]])
        ii = np.where(m_eval)[0]
        s_hat = reg.predict(S[ctx_of[ii]].reshape(len(ii), -1)).astype(np.float32)
        tgt = S[tgt_of[k][ii]]; cur = S[ctx_of[ii][:, -1]]
        out[k] = (((s_hat - tgt) ** 2).sum(1), ((cur - tgt) ** 2).sum(1))
    return out


def zmargin(e_pred, e_pers):
    d = e_pers - e_pred
    _, se = block_z(d, block=4, n_boot=1000, seed=7)             # 4 blocks ~ 20 trading days
    return float(d.mean() / (e_pers.mean() + 1e-12)), float(d.mean() / se)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W0 v3 — disjoint-block JEPA + KT-A ===")
    F = build_features()
    blocks, b_end, idx = make_blocks(F)
    valid = [i for i, bl in enumerate(blocks) if bl is not None]
    Xb = np.stack([blocks[i] for i in valid]).astype(np.float32)
    remap = {i: j for j, i in enumerate(valid)}
    n = len(blocks)
    samples = []
    for i in range(CTX - 1, n - max(KS)):
        ids = list(range(i - CTX + 1, i + 1)) + [i + k for k in KS]
        if all(blocks[j] is not None for j in ids):
            samples.append(i)
    samples = np.array(samples)
    ctx_of = np.array([[remap[j] for j in range(i - CTX + 1, i + 1)] for i in samples])
    tgt_of = {k: np.array([remap[i + k] for i in samples]) for k in KS}
    print(f"  blocks: {len(Xb)} x {Xb.shape[1]} | samples: {len(samples)}")
    masks = {w: window_mask(np.arange(len(samples)), b_end[samples], idx,
                             *dict(train=TRAIN, dev=DEV, hold=HOLD, oos=OOS)[w])
             for w in ("train", "dev", "hold", "oos")}
    # placebo: year-block-shuffled targets within train
    rng = np.random.default_rng(1234)
    yrs = np.asarray(idx[b_end[samples]].year)
    uy = np.unique(yrs[masks["train"]]); pm = dict(zip(uy, rng.permutation(uy)))
    plc_tgt = {}
    for k in KS:
        arr = tgt_of[k].copy()
        for j in np.where(masks["train"])[0]:
            cands = np.where(yrs == pm.get(yrs[j], yrs[j]))[0]
            if len(cands):
                arr[j] = tgt_of[k][cands[j % len(cands)]]
        plc_tgt[k] = arr

    report = {"preregistration": {"bar": "HOLD, some k in {1,2}: margin>0 z>=1.28 AND >= pca margin AND placebo < 0.5x",
                                   "design": "disjoint 5d blocks, ctx=12 blocks, EMA-space eval",
                                   "iteration_note": "v1 predictor collapse; v2 EMA/online mismatch + overlap artifact — see docstring"},
              "seeds": {}}
    agg_m = {k: [] for k in KS}; agg_z = {k: [] for k in KS}
    agg_p = {k: [] for k in KS}; agg_pca = {k: [] for k in KS}
    for seed in SEEDS:
        enc, enc_t, agg, preds = train_v3(Xb, samples, ctx_of, tgt_of, masks["train"], masks["dev"], seed)
        with torch.no_grad():
            S = enc_t(torch.tensor(Xb)).numpy()
        stds = S.std(0); lam = np.clip(np.linalg.eigvalsh(np.cov(S.T)), 0, None)
        srec = {"collapse": {"min_dim_std": round(float(stds.min()), 3),
                              "eff_rank": round(float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12)), 2)},
                "windows": {}}
        enc_p, enc_tp, agg_p_m, preds_p = train_v3(Xb, samples, ctx_of, plc_tgt, masks["train"], masks["dev"], seed + 50)
        for w in ("hold", "oos"):
            ev = eval_v3(enc, enc_t, agg, preds, Xb, ctx_of, tgt_of, masks[w])
            evp = eval_v3(enc_p, enc_tp, agg_p_m, preds_p, Xb, ctx_of, tgt_of, masks[w])
            pc = pca_arm(Xb, ctx_of, tgt_of, masks["train"], masks[w])
            row = {}
            for k in KS:
                m, z = zmargin(*ev[k]); mp, _ = zmargin(*evp[k]); mpc, _ = zmargin(*pc[k])
                row[k] = {"margin": round(m, 4), "z": round(z, 2),
                          "placebo": round(mp, 4), "pca": round(mpc, 4)}
                if w == "hold":
                    agg_m[k].append(m); agg_z[k].append(z); agg_p[k].append(mp); agg_pca[k].append(mpc)
            srec["windows"][w] = row
            print(f"  seed {seed} {w}: " + " | ".join(
                f"k={k} m {row[k]['margin']:+.4f} z {row[k]['z']:+.2f} plc {row[k]['placebo']:+.4f} "
                f"pca {row[k]['pca']:+.4f}" for k in KS)
                + f" | collapse std_min {srec['collapse']['min_dim_std']} er {srec['collapse']['eff_rank']}")
        report["seeds"][seed] = srec

    passes = {}
    for k in KS:
        m = float(np.mean(agg_m[k])); z = float(np.mean(agg_z[k]))
        mp = float(np.mean(agg_p[k])); mpc = float(np.mean(agg_pca[k]))
        passes[k] = bool(m > 0 and z >= 1.28 and m >= mpc and (mp < 0.5 * m))
    kta = any(passes.values())
    verdict = ("KT-A PASS — disjoint-block latent prediction beats persistence and the linear "
               "baseline with a clean placebo; W1 (L2/L3 hierarchy + KT-B) is unlocked."
               if kta else
               "KT-A KILL (v3, artifact-free) — the L1 daily micro-level does not support latent "
               "prediction beyond trivial dynamics: the third independent confirmation of the "
               "substrate's VoI~0 at the fine level, now in representation space. Per the "
               "methodology this moves the program's question to L2 (KT-B on slower pooled "
               "inputs), where the certified regime belief already proves skill exists.")
    report["bar_by_k"] = {str(k): {"mean_margin": round(float(np.mean(agg_m[k])), 4),
                                    "mean_z": round(float(np.mean(agg_z[k])), 2),
                                    "mean_placebo": round(float(np.mean(agg_p[k])), 4),
                                    "mean_pca": round(float(np.mean(agg_pca[k])), 4),
                                    "pass": passes[k]} for k in KS}
    report["kta_pass"] = bool(kta)
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
