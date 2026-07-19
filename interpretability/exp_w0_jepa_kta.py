"""W0 — the first CRYSTAL-WORLD probe: a single-level JEPA (L1) on the v2 panel, judged by KT-A.

Per business/CRYSTAL_WORLD_METHODOLOGY.md (W0 + KT-A), pre-registered before running:

  MODEL: encoder Enc: 60d x 12-feature window -> s in R^16 (MLP), EMA target encoder (tau .996),
  per-horizon predictors P_h(s_t) for h in {1, 5} (L1 horizons; h=10 reported exploratory),
  VICReg-style anti-collapse (variance floor + covariance decorrelation) — G1 is architectural.
  Train 2010-18 (train-frozen feature scaling), DEV 2019-21 for early-stop selection,
  HOLD 2022-23 = the primary KT-A read, OOS 2024-26 = confirmation. 2 seeds, bar on the mean.

  KT-A METRICS (held-out, per window; v2 after the v1 iteration — see ITERATION NOTE):
    PRIMARY = ENERGY MARGIN: relative reduction of per-day squared latent error vs the
    persistence prediction (s_t), paired per day; the space-invariant comparison is each model's
    margin over ITS OWN persistence (JEPA space vs PCA-16+ridge space). SECONDARY (reported) =
    retrieval percentile skill.
  ITERATION NOTE (honest record): the v1 run scored the predictor at CHANCE retrieval (0.51) vs
  persistence 0.996 with placebo == real — a PREDICTOR COLLAPSE to the global mean (VICReg
  guarded the encoder, not the predictor output) plus a persistence-saturated retrieval metric on
  overlapping windows. v2 fixes: residual zero-init predictor (starts AT persistence) + the
  energy-margin primary metric. The substrate verdict comes from v2 only.

  KT-A BAR (PASS iff, on HOLD, for at least one h in {1,5}):
    (a) JEPA energy margin > 0 with paired block-bootstrap (20d, 1000) z >= 1.28;
    (b) JEPA margin >= the PCA+ridge margin (each vs its own persistence);
    (c) the shuffled-target placebo JEPA's margin < half the real JEPA margin.
  KILL otherwise: the substrate does not support L1 latent prediction beyond trivial dynamics —
  informative per the methodology (the interesting level moves to L2 where the HMM already works).

  Collapse diagnostics reported regardless: per-dim std floor, effective rank (participation
  ratio); a collapsed run voids the read (KT-E overlap).

Run: python interpretability/exp_w0_jepa_kta.py     (~10-15 min CPU)
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
from interpretability.exp_cl1_new_eyes_continual import load_v2, MACRO4  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402

OUT = HERE / "exp_w0_jepa_kta_report.json"
WIN = 60
DIM = 16
HS = (1, 5, 10)          # 10 = exploratory only, not in the bar
HS_BAR = (1, 5)
EPOCHS = 150
BATCH = 256
EMA_TAU = 0.996
SEEDS = (0, 1)


def build_features():
    r, obs, rf = load_v2()
    sigma = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean())
    eq = np.cumprod(1 + r.fillna(0)); dd = eq / np.maximum.accumulate(eq) - 1
    F = pd.DataFrame({"ret": r, "vix": obs["VIX"], "trend": obs["SP500_Trend"],
                      "turb": obs["turbulence"], "y10": obs["10Y_Yield"],
                      "vrp": obs["vrp"], "ebp": obs["ebp"],
                      "ret5": r.rolling(5).sum(), "ret21": r.rolling(21).sum(),
                      "ret63": r.rolling(63).sum(), "ewma_vol": sigma, "dd": dd}, index=r.index)
    return F


def make_windows(F):
    idx = F.index
    m_tr = np.asarray((idx >= pd.Timestamp(TRAIN[0])) & (idx <= pd.Timestamp(TRAIN[1])))
    mu = F[m_tr].mean(); sd = F[m_tr].std() + 1e-9
    Z = ((F - mu) / sd).to_numpy(dtype=np.float32)
    ok_row = np.isfinite(Z).all(axis=1)
    X, ts = [], []
    for t in range(WIN - 1, len(idx)):
        if ok_row[t - WIN + 1:t + 1].all():
            X.append(Z[t - WIN + 1:t + 1].ravel()); ts.append(t)
    return np.stack(X), np.array(ts), idx


class Enc(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                               nn.Linear(64, DIM))
    def forward(self, x):
        return self.f(x)


class Pred(nn.Module):
    """RESIDUAL predictor, zero-initialized last layer: starts AT persistence (f=0) and can only
    learn deviations — the v2 fix for the v1 run's predictor collapse to the global mean (the
    first run scored predictor=chance vs persistence=0.996; documented in the logbook)."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, DIM))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
    def forward(self, s):
        return s + self.f(s)


def vicreg(s):
    sd = torch.sqrt(s.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - sd).pow(2).mean()
    sc = s - s.mean(dim=0)
    cov = (sc.T @ sc) / (len(s) - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / DIM
    return var_loss, cov_loss


def train_jepa(X, ts, tr_mask, dev_mask, pair_map, seed, log_prefix=""):
    """pair_map[h][i] = index j into X of the target for sample i at horizon h (or -1)."""
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = X.shape[1]
    enc = Enc(d_in); enc_t = Enc(d_in); enc_t.load_state_dict(enc.state_dict())
    for p in enc_t.parameters():
        p.requires_grad_(False)
    preds = {h: Pred() for h in HS}
    params = list(enc.parameters()) + [p for m in preds.values() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    Xt = torch.tensor(X)
    tr_idx = np.where(tr_mask)[0]
    best_dev, best_state, patience = np.inf, None, 0
    for ep in range(EPOCHS):
        perm = np.random.permutation(tr_idx)
        for b0 in range(0, len(perm), BATCH):
            bi = perm[b0:b0 + BATCH]
            keep = [i for i in bi if all(pair_map[h][i] >= 0 for h in HS)]
            if len(keep) < 32:
                continue
            keep = np.array(keep)
            s = enc(Xt[keep])
            loss = 0.0
            for h in HS:
                tj = torch.tensor(pair_map[h][keep])
                with torch.no_grad():
                    s_tgt = enc_t(Xt[tj])
                loss = loss + nn.functional.mse_loss(preds[h](s), s_tgt)
            v, c = vicreg(s)
            loss = loss + 5.0 * v + 1.0 * c
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, pt in zip(enc.parameters(), enc_t.parameters()):
                    pt.mul_(EMA_TAU).add_(p, alpha=1 - EMA_TAU)
        # dev early-stop on h=5 prediction mse
        with torch.no_grad():
            di = np.where(dev_mask & (pair_map[5] >= 0))[0]
            s = enc(Xt[di]); s_tgt = enc_t(Xt[torch.tensor(pair_map[5][di])])
            dl = float(nn.functional.mse_loss(preds[5](s), s_tgt))
        if dl < best_dev - 1e-5:
            best_dev, patience = dl, 0
            best_state = ({k: v.clone() for k, v in enc.state_dict().items()},
                          {h: {k: v.clone() for k, v in preds[h].state_dict().items()} for h in HS})
        else:
            patience += 1
            if patience >= 15:
                break
    if best_state is not None:
        enc.load_state_dict(best_state[0])
        for h in HS:
            preds[h].load_state_dict(best_state[1][h])
    return enc, preds


def retrieval_skill(S_pred, S_true_all, true_pos):
    """S_pred: predictions for eval samples; S_true_all: representations of ALL days in the
    window; true_pos: index of each sample's true target within S_true_all. Returns per-sample
    skill = 1 - percentile rank of the true target."""
    d = ((S_pred[:, None, :] - S_true_all[None, :, :]) ** 2).sum(-1)
    ranks = (d < d[np.arange(len(S_pred)), true_pos][:, None]).sum(1)
    return 1.0 - ranks / (S_true_all.shape[0] - 1)


def eval_window(enc, preds, X, ts, mask, pair_map):
    """Per-h: (per-day squared error of predictor, of persistence) — the ENERGY-margin primary
    metric (relative error reduction vs persistence), plus retrieval skills as secondary."""
    with torch.no_grad():
        S_all = enc(torch.tensor(X)).numpy()
    out = {}
    for h in HS:
        ii = np.where(mask & (pair_map[h] >= 0))[0]
        tgt = pair_map[h][ii]
        pool_idx = np.unique(tgt)
        pos = np.searchsorted(pool_idx, tgt)
        S_pool = S_all[pool_idx]
        with torch.no_grad():
            S_hat = preds[h](torch.tensor(S_all[ii])).numpy()
        e_pred = ((S_hat - S_all[tgt]) ** 2).sum(1)
        e_pers = ((S_all[ii] - S_all[tgt]) ** 2).sum(1)
        sk_pred = retrieval_skill(S_hat, S_pool, pos)
        sk_pers = retrieval_skill(S_all[ii], S_pool, pos)
        out[h] = (e_pred, e_pers, sk_pred, sk_pers)
    return out


def pca_baseline(X, tr_mask, mask_eval, pair_map, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    pca = PCA(n_components=DIM, random_state=seed).fit(X[tr_mask])
    S_all = pca.transform(X).astype(np.float32)
    out = {}
    for h in HS:
        tri = np.where(tr_mask & (pair_map[h] >= 0))[0]
        reg = Ridge(alpha=1.0).fit(S_all[tri], S_all[pair_map[h][tri]])
        ii = np.where(mask_eval & (pair_map[h] >= 0))[0]
        tgt = pair_map[h][ii]
        S_hat = reg.predict(S_all[ii]).astype(np.float32)
        e_pred = ((S_hat - S_all[tgt]) ** 2).sum(1)
        e_pers = ((S_all[ii] - S_all[tgt]) ** 2).sum(1)
        out[h] = (e_pred, e_pers)
    return out


def zmargin(e_pred, e_pers):
    """Energy margin: relative error reduction vs persistence, with a paired block-boot z on the
    per-day error differences (positive margin = predictor better)."""
    d = e_pers - e_pred
    _, se = block_z(d, block=20, n_boot=1000, seed=7)
    rel = float(d.mean() / (e_pers.mean() + 1e-12))
    return rel, float(d.mean() / se)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W0 — single-level JEPA probe + KT-A (CRYSTAL-WORLD first read) ===")
    F = build_features()
    X, ts, idx = make_windows(F)
    dates = idx[ts]
    print(f"  windows: {len(X)} x {X.shape[1]} ({dates[0].date()} -> {dates[-1].date()})")
    pos_of_t = {t: i for i, t in enumerate(ts)}
    pair_map = {h: np.array([pos_of_t.get(t + h, -1) for t in ts]) for h in HS}
    masks = {}
    for name, (a, b) in (("train", TRAIN), ("dev", DEV), ("hold", HOLD), ("oos", OOS)):
        masks[name] = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))

    # placebo pair map: year-block-shuffled targets (train only needs it, but eval uses real)
    rng = np.random.default_rng(1234)
    years = dates.year.to_numpy() if hasattr(dates.year, "to_numpy") else np.asarray(dates.year)
    uy = np.unique(years[masks["train"]])
    perm_years = dict(zip(uy, rng.permutation(uy)))
    plc_pair = {}
    for h in HS:
        arr = pair_map[h].copy()
        for i, t in enumerate(ts):
            if not masks["train"][i] or arr[i] < 0:
                continue
            y_src = perm_years[years[i]] if years[i] in perm_years else years[i]
            cands = np.where((years == y_src) & (pair_map[h] >= 0))[0]
            if len(cands):
                arr[i] = pair_map[h][cands[i % len(cands)]]
        plc_pair[h] = arr

    report = {"preregistration": {"bar": "on HOLD, some h in {1,5}: pred>pers z>=1.28 AND jepa margin >= pca margin AND placebo margin < 0.5x real",
                                   "dims": DIM, "window_days": WIN, "seeds": list(SEEDS),
                                   "collapse_guard": "per-dim std + effective rank reported"},
              "seeds": {}}
    margins_hold = {h: [] for h in HS_BAR}
    z_hold = {h: [] for h in HS_BAR}
    plc_margins = {h: [] for h in HS_BAR}
    pca_margins = {h: [] for h in HS_BAR}
    for seed in SEEDS:
        enc, preds = train_jepa(X, ts, masks["train"], masks["dev"], pair_map, seed)
        with torch.no_grad():
            S = enc(torch.tensor(X)).numpy()
        stds = S.std(0)
        lam = np.linalg.eigvalsh(np.cov(S.T)); lam = np.clip(lam, 0, None)
        eff_rank = float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))
        srec = {"collapse": {"min_dim_std": round(float(stds.min()), 3),
                              "mean_dim_std": round(float(stds.mean()), 3),
                              "eff_rank": round(eff_rank, 2)}, "windows": {}}
        enc_p, preds_p = train_jepa(X, ts, masks["train"], masks["dev"], plc_pair, seed + 50)
        for wname in ("hold", "oos"):
            ev = eval_window(enc, preds, X, ts, masks[wname], pair_map)
            ev_p = eval_window(enc_p, preds_p, X, ts, masks[wname], pair_map)
            pca_w = pca_baseline(X, masks["train"], masks[wname], pair_map)
            row = {}
            for h in HS:
                m, z = zmargin(ev[h][0], ev[h][1])
                mp, _ = zmargin(ev_p[h][0], ev_p[h][1])
                mpca, _ = zmargin(*pca_w[h])
                row[h] = {"energy_margin_rel": round(m, 4), "z": round(z, 2),
                          "placebo_margin": round(mp, 4), "pca_margin": round(mpca, 4),
                          "retr_pred": round(float(ev[h][2].mean()), 4),
                          "retr_pers": round(float(ev[h][3].mean()), 4)}
                if wname == "hold" and h in HS_BAR:
                    margins_hold[h].append(m); z_hold[h].append(z)
                    plc_margins[h].append(mp); pca_margins[h].append(mpca)
            srec["windows"][wname] = row
            print(f"  seed {seed} {wname}: " + " | ".join(
                f"h={h} m {row[h]['energy_margin_rel']:+.4f} z {row[h]['z']:+.2f} "
                f"plc {row[h]['placebo_margin']:+.4f} pca {row[h]['pca_margin']:+.4f} "
                f"retr {row[h]['retr_pred']:.3f}/{row[h]['retr_pers']:.3f}" for h in HS))
        report["seeds"][seed] = srec

    passes = {}
    for h in HS_BAR:
        m = float(np.mean(margins_hold[h])); z = float(np.mean(z_hold[h]))
        mp = float(np.mean(plc_margins[h])); mpca = float(np.mean(pca_margins[h]))
        passes[h] = bool(z >= 1.28 and m >= mpca and (mp < 0.5 * m if m > 0 else True) and m > 0)
    kta = any(passes.values())
    verdict = ("KT-A PASS — the L1 JEPA predictor beats persistence and the linear baseline with a "
               "clean placebo on the frozen window; W1 (hierarchy + KT-B) is unlocked."
               if kta else
               "KT-A KILL — L1 latent prediction does not beat trivial dynamics on the frozen "
               "window (per prereg this is informative, not embarrassing: the daily micro-level is "
               "where VoI~0 lives; the program's interesting question moves to L2/KT-B, and the W1 "
               "design should build L2 on slower pooled inputs rather than expecting L1 skill).")
    report["bar_by_h"] = {str(h): {"mean_margin": round(float(np.mean(margins_hold[h])), 4),
                                    "mean_z": round(float(np.mean(z_hold[h])), 2),
                                    "mean_placebo": round(float(np.mean(plc_margins[h])), 4),
                                    "mean_pca_margin": round(float(np.mean(pca_margins[h])), 4),
                                    "pass": passes[h]} for h in HS_BAR}
    report["kta_pass"] = bool(kta)
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
