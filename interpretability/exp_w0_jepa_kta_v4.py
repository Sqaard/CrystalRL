"""W0 v4 — KT-A with every referee fix: within-block features, fair arms, fair nulls.

ITERATION RECORD (all four preserved; the arc is a methodology result in itself):
  v1 predictor collapse (flattering-null artifact); v2 EMA/online mismatch + day-level overlap
  mechanics; v3 KILLED BY REVIEW on both legs — the linear arm was untuned (dev-tuned ridge
  alpha=100 PASSED the bar, but substantially mechanically: trailing-aggregate target features
  embed context days — the feature-level overlap artifact), and the JEPA arm lost v2's residual
  fix (its negative margin measured optimization failure; dev-best at epoch 7, 6s runtime).

V4 FIXES (each referee-mandated):
  * TARGET features are WITHIN-BLOCK only + end-of-block levels: [block ret sum, block realized
    vol, block max drawdown, dVIX over block, VIX_end, trend_end, turb_end, y10_end, vrp_end,
    ebp_end] — no trailing window crosses a block boundary, so persistence/ridge cannot harvest
    rolling-dropout mechanics; levels at block end are genuine new state.
  * NULL FAMILY per space: the better-on-DEV of {persistence (current block rep), train-mean};
    frozen per space before any HOLD read.
  * LINEAR ARM: ridge with alpha selected on DEV (grid 10^-1..10^4), in PCA-16-of-blocks space.
  * JEPA ARM: skip predictor s_hat = s_cur + f(s_ctx) (zero-init last layer), enc_t CHECKPOINTED
    with the best-epoch state, min 40 epochs before early stop (patience 25), block bootstrap
    length 12 (the context-overlap span).
  * Placebo: year-shuffled targets, unchanged.

KT-A BAR (pre-registered, unchanged in spirit): PASS iff on HOLD for some k in {1,2}:
  JEPA margin-vs-null > 0 with z >= 1.28 AND >= the linear arm's margin-vs-null AND
  placebo margin < half the real margin. KILL otherwise.
INTERPRETATION KEY (written before running): with within-block targets, block realized vol is
the one component KNOWN to be predictable (vol clustering) — and persistence carries it, so the
null is strong and honest. A margin ~0 means "nothing beyond vol-clustering persistence"; a
positive margin means structure beyond it. Either way this v4 read is the substrate verdict.

Run: python interpretability/exp_w0_jepa_kta_v4.py     (~10 min CPU)
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
from interpretability.exp_cl1_new_eyes_continual import load_v2  # noqa: E402
from interpretability.exp_w0_jepa_kta import vicreg  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402

OUT = HERE / "exp_w0_jepa_kta_v4_report.json"
B = 5
CTX = 12
DIM = 16
KS = (1, 2)
EPOCHS = 200
MIN_EPOCHS = 40
PATIENCE = 25
BATCH = 256
EMA_TAU = 0.996
SEEDS = (0, 1)
BOOT_BLOCK = 12


def build_block_features():
    """Per non-overlapping 5d block: within-block stats + end-of-block levels. No cross-boundary windows."""
    r, obs, rf = load_v2()
    idx = r.index
    ends = list(range(len(idx) - 1, B - 2, -B))[::-1]
    rows, b_end = [], []
    rv = r.to_numpy()
    for e in ends:
        s = e - B + 1
        if s < 0:
            continue
        blk_r = rv[s:e + 1]
        lvl = obs.iloc[e]
        lvl0 = obs.iloc[s]
        if not (np.isfinite(blk_r).all() and np.isfinite(lvl[["VIX", "SP500_Trend", "turbulence",
                                                              "10Y_Yield", "vrp", "ebp"]]).all()):
            rows.append(None); b_end.append(e); continue
        eqb = np.cumprod(1 + blk_r)
        ddb = float((eqb / np.maximum.accumulate(eqb) - 1).min())
        rows.append([float(blk_r.sum()), float(blk_r.std()), ddb,
                     float(lvl["VIX"] - lvl0["VIX"]), float(lvl["VIX"]), float(lvl["SP500_Trend"]),
                     float(lvl["turbulence"]), float(lvl["10Y_Yield"]), float(lvl["vrp"]),
                     float(lvl["ebp"])])
        b_end.append(e)
    return rows, np.array(b_end), idx


class EncB(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, DIM))
    def forward(self, x):
        return self.f(x)


class Agg(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(CTX * DIM, 128), nn.ReLU(), nn.Linear(128, DIM))
    def forward(self, s):
        return self.f(s)


class PredSkip(nn.Module):
    """s_hat = s_cur + f(s_ctx): starts at persistence, zero-init last layer (the v2 fix, kept)."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, DIM))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
    def forward(self, s_ctx, s_cur):
        return s_cur + self.f(s_ctx)


def train_v4(Xb, ctx_of, cur_of, tgt_of, tr_m, dev_m, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = Xb.shape[1]
    enc = EncB(d_in); enc_t = EncB(d_in); enc_t.load_state_dict(enc.state_dict())
    for p in enc_t.parameters():
        p.requires_grad_(False)
    agg = Agg(); preds = {k: PredSkip() for k in KS}
    params = (list(enc.parameters()) + list(agg.parameters())
              + [p for m in preds.values() for p in m.parameters()])
    opt = torch.optim.Adam(params, lr=1e-3)
    Xt = torch.tensor(Xb)
    tr_idx = np.where(tr_m)[0]
    best_dev, best_state, patience = np.inf, None, 0
    for ep in range(EPOCHS):
        perm = np.random.permutation(tr_idx)
        for b0 in range(0, len(perm), BATCH):
            bi = perm[b0:b0 + BATCH]
            if len(bi) < 32:
                continue
            sb = enc(Xt[ctx_of[bi].reshape(-1)]).reshape(len(bi), CTX, DIM)
            s_ctx = agg(sb.reshape(len(bi), -1))
            with torch.no_grad():
                s_cur = enc_t(Xt[cur_of[bi]])
            loss = 0.0
            for k in KS:
                with torch.no_grad():
                    s_tgt = enc_t(Xt[tgt_of[k][bi]])
                loss = loss + nn.functional.mse_loss(preds[k](s_ctx, s_cur), s_tgt)
            v, c = vicreg(sb.reshape(-1, DIM)); v2, c2 = vicreg(s_ctx)
            loss = loss + 5.0 * (v + v2) + 1.0 * (c + c2)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, pt in zip(enc.parameters(), enc_t.parameters()):
                    pt.mul_(EMA_TAU).add_(p, alpha=1 - EMA_TAU)
        with torch.no_grad():
            di = np.where(dev_m)[0]
            sb = enc(Xt[ctx_of[di].reshape(-1)]).reshape(len(di), CTX, DIM)
            s_ctx = agg(sb.reshape(len(di), -1))
            s_cur = enc_t(Xt[cur_of[di]])
            dl = float(sum(nn.functional.mse_loss(preds[k](s_ctx, s_cur), enc_t(Xt[tgt_of[k][di]]))
                           for k in KS))
        if dl < best_dev - 1e-5:
            best_dev, patience = dl, 0
            best_state = ({a: b.clone() for a, b in enc.state_dict().items()},
                          {a: b.clone() for a, b in enc_t.state_dict().items()},     # v4: checkpoint enc_t
                          {a: b.clone() for a, b in agg.state_dict().items()},
                          {k: {a: b.clone() for a, b in preds[k].state_dict().items()} for k in KS})
        elif ep >= MIN_EPOCHS:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is not None:
        enc.load_state_dict(best_state[0]); enc_t.load_state_dict(best_state[1])
        agg.load_state_dict(best_state[2])
        for k in KS:
            preds[k].load_state_dict(best_state[3][k])
    return enc, enc_t, agg, preds


def null_errors(S, cur_idx, tgt_idx, mean_vec):
    e_pers = ((S[cur_idx] - S[tgt_idx]) ** 2).sum(1)
    e_mean = ((mean_vec[None, :] - S[tgt_idx]) ** 2).sum(1)
    return e_pers, e_mean


def zmargin(e_pred, e_null):
    d = e_null - e_pred
    _, se = block_z(d, block=BOOT_BLOCK, n_boot=1000, seed=7)
    return float(d.mean() / (e_null.mean() + 1e-12)), float(d.mean() / se)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W0 v4 — KT-A, referee-fixed (within-block features, fair arms, fair nulls) ===")
    rows, b_end, idx = build_block_features()
    valid = [i for i, x in enumerate(rows) if x is not None]
    Xraw = np.array([rows[i] for i in valid], dtype=np.float32)
    remap = {i: j for j, i in enumerate(valid)}
    n = len(rows)
    samples = [i for i in range(CTX - 1, n - max(KS))
               if all(rows[j] is not None for j in list(range(i - CTX + 1, i + 1)) + [i + k for k in KS])]
    samples = np.array(samples)
    dates = idx[b_end[samples]]
    masks = {w: np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
             for w, (a, b) in dict(train=TRAIN, dev=DEV, hold=HOLD, oos=OOS).items()}
    mu = Xraw[[remap[i] for i in samples[masks["train"]]]].mean(0)
    sd = Xraw[[remap[i] for i in samples[masks["train"]]]].std(0) + 1e-9
    Xb = ((Xraw - mu) / sd).astype(np.float32)
    ctx_of = np.array([[remap[j] for j in range(i - CTX + 1, i + 1)] for i in samples])
    cur_of = ctx_of[:, -1]
    tgt_of = {k: np.array([remap[i + k] for i in samples]) for k in KS}
    print(f"  blocks {len(Xb)} x {Xb.shape[1]} | samples {len(samples)} "
          f"(tr {masks['train'].sum()} dev {masks['dev'].sum()} hold {masks['hold'].sum()} oos {masks['oos'].sum()})")

    rng = np.random.default_rng(1234)
    yrs = np.asarray(dates.year)
    uy = np.unique(yrs[masks["train"]]); pm = dict(zip(uy, rng.permutation(uy)))
    plc_tgt = {}
    for k in KS:
        arr = tgt_of[k].copy()
        for j in np.where(masks["train"])[0]:
            cands = np.where(yrs == pm.get(yrs[j], yrs[j]))[0]
            if len(cands):
                arr[j] = tgt_of[k][cands[j % len(cands)]]
        plc_tgt[k] = arr

    # ---- linear arm: PCA-16 + dev-tuned ridge; nulls chosen on dev per space ----
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    tri, di = np.where(masks["train"])[0], np.where(masks["dev"])[0]
    pca = PCA(n_components=min(DIM, Xb.shape[1]), random_state=0).fit(Xb[np.unique(ctx_of[tri].reshape(-1))])
    Sp = pca.transform(Xb).astype(np.float32)
    mean_p = Sp[np.unique(tgt_of[1][tri])].mean(0)
    lin = {}
    for k in KS:
        best = None
        for alpha in (0.1, 1, 10, 100, 1000, 10000):
            reg = Ridge(alpha=alpha).fit(Sp[ctx_of[tri]].reshape(len(tri), -1), Sp[tgt_of[k][tri]])
            e_dev = ((reg.predict(Sp[ctx_of[di]].reshape(len(di), -1)) - Sp[tgt_of[k][di]]) ** 2).sum(1)
            if best is None or e_dev.mean() < best[0]:
                best = (e_dev.mean(), alpha, reg)
        _, alpha, reg = best
        ep_dev, em_dev = null_errors(Sp, cur_of[di], tgt_of[k][di], mean_p)
        null_kind = "pers" if ep_dev.mean() <= em_dev.mean() else "mean"
        lin[k] = (reg, alpha, null_kind)
        print(f"  linear arm k={k}: alpha {alpha}, dev-chosen null = {null_kind}")

    report = {"preregistration": {"bar": "HOLD, some k: jepa margin>0 z>=1.28 AND >= linear margin AND placebo < 0.5x",
                                   "nulls": "better-on-DEV of {persistence, train-mean} per space",
                                   "boot_block": BOOT_BLOCK, "target_features": "within-block + end levels",
                                   "iteration_note": "v1-v3 artifacts documented in docstring; v4 = referee-fixed"},
              "linear_alpha": {str(k): lin[k][1] for k in KS}, "seeds": {}}
    agg_m = {k: [] for k in KS}; agg_z = {k: [] for k in KS}
    agg_pl = {k: [] for k in KS}; agg_li = {k: [] for k in KS}
    for seed in SEEDS:
        enc, enc_t, aggm, preds = train_v4(Xb, ctx_of, cur_of, tgt_of, masks["train"], masks["dev"], seed)
        encp, enc_tp, aggp, predsp = train_v4(Xb, ctx_of, cur_of, plc_tgt, masks["train"], masks["dev"], seed + 50)
        with torch.no_grad():
            S = enc_t(torch.tensor(Xb)).numpy()
            Spl = enc_tp(torch.tensor(Xb)).numpy()
        stds = S.std(0); lam = np.clip(np.linalg.eigvalsh(np.cov(S.T)), 0, None)
        mean_j = S[np.unique(tgt_of[1][tri])].mean(0)
        mean_jp = Spl[np.unique(tgt_of[1][tri])].mean(0)
        # dev null choice in the JEPA space
        srec = {"collapse": {"min_dim_std": round(float(stds.min()), 3),
                              "eff_rank": round(float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12)), 2)},
                "windows": {}}
        ep_dev, em_dev = null_errors(S, cur_of[di], tgt_of[1][di], mean_j)
        null_kind_j = "pers" if ep_dev.mean() <= em_dev.mean() else "mean"
        srec["jepa_null"] = null_kind_j
        for w in ("hold", "oos"):
            ii = np.where(masks[w])[0]
            with torch.no_grad():
                sb = enc(torch.tensor(Xb[ctx_of[ii].reshape(-1)])).reshape(len(ii), CTX, DIM)
                s_ctx = aggm(sb.reshape(len(ii), -1))
                s_cur = enc_t(torch.tensor(Xb[cur_of[ii]]))
                sbp = encp(torch.tensor(Xb[ctx_of[ii].reshape(-1)])).reshape(len(ii), CTX, DIM)
                s_ctxp = aggp(sbp.reshape(len(ii), -1))
                s_curp = enc_tp(torch.tensor(Xb[cur_of[ii]]))
            row = {}
            for k in KS:
                with torch.no_grad():
                    e_pred = ((preds[k](s_ctx, s_cur).numpy() - S[tgt_of[k][ii]]) ** 2).sum(1)
                    e_predp = ((predsp[k](s_ctxp, s_curp).numpy() - Spl[tgt_of[k][ii]]) ** 2).sum(1)
                ep_, em_ = null_errors(S, cur_of[ii], tgt_of[k][ii], mean_j)
                e_null = ep_ if null_kind_j == "pers" else em_
                epp_, emp_ = null_errors(Spl, cur_of[ii], tgt_of[k][ii], mean_jp)
                e_nullp = epp_ if null_kind_j == "pers" else emp_
                m, z = zmargin(e_pred, e_null)
                mp, _ = zmargin(e_predp, e_nullp)
                reg, alpha, nk = lin[k]
                e_lin = ((reg.predict(Sp[ctx_of[ii]].reshape(len(ii), -1)) - Sp[tgt_of[k][ii]]) ** 2).sum(1)
                epl_, eml_ = null_errors(Sp, cur_of[ii], tgt_of[k][ii], mean_p)
                e_nulll = epl_ if nk == "pers" else eml_
                ml, zl = zmargin(e_lin, e_nulll)
                row[k] = {"jepa_margin": round(m, 4), "z": round(z, 2), "placebo": round(mp, 4),
                          "linear_margin": round(ml, 4), "linear_z": round(zl, 2)}
                if w == "hold":
                    agg_m[k].append(m); agg_z[k].append(z); agg_pl[k].append(mp); agg_li[k].append(ml)
            srec["windows"][w] = row
            print(f"  seed {seed} {w}: " + " | ".join(
                f"k={k} jepa {row[k]['jepa_margin']:+.4f} z {row[k]['z']:+.2f} "
                f"plc {row[k]['placebo']:+.4f} lin {row[k]['linear_margin']:+.4f} (z {row[k]['linear_z']:+.2f})"
                for k in KS) + f" | null {null_kind_j} std_min {srec['collapse']['min_dim_std']}")
        report["seeds"][seed] = srec

    passes = {}
    for k in KS:
        m = float(np.mean(agg_m[k])); z = float(np.mean(agg_z[k]))
        mp = float(np.mean(agg_pl[k])); ml = float(np.mean(agg_li[k]))
        passes[k] = bool(m > 0 and z >= 1.28 and m >= ml and mp < 0.5 * m)
    kta = any(passes.values())
    lin_pass = any(float(np.mean(agg_li[k])) > 0 for k in KS)
    if kta:
        verdict = ("KT-A PASS — the JEPA arm beats the fair null family and the tuned linear arm "
                   "with a clean placebo; W1 (hierarchy + KT-B) unlocked.")
    elif lin_pass:
        verdict = ("KT-A SPLIT — the JEPA arm fails but the tuned LINEAR arm shows positive margin "
                   "over the fair null on within-block targets: L1 has (modest, linear) predictable "
                   "structure; the JEPA encoder adds nothing over PCA at this scale. Per the "
                   "methodology, W1 should build L2 on the LINEAR representation first and revisit "
                   "nonlinear encoders only with more data or features.")
    else:
        verdict = ("KT-A KILL (v4, referee-fixed harness) — neither arm beats the fair null on "
                   "within-block targets: no L1 latent predictability beyond the null family on "
                   "this substrate; the program's question moves to L2/KT-B as pre-registered.")
    report["bar_by_k"] = {str(k): {"jepa_margin": round(float(np.mean(agg_m[k])), 4),
                                    "z": round(float(np.mean(agg_z[k])), 2),
                                    "placebo": round(float(np.mean(agg_pl[k])), 4),
                                    "linear_margin": round(float(np.mean(agg_li[k])), 4),
                                    "pass": passes[k]} for k in KS}
    report["kta_pass"] = bool(kta)
    report["verdict"] = verdict
    report["runtime_s"] = int(time.time() - t0)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
