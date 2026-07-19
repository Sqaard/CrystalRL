"""W2 (part 1) — the paper-probe loop replay: perceive -> predict -> compare, with detour tests
and the action-conditioning read.

Per business/CRYSTAL_WORLD_METHODOLOGY.md §4 + W2. Three pre-registered reads:

1. DETOUR TESTS (G3, the compare step as a schema-violation detector): replay the loop over
   2019-2026 with W1-v2-protocol encoders/predictors (fit TRAIN 2010-18, tuned DEV); per L1 block
   compute the SURPRISE RATIO = e_pred / median_train(e_pred). Break windows (fixed before
   running): COVID 2020-02-15..2020-04-30; RATES 2022-01-01..2022-06-30; TARIFF
   2025-03-15..2025-05-15. PASS iff the median surprise in >=2 of 3 break windows exceeds the
   calm-period median with a block-bootstrap z >= 2 — i.e., prediction error IS the salient
   schema-violation signal the enrichment loop needs (Tse et al., theory §2). Disclosure: 2020
   lies in the DEV tuning span — the COVID read is the weakest of the three; RATES (hold) and
   TARIFF (oos) are clean.
2. ACTION-CONDITIONING (G5 made concrete): augment the L1 k in {1,4} ridge context with the
   certified rule's per-block action stats (mean exposure, switch count). PREREGISTERED
   EXPECTATION = NULL: on historical data a_t is INFORMATION-DERIVED (a function of the belief),
   not a causal footprint, so any margin gain must be matched by the shuffled-action capacity
   placebo; a gain above the placebo would indicate re-encoded belief info, not footprint.
   The interface (predictor accepts an action channel) is the deliverable; the honest read is
   whether the channel is inert on history, as theory says it must be at our size.
3. EPISODIC STORE: every replay step appends a record {date, level, k, energy, null_energy,
   surprise_ratio, action} to data/_crystal_world/episodic_replay.jsonl — the CLS episodic
   store's first real artifact (schema v0; the daily hook rides the live_executor paper track
   in W3).

Run: python interpretability/exp_w2_probe_loop.py     (~2 min, sklearn only)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.exp_cl1_new_eyes_continual import load_v2, fit_belief, MACRO4  # noqa: E402
from interpretability.exp_w1_ktb_v2 import build_blocks, assemble  # noqa: E402
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from sklearn.linear_model import Ridge

OUT = HERE / "exp_w2_probe_loop_report.json"
STORE = ROOT / "data" / "_crystal_world"; STORE.mkdir(parents=True, exist_ok=True)
EPISODIC = STORE / "episodic_replay.jsonl"
CERT = (0.66, 0.74)
BREAKS = {"COVID_dev": ("2020-02-15", "2020-04-30"), "RATES_hold": ("2022-01-01", "2022-06-30"),
          "TARIFF_oos": ("2025-03-15", "2025-05-15")}
CELLS = {("L1", 1): {"B": 5, "D": 16, "CTX": 2, "alpha": 100},
         ("L1", 4): {"B": 5, "D": 16, "CTX": 4, "alpha": 1000},
         ("L2", 1): {"B": 21, "D": 8, "CTX": 2, "alpha": 1000}}
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)


def cert_exposure_path(r, obs, rf):
    """Daily exposure path of the certified rule (eyes4 belief, CERT config, H=10, lag 1)."""
    bel = fit_belief(obs[MACRO4], TRAIN[1])
    sig = bel.to_numpy()[:-1]
    ex, exs = 1.0, np.empty(len(sig))
    for t in range(len(sig)):
        if t % 10 == 0:
            ex = CERT[1] if sig[t] > CERT[0] else 1.0
        exs[t] = ex
    return pd.Series(np.concatenate([[1.0], exs]), index=r.index)


def fit_cell(bl, asm, k, alpha, extra=None, extra_tr_shuffle_seed=None):
    """Ridge on [ctx (+ extra)] with purged train; returns per-sample eval errors + train medians."""
    S = bl["S"]
    X = S[asm["ctx_of"]].reshape(len(asm["samples"]), -1)
    if extra is not None:
        ex = extra.copy()
        if extra_tr_shuffle_seed is not None:
            rng = np.random.default_rng(extra_tr_shuffle_seed)
            ex = ex[rng.permutation(len(ex))]
        X = np.concatenate([X, ex], axis=1)
    tr_all = np.where(asm["masks"]["train"])[0]
    purged = tr_all[asm["tgt_dates"][k][tr_all] <= pd.Timestamp(TRAIN[1])]
    reg = Ridge(alpha=alpha).fit(X[purged], S[asm["tgt_of"][k][purged]])
    e = ((reg.predict(X) - S[asm["tgt_of"][k]]) ** 2).sum(1)
    med_tr = float(np.median(e[purged]))
    return e, med_tr, reg


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    t0 = time.time()
    print("=== W2 — probe-loop replay: detours + action-conditioning + episodic store ===")
    r, obs, rf = load_v2()
    expo = cert_exposure_path(r, obs, rf)

    records = []
    detour, action_read = {}, {}
    for (lname, k), cfg in CELLS.items():
        bl = build_blocks(r, obs, cfg["B"], cfg["D"])
        asm = assemble(bl, cfg["CTX"], (1, 2, 4, 8))
        e, med_tr, _ = fit_cell(bl, asm, k, cfg["alpha"])
        surprise = e / (med_tr + 1e-12)
        dates = asm["dates"]
        eval_m = np.asarray(dates >= pd.Timestamp(DEV[0]))
        # persistence null energy for the store
        S = bl["S"]
        e_null = ((S[asm["cur_of"]] - S[asm["tgt_of"][k]]) ** 2).sum(1)
        # episodic records
        for j in np.where(eval_m)[0]:
            d = dates[j]
            a_blk = float(expo.loc[:d].tail(cfg["B"]).mean())
            records.append({"date": str(d.date()), "level": lname, "k": int(k),
                            "energy": round(float(e[j]), 4), "null_energy": round(float(e_null[j]), 4),
                            "surprise_ratio": round(float(surprise[j]), 3),
                            "action_mean_exposure": round(a_blk, 3)})
        # detour stats on the L1 k=1 cell (the finest clock) and L2 k=1
        if (lname, k) in (("L1", 1), ("L2", 1)):
            calm_m = eval_m.copy()
            rows = {}
            for bname, (a, b) in BREAKS.items():
                m = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
                calm_m &= ~m
                rows[bname] = m
            calm_med = float(np.median(surprise[calm_m]))
            det = {"calm_median": round(calm_med, 3), "n_calm": int(calm_m.sum())}
            for bname, m in rows.items():
                blk = max(2, min(4, m.sum() // 2))
                if m.sum() - blk + 1 < 2:                     # degenerate bootstrap (CL-1b rule):
                    det[bname] = {"n": int(m.sum()),          # a single start = SE 0 = fake z
                                  "median": round(float(np.median(surprise[m])), 3) if m.sum() else None,
                                  "z": None, "note": "n too small for block bootstrap — z nulled"}
                    continue
                d_arr = surprise[m] - calm_med
                _, se = block_z(d_arr, block=blk, n_boot=1000, seed=7)
                det[bname] = {"n": int(m.sum()), "median": round(float(np.median(surprise[m])), 3),
                              "z": round(float(d_arr.mean() / se), 2)}
            detour[f"{lname}_k{k}"] = det
            print(f"  detour {lname} k={k}: calm med {det['calm_median']} | " +
                  " ".join(f"{b}: med {det[b]['median']} z {det[b]['z']}" for b in BREAKS))

        # action-conditioning read on L1 cells
        if lname == "L1":
            a_stats = []
            for j in range(len(asm["samples"])):
                d = dates[j]
                w = expo.loc[:d].tail(cfg["B"] * cfg["CTX"])
                a_stats.append([float(w.mean()), float((w.diff().abs() > 1e-9).sum())])
            a_stats = np.asarray(a_stats, dtype=np.float32)
            hold_m = np.asarray((dates >= pd.Timestamp(HOLD[0])) & (dates <= pd.Timestamp(HOLD[1])))
            e_plain, _, _ = fit_cell(bl, asm, k, cfg["alpha"])
            e_act, _, _ = fit_cell(bl, asm, k, cfg["alpha"], extra=a_stats)
            e_plc, _, _ = fit_cell(bl, asm, k, cfg["alpha"], extra=a_stats, extra_tr_shuffle_seed=99)
            def marg(e_variant):
                d_arr = e_plain[hold_m] - e_variant[hold_m]
                _, se = block_z(d_arr, block=cfg["CTX"], n_boot=1000, seed=7)
                return round(float(d_arr.mean() / (e_plain[hold_m].mean() + 1e-12)), 4), round(float(d_arr.mean() / se), 2)
            m_act, z_act = marg(e_act)
            m_plc, z_plc = marg(e_plc)
            action_read[f"L1_k{k}"] = {"action_margin": m_act, "z": z_act,
                                        "shuffled_action_margin": m_plc, "z_plc": z_plc}
            print(f"  action-cond L1 k={k}: margin {m_act:+.4f} (z {z_act:+.2f}) vs shuffled {m_plc:+.4f} (z {z_plc:+.2f})")

    EPISODIC.write_text("\n".join(json.dumps(x) for x in records), encoding="utf-8")

    l1_det = detour.get("L1_k1", {})
    n_break_hits = sum(1 for b in BREAKS
                       if l1_det.get(b, {}).get("z") is not None and l1_det[b]["z"] >= 2
                       and l1_det[b]["median"] > l1_det["calm_median"])
    detour_ok = n_break_hits >= 2
    act_inert = all(abs(v["action_margin"]) < 0.02 or v["action_margin"] <= v["shuffled_action_margin"] + 0.01
                    for v in action_read.values())
    verdict = (f"DETOUR {'PASS' if detour_ok else 'FAIL'} ({n_break_hits}/3 break windows elevated at z>=2 on L1) — "
               + ("the compare step detects schema violations; " if detour_ok else
                  "surprise does NOT reliably flag breaks — the enrichment trigger needs redesign; ")
               + f"ACTION CHANNEL {'INERT as pre-registered' if act_inert else 'NOT inert — investigate re-encoding'} "
               + "(historical a_t is derived info; footprint untestable without live size). "
               + f"Episodic store v0 written ({len(records)} records).")
    rep = {"preregistration": {"detour_pass": ">=2 of 3 breaks with median>calm and z>=2 on L1 k=1",
                                "action_expectation": "NULL (derived info; gain must match shuffled placebo)",
                                "breaks": BREAKS, "covid_caveat": "2020 lies in the DEV tuning span"},
           "detour": detour, "action_conditioning": action_read,
           "episodic_store": {"path": str(EPISODIC.relative_to(ROOT)), "n_records": len(records),
                               "schema": "date/level/k/energy/null_energy/surprise_ratio/action_mean_exposure"},
           "verdict": verdict, "runtime_s": int(time.time() - t0)}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
