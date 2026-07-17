"""BH2 — the LIBRARY continuous jump-model belief, through the SAME kill bar that killed KT4.

Context (honest lineage): E-24b/KT4 already tested OUR OWN Nystrup-style jump-penalized clustering
belief and KILLED it (flips −23.5% vs the −30% bar; hold legs slightly worse). BH2 is the
pre-registered RE-TEST with the published machinery instead of ours: Shu–Yu–Mulvey's `jumpmodels`
package, CONTINUOUS jump model (probabilistic state, mode loss), online inference
(`predict_proba_online` — no look-ahead). Everything else is IDENTICAL to KT4 by design, so the
delta is purely the algorithm:

  * same 4 standardized observables (train-frozen scaling), same TRAIN/DEV/HOLD windows;
  * same deployed step rule (the E-15 thresholds, H=10, 10bp costs) driven by the belief signal;
  * same lambda discipline: jump_penalty chosen on TRAIN ONLY = the first penalty on the grid whose
    train-window armed-decision flips fall to <= 0.7x the HMM's train flips (persistence by
    construction, no dev/hold touch);
  * SAME KILL BAR (pre-registered, unchanged from KT4): armed-decision flips on dev+hold must drop
    >= 30% vs the HMM AND the hold-window certified legs must not be worse
    (z_dsd >= z_hmm − 0.5 AND ni_z >= ni_hmm − 0.5). KILL otherwise.

Run: python interpretability/exp_bh2_jumpmodels.py        (~1-2 min, no network)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from interpretability.hl_v9_fresh_oos import load_extended, build_belief, TRAIN, DEV, HOLD  # noqa: E402
from interpretability.hl_v4_over_crystal1 import risk_boot_z  # noqa: E402
from src.hl.r6c_tension_adapter import block_z  # noqa: E402
from interpretability.exp_e24b_killtests2 import step_pnl, flips  # noqa: E402

OUT = HERE / "exp_bh2_jumpmodels_report.json"
PENALTY_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from jumpmodels.jump import JumpModel
    print("=== BH2 — library continuous jump-model belief through the KT4 kill bar ===")
    r, macro = load_extended()
    bel, _ = build_belief(r, macro)
    dates = r.index; ro_all = r.to_numpy(); b_hmm = bel.to_numpy()

    # same train-frozen standardization of the SAME observables as KT4
    m_tr = (macro.index >= pd.Timestamp(TRAIN[0])) & (macro.index <= pd.Timestamp(TRAIN[1]))
    X = macro.to_numpy(dtype=float)
    mu_s, sd_s = np.nanmean(X[np.asarray(m_tr)], 0), np.nanstd(X[np.asarray(m_tr)], 0) + 1e-9
    Z = np.nan_to_num((X - mu_s) / sd_s, nan=0.0)
    Z_tr = Z[np.asarray(m_tr)]

    win = (dates >= pd.Timestamp(DEV[0])) & (dates <= pd.Timestamp(HOLD[1]))
    hold_m = (dates >= pd.Timestamp(HOLD[0])) & (dates <= pd.Timestamp(HOLD[1]))

    # HMM reference (identical to KT4)
    ro_w = ro_all[np.asarray(win)][1:]; sig_hmm = b_hmm[np.asarray(win)][:-1]
    _, ex_hmm = step_pnl(ro_w, sig_hmm)
    ro_tr = ro_all[np.asarray(m_tr)][1:]; sig_tr_hmm = b_hmm[np.asarray(m_tr)][:-1]
    _, ex_tr_hmm = step_pnl(ro_tr, sig_tr_hmm)
    hmm_train_flips = flips(ex_tr_hmm)

    # TRAIN-only penalty selection: first penalty whose train flips <= 0.7 x HMM train flips
    chosen = None
    for pen in PENALTY_GRID:
        jm = JumpModel(n_components=2, jump_penalty=pen, cont=True, random_state=0)
        jm.fit(pd.DataFrame(Z_tr, index=macro.index[np.asarray(m_tr)]), sort_by="cumret",
               ret_ser=pd.Series(ro_all[np.asarray(m_tr)], index=macro.index[np.asarray(m_tr)]))
        # bear = the LOW-cumret state; with sort_by='cumret' state 0 = high-return -> bear prob = proba[:,1]
        proba_tr = jm.predict_proba(pd.DataFrame(Z_tr, index=macro.index[np.asarray(m_tr)]))
        sig_tr = np.asarray(proba_tr)[:, 1][:-1]
        _, ex_tr_j = step_pnl(ro_tr, sig_tr)
        f = flips(ex_tr_j)
        print(f"  penalty {pen:>7}: train flips {f} (HMM {hmm_train_flips}, bar {0.7*hmm_train_flips:.0f})")
        if f <= 0.7 * max(hmm_train_flips, 1):
            chosen = (pen, jm); break
    if chosen is None:
        pen = PENALTY_GRID[-1]
        jm = JumpModel(n_components=2, jump_penalty=pen, cont=True, random_state=0)
        jm.fit(pd.DataFrame(Z_tr, index=macro.index[np.asarray(m_tr)]), sort_by="cumret",
               ret_ser=pd.Series(ro_all[np.asarray(m_tr)], index=macro.index[np.asarray(m_tr)]))
        chosen = (pen, jm)
    pen, jm = chosen

    # ONLINE inference over the full span (no look-ahead), then the SAME windows
    proba_all = jm.predict_proba_online(pd.DataFrame(Z, index=macro.index))
    b_jump = np.asarray(proba_all)[:, 1]
    sig_j = b_jump[np.asarray(win)][:-1]
    _, ex_j = step_pnl(ro_w, sig_j)
    f_hmm, f_j = flips(ex_hmm), flips(ex_j)
    flip_cut = 1 - f_j / max(f_hmm, 1)

    # hold-window certified legs, identical estimator settings to KT4
    ro_h = ro_all[np.asarray(hold_m)][1:]; bh_h = ro_h.copy()
    pnl_j_h, _ = step_pnl(ro_h, b_jump[np.asarray(hold_m)][:-1])
    pnl_hmm_h, _ = step_pnl(ro_h, b_hmm[np.asarray(hold_m)][:-1])
    zj, _ = risk_boot_z(pnl_j_h, bh_h, block=20, n_boot=1000, seed=4)
    zh, _ = risk_boot_z(pnl_hmm_h, bh_h, block=20, n_boot=1000, seed=4)
    dj = pnl_j_h - bh_h; _, sej = block_z(dj, block=5, n_boot=1000, seed=4); nij = (dj.mean() + 2e-4) / sej
    dh = pnl_hmm_h - bh_h; _, seh = block_z(dh, block=5, n_boot=1000, seed=4); nih = (dh.mean() + 2e-4) / seh

    survives = (flip_cut >= 0.30) and (zj >= zh - 0.5) and (float(nij) >= float(nih) - 0.5)
    verdict = ("SURVIVES the KT4 kill bar — the library continuous JM earns a shot at the frozen "
               "gate (next: the full certified battery)" if survives else
               "KILLED on the same bar as KT4 — the library continuous JM does not rescue the "
               "jump-belief hypothesis on this substrate (BH2 closed honestly)")
    rep = {"experiment": "BH2 — library continuous jump model (jumpmodels pkg) vs the frozen HMM belief",
           "lineage": "re-test of E-24b/KT4 (our Nystrup-style variant KILLED at flips −23.5%, bar −30%)",
           "preregistration": {"kill_bar": "flip_cut>=0.30 AND z_dsd>=hmm−0.5 AND ni_z>=hmm−0.5 (unchanged from KT4)",
                                "penalty_rule": "TRAIN-only: first grid penalty with train flips <= 0.7x HMM",
                                "inference": "predict_proba_online (no look-ahead)"},
           "jump_penalty_chosen": pen,
           "flips": {"hmm_devhold": f_hmm, "jump_devhold": f_j, "flip_cut": round(flip_cut, 3), "bar": 0.30},
           "hold_legs": {"jump": {"z_dsd": round(zj, 2), "ni_z": round(float(nij), 2)},
                          "hmm": {"z_dsd": round(zh, 2), "ni_z": round(float(nih), 2)}},
           "survives_kill_bar": bool(survives),
           "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"flips: HMM {f_hmm} -> JM {f_j} (cut {flip_cut:.1%}, bar 30%) | hold legs JM z {zj:+.2f}/ni {float(nij):+.2f} "
          f"vs HMM z {zh:+.2f}/ni {float(nih):+.2f}")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
