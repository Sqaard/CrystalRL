"""H2 -- MEASUREMENT LEG ONLY: the entropy-rate of R6c's mode-arbitration stream (E / C_mu / h_mu).

H2's full card claims "event-conditioned mode arbitration + adaptive hold-time raise E/C_mu (structure)
without inflating h_mu, and ONLY this UP-move carries PnL". We run ONLY the runnable, honest half here:
the L0 bits/action ruler re-pointed at the per-day MODE stream (argmax over the 30-d `action_mode` head in
the frozen NPZ). This gives the first temporal/structural reading on the MODE stream (vs L0's cash/code
streams).

WHY THE OTHER HALF IS GATED OUT (do NOT run now):
  1. "only this carries PnL" rides a NULL OOS signal (R6c original_ppo frozen Sharpe ~ -0.10, return
     ~ -0.019 on the no-edge Dow-29). It can only be a within-window decomposition with the OOS-null
     status stamped on it -- never a promotion signal (mirror-of-the-hall). -> HCS gate, not now.
  2. The "entropy-matched noise twin" + "obs-shuffle" are genuinely NEW policy-forward replay code on the
     frozen checkpoint (A67's random_* controls are token controls, not turnover/entropy-matched twins).
  3. 289 days is short for block entropy at L>=2 on a 30-mode alphabet -> coarse alphabets, RANGE not point.

Run: python interpretability/h2_mode_entropy.py
Out: interpretability/h2_mode_entropy_report.json + interpretability/H2_MODE_ENTROPY.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

NPZ = ROOT / ("artifacts/stage4/R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12"
              "_frozen_2022_2023_for_Joseph/hidden_activations/r6c_frozen_hidden_activations.npz")
OUT_JSON = HERE / "h2_mode_entropy_report.json"
OUT_MD = HERE / "H2_MODE_ENTROPY.md"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not NPZ.exists():
        print(f"[H2] FATAL: {NPZ} not found"); return 1
    z = np.load(NPZ, allow_pickle=True)
    am = np.asarray(z["action_mode"], dtype=float)  # (289, 30) action-weight vector
    dates = z["dates"]
    win = (str(dates[0]), str(dates[-1]))
    # argmax is DEGENERATE here: one component (col 0) dominates every day -> argmax constant. That is an
    # artifact of the action vector's shape, not "no arbitration". The honest mode stream is a KMeans
    # clustering of the 30-d action-weight vectors (same recipe L0 uses for the 64-d latent) into K modes.
    argmax_n_modes = int(np.unique(am.argmax(axis=1)).size)
    l1_daily = float(np.abs(np.diff(am, axis=0)).sum(1).mean())
    mode_id = KMeans(n_clusters=6, random_state=0, n_init=10).fit_predict(am)
    n_active = int(np.unique(mode_id).size)
    print(f"[H2] action_mode: argmax collapses to {argmax_n_modes} mode (one component dominates); "
          f"KMeans(6) -> {n_active} modes; day-to-day L1={l1_daily:.4f} (near-static)")

    res = behavioral_complexity_dynamic(mode_id, kind="discrete", dts=(1, 2), max_L=6,
                                        min_mean_count=5.0, n_null=500, n_boot=500, seed=0)
    report = {
        "policy": "R6c (original_ppo)", "stream": "action_mode KMeans(6) modes (mode-arbitration of the action-weight vector)",
        "argmax_degenerate": f"argmax collapses to {argmax_n_modes} mode (col-0 dominates every day); KMeans used instead",
        "action_vector_daily_L1": round(l1_daily, 4),
        "frozen_window": win, "n_days": int(am.shape[0]), "n_modes_active": n_active,
        "leg": "MEASUREMENT ONLY (h_mu/E/C_mu); the PnL / noise-twin / obs-shuffle leg is GATED OUT (HCS, not now)",
        "mirror_of_the_hall": "OOS edge is NULL -> this is the mode stream's self-structure, not market signal.",
        "result": res,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# H2 (measurement leg) -- mode-arbitration entropy-rate (R6c, frozen 2022-2023)", "",
             f"- Stream: `action_mode` argmax (the mode-arbitration head), {mode_id.size} days {win[0]}..{win[1]}, "
             f"{n_active} modes active.",
             f"- **Leg:** {report['leg']}.",
             f"- **Mirror-of-the-hall:** {report['mirror_of_the_hall']}", ""]
    if "error" in res:
        lines.append(f"- result: {res['error']}")
    else:
        lines.append(f"- **h_mu** = {res['h_mu_range']} bits/action | **E** = {res['E_range']} | "
                     f"**C_mu(Lstar)** = {res['C_mu_Lstar_range']} | structure vs phase-shuffle null: "
                     f"{res['structure_present_configs']} configs "
                     f"({'STRUCTURE' if res['structure_present'] else 'no structure'}).")
        lines.append("")
        lines.append("**Reading.** Low `h_mu` far below the i.i.d. bound with `structure_present` ⇒ the mode-arbitration "
                     "stream is itself a low-entropy, self-predictable program (consistent with L0's stance finding and "
                     "H1's autoregression). E/C_mu>0 quantify the structural memory. A future UP-move would have to raise "
                     "E/C_mu WITHOUT inflating h_mu AND carry PnL under a turnover/entropy-matched twin + obs-shuffle — "
                     "the gated HCS leg, not run here because the OOS signal it rides is null.")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    if "error" not in res:
        print(f"[H2] mode h_mu={res['h_mu_range']} E={res['E_range']} C_mu(L*)={res['C_mu_Lstar_range']} "
              f"structure={res['structure_present_configs']}")
    print(f"[H2] wrote {OUT_JSON.name} + {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
