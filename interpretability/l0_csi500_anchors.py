"""L0 on the FOUR csi500 policies (R6c / P22 x deadline / PIT) -> the second frontier anchor.

The other agent retrained R6c + genuine P22 two-agent on the wide csi500 universe (deadline + proper-PIT).
Those policies sit at HIGH static book_dispersion (1.15-1.36) vs Dow-29's collapse band (<=0.17). The
sharp question the L0 bits/action ruler can now answer:

   Does the wide universe's high STATIC concentration (book_dispersion) correspond to high DYNAMIC
   complexity (h_mu / E / C_mu), or is it still a low-h_mu near-frozen program?

We re-rolled each saved checkpoint (no retrain) to dump per-day {cash, within-book weights} on the frozen
2022-2023 window, then run the SAME L0 estimator used on Dow-29 R6c (cash h_mu ~ 0.29-0.63) on two streams:
  - STANCE : cash fraction (continuous, binned).
  - BOOK-MODE : KMeans(6) of the within-book composition (cash removed, renormalized) -> mode sequence.

MIRROR-OF-THE-HALL: none of these beat equal-weight deflated (no edge) -- complexity here is of a null-edge
policy, not understanding. Reported next to every number.

Run: python interpretability/l0_csi500_anchors.py
Out: interpretability/l0_csi500_anchors_report.json + interpretability/L0_CSI500_ANCHORS.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STREAMS = HERE / "_streams"
sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

# (arch, panel, daily-dump, score-json) ; book_dispersion/cash/excess pulled from the score JSON
POLICIES = [
    ("R6c", "deadline", "r6c_deadline_daily.csv", "artifacts/stage0_1/csi500_r6c_v1/csi300/csi300_score.json"),
    ("R6c", "PIT", "r6c_pit_daily.csv", "artifacts/stage0_1/csi500_pit_r6c/csi300/csi300_score.json"),
    ("P22", "deadline", "p22_deadline_daily.csv", "artifacts/stage0_1/csi500_p22_v1/csi500/csi500_p22_score.json"),
    ("P22", "PIT", "p22_pit_daily.csv", "artifacts/stage0_1/csi500_pit_p22/csi500/csi500_p22_score.json"),
]
DOW29_R6C_CASH_HMU = [0.2882, 0.6276]  # measured Dow-29 R6c reference (L0_BITS_PER_ACTION.md)


def _score(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    j = json.loads(p.read_text(encoding="utf-8"))
    return {"book_dispersion": j.get("book_dispersion"), "mean_cash": j.get("mean_cash"),
            "excess_sharpe_vs_ew": j.get("excess_sharpe_vs_ew"), "beats_ew_deflated": j.get("beats_ew_deflated")}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rows = []
    for arch, timing, dump, score_json in POLICIES:
        f = STREAMS / dump
        if not f.exists():
            print(f"[L0-csi500] {arch}/{timing}: dump missing ({dump}), skipped")
            continue
        d = pd.read_csv(f)
        cash = d["cash"].to_numpy(float)
        wcols = [c for c in d.columns if c.startswith("w_")]
        W = np.clip(d[wcols].to_numpy(float), 0, None)
        rs = W.sum(axis=1, keepdims=True)
        within = np.divide(W, rs, out=np.full_like(W, 1.0 / W.shape[1]), where=rs > 1e-9)  # within-book, exposure-free
        modes = KMeans(n_clusters=6, random_state=0, n_init=10).fit_predict(within)

        cash_res = behavioral_complexity_dynamic(cash, kind="continuous", alphabets=(3, 5, 10), dts=(1, 2),
                                                 n_null=400, n_boot=400, seed=0)
        mode_res = behavioral_complexity_dynamic(modes, kind="discrete", dts=(1, 2), n_null=400, n_boot=400, seed=0)
        sc = _score(score_json)
        cash_var = float(np.std(cash))
        row = {
            "arch": arch, "timing": timing, "n_days": int(len(d)),
            "static_book_dispersion": sc.get("book_dispersion"), "mean_cash": sc.get("mean_cash"),
            "excess_vs_ew": sc.get("excess_sharpe_vs_ew"), "beats_ew_deflated": sc.get("beats_ew_deflated"),
            "cash_std": round(cash_var, 4),
            "cash_h_mu_range": cash_res.get("h_mu_range"), "cash_E_range": cash_res.get("E_range"),
            "cash_structure": cash_res.get("structure_present_configs"),
            "bookmode_h_mu_range": mode_res.get("h_mu_range"), "bookmode_E_range": mode_res.get("E_range"),
            "bookmode_structure": mode_res.get("structure_present_configs"), "n_book_modes": int(np.unique(modes).size),
            "_cash_full": cash_res, "_mode_full": mode_res,
        }
        rows.append(row)
        print(f"[L0-csi500] {arch}/{timing:8s}: static_disp={row['static_book_dispersion']} cash={row['mean_cash']} "
              f"| DYNAMIC cash h_mu={row['cash_h_mu_range']} ({row['cash_structure']}) "
              f"bookmode h_mu={row['bookmode_h_mu_range']} ({row['bookmode_structure']})")

    report = {
        "what": "L0 bits/action on the four csi500 policies vs Dow-29 R6c reference",
        "dow29_r6c_cash_h_mu_reference": DOW29_R6C_CASH_HMU,
        "mirror_of_the_hall": "none of these beat equal-weight deflated; complexity is of a null-edge policy.",
        "rows": rows,
    }
    (HERE / "l0_csi500_anchors_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- markdown: static vs dynamic, the central comparison ----
    lines = ["# L0 second anchor -- dynamic complexity of the csi500 policies", "",
             f"- Reference: **Dow-29 R6c** stance h_mu = {DOW29_R6C_CASH_HMU} bits/action (measured, "
             "L0_BITS_PER_ACTION.md). Dow-29 static book_dispersion <= 0.17 (collapse band).",
             f"- **Mirror-of-the-hall:** {report['mirror_of_the_hall']}", "",
             "| policy | STATIC book_disp | mean_cash | cash std | DYNAMIC stance h_mu | stance struct | book-mode h_mu | mode struct |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| **{r['arch']} {r['timing']}** | {r['static_book_dispersion']} | {r['mean_cash']} | "
                     f"{r['cash_std']} | {r['cash_h_mu_range']} | {r['cash_structure']} | "
                     f"{r['bookmode_h_mu_range']} | {r['bookmode_structure']} |")
    lines.append("")
    lines.append("**The static-vs-dynamic test.** If the wide universe induced genuine high DYNAMIC complexity, "
                 "stance/book-mode h_mu on csi500 would sit ABOVE the Dow-29 reference with structure present. If "
                 "instead csi500 policies park in cash (near-constant stance, tiny cash_std) and concentrate a frozen "
                 "sleeve, their static book_dispersion is high while their DYNAMIC bits/action is LOW -- i.e. static "
                 "concentration and dynamic complexity DISAGREE, and 'universe drives complexity' holds only on the "
                 "static axis. The table above resolves it per policy.")
    lines.append("")
    lines.append("**Auto-applies + frozen.** Streams re-rolled from the saved checkpoints (no retrain); same L0 "
                 "estimator + alphabets+dt as the Dow-29 R6c reading, so the anchors are directly comparable.")
    (HERE / "L0_CSI500_ANCHORS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[L0-csi500] wrote l0_csi500_anchors_report.json + L0_CSI500_ANCHORS.md ({len(rows)}/4 policies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
