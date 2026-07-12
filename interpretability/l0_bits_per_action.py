"""L0 -- the bits/action complexity ruler for R6c's FROZEN 2022-2023 action log.

WHAT / WHY
----------
The cross-arch frontier x-axis (behavioral_complexity) is ORDER-INVARIANT -- it reads the per-day
marginal and is blind to the action SEQUENCE. Both finals (FINAL_A / FINAL_G) and FRONTIER_ARC name
the same top-open item: a single ACTION-ONLY bits/action ruler (h_mu entropy rate, E excess entropy,
C_mu statistical complexity) computed IDENTICALLY on R6c's existing log and any future policy log, so
two ordinal "corners" become two readings on ONE frontier. This is that ruler, run on R6c.

FROZEN-LEGALITY (leakage guard)
-------------------------------
The clean discrete `code_id` stream in artifacts/stage4/.../stage4_joined_daily.csv is DEV-WINDOW ONLY
(2010-01-14 .. 2021-12-17, ZERO rows >= 2022) -- using it would leak the score across the frozen
boundary. We therefore score ONLY the frozen 2022-2023 window, taken from
artifacts/action_vq/A67.../ja67_joint_controls_daily.csv filtered to counterfactual_variant=='original_ppo'
(289 trading days, split=='frozen_test'). That file carries native frozen-legal discrete streams
(hidden_code_id, action_token_code) AND the continuous stance cash_target.

DISCIPLINE (guardrail #4 + mirror-of-the-hall)
----------------------------------------------
  * Publish a RANGE: >=3 alphabets x >=2 temporal resolutions (dt), min-occupancy depth guard,
    block-bootstrap 95% CI, and the phase-shuffle NULL (a stream has readable temporal structure only
    if observed h_mu sits below the 5th percentile of the day-order-shuffled h_mu).
  * MIRROR-OF-THE-HALL: R6c's out-of-sample edge on the liquid Dow-29 universe is NULL (this is the
    interpretability pivot's central finding). So every bits/action number here describes the complexity
    of a policy that EXPLAINS A NULL OOS SIGNAL -- it is NOT evidence of a real edge. Stamp that next to
    any number; a low h_mu means "simple, predictable cash-timing", not "understands the market".

Run: python interpretability/l0_bits_per_action.py
Out: interpretability/l0_bits_per_action_report.json + interpretability/L0_BITS_PER_ACTION.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cross_policy_crystal import behavioral_complexity_dynamic  # noqa: E402

JA67 = ROOT / "artifacts/action_vq/A67_joint_hidden_action_controls_fullenv_from_R6c_v1/ja67_joint_controls_daily.csv"
OUT_JSON = HERE / "l0_bits_per_action_report.json"
OUT_MD = HERE / "L0_BITS_PER_ACTION.md"

# the streams we score on the frozen window: (column, kind, human label)
STREAMS = [
    ("cash_target", "continuous", "stance: invested-vs-cash fraction (continuous, binned)"),
    ("hidden_code_id", "discrete", "latent code: KMeans cluster of the 64-d final policy latent"),
    ("action_token_code", "discrete", "action token: discrete controller-action code (incl -1 = no token)"),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not JA67.exists():
        print(f"[L0] FATAL: frozen log not found: {JA67}")
        return 1
    df = pd.read_csv(JA67)
    if "counterfactual_variant" in df.columns:
        df = df[df["counterfactual_variant"] == "original_ppo"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    n_days = df["date"].nunique()
    win = (str(df["date"].min().date()), str(df["date"].max().date()))
    assert df["date"].min() >= pd.Timestamp("2022-01-01"), "frozen-legality violated: pre-2022 rows present"
    print(f"[L0] frozen original_ppo: {len(df)} rows, {n_days} unique days, window {win[0]}..{win[1]}")

    results = {}
    for col, kind, label in STREAMS:
        if col not in df.columns:
            results[col] = {"error": f"column {col} absent"}
            print(f"[L0] {col}: ABSENT, skipped")
            continue
        series = df[col].to_numpy()
        res = behavioral_complexity_dynamic(series, kind=kind, alphabets=(3, 5, 10), dts=(1, 2),
                                            max_L=6, min_mean_count=5.0, n_null=500, n_boot=500, seed=0)
        res["stream_label"] = label
        res["native_n_unique"] = int(pd.Series(series).nunique())
        results[col] = res
        if "error" in res:
            print(f"[L0] {col}: {res['error']}")
            continue
        print(f"[L0] {col:18s} h_mu={res['h_mu_range']}  E={res['E_range']}  "
              f"C_mu(Lstar)={res['C_mu_Lstar_range']}  structure={res['structure_present_configs']}")

    report = {
        "policy": "R6c (original_ppo)", "universe": "Dow-29 (liquid large-cap)",
        "frozen_window": win, "n_frozen_days": int(n_days), "source": str(JA67.relative_to(ROOT)),
        "units": "bits per action", "method": "entropy-rate / excess-entropy / finite-L causal-state (epsilon-machine) reconstruction",
        "mirror_of_the_hall": ("R6c's OOS edge on Dow-29 is NULL -- these bits/action describe the complexity of a policy "
                               "explaining a null OOS signal, NOT evidence of a real edge."),
        "frozen_legality": ("scored ONLY on the 289-day frozen 2022-2023 original_ppo log; the clean stage4 code_id stream "
                            "is dev-window-only (<=2021-12-17) and was deliberately NOT used."),
        "streams": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- human-readable summary ----
    lines = ["# L0 -- bits/action complexity ruler (R6c, frozen 2022-2023)", ""]
    lines.append(f"- Policy: **R6c (original_ppo)**, universe Dow-29, frozen window {win[0]}..{win[1]} ({n_days} days).")
    lines.append(f"- Units: **bits per action**. Method: h_mu (entropy rate) / E (excess entropy) / C_mu (finite-L causal-state).")
    lines.append(f"- **Mirror-of-the-hall:** {report['mirror_of_the_hall']}")
    lines.append(f"- **Frozen-legality:** {report['frozen_legality']}")
    lines.append("")
    lines.append("| stream | h_mu (bits/action) | i.i.d. bound H1 | E (bits) | C_mu(Lstar) | structure vs phase-shuffle null |")
    lines.append("|---|---|---|---|---|---|")
    for col, _, label in STREAMS:
        r = results.get(col, {})
        if "error" in r or not r.get("configs"):
            lines.append(f"| `{col}` | n/a ({r.get('error','no config')}) | | | | |")
            continue
        h1 = [c["H1_iid_bound"] for c in r["configs"]]
        lines.append(f"| `{col}`<br><sub>{label}</sub> | {r['h_mu_range']} | [{min(h1)}, {max(h1)}] | "
                     f"{r['E_range']} | {r['C_mu_Lstar_range']} | {r['structure_present_configs']} configs "
                     f"({'STRUCTURE' if r['structure_present'] else 'no structure'}) |")
    lines.append("")
    lines.append("**Reading.** `h_mu` near its i.i.d. bound `H1` ⇒ each action is near-memoryless (predictable only "
                 "from its marginal, no temporal program); `h_mu` far below `H1` with `structure_present` ⇒ the stream "
                 "carries readable sequential structure (the order matters). `E`>0 and `C_mu`>0 quantify how much past "
                 "must be remembered. All values are RANGES across the alphabet/dt sweep (guardrail #4).")
    lines.append("")
    lines.append("**Auto-applies to the new policy.** Re-run this exact script pointed at the parallel agent's new "
                 "(PIT-retrained R6c / genuine P22) frozen log, on the SAME alphabets+dt, to drop the second anchor on "
                 "the one frontier with zero re-tooling.")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[L0] wrote {OUT_JSON.name} + {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
