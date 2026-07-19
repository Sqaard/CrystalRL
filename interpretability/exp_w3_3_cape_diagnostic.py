"""W3.3 diagnostic falsifier: Shiller CAPE vs the active DY-reversion US CMA anchor.

This is intentionally NOT a PIT-vintage-clean calibration experiment: Shiller's downloaded historical
spreadsheet is a latest-vintage series.  It is a conservative diagnostic of whether expanding-mean CAPE
reversion improves the same 5y backcast.  It cannot promote CALIBRATED_V1; it can only reject CAPE.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from interpretability.personal_invest_forecast import (  # noqa: E402
    BOOKS_W3, CAPE_CSV, CAPE_XLS, cma_central, load_cape, load_universe)

OUT = HERE / "exp_w3_3_cape_diagnostic_report.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def cape_variant(uni, book, origin):
    active = cma_central(uni, book, origin)
    dy_hist = uni["div_yield"][uni["div_yield"].index <= origin].dropna()
    dy = float(dy_hist.iloc[-1]); dy_mean = float(dy_hist.mean())
    dy_drag = ((dy / dy_mean) ** 0.1 - 1.0) if dy > 0 and dy_mean > 0 else 0.0
    cape_hist = uni["cape"][uni["cape"].index <= origin].dropna()
    if not len(cape_hist):
        raise ValueError(f"no lagged CAPE available at {origin}")
    cape_now = float(cape_hist.iloc[-1]); cape_mean = float(cape_hist.mean())
    cape_drag = ((cape_mean / cape_now) ** 0.1 - 1.0) if cape_now > 0 and cape_mean > 0 else 0.0
    # CAPE changes the US equity anchor only.  Bonds, TIPS and gold retain their
    # own CMA anchors; treating every non-cash sleeve as equity overstates the
    # CAPE drag for SPY6040.
    equity_weight = float(sum(book["weights"].get(key, 0.0) for key in ("SPY", "EFA")))
    return active + equity_weight * (cape_drag - dy_drag), cape_now, cape_mean, cape_drag


def main():
    uni = load_universe("US")
    uni["cape"] = load_cape()
    calibrated = json.loads((HERE / "exp_w3_calibration_report.json").read_text(encoding="utf-8"))
    source_rows = [r for r in calibrated["rows"] if r["universe"] == "US" and r["h"] == 5]
    books = {b["name"]: b for b in BOOKS_W3["US"]}
    rows = []
    for src in source_rows:
        book = books[src["book"]]; o = pd.Timestamp(src["origin"]); realized = float(src["realized"])
        dy = cma_central(uni, book, o)
        cape, cape_now, cape_mean, cape_drag = cape_variant(uni, book, o)
        rows.append({"book": book["name"], "origin": str(o.date()), "realized": realized,
                     "dy_central": dy, "cape_central": cape,
                     "dy_abs_error": float(src["cma_err"]), "cape_abs_error": abs(cape-realized),
                     "sample_abs_error": float(src["sm_err"]), "cape": cape_now,
                     "cape_expanding_mean": cape_mean, "cape_drag": cape_drag})
    mae = {"dy_reversion": float(np.mean([r["dy_abs_error"] for r in rows])),
           "cape_reversion": float(np.mean([r["cape_abs_error"] for r in rows])),
           "sample_mean": float(np.mean([r["sample_abs_error"] for r in rows]))}
    cape_pass = mae["cape_reversion"] <= mae["dy_reversion"] and mae["cape_reversion"] <= mae["sample_mean"]
    rep = {"schema_version": "w3.3-cape-diagnostic.v1", "experiment": "W3.3 CAPE anchor falsifier",
           "evidence_status": "LATEST_VINTAGE_DIAGNOSTIC_NOT_PIT_VINTAGE_CLEAN",
           "n_rows": len(rows), "horizon_years": 5, "mae": mae,
           "gate": {"CAPE_must_beat_DY_and_sample_mean": cape_pass},
           "source": {"cape_csv": str(CAPE_CSV.relative_to(ROOT)), "cape_csv_sha256": sha(CAPE_CSV),
                      "source_xls": str(CAPE_XLS.relative_to(ROOT)), "source_xls_sha256": sha(CAPE_XLS),
                      "availability_lag_months": 3,
                      "vintage_caveat": "latest downloaded history; no historical revision vintages"},
           "active_engine_after_test": "DY reversion retained; CAPE is never called by cma_central",
           "probability_status": "UNCALIBRATED_RESEARCH_ESTIMATES",
           "verdict": ("CAPE candidate survives diagnostic" if cape_pass else
                       "CAPE candidate FALSIFIED; retain DY reversion; no calibration-status change"),
           "rows": rows}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_rows", "mae", "gate", "evidence_status", "verdict")},
                     indent=2))


if __name__ == "__main__":
    main()
