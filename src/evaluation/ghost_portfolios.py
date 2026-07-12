"""N5 — Ghost portfolios (shadow P&L) for repairs/interventions.

For every repair decision, keep the P&L of FOUR shadow worlds aligned by date:

  original   the un-edited policy (baseline arm)
  repaired   the real targeted repair (promotion arm)
  opposite   a wrong-direction / matched-random control arm
  no_trade   hold cash / don't act (synthesized if not logged)

This turns "the repair improved the average" into "WHEN exactly was the repair right
vs wrong", which is what S3 (weight-vs-conviction sells) and I1 (LLM-authored hypotheses)
need to read. It is the control-adjusted, finance-safe substrate the courtroom (N6) judges:
every claim is reported as repaired-minus-control with a block-bootstrap CI, and split by
regime so a single-regime win is exposed.

Consumes the `counterfactual_variant` daily schema the rollouts already emit
(run_action_stage7..., run_pretrain_branch_hcs_cycle, frozen_test_w2_latent_rollout): a long
CSV with `date`, `counterfactual_variant`, and a return column. So wiring is non-intrusive —
run a rollout, then point this at its daily CSV. (Per-name shadow P&L for S3 plugs into the
same ledger once the rollout logs per-name weights; `per_name_frame` defines that schema.)

CLI:  python -m src.evaluation.ghost_portfolios --daily path/to/a7_contextual_action_fullenv_daily.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.evaluation import firewall as fw
except Exception:  # allow running as a loose script
    import firewall as fw  # type: ignore

ANN = math.sqrt(252.0)

# world -> list of counterfactual_variant prefixes that map to it (first match wins per date)
DEFAULT_WORLDS = {
    "original": ["original_ppo"],
    "repaired": ["a7b_promote_context_best_broad", "s55_promote_context_best", "promote_context_best"],
    "opposite": ["a7b_random_control_context_schedule", "s6_random_promote", "random_control"],
}
RETURN_CANDIDATES = ["net_return", "portfolio_return_1d", "portfolio_return", "daily_return", "return_1d", "reward"]


def _detect_return_col(cols, override=None):
    if override:
        return override
    for c in RETURN_CANDIDATES:
        if c in cols:
            return c
    rx = [c for c in cols if "return" in str(c).lower()]
    if rx:
        return rx[0]
    raise SystemExit(f"no return column; columns: {list(cols)[:10]}")


def _finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _ann_sharpe(x):
    x = _finite(x)
    if x.size < 2:
        return float("nan")
    s = x.std(ddof=1)
    return float(x.mean() / s * ANN) if s > 0 else float("nan")


def _cumret(x):
    x = _finite(x)
    return float(np.prod(1.0 + x) - 1.0) if x.size else float("nan")


def _maxdd(x):
    x = _finite(x)
    if x.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + x)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


class GhostLedger:
    """Aligns the shadow worlds and computes when-right/when-wrong with control-adjusted CIs."""

    def __init__(self, daily: pd.DataFrame, *, return_col=None, worlds=None,
                 no_trade_return=0.0, regime_col=None):
        worlds = worlds or DEFAULT_WORLDS
        rc = _detect_return_col(daily.columns, return_col)
        d = daily.copy()
        d["_w"] = None
        v = d["counterfactual_variant"].astype(str)
        for world, prefixes in worlds.items():
            for p in prefixes:
                d.loc[d["_w"].isna() & v.str.startswith(p), "_w"] = world
        d = d.dropna(subset=["_w"]).copy()
        # one variant per world: if several variant strings map to the same world, keep the
        # most-frequent and warn, rather than silently mean-blending different arms (e.g. d10 vs d20).
        for world in list(d["_w"].unique()):
            vc = d.loc[d["_w"] == world, "counterfactual_variant"].astype(str).value_counts()
            if len(vc) > 1:
                keep = vc.index[0]
                print(f"[N5][warn] world '{world}' maps {len(vc)} variants {list(vc.index)}; keeping '{keep}'")
                d = d[~((d["_w"] == world) & (d["counterfactual_variant"].astype(str) != keep))]
        wide = (d.pivot_table(index="date", columns="_w", values=rc, aggfunc="mean")
                .sort_index())
        if "original" not in wide.columns or "repaired" not in wide.columns:
            raise SystemExit(f"need at least 'original' and 'repaired' worlds; got {list(wide.columns)}")
        if "no_trade" not in wide.columns:
            wide["no_trade"] = float(no_trade_return)   # synthesized: 'did nothing' = flat cash
        self.return_col = rc
        self.wide = wide
        self.regime_col = regime_col
        if regime_col and regime_col in d.columns:
            rmap = d.groupby("date")[regime_col].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            self.regime = wide.index.to_series().map(rmap).fillna("unknown")
        else:
            def _reg(s):
                s = str(s)[:10]
                if "2022-01-01" <= s < "2023-01-01":
                    return "2022_bear"
                if "2023-01-01" <= s < "2023-07-01":
                    return "2023_recovery"
                return s[:4]  # honest: label by year outside the known frozen window
            self.regime = wide.index.to_series().map(_reg)

    def shadow_frame(self) -> pd.DataFrame:
        w = self.wide.copy()
        w["repair_minus_original"] = w["repaired"] - w["original"]
        w["repair_minus_opposite"] = w["repaired"] - w.get("opposite", np.nan)
        w["repair_minus_no_trade"] = w["repaired"] - w["no_trade"]
        w["repair_helped"] = (w["repair_minus_original"] > 0).astype(int)
        w["regime"] = self.regime.values
        return w

    def _adj_ci(self, diff: np.ndarray, n_boot=10000, block=5):
        diff = np.asarray(diff, float)
        diff = diff[np.isfinite(diff)]
        if diff.size < 10:
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                    "sig": False, "pos_rate": float("nan"), "n": int(diff.size)}
        m, lo, hi = fw.block_bootstrap_ci(diff, n_boot=n_boot, block=block, seed=0)
        return {"mean": m, "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0),
                "pos_rate": float((diff > 0).mean()), "n": int(diff.size)}

    def summary(self) -> dict:
        w = self.shadow_frame()
        worlds = [c for c in ["original", "repaired", "opposite", "no_trade"] if c in w.columns]
        per_world = {c: {"ann_sharpe": round(_ann_sharpe(w[c]), 4),
                         "cum_return_pct": round(_cumret(w[c]) * 100, 3),
                         "max_dd_pct": round(_maxdd(w[c]) * 100, 3)} for c in worlds}
        contrasts = {
            "repaired_vs_original": self._adj_ci(w["repair_minus_original"].to_numpy()),
            "repaired_vs_opposite": self._adj_ci(w["repair_minus_opposite"].to_numpy()),
            "repaired_vs_no_trade": self._adj_ci(w["repair_minus_no_trade"].to_numpy()),
        }
        by_regime = {}
        for reg, g in w.groupby("regime"):
            by_regime[reg] = {
                "n": int(len(g)),
                "repair_helped_rate": round(float(g["repair_helped"].mean()), 3),
                "repaired_vs_opposite_mean": round(float(g["repair_minus_opposite"].mean()), 6),
            }
        return {"return_col": self.return_col, "n_days": int(len(w)),
                "per_world": per_world, "control_adjusted": contrasts, "by_regime": by_regime}


PER_NAME_COLUMNS = ["date", "tic", "world", "pre_weight", "desired_weight", "target_weight", "wt_conviction_gap"]


def build_per_name_frame(arm_csv_by_world: dict) -> pd.DataFrame:
    """Populate the S3 per-name shadow ledger from `evaluate`'s `<split>_daily.csv` arm files
    (which already log pre_weight_<tic> / desired_weight_<tic> / target_weight_<tic>). One row per
    (date, tic, world) with pre_weight (held/drifted), desired_weight (fresh conviction),
    target_weight, and wt_conviction_gap = pre_weight - desired_weight (S3: >0 = held overweight
    relative to current conviction = a trim candidate). No rollout instrumentation needed."""
    frames = []
    for world, path in arm_csv_by_world.items():
        d = pd.read_csv(path)
        if "date" not in d.columns:
            continue
        names = [c[len("pre_weight_"):] for c in d.columns
                 if c.startswith("pre_weight_") and not c.endswith("_CASH")]
        for nm in names:
            frames.append(pd.DataFrame({
                "date": d["date"].astype(str), "tic": nm, "world": world,
                "pre_weight": pd.to_numeric(d.get(f"pre_weight_{nm}"), errors="coerce"),
                "desired_weight": pd.to_numeric(d.get(f"desired_weight_{nm}"), errors="coerce"),
                "target_weight": pd.to_numeric(d.get(f"target_weight_{nm}"), errors="coerce"),
            }))
    if not frames:
        return pd.DataFrame(columns=PER_NAME_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["wt_conviction_gap"] = out["pre_weight"] - out["desired_weight"]
    return out[PER_NAME_COLUMNS]


def write_report(ledger: GhostLedger, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = ledger.summary()
    sf = ledger.shadow_frame()
    sf.to_csv(out_dir / f"ghost_{name}_daily.csv")
    L = [f"# N5 — Ghost portfolios (shadow P&L): {name}\n",
         f"Return col `{s['return_col']}`, {s['n_days']} days. Four shadow worlds aligned by date; "
         f"`opposite` = matched-random/wrong-direction control; `no_trade` = flat cash (synthesized). "
         f"Every contrast is control-adjusted with a block-bootstrap 95% CI.\n",
         "## Per-world", "| world | ann Sharpe | cum return % | max DD % |", "|---|---:|---:|---:|"]
    for wname, m in s["per_world"].items():
        L.append(f"| {wname} | {m['ann_sharpe']:+.3f} | {m['cum_return_pct']:+.2f} | {m['max_dd_pct']:+.2f} |")
    L += ["", "## Control-adjusted contrasts (repaired minus ...)",
          "| contrast | mean/day | 95% CI | positive-day rate | sig |", "|---|---:|---|---:|:--:|"]
    for k, c in s["control_adjusted"].items():
        pr = round(c['pos_rate'], 3) if c['pos_rate'] == c['pos_rate'] else float('nan')
        L.append(f"| {k} | {c['mean']:+.6f} | [{c['lo']:+.6f}, {c['hi']:+.6f}] | {pr} | {'YES' if c['sig'] else '·'} |")
    L += ["", "## When was the repair right? (by regime)",
          "| regime | days | repair-helped rate | repaired−opposite mean |", "|---|---:|---:|---:|"]
    for reg, m in s["by_regime"].items():
        L.append(f"| {reg} | {m['n']} | {m['repair_helped_rate']} | {m['repaired_vs_opposite_mean']:+.6f} |")
    co = s["control_adjusted"]["repaired_vs_opposite"]
    L += ["", "## Verdict",
          (f"Repair beats its control with a CI excluding 0 (mean {co['mean']:+.6f}/day) — a real, "
           "control-adjusted edge; check the by-regime table it isn't single-regime."
           if co["sig"] else
           f"Repair does NOT beat its matched control (mean {co['mean']:+.6f}/day, CI includes 0). "
           "Per N5 the 'improvement' is not distinguishable from a wrong-direction edit — do not promote.")]
    (out_dir / f"ghost_{name}_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True, help="long CSV with date, counterfactual_variant, return")
    ap.add_argument("--return-col", default=None)
    ap.add_argument("--no-trade-return", type=float, default=0.0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    daily = pd.read_csv(args.daily)
    ledger = GhostLedger(daily, return_col=args.return_col, no_trade_return=args.no_trade_return)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.daily).resolve().parent
    name = args.name or Path(args.daily).stem
    s = write_report(ledger, out_dir, name)
    import json
    print(json.dumps(s, indent=2))
    print(f"\n[N5] wrote ghost_{name}_report.md + ghost_{name}_daily.csv to {out_dir}")


if __name__ == "__main__":
    main()
