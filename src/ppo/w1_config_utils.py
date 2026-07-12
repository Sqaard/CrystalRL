"""Lightweight config helpers for W1 PM/Trader runners.

The generic `stage0_1_train` module imports the full SB3/weight-env stack at
module import time.  W1 runners only need variant inheritance and the prepared
feature CSV path, so these helpers keep Huawei packages small and avoid
unrelated transitive imports.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_variant_inheritance(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {variant["name"]: variant for variant in variants}
    resolved: dict[str, dict[str, Any]] = {}
    resolving: set[str] = set()

    def resolve_one(name: str) -> dict[str, Any]:
        if name in resolved:
            return copy.deepcopy(resolved[name])
        if name in resolving:
            raise ValueError(f"Cyclic variant inheritance detected at {name}")
        if name not in by_name:
            raise ValueError(f"Unknown base variant in inheritance: {name}")
        resolving.add(name)
        variant = by_name[name]
        parent_name = variant.get("inherits")
        if parent_name:
            parent = resolve_one(str(parent_name))
            merged = deep_merge_dicts(parent, variant)
        else:
            merged = copy.deepcopy(variant)
        merged.pop("inherits", None)
        resolved[name] = merged
        resolving.remove(name)
        return copy.deepcopy(merged)

    return [resolve_one(variant["name"]) for variant in variants]


def load_folds_from_config(
    config: dict[str, Any],
    names: list[str] | None,
    trading_dates: Any = None,
) -> pd.DataFrame:
    folds = pd.read_csv(resolve(config["walk_forward"]["folds_csv"]))
    if names:
        folds = folds[folds["fold"].astype(str).isin(names)].copy()
    if folds.empty:
        raise ValueError(f"No folds selected: {names}")

    # firewall (audit H5): train->validation embargo. With <=252-day rolling features, the
    # first ~year of validation overlaps train windows and inflates selection metrics. Set
    # walk_forward.embargo_trading_days > 0 AND pass the panel's trading_dates to push
    # validation_start past the embargo; consumers then read `validation_start_embargoed`.
    # ADDITIVE: default embargo 0 -> validation_start_embargoed == validation_start (no change).
    embargo = int(config.get("walk_forward", {}).get("embargo_trading_days", 0) or 0)
    folds["embargo_trading_days"] = embargo
    folds["validation_start_embargoed"] = folds["validation_start"]
    if embargo > 0 and trading_dates is not None:
        try:
            from src.evaluation import firewall as fw
        except Exception:
            fw = None
        dates = sorted(pd.to_datetime(pd.Series(list(trading_dates))).dt.strftime("%Y-%m-%d").unique().tolist())
        if fw is None:
            dates = []
        starts = []
        for _, r in folds.iterrows():
            try:
                starts.append(fw.embargoed_validation_start(dates, str(r["train_end_inclusive"]), embargo))
            except Exception:
                starts.append(str(r["validation_start"]))
        folds["validation_start_embargoed"] = starts
    return folds


def feature_csv_for_fold(
    config: dict[str, Any],
    variant: dict[str, Any],
    fold: pd.Series,
    out_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data"]
    normalization_cfg = config.get("normalization", {})
    if not normalization_cfg.get("enabled", False):
        return {
            "model_ready_csv": resolve(data_cfg["model_ready_csv"]),
            "transform_stats_csv": resolve(data_cfg["transform_stats_csv"])
            if data_cfg.get("transform_stats_csv")
            else None,
            "diagnostics_csv": None,
            "manifest_json": None,
            "status": "disabled",
        }

    raw_csv = data_cfg.get("raw_features_csv")
    if not raw_csv:
        raise ValueError("normalization.enabled=true requires data.raw_features_csv.")

    # Lazy import keeps W1 packages free from the full Stage0 PPO stack when
    # normalization is disabled, which is the default W1/Huawei case.
    from src.data.stage0_1_normalization import prepare_fold_scaled_features

    feature_subset_name = variant.get("feature_subset")
    feature_subset = None
    scaler_out_dir = out_root / "feature_scalers"
    if feature_subset_name:
        subsets = config.get("feature_subsets", {})
        if feature_subset_name not in subsets:
            raise ValueError(
                f"Variant {variant['name']} requests unknown feature_subset={feature_subset_name}. "
                f"Available: {sorted(subsets)}"
            )
        feature_subset = list(subsets[feature_subset_name])
        scaler_out_dir = scaler_out_dir / str(feature_subset_name)

    return prepare_fold_scaled_features(
        raw_csv=resolve(raw_csv),
        out_dir=scaler_out_dir,
        fold_id=str(fold["fold"]),
        train_start=str(fold["train_start"]),
        train_end=str(fold["train_end_inclusive"]),
        validation_end=str(fold["validation_end_inclusive"]),
        feature_subset=feature_subset,
        feature_subset_name=str(feature_subset_name) if feature_subset_name else None,
        lower_quantile=float(normalization_cfg.get("lower_quantile", 0.01)),
        upper_quantile=float(normalization_cfg.get("upper_quantile", 0.99)),
        force=force,
    )
