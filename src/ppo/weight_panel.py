"""Market panel loading utilities shared by Stage 0.1 environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ppo.execution.helpers import EPS


@dataclass(frozen=True)
class WeightPanel:
    dates: np.ndarray
    tickers: list[str]
    feature_columns: list[str]
    features: np.ndarray
    prices: np.ndarray
    returns_next: np.ndarray


def load_weight_panel(csv_path: str | Path, start: str, end: str) -> WeightPanel:
    """Load a complete date/ticker panel and compute next-day returns."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"date", "tic", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No rows in {path} for {start}..{end}")

    feature_columns = [c for c in df.columns if c not in {"date", "tic", "close"}]
    tickers = sorted(df["tic"].unique().tolist())
    dates = np.array(sorted(df["date"].unique()))

    expected_rows = len(dates) * len(tickers)
    if len(df) != expected_rows:
        counts = df.groupby("date")["tic"].nunique()
        bad_dates = counts[counts != len(tickers)].head(10).index.astype(str).tolist()
        raise ValueError(
            f"Panel is incomplete: rows={len(df)}, expected={expected_rows}, "
            f"bad_dates={bad_dates}"
        )

    price_panel = df.pivot(index="date", columns="tic", values="close").loc[dates, tickers]
    prices = price_panel.to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all():
        raise ValueError("Price panel contains NaN or inf values.")
    returns_next = prices[1:] / np.maximum(prices[:-1], EPS) - 1.0

    feature_arrays = []
    for col in feature_columns:
        arr = df.pivot(index="date", columns="tic", values=col).loc[dates, tickers].to_numpy(dtype=np.float64)
        feature_arrays.append(arr)
    features = np.stack(feature_arrays, axis=-1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature panel contains NaN or inf values.")

    return WeightPanel(
        dates=dates,
        tickers=tickers,
        feature_columns=feature_columns,
        features=features,
        prices=prices,
        returns_next=returns_next,
    )
