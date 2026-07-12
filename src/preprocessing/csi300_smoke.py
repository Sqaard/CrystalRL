"""SMOKE: the PROJECT-FAITHFUL csi300 panel loads into the env end-to-end.

Proves that ``data/adapters/_csi300_processed/csi300_model_ready.csv`` (produced by
``csi300_pipeline.run_pipeline``) is consumable by the PPO weight-env, reusing the exact pattern
of the existing real-smoke (``data/adapters/qlib_real_smoke.py``):

  1. load the model_ready panel,
  2. register an all-"U" single-group sector map for the csi300 universe (the env validates every
     ticker against a sector map),
  3. load_weight_panel + make_env_from_config + STEP the env with an equal-weight action,
  4. report obs_dim/action_dim/step-rewards + an equal-weight buy-and-hold baseline.

Run with the Py-3.9 venv:
    C:/Users/ivanp/.qlib_venv39/Scripts/python.exe -m src.preprocessing.csi300_smoke
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PANEL = _REPO_ROOT / "data" / "adapters" / "_csi300_processed" / "csi300_model_ready.csv"
_SECTOR_MAP_NAME = "csi300_faithful_singlegroup"
_START = "2018-01-01"
_END = "2021-12-31"


def _equal_weight_baseline(returns_next: np.ndarray) -> dict:
    port_ret = returns_next.mean(axis=1)
    equity = np.cumprod(1.0 + port_ret)
    total_return = float(equity[-1] - 1.0)
    mean_d, std_d = float(port_ret.mean()), float(port_ret.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * mean_d / std_d) if std_d > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    return {
        "n_days": int(len(port_ret)),
        "total_return": total_return,
        "annualized_sharpe": sharpe,
        "max_drawdown": float((equity / peak - 1.0).min()),
    }


def run() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("PROJECT-FAITHFUL csi300 PANEL SMOKE")
    print("=" * 78)
    if not _PANEL.exists():
        print(f"BLOCKED: panel not found at {_PANEL}. Run csi300_pipeline first.")
        return 2

    import pandas as pd

    from src.data.dow30_sectors import register_single_group_map
    from src.ppo.stage0_1_weight_env import make_env_from_config
    from src.ppo.weight_panel import load_weight_panel

    tickers = sorted(pd.read_csv(_PANEL, usecols=["tic"])["tic"].unique().tolist())
    register_single_group_map(_SECTOR_MAP_NAME, tickers, group="U")
    print(f"  registered sector map '{_SECTOR_MAP_NAME}' (all-'U') for {len(tickers)} tickers")

    panel = load_weight_panel(_PANEL, start=_START, end=_END)
    print(
        f"  load_weight_panel OK: dates={len(panel.dates)} tickers={len(panel.tickers)} "
        f"features={len(panel.feature_columns)} returns_next={panel.returns_next.shape}"
    )

    config = {"environment": {}, "universe": {"sector_map": _SECTOR_MAP_NAME}}
    variant = {"action_transform": "direct_weights", "controller": {"type": "P", "kp": 1.0}, "root_split": {}}
    env = make_env_from_config(panel, config, variant)
    obs, info = env.reset(seed=0)
    obs_dim, action_dim = int(obs.shape[0]), int(env.action_space.shape[0])
    print(f"  make_env_from_config OK: obs_dim={obs_dim} action_dim={action_dim}")

    ew = np.full(env.action_space.shape, 1.0 / action_dim, dtype=np.float32)
    rewards, last_pv = [], None
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(ew)
        rewards.append(float(reward))
        last_pv = info.get("portfolio_value")
        if terminated:
            break
    print(f"  STEPPED {len(rewards)} steps: rewards={[round(r,4) for r in rewards]} last_pv={last_pv}")

    base = _equal_weight_baseline(panel.returns_next)
    print("-" * 78)
    print(f"  EQUAL-WEIGHT BUY-AND-HOLD: days={base['n_days']} "
          f"total_return={base['total_return']:+.4%} sharpe={base['annualized_sharpe']:+.3f} "
          f"max_dd={base['max_drawdown']:+.4%}")

    summary = {
        "status": "PASS",
        "panel": {"n_dates": len(panel.dates), "n_tickers": len(panel.tickers),
                  "n_features": len(panel.feature_columns)},
        "env": {"obs_dim": obs_dim, "action_dim": action_dim,
                "equal_weight_step_rewards": rewards},
        "equal_weight_baseline": base,
        "feature_columns": list(panel.feature_columns),
    }
    (_PANEL.parent / "csi300_smoke_result.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {_PANEL.parent / 'csi300_smoke_result.json'}")
    print("=" * 78)
    print("STATUS: PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
