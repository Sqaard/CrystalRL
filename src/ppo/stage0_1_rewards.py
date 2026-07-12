"""Reward helpers for Stage 0.1 weight-based environments."""

from __future__ import annotations

from typing import Any


def compute_stage0_reward(
    reward_config: dict[str, Any],
    *,
    net_return: float,
    turnover: float,
    drawdown_increment: float,
    concentration: float,
    action_change: float,
    extra_penalty: float = 0.0,
    reward_scale: float = 100.0,
) -> float:
    """Classic Stage 0.1 scalar reward used by E/R candidates."""
    unscaled = (
        float(reward_config.get("return_weight", 1.0)) * float(net_return)
        - float(reward_config.get("turnover_penalty", 0.0)) * float(turnover)
        - float(reward_config.get("drawdown_penalty", 0.0)) * float(drawdown_increment)
        - float(reward_config.get("concentration_penalty", 0.0)) * float(concentration)
        - float(reward_config.get("action_change_penalty", 0.0)) * float(action_change)
        - float(extra_penalty)
    )
    return float(reward_scale) * unscaled
