"""Vectorized environment construction helpers for Stage 0.1 PPO runs."""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.ppo.stage0_1_weight_env import make_env_from_config
from src.ppo.weight_panel import WeightPanel


def make_vec_env(panel: WeightPanel, config: dict[str, Any], variant: dict[str, Any]):
    """Build the SB3 VecEnv requested by config/variant."""
    vec_cfg = dict(config.get("fast_vectorized_train", {}))
    vec_cfg.update(variant.get("fast_vectorized_train", {}))
    num_envs = int(vec_cfg.get("num_envs", 1))
    backend = str(vec_cfg.get("backend", "dummy")).lower()

    def _factory():
        env = make_env_from_config(panel, config, variant)
        return Monitor(env)

    if num_envs <= 1:
        return DummyVecEnv([_factory])

    factories = [_factory for _ in range(num_envs)]
    if backend == "subproc":
        return SubprocVecEnv(factories, start_method=str(vec_cfg.get("start_method", "spawn")))
    if backend != "dummy":
        raise ValueError(f"Unsupported fast_vectorized_train backend: {backend}")
    return DummyVecEnv(factories)
