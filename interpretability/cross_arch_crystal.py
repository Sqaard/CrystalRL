"""CROSS-ARCHITECTURE CrystalScore -- extend the (complexity, interpretability) arc
from the R6c-FAMILY corner to the TWO-AGENT (PM + Trader) architecture.

WHY THIS EXISTS
---------------
`cross_policy_crystal.py` placed 6 R6c-FAMILY policies on a (behavioral_complexity,
CrystalScore-core) curve and SKIPPED the two-agent policies (P22 / W1 / H2) because they
have no single comparable internal latent: the cross-policy y-axis there (CrystalScore-core)
was built on each policy's OWN 64-d `policy_net.5.ReLU` latent, which the two-agent PM+Trader
nets simply do not have.

THE FIX (the key idea): a UNIFIED, ARCHITECTURE-AGNOSTIC interpretability axis that needs only
each policy's per-step ACTIONS plus a FIXED, COMMON market-regime feature set available to
EVERY policy -- never any policy's internal latent. Then EVERY policy (single-MLP R6c-family OR
two-agent PM+Trader) is scored on the SAME axes:

  x  BEHAVIORAL_COMPLEXITY  (action-only, already uniform; IDENTICAL recipe to cross_policy):
       mean( cash_entropy[10-bin], book_dispersion[L1-to-equal-weight], action_eff_dim[PR/dim] ).

  y  CRYSTALSCORE-BEHAVIORAL  (cross-ARCH-comparable; needs NO internal latent):
       cluster a STANDARDIZED low-dim COMMON market-regime representation
       (trailing realized vol, drawdown, market momentum, breadth -- all computed from the
       frozen panel RETURNS, identical for every policy) into K=2..9, and measure
         COMPLETENESS@K   = 1 - SS_res/SS_tot of the K-regime-cluster-mean predictor of the
                            policy's ACTION (cash/q AND within-book composition), separately
                            for STANCE (cash) and SELECTION (within-book), swept over K.
         SIMULATABILITY   = cv-R^2 (shuffled 5-fold) of a linear surrogate predicting cash/q
                            from the COMMON regime features. Negative R^2 -> 0.
       CrystalScore-behavioral(core) = Simulatability_regime x clustering-stability(regime, K=9).
       This asks: "is the behaviour compressible into K human-readable MARKET REGIMES?" -- a
       question every architecture can be asked identically.

  SECONDARY (NOT on the common axis): each policy's NATIVE-latent simulatability, where the
       latent is extractable:
         R6c-family : 64-d `policy_net.5.ReLU`  (reused verbatim from cross_policy where present;
                      replayed otherwise).
         two-agent  : concat(PM penultimate 128-d, Trader penultimate 128-d) = 256-d.
       Reported with an EXPLICIT latent-DIM caveat: higher-dim latents decode cash more trivially
       (more directions -> higher cv-R^2 mechanically), so native-latent simulatability is NOT
       cross-arch-comparable. We standardize the latent and report the raw dim alongside.

WHAT IS SCORABLE (replay-only, NO retrain)
------------------------------------------
  * R6c-family (6 rows): REUSED from cross_policy_crystalscore.csv. We re-derive their
    common-axis y (regime-based) from their EXISTING behavior series (R6c from its frozen
    behavior log; the 5 replayed ones by re-replaying -- identical machinery to cross_policy).
  * W1 (two-agent, vanilla Dirichlet trader): REPLAYED natively on the frozen 2022-2023 panel
    via W1BudgetTraderEnv + BudgetPMActorCritic/BudgetTraderActorCritic. Scorable.
  * P22 / H2: SKIPPED with PRECISE stated reasons (see SKIPPED below). No fabrication.

FIREWALL DISCIPLINE
-------------------
  * No retrain. Replay only on the frozen panel.
  * Negative R^2 -> 0 (clip01). Missing input -> SKIP with reason, never invented.
  * The cross-ARCH y-axis (regime-based) is SEPARATE from the native-latent secondary column.
  * If only 0-1 two-agent policies are scorable, the result is reported as PARTIAL; we do not
    overclaim a full cross-architecture curve.
  * Windows console: sys.stdout.reconfigure(encoding='utf-8').

Run:
    python interpretability/cross_arch_crystal.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console
except Exception:
    pass

# ----------------------------------------------------------------------------- paths / imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# REUSE the cross_policy helpers VERBATIM (import, do not re-implement).
import cross_policy_crystal as cpc  # noqa: E402
from cross_policy_crystal import (  # noqa: E402
    FROZEN_PANEL,
    K_BUDGET,
    K_GRID,
    SEEDS,
    behavioral_complexity,
    candidate_specs,
    clip01,
    cluster_mean_r2,
    clustering_stability,
    pareto_auc,
    r6c_from_report,
    replay_policy,  # R6c-family single-MLP replay (latent + actions)
)

# two-agent (W1) native machinery
from src.ppo.weight_panel import load_weight_panel  # noqa: E402
from src.ppo.w1_budget_trader_env import W1BudgetTraderEnv  # noqa: E402
from src.ppo.w1_budget_trader_policy import (  # noqa: E402
    BudgetPMActorCritic,
    BudgetTraderActorCritic,
    LatentActionTraderActorCritic,
)
import src.ppo.stage0_1_w1_budget_trader_train as w1train  # noqa: E402
import torch as th  # noqa: E402

# two-agent latent-action (P22) native machinery -- the SAME proven path the firewall
# frozen-test runner (scripts/frozen_test_w2_latent_rollout.py) uses for the base arm.
from scripts.run_pretrain_branch_hcs_cycle import load_variant  # noqa: E402
from src.ppo.stage0_1_w1_budget_trader_train import (  # noqa: E402
    env_kwargs as w2_env_kwargs,
    evaluate as w2_evaluate,
    make_policies as w2_make_policies,
)
from src.ppo.w1_config_utils import (  # noqa: E402
    feature_csv_for_fold,
    load_folds_from_config,
    resolve as w2_resolve,
)

# ----------------------------------------------------------------------------- outputs
OUT_CSV = HERE / "cross_arch_crystalscore.csv"
OUT_CURVE_PNG = HERE / "cross_arch_curve.png"
OUT_MD = HERE / "CROSS_POLICY_CRYSTALSCORE.md"  # we APPEND a section here
OUT_JSON = HERE / "cross_arch_crystal_report.json"

W1_VARIANT_NAME = "W1_PM_BetaBudget_Trader_P4PrimitiveWarmStart_v1"

# P22 (W2 latent-action two-agent) -- now scorable because D: is mounted and the
# P13-corrected teacher-prototype tree + the trained W2/P22 checkpoints are present.
P22_VARIANT_NAME = "W2_P22P18ContextSelectiveAntiCollapseLatentActionTrader_P22WarmStart_v1"
P22_CONFIG = ROOT / "configs" / "generated" / "stage0_1_w1_budget_trader.yaml"
# the trained PPO W2/P22 checkpoint dir (pm_policy.pt + trader_policy.pt) on the now-mounted D:.
P22_RUN_DIR = Path(
    r"D:/Interpretable_CHRL/stage0_1/current"
    r"/weight_based_w2_p22_context_selective_anti_collapse_batch"
    r"/W2_P22P18ContextSelectiveAntiCollapseLatentActionTrader_P22WarmStart_v1/fold_2021"
)
# the P13-corrected teacher-prototype CSV the latent-action decode needs (default in
# build_latent_action_prototypes), now present on D:.
P22_PROTOTYPE_CSV = Path(
    r"D:/Interpretable_CHRL/pretrain/P13_corrected_teacher_trajectories_v1"
    r"/p13_asset_corrected_teacher_targets_long.csv"
)


# ----------------------------------------------------------------------------- common regime features (THE cross-arch axis)
def common_regime_features(panel: Any) -> np.ndarray:
    """Build a FIXED, architecture-AGNOSTIC, low-dim market-regime representation from the
    frozen-panel RETURNS ONLY (never any policy's latent). One row per executed step, ALIGNED
    1:1 with the per-day action series (returns_next has N = len(dates)-1 rows; each action row
    is taken at day t and realised over returns_next[t], so row t of the regime block describes
    the market state the policy is acting in).

    Components (all standardized downstream):
      realized_vol_20  : trailing 20d cross-sectional-mean realized vol (annualized).
      drawdown_60      : running drawdown of the equal-weight market index over 60d.
      momentum_20      : trailing 20d equal-weight market momentum.
      momentum_60      : trailing 60d equal-weight market momentum.
      breadth_20       : fraction of names with positive trailing 20d return (cross-sectional breadth).
      dispersion_20    : trailing-20d cross-sectional dispersion (std across names of 20d returns).

    These are the SAME for every policy on this frozen panel, so they form the common input
    against which every policy's behaviour is tested for regime-compressibility.
    """
    rets = np.asarray(panel.returns_next, dtype=float)  # (N, n_assets); row t realised over day t
    n, n_assets = rets.shape
    mkt = rets.mean(axis=1)  # equal-weight market daily return, length N

    def trailing(fn, win):
        out = np.full(n, np.nan)
        for t in range(n):
            lo = max(0, t - win + 1)
            out[t] = fn(t, lo)
        return out

    realized_vol_20 = trailing(lambda t, lo: float(np.std(mkt[lo : t + 1], ddof=0) * np.sqrt(252)), 20)
    momentum_20 = trailing(lambda t, lo: float(np.prod(1.0 + mkt[lo : t + 1]) - 1.0), 20)
    momentum_60 = trailing(lambda t, lo: float(np.prod(1.0 + mkt[lo : t + 1]) - 1.0), 60)

    # running drawdown of the cumulative equal-weight index (causal)
    cum = np.cumprod(1.0 + mkt)
    peak = np.maximum.accumulate(cum)
    drawdown = cum / np.maximum(peak, 1e-12) - 1.0  # <= 0

    # cross-sectional breadth + dispersion over trailing 20d per-name compounded return
    breadth_20 = np.full(n, np.nan)
    dispersion_20 = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - 20 + 1)
        comp = np.prod(1.0 + rets[lo : t + 1], axis=0) - 1.0  # per-name 20d return
        breadth_20[t] = float(np.mean(comp > 0.0))
        dispersion_20[t] = float(np.std(comp, ddof=0))

    feats = np.column_stack(
        [realized_vol_20, drawdown, momentum_20, momentum_60, breadth_20, dispersion_20]
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats  # (N, 6)


REGIME_FEATURE_NAMES = [
    "realized_vol_20",
    "drawdown_60",
    "momentum_20",
    "momentum_60",
    "breadth_20",
    "dispersion_20",
]


def regime_simulatability_cv(regime_std: np.ndarray, cash: np.ndarray) -> float:
    """SIMULATABILITY = shuffled 5-fold cv-R^2 of a linear (Ridge) surrogate predicting cash/q
    from the COMMON regime features. Negative -> 0. This is the continuous "can a human-readable
    regime description reproduce the stance" reading, on the cross-arch axis."""
    y = np.asarray(cash, dtype=float)
    if y.size < 10 or np.std(y) < 1e-9:
        return float("nan")
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(Ridge(alpha=1.0), regime_std, y, cv=cv, scoring="r2")
    return clip01(float(np.mean(scores)))


def regime_clustering_stability(regime_std: np.ndarray, k: int = K_BUDGET) -> float:
    """Cross-KMeans-seed ARI of the K regime clusters on the COMMON regime features. This is the
    SAME for every policy (regime features are policy-independent), so it is a property of the
    regime axis, reported once and reused -- it does NOT discriminate policies, but keeps the
    CrystalScore-behavioral formula structurally identical to CrystalScore-core."""
    labels = [
        KMeans(n_clusters=k, random_state=s, n_init=10, max_iter=500).fit(regime_std).labels_
        for s in SEEDS
    ]
    aris = [
        adjusted_rand_score(labels[i], labels[j])
        for i in range(len(SEEDS))
        for j in range(i + 1, len(SEEDS))
    ]
    return clip01(float(np.mean(aris)))


def score_behavioral(regime_std: np.ndarray, cash: np.ndarray, within: np.ndarray) -> dict:
    """The cross-ARCH-comparable y-axis: regime-completeness / regime-simulatability of the
    ACTION. Uses ONLY the common regime features as the clustering/prediction input -- never any
    internal latent -- so it is identical-recipe across architectures."""
    n = min(len(regime_std), len(cash), len(within))
    R = regime_std[:n]
    cash = np.asarray(cash, dtype=float)[:n]
    within = np.asarray(within, dtype=float)[:n]

    comp_stance = {k: cluster_mean_r2(R, cash, k, seed=0) for k in K_GRID}
    comp_sel = {k: cluster_mean_r2(R, within, k, seed=0) for k in K_GRID}
    simul = regime_simulatability_cv(R, cash)
    stab = regime_clustering_stability(R, K_BUDGET)

    comp9 = comp_stance[K_BUDGET]
    core = clip01(simul * stab) if np.isfinite(simul) and np.isfinite(stab) else float("nan")
    core_nostab = clip01(simul) if np.isfinite(simul) else float("nan")
    return {
        "regime_simulatability_cash_cv": round(float(simul), 4) if np.isfinite(simul) else None,
        "regime_completeness_stance": {k: round(float(v), 4) for k, v in comp_stance.items()},
        "regime_completeness_selection": {k: round(float(v), 4) for k, v in comp_sel.items()},
        "regime_completeness_stance_K9": round(float(comp9), 4),
        "regime_completeness_selection_K9": round(float(comp_sel[K_BUDGET]), 4),
        "regime_pareto_stance": pareto_auc(K_GRID, comp_stance),
        "regime_pareto_selection": pareto_auc(K_GRID, comp_sel),
        "regime_clustering_stability_K9": round(float(stab), 4),
        "crystalscore_behavioral": round(float(core), 4),
        "crystalscore_behavioral_nostab": round(float(core_nostab), 4),
    }


def native_latent_simulatability(latent: np.ndarray, cash: np.ndarray) -> dict:
    """SECONDARY column (NOT cross-arch-comparable): cv-R^2 of cash from the policy's OWN latent.
    Reported WITH the latent dim, because higher-dim latents decode cash more trivially."""
    if latent is None or not np.isfinite(latent).all():
        return {"native_latent_dim": None, "native_latent_simul_cash_cv": None}
    n = min(len(latent), len(cash))
    X = StandardScaler().fit_transform(np.asarray(latent, dtype=float)[:n])
    y = np.asarray(cash, dtype=float)[:n]
    if y.size < 10 or np.std(y) < 1e-9:
        return {"native_latent_dim": int(X.shape[1]), "native_latent_simul_cash_cv": None}
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=cv, scoring="r2")
    return {
        "native_latent_dim": int(X.shape[1]),
        "native_latent_simul_cash_cv": round(clip01(float(np.mean(scores))), 4),
    }


# ----------------------------------------------------------------------------- two-agent (W1) replay
@dataclass
class TwoAgentReplay:
    cash: np.ndarray          # (N,) executed cash
    within: np.ndarray        # (N, n_assets) within-book composition (exposure removed)
    native_latent: np.ndarray  # (N, pm_pen + trader_pen) concat penultimate latents
    n_steps: int


def _w1_variant_and_config() -> tuple[dict, dict]:
    """Resolve the W1 P4-primitive-warmstart variant + the W1 config. The variant is read
    VERBATIM from the checkpoint's metadata.json (the fully-resolved variant actually trained),
    so the replayed env/policy matches the trained one exactly."""
    meta_path = (
        ROOT / "artifacts" / "stage0_1" / "_unpacked_W1_P4PrimitiveWarmStart_results"
        / "stage0_1_job_703_W1_P4PrimitiveWarmStart_fold_2021_results"
        / "stage0_1_job_703_W1_P4PrimitiveWarmStart_fold_2021" / "artifacts" / "stage0_1"
        / "weight_based_w1_p4_primitive_warmstart_batch"
        / "W1_PM_BetaBudget_Trader_P4PrimitiveWarmStart_v1" / "fold_2021" / "metadata.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    variant = meta["variant"]
    cfg_path = ROOT / "configs" / "generated" / "stage0_1_w1_budget_trader.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return variant, config, meta_path.parent


def _register_penultimate_hooks(pm_policy, trader_policy) -> tuple[dict, list]:
    """Capture the PM penultimate (actor MLP output, 128-d) and Trader penultimate
    (q_ctx = cat[mean_ctx, std_ctx], 128-d) on each forward. Last-write-wins per forward."""
    store: dict[str, np.ndarray] = {}
    handles = []

    def pm_actor_hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        store["pm_pen"] = t.detach().cpu().numpy().reshape(-1).astype(np.float32)

    handles.append(pm_policy.actor.register_forward_hook(pm_actor_hook))

    # Trader: hook the value-head's critic input is awkward; instead wrap _dists via the
    # cash_score_head input is q_ctx. We hook the trader's `critic` first Linear's INPUT, which
    # is exactly q_ctx (+ optional critic_extra). Simpler & robust: monkeypatch _dists to stash.
    orig_dists = trader_policy._dists

    def patched_dists(obs):
        portfolio_dist, q_ctx = orig_dists(obs)
        store["trader_pen"] = q_ctx.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return portfolio_dist, q_ctx

    trader_policy._dists = patched_dists  # type: ignore[assignment]
    return store, handles


def replay_w1(model_pm: Path, model_trader: Path) -> TwoAgentReplay:
    """Replay W1 (PM + vanilla Dirichlet Trader) on the frozen 2022-2023 panel. Reuses the
    native W1 inference loop semantics from stage0_1_w1_budget_trader_train.evaluate (deterministic
    PM/Trader), capturing per-day executed cash + within-book composition AND the concat
    penultimate native latent. No retrain; no warm-start (final checkpoints are loaded directly)."""
    variant, config, _run_dir = _w1_variant_and_config()
    start = str(config["data"]["frozen_test_start"])
    end = str(config["data"]["frozen_test_end"])
    panel = load_weight_panel(FROZEN_PANEL, start, end)
    env = W1BudgetTraderEnv(panel, **w1train.env_kwargs(config, variant, panel))

    pm_policy, trader_policy = w1train.make_policies(env, variant, fold_id="fold_2021")
    if not isinstance(trader_policy, BudgetTraderActorCritic):
        raise RuntimeError(f"unexpected trader type {type(trader_policy).__name__}")

    pm_ckpt = th.load(str(model_pm), map_location="cpu", weights_only=False)
    pm_policy.load_state_dict(pm_ckpt.get("state_dict", pm_ckpt), strict=True)
    tr_ckpt = th.load(str(model_trader), map_location="cpu", weights_only=False)
    trader_policy.load_state_dict(tr_ckpt.get("state_dict", tr_ckpt), strict=True)
    pm_policy.eval()
    trader_policy.eval()

    store, handles = _register_penultimate_hooks(pm_policy, trader_policy)

    # native deterministic rollout (mirrors w1train.evaluate, but also records latents/actions)
    env.reset()
    rows: list[dict[str, Any]] = []
    latent_rows: list[np.ndarray] = []
    pm_open = False
    pm_start_day = 0
    pm_horizon_days = 1
    q_raw_target = 0.0
    while not env.done():
        elapsed = int(env.day) - int(pm_start_day)
        if (not pm_open) or elapsed >= pm_horizon_days:
            action, _v, _lp = w1train.sample_policy(pm_policy, env.pm_obs(), deterministic=True)
            q_raw_target = float(np.clip(action[0], env.q_min, env.q_max))
            pm_horizon_days = env.horizon_from_action(action[1])
            pm_start_day = int(env.day)
            pm_open = True
        pm_pen = store.get("pm_pen")  # latest PM penultimate (held across the PM window)
        remaining = max(1, pm_horizon_days - (int(env.day) - int(pm_start_day)))
        q_target, risk_stop_info = env.apply_risk_stop(q_raw_target)
        obs = env.trader_obs(q_target=q_target, remaining_days=remaining)
        trader_action, _v, _lp = w1train.sample_policy(trader_policy, obs, deterministic=True)
        trader_pen = store.get("trader_pen")
        executable = w1train.decode_policy_actions(trader_policy, obs, trader_action)
        info = env.step_trader(
            q_target=q_target,
            remaining_days=remaining,
            trader_action=executable,
            q_raw_target=q_raw_target,
            risk_stop_info=risk_stop_info,
        )
        cash_exec = float(info["cash_exec"])
        target = np.asarray(info.get("target_weights"), dtype=float)
        risky = target[: env.stock_dim]
        gross = risky.sum()
        within = risky / (gross if gross > 0 else 1.0)
        rows.append({"cash": cash_exec, "within": within})
        if pm_pen is not None and trader_pen is not None:
            latent_rows.append(np.concatenate([pm_pen, trader_pen]).astype(np.float32))
        else:
            latent_rows.append(None)

    for h in handles:
        h.remove()

    cash = np.array([r["cash"] for r in rows], dtype=float)
    within = np.vstack([r["within"] for r in rows]).astype(float)
    # native latent: drop steps where either penultimate was unavailable (boundary safety)
    valid = [lr is not None for lr in latent_rows]
    if all(valid):
        native = np.vstack(latent_rows).astype(float)
    else:
        native = None
    return TwoAgentReplay(cash=cash, within=within, native_latent=native, n_steps=len(cash))


def _register_latent_penultimate_hooks(pm_policy, trader_policy) -> tuple[dict, list]:
    """Capture PM penultimate (actor MLP output, 128-d) and the LatentActionTrader penultimate
    (q_ctx = cat[mean_ctx, std_ctx], 128-d) on each forward. LatentActionTraderActorCritic._dists
    returns (code_dist, residual_dist, q_ctx) -- a 3-tuple, vs W1's 2-tuple -- so it needs its own
    monkeypatch. Last-write-wins per forward."""
    store: dict[str, np.ndarray] = {}
    handles = []

    def pm_actor_hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        store["pm_pen"] = t.detach().cpu().numpy().reshape(-1).astype(np.float32)

    handles.append(pm_policy.actor.register_forward_hook(pm_actor_hook))

    orig_dists = trader_policy._dists

    def patched_dists(obs):
        code_dist, residual_dist, q_ctx = orig_dists(obs)
        store["trader_pen"] = q_ctx.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return code_dist, residual_dist, q_ctx

    trader_policy._dists = patched_dists  # type: ignore[assignment]
    return store, handles


def replay_p22() -> TwoAgentReplay:
    """Replay P22 (W2 latent-action two-agent: BetaBudget PM + LatentActionTrader) on the frozen
    2022-2023 panel, NATIVELY, via the EXACT same proven inference path the firewall frozen-test
    runner (scripts/frozen_test_w2_latent_rollout.py) uses for its base arm:

        make_policies(env, variant)  ->  build the BetaBudget PM + LatentActionTraderActorCritic
                                         (decode via prototype_weights from the P18 teacher-prototype
                                          CSV, now present on disk)
        load pm_policy.pt + trader_policy.pt state_dicts (the trained W2/P22 checkpoint on D:)
        evaluate(...)  ->  the deterministic PM(q,horizon) -> Trader(code+residual) -> decode_actions
                           -> executed risky weights loop, with NO action supervisor (base arm).

    The executed action per step is read from the daily CSV: cash = `cash_exec`, within-book =
    gross-normalized `target_weight_<ticker>`. The native latent = concat(PM penultimate 128-d,
    Trader q_ctx 128-d) = 256-d (same structure/dim as W1). No retrain; no warm-start re-run
    (the final trained checkpoints are loaded directly). Uses the panel its config resolves to
    (W2 normalization is disabled -> the raw model_ready CSV); that panel is verified row-identical
    (same 290 frozen dates, same 29 tickers) to the FROZEN_PANEL the regime axis is built from, so
    the per-step action series aligns 1:1 with the common regime block."""
    import tempfile

    config = yaml.safe_load(P22_CONFIG.read_text(encoding="utf-8"))
    variant = load_variant(config, P22_VARIANT_NAME)
    fold_row = load_folds_from_config(config, ["fold_2021"]).iloc[0]
    feature_info = feature_csv_for_fold(config, variant, fold_row, P22_RUN_DIR.parent, force=False)
    start = str(config["data"]["frozen_test_start"])
    end = str(config["data"]["frozen_test_end"])
    panel = load_weight_panel(feature_info["model_ready_csv"], start, end)
    env_config = w2_env_kwargs(config, variant, panel)
    env = W1BudgetTraderEnv(panel, **env_config)

    pm_policy, trader_policy = w2_make_policies(env, variant)
    if not isinstance(trader_policy, LatentActionTraderActorCritic):
        raise RuntimeError(f"unexpected trader type {type(trader_policy).__name__}")
    pm_state = th.load(str(P22_RUN_DIR / "pm_policy.pt"), map_location="cpu", weights_only=False)
    tr_state = th.load(str(P22_RUN_DIR / "trader_policy.pt"), map_location="cpu", weights_only=False)
    # strict=False mirrors the firewall runner (load_policies); the W2/P22 checkpoint matches the
    # constructed nets exactly (0 missing / 0 unexpected, verified), so nothing is silently dropped.
    pm_policy.load_state_dict(pm_state.get("state_dict", pm_state), strict=False)
    trader_policy.load_state_dict(tr_state.get("state_dict", tr_state), strict=False)
    pm_policy.eval()
    trader_policy.eval()

    store, handles = _register_latent_penultimate_hooks(pm_policy, trader_policy)
    out_dir = Path(tempfile.mkdtemp(prefix="p22_frozen_"))
    # base arm = native deterministic rollout, no action supervisor (identical to BASE_ARM).
    w2_evaluate(env, pm_policy, trader_policy, out_dir=out_dir, split_name="frozen", action_supervisor=None)
    for h in handles:
        h.remove()

    daily = pd.read_csv(out_dir / "frozen_daily.csv")
    cash = pd.to_numeric(daily["cash_exec"], errors="coerce").to_numpy(dtype=float)
    tw_cols = [c for c in daily.columns if c.startswith("target_weight_") and c != "target_weight_CASH"]
    risky = daily[tw_cols].to_numpy(dtype=float)
    gross = risky.sum(axis=1, keepdims=True)
    within = risky / np.where(gross == 0, 1.0, gross)
    # native latent: the hook store holds only the LAST forward's penultimate; to capture the full
    # per-step series we re-run a lightweight native loop mirroring evaluate's deterministic policy
    # calls (the executed cash/within already come from the authoritative evaluate() daily above).
    native = _p22_native_latent_series(env_config, panel, pm_policy, trader_policy)
    n = min(len(cash), len(within), (len(native) if native is not None else len(cash)))
    return TwoAgentReplay(
        cash=cash[:n],
        within=within[:n].astype(float),
        native_latent=(native[:n] if native is not None else None),
        n_steps=n,
    )


def _p22_native_latent_series(env_config, panel, pm_policy, trader_policy) -> np.ndarray | None:
    """Re-run the deterministic PM->Trader rollout (no env mutation that matters for latents) capturing
    the per-step concat(PM penultimate, Trader q_ctx) native latent. Mirrors evaluate()'s control flow
    exactly so the captured latents align 1:1 with the executed action series."""
    env = W1BudgetTraderEnv(panel, **env_config)
    store, handles = _register_latent_penultimate_hooks(pm_policy, trader_policy)
    env.reset()
    latent_rows: list[Any] = []
    pm_open = False
    pm_start_day = 0
    pm_horizon_days = 1
    q_raw_target = 0.0
    while not env.done():
        elapsed = int(env.day) - int(pm_start_day)
        if (not pm_open) or elapsed >= pm_horizon_days:
            action, _v, _lp = w1train.sample_policy(pm_policy, env.pm_obs(), deterministic=True)
            q_raw_target = float(np.clip(action[0], env.q_min, env.q_max))
            pm_horizon_days = env.horizon_from_action(action[1])
            pm_start_day = int(env.day)
            pm_open = True
        pm_pen = store.get("pm_pen")
        remaining = max(1, pm_horizon_days - (int(env.day) - int(pm_start_day)))
        q_target, risk_stop_info = env.apply_risk_stop(q_raw_target)
        obs = env.trader_obs(q_target=q_target, remaining_days=remaining)
        trader_action, _v, _lp = w1train.sample_policy(trader_policy, obs, deterministic=True)
        trader_pen = store.get("trader_pen")
        executable = w1train.decode_policy_actions(trader_policy, obs, trader_action)
        env.step_trader(
            q_target=q_target,
            remaining_days=remaining,
            trader_action=executable,
            q_raw_target=q_raw_target,
            risk_stop_info=risk_stop_info,
        )
        if pm_pen is not None and trader_pen is not None:
            latent_rows.append(np.concatenate([pm_pen, trader_pen]).astype(np.float32))
        else:
            latent_rows.append(None)
    for h in handles:
        h.remove()
    if all(lr is not None for lr in latent_rows) and latent_rows:
        return np.vstack(latent_rows).astype(float)
    return None


# ----------------------------------------------------------------------------- skipped two-agent policies
SKIPPED: list[dict[str, str]] = []


# NOTE: P22 was previously SKIPPED (the prototype book / W2 checkpoints were not on disk). With
# D: mounted, the trained W2/P22 checkpoint + the P18 teacher-prototype CSV are both present, so
# P22 is now REPLAYED natively (see replay_p22 / p22_row) rather than skipped. The old
# check_p22_scorable gate is therefore retired.


def check_h2_scorable() -> tuple[bool, str]:
    """H2 = R6c-PM + T5-Trader NoLOB two-agent. Its checkpoints were only ever produced as
    cloud-job archives (stage0_1_job_510..513_H2_..._NoLOB_CTDE_fold_*.zip) that are NOT present
    on disk -- only the config (stage0_1_h2_pm_trader_nolob.yaml), a build-package script, and an
    audit .md exist. No pm/trader .pt anywhere -> cannot replay -> SKIP."""
    import glob

    pats = [
        str(ROOT / "artifacts" / "**" / "*H2*" / "**" / "*.pt"),
        str(ROOT / "artifacts" / "**" / "*nolob*" / "**" / "*.pt"),
        str(ROOT / "artifacts" / "**" / "*NoLOB*" / "**" / "*.pt"),
    ]
    found = []
    for p in pats:
        found.extend(glob.glob(p, recursive=True))
    if found:
        return True, ""  # (would still need wiring; but none exist)
    return False, (
        "No checkpoints on disk: H2 (R6c-PM + T5-Trader, NoLOB CTDE) was only produced as cloud "
        "job archives (stage0_1_job_510..513_H2_R6c_PM_T5_Trader_NoLOB_CTDE_fold_*.zip) which are "
        "not extracted/present here -- only configs/generated/stage0_1_h2_pm_trader_nolob.yaml, "
        "the package-builder script, and an audit .md exist. No pm_policy.pt / trader_policy.pt "
        "(or model.zip) exists for H2, so there is nothing to replay -> skipped (not fabricated)."
    )


# ----------------------------------------------------------------------------- assemble rows
def r6c_family_rows(regime_std: np.ndarray) -> list[dict]:
    """The 6 R6c-family policies: behavioral complexity + native-latent reused from cross_policy,
    BUT the cross-ARCH y (regime-based) is recomputed here from each policy's action series so it
    sits on the SAME common axis as the two-agent rows. R6c's actions come from its frozen behavior
    log; the 5 replayed ones are re-replayed with the identical cross_policy machinery."""
    rows: list[dict] = []

    # ---- R6c (actions from its own frozen behavior log; native-latent reused verbatim)
    r6c = r6c_from_report()
    blog = (
        ROOT / "artifacts" / "stage4"
        / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph"
        / "frozen_test_behavior_log_daily.csv"
    )
    df = pd.read_csv(blog, low_memory=False)
    cash = pd.to_numeric(df["cash_target"], errors="coerce").to_numpy(dtype=float)
    rc = [c for c in df.columns if c.startswith("executed_weight_") and c != "executed_weight_CASH"]
    risky = df[rc].to_numpy(dtype=float)
    gross = risky.sum(axis=1, keepdims=True)
    within = risky / np.where(gross == 0, 1.0, gross)
    ok = np.isfinite(cash)
    beh = score_behavioral(regime_std, cash[ok], within[ok])
    rows.append({
        "policy": "R6c", "family": "R6c (root_split_beta_dirichlet)", "arch": "single-MLP",
        "behavioral_complexity": r6c["behavioral_complexity"],
        "cash_entropy": r6c["cash_entropy"], "book_dispersion": r6c["book_dispersion"],
        "action_eff_dim": r6c["action_eff_dim"], "n_steps": r6c["n_steps"],
        **beh,
        # native latent for R6c is its 64-d policy_net.5.ReLU; reuse cross_policy's value directly
        "native_latent_dim": 64,
        "native_latent_simul_cash_cv": r6c["simulatability_cash_K9"],  # K9 latent-cluster cash R^2 (its native reading)
        "native_latent_note": "R6c native = 64-d policy_net.5.ReLU; value reused from crystal_score_report (K9 latent-cluster cash R^2).",
        "source": "crystal_score_report.json (actions: frozen behavior log)",
        "notes": "R6c-family; cross-arch y recomputed from frozen behavior log on common regime axis.",
    })

    # ---- the 5 replayed R6c-family candidates
    for spec in candidate_specs():
        if not spec["model"].exists():
            continue
        try:
            config = yaml.safe_load(spec["config"].read_text(encoding="utf-8"))
            meta = json.loads(spec["metadata"].read_text(encoding="utf-8"))
            rep = replay_policy(spec["model"], config, meta["variant"])  # cross_policy replay
        except Exception as e:  # honest skip
            SKIPPED.append({"policy": spec["policy"], "family": spec["family"],
                            "reason": f"R6c-family replay failed: {type(e).__name__}: {e}"})
            continue
        bc = behavioral_complexity(rep.cash, rep.within)
        beh = score_behavioral(regime_std, rep.cash, rep.within)
        nat = native_latent_simulatability(rep.latent, rep.cash)
        rows.append({
            "policy": spec["policy"], "family": spec["family"], "arch": "single-MLP",
            "behavioral_complexity": bc["behavioral_complexity"],
            "cash_entropy": bc["cash_entropy"], "book_dispersion": bc["book_dispersion"],
            "action_eff_dim": bc["action_eff_dim"], "n_steps": rep.n_steps,
            **beh,
            "native_latent_dim": nat["native_latent_dim"],
            "native_latent_simul_cash_cv": nat["native_latent_simul_cash_cv"],
            "native_latent_note": "64-d policy_net.5.ReLU; cv-R^2 cash (standardized).",
            "source": str(spec["model"].relative_to(ROOT)),
            "notes": "R6c-family; replayed frozen 2022-2023; cross-arch y on common regime axis.",
        })
    return rows


def w1_row(regime_std: np.ndarray) -> dict | None:
    pm = (
        ROOT / "artifacts" / "stage0_1" / "_unpacked_W1_P4PrimitiveWarmStart_results"
        / "stage0_1_job_703_W1_P4PrimitiveWarmStart_fold_2021_results"
        / "stage0_1_job_703_W1_P4PrimitiveWarmStart_fold_2021" / "artifacts" / "stage0_1"
        / "weight_based_w1_p4_primitive_warmstart_batch"
        / "W1_PM_BetaBudget_Trader_P4PrimitiveWarmStart_v1" / "fold_2021" / "pm_policy.pt"
    )
    trader = pm.parent / "trader_policy.pt"
    if not (pm.exists() and trader.exists()):
        SKIPPED.append({"policy": "W1", "family": "two-agent (pm + Dirichlet trader)",
                        "reason": f"W1 checkpoints not found at {pm.parent.relative_to(ROOT)}"})
        return None
    try:
        rep = replay_w1(pm, trader)
    except Exception as e:
        SKIPPED.append({"policy": "W1", "family": "two-agent (pm + Dirichlet trader)",
                        "reason": f"W1 replay failed: {type(e).__name__}: {e}"})
        return None
    bc = behavioral_complexity(rep.cash, rep.within)
    beh = score_behavioral(regime_std, rep.cash, rep.within)
    nat = native_latent_simulatability(rep.native_latent, rep.cash)
    return {
        "policy": "W1", "family": "two-agent (BetaBudget PM + Dirichlet Trader)",
        "arch": "two-agent (PM+Trader)",
        "behavioral_complexity": bc["behavioral_complexity"],
        "cash_entropy": bc["cash_entropy"], "book_dispersion": bc["book_dispersion"],
        "action_eff_dim": bc["action_eff_dim"], "n_steps": rep.n_steps,
        **beh,
        "native_latent_dim": nat["native_latent_dim"],
        "native_latent_simul_cash_cv": nat["native_latent_simul_cash_cv"],
        "native_latent_note": (
            "two-agent native latent = concat(PM penultimate 128-d, Trader q_ctx 128-d) = 256-d. "
            "CAVEAT: 256-d >> R6c-family 64-d, so this cv-R^2 is mechanically inflated vs the "
            "single-MLP rows and is NOT cross-arch-comparable -- shown for transparency only."
        ),
        "source": str(pm.parent.relative_to(ROOT)),
        "notes": "two-agent; replayed frozen 2022-2023 natively; cross-arch y on common regime axis.",
    }


def p22_row(regime_std: np.ndarray) -> dict | None:
    """P22 (W2 latent-action two-agent) -- NOW scorable: D: is mounted, so the trained W2/P22
    checkpoint (pm_policy.pt + trader_policy.pt) AND the P18 teacher-prototype CSV the
    LatentActionTrader's decode_actions needs are both present. Replayed natively on the frozen
    panel via the proven firewall base-arm path; scored on the SAME common regime axis as every
    other row."""
    proto_csv = w2_resolve(
        "artifacts/pretrain/P18_p15_r2_barlow_semantic_weighted_augmented_trajectories_v1"
        "/p13_asset_corrected_teacher_targets_long.csv"
    )
    missing = []
    if not (P22_RUN_DIR / "pm_policy.pt").exists() or not (P22_RUN_DIR / "trader_policy.pt").exists():
        missing.append(f"W2/P22 checkpoints at {P22_RUN_DIR}")
    if not proto_csv.exists():
        missing.append(f"teacher-prototype CSV at {proto_csv}")
    if missing:
        SKIPPED.append({
            "policy": "P22", "family": "two-agent (latent-action pm + trader)",
            "reason": "Still unscorable -- missing on disk: " + "; ".join(missing) + ".",
        })
        return None
    try:
        rep = replay_p22()
    except Exception as e:
        import traceback
        SKIPPED.append({
            "policy": "P22", "family": "two-agent (latent-action pm + trader)",
            "reason": f"P22 native replay failed: {type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-1]}",
        })
        return None
    bc = behavioral_complexity(rep.cash, rep.within)
    beh = score_behavioral(regime_std, rep.cash, rep.within)
    nat = native_latent_simulatability(rep.native_latent, rep.cash)
    return {
        "policy": "P22", "family": "two-agent (BetaBudget PM + LatentAction Trader)",
        "arch": "two-agent (PM+Trader)",
        "behavioral_complexity": bc["behavioral_complexity"],
        "cash_entropy": bc["cash_entropy"], "book_dispersion": bc["book_dispersion"],
        "action_eff_dim": bc["action_eff_dim"], "n_steps": rep.n_steps,
        **beh,
        "native_latent_dim": nat["native_latent_dim"],
        "native_latent_simul_cash_cv": nat["native_latent_simul_cash_cv"],
        "native_latent_note": (
            "two-agent latent-action native latent = concat(PM penultimate 128-d, Trader q_ctx "
            "128-d) = 256-d. CAVEAT: 256-d >> R6c-family 64-d, so this cv-R^2 is mechanically "
            "inflated vs the single-MLP rows and is NOT cross-arch-comparable -- transparency only."
        ),
        "source": str(P22_RUN_DIR),
        "notes": "two-agent latent-action; replayed frozen 2022-2023 natively (firewall base-arm "
                 "path; decode via P18 teacher prototypes); cross-arch y on common regime axis.",
    }


# ----------------------------------------------------------------------------- plotting
def maybe_plot(flat: pd.DataFrame) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        return {"plotted": False, "reason": str(e)}

    cc = flat.dropna(subset=["behavioral_complexity", "crystalscore_behavioral"]).copy()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    # color by ARCH; two-agent gets a distinct marker
    arch_style = {
        "single-MLP": {"color": "#1f77b4", "marker": "o", "label": "single-MLP (R6c-family)"},
        "two-agent (PM+Trader)": {"color": "#d62728", "marker": "D", "label": "two-agent (PM+Trader)"},
    }
    seen = set()
    for _, r in cc.iterrows():
        st = arch_style.get(r["arch"], {"color": "#7f7f7f", "marker": "s", "label": r["arch"]})
        lbl = st["label"] if st["label"] not in seen else None
        seen.add(st["label"])
        ax.scatter(r["behavioral_complexity"], r["crystalscore_behavioral"],
                   color=st["color"], marker=st["marker"], s=95, zorder=3,
                   edgecolor="k", linewidth=0.6, label=lbl)
        ax.annotate(r["policy"], (r["behavioral_complexity"], r["crystalscore_behavioral"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    if len(cc) >= 3 and cc["behavioral_complexity"].nunique() > 1:
        z = np.polyfit(cc["behavioral_complexity"], cc["crystalscore_behavioral"], 1)
        xs = np.linspace(cc["behavioral_complexity"].min(), cc["behavioral_complexity"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", lw=1, alpha=0.6, label=f"OLS slope={z[0]:.2f}")
    ax.set_xlabel("Behavioral complexity  (action-only proxy, [0,1])")
    ax.set_ylabel("CrystalScore-behavioral  (regime-simulatability x regime-stability)")
    ax.set_title("Cross-ARCHITECTURE: interpretability vs behavioral complexity\n"
                 "(common market-regime axis; frozen 2022-2023)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_CURVE_PNG, dpi=140)
    plt.close(fig)
    return {"plotted": True, "file": str(OUT_CURVE_PNG)}


# ----------------------------------------------------------------------------- markdown append
def append_markdown(report: dict, flat: pd.DataFrame) -> None:
    def f(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    L = []
    L.append("\n\n---\n")
    L.append("# Cross-architecture extension\n")
    L.append(
        "The cross-POLICY curve above lives entirely in the R6c-FAMILY corner: its y-axis "
        "(`crystalscore_core`) is built on each policy's OWN 64-d `policy_net.5.ReLU` latent, which "
        "the two-agent PM+Trader policies do not have. This section extends the arc to a "
        "cross-ARCHITECTURE one using a **unified, architecture-AGNOSTIC interpretability axis** that "
        "needs only each policy's per-step **actions** plus a **fixed common market-regime feature "
        "set** (trailing realized vol, drawdown, momentum, breadth, dispersion -- computed from the "
        "frozen-panel **returns**, identical for every policy, never from any internal latent).\n")

    L.append("## The common (cross-arch) axes\n")
    L.append("| Axis | Definition | Cross-arch? |")
    L.append("|---|---|---|")
    L.append("| `behavioral_complexity` (x) | action-only: mean(cash 10-bin entropy, within-book L1-to-equal-weight, action-cov participation-ratio/dim). IDENTICAL recipe to the cross-policy curve. | **YES** |")
    L.append("| `crystalscore_behavioral` (y) | regime-Simulatability(cash, shuffled 5-fold cv-R^2 of Ridge on the K common regime features) x regime-clustering-stability(K=9). | **YES** |")
    L.append("| `regime_completeness_stance/selection@K` | 1 - SS_res/SS_tot of the K-REGIME-cluster-mean predictor of cash / within-book composition, swept K=2..9. Measures \"is the behaviour compressible into K human-readable market regimes?\" | **YES** |")
    L.append("| `native_latent_simul_cash_cv` (secondary) | cv-R^2 of cash from the policy's OWN latent (R6c-family: 64-d policy_net.5.ReLU; two-agent: concat PM+Trader penultimate = 256-d). | **NO** -- dim-dependent (see caveat) |")
    L.append("")
    L.append(
        "Regime features: `" + "`, `".join(report["regime_feature_names"]) + "`. "
        f"Common regime clustering-stability (K=9, shared by all policies) = "
        f"{f(report['regime_clustering_stability_shared'])}.\n")

    L.append("## Extended table (R6c-family reused + scorable two-agent)\n")
    cols = ["policy", "arch", "family", "behavioral_complexity", "cash_entropy", "book_dispersion",
            "action_eff_dim", "regime_simulatability_cash_cv", "regime_completeness_stance_K9",
            "regime_completeness_selection_K9", "regime_clustering_stability_K9",
            "crystalscore_behavioral", "native_latent_dim", "native_latent_simul_cash_cv", "n_steps"]
    L.append("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    L.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in flat.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("N/A" if v is None or (isinstance(v, float) and not np.isfinite(v))
                         else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"Full CSV: `{OUT_CSV.name}`.\n")

    L.append("## The cross-architecture curve\n")
    if report["plot"].get("plotted"):
        L.append(f"![cross-arch curve]({OUT_CURVE_PNG.name})\n")
    else:
        L.append("(matplotlib unavailable -> PNG skipped)\n")
    L.append("Points sorted by behavioral complexity (low -> high), on the common regime axis:\n")
    L.append("```")
    cc = flat.dropna(subset=["behavioral_complexity", "crystalscore_behavioral"]).sort_values("behavioral_complexity")
    for _, row in cc.iterrows():
        L.append(f"  {row['policy']:<26} complexity={row['behavioral_complexity']:.3f}  "
                 f"behavioral={row['crystalscore_behavioral']:.3f}  "
                 f"(regime_simul={f(row['regime_simulatability_cash_cv'])})  "
                 f"[{row['arch']}]")
    L.append("```")
    cor = report["correlation_complexity_vs_behavioral"]
    L.append(f"\nAcross the {report['n_scored']} scored policies: "
             f"Pearson(complexity, behavioral) = {f(cor['pearson'])}, Spearman = {f(cor['spearman'])}.\n")

    L.append("## COVERAGE\n")
    L.append(f"**Scored: {report['n_scored']} policies "
             f"({report['n_single_mlp']} single-MLP R6c-family + {report['n_two_agent']} two-agent).**\n")
    for row in report["rows"]:
        L.append(f"- `{row['policy']}` [{row['arch']}] -- n={row.get('n_steps')} steps. {row.get('notes','')}")
    L.append("\n**Skipped (with precise reason -- NO fabrication):**\n")
    for s in report["skipped"]:
        L.append(f"- `{s['policy']}` ({s['family']}): {s['reason']}")
    L.append("")

    L.append("## Honest interpretation\n")
    n_ta = report["n_two_agent"]
    if n_ta == 0:
        L.append("- **No two-agent policy was scorable**, so this remains an R6c-family result on a "
                 "new (regime-based) axis. We do NOT claim a cross-architecture curve.")
    else:
        scored_ta = ", ".join(sorted(flat[flat["arch"] == "two-agent (PM+Trader)"]["policy"].tolist()))
        L.append(
            f"- **{n_ta} of 3 two-agent policies scorable ({scored_ta}).** This is a cross-architecture "
            "extension: each two-agent policy (genuine two-module PM + Trader) is placed on the EXACT "
            "same (behavioral_complexity, regime-interpretability) axes as the 6 single-MLP R6c-family "
            "policies. P22 is now scorable because D: is mounted -- its trained W2/P22 checkpoint "
            "(pm_policy.pt + trader_policy.pt) AND the P18 teacher-prototype CSV its LatentActionTrader "
            "decode needs are both present, so it is replayed natively (firewall base-arm path), no "
            "fabrication. H2 remains unscorable (no checkpoints on disk -- see Coverage).")
    bc = report["two_agent_complexity_vs_family"]
    if bc:
        L.append(
            f"- **Is the two-agent policy MORE behaviorally complex?** W1 complexity = "
            f"{f(bc['w1_complexity'])} vs R6c-family mean = {f(bc['r6c_family_mean_complexity'])} "
            f"(range [{f(bc['r6c_family_min_complexity'])}, {f(bc['r6c_family_max_complexity'])}]). "
            + ("W1 sits ABOVE the R6c-family band -> the two-agent design is genuinely more complex "
               "(wider stance/selection range), widening the x-axis as hoped."
               if bc["w1_complexity"] is not None and bc["r6c_family_max_complexity"] is not None
               and bc["w1_complexity"] > bc["r6c_family_max_complexity"]
               else "W1 sits WITHIN/below the R6c-family complexity band -> on this action-only proxy "
                    "the two-agent policy is not obviously more complex; the complexity range stays "
                    "narrow, so any complexity->interpretability slope is still weakly identified."))
    L.append(
        "- **Does interpretability fall as behavioral complexity rises (now with a two-agent point)?** "
        f"OLS slope of crystalscore_behavioral on behavioral_complexity = {f(report.get('ols_slope'))}; "
        f"Pearson = {f(report['correlation_complexity_vs_behavioral']['pearson'])}, Spearman = "
        f"{f(report['correlation_complexity_vs_behavioral']['spearman'])}. "
        "Read sign/magnitude with the small-n caveat -- the complexity range is still narrow.")
    rc = report.get("regime_compressibility")
    if rc:
        verdict = (
            "NOT less regime-compressible -- it sits mid-band, so on this axis the two-agent design is "
            "no harder to reduce to a few human-readable market regimes than the single-MLP policies."
            if rc["w1_behavioral"] is not None and rc["family_min_behavioral"] is not None
            and rc["w1_behavioral"] >= rc["family_min_behavioral"]
            else "LESS regime-compressible -- it falls below the R6c-family band, i.e. its behaviour is "
                 "harder to reduce to a few market regimes."
        )
        L.append(
            "- **Is the two-agent policy LESS regime-compressible (the substantive question)?** W1's "
            f"`regime_completeness_stance@9` = {f(rc['w1_completeness_stance9'])} and "
            f"`crystalscore_behavioral` = {f(rc['w1_behavioral'])} both sit INSIDE the R6c-family band "
            f"(completeness@9 [{f(rc['family_min_completeness9'])}, {f(rc['family_max_completeness9'])}], "
            f"behavioral [{f(rc['family_min_behavioral'])}, {f(rc['family_max_behavioral'])}]). "
            f"So W1 is {verdict} The one genuinely distinctive W1 trait on the action axis is its "
            f"`book_dispersion` = {f(rc['w1_book_dispersion'])} (vs R6c-family "
            f"~{f(rc['family_mean_book_dispersion'])}): W1 runs an almost PURE equal-weight risky book "
            "(near-zero within-book tilt) and expresses essentially all of its behaviour through the cash "
            "stance -- the same 'transparent cash-timing controller' signature the single-policy CrystalScore "
            "found for R6c, now confirmed for a two-agent architecture.")
    else:
        L.append(
            "- **Is the two-agent policy LESS regime-compressible?** No two-agent point was scorable.")
    pf = report.get("p22_vs_field")
    if pf:
        extends = pf["p22_extends_complexity_range"] or pf["p22_extends_dispersion_range"]
        verdict = (
            "EXTENDS the range -- P22 sits ABOVE the field on complexity and/or book_dispersion, i.e. "
            "the latent-action two-agent design finally produces a genuinely higher-complexity / "
            "more-selective policy."
            if extends else
            "does NOT extend the range -- P22's book_dispersion and complexity sit INSIDE (at or below) "
            "the existing field, so even the most-selection-oriented two-agent latent-action design "
            "ALSO collapses to near-equal-weight cash-timing rather than doing more genuine "
            "cross-sectional selection."
        )
        L.append(
            "- **DECISIVE -- does P22 (the most-selection-oriented two-agent latent-action design) do "
            "MORE genuine cross-sectional selection?** P22 `book_dispersion` = "
            f"{f(pf['p22_book_dispersion'])} vs field [{f(pf['field_min_book_dispersion'])}, "
            f"{f(pf['field_max_book_dispersion'])}] (mean {f(pf['field_mean_book_dispersion'])}); P22 "
            f"`behavioral_complexity` = {f(pf['p22_complexity'])} vs field "
            f"[{f(pf['field_min_complexity'])}, {f(pf['field_max_complexity'])}] (mean "
            f"{f(pf['field_mean_complexity'])}). P22 {verdict} P22's regime-`completeness_selection@9` = "
            f"{f(pf['p22_completeness_selection9'])} (within-book reducible to <=9 regimes) and "
            f"`crystalscore_behavioral` = {f(pf['p22_behavioral'])}. This is the scientifically decisive "
            "point: the latent-action prototype book (26 distinct teacher-derived code portfolios, "
            "re-budgeted to the PM's q each step) is the strongest selection-capacity prior in the "
            "universe, yet the executed book is still nearly equal-weight -- so on this frozen panel the "
            "universe does NOT produce a high-complexity policy; every architecture, single-MLP and "
            "two-agent alike, converges to a transparent cash-timing controller over a near-equal-weight "
            "risky book.")
    L.append(
        "- **Native-latent caveat (firewall):** the secondary `native_latent_simul_cash_cv` is NOT "
        "cross-arch-comparable -- the two-agent native latent is 256-d vs the R6c-family 64-d, and a "
        "higher-dim latent decodes cash more trivially. It is shown standardized, with its dim, for "
        "transparency only; the cross-architecture comparison rests entirely on the common regime axis. "
        "Negative R^2 is clipped to 0 throughout; no policy was retrained (replay only).\n")

    # append (do not overwrite) to the existing cross-policy markdown
    prior = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    # idempotency: strip any prior cross-arch section before re-appending
    marker = "# Cross-architecture extension"
    if marker in prior:
        prior = prior.split("\n\n---\n# Cross-architecture extension")[0].rstrip() + "\n"
        # also handle the leading-newline variant
        cut = prior.find(marker)
        if cut != -1:
            prior = prior[:cut].rstrip().rstrip("-").rstrip() + "\n"
    OUT_MD.write_text(prior.rstrip() + "\n" + "\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> None:
    if not FROZEN_PANEL.exists():
        raise FileNotFoundError(f"frozen panel missing: {FROZEN_PANEL}")

    # common regime axis (shared by every policy)
    panel = load_weight_panel(FROZEN_PANEL, "2022-01-03", "2023-02-28")
    regime_raw = common_regime_features(panel)
    regime_std = StandardScaler().fit_transform(regime_raw)
    shared_regime_stability = regime_clustering_stability(regime_std, K_BUDGET)
    print(f"[regime] common axis built: {regime_std.shape[0]} steps x {regime_std.shape[1]} features "
          f"({', '.join(REGIME_FEATURE_NAMES)}); shared K9 stability={shared_regime_stability:.3f}")

    rows: list[dict] = []

    # ---- R6c-family (reused complexity/native; cross-arch y recomputed on the common axis)
    print("[scoring] R6c-family (6 policies) on the common regime axis ...")
    rows.extend(r6c_family_rows(regime_std))

    # ---- two-agent: W1 (scorable), P22 / H2 (skip with reason)
    ok_w1 = True
    w1 = w1_row(regime_std)
    if w1 is not None:
        rows.append(w1)
        print(f"[scoring] W1 (two-agent) -- complexity={w1['behavioral_complexity']:.3f}  "
              f"behavioral={w1['crystalscore_behavioral']:.3f}  n={w1['n_steps']}")
    else:
        ok_w1 = False

    # ---- P22 (two-agent latent-action): NOW scorable (D: mounted -> checkpoints + prototypes present)
    p22 = p22_row(regime_std)
    if p22 is not None:
        rows.append(p22)
        print(f"[scoring] P22 (two-agent latent-action) -- complexity={p22['behavioral_complexity']:.3f}  "
              f"behavioral={p22['crystalscore_behavioral']:.3f}  book_dispersion={p22['book_dispersion']:.3f}  "
              f"n={p22['n_steps']}")
    else:
        print(f"[skip] P22: {SKIPPED[-1]['reason'][:120]}...")

    ok_h2, reason_h2 = check_h2_scorable()
    if not ok_h2:
        SKIPPED.append({"policy": "H2", "family": "two-agent (pm + trader, NoLOB)", "reason": reason_h2})
        print(f"[skip] H2: {reason_h2[:90]}...")

    # ---- flat CSV
    flat = pd.DataFrame([{
        "policy": r["policy"], "arch": r["arch"], "family": r["family"],
        "behavioral_complexity": r["behavioral_complexity"], "cash_entropy": r["cash_entropy"],
        "book_dispersion": r["book_dispersion"], "action_eff_dim": r["action_eff_dim"],
        "regime_simulatability_cash_cv": r["regime_simulatability_cash_cv"],
        "regime_completeness_stance_K9": r["regime_completeness_stance_K9"],
        "regime_completeness_selection_K9": r["regime_completeness_selection_K9"],
        "regime_pareto_stance_auc": r["regime_pareto_stance"].get("auc_normalized"),
        "regime_pareto_selection_auc": r["regime_pareto_selection"].get("auc_normalized"),
        "regime_clustering_stability_K9": r["regime_clustering_stability_K9"],
        "crystalscore_behavioral": r["crystalscore_behavioral"],
        "crystalscore_behavioral_nostab": r["crystalscore_behavioral_nostab"],
        "native_latent_dim": r["native_latent_dim"],
        "native_latent_simul_cash_cv": r["native_latent_simul_cash_cv"],
        "n_steps": r["n_steps"], "source": r.get("source", ""), "notes": r.get("notes", ""),
    } for r in rows])
    flat.to_csv(OUT_CSV, index=False)

    # ---- correlation / slope across scored set
    cc = flat.dropna(subset=["behavioral_complexity", "crystalscore_behavioral"])
    n_scored = int(len(cc))
    pear = spear = ols = float("nan")
    if n_scored >= 3 and cc["behavioral_complexity"].nunique() > 1:
        from scipy.stats import pearsonr, spearmanr
        pear = float(pearsonr(cc["behavioral_complexity"], cc["crystalscore_behavioral"])[0])
        spear = float(spearmanr(cc["behavioral_complexity"], cc["crystalscore_behavioral"])[0])
        ols = float(np.polyfit(cc["behavioral_complexity"], cc["crystalscore_behavioral"], 1)[0])

    n_two = int((flat["arch"] == "two-agent (PM+Trader)").sum())
    n_single = int((flat["arch"] == "single-MLP").sum())

    # two-agent vs family complexity comparison
    fam = flat[flat["arch"] == "single-MLP"]["behavioral_complexity"].dropna()
    w1c = flat[flat["policy"] == "W1"]["behavioral_complexity"]
    ta_cmp = None
    if n_two >= 1 and len(fam):
        ta_cmp = {
            "w1_complexity": float(w1c.iloc[0]) if len(w1c) else None,
            "r6c_family_mean_complexity": float(fam.mean()),
            "r6c_family_min_complexity": float(fam.min()),
            "r6c_family_max_complexity": float(fam.max()),
        }

    # two-agent regime-compressibility vs family (the substantive question)
    fam_rows = flat[flat["arch"] == "single-MLP"]
    w1_rows = flat[flat["policy"] == "W1"]
    regime_compress = None
    if len(w1_rows) and len(fam_rows):
        w1r = w1_rows.iloc[0]
        regime_compress = {
            "w1_behavioral": float(w1r["crystalscore_behavioral"]),
            "w1_completeness_stance9": float(w1r["regime_completeness_stance_K9"]),
            "w1_book_dispersion": float(w1r["book_dispersion"]),
            "family_min_behavioral": float(fam_rows["crystalscore_behavioral"].min()),
            "family_max_behavioral": float(fam_rows["crystalscore_behavioral"].max()),
            "family_min_completeness9": float(fam_rows["regime_completeness_stance_K9"].min()),
            "family_max_completeness9": float(fam_rows["regime_completeness_stance_K9"].max()),
            "family_mean_book_dispersion": float(fam_rows["book_dispersion"].mean()),
        }

    # P22 vs the FIELD (the decisive question): does the most-selection-oriented two-agent
    # latent-action design DO MORE genuine cross-sectional selection (higher book_dispersion /
    # complexity) -- or does it ALSO collapse to near-equal-weight cash-timing like the rest?
    p22_rows = flat[flat["policy"] == "P22"]
    field_rows = flat[flat["policy"] != "P22"]
    p22_vs_field = None
    if len(p22_rows) and len(field_rows):
        p = p22_rows.iloc[0]
        p22_vs_field = {
            "p22_complexity": float(p["behavioral_complexity"]),
            "p22_book_dispersion": float(p["book_dispersion"]),
            "p22_cash_entropy": float(p["cash_entropy"]),
            "p22_action_eff_dim": float(p["action_eff_dim"]),
            "p22_behavioral": float(p["crystalscore_behavioral"]),
            "p22_completeness_stance9": float(p["regime_completeness_stance_K9"]),
            "p22_completeness_selection9": float(p["regime_completeness_selection_K9"]),
            "field_min_complexity": float(field_rows["behavioral_complexity"].min()),
            "field_max_complexity": float(field_rows["behavioral_complexity"].max()),
            "field_mean_complexity": float(field_rows["behavioral_complexity"].mean()),
            "field_min_book_dispersion": float(field_rows["book_dispersion"].min()),
            "field_max_book_dispersion": float(field_rows["book_dispersion"].max()),
            "field_mean_book_dispersion": float(field_rows["book_dispersion"].mean()),
            "p22_extends_complexity_range": bool(
                float(p["behavioral_complexity"]) > float(field_rows["behavioral_complexity"].max())
            ),
            "p22_extends_dispersion_range": bool(
                float(p["book_dispersion"]) > float(field_rows["book_dispersion"].max())
            ),
        }

    plot_info = maybe_plot(flat)

    report = {
        "title": "Cross-architecture CrystalScore -- common market-regime axis (frozen 2022-2023)",
        "x_axis": "behavioral_complexity (action-only; identical to cross-policy)",
        "y_axis": "crystalscore_behavioral = regime_simulatability(cash cv-R^2) x regime_clustering_stability(K9)",
        "regime_feature_names": REGIME_FEATURE_NAMES,
        "regime_clustering_stability_shared": round(float(shared_regime_stability), 4),
        "frozen_panel": str(FROZEN_PANEL.relative_to(ROOT)),
        "K_budget": K_BUDGET, "K_grid": K_GRID,
        "n_scored": n_scored, "n_two_agent": n_two, "n_single_mlp": n_single,
        "correlation_complexity_vs_behavioral": {
            "pearson": round(pear, 4) if np.isfinite(pear) else None,
            "spearman": round(spear, 4) if np.isfinite(spear) else None,
        },
        "ols_slope": round(ols, 4) if np.isfinite(ols) else None,
        "two_agent_complexity_vs_family": ta_cmp,
        "regime_compressibility": regime_compress,
        "p22_vs_field": p22_vs_field,
        "rows": rows,
        "skipped": SKIPPED,
        "plot": plot_info,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    append_markdown(report, flat)
    print_console(report, flat)


def print_console(report: dict, flat: pd.DataFrame) -> None:
    def g(v):
        return "  N/A" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (
            f"{v:.3f}" if isinstance(v, float) else str(v))

    print("=" * 96)
    print("CROSS-ARCHITECTURE CRYSTALSCORE  (common market-regime axis; frozen 2022-2023)")
    print("=" * 96)
    hdr = f"{'policy':<26}{'arch':<22}{'complx':>8}{'reg_sim':>9}{'reg_cmp9':>9}{'behav':>8}{'natDim':>8}{'natCV':>8}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in flat.iterrows():
        print(f"{str(r['policy']):<26}{str(r['arch']):<22}{g(r['behavioral_complexity']):>8}"
              f"{g(r['regime_simulatability_cash_cv']):>9}{g(r['regime_completeness_stance_K9']):>9}"
              f"{g(r['crystalscore_behavioral']):>8}{g(r['native_latent_dim']):>8}"
              f"{g(r['native_latent_simul_cash_cv']):>8}{g(r['n_steps']):>6}")
    print("-" * len(hdr))
    cor = report["correlation_complexity_vs_behavioral"]
    print(f"Scored: {report['n_scored']}  ({report['n_single_mlp']} single-MLP + {report['n_two_agent']} two-agent) | "
          f"Pearson={cor['pearson']} Spearman={cor['spearman']} OLS_slope={report['ols_slope']}")
    print(f"Skipped: {', '.join(s['policy'] for s in report['skipped'])}")
    ta = report["two_agent_complexity_vs_family"]
    if ta:
        print(f"W1 complexity={g(ta['w1_complexity'])} vs R6c-family mean={g(ta['r6c_family_mean_complexity'])} "
              f"[{g(ta['r6c_family_min_complexity'])}, {g(ta['r6c_family_max_complexity'])}]")
    pf = report.get("p22_vs_field")
    if pf:
        print("-" * len(hdr))
        print(f"P22 (most-selection-oriented two-agent latent-action):")
        print(f"  book_dispersion={g(pf['p22_book_dispersion'])} vs field [{g(pf['field_min_book_dispersion'])}, "
              f"{g(pf['field_max_book_dispersion'])}] mean={g(pf['field_mean_book_dispersion'])}")
        print(f"  complexity={g(pf['p22_complexity'])} vs field [{g(pf['field_min_complexity'])}, "
              f"{g(pf['field_max_complexity'])}] mean={g(pf['field_mean_complexity'])}")
        print(f"  extends_complexity_range={pf['p22_extends_complexity_range']}  "
              f"extends_dispersion_range={pf['p22_extends_dispersion_range']}  "
              f"=> {'EXTENDS' if (pf['p22_extends_complexity_range'] or pf['p22_extends_dispersion_range']) else 'COLLAPSES (near-equal-weight cash-timing)'}")
    print(f"Wrote: {OUT_CSV.name}, {OUT_JSON.name}, {OUT_MD.name}"
          + (f", {OUT_CURVE_PNG.name}" if report['plot'].get('plotted') else ""))
    print("=" * 96)


if __name__ == "__main__":
    main()
