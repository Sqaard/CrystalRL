"""Joint W1 solver for an Investor Contract and research-only path distributions.

The solver is intentionally not a recommender.  It evaluates the two inverse query forms on the same path
outcomes and emits a ``RESEARCH_ONLY_INTERFACE_PROTOTYPE`` selection.  W3 must replace the historical path
source, W4 must supply the production DP/MPC policy set, and W8 must certify the profile grid before any
result can enter a client menu or execution path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from interpretability.investor_contract import (
    INTERFACE_STATUS, ContractValidationError,
    DrawdownMetric,
    FeasibilityEvidence,
    InvestorContract,
    QueryKind,
    ReturnQuery,
    ValueBasis,
    assess_feasibility,
)
from interpretability.personal_invest_scenarios import (
    HISTORICAL_SCENARIO_STATUS,
    PathDistribution,
    conservative_return_floor,
    conservative_return_floor_lcb,
)


@dataclass(frozen=True)
class CandidatePolicy:
    candidate_id: str
    source_id: str
    distribution: PathDistribution
    currency: str
    value_basis: ValueBasis
    instrument_positions: tuple[tuple[str, str, float], ...]
    max_single_instrument_weight: float
    liquid_fraction: float
    max_lockup_days: int
    gross_leverage: float = 1.0
    uses_shorting: bool = False
    diagnostic_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id is required")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("candidate_id and source_id are required")
        if not isinstance(self.currency, str) or len(self.currency) != 3 or not self.currency.isupper():
            raise ValueError("currency must be an uppercase three-letter code")
        if not isinstance(self.value_basis, ValueBasis):
            raise ValueError("value_basis must be a ValueBasis")
        if not isinstance(self.distribution, PathDistribution):
            raise TypeError("distribution must be a PathDistribution")
        if self.distribution.status != HISTORICAL_SCENARIO_STATUS:
            raise ValueError("W1 accepts only explicitly uncalibrated historical scenarios")
        if isinstance(self.max_single_instrument_weight, bool) or not np.isfinite(self.max_single_instrument_weight):
            raise ValueError("max_single_instrument_weight must be finite")
        if not 0.0 <= self.max_single_instrument_weight <= 1.0:
            raise ValueError("max_single_instrument_weight must be in [0, 1]")
        positions = tuple(
            (str(name).strip(), str(asset_class).strip(), float(weight))
            for name, asset_class, weight in self.instrument_positions
        )
        if not positions or any(
            not name or not asset_class or not np.isfinite(weight) or weight <= 0.0
            for name, asset_class, weight in positions
        ):
            raise ValueError("instrument_positions must contain typed, unique, positive finite weights")
        if len({name.casefold() for name, _, _ in positions}) != len(positions):
            raise ValueError("instrument_positions contains duplicate instrument IDs")
        cash_like = {"cash", "money_market", "deposit"}
        if any(
            name.casefold() in cash_like and asset_class.casefold() not in cash_like
            for name, asset_class, _ in positions
        ):
            raise ValueError("cash-like instrument ID conflicts with its typed asset class")
        total_weight = sum(weight for _, _, weight in positions)
        if not np.isclose(total_weight, 1.0, atol=1e-9):
            raise ValueError("instrument_positions must sum to 1")
        observed_max = max(weight for _, _, weight in positions)
        if not np.isclose(observed_max, self.max_single_instrument_weight, atol=1e-9):
            raise ValueError("max_single_instrument_weight disagrees with instrument_positions")
        object.__setattr__(self, "instrument_positions", positions)
        if isinstance(self.liquid_fraction, bool) or not np.isfinite(self.liquid_fraction) or not 0.0 <= self.liquid_fraction <= 1.0:
            raise ValueError("liquid_fraction must be in [0, 1]")
        if isinstance(self.max_lockup_days, bool) or not isinstance(self.max_lockup_days, int) or self.max_lockup_days < 0:
            raise ValueError("max_lockup_days must be a non-negative integer")
        if isinstance(self.gross_leverage, bool) or not isinstance(self.gross_leverage, (int, float)):
            raise ValueError("gross_leverage must be a finite number")
        if not np.isfinite(self.gross_leverage) or self.gross_leverage <= 0.0:
            raise ValueError("invalid lockup or leverage")
        if not isinstance(self.uses_shorting, bool) or not isinstance(self.diagnostic_only, bool):
            raise ValueError("uses_shorting and diagnostic_only must be boolean")
        observed_gross = sum(abs(weight) for _, _, weight in positions)
        if not np.isclose(float(self.gross_leverage), observed_gross, atol=1e-9):
            raise ValueError("gross_leverage disagrees with typed instrument positions")
        if self.uses_shorting:
            raise ValueError("W1 typed positions are long-only; signed positions require a later engine")

    @property
    def asset_classes(self) -> tuple[str, ...]:
        return tuple(sorted({asset_class for _, asset_class, _ in self.instrument_positions}))


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    eligible: bool
    hard_capacity_satisfied: bool
    goal_probability_satisfied: bool
    instrument_constraints_satisfied: bool
    capacity_probability: float
    capacity_expected_excess_given_breach: float | None
    wealth_floor_probability: float
    wealth_floor_expected_shortfall_given_breach: float | None
    goal_probability: float
    goal_expected_shortfall_given_miss: float | None
    return_floor: float
    return_floor_probability: float
    return_expected_shortfall_below_floor: float | None
    target_return_probability: float | None
    target_expected_shortfall_given_miss: float | None
    selected_drawdown_metric: str
    drawdown_p95: float
    tolerance_penalty: float
    rejection_reasons: tuple[str, ...]
    # P3: the product-grade floor r*(p,H,u) = LCB[Q_{1-p}]; return_floor stays the exact quantile.
    return_floor_lcb: float = 0.0
    return_floor_lcb_se: float = 0.0
    # P3: CVaR of the shortfall below the goal CAGR, and whether an optional CVaR budget binds.
    cvar_shortfall_below_goal: float = 0.0
    cvar_shortfall_budget: float | None = None
    cvar_budget_satisfied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointSolution:
    feasible: bool
    contract_id: str
    goal_id: str
    query_kind: str
    selected_candidate_id: str | None
    query_result: dict[str, Any]
    evaluations: tuple[CandidateEvaluation, ...]
    feasibility: dict[str, Any]
    interface_status: str = INTERFACE_STATUS
    recommendable: bool = False
    certification_status: str = "NOT_CERTIFIED_FOR_PROFILE_GRID"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "contract_id": self.contract_id,
            "goal_id": self.goal_id,
            "query_kind": self.query_kind,
            "selected_candidate_id": self.selected_candidate_id,
            "query_result": self.query_result,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "feasibility": self.feasibility,
            "interface_status": self.interface_status,
            "recommendable": self.recommendable,
            "certification_status": self.certification_status,
        }


def _conditional_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(np.mean(values[mask])) if bool(mask.any()) else None


def _instrument_reasons(contract: InvestorContract, candidate: CandidatePolicy) -> list[str]:
    restrictions = contract.instruments
    reasons: list[str] = []
    allowed = {item.casefold() for item in restrictions.allowed_asset_classes}
    if not set(item.casefold() for item in candidate.asset_classes).issubset(allowed):
        reasons.append("candidate uses an asset class outside the Investor Contract")
    prohibited = {item.casefold() for item in restrictions.prohibited_instruments}
    prohibited_used = sorted(
        instrument_id for instrument_id, _, _ in candidate.instrument_positions
        if instrument_id.casefold() in prohibited
    )
    if prohibited_used:
        reasons.append("candidate uses prohibited instruments: " + ", ".join(prohibited_used))
    if candidate.max_single_instrument_weight > restrictions.max_single_instrument_weight:
        reasons.append("candidate exceeds max_single_instrument_weight")
    if candidate.uses_shorting and not restrictions.allow_shorting:
        reasons.append("candidate uses prohibited shorting")
    if candidate.gross_leverage > restrictions.max_gross_leverage:
        reasons.append("candidate exceeds max_gross_leverage")
    if candidate.gross_leverage > 1.0 and not restrictions.allow_leverage:
        reasons.append("candidate uses prohibited leverage")
    if candidate.liquid_fraction < contract.liquidity.min_liquid_fraction:
        reasons.append("candidate violates min_liquid_fraction")
    if candidate.max_lockup_days > contract.liquidity.max_lockup_days:
        reasons.append("candidate exceeds max_lockup_days")
    reserve_fraction = contract.liquidity.emergency_reserve / contract.initial_wealth
    cash_like_weight = sum(
        weight for _, asset_class, weight in candidate.instrument_positions
        if asset_class.casefold() in {"cash", "money_market", "deposit"}
    )
    if cash_like_weight + 1e-12 < reserve_fraction:
        reasons.append("candidate does not preserve the emergency reserve in cash-like instruments")
    return reasons


def _evaluate_candidate(
    contract: InvestorContract,
    query: ReturnQuery,
    candidate: CandidatePolicy,
) -> CandidateEvaluation:
    goal = contract.goal(query.goal_id)
    distribution = candidate.distribution
    reasons = _instrument_reasons(contract, candidate)
    if candidate.diagnostic_only:
        reasons.append("diagnostic-only candidate is excluded from selection")
    if candidate.currency != contract.currency:
        reasons.append("candidate currency differs from the Investor Contract")
    if candidate.value_basis is not contract.reporting_basis or goal.value_basis is not contract.reporting_basis:
        reasons.append("candidate/goal reporting basis requires the W3 inflation/FX model")

    horizon = (goal.deadline - contract.as_of_date).days / 365.2425
    if abs(distribution.horizon_years - horizon) > 0.10:
        reasons.append("candidate path horizon differs from the goal horizon")

    metric = contract.capacity.drawdown_metric.value
    dd = distribution.drawdowns(metric)
    breach = dd > contract.capacity.max_drawdown
    capacity_probability = float(np.mean(~breach))
    capacity_excess = _conditional_mean(dd - contract.capacity.max_drawdown, breach)

    if distribution.minimum_wealth is None:
        # Conservative fallback: 1-MDD never overstates the initial-wealth-relative floor.
        min_wealth_fraction = 1.0 - distribution.full_horizon_max_drawdown
    else:
        min_wealth_fraction = distribution.minimum_wealth
    floor_breach = min_wealth_fraction * contract.initial_wealth < contract.capacity.wealth_floor
    wealth_floor_probability = float(np.mean(~floor_breach))
    wealth_floor_shortfall = _conditional_mean(
        contract.capacity.wealth_floor - min_wealth_fraction * contract.initial_wealth,
        floor_breach,
    )
    hard_capacity = (
        capacity_probability >= contract.capacity.confidence
        and wealth_floor_probability >= contract.capacity.confidence
    )
    if capacity_probability < contract.capacity.confidence:
        reasons.append("drawdown capacity confidence is not met")
    if wealth_floor_probability < contract.capacity.confidence:
        reasons.append("wealth-floor confidence is not met")

    required_cagr = contract.required_cagr(goal.goal_id)
    goal_miss = distribution.cagr < required_cagr
    goal_probability = float(np.mean(~goal_miss))
    goal_shortfall = _conditional_mean(required_cagr - distribution.cagr, goal_miss)
    goal_ok = goal_probability >= goal.success_probability
    if not goal_ok:
        reasons.append("goal success probability is not met")

    # P3 (binding CVaR): CVaR_beta of the shortfall below the goal CAGR. If the capacity object
    # carries a cvar_shortfall_budget it BINDS (a candidate whose tail shortfall exceeds the budget
    # is capacity-ineligible); otherwise it is computed and reported but does not gate selection.
    cvar_beta = float(getattr(contract.capacity, "cvar_beta", 0.10))
    shortfall_below_goal = np.maximum(required_cagr - distribution.cagr, 0.0)
    tail_n = max(1, int(np.ceil(cvar_beta * len(shortfall_below_goal))))
    cvar_shortfall_below_goal = float(np.sort(shortfall_below_goal)[-tail_n:].mean())
    cvar_budget = getattr(contract.capacity, "cvar_shortfall_budget", None)
    cvar_budget = None if cvar_budget is None else float(cvar_budget)
    cvar_budget_satisfied = cvar_budget is None or cvar_shortfall_below_goal <= cvar_budget
    if not cvar_budget_satisfied:
        reasons.append("CVaR shortfall budget is not met")

    query_probability = (
        query.probability
        if query.kind is QueryKind.RETURN_FLOOR_AT_PROBABILITY
        else goal.success_probability
    )
    # P3: keep the exact empirical-inverse quantile as return_floor (reproducible, inverse-consistent),
    # and ADD the lower-confidence bound r*(p,H,u) = LCB[Q_{1-p}] as the product-grade conservative
    # number that drives selection (max over candidates of the LCB).
    return_floor = conservative_return_floor(distribution, float(query_probability))
    floor_info = conservative_return_floor_lcb(distribution, float(query_probability))
    return_floor_lcb = floor_info["lcb"]
    return_floor_lcb_se = floor_info["se"]
    floor_miss = distribution.cagr < return_floor
    return_floor_shortfall = _conditional_mean(return_floor - distribution.cagr, floor_miss)
    target_probability: float | None = None
    target_shortfall: float | None = None
    if query.kind is QueryKind.PROBABILITY_AT_TARGET_RETURN:
        target_miss = distribution.cagr < float(query.target_cagr)
        target_probability = float(np.mean(~target_miss))
        target_shortfall = _conditional_mean(float(query.target_cagr) - distribution.cagr, target_miss)

    dd_p95 = float(np.quantile(dd, 0.95, method="inverted_cdf"))
    tolerance_penalty = contract.tolerance.penalty(min(dd_p95, 1.0))
    instrument_ok = not any(
        reason for reason in reasons
        if reason not in {
            "drawdown capacity confidence is not met",
            "wealth-floor confidence is not met",
            "goal success probability is not met",
        }
    )
    eligible = instrument_ok and hard_capacity and goal_ok and cvar_budget_satisfied
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        eligible=eligible,
        hard_capacity_satisfied=hard_capacity,
        goal_probability_satisfied=goal_ok,
        instrument_constraints_satisfied=instrument_ok,
        capacity_probability=capacity_probability,
        capacity_expected_excess_given_breach=capacity_excess,
        wealth_floor_probability=wealth_floor_probability,
        wealth_floor_expected_shortfall_given_breach=wealth_floor_shortfall,
        goal_probability=goal_probability,
        goal_expected_shortfall_given_miss=goal_shortfall,
        return_floor=return_floor,
        return_floor_probability=float(query_probability),
        return_expected_shortfall_below_floor=return_floor_shortfall,
        target_return_probability=target_probability,
        target_expected_shortfall_given_miss=target_shortfall,
        selected_drawdown_metric=metric,
        drawdown_p95=dd_p95,
        tolerance_penalty=tolerance_penalty,
        rejection_reasons=tuple(reasons),
        return_floor_lcb=return_floor_lcb,
        return_floor_lcb_se=return_floor_lcb_se,
        cvar_shortfall_below_goal=cvar_shortfall_below_goal,
        cvar_shortfall_budget=cvar_budget,
        cvar_budget_satisfied=cvar_budget_satisfied,
    )


def _assessment_dict(assessment: Any) -> dict[str, Any]:
    return {
        "feasible": assessment.feasible,
        "return_hurdle_satisfied": assessment.return_hurdle_satisfied,
        "goal_probability_satisfied": assessment.goal_probability_satisfied,
        "hard_capacity_satisfied": assessment.hard_capacity_satisfied,
        "reasons": list(assessment.reasons),
        "suggestions": [
            {"kind": item.kind.value, "rationale": item.rationale}
            for item in assessment.suggestions
        ],
        "required_risk": {
            "required_cagr": assessment.required_risk.required_cagr,
            "required_success_probability": assessment.required_risk.required_success_probability,
            "capacity_metric": assessment.required_risk.capacity_metric.value,
            "capacity_limit": assessment.required_risk.capacity_limit,
            "minimum_required_drawdown": assessment.required_risk.minimum_required_drawdown,
            "risk_mapping_status": assessment.required_risk.risk_mapping_status,
        },
    }


def solve_joint(
    contract: InvestorContract,
    query: ReturnQuery,
    candidates: Iterable[CandidatePolicy],
) -> JointSolution:
    """Evaluate both query forms on one shared candidate/path measure and apply hard constraints."""
    contract.validate_query(query)
    if contract.cash_flows:
        raise ContractValidationError(
            "dated cash flows require the W3 pathwise wealth/cash-flow simulator; W1 refuses CAGR-only "
            "feasibility for contribution or withdrawal contracts"
        )
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one candidate is required")
    evaluations = tuple(_evaluate_candidate(contract, query, item) for item in candidates)
    feasible = [item for item in evaluations if item.eligible]

    goal = contract.goal(query.goal_id)
    paired = tuple(zip(candidates, evaluations))
    capacity_feasible = [
        (candidate, item) for candidate, item in paired
        if item.hard_capacity_satisfied and item.instrument_constraints_satisfied
    ]
    # FeasibilityEvidence is evaluated at the goal's requested probability, independent of which inverse
    # query form the user asked for. This prevents a p80 display query from changing a p60 goal verdict.
    max_floor = max(
        (conservative_return_floor(candidate.distribution, goal.success_probability)
         for candidate, _ in capacity_feasible),
        default=-0.999,
    )
    max_goal_probability = max((item.goal_probability for _, item in capacity_feasible), default=0.0)
    max_capacity_probability = max(
        (item.capacity_probability for item in evaluations if item.instrument_constraints_satisfied),
        default=0.0,
    )
    goal_candidates = [
        (candidate, item) for candidate, item in paired
        if item.goal_probability_satisfied and item.instrument_constraints_satisfied
    ]
    minimum_drawdown = min(
        (
            float(np.quantile(
                candidate.distribution.drawdowns(contract.capacity.drawdown_metric.value),
                contract.capacity.confidence,
                method="inverted_cdf",
            ))
            for candidate, _ in goal_candidates
        ),
        default=None,
    )
    evidence = FeasibilityEvidence(
        goal_id=goal.goal_id,
        source_id="w1-joint-solver/uncalibrated-historical-scenarios",
        capacity_metric=contract.capacity.drawdown_metric,
        evaluated_probability=goal.success_probability,
        max_lower_quantile_cagr_within_capacity=max_floor,
        max_goal_probability_within_capacity=max_goal_probability,
        capacity_compliance_probability=max_capacity_probability,
        minimum_drawdown_needed_for_goal=minimum_drawdown,
    )
    assessment = assess_feasibility(contract, goal.goal_id, evidence)

    selected: CandidateEvaluation | None = None
    query_result: dict[str, Any]
    if feasible:
        if query.kind is QueryKind.RETURN_FLOOR_AT_PROBABILITY:
            # r*(p,H,u): select the candidate with the highest LOWER-CONFIDENCE-BOUND floor
            # (tolerance breaks ties), not the highest point estimate.
            selected = max(
                feasible,
                key=lambda item: (
                    item.return_floor_lcb - item.tolerance_penalty,
                    item.return_floor_lcb,
                    item.goal_probability,
                ),
            )
            query_result = {
                "probability": query.probability,
                "lower_cagr_floor": selected.return_floor,
                "lower_cagr_floor_lcb": selected.return_floor_lcb,
                "lower_cagr_floor_lcb_se": selected.return_floor_lcb_se,
                "cvar_shortfall_below_goal": selected.cvar_shortfall_below_goal,
                "expected_shortfall_below_floor": selected.return_expected_shortfall_below_floor,
                "preference_adjusted_score": selected.return_floor_lcb - selected.tolerance_penalty,
            }
        else:
            selected = max(
                feasible,
                key=lambda item: (
                    float(item.target_return_probability) - item.tolerance_penalty,
                    float(item.target_return_probability),
                    item.return_floor,
                ),
            )
            query_result = {
                "target_cagr": query.target_cagr,
                "probability_reach": selected.target_return_probability,
                "expected_shortfall_given_miss": selected.target_expected_shortfall_given_miss,
                "preference_adjusted_score": float(selected.target_return_probability) - selected.tolerance_penalty,
            }
    else:
        query_result = {
            "status": "INFEASIBLE_WITHIN_HARD_CAPACITY",
            "allowed_changes_only": [item.kind.value for item in assessment.suggestions],
        }

    return JointSolution(
        feasible=bool(feasible),
        contract_id=contract.contract_id,
        goal_id=goal.goal_id,
        query_kind=query.kind.value,
        selected_candidate_id=None if selected is None else selected.candidate_id,
        query_result=query_result,
        evaluations=evaluations,
        feasibility=_assessment_dict(assessment),
    )


__all__ = ["CandidateEvaluation", "CandidatePolicy", "JointSolution", "solve_joint"]
