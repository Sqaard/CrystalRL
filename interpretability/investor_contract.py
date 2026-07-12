"""Versioned Investor Contract interface for personalized-investing research.

This module implements the W1 contract boundary.  It is deliberately independent
from portfolio selection, PPO, and the wealth-distribution engine.  Objects and
feasibility results produced here are ``RESEARCH_ONLY_INTERFACE_PROTOTYPE`` until
the W8 certification gate is passed; they are neither forecasts nor investment
recommendations.

The important separation is structural:

* ``CapacityConstraint`` is a hard financial constraint.
* ``TolerancePreference`` is a preference inside the capacity-feasible set.
* required return is computed from wealth, dated cash flows, goal, and horizon.
* required *risk* is not inferred from required return.  It is only populated
  from an external scenario/frontier evidence object, which W3 will own.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "investor-contract.v1"
INTERFACE_STATUS = "RESEARCH_ONLY_INTERFACE_PROTOTYPE"
RISK_MAPPING_REQUIRED = "REQUIRES_W3_SCENARIO_ENGINE"
RISK_MAPPING_RESEARCH_ONLY = "RESEARCH_ONLY_SCENARIO_EVIDENCE"


class ContractValidationError(ValueError):
    """Raised when an Investor Contract is ambiguous or internally invalid."""


class ValueBasis(str, Enum):
    NOMINAL = "nominal"
    REAL = "real"


class DrawdownMetric(str, Enum):
    """The path scope to which a drawdown capacity limit applies."""

    ANNUAL_MAX_DRAWDOWN = "annual_max_drawdown"
    FULL_HORIZON_MAX_DRAWDOWN = "full_horizon_max_drawdown"


class CashFlowKind(str, Enum):
    CONTRIBUTION = "contribution"
    WITHDRAWAL = "withdrawal"


class QueryKind(str, Enum):
    RETURN_FLOOR_AT_PROBABILITY = "return_floor_at_probability"
    PROBABILITY_AT_TARGET_RETURN = "probability_at_target_return"


class SuggestionKind(str, Enum):
    EXTEND_HORIZON = "extend_horizon"
    INCREASE_CONTRIBUTIONS = "increase_contributions"
    REDUCE_GOAL = "reduce_goal"
    REDUCE_SUCCESS_PROBABILITY = "reduce_success_probability"


ALLOWED_FEASIBILITY_SUGGESTIONS = frozenset(SuggestionKind)


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite")
    return result


def _positive(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    if result <= 0.0:
        raise ContractValidationError(f"{field_name} must be > 0")
    return result


def _nonnegative(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    if result < 0.0:
        raise ContractValidationError(f"{field_name} must be >= 0")
    return result


def _closed_fraction(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ContractValidationError(f"{field_name} must be in [0, 1]")
    return result


def _probability(value: Any, field_name: str) -> float:
    result = _number(value, field_name)
    if not 0.0 < result <= 1.0:
        raise ContractValidationError(f"{field_name} must be in (0, 1]")
    return result


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _enum(enum_type: type[Enum], value: Any, field_name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} must be an ISO date") from exc


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return value


def _assert_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field_name: str,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise ContractValidationError(
            f"{field_name} is missing fields: {', '.join(sorted(missing))}"
        )
    if extra:
        raise ContractValidationError(
            f"{field_name} has unknown fields: {', '.join(sorted(extra))}"
        )


@dataclass(frozen=True)
class Goal:
    goal_id: str
    target_amount: float
    deadline: date
    value_basis: ValueBasis
    priority: int
    success_probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "target_amount", _positive(self.target_amount, "target_amount")
        )
        object.__setattr__(self, "deadline", _date(self.deadline, "deadline"))
        object.__setattr__(
            self,
            "value_basis",
            _enum(ValueBasis, self.value_basis, "value_basis"),
        )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ContractValidationError("priority must be a positive integer")
        if self.priority < 1:
            raise ContractValidationError("priority must be a positive integer")
        object.__setattr__(
            self,
            "success_probability",
            _probability(self.success_probability, "success_probability"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "target_amount": self.target_amount,
            "deadline": self.deadline.isoformat(),
            "value_basis": self.value_basis.value,
            "priority": self.priority,
            "success_probability": self.success_probability,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Goal":
        raw = _mapping(raw, "goal")
        _assert_keys(
            raw,
            required={
                "goal_id",
                "target_amount",
                "deadline",
                "value_basis",
                "priority",
                "success_probability",
            },
            field_name="goal",
        )
        return cls(**raw)


@dataclass(frozen=True)
class ScheduledCashFlow:
    flow_id: str
    date: date
    amount: float
    kind: CashFlowKind
    value_basis: ValueBasis
    goal_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "flow_id", _identifier(self.flow_id, "flow_id"))
        object.__setattr__(self, "date", _date(self.date, "cash_flow.date"))
        object.__setattr__(self, "amount", _positive(self.amount, "cash_flow.amount"))
        object.__setattr__(
            self, "kind", _enum(CashFlowKind, self.kind, "cash_flow.kind")
        )
        object.__setattr__(
            self,
            "value_basis",
            _enum(ValueBasis, self.value_basis, "cash_flow.value_basis"),
        )
        if self.goal_id is not None:
            object.__setattr__(
                self, "goal_id", _identifier(self.goal_id, "cash_flow.goal_id")
            )

    @property
    def signed_amount(self) -> float:
        if self.kind is CashFlowKind.CONTRIBUTION:
            return self.amount
        return -self.amount

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "date": self.date.isoformat(),
            "amount": self.amount,
            "kind": self.kind.value,
            "value_basis": self.value_basis.value,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScheduledCashFlow":
        raw = _mapping(raw, "cash_flow")
        _assert_keys(
            raw,
            required={"flow_id", "date", "amount", "kind", "value_basis"},
            optional={"goal_id"},
            field_name="cash_flow",
        )
        return cls(**raw)


@dataclass(frozen=True)
class CapacityConstraint:
    max_drawdown: float
    drawdown_metric: DrawdownMetric
    confidence: float
    wealth_floor: float
    # P3: optional binding tail constraint. When cvar_shortfall_budget is set, a candidate whose
    # CVaR_beta of the shortfall below the goal CAGR exceeds it is capacity-INELIGIBLE (not merely
    # reported). None => the CVaR is computed and reported but does not gate selection.
    cvar_shortfall_budget: float | None = None
    cvar_beta: float = 0.10

    def __post_init__(self) -> None:
        max_drawdown = _closed_fraction(self.max_drawdown, "capacity.max_drawdown")
        if max_drawdown >= 1.0:
            raise ContractValidationError("capacity.max_drawdown must be < 1")
        object.__setattr__(self, "max_drawdown", max_drawdown)
        object.__setattr__(
            self,
            "drawdown_metric",
            _enum(
                DrawdownMetric,
                self.drawdown_metric,
                "capacity.drawdown_metric",
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "capacity.confidence"),
        )
        object.__setattr__(
            self,
            "wealth_floor",
            _nonnegative(self.wealth_floor, "capacity.wealth_floor"),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "max_drawdown": self.max_drawdown,
            "drawdown_metric": self.drawdown_metric.value,
            "confidence": self.confidence,
            "wealth_floor": self.wealth_floor,
        }
        if self.cvar_shortfall_budget is not None:
            out["cvar_shortfall_budget"] = self.cvar_shortfall_budget
            out["cvar_beta"] = self.cvar_beta
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CapacityConstraint":
        raw = _mapping(raw, "capacity")
        _assert_keys(
            raw,
            required={
                "max_drawdown",
                "drawdown_metric",
                "confidence",
                "wealth_floor",
            },
            optional={"cvar_shortfall_budget", "cvar_beta"},
            field_name="capacity",
        )
        return cls(**raw)


@dataclass(frozen=True)
class TolerancePreference:
    """Psychological preference; never used as a hard feasibility limit."""

    comfortable_drawdown: float
    discomfort_weight: float = 1.0

    def __post_init__(self) -> None:
        comfortable = _closed_fraction(
            self.comfortable_drawdown, "tolerance.comfortable_drawdown"
        )
        if comfortable >= 1.0:
            raise ContractValidationError("tolerance.comfortable_drawdown must be < 1")
        object.__setattr__(self, "comfortable_drawdown", comfortable)
        object.__setattr__(
            self,
            "discomfort_weight",
            _nonnegative(self.discomfort_weight, "tolerance.discomfort_weight"),
        )

    def penalty(self, candidate_drawdown: float) -> float:
        """Preference-only penalty for ranking capacity-feasible candidates."""

        drawdown = _closed_fraction(candidate_drawdown, "candidate_drawdown")
        return self.discomfort_weight * max(
            0.0, drawdown - self.comfortable_drawdown
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "comfortable_drawdown": self.comfortable_drawdown,
            "discomfort_weight": self.discomfort_weight,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TolerancePreference":
        raw = _mapping(raw, "tolerance")
        _assert_keys(
            raw,
            required={"comfortable_drawdown"},
            optional={"discomfort_weight"},
            field_name="tolerance",
        )
        return cls(**raw)


@dataclass(frozen=True)
class LiquidityConstraint:
    emergency_reserve: float
    min_liquid_fraction: float
    max_lockup_days: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "emergency_reserve",
            _nonnegative(self.emergency_reserve, "liquidity.emergency_reserve"),
        )
        object.__setattr__(
            self,
            "min_liquid_fraction",
            _closed_fraction(
                self.min_liquid_fraction, "liquidity.min_liquid_fraction"
            ),
        )
        if isinstance(self.max_lockup_days, bool) or not isinstance(
            self.max_lockup_days, int
        ):
            raise ContractValidationError(
                "liquidity.max_lockup_days must be a non-negative integer"
            )
        if self.max_lockup_days < 0:
            raise ContractValidationError(
                "liquidity.max_lockup_days must be a non-negative integer"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "emergency_reserve": self.emergency_reserve,
            "min_liquid_fraction": self.min_liquid_fraction,
            "max_lockup_days": self.max_lockup_days,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LiquidityConstraint":
        raw = _mapping(raw, "liquidity")
        _assert_keys(
            raw,
            required={
                "emergency_reserve",
                "min_liquid_fraction",
                "max_lockup_days",
            },
            field_name="liquidity",
        )
        return cls(**raw)


@dataclass(frozen=True)
class InstrumentRestrictions:
    allowed_asset_classes: tuple[str, ...]
    prohibited_instruments: tuple[str, ...] = ()
    max_single_instrument_weight: float = 1.0
    allow_shorting: bool = False
    allow_leverage: bool = False
    max_gross_leverage: float = 1.0

    def __post_init__(self) -> None:
        allowed = tuple(
            _identifier(item, "instruments.allowed_asset_classes item")
            for item in self.allowed_asset_classes
        )
        prohibited = tuple(
            _identifier(item, "instruments.prohibited_instruments item")
            for item in self.prohibited_instruments
        )
        if not allowed:
            raise ContractValidationError(
                "instruments.allowed_asset_classes must not be empty"
            )
        if len({item.casefold() for item in allowed}) != len(allowed):
            raise ContractValidationError(
                "instruments.allowed_asset_classes contains duplicates"
            )
        if len({item.casefold() for item in prohibited}) != len(prohibited):
            raise ContractValidationError(
                "instruments.prohibited_instruments contains duplicates"
            )
        object.__setattr__(self, "allowed_asset_classes", allowed)
        object.__setattr__(self, "prohibited_instruments", prohibited)
        object.__setattr__(
            self,
            "max_single_instrument_weight",
            _closed_fraction(
                self.max_single_instrument_weight,
                "instruments.max_single_instrument_weight",
            ),
        )
        if not isinstance(self.allow_shorting, bool):
            raise ContractValidationError("instruments.allow_shorting must be boolean")
        if not isinstance(self.allow_leverage, bool):
            raise ContractValidationError("instruments.allow_leverage must be boolean")
        max_leverage = _positive(
            self.max_gross_leverage, "instruments.max_gross_leverage"
        )
        if not self.allow_leverage and max_leverage > 1.0:
            raise ContractValidationError(
                "max_gross_leverage cannot exceed 1 when leverage is disabled"
            )
        object.__setattr__(self, "max_gross_leverage", max_leverage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_asset_classes": list(self.allowed_asset_classes),
            "prohibited_instruments": list(self.prohibited_instruments),
            "max_single_instrument_weight": self.max_single_instrument_weight,
            "allow_shorting": self.allow_shorting,
            "allow_leverage": self.allow_leverage,
            "max_gross_leverage": self.max_gross_leverage,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InstrumentRestrictions":
        raw = _mapping(raw, "instruments")
        _assert_keys(
            raw,
            required={"allowed_asset_classes"},
            optional={
                "prohibited_instruments",
                "max_single_instrument_weight",
                "allow_shorting",
                "allow_leverage",
                "max_gross_leverage",
            },
            field_name="instruments",
        )
        return cls(**raw)


@dataclass(frozen=True)
class ReturnQuery:
    """One of the two inverse-compatible W1 query forms."""

    goal_id: str
    kind: QueryKind
    probability: float | None = None
    target_cagr: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "query.goal_id"))
        object.__setattr__(self, "kind", _enum(QueryKind, self.kind, "query.kind"))
        if self.kind is QueryKind.RETURN_FLOOR_AT_PROBABILITY:
            if self.probability is None or self.target_cagr is not None:
                raise ContractValidationError(
                    "return-floor query requires probability and forbids target_cagr"
                )
            object.__setattr__(
                self,
                "probability",
                _probability(self.probability, "query.probability"),
            )
        else:
            if self.target_cagr is None or self.probability is not None:
                raise ContractValidationError(
                    "target-return query requires target_cagr and forbids probability"
                )
            object.__setattr__(
                self,
                "target_cagr",
                _number(self.target_cagr, "query.target_cagr"),
            )
            if self.target_cagr <= -1.0:
                raise ContractValidationError("query.target_cagr must be > -1")


@dataclass(frozen=True)
class FeasibilityEvidence:
    """Research-only frontier/scenario summary supplied by a future W3 engine."""

    goal_id: str
    source_id: str
    capacity_metric: DrawdownMetric
    evaluated_probability: float
    max_lower_quantile_cagr_within_capacity: float
    max_goal_probability_within_capacity: float
    capacity_compliance_probability: float
    minimum_drawdown_needed_for_goal: float | None = None
    evidence_status: str = "UNCALIBRATED_RESEARCH_ESTIMATE"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "evidence.goal_id"))
        object.__setattr__(
            self, "source_id", _identifier(self.source_id, "evidence.source_id")
        )
        object.__setattr__(
            self,
            "capacity_metric",
            _enum(
                DrawdownMetric,
                self.capacity_metric,
                "evidence.capacity_metric",
            ),
        )
        object.__setattr__(
            self,
            "evaluated_probability",
            _probability(
                self.evaluated_probability, "evidence.evaluated_probability"
            ),
        )
        object.__setattr__(
            self,
            "max_lower_quantile_cagr_within_capacity",
            _number(
                self.max_lower_quantile_cagr_within_capacity,
                "evidence.max_lower_quantile_cagr_within_capacity",
            ),
        )
        if self.max_lower_quantile_cagr_within_capacity <= -1.0:
            raise ContractValidationError(
                "evidence.max_lower_quantile_cagr_within_capacity must be > -1"
            )
        object.__setattr__(
            self,
            "max_goal_probability_within_capacity",
            _closed_fraction(
                self.max_goal_probability_within_capacity,
                "evidence.max_goal_probability_within_capacity",
            ),
        )
        object.__setattr__(
            self,
            "capacity_compliance_probability",
            _closed_fraction(
                self.capacity_compliance_probability,
                "evidence.capacity_compliance_probability",
            ),
        )
        if self.minimum_drawdown_needed_for_goal is not None:
            minimum_drawdown = _closed_fraction(
                self.minimum_drawdown_needed_for_goal,
                "evidence.minimum_drawdown_needed_for_goal",
            )
            if minimum_drawdown >= 1.0:
                raise ContractValidationError(
                    "evidence.minimum_drawdown_needed_for_goal must be < 1"
                )
            object.__setattr__(
                self, "minimum_drawdown_needed_for_goal", minimum_drawdown
            )
        if self.evidence_status != "UNCALIBRATED_RESEARCH_ESTIMATE":
            raise ContractValidationError(
                "W1 accepts only UNCALIBRATED_RESEARCH_ESTIMATE evidence before W8"
            )


@dataclass(frozen=True)
class RequiredRiskEstimate:
    goal_id: str
    required_cagr: float
    required_success_probability: float
    capacity_metric: DrawdownMetric
    capacity_limit: float
    minimum_required_drawdown: float | None
    risk_mapping_status: str


@dataclass(frozen=True)
class FeasibilitySuggestion:
    kind: SuggestionKind
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(SuggestionKind, self.kind, "kind"))
        object.__setattr__(
            self, "rationale", _identifier(self.rationale, "rationale")
        )


@dataclass(frozen=True)
class FeasibilityAssessment:
    feasible: bool
    goal_id: str
    required_risk: RequiredRiskEstimate
    return_hurdle_satisfied: bool
    goal_probability_satisfied: bool
    hard_capacity_satisfied: bool
    reasons: tuple[str, ...]
    suggestions: tuple[FeasibilitySuggestion, ...]
    interface_status: str = INTERFACE_STATUS


@dataclass(frozen=True)
class InvestorContract:
    contract_id: str
    as_of_date: date
    initial_wealth: float
    currency: str
    reporting_basis: ValueBasis
    goals: tuple[Goal, ...]
    cash_flows: tuple[ScheduledCashFlow, ...]
    capacity: CapacityConstraint
    tolerance: TolerancePreference
    liquidity: LiquidityConstraint
    instruments: InstrumentRestrictions
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        object.__setattr__(
            self, "contract_id", _identifier(self.contract_id, "contract_id")
        )
        object.__setattr__(self, "as_of_date", _date(self.as_of_date, "as_of_date"))
        object.__setattr__(
            self, "initial_wealth", _positive(self.initial_wealth, "initial_wealth")
        )
        if not isinstance(self.currency, str) or not re.fullmatch(
            r"[A-Z]{3}", self.currency
        ):
            raise ContractValidationError(
                "currency must be an uppercase three-letter ISO-4217-style code"
            )
        object.__setattr__(
            self,
            "reporting_basis",
            _enum(ValueBasis, self.reporting_basis, "reporting_basis"),
        )
        goals = tuple(self.goals)
        cash_flows = tuple(self.cash_flows)
        if not goals:
            raise ContractValidationError("goals must not be empty")
        if not all(isinstance(item, Goal) for item in goals):
            raise ContractValidationError("goals must contain Goal objects")
        if not all(isinstance(item, ScheduledCashFlow) for item in cash_flows):
            raise ContractValidationError(
                "cash_flows must contain ScheduledCashFlow objects"
            )
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "cash_flows", cash_flows)
        goal_ids = [goal.goal_id for goal in goals]
        if len(set(goal_ids)) != len(goal_ids):
            raise ContractValidationError("goal_id values must be unique")
        flow_ids = [flow.flow_id for flow in cash_flows]
        if len(set(flow_ids)) != len(flow_ids):
            raise ContractValidationError("flow_id values must be unique")
        goal_map = {goal.goal_id: goal for goal in goals}
        for goal in goals:
            if goal.deadline <= self.as_of_date:
                raise ContractValidationError(
                    f"goal {goal.goal_id!r} deadline must be after as_of_date"
                )
        latest_deadline = max(goal.deadline for goal in goals)
        for flow in cash_flows:
            if flow.date < self.as_of_date:
                raise ContractValidationError(
                    f"cash flow {flow.flow_id!r} predates as_of_date"
                )
            if flow.date > latest_deadline:
                raise ContractValidationError(
                    f"cash flow {flow.flow_id!r} occurs after every goal deadline"
                )
            if flow.goal_id is not None:
                if flow.goal_id not in goal_map:
                    raise ContractValidationError(
                        f"cash flow {flow.flow_id!r} references unknown goal_id"
                    )
                goal = goal_map[flow.goal_id]
                if flow.date > goal.deadline:
                    raise ContractValidationError(
                        f"cash flow {flow.flow_id!r} occurs after its goal deadline"
                    )
                if flow.value_basis is not goal.value_basis:
                    raise ContractValidationError(
                        f"cash flow {flow.flow_id!r} basis differs from its goal"
                    )
        if not isinstance(self.capacity, CapacityConstraint):
            raise ContractValidationError("capacity must be a CapacityConstraint")
        if not isinstance(self.tolerance, TolerancePreference):
            raise ContractValidationError("tolerance must be a TolerancePreference")
        if not isinstance(self.liquidity, LiquidityConstraint):
            raise ContractValidationError("liquidity must be a LiquidityConstraint")
        if not isinstance(self.instruments, InstrumentRestrictions):
            raise ContractValidationError(
                "instruments must be an InstrumentRestrictions object"
            )
        if self.capacity.wealth_floor > self.initial_wealth:
            raise ContractValidationError(
                "capacity.wealth_floor cannot exceed initial_wealth"
            )
        if self.liquidity.emergency_reserve > self.initial_wealth:
            raise ContractValidationError(
                "liquidity.emergency_reserve cannot exceed initial_wealth"
            )

    @property
    def interface_status(self) -> str:
        return INTERFACE_STATUS

    def goal(self, goal_id: str) -> Goal:
        goal_id = _identifier(goal_id, "goal_id")
        for goal in self.goals:
            if goal.goal_id == goal_id:
                return goal
        raise ContractValidationError(f"unknown goal_id {goal_id!r}")

    def _goal_cash_flows(self, goal: Goal) -> tuple[ScheduledCashFlow, ...]:
        relevant = tuple(
            flow
            for flow in self.cash_flows
            if flow.goal_id is None or flow.goal_id == goal.goal_id
        )
        mismatched = [
            flow.flow_id for flow in relevant if flow.value_basis is not goal.value_basis
        ]
        if mismatched:
            raise ContractValidationError(
                "cannot compute required CAGR across nominal/real bases without "
                "the W3 inflation model; mismatched flows: " + ", ".join(mismatched)
            )
        return relevant

    def required_cagr(self, goal_id: str) -> float:
        """Compute the constant annual return hurdle for one goal.

        Dated contributions are positive and withdrawals negative.  This is a
        deterministic funding hurdle, not a forecast and not a risk estimate.
        If the dated cash-flow equation has no unique monotone root, the method
        rejects it instead of selecting an arbitrary IRR.
        """

        goal = self.goal(goal_id)
        flows = self._goal_cash_flows(goal)
        horizon = (goal.deadline - self.as_of_date).days / 365.2425
        if horizon <= 0.0:  # also guarded in __post_init__
            raise ContractValidationError("goal horizon must be positive")

        def terminal_wealth(rate: float) -> float:
            if rate <= -1.0:
                return -math.inf
            base = 1.0 + rate
            wealth = self.initial_wealth * base**horizon
            for flow in flows:
                remaining = (goal.deadline - flow.date).days / 365.2425
                wealth += flow.signed_amount * base**remaining
            return wealth

        def gap(rate: float) -> float:
            return terminal_wealth(rate) - goal.target_amount

        lower = -0.999999
        upper = 1.0
        while gap(upper) < 0.0 and upper < 1023.0:
            upper = 2.0 * upper + 1.0
        if gap(upper) < 0.0:
            raise ContractValidationError(
                "required CAGR exceeds the numerical research-interface bound"
            )

        # Multiple IRRs are possible with sign-changing cash-flow schedules.
        # Ordinary withdrawals can make terminal wealth locally non-monotone near
        # -100% without creating another funding root, so test root uniqueness
        # rather than requiring the entire curve to be monotone.
        grid = [lower + (upper - lower) * i / 2048.0 for i in range(2049)]
        values = [gap(rate) for rate in grid]
        crossings = [
            index
            for index, (left, right) in enumerate(zip(values, values[1:]))
            if (left < 0.0 <= right) or (left > 0.0 >= right)
        ]
        if len(crossings) != 1:
            raise ContractValidationError(
                "cash-flow schedule does not have a unique required CAGR"
            )
        crossing = crossings[0]
        if values[crossing] > values[crossing + 1]:
            raise ContractValidationError(
                "cash-flow schedule crosses the target in the wrong direction"
            )
        lo, hi = grid[crossing], grid[crossing + 1]
        for _ in range(160):
            mid = (lo + hi) / 2.0
            if gap(mid) >= 0.0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    def required_risk(
        self,
        goal_id: str,
        evidence: FeasibilityEvidence | None = None,
    ) -> RequiredRiskEstimate:
        goal = self.goal(goal_id)
        minimum_drawdown: float | None = None
        status = RISK_MAPPING_REQUIRED
        if evidence is not None:
            if evidence.goal_id != goal.goal_id:
                raise ContractValidationError(
                    "feasibility evidence goal_id does not match requested goal"
                )
            if evidence.capacity_metric is not self.capacity.drawdown_metric:
                raise ContractValidationError(
                    "evidence and capacity use different drawdown metrics"
                )
            minimum_drawdown = evidence.minimum_drawdown_needed_for_goal
            status = RISK_MAPPING_RESEARCH_ONLY
        return RequiredRiskEstimate(
            goal_id=goal.goal_id,
            required_cagr=self.required_cagr(goal.goal_id),
            required_success_probability=goal.success_probability,
            capacity_metric=self.capacity.drawdown_metric,
            capacity_limit=self.capacity.max_drawdown,
            minimum_required_drawdown=minimum_drawdown,
            risk_mapping_status=status,
        )

    def validate_query(self, query: ReturnQuery) -> None:
        if not isinstance(query, ReturnQuery):
            raise ContractValidationError("query must be a ReturnQuery")
        self.goal(query.goal_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "as_of_date": self.as_of_date.isoformat(),
            "initial_wealth": self.initial_wealth,
            "currency": self.currency,
            "reporting_basis": self.reporting_basis.value,
            "goals": [goal.to_dict() for goal in self.goals],
            "cash_flows": [flow.to_dict() for flow in self.cash_flows],
            "capacity": self.capacity.to_dict(),
            "tolerance": self.tolerance.to_dict(),
            "liquidity": self.liquidity.to_dict(),
            "instruments": self.instruments.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InvestorContract":
        raw = _mapping(raw, "investor_contract")
        _assert_keys(
            raw,
            required={
                "schema_version",
                "contract_id",
                "as_of_date",
                "initial_wealth",
                "currency",
                "reporting_basis",
                "goals",
                "cash_flows",
                "capacity",
                "tolerance",
                "liquidity",
                "instruments",
            },
            field_name="investor_contract",
        )
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {raw['schema_version']!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        goals_raw = raw["goals"]
        flows_raw = raw["cash_flows"]
        if not isinstance(goals_raw, (list, tuple)):
            raise ContractValidationError("goals must be an array")
        if not isinstance(flows_raw, (list, tuple)):
            raise ContractValidationError("cash_flows must be an array")
        return cls(
            schema_version=raw["schema_version"],
            contract_id=raw["contract_id"],
            as_of_date=raw["as_of_date"],
            initial_wealth=raw["initial_wealth"],
            currency=raw["currency"],
            reporting_basis=raw["reporting_basis"],
            goals=tuple(Goal.from_dict(item) for item in goals_raw),
            cash_flows=tuple(ScheduledCashFlow.from_dict(item) for item in flows_raw),
            capacity=CapacityConstraint.from_dict(raw["capacity"]),
            tolerance=TolerancePreference.from_dict(raw["tolerance"]),
            liquidity=LiquidityConstraint.from_dict(raw["liquidity"]),
            instruments=InstrumentRestrictions.from_dict(raw["instruments"]),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "InvestorContract":
        if not isinstance(raw, str):
            raise ContractValidationError("JSON contract must be a string")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("invalid Investor Contract JSON") from exc
        return cls.from_dict(decoded)


def assess_feasibility(
    contract: InvestorContract,
    goal_id: str,
    evidence: FeasibilityEvidence,
) -> FeasibilityAssessment:
    """Apply a hard-capacity feasibility check without using tolerance as a cap.

    The supplied evidence remains research-only.  For an infeasible goal, this
    function can only suggest changing horizon, contributions, goal, or requested
    probability; it has no suggestion type capable of raising capacity.
    """

    if not isinstance(contract, InvestorContract):
        raise ContractValidationError("contract must be an InvestorContract")
    if not isinstance(evidence, FeasibilityEvidence):
        raise ContractValidationError("evidence must be FeasibilityEvidence")
    goal = contract.goal(goal_id)
    if evidence.goal_id != goal.goal_id:
        raise ContractValidationError("evidence goal_id does not match goal_id")
    if evidence.capacity_metric is not contract.capacity.drawdown_metric:
        raise ContractValidationError(
            "evidence and capacity use different drawdown metrics"
        )
    if not math.isclose(
        evidence.evaluated_probability,
        goal.success_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractValidationError(
            "evidence must be evaluated at the goal success_probability"
        )

    required = contract.required_risk(goal.goal_id, evidence)
    return_ok = (
        evidence.max_lower_quantile_cagr_within_capacity
        >= required.required_cagr
    )
    probability_ok = (
        evidence.max_goal_probability_within_capacity
        >= goal.success_probability
    )
    drawdown_ok = (
        evidence.minimum_drawdown_needed_for_goal is None
        or evidence.minimum_drawdown_needed_for_goal
        <= contract.capacity.max_drawdown
    )
    compliance_ok = (
        evidence.capacity_compliance_probability >= contract.capacity.confidence
    )
    hard_capacity_ok = drawdown_ok and compliance_ok
    feasible = return_ok and probability_ok and hard_capacity_ok

    reasons: list[str] = []
    if not return_ok:
        reasons.append("required return hurdle is outside the capacity-feasible frontier")
    if not probability_ok:
        reasons.append("requested goal probability is not attainable within capacity")
    if not drawdown_ok:
        reasons.append("minimum estimated drawdown for the goal exceeds hard capacity")
    if not compliance_ok:
        reasons.append("capacity compliance probability is below the hard confidence")

    suggestions: tuple[FeasibilitySuggestion, ...] = ()
    if not feasible:
        suggestions = (
            FeasibilitySuggestion(
                SuggestionKind.EXTEND_HORIZON,
                "A longer deadline may lower the required annual return hurdle.",
            ),
            FeasibilitySuggestion(
                SuggestionKind.INCREASE_CONTRIBUTIONS,
                "Additional scheduled contributions may lower the required return hurdle.",
            ),
            FeasibilitySuggestion(
                SuggestionKind.REDUCE_GOAL,
                "A smaller target may become attainable without relaxing capacity.",
            ),
            FeasibilitySuggestion(
                SuggestionKind.REDUCE_SUCCESS_PROBABILITY,
                "A lower requested probability may be feasible, subject to user confirmation.",
            ),
        )
    if any(item.kind not in ALLOWED_FEASIBILITY_SUGGESTIONS for item in suggestions):
        raise AssertionError("forbidden feasibility suggestion escaped the W1 contract")

    return FeasibilityAssessment(
        feasible=feasible,
        goal_id=goal.goal_id,
        required_risk=required,
        return_hurdle_satisfied=return_ok,
        goal_probability_satisfied=probability_ok,
        hard_capacity_satisfied=hard_capacity_ok,
        reasons=tuple(reasons),
        suggestions=suggestions,
    )


__all__ = [
    "ALLOWED_FEASIBILITY_SUGGESTIONS",
    "CashFlowKind",
    "CapacityConstraint",
    "ContractValidationError",
    "DrawdownMetric",
    "FeasibilityAssessment",
    "FeasibilityEvidence",
    "FeasibilitySuggestion",
    "Goal",
    "InstrumentRestrictions",
    "INTERFACE_STATUS",
    "InvestorContract",
    "LiquidityConstraint",
    "QueryKind",
    "RequiredRiskEstimate",
    "ReturnQuery",
    "RISK_MAPPING_REQUIRED",
    "RISK_MAPPING_RESEARCH_ONLY",
    "SCHEMA_VERSION",
    "ScheduledCashFlow",
    "SuggestionKind",
    "TolerancePreference",
    "ValueBasis",
    "assess_feasibility",
]
