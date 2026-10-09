"""The evaluation cases and the result each one is expected to reach.

eval/cases.yaml is written once by casefile.eval.select_cases, before the
agents exist, and is then treated as frozen. This module defines its shape and
loads it.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal, Self, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from casefile.config import PROJECT_ROOT
from casefile.control.state import EscalationReason
from casefile.rules import DecisionReason

CASES_PATH = PROJECT_ROOT / "eval" / "cases.yaml"

CaseGroup = Literal["clean", "natural", "document_defect", "system_stress"]
Outcome = Literal["recommend_pay", "recommend_deny", "refer_to_adjuster", "escalated"]

# A defect is planted in the generated documents of a case.
DefectType = Literal[
    "amount_mismatch",
    "date_mismatch",
    "missing_document",
    "wrong_policy_number",
    "incident_outside_coverage",
]
# A fault is injected into the running system, not into the documents.
InjectionType = Literal[
    "extractor_error_first_attempt",
    "reviewer_never_approves",
    "repeated_schema_errors",
    "budget_too_small",
    "step_limit_too_small",
]

_ESCALATION_REASONS = frozenset(get_args(EscalationReason))


class ExpectedResult(BaseModel):
    """The state a case must end in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome
    reason: DecisionReason | EscalationReason
    payable_amount: Decimal | None = None
    # True when the reviewer must return the work at least once on the way.
    reviewer_send_back: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (self.outcome == "recommend_pay") != (self.payable_amount is not None):
            raise ValueError("payable_amount is set exactly when the outcome is pay")
        if (self.outcome == "escalated") != (self.reason in _ESCALATION_REASONS):
            raise ValueError(
                f"reason {self.reason!r} does not fit outcome {self.outcome!r}"
            )
        return self


class EvalCase(BaseModel):
    """One evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(pattern=r"^C\d{2}$")
    group: CaseGroup
    claim_id: str
    policy_number: int
    document_defect: DefectType | None = None
    fault_injection: InjectionType | None = None
    note: str
    expected: ExpectedResult

    @model_validator(mode="after")
    def _check_group(self) -> Self:
        if (self.group == "document_defect") != (self.document_defect is not None):
            raise ValueError("only document_defect cases carry a document defect")
        if (self.group == "system_stress") != (self.fault_injection is not None):
            raise ValueError("only system_stress cases carry a fault injection")
        return self


class RuleThresholds(BaseModel):
    """The rule parameters the expected results were computed with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frequent_claims_count: int
    frequent_claims_window_months: int


class EvalSuite(BaseModel):
    """All evaluation cases, as stored in eval/cases.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    seed: int
    rules: RuleThresholds
    cases: tuple[EvalCase, ...]

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        for field_name in ("case_id", "claim_id"):
            values = [getattr(case, field_name) for case in self.cases]
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values are not unique")
        return self


def load_cases(path: Path = CASES_PATH) -> EvalSuite:
    """Read and validate the evaluation cases."""
    with path.open(encoding="utf-8") as handle:
        return EvalSuite.model_validate(yaml.safe_load(handle))
