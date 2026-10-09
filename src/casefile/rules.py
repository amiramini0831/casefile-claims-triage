"""Deterministic triage rules.

Given the facts of a claim, ``decide`` says what should happen to it. The
rules are checked in a fixed order and the first one that applies wins, so
every decision has exactly one reason.

The rules are deliberately simple and hypothetical. They are not taken from
any insurer's guidelines; they exist so that the expected result of an
evaluation case can be computed instead of being typed in by hand.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from casefile.data.records import (
    ClaimRecord,
    PolicyRecord,
    PriorClaimRecord,
    RepairCostReferenceRecord,
)
from casefile.data.transform import subtract_months

# A policy with this many earlier claims inside the window goes to a person.
FREQUENT_CLAIMS_COUNT = 3
FREQUENT_CLAIMS_WINDOW_MONTHS = 36

RuleOutcome = Literal["recommend_pay", "recommend_deny", "refer_to_adjuster"]
DecisionReason = Literal[
    "policy_not_found",
    "incident_outside_policy_period",
    "missing_document",
    "document_mismatch",
    "amount_above_reference",
    "frequent_prior_claims",
    "below_deductible",
    "covered",
]
DocumentIssue = Literal["missing_document", "amount_mismatch", "date_mismatch"]


class Decision(BaseModel):
    """What should happen to a claim, and the single rule that decided it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: RuleOutcome
    reason: DecisionReason
    payable_amount: Decimal | None = None

    @model_validator(mode="after")
    def _amount_only_when_paying(self) -> Self:
        if (self.outcome == "recommend_pay") != (self.payable_amount is not None):
            raise ValueError("payable_amount is set exactly when the outcome is pay")
        return self


def count_recent_prior_claims(
    claim: ClaimRecord, prior_claims: Iterable[PriorClaimRecord]
) -> int:
    """Count the policy's earlier claims inside the look-back window."""
    window_start = subtract_months(claim.incident_date, FREQUENT_CLAIMS_WINDOW_MONTHS)
    return sum(
        prior.policy_number == claim.policy_number
        and window_start <= prior.incident_date < claim.incident_date
        for prior in prior_claims
    )


def decide(
    *,
    claim: ClaimRecord,
    policy: PolicyRecord | None,
    prior_claims: Iterable[PriorClaimRecord] = (),
    reference: RepairCostReferenceRecord | None = None,
    document_issues: Collection[DocumentIssue] = (),
) -> Decision:
    """Apply the triage rules to one claim.

    ``policy`` is ``None`` when the policy number on the documents matches no
    policy. ``reference`` is ``None`` when there is no repair cost range for
    this kind of claim, in which case the amount is not judged.
    """
    if policy is None:
        return _refer("policy_not_found")
    if claim.incident_date < policy.policy_start_date:
        return Decision(
            outcome="recommend_deny", reason="incident_outside_policy_period"
        )
    if "missing_document" in document_issues:
        return _refer("missing_document")
    if document_issues:
        return _refer("document_mismatch")
    if reference is not None and claim.vehicle_claim > reference.high_amount:
        return _refer("amount_above_reference")
    if count_recent_prior_claims(claim, prior_claims) >= FREQUENT_CLAIMS_COUNT:
        return _refer("frequent_prior_claims")
    if claim.total_claim_amount <= policy.policy_deductible:
        return Decision(outcome="recommend_deny", reason="below_deductible")
    return Decision(
        outcome="recommend_pay",
        reason="covered",
        payable_amount=claim.total_claim_amount - policy.policy_deductible,
    )


def _refer(reason: DecisionReason) -> Decision:
    return Decision(outcome="refer_to_adjuster", reason=reason)
