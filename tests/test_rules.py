"""The triage rules, one rule at a time and in their order of precedence."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from builders import (
    INCIDENT_DATE,
    build_claim,
    build_policy,
    build_prior_claim,
    build_reference,
)

from casefile.rules import Decision, count_recent_prior_claims, decide

THREE_RECENT_PRIORS = [
    build_prior_claim(1, date(2013, 3, 1)),
    build_prior_claim(2, date(2014, 3, 1)),
    build_prior_claim(3, date(2014, 9, 1)),
]


def test_ordinary_claim_is_paid_minus_the_deductible() -> None:
    decision = decide(
        claim=build_claim(), policy=build_policy(), reference=build_reference()
    )

    assert decision == Decision(
        outcome="recommend_pay", reason="covered", payable_amount=Decimal(4000)
    )


def test_unknown_policy_is_referred() -> None:
    decision = decide(claim=build_claim(), policy=None)

    assert (decision.outcome, decision.reason) == (
        "refer_to_adjuster",
        "policy_not_found",
    )
    assert decision.payable_amount is None


def test_incident_before_policy_start_is_denied() -> None:
    policy = build_policy(policy_start_date=date(2015, 1, 26), months_as_customer=0)

    decision = decide(claim=build_claim(), policy=policy)

    assert (decision.outcome, decision.reason) == (
        "recommend_deny",
        "incident_outside_policy_period",
    )


def test_incident_on_the_policy_start_date_is_covered() -> None:
    policy = build_policy(policy_start_date=INCIDENT_DATE, months_as_customer=0)

    assert decide(claim=build_claim(), policy=policy).reason == "covered"


@pytest.mark.parametrize(
    ("issues", "reason"),
    [
        (["missing_document"], "missing_document"),
        (["amount_mismatch"], "document_mismatch"),
        (["date_mismatch"], "document_mismatch"),
        (["date_mismatch", "missing_document"], "missing_document"),
    ],
)
def test_document_problems_are_referred(issues: list[str], reason: str) -> None:
    decision = decide(
        claim=build_claim(), policy=build_policy(), document_issues=issues
    )

    assert (decision.outcome, decision.reason) == ("refer_to_adjuster", reason)


@pytest.mark.parametrize(
    ("high_amount", "reason"), [(3999, "amount_above_reference"), (4000, "covered")]
)
def test_only_amounts_above_the_reference_high_are_referred(
    high_amount: int, reason: str
) -> None:
    decision = decide(
        claim=build_claim(),
        policy=build_policy(),
        reference=build_reference(high_amount),
    )

    assert decision.reason == reason


def test_amount_is_not_judged_without_a_reference() -> None:
    claim = build_claim(
        total_claim_amount=Decimal(901_000), vehicle_claim=Decimal(900_000)
    )

    assert (
        decide(claim=claim, policy=build_policy(), reference=None).reason == "covered"
    )


def test_three_recent_prior_claims_are_referred() -> None:
    decision = decide(
        claim=build_claim(), policy=build_policy(), prior_claims=THREE_RECENT_PRIORS
    )

    assert (decision.outcome, decision.reason) == (
        "refer_to_adjuster",
        "frequent_prior_claims",
    )


def test_two_recent_prior_claims_are_not_enough() -> None:
    decision = decide(
        claim=build_claim(),
        policy=build_policy(),
        prior_claims=THREE_RECENT_PRIORS[:2],
    )

    assert decision.reason == "covered"


def test_prior_claims_are_counted_inside_the_window_of_one_policy() -> None:
    claim = build_claim()
    priors = [
        build_prior_claim(1, date(2012, 1, 25)),  # first day of the window
        build_prior_claim(2, date(2012, 1, 24)),  # one day too old
        build_prior_claim(3, date(2015, 1, 24)),  # the day before the incident
        build_prior_claim(4, INCIDENT_DATE),  # not earlier than the incident
        build_prior_claim(5, date(2014, 6, 1), policy_number=2),  # other policy
    ]

    assert count_recent_prior_claims(claim, priors) == 2


@pytest.mark.parametrize(
    ("deductible", "outcome"),
    [(5000, "recommend_deny"), (6000, "recommend_deny"), (4999, "recommend_pay")],
)
def test_claim_must_exceed_the_deductible(deductible: int, outcome: str) -> None:
    decision = decide(
        claim=build_claim(), policy=build_policy(policy_deductible=deductible)
    )

    assert decision.outcome == outcome
    if outcome == "recommend_deny":
        assert decision.reason == "below_deductible"
    else:
        assert decision.payable_amount == Decimal(1)


def test_rules_apply_in_order() -> None:
    """Every rule's condition holds at once; they must win one after another."""
    claim = build_claim()
    late_policy = build_policy(
        policy_start_date=date(2015, 2, 1), months_as_customer=0, policy_deductible=9000
    )
    policy = build_policy(policy_deductible=9000)
    everything = {
        "prior_claims": THREE_RECENT_PRIORS,
        "reference": build_reference(high_amount=3000),
    }
    issues = ["amount_mismatch", "missing_document"]

    reasons = [
        decide(claim=claim, policy=None, document_issues=issues, **everything),
        decide(claim=claim, policy=late_policy, document_issues=issues, **everything),
        decide(claim=claim, policy=policy, document_issues=issues, **everything),
        decide(claim=claim, policy=policy, document_issues=issues[:1], **everything),
        decide(claim=claim, policy=policy, **everything),
        decide(claim=claim, policy=policy, prior_claims=THREE_RECENT_PRIORS),
        decide(claim=claim, policy=policy),
    ]

    assert [decision.reason for decision in reasons] == [
        "policy_not_found",
        "incident_outside_policy_period",
        "missing_document",
        "document_mismatch",
        "amount_above_reference",
        "frequent_prior_claims",
        "below_deductible",
    ]


def test_payable_amount_belongs_to_pay_decisions_only() -> None:
    with pytest.raises(ValueError, match="payable_amount"):
        Decision(outcome="recommend_deny", reason="below_deductible", payable_amount=1)
    with pytest.raises(ValueError, match="payable_amount"):
        Decision(outcome="recommend_pay", reason="covered")
