"""Handoffs, stand-in agents, injected faults and the 30 cases end to end."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from builders import (
    INCIDENT_DATE,
    POLICY_NUMBER,
    build_claim,
    build_policy,
    build_prior_claim,
    build_reference,
)

from casefile.agents.mock import (
    MockExtractor,
    MockInvestigator,
    MockReviewer,
    extraction_from_documents,
)
from casefile.agents.triage import assess
from casefile.config import RAW_DATA_DIR
from casefile.control.handoff import Extraction, PoliceReportFields
from casefile.control.state import Limits
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.records import PolicyRecord, PriorClaimRecord
from casefile.data.transform import RAW_FILE_NAME
from casefile.documents.content import build_case_documents
from casefile.eval.cases import EvalCase, ExpectedResult, load_cases
from casefile.eval.run_mock import MOCK_LIMITS, mock_agent_factory
from casefile.eval.runner import CaseRun, compare, run_eval_case, run_suite
from casefile.eval.select_cases import build_portfolio

CLAIMS_PATH = RAW_DATA_DIR / RAW_FILE_NAME
ORACLE_PATH = RAW_DATA_DIR / ORACLE_FILE_NAME
needs_raw_files = pytest.mark.skipif(
    not (CLAIMS_PATH.exists() and ORACLE_PATH.exists()),
    reason="the raw datasets have not been downloaded",
)
PAY_4000 = ExpectedResult(
    outcome="recommend_pay", reason="covered", payable_amount=Decimal(4000)
)


class FakeLookup:
    """Domain data held in memory: one policy, its history, one reference."""

    def __init__(
        self,
        policy: PolicyRecord | None = None,
        prior_claims: tuple[PriorClaimRecord, ...] = (),
    ) -> None:
        self._policy = policy or build_policy()
        self._prior_claims = prior_claims

    def find_policy(self, policy_number: int) -> PolicyRecord | None:
        return self._policy if policy_number == self._policy.policy_number else None

    def find_prior_claims(self, policy_number: int) -> tuple[PriorClaimRecord, ...]:
        return self._prior_claims

    def find_reference(self, incident_type: str, incident_severity: str):
        return build_reference()


def build_case(
    defect: str | None = None,
    injection: str | None = None,
    expected: ExpectedResult = PAY_4000,
) -> EvalCase:
    group = "document_defect" if defect else "system_stress" if injection else "clean"
    return EvalCase(
        case_id="C01",
        group=group,
        claim_id="CLM-2015-00001",
        policy_number=POLICY_NUMBER,
        document_defect=defect,
        fault_injection=injection,
        note="test case",
        expected=expected,
    )


def run(case: EvalCase, lookup: FakeLookup | None = None) -> CaseRun:
    """Run ``case`` on the builder records with the stand-in agents."""
    lookup = lookup or FakeLookup()
    documents = build_case_documents(case, build_policy(), build_claim(), {1, 2, 3})
    agents = {
        "extractor": MockExtractor(documents),
        "investigator": MockInvestigator(lookup),
        "reviewer": MockReviewer(lookup),
    }
    return run_eval_case(case, agents, MOCK_LIMITS)


def extraction(defect: str | None = None) -> Extraction:
    documents = build_case_documents(
        build_case(defect), build_policy(), build_claim(), {1, 2, 3}
    )
    return extraction_from_documents(documents)


def test_clean_documents_have_no_issues() -> None:
    clean = extraction()

    assert clean.document_issues() == ()
    assert clean.amounts_add_up()
    assert clean.claim_form.policy_number == POLICY_NUMBER
    assert clean.repair_estimate.total_amount == Decimal(4000)
    assert clean.police_report.incident_date == INCIDENT_DATE


@pytest.mark.parametrize(
    ("defect", "issues"),
    [
        ("amount_mismatch", ("amount_mismatch",)),
        ("date_mismatch", ("date_mismatch",)),
        ("missing_document", ("missing_document",)),
        ("wrong_policy_number", ()),
        ("incident_outside_coverage", ()),
    ],
)
def test_document_issues_are_derived_from_the_extracted_values(
    defect: str, issues: tuple[str, ...]
) -> None:
    assert extraction(defect).document_issues() == issues


def test_several_issues_are_all_reported() -> None:
    clean = extraction()
    other_day = PoliceReportFields(policy_number=1, incident_date=date(2015, 1, 20))

    broken = clean.model_copy(
        update={"repair_estimate": None, "police_report": other_day}
    )

    assert broken.document_issues() == ("missing_document", "date_mismatch")


def test_a_missing_police_report_is_not_an_issue() -> None:
    assert (
        extraction().model_copy(update={"police_report": None}).document_issues() == ()
    )


def test_assess_uses_the_policy_named_on_the_documents() -> None:
    known = assess(extraction(), FakeLookup())
    unknown = assess(extraction("wrong_policy_number"), FakeLookup())

    assert (known.outcome, known.payable_amount) == ("recommend_pay", Decimal(4000))
    assert (unknown.outcome, unknown.reason) == (
        "refer_to_adjuster",
        "policy_not_found",
    )


def test_assess_counts_the_claim_history() -> None:
    history = tuple(
        build_prior_claim(number, date(2014, number, 1)) for number in (1, 2, 3)
    )

    decision = assess(extraction(), FakeLookup(prior_claims=history))

    assert decision.reason == "frequent_prior_claims"


def test_clean_case_runs_straight_through() -> None:
    result = run(build_case())

    assert result.passed
    assert result.state.path == ("extractor", "investigator", "reviewer")
    assert result.state.budget_spent == Decimal(3)
    assert result.state.recommendation.rationale == "Rule applied: covered."


@pytest.mark.parametrize(
    ("defect", "outcome", "reason"),
    [
        ("amount_mismatch", "refer_to_adjuster", "document_mismatch"),
        ("date_mismatch", "refer_to_adjuster", "document_mismatch"),
        ("missing_document", "refer_to_adjuster", "missing_document"),
        ("wrong_policy_number", "refer_to_adjuster", "policy_not_found"),
        (
            "incident_outside_coverage",
            "recommend_deny",
            "incident_outside_policy_period",
        ),
    ],
)
def test_planted_defect_leads_to_its_expected_result(
    defect: str, outcome: str, reason: str
) -> None:
    result = run(
        build_case(defect, expected=ExpectedResult(outcome=outcome, reason=reason))
    )

    assert result.passed, result.mismatches
    assert result.state.steps_taken == 3


def test_misread_amount_is_sent_back_once_and_then_corrected() -> None:
    expected = PAY_4000.model_copy(update={"reviewer_send_back": True})

    result = run(
        build_case(injection="extractor_error_first_attempt", expected=expected)
    )

    assert result.passed, result.mismatches
    assert result.state.path == ("extractor", "investigator", "reviewer") * 2
    assert result.state.revisions.extractor == 1
    assert result.state.trace[2].detail.startswith("The claimed amounts do not add up")
    assert result.state.extraction.claim_form.vehicle_claim == Decimal(4000)


@pytest.mark.parametrize(
    ("injection", "reason", "steps", "send_back"),
    [
        ("reviewer_never_approves", "max_revisions", 9, True),
        ("repeated_schema_errors", "schema_errors", 3, False),
        ("budget_too_small", "budget_exceeded", 0, False),
        ("step_limit_too_small", "max_steps", 2, False),
    ],
)
def test_injected_fault_trips_its_guard(
    injection: str, reason: str, steps: int, send_back: bool
) -> None:
    expected = ExpectedResult(
        outcome="escalated", reason=reason, reviewer_send_back=send_back
    )

    result = run(build_case(injection=injection, expected=expected))

    assert result.passed, result.mismatches
    assert result.state.steps_taken == steps
    assert result.observed.payable_amount is None


def test_wrong_outcome_reason_and_amount_are_reported() -> None:
    expected = ExpectedResult(outcome="recommend_deny", reason="below_deductible")

    mismatches = compare(expected, PAY_4000)

    assert len(mismatches) == 3
    assert mismatches[0] == "outcome: expected recommend_deny, got recommend_pay"


def test_send_back_is_required_only_where_the_case_demands_it() -> None:
    with_send_back = PAY_4000.model_copy(update={"reviewer_send_back": True})

    assert compare(PAY_4000, with_send_back) == ()
    assert compare(with_send_back, PAY_4000) == (
        "the reviewer never sent the work back",
    )


@needs_raw_files
def test_all_thirty_cases_end_as_pre_registered() -> None:
    suite = load_cases()
    portfolio = build_portfolio(CLAIMS_PATH, ORACLE_PATH)

    runs = run_suite(suite, mock_agent_factory(suite, portfolio), MOCK_LIMITS)

    failures = {run.case.case_id: run.mismatches for run in runs if not run.passed}
    assert failures == {}
    assert len(runs) == 30
    limits = Limits(budget=MOCK_LIMITS.budget)
    assert all(run.state.steps_taken <= limits.max_steps for run in runs)
    assert all(run.state.status in ("completed", "escalated") for run in runs)
    assert sum(run.observed.reviewer_send_back for run in runs) == 3
