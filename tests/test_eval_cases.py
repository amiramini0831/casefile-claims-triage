"""The pre-registered evaluation cases in eval/cases.yaml."""

from __future__ import annotations

from collections import Counter

import pytest

from casefile.config import RAW_DATA_DIR
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME
from casefile.eval.cases import (
    CASES_PATH,
    EvalCase,
    EvalSuite,
    ExpectedResult,
    load_cases,
)
from casefile.eval.select_cases import (
    SEED,
    Portfolio,
    build_portfolio,
    render,
    select_cases,
)
from casefile.rules import FREQUENT_CLAIMS_COUNT, FREQUENT_CLAIMS_WINDOW_MONTHS

CLAIMS_PATH = RAW_DATA_DIR / RAW_FILE_NAME
ORACLE_PATH = RAW_DATA_DIR / ORACLE_FILE_NAME
needs_raw_files = pytest.mark.skipif(
    not (CLAIMS_PATH.exists() and ORACLE_PATH.exists()),
    reason="the raw datasets have not been downloaded",
)


@pytest.fixture(scope="module")
def suite() -> EvalSuite:
    return load_cases()


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    return build_portfolio(CLAIMS_PATH, ORACLE_PATH)


def test_file_holds_thirty_cases_in_the_planned_groups(suite: EvalSuite) -> None:
    assert [case.case_id for case in suite.cases] == [
        f"C{number:02d}" for number in range(1, 31)
    ]
    assert Counter(case.group for case in suite.cases) == {
        "clean": 8,
        "natural": 6,
        "document_defect": 10,
        "system_stress": 6,
    }


def test_file_records_the_thresholds_the_rules_use(suite: EvalSuite) -> None:
    assert suite.seed == SEED
    assert suite.rules.frequent_claims_count == FREQUENT_CLAIMS_COUNT
    assert suite.rules.frequent_claims_window_months == FREQUENT_CLAIMS_WINDOW_MONTHS


def test_every_defect_and_fault_is_represented(suite: EvalSuite) -> None:
    defects = Counter(c.document_defect for c in suite.cases if c.document_defect)
    faults = Counter(c.fault_injection for c in suite.cases if c.fault_injection)

    assert defects == {
        "amount_mismatch": 2,
        "date_mismatch": 2,
        "missing_document": 2,
        "wrong_policy_number": 2,
        "incident_outside_coverage": 2,
    }
    assert faults == {
        "extractor_error_first_attempt": 2,
        "reviewer_never_approves": 1,
        "repeated_schema_errors": 1,
        "budget_too_small": 1,
        "step_limit_too_small": 1,
    }


def test_expected_outcomes_add_up(suite: EvalSuite) -> None:
    assert Counter(case.expected.outcome for case in suite.cases) == {
        "recommend_pay": 10,
        "refer_to_adjuster": 13,
        "recommend_deny": 3,
        "escalated": 4,
    }


def test_every_stopping_guard_is_exercised_once(suite: EvalSuite) -> None:
    reasons = sorted(
        case.expected.reason
        for case in suite.cases
        if case.expected.outcome == "escalated"
    )

    assert reasons == ["budget_exceeded", "max_revisions", "max_steps", "schema_errors"]


def test_a_reviewer_send_back_must_still_end_normally(suite: EvalSuite) -> None:
    recovered = [
        case
        for case in suite.cases
        if case.expected.reviewer_send_back and case.expected.outcome != "escalated"
    ]

    assert len(recovered) == 2


def test_pay_cases_state_a_positive_amount(suite: EvalSuite) -> None:
    paying = [c for c in suite.cases if c.expected.outcome == "recommend_pay"]

    assert all(case.expected.payable_amount > 0 for case in paying)


def test_escalation_needs_a_stopping_reason() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        ExpectedResult(outcome="escalated", reason="covered")
    with pytest.raises(ValueError, match="does not fit"):
        ExpectedResult(outcome="refer_to_adjuster", reason="max_steps")


def test_case_group_must_match_its_defect(suite: EvalSuite) -> None:
    clean = suite.cases[0].model_dump()

    with pytest.raises(ValueError, match="document defect"):
        EvalCase(**{**clean, "document_defect": "amount_mismatch"})
    with pytest.raises(ValueError, match="fault injection"):
        EvalCase(**{**clean, "fault_injection": "budget_too_small"})


def test_claims_cannot_be_reused(suite: EvalSuite) -> None:
    duplicated = suite.model_dump()
    duplicated["cases"][1]["claim_id"] = duplicated["cases"][0]["claim_id"]

    with pytest.raises(ValueError, match="claim_id values are not unique"):
        EvalSuite(**duplicated)


@needs_raw_files
def test_file_is_exactly_what_the_selection_produces(portfolio: Portfolio) -> None:
    assert CASES_PATH.read_text(encoding="utf-8") == render(select_cases(portfolio))


@needs_raw_files
def test_selection_depends_only_on_the_seed(portfolio: Portfolio) -> None:
    assert select_cases(portfolio, seed=1) == select_cases(portfolio, seed=1)
    assert select_cases(portfolio, seed=1) != select_cases(portfolio, seed=2)


@needs_raw_files
def test_cases_without_defects_follow_the_rules(
    suite: EvalSuite, portfolio: Portfolio
) -> None:
    claims = {claim.claim_id: claim for claim in portfolio.claims}

    for case in suite.cases:
        if case.group in ("clean", "natural"):
            decision = portfolio.decide(claims[case.claim_id])
            assert case.expected.outcome == decision.outcome
            assert case.expected.reason == decision.reason
            assert case.expected.payable_amount == decision.payable_amount


@needs_raw_files
def test_clean_cases_cover_every_incident_type(
    suite: EvalSuite, portfolio: Portfolio
) -> None:
    claims = {claim.claim_id: claim for claim in portfolio.claims}

    incident_types = Counter(
        claims[case.claim_id].incident_type
        for case in suite.cases
        if case.group == "clean"
    )

    assert incident_types == {
        "Single Vehicle Collision": 2,
        "Multi-vehicle Collision": 2,
        "Vehicle Theft": 2,
        "Parked Car": 2,
    }


@needs_raw_files
def test_defect_and_stress_cases_start_from_ordinary_claims(
    suite: EvalSuite, portfolio: Portfolio
) -> None:
    claims = {claim.claim_id: claim for claim in portfolio.claims}

    for case in suite.cases:
        if case.group in ("document_defect", "system_stress"):
            assert portfolio.decide(claims[case.claim_id]).reason == "covered"
        if case.document_defect == "date_mismatch":
            assert claims[case.claim_id].police_report_available is True
