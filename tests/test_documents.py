"""The synthetic case documents and the defects planted in them."""

from __future__ import annotations

import io
import random
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from builders import INCIDENT_DATE, POLICY_NUMBER, build_claim, build_policy
from pypdf import PdfReader

from casefile.config import RAW_DATA_DIR
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME
from casefile.documents.content import (
    CaseDocuments,
    build_case_documents,
    split_amount,
)
from casefile.documents.generate import (
    CASES_DIR,
    DEFECTS_PATH,
    build_all_documents,
    render_defects,
    write_case_folders,
)
from casefile.documents.render import (
    CLAIM_FORM_FILE,
    POLICE_REPORT_FILE,
    REPAIR_ESTIMATE_FILE,
    render_case,
    render_pdf,
)
from casefile.eval.cases import EvalCase, EvalSuite, ExpectedResult, load_cases
from casefile.eval.select_cases import Portfolio, build_portfolio

CLAIMS_PATH = RAW_DATA_DIR / RAW_FILE_NAME
ORACLE_PATH = RAW_DATA_DIR / ORACLE_FILE_NAME
needs_raw_files = pytest.mark.skipif(
    not (CLAIMS_PATH.exists() and ORACLE_PATH.exists()),
    reason="the raw datasets have not been downloaded",
)
KNOWN_POLICY_NUMBERS = {POLICY_NUMBER, 2, 3}


def build_documents(
    defect: str | None = None, **claim_changes: object
) -> CaseDocuments:
    """Build the documents of a one-off case around the builder records."""
    claim = build_claim(**claim_changes)
    case = EvalCase(
        case_id="C01",
        group="document_defect" if defect else "clean",
        claim_id=claim.claim_id,
        policy_number=claim.policy_number,
        document_defect=defect,
        note="test case",
        expected=ExpectedResult(outcome="recommend_deny", reason="below_deductible"),
    )
    return build_case_documents(case, build_policy(), claim, KNOWN_POLICY_NUMBERS)


def pdf_text(content: bytes) -> str:
    return "\n".join(
        page.extract_text() for page in PdfReader(io.BytesIO(content)).pages
    )


@pytest.mark.parametrize(("total", "parts"), [(45_200, 6), (70, 5), (6, 6), (3, 6)])
def test_split_amount_is_exact_and_positive(total: int, parts: int) -> None:
    amounts = split_amount(total, parts, random.Random(1))

    assert sum(amounts) == total
    assert all(amount >= 1 for amount in amounts)
    assert len(amounts) == min(parts, total)


def test_clean_documents_agree_with_each_other_and_the_records() -> None:
    documents = build_documents()
    form, estimate, report = (
        documents.claim_form,
        documents.repair_estimate,
        documents.police_report,
    )

    assert documents.defects == ()
    assert form.policy_number == estimate.policy_number == report.policy_number == 1
    assert form.incident_date == report.incident_date == INCIDENT_DATE
    assert form.policyholder_name == estimate.customer_name == report.driver_name
    assert estimate.total_amount == form.vehicle_claim == Decimal(4000)
    assert sum(item.amount for item in estimate.items) == estimate.total_amount
    assert form.total_claim_amount == Decimal(5000)
    assert form.date_filed > INCIDENT_DATE
    assert form.collision_type == "Not applicable"


@pytest.mark.parametrize(("available", "stated"), [(False, "No"), (None, "Not stated")])
def test_no_police_report_unless_one_is_available(
    available: bool | None, stated: str
) -> None:
    documents = build_documents(police_report_available=available)

    assert documents.police_report is None
    assert documents.claim_form.police_report == stated


def test_collision_description_names_the_impact_area() -> None:
    documents = build_documents(
        incident_type="Multi-vehicle Collision",
        collision_type="Rear Collision",
        number_of_vehicles_involved=3,
    )

    description = documents.claim_form.description
    assert "collision with 2 other vehicles" in description
    assert "rear of the vehicle" in description
    assert documents.claim_form.collision_type == "Rear Collision"


def test_same_case_always_gets_the_same_documents() -> None:
    assert build_documents() == build_documents()


def test_amount_mismatch_inflates_only_the_repair_estimate() -> None:
    documents = build_documents("amount_mismatch")
    estimate = documents.repair_estimate

    assert documents.claim_form.vehicle_claim == Decimal(4000)
    assert estimate.total_amount > Decimal(4000)
    assert sum(item.amount for item in estimate.items) == estimate.total_amount
    (defect,) = documents.defects
    assert (defect.document, defect.field) == ("repair_estimate", "total_amount")
    assert defect.true_value == "4000"
    assert defect.planted_value == str(estimate.total_amount)


def test_date_mismatch_changes_only_the_police_report() -> None:
    documents = build_documents("date_mismatch")

    assert documents.claim_form.incident_date == INCIDENT_DATE
    assert documents.police_report.incident_date != INCIDENT_DATE
    (defect,) = documents.defects
    assert (defect.document, defect.field) == ("police_report", "incident_date")
    assert defect.true_value == INCIDENT_DATE.isoformat()
    assert defect.planted_value == documents.police_report.incident_date.isoformat()


def test_date_mismatch_needs_a_police_report() -> None:
    with pytest.raises(ValueError, match="needs a police report"):
        build_documents("date_mismatch", police_report_available=False)


def test_missing_document_leaves_out_the_repair_estimate() -> None:
    documents = build_documents("missing_document")

    assert documents.repair_estimate is None
    assert REPAIR_ESTIMATE_FILE not in render_case(documents)
    (defect,) = documents.defects
    assert (defect.document, defect.planted_value) == ("repair_estimate", "absent")


def test_wrong_policy_number_is_unknown_and_on_every_document() -> None:
    documents = build_documents("wrong_policy_number")
    planted = documents.claim_form.policy_number

    assert planted not in KNOWN_POLICY_NUMBERS
    assert documents.repair_estimate.policy_number == planted
    assert documents.police_report.policy_number == planted
    (defect,) = documents.defects
    assert (defect.true_value, defect.planted_value) == ("1", str(planted))


def test_incident_outside_coverage_predates_the_policy() -> None:
    documents = build_documents("incident_outside_coverage")
    policy_start = date(2005, 1, 25)

    assert documents.claim_form.incident_date < policy_start
    assert documents.police_report.incident_date < policy_start
    assert documents.claim_form.date_filed > INCIDENT_DATE
    (defect,) = documents.defects
    assert (defect.document, defect.field) == ("all", "incident_date")


def test_pdf_is_reproducible_and_readable() -> None:
    documents = build_documents()

    first = render_case(documents)
    second = render_case(documents)

    assert first == second
    assert set(first) == {CLAIM_FORM_FILE, REPAIR_ESTIMATE_FILE, POLICE_REPORT_FILE}
    form_text = pdf_text(first[CLAIM_FORM_FILE])
    assert "Claim number: CLM-2015-00001" in form_text
    assert "Date of incident: January 25, 2015" in form_text
    assert "Total claimed 5,000.00" in form_text
    assert "Estimate total 4,000.00" in pdf_text(first[REPAIR_ESTIMATE_FILE])
    assert "Date of incident: Jan 25, 2015" in pdf_text(first[POLICE_REPORT_FILE])


def test_text_is_escaped_in_the_pdf() -> None:
    documents = build_documents(incident_location="5 <b>Main</b> & 1st")

    assert "5 <b>Main</b> & 1st" in pdf_text(render_pdf(documents.claim_form))


@pytest.fixture(scope="module")
def suite() -> EvalSuite:
    return load_cases()


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    return build_portfolio(CLAIMS_PATH, ORACLE_PATH)


@pytest.fixture(scope="module")
def all_documents(suite: EvalSuite, portfolio: Portfolio) -> list[CaseDocuments]:
    return build_all_documents(suite, portfolio)


@needs_raw_files
def test_files_on_disk_are_exactly_what_the_generator_produces(
    all_documents: list[CaseDocuments], tmp_path: Path
) -> None:
    write_case_folders(all_documents, tmp_path)

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*.pdf"))
        }

    assert snapshot(CASES_DIR) == snapshot(tmp_path)
    assert DEFECTS_PATH.read_text(encoding="utf-8") == render_defects(all_documents)


@needs_raw_files
def test_stale_documents_are_removed(
    all_documents: list[CaseDocuments], tmp_path: Path
) -> None:
    missing = next(d for d in all_documents if d.repair_estimate is None)
    leftover = tmp_path / missing.case_id / REPAIR_ESTIMATE_FILE
    leftover.parent.mkdir()
    leftover.write_bytes(b"old run")

    write_case_folders([missing], tmp_path)

    assert not leftover.exists()


@needs_raw_files
def test_answer_key_matches_the_pre_registered_defects(
    suite: EvalSuite, all_documents: list[CaseDocuments]
) -> None:
    planned = {c.case_id: c.document_defect for c in suite.cases if c.document_defect}
    planted = {
        defect.case_id: defect.defect
        for documents in all_documents
        for defect in documents.defects
    }

    assert planted == planned
    assert len(planted) == 10


@needs_raw_files
def test_planted_policy_numbers_do_not_exist(
    all_documents: list[CaseDocuments], portfolio: Portfolio
) -> None:
    planted = [
        int(defect.planted_value)
        for documents in all_documents
        for defect in documents.defects
        if defect.defect == "wrong_policy_number"
    ]

    assert len(planted) == 2
    assert not set(planted) & portfolio.policies.keys()


@needs_raw_files
def test_documents_without_defects_state_the_true_values(
    suite: EvalSuite, portfolio: Portfolio, all_documents: list[CaseDocuments]
) -> None:
    claims = {claim.claim_id: claim for claim in portfolio.claims}

    for case, documents in zip(suite.cases, all_documents, strict=True):
        if case.document_defect is not None:
            continue
        claim = claims[case.claim_id]
        form, estimate = documents.claim_form, documents.repair_estimate
        assert documents.defects == ()
        assert form.policy_number == estimate.policy_number == claim.policy_number
        assert form.incident_date == claim.incident_date
        assert form.total_claim_amount == claim.total_claim_amount
        assert estimate.total_amount == claim.vehicle_claim
        assert (documents.police_report is not None) == bool(
            claim.police_report_available
        )


@needs_raw_files
def test_every_generated_pdf_can_be_read_back(suite: EvalSuite) -> None:
    claim_ids = {case.case_id: case.claim_id for case in suite.cases}
    paths = sorted(CASES_DIR.rglob("*.pdf"))

    assert {path.parent.name for path in paths} == set(claim_ids)
    for path in paths:
        text = pdf_text(path.read_bytes())
        if path.name != POLICE_REPORT_FILE:
            assert claim_ids[path.parent.name] in text
        assert "Policy number:" in text or "policy number:" in text
