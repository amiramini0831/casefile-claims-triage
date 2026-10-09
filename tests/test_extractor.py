"""The model-backed extractor, with a scripted model instead of a real one."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from builders import build_claim, build_eval_case, build_policy
from crash_run import Lookup

from casefile.agents.extractor import (
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    LLMExtractor,
)
from casefile.agents.mock import (
    MockInvestigator,
    MockReviewer,
    extraction_from_documents,
)
from casefile.config import RAW_DATA_DIR
from casefile.control.handoff import ClaimFormFields, Extraction
from casefile.control.state import Extracted, InvalidOutput, Limits
from casefile.control.supervisor import run_case, start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME
from casefile.documents.content import CaseDocuments, build_case_documents
from casefile.documents.render import (
    POLICE_REPORT_FILE,
    REPAIR_ESTIMATE_FILE,
    render_case,
)
from casefile.eval.run_extraction import (
    EXTRACTION_RECORDINGS,
    REPORT_PATH,
    compare_extractions,
    run_extraction,
)
from casefile.llm.budget import worst_case_tokens
from casefile.llm.client import ChatRequest, ChatResponse, Usage

LIMITS = Limits(budget=Decimal(1_000_000))
TOKENS_PER_CALL = 100


class ScriptedLLM:
    """Answers each kind of document from a list, and remembers the requests."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self._answers = {name: list(contents) for name, contents in answers.items()}
        self.requests: list[ChatRequest] = []

    def worst_case(self, request: ChatRequest) -> int:
        return worst_case_tokens(request)

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            content=self._answers[request.schema_name].pop(0),
            finish_reason="stop",
            usage=Usage(prompt_tokens=TOKENS_PER_CALL - 10, completion_tokens=10),
            latency_seconds=0.1,
        )


def write_case(folder: Path, defect: str | None = None) -> CaseDocuments:
    """Write the PDFs of a builder case into ``folder``."""
    case = build_eval_case()
    if defect:
        case = case.model_copy(
            update={"group": "document_defect", "document_defect": defect}
        )
    documents = build_case_documents(case, build_policy(), build_claim(), {1})
    for name, content in render_case(documents).items():
        (folder / name).write_bytes(content)
    return documents


def perfect_answers(documents: CaseDocuments) -> dict[str, list[str]]:
    """What a model that reads flawlessly would answer for each document."""
    ideal = extraction_from_documents(documents)
    return {
        name: [getattr(ideal, name).model_dump_json()]
        for name in ("claim_form", "repair_estimate", "police_report")
        if getattr(ideal, name) is not None
    }


def test_each_document_is_read_on_its_own_and_validated(tmp_path: Path) -> None:
    documents = write_case(tmp_path)
    llm = ScriptedLLM(perfect_answers(documents))

    event = LLMExtractor(llm, tmp_path).run(start("C01", LIMITS))

    assert isinstance(event, Extracted)
    assert event.extraction == extraction_from_documents(documents)
    assert event.cost == Decimal(3 * TOKENS_PER_CALL)
    assert [request.schema_name for request in llm.requests] == [
        "claim_form",
        "repair_estimate",
        "police_report",
    ]


def test_request_holds_the_document_text_and_its_schema(tmp_path: Path) -> None:
    llm = ScriptedLLM(perfect_answers(write_case(tmp_path)))

    LLMExtractor(llm, tmp_path).run(start("C01", LIMITS))

    form_request, estimate_request, _ = llm.requests
    system, user = form_request.messages
    assert system.content == SYSTEM_PROMPT
    assert "automobile claim form" in user.content
    assert "Claim number: CLM-2015-00001" in user.content
    assert "Total claimed 5,000.00" in user.content
    assert form_request.json_schema == ClaimFormFields.model_json_schema()
    assert form_request.max_tokens == MAX_OUTPUT_TOKENS
    # Documents are not mixed: the estimate request has nothing of the form.
    assert "Estimate total 4,000.00" in estimate_request.messages[1].content
    assert "Total claimed" not in estimate_request.messages[1].content


def test_documents_missing_from_the_folder_are_simply_absent(tmp_path: Path) -> None:
    documents = write_case(tmp_path, "missing_document")
    (tmp_path / POLICE_REPORT_FILE).unlink()
    answers = perfect_answers(documents)
    llm = ScriptedLLM({"claim_form": answers["claim_form"]})

    event = LLMExtractor(llm, tmp_path).run(start("C01", LIMITS))

    assert not (tmp_path / REPAIR_ESTIMATE_FILE).exists()
    assert event.extraction.repair_estimate is None
    assert event.extraction.police_report is None
    assert event.extraction.document_issues() == ("missing_document",)
    assert len(llm.requests) == 1


def test_planted_mismatch_is_reported_as_the_documents_state_it(
    tmp_path: Path,
) -> None:
    documents = write_case(tmp_path, "amount_mismatch")
    llm = ScriptedLLM(perfect_answers(documents))

    event = LLMExtractor(llm, tmp_path).run(start("C01", LIMITS))

    assert event.extraction.claim_form.vehicle_claim == Decimal(4000)
    assert event.extraction.repair_estimate.total_amount > Decimal(4000)
    assert event.extraction.document_issues() == ("amount_mismatch",)


@pytest.mark.parametrize(
    "bad_answer",
    [
        "The claim number is CLM-2015-00001.",
        "",
        '{"claim_number": "CLM-2015-00001"}',
        json.dumps(
            {
                "claim_number": "CLM-2015-00001",
                "policy_number": 1,
                "incident_date": "2015-01-25",
                "incident_type": "Meteor Strike",
                "incident_severity": "Minor Damage",
                "injury_claim": 0,
                "property_claim": 1000,
                "vehicle_claim": 4000,
                "total_claim_amount": 5000,
            }
        ),
    ],
    ids=["prose", "empty", "missing-fields", "unknown-incident-type"],
)
def test_answer_that_does_not_fit_the_schema_is_invalid_output(
    tmp_path: Path, bad_answer: str
) -> None:
    write_case(tmp_path)
    llm = ScriptedLLM({"claim_form": [bad_answer]})

    event = LLMExtractor(llm, tmp_path).run(start("C01", LIMITS))

    assert isinstance(event, InvalidOutput)
    assert event.detail.startswith("claim_form:")
    # The tokens of the failed call are still paid for, and nothing more is asked.
    assert event.cost == Decimal(TOKENS_PER_CALL)
    assert len(llm.requests) == 1


def test_worst_case_covers_every_document(tmp_path: Path) -> None:
    llm = ScriptedLLM(perfect_answers(write_case(tmp_path)))
    extractor = LLMExtractor(llm, tmp_path)
    state = start("C01", LIMITS)

    worst_case = extractor.worst_case_cost(state)

    requests = [request for _, request in extractor.requests(state)]
    assert worst_case == sum(worst_case_tokens(request) for request in requests)
    assert worst_case > 3 * MAX_OUTPUT_TOKENS
    assert llm.requests == []


def test_reviewer_feedback_is_passed_to_the_model(tmp_path: Path) -> None:
    llm = ScriptedLLM(perfect_answers(write_case(tmp_path)))
    extractor = LLMExtractor(llm, tmp_path)
    fresh = start("C01", LIMITS)
    sent_back = fresh.model_copy(update={"feedback": "The amounts do not add up."})

    first = extractor.requests(fresh)[0][1].messages[1].content
    second = extractor.requests(sent_back)[0][1].messages[1].content

    assert "rejected by a reviewer" not in first
    assert "The amounts do not add up." in second


def test_folder_without_a_claim_form_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="claim_form.pdf"):
        LLMExtractor(ScriptedLLM({}), tmp_path)


def test_model_that_keeps_answering_badly_is_stopped(tmp_path: Path) -> None:
    write_case(tmp_path)
    agents = {
        "extractor": LLMExtractor(
            ScriptedLLM({"claim_form": ["nonsense", "more nonsense"]}), tmp_path
        ),
        "investigator": MockInvestigator(Lookup()),
        "reviewer": MockReviewer(Lookup()),
    }

    state = run_case(start("C01", LIMITS), agents)

    assert (state.status, state.escalation_reason) == ("escalated", "schema_errors")
    assert state.path == ("extractor", "extractor")
    assert state.budget_spent == Decimal(2 * TOKENS_PER_CALL)


def test_one_bad_answer_is_retried_and_the_case_completes(tmp_path: Path) -> None:
    answers = perfect_answers(write_case(tmp_path))
    answers["claim_form"].insert(0, "nonsense")
    agents = {
        "extractor": LLMExtractor(ScriptedLLM(answers), tmp_path),
        "investigator": MockInvestigator(Lookup()),
        "reviewer": MockReviewer(Lookup()),
    }

    state = run_case(start("C01", LIMITS), agents)

    assert state.status == "completed"
    assert state.path == ("extractor", "extractor", "investigator", "reviewer")
    assert state.recommendation.rule_decision.payable_amount == Decimal(4000)


# --- comparing an extraction with what the documents say ----------------------


def test_identical_extractions_have_no_differences(tmp_path: Path) -> None:
    ideal = extraction_from_documents(write_case(tmp_path))

    assert compare_extractions(ideal, ideal) == (13, [])


def test_wrong_and_missing_values_are_listed(tmp_path: Path) -> None:
    ideal = extraction_from_documents(write_case(tmp_path))
    form = ideal.claim_form.model_copy(update={"vehicle_claim": Decimal(400)})
    actual = Extraction(claim_form=form, repair_estimate=ideal.repair_estimate)

    fields, differences = compare_extractions(ideal, actual)

    assert fields == 13
    assert differences[0] == {
        "field": "claim_form.vehicle_claim",
        "expected": "4000",
        "extracted": "400",
    }
    assert [d["field"] for d in differences[1:]] == [
        "police_report.policy_number",
        "police_report.incident_date",
    ]
    assert {d["extracted"] for d in differences[1:]} == {"nothing"}


def test_no_extraction_counts_every_field_as_wrong(tmp_path: Path) -> None:
    ideal = extraction_from_documents(write_case(tmp_path))

    fields, differences = compare_extractions(ideal, None)

    assert (fields, len(differences)) == (13, 13)


def test_inventing_a_missing_document_is_an_error(tmp_path: Path) -> None:
    ideal = extraction_from_documents(write_case(tmp_path))
    without_report = ideal.model_copy(update={"police_report": None})

    fields, differences = compare_extractions(without_report, ideal)

    assert fields == 12
    assert differences == [
        {"field": "police_report", "expected": "absent", "extracted": "present"}
    ]


# --- the recorded run with the real model --------------------------------------


@pytest.mark.skipif(
    not (REPORT_PATH.exists() and EXTRACTION_RECORDINGS.exists()),
    reason="the extraction has not been run and recorded yet",
)
def test_recorded_answers_reproduce_the_report_offline() -> None:
    """Replaying the recordings needs no model, no key and no network."""
    claims, oracle = RAW_DATA_DIR / RAW_FILE_NAME, RAW_DATA_DIR / ORACLE_FILE_NAME
    if not (claims.exists() and oracle.exists()):
        pytest.skip("the raw datasets have not been downloaded")
    stored = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    replayed = run_extraction("replay", None, stored["model"], claims, oracle)

    assert replayed == stored
