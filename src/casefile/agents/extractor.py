"""The extractor: read the documents of a case with a language model.

The agent is given a folder of PDFs and nothing else. It never sees the
database or the answer key. Each document is sent to the model on its own,
with the JSON schema of the fields wanted from it, and the answer is validated
against that schema before it is passed on.

One request per document is deliberate. Shown all the documents at once, a
model tends to reconcile them: it would quietly copy the amount from the claim
form onto the repair estimate, and the contradiction this system exists to
find would disappear in the reading.

An answer that does not fit its schema is reported as invalid output and left
to the supervisor, which lets the agent try again and stops the run if it
fails twice in a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ValidationError
from pypdf import PdfReader

from casefile.control.handoff import (
    ClaimFormFields,
    Extraction,
    PoliceReportFields,
    RepairEstimateFields,
)
from casefile.control.state import CaseState, Extracted, InvalidOutput, StepEvent
from casefile.documents.render import (
    CLAIM_FORM_FILE,
    POLICE_REPORT_FILE,
    REPAIR_ESTIMATE_FILE,
)
from casefile.llm.budget import MeteredLLM
from casefile.llm.client import ChatRequest, Message

MAX_OUTPUT_TOKENS = 400
SYSTEM_PROMPT = (
    "You read one insurance document and report the requested fields exactly "
    "as the document states them. Do not correct, complete or infer values. "
    "Write dates as YYYY-MM-DD and amounts as plain numbers without currency "
    "symbols or thousands separators. Answer with JSON only."
)


@dataclass(frozen=True)
class DocumentSpec:
    """One kind of document and the fields wanted from it."""

    name: str
    file_name: str
    title: str
    fields: type[BaseModel]


CLAIM_FORM = DocumentSpec(
    "claim_form", CLAIM_FORM_FILE, "automobile claim form", ClaimFormFields
)
OPTIONAL_DOCUMENTS = (
    DocumentSpec(
        "repair_estimate",
        REPAIR_ESTIMATE_FILE,
        "vehicle repair estimate",
        RepairEstimateFields,
    ),
    DocumentSpec(
        "police_report",
        POLICE_REPORT_FILE,
        "police incident report",
        PoliceReportFields,
    ),
)


def read_pdf_text(path: Path) -> str:
    """Return the text of a PDF, page after page."""
    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


class LLMExtractor:
    """Reads the documents in a case folder with a language model."""

    def __init__(self, llm: MeteredLLM, case_folder: Path) -> None:
        form_path = case_folder / CLAIM_FORM_FILE
        if not form_path.exists():
            raise FileNotFoundError(f"{case_folder} has no {CLAIM_FORM_FILE}")
        self._llm = llm
        # A document that is not in the folder is simply not there; whether
        # that is a problem is for the rules to say, not for the reader.
        self._documents = [(CLAIM_FORM, read_pdf_text(form_path))] + [
            (spec, read_pdf_text(case_folder / spec.file_name))
            for spec in OPTIONAL_DOCUMENTS
            if (case_folder / spec.file_name).exists()
        ]

    def requests(self, state: CaseState) -> list[tuple[DocumentSpec, ChatRequest]]:
        """Return the request for each document, given where the case stands."""
        return [
            (spec, build_request(spec, text, state.feedback))
            for spec, text in self._documents
        ]

    def worst_case_cost(self, state: CaseState) -> Decimal:
        """Return the most tokens reading every document can take."""
        return Decimal(
            sum(self._llm.worst_case(request) for _, request in self.requests(state))
        )

    def run(self, state: CaseState) -> StepEvent:
        """Read every document; report the extraction or the first bad answer."""
        tokens = 0
        fields: dict[str, BaseModel] = {}
        for spec, request in self.requests(state):
            response = self._llm.complete(request)
            tokens += response.usage.total_tokens
            try:
                fields[spec.name] = spec.fields.model_validate_json(response.content)
            except ValidationError as error:
                problem = error.errors()[0]
                return InvalidOutput(
                    cost=Decimal(tokens),
                    detail=f"{spec.name}: {problem['msg']} at {list(problem['loc'])}",
                )
        return Extracted(cost=Decimal(tokens), extraction=Extraction(**fields))


def build_request(spec: DocumentSpec, text: str, feedback: str | None) -> ChatRequest:
    """Build the request that asks the model for the fields of one document."""
    instruction = f"This is the text of a {spec.title}. Report its fields."
    if feedback:
        instruction += (
            " An earlier reading of this case was rejected by a reviewer for "
            f"this reason: {feedback} Read the document again with care."
        )
    return ChatRequest(
        messages=(
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=f"{instruction}\n\n{text}"),
        ),
        max_tokens=MAX_OUTPUT_TOKENS,
        json_schema=spec.fields.model_json_schema(),
        schema_name=spec.name,
    )
