"""Measure how accurately the extractor reads the documents of the 30 cases.

    python -m casefile.eval.run_extraction            # call the model, record
    python -m casefile.eval.run_extraction --mode replay   # recorded answers only

Every field the extractor reports is compared with what the document really
says. "What the document says" comes from the generator that wrote the PDFs,
so a planted defect counts as correct when it is read as planted: the
extractor's job is to report the documents, not to fix them.

Model answers are recorded under recordings/extraction/. With them, the whole
measurement can be repeated offline, without a key, and gives the same report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from casefile.agents.extractor import LLMExtractor
from casefile.agents.mock import extraction_from_documents
from casefile.config import (
    RAW_DATA_DIR,
    RECORDINGS_DIR,
    REPORTS_DIR,
    ConfigurationError,
    load_llm_settings,
)
from casefile.control.handoff import Extraction
from casefile.control.state import Extracted, Limits
from casefile.control.supervisor import start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME, RawDataError
from casefile.documents.generate import CASES_DIR, build_all_documents
from casefile.eval.cases import load_cases
from casefile.eval.select_cases import build_portfolio
from casefile.llm.budget import BudgetedLLM
from casefile.llm.client import LLM, LLMError, OpenAICompatibleClient
from casefile.llm.recording import MissingRecordingError, RecordingLLM, RecordingMode

REPORT_PATH = REPORTS_DIR / "extraction_report.json"
EXTRACTION_RECORDINGS = RECORDINGS_DIR / "extraction"
DOCUMENT_NAMES = ("claim_form", "repair_estimate", "police_report")
# This measurement is not about the stopping guards.
NO_LIMIT = Limits(budget=Decimal(10**9))


def compare_extractions(
    expected: Extraction, actual: Extraction | None
) -> tuple[int, list[dict[str, str]]]:
    """Return the number of expected fields and those that were read wrongly.

    ``actual`` is ``None`` when the extractor produced no valid extraction, in
    which case every expected field counts as wrong.
    """
    total = 0
    differences: list[dict[str, str]] = []
    for name in DOCUMENT_NAMES:
        wanted = getattr(expected, name)
        found = getattr(actual, name) if actual is not None else None
        if wanted is None:
            if found is not None:
                # Reporting a document that is not there is one wrong field.
                total += 1
                differences.append(
                    {"field": name, "expected": "absent", "extracted": "present"}
                )
            continue
        for field, value in wanted:
            total += 1
            extracted = getattr(found, field) if found is not None else None
            if extracted != value:
                differences.append(
                    {
                        "field": f"{name}.{field}",
                        "expected": str(value),
                        "extracted": "nothing" if extracted is None else str(extracted),
                    }
                )
    return total, differences


def run_extraction(
    mode: RecordingMode,
    model: LLM | None,
    model_name: str,
    claims_path: Path,
    oracle_path: Path,
    only: Sequence[str] = (),
) -> dict[str, Any]:
    """Extract every case and return the report."""
    suite = load_cases()
    truth = build_all_documents(suite, build_portfolio(claims_path, oracle_path))
    cases = []
    for case, documents in zip(suite.cases, truth, strict=True):
        if only and case.case_id not in only:
            continue
        recorder = RecordingLLM(
            EXTRACTION_RECORDINGS / case.case_id, mode, model, model=model_name
        )
        llm = BudgetedLLM(recorder)
        event = LLMExtractor(llm, CASES_DIR / case.case_id).run(
            start(case.case_id, NO_LIMIT)
        )
        extraction = event.extraction if isinstance(event, Extracted) else None
        fields, differences = compare_extractions(
            extraction_from_documents(documents), extraction
        )
        cases.append(
            {
                "case_id": case.case_id,
                "document_defect": case.document_defect,
                "result": event.kind,
                "detail": getattr(event, "detail", ""),
                "fields": fields,
                "fields_correct": fields - len(differences),
                "differences": differences,
                "model_calls": llm.calls,
                "tokens": llm.tokens_used,
            }
        )
        mark = "ok  " if not differences else "DIFF"
        print(
            f"{mark} {case.case_id} {cases[-1]['fields_correct']}/{fields} fields, "
            f"{llm.calls} calls, {llm.tokens_used} tokens"
            + (f" ({recorder.replayed} replayed)" if recorder.replayed else ""),
            flush=True,
        )

    fields = sum(case["fields"] for case in cases)
    correct = sum(case["fields_correct"] for case in cases)
    return {
        "model": model_name,
        "measures": "field-level accuracy of the extractor against the documents",
        "summary": {
            "cases": len(cases),
            "cases_read_without_error": sum(not c["differences"] for c in cases),
            "invalid_outputs": sum(c["result"] != "extracted" for c in cases),
            "fields": fields,
            "fields_correct": correct,
            "model_calls": sum(case["model_calls"] for case in cases),
            "tokens": sum(case["tokens"] for case in cases),
        },
        "cases": cases,
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=("auto", "record", "replay"),
        default="auto",
        help="auto: reuse recorded answers and call the model for the rest "
        "(default); record: call the model for everything; replay: never call it",
    )
    parser.add_argument(
        "--cases", nargs="*", default=(), help="case ids to run (default: all)"
    )
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--oracle-path", type=Path, default=RAW_DATA_DIR / ORACLE_FILE_NAME
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write the report. Returns an exit code."""
    args = parse_args(argv)
    client = None
    model_name = recorded_model_name()
    try:
        if args.mode != "replay":
            settings = load_llm_settings()
            client = OpenAICompatibleClient(settings)
            model_name = settings.model
        report = run_extraction(
            args.mode,
            client,
            model_name,
            args.claims_path,
            args.oracle_path,
            args.cases,
        )
    except (
        ConfigurationError,
        LLMError,
        MissingRecordingError,
        RawDataError,
        FileNotFoundError,
    ) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        print(
            "Answers recorded so far are kept; run again to continue.", file=sys.stderr
        )
        return 1
    finally:
        if client is not None:
            client.close()

    summary = report["summary"]
    print(
        f"\n{summary['fields_correct']} of {summary['fields']} fields read correctly; "
        f"{summary['cases_read_without_error']} of {summary['cases']} cases without "
        f"any error; {summary['tokens']} tokens in {summary['model_calls']} calls."
    )
    if not args.cases:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Report written to {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0


def recorded_model_name() -> str:
    """Return the model named in the existing recordings, if there are any."""
    for path in sorted(EXTRACTION_RECORDINGS.glob("*/*.json")):
        return json.loads(path.read_text(encoding="utf-8")).get("model", "")
    return ""


if __name__ == "__main__":
    sys.exit(main())
