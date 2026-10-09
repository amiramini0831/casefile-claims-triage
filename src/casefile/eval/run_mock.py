"""Run the 30 evaluation cases through the control plane with stand-in agents.

    python -m casefile.eval.run_mock

This checks the control plane, not the system. The stand-in extractor is given
the document contents instead of reading the PDFs, so a perfect score here
says that routing, the stopping guards and the typed handoffs work. It says
nothing about how a real model performs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from casefile.agents.mock import MockExtractor, MockInvestigator, MockReviewer
from casefile.config import RAW_DATA_DIR, REPORTS_DIR
from casefile.control.state import Limits
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME, RawDataError
from casefile.documents.generate import build_all_documents
from casefile.eval.cases import EvalCase, EvalSuite, load_cases
from casefile.eval.runner import AgentFactory, CaseRun, run_suite
from casefile.eval.select_cases import Portfolio, build_portfolio

REPORT_PATH = REPORTS_DIR / "mock_run_report.json"
# Stand-in calls cost one unit each, so this covers the 12 allowed steps.
MOCK_LIMITS = Limits(budget=Decimal(20))


def mock_agent_factory(suite: EvalSuite, portfolio: Portfolio) -> AgentFactory:
    """Return a function that builds the stand-in agents of a case."""
    documents = {item.case_id: item for item in build_all_documents(suite, portfolio)}

    def build(case: EvalCase) -> dict[str, Any]:
        return {
            "extractor": MockExtractor(documents[case.case_id]),
            "investigator": MockInvestigator(portfolio),
            "reviewer": MockReviewer(portfolio),
        }

    return build


def build_report(runs: Sequence[CaseRun]) -> dict[str, Any]:
    """Describe the runs, including the exact path each one took."""
    return {
        "agents": "stand-ins without a model",
        "measures": "the control plane only, not model performance",
        "cases": len(runs),
        "matched": sum(run.passed for run in runs),
        "runs": [
            {
                "case_id": run.case.case_id,
                "group": run.case.group,
                "fault_injection": run.case.fault_injection,
                "expected": run.case.expected.model_dump(mode="json"),
                "observed": run.observed.model_dump(mode="json"),
                "steps": run.state.steps_taken,
                "path": list(run.state.path),
                "matched": run.passed,
                "mismatches": list(run.mismatches),
            }
            for run in runs
        ],
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--oracle-path", type=Path, default=RAW_DATA_DIR / ORACLE_FILE_NAME
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cases and report. Returns 1 if any case ended unexpectedly."""
    args = parse_args(argv)
    try:
        suite = load_cases()
        portfolio = build_portfolio(args.claims_path, args.oracle_path)
    except (RawDataError, FileNotFoundError) as error:
        print(f"Cannot run the cases: {error}", file=sys.stderr)
        return 1

    runs = run_suite(suite, mock_agent_factory(suite, portfolio), MOCK_LIMITS)
    for run in runs:
        mark = "ok  " if run.passed else "FAIL"
        reason = run.observed.reason
        print(
            f"{mark} {run.case.case_id} {run.observed.outcome:<18} {reason:<31}"
            f"{run.state.steps_taken:>2} steps  {'; '.join(run.mismatches)}"
        )
    matched = sum(run.passed for run in runs)
    print(
        f"\n{matched} of {len(runs)} cases ended as pre-registered (stand-in agents)."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(build_report(runs), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Report written to {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0 if matched == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
