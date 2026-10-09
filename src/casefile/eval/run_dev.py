"""Try the investigator and reviewer on development cases, not on the evaluation.

    python -m casefile.eval.run_dev
    python -m casefile.eval.run_dev --mode replay

The prompts of the investigator and the reviewer have to be tried out and
adjusted somewhere. Doing that on the 30 pre-registered cases would tune the
system to the very cases it is later judged on. So a second set of 30 cases is
drawn here, by the same procedure and with another seed, from the claims that
are NOT in eval/cases.yaml. All prompt work happens on this development set;
the evaluation cases are run once, afterwards.

The extractor is not part of this: each case starts from a flawless reading of
its documents, so that what is measured is deciding and reviewing alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from casefile.agents.domain import SqlDomainLookup
from casefile.agents.investigator import LLMInvestigator
from casefile.agents.mock import extraction_from_documents
from casefile.agents.reviewer import LLMReviewer
from casefile.agents.triage import DomainLookup
from casefile.config import (
    RAW_DATA_DIR,
    RECORDINGS_DIR,
    REPORTS_DIR,
    ConfigurationError,
    load_llm_settings,
)
from casefile.control.state import Limits, Recommended
from casefile.control.supervisor import advance, start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME, RawDataError
from casefile.db import create_domain_engine
from casefile.documents.generate import build_all_documents
from casefile.eval.cases import EvalSuite, load_cases
from casefile.eval.select_cases import Portfolio, build_portfolio, select_cases
from casefile.llm.budget import BudgetedLLM
from casefile.llm.client import LLM, LLMError, OpenAICompatibleClient
from casefile.llm.recording import MissingRecordingError, RecordingLLM, RecordingMode
from casefile.rules import DecisionReason

REPORT_PATH = REPORTS_DIR / "dev_decisions_report.json"
DEV_RECORDINGS = RECORDINGS_DIR / "dev"
DEV_SEED = 7
# The data holds a single below-deductible claim, and the evaluation has it.
DEV_NATURAL_QUOTAS: tuple[tuple[DecisionReason, int], ...] = (
    ("amount_above_reference", 3),
    ("frequent_prior_claims", 3),
)
NO_LIMIT = Limits(budget=Decimal(10**9))


def build_dev_suite(portfolio: Portfolio, evaluation: EvalSuite) -> EvalSuite:
    """Draw development cases from the claims the evaluation does not use."""
    taken = {case.claim_id for case in evaluation.cases}
    remaining = Portfolio(
        policies=portfolio.policies,
        claims=tuple(c for c in portfolio.claims if c.claim_id not in taken),
        prior_claims=portfolio.prior_claims,
        reference=portfolio.reference,
    )
    return select_cases(remaining, seed=DEV_SEED, natural_quotas=DEV_NATURAL_QUOTAS)


def run_dev(
    mode: RecordingMode,
    model: LLM | None,
    model_name: str,
    lookup: DomainLookup,
    portfolio: Portfolio,
) -> dict[str, Any]:
    """Decide and review every development case and return the report."""
    suite = build_dev_suite(portfolio, load_cases())
    cases = []
    for case, documents in zip(
        suite.cases, build_all_documents(suite, portfolio), strict=True
    ):
        llm = BudgetedLLM(
            RecordingLLM(DEV_RECORDINGS / case.case_id, mode, model, model=model_name)
        )
        state = start(case.case_id, NO_LIMIT).model_copy(
            update={
                "extraction": extraction_from_documents(documents),
                "next_node": "investigator",
            }
        )
        investigated = LLMInvestigator(llm, lookup).run(state)
        entry: dict[str, Any] = {
            "case_id": case.case_id,
            "claim_id": case.claim_id,
            "document_defect": case.document_defect,
            "investigator": investigated.kind,
        }
        if isinstance(investigated, Recommended):
            recommendation = investigated.recommendation
            reviewed = LLMReviewer(llm, lookup).run(advance(state, investigated))
            entry |= {
                "rule_decision": recommendation.rule_decision.model_dump(mode="json"),
                "model_decision": recommendation.model_decision.model_dump(mode="json"),
                "outcomes_agree": recommendation.outcomes_agree,
                "decisions_agree": recommendation.decisions_agree,
                "rule_explanation": recommendation.rule_explanation,
                "model_explanation": recommendation.model_explanation,
                "reviewer": reviewed.kind,
                "reviewer_detail": getattr(reviewed, "reason", "")
                or getattr(reviewed, "detail", ""),
            }
        else:
            entry["detail"] = investigated.detail
        entry |= {"model_calls": llm.calls, "tokens": llm.tokens_used}
        cases.append(entry)
        agreement = (
            "agree"
            if entry.get("decisions_agree")
            else "outcome only"
            if entry.get("outcomes_agree")
            else "DISAGREE"
        )
        rule_reason = entry.get("rule_decision", {}).get("reason", "-")
        model_reason = entry.get("model_decision", {}).get("reason", "-")
        print(
            f"{case.case_id} rules={rule_reason:<31}model={model_reason:<31}"
            f"{agreement:<13}reviewer={entry.get('reviewer', '-')}",
            flush=True,
        )

    decided = [case for case in cases if case["investigator"] == "recommended"]
    return {
        "model": model_name,
        "measures": "development set; not the pre-registered evaluation",
        "seed": DEV_SEED,
        "summary": {
            "cases": len(cases),
            "investigator_invalid_outputs": len(cases) - len(decided),
            "same_outcome": sum(case["outcomes_agree"] for case in decided),
            "same_outcome_reason_and_amount": sum(
                case["decisions_agree"] for case in decided
            ),
            "reviewer_approved": sum(c["reviewer"] == "approved" for c in decided),
            "reviewer_sent_back": sum(c["reviewer"] == "sent_back" for c in decided),
            "reviewer_invalid_outputs": sum(
                c["reviewer"] == "invalid_output" for c in decided
            ),
            "model_calls": sum(case["model_calls"] for case in cases),
            "tokens": sum(case["tokens"] for case in cases),
        },
        "cases": cases,
    }


def recorded_model_name() -> str:
    """Return the model named in the development recordings, if there are any."""
    for path in sorted(DEV_RECORDINGS.glob("*/*.json")):
        return json.loads(path.read_text(encoding="utf-8")).get("model", "")
    return ""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("auto", "record", "replay"), default="auto")
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--oracle-path", type=Path, default=RAW_DATA_DIR / ORACLE_FILE_NAME
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the development cases and write the report. Returns an exit code."""
    args = parse_args(argv)
    client = None
    model_name = ""
    try:
        portfolio = build_portfolio(args.claims_path, args.oracle_path)
        if args.mode != "replay":
            settings = load_llm_settings()
            client = OpenAICompatibleClient(settings)
            model_name = settings.model
        if args.mode == "replay":
            # A replay needs no database: the same lookups are answered from
            # the raw files, and the requests come out as the same text.
            model_name = recorded_model_name()
            report = run_dev(args.mode, None, model_name, portfolio, portfolio)
        else:
            engine = create_domain_engine("agent")
            with engine.connect() as connection:
                lookup = SqlDomainLookup(connection)
                report = run_dev(args.mode, client, model_name, lookup, portfolio)
            engine.dispose()
    except (
        ConfigurationError,
        LLMError,
        MissingRecordingError,
        RawDataError,
        FileNotFoundError,
    ) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    summary = report["summary"]
    print(
        f"\nOf {summary['cases']} development cases the model reached the rules' "
        f"outcome in {summary['same_outcome']} and the same outcome, reason and "
        f"amount in {summary['same_outcome_reason_and_amount']}. The reviewer "
        f"approved {summary['reviewer_approved']} and sent back "
        f"{summary['reviewer_sent_back']}. {summary['tokens']} tokens in "
        f"{summary['model_calls']} calls."
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Report written to {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
