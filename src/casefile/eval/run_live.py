"""Run the evaluation cases with the real agents, and show each step as it happens.

    python -m casefile.eval.run_live                  # the 30 pre-registered cases
    python -m casefile.eval.run_live --mode replay    # the same run, from recordings
    python -m casefile.eval.run_live --cases C15 C27  # some cases only
    python -m casefile.eval.run_live --suite dev --cases C01   # a development case

Every case goes the whole way: the extractor reads the PDFs, the investigator
produces the rules' decision and the model's own, the reviewer checks the
work, and a finished case is put in front of a person at the human gate.
Faults named in eval/cases.yaml are injected. Each step is printed as soon as
it is done.

In a live run the state of each case is checkpointed in MySQL after every
step and all model answers are recorded under recordings/. If the run is
interrupted, run the same command again: finished cases are read back and the
interrupted one continues from its last finished step.

``--mode replay`` repeats a recorded run exactly, without the model, an API
key, a network connection or a database, and produces the same report.

The development suite exists for trying things out; it uses separate claims
and never touches the checkpoints, the gate or the evaluation report.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from casefile.agents.domain import SqlDomainLookup
from casefile.agents.extractor import LLMExtractor
from casefile.agents.investigator import LLMInvestigator
from casefile.agents.reviewer import LLMReviewer
from casefile.agents.triage import DomainLookup
from casefile.config import (
    RAW_DATA_DIR,
    RECORDINGS_DIR,
    REPORTS_DIR,
    ConfigurationError,
    load_llm_settings,
)
from casefile.control.checkpoint import open_checkpointer
from casefile.control.graph import build_graph, run_graph, saved_state
from casefile.control.state import (
    CaseState,
    Extracted,
    Limits,
    Node,
    Recommended,
    SentBack,
    StepEvent,
)
from casefile.control.supervisor import Agent, run_case, start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME, RawDataError
from casefile.db import create_domain_engine
from casefile.documents.generate import (
    CASES_DIR,
    build_all_documents,
    write_case_folders,
)
from casefile.eval.cases import EvalCase, EvalSuite, load_cases
from casefile.eval.faults import inject
from casefile.eval.run_dev import build_dev_suite
from casefile.eval.runner import compare, observed_result
from casefile.eval.select_cases import Portfolio, build_portfolio
from casefile.gate import submit_for_approval
from casefile.llm.budget import BudgetedLLM
from casefile.llm.client import LLM, LLMError, OpenAICompatibleClient
from casefile.llm.recording import MissingRecordingError, RecordingLLM, RecordingMode

REPORT_PATH = REPORTS_DIR / "live_run_report.json"
# A ceiling, not a target: far above what a case needs, so that only a run
# that has gone wrong can reach it. Case C29 tightens it on purpose.
TOKEN_CEILING = Decimal(200_000)
LIMITS = Limits(budget=TOKEN_CEILING)


def say(text: str = "") -> None:
    """Print one line at once, so the run can be followed while it happens."""
    print(text, flush=True)


class Narrated:
    """Wrap an agent and describe each of its steps as soon as it is done.

    It changes nothing: the event it returns is the one the agent returned.
    """

    def __init__(self, inner: Agent, name: Node) -> None:
        self._inner = inner
        self._name = name

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return self._inner.worst_case_cost(state)

    def run(self, state: CaseState) -> StepEvent:
        started = time.perf_counter()
        event = self._inner.run(state)
        seconds = time.perf_counter() - started
        say(
            f"  [{state.steps_taken + 1}] {self._name:<13}{seconds:5.1f}s "
            f"{int(event.cost):>6} tok  {headline(event)}"
        )
        for line in details(event):
            say(f"        {line}")
        return event


def headline(event: StepEvent) -> str:
    """Sum up one step in a line."""
    if isinstance(event, Extracted):
        form = event.extraction.claim_form
        estimate = event.extraction.repair_estimate
        report = event.extraction.police_report
        return (
            f"read: policy {form.policy_number} | incident {form.incident_date} | "
            f"claimed {form.total_claim_amount} (vehicle {form.vehicle_claim}) | "
            f"estimate {estimate.total_amount if estimate else 'MISSING'} | "
            f"police report {report.incident_date if report else 'none'}"
        )
    if isinstance(event, Recommended):
        agree = "agree" if event.recommendation.decisions_agree else "DISAGREE"
        return f"two decisions, which {agree}"
    if isinstance(event, SentBack):
        return f"SENT BACK to the {event.target}"
    if event.kind == "approved":
        return "approved"
    return f"INVALID OUTPUT: {event.detail}"


def details(event: StepEvent) -> list[str]:
    """Return the lines that explain a step, if it needs explaining."""
    if isinstance(event, Extracted):
        issues = event.extraction.document_issues()
        if not issues:
            return []
        return [f"documents disagree or are incomplete: {', '.join(issues)}"]
    if isinstance(event, Recommended):
        recommendation = event.recommendation
        return [
            f"rules: {describe(recommendation.rule_decision)}",
            f"       {recommendation.rule_explanation}",
            f"model: {describe(recommendation.model_decision)}",
            f"       {recommendation.model_explanation}",
        ]
    if isinstance(event, SentBack):
        return [f"reason: {event.reason}"]
    return []


def describe(decision: Any) -> str:
    amount = f", pay {decision.payable_amount}" if decision.payable_amount else ""
    return f"{decision.outcome} ({decision.reason}{amount})"


def build_agents(
    case: EvalCase, llm: BudgetedLLM, lookup: DomainLookup, cases_dir: Path
) -> tuple[dict[Node, Agent], Limits]:
    """Build the real agents of a case, inject its fault and narrate them."""
    agents: Mapping[Node, Agent] = {
        "extractor": LLMExtractor(llm, cases_dir / case.case_id),
        "investigator": LLMInvestigator(llm, lookup),
        "reviewer": LLMReviewer(llm, lookup),
    }
    agents, limits = inject(case.fault_injection, case.case_id, agents, LIMITS)
    # Narration goes outermost, so it shows what the supervisor is given,
    # injected faults included.
    return {name: Narrated(agent, name) for name, agent in agents.items()}, limits


def case_entry(case: EvalCase, state: CaseState, recordings: Path) -> dict[str, Any]:
    """Describe how a case ended, in terms that do not change on replay.

    Model calls are counted from the recordings rather than from this
    process, so a case that was read back from a checkpoint reports the calls
    it really took.
    """
    observed = observed_result(state)
    mismatches = compare(case.expected, observed)
    entry: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group,
        "document_defect": case.document_defect,
        "fault_injection": case.fault_injection,
        "expected": case.expected.model_dump(mode="json"),
        "observed": observed.model_dump(mode="json"),
        "matched": not mismatches,
        "mismatches": list(mismatches),
        "status": state.status,
        "steps": state.steps_taken,
        "path": list(state.path),
        "events": [step.event for step in state.trace],
        "tokens": int(state.budget_spent),
        "model_calls": len(list(recordings.glob("*.json"))),
    }
    if state.status == "completed" and state.recommendation is not None:
        recommendation = state.recommendation
        entry |= {
            "rule_decision": recommendation.rule_decision.model_dump(mode="json"),
            "model_decision": recommendation.model_decision.model_dump(mode="json"),
            "outcomes_agree": recommendation.outcomes_agree,
            "decisions_agree": recommendation.decisions_agree,
            "rule_explanation": recommendation.rule_explanation,
            "model_explanation": recommendation.model_explanation,
        }
    return entry


def build_report(entries: Sequence[dict[str, Any]], model_name: str) -> dict[str, Any]:
    """Summarise the run."""
    completed = [entry for entry in entries if entry["status"] == "completed"]
    return {
        "model": model_name,
        "agents": "real: extractor, investigator and reviewer backed by the model",
        "token_ceiling_per_case": int(TOKEN_CEILING),
        "summary": {
            "cases": len(entries),
            "stopped": sum(e["status"] in ("completed", "escalated") for e in entries),
            "ended_as_pre_registered": sum(entry["matched"] for entry in entries),
            "completed": len(completed),
            "escalated": len(entries) - len(completed),
            "model_reached_the_rules_outcome": sum(
                entry["outcomes_agree"] for entry in completed
            ),
            "model_matched_outcome_reason_and_amount": sum(
                entry["decisions_agree"] for entry in completed
            ),
            "cases_with_a_reviewer_send_back": sum(
                "sent_back" in entry["events"] for entry in entries
            ),
            "invalid_outputs": sum(
                entry["events"].count("invalid_output") for entry in entries
            ),
            "tokens": sum(entry["tokens"] for entry in entries),
            "most_tokens_in_one_case": max(entry["tokens"] for entry in entries),
            "model_calls": sum(entry["model_calls"] for entry in entries),
        },
        "cases": list(entries),
    }


@contextmanager
def dev_case_folders(suite: EvalSuite, portfolio: Portfolio) -> Iterator[Path]:
    """Write the PDFs of the development cases to a temporary folder."""
    with tempfile.TemporaryDirectory() as folder:
        write_case_folders(build_all_documents(suite, portfolio), Path(folder))
        yield Path(folder)


def run_live(
    suite_name: str,
    mode: RecordingMode,
    model: LLM | None,
    model_name: str,
    portfolio: Portfolio,
    only: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the cases and return the report.

    Only a live run of the evaluation suite uses MySQL: for the agents'
    lookups, the checkpoints and the human gate. Replays and development runs
    look things up in memory and keep nothing.
    """
    evaluation = load_cases()
    is_eval = suite_name == "eval"
    suite = evaluation if is_eval else build_dev_suite(portfolio, evaluation)
    persistent = is_eval and mode != "replay"
    recordings = RECORDINGS_DIR / ("eval" if is_eval else "dev_live")
    entries = []

    with ExitStack() as stack:
        cases_dir = (
            CASES_DIR
            if is_eval
            else stack.enter_context(dev_case_folders(suite, portfolio))
        )
        lookup: DomainLookup = portfolio
        checkpointer = writer = None
        if persistent:
            agent_engine = create_domain_engine("agent")
            lookup = SqlDomainLookup(stack.enter_context(agent_engine.connect()))
            checkpointer = stack.enter_context(open_checkpointer())
            writer = create_domain_engine("writer")

        for case in suite.cases:
            if only and case.case_id not in only:
                continue
            what = case.document_defect or case.fault_injection or case.group
            say(f"\n== {case.case_id}  {what}  ({case.note})")
            llm = BudgetedLLM(
                RecordingLLM(recordings / case.case_id, mode, model, model=model_name)
            )
            agents, limits = build_agents(case, llm, lookup, cases_dir)
            run_id = f"{suite_name}-{case.case_id}"

            if checkpointer is None:
                state = run_case(start(case.case_id, limits), agents)
            else:
                graph = build_graph(agents, checkpointer)
                earlier = saved_state(graph, run_id, limits)
                if earlier is not None:
                    say(f"  (read from a checkpoint: {earlier.steps_taken} steps done)")
                state = run_graph(
                    graph, run_id=run_id, case_id=case.case_id, limits=limits
                )
            if writer is not None:
                with writer.begin() as connection:
                    recommendation_id = submit_for_approval(
                        connection, run_id=run_id, state=state
                    )
                if recommendation_id is not None:
                    say(f"  waiting for a person: recommendation {recommendation_id}")

            entry = case_entry(case, state, recordings / case.case_id)
            entries.append(entry)
            observed = entry["observed"]
            verdict = "MATCH" if entry["matched"] else "DIFFERENT"
            say(
                f"  => {observed['outcome']} / {observed['reason']} after "
                f"{state.steps_taken} steps, {entry['tokens']} tokens. "
                f"Pre-registered: {case.expected.outcome} / {case.expected.reason}. "
                f"{verdict}"
            )
            for mismatch in entry["mismatches"]:
                say(f"     {mismatch}")
    return build_report(entries, model_name)


def recorded_model_name() -> str:
    """Return the model named in the evaluation recordings, if there are any."""
    for path in sorted((RECORDINGS_DIR / "eval").glob("*/*.json")):
        return json.loads(path.read_text(encoding="utf-8")).get("model", "")
    return ""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=("eval", "dev"), default="eval")
    parser.add_argument("--mode", choices=("auto", "record", "replay"), default="auto")
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
    """Run the cases, narrating as they go. Returns an exit code."""
    args = parse_args(argv)
    # A model may answer with characters the Windows console cannot show.
    sys.stdout.reconfigure(errors="replace")
    client = None
    model_name = recorded_model_name()
    try:
        portfolio = build_portfolio(args.claims_path, args.oracle_path)
        if args.mode != "replay":
            settings = load_llm_settings()
            client = OpenAICompatibleClient(settings)
            model_name = settings.model
        say(f"Model: {model_name} | suite: {args.suite} | mode: {args.mode}")
        report = run_live(
            args.suite, args.mode, client, model_name, portfolio, args.cases
        )
    except (
        ConfigurationError,
        LLMError,
        MissingRecordingError,
        RawDataError,
        FileNotFoundError,
    ) as error:
        say(f"\nStopped: {error}")
        say("Nothing is lost. Run the same command again to continue.")
        return 1
    finally:
        if client is not None:
            client.close()

    summary = report["summary"]
    say("\n" + "=" * 72)
    agreed = summary["model_reached_the_rules_outcome"]
    say(f"Cases run:                        {summary['cases']}")
    say(f"Stopped on their own:             {summary['stopped']}")
    say(f"Ended as pre-registered:          {summary['ended_as_pre_registered']}")
    say(
        f"Model reached the rules' outcome: {agreed} of {summary['completed']} "
        "completed cases"
    )
    sent_back = summary["cases_with_a_reviewer_send_back"]
    say(f"Cases with a reviewer send-back:  {sent_back}")
    say(
        f"Tokens: {summary['tokens']} in {summary['model_calls']} model calls; "
        f"most in one case {summary['most_tokens_in_one_case']} "
        f"(ceiling {int(TOKEN_CEILING)})"
    )
    if args.suite == "eval" and not args.cases:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        say(f"Report written to {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
