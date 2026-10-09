"""Print the evaluation results as Markdown, for the README.

    python -m casefile.eval.summary

Reads reports/live_run_report.json. The tables in the README are this
script's output, so they are never typed by hand.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from casefile.config import REPORTS_DIR

REPORT_PATH = REPORTS_DIR / "live_run_report.json"


def describe(result: dict[str, Any]) -> str:
    """Write an expected or observed result on one line."""
    text = f"{result['outcome']} / {result['reason']}"
    if result.get("payable_amount"):
        text += f" / {result['payable_amount']}"
    return text


def model_column(case: dict[str, Any]) -> str:
    """Say how the model's own decision compared with the rules' decision."""
    if case["status"] != "completed":
        return "no decision (run stopped)"
    if case["decisions_agree"]:
        return "same"
    return f"differs: {describe(case['model_decision'])}"


def case_table(report: dict[str, Any]) -> str:
    """Return one row per case."""
    lines = [
        "| Case | Kind | Pre-registered | Observed | Match | Model vs rules "
        "| Steps | Tokens |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for case in report["cases"]:
        kind = case["document_defect"] or case["fault_injection"] or case["group"]
        lines.append(
            f"| {case['case_id']} | {kind} | {describe(case['expected'])} "
            f"| {describe(case['observed'])} | {'yes' if case['matched'] else 'NO'} "
            f"| {model_column(case)} | {case['steps']} | {case['tokens']:,} |"
        )
    return "\n".join(lines)


def summary_table(report: dict[str, Any]) -> str:
    """Return the headline numbers."""
    summary = report["summary"]
    completed = summary["completed"]
    rows = [
        ("Cases", summary["cases"]),
        ("Stopped on their own", f"{summary['stopped']} of {summary['cases']}"),
        (
            "Ended as pre-registered",
            f"{summary['ended_as_pre_registered']} of {summary['cases']}",
        ),
        (
            "Model reached the rules' outcome",
            f"{summary['model_reached_the_rules_outcome']} of {completed} completed",
        ),
        (
            "Model matched outcome, reason and amount",
            f"{summary['model_matched_outcome_reason_and_amount']} of {completed} "
            "completed",
        ),
        (
            "Cases in which the reviewer sent work back",
            summary["cases_with_a_reviewer_send_back"],
        ),
        ("Invalid outputs, injected ones included", summary["invalid_outputs"]),
        ("Model calls", summary["model_calls"]),
        ("Tokens, all cases", f"{summary['tokens']:,}"),
        ("Tokens, largest case", f"{summary['most_tokens_in_one_case']:,}"),
        ("Token ceiling per case", f"{report['token_ceiling_per_case']:,}"),
    ]
    lines = ["| Measure | Result |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def main() -> int:
    """Print both tables. Returns an exit code."""
    if not REPORT_PATH.exists():
        print(
            f"{REPORT_PATH} does not exist. Run casefile.eval.run_live first.",
            file=sys.stderr,
        )
        return 1
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"Model: {report['model']}\n")
    print(summary_table(report))
    print()
    print(case_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
