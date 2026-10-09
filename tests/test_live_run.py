"""The live runner: narration must not change a run, and a replay must match."""

from __future__ import annotations

import json

import pytest
from crash_run import CASE_ID, LIMITS, clean_agents

from casefile.config import RAW_DATA_DIR, RECORDINGS_DIR
from casefile.control.state import Approved, InvalidOutput, SentBack
from casefile.control.supervisor import run_case, start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME
from casefile.eval.faults import inject
from casefile.eval.run_live import (
    REPORT_PATH,
    Narrated,
    build_report,
    details,
    headline,
    run_live,
)
from casefile.eval.select_cases import build_portfolio


def narrated(agents):
    return {name: Narrated(agent, name) for name, agent in agents.items()}


def test_narration_does_not_change_the_run(capsys: pytest.CaptureFixture[str]) -> None:
    plain = run_case(start(CASE_ID, LIMITS), clean_agents())

    told = run_case(start(CASE_ID, LIMITS), narrated(clean_agents()))

    assert told == plain
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[1] for line in lines if line.startswith("  [")] == [
        "extractor",
        "investigator",
        "reviewer",
    ]
    assert "read: policy 1 | incident 2015-01-25" in lines[0]
    assert any("rules: recommend_pay (covered, pay 4000)" in line for line in lines)


def test_injected_fault_is_narrated_as_the_supervisor_sees_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents, limits = inject(
        "extractor_error_first_attempt", CASE_ID, clean_agents(), LIMITS
    )

    state = run_case(start(CASE_ID, limits), narrated(agents))

    output = capsys.readouterr().out
    assert state.status == "completed"
    assert "(vehicle 400)" in output  # the misread amount, before the send-back
    assert "SENT BACK to the extractor" in output
    assert "(vehicle 4000)" in output


def test_every_kind_of_step_has_a_headline() -> None:
    sent_back = SentBack(target="investigator", reason="An invented date.")

    assert headline(Approved()) == "approved"
    assert headline(sent_back) == "SENT BACK to the investigator"
    assert details(sent_back) == ["reason: An invented date."]
    assert headline(InvalidOutput(detail="not JSON")) == "INVALID OUTPUT: not JSON"
    assert details(Approved()) == []


def test_report_counts_agreement_only_among_completed_cases() -> None:
    completed = {
        "status": "completed",
        "matched": True,
        "outcomes_agree": True,
        "decisions_agree": False,
        "events": ["extracted", "recommended", "sent_back", "recommended", "approved"],
        "tokens": 900,
        "model_calls": 7,
    }
    escalated = {
        "status": "escalated",
        "matched": False,
        "events": ["extracted", "invalid_output", "invalid_output"],
        "tokens": 300,
        "model_calls": 2,
    }

    summary = build_report([completed, escalated], "test/model")["summary"]

    assert summary == {
        "cases": 2,
        "stopped": 2,
        "ended_as_pre_registered": 1,
        "completed": 1,
        "escalated": 1,
        "model_reached_the_rules_outcome": 1,
        "model_matched_outcome_reason_and_amount": 0,
        "cases_with_a_reviewer_send_back": 1,
        "invalid_outputs": 2,
        "tokens": 1200,
        "most_tokens_in_one_case": 900,
        "model_calls": 9,
    }


@pytest.mark.skipif(
    not (REPORT_PATH.exists() and (RECORDINGS_DIR / "eval").exists()),
    reason="the evaluation has not been run with the real agents yet",
)
def test_recorded_evaluation_reproduces_its_report_offline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole evaluation again: no model, no key, no network, no database."""
    claims, oracle = RAW_DATA_DIR / RAW_FILE_NAME, RAW_DATA_DIR / ORACLE_FILE_NAME
    if not (claims.exists() and oracle.exists()):
        pytest.skip("the raw datasets have not been downloaded")
    stored = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    replayed = run_live(
        "eval", "replay", None, stored["model"], build_portfolio(claims, oracle)
    )

    assert replayed == stored
    assert replayed["summary"]["stopped"] == 30
