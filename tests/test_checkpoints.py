"""Running a case through LangGraph, with checkpoints, crashes and resumes."""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from crash_run import CASE_ID, CRASH_EXIT_CODE, LIMITS, clean_agents
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import Connection, text

from casefile.config import RAW_DATA_DIR, ConfigurationError, DatabaseSettings
from casefile.control.checkpoint import open_checkpointer
from casefile.control.graph import (
    build_graph,
    from_graph_state,
    run_graph,
    saved_state,
    to_graph_state,
)
from casefile.control.state import CaseState, StepEvent
from casefile.control.supervisor import Agent, run_case, start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME
from casefile.eval.cases import load_cases
from casefile.eval.faults import inject
from casefile.eval.run_mock import MOCK_LIMITS, mock_agent_factory
from casefile.eval.select_cases import build_portfolio
from casefile.gate import GateError, submit_for_approval

CLAIMS_PATH = RAW_DATA_DIR / RAW_FILE_NAME
ORACLE_PATH = RAW_DATA_DIR / ORACLE_FILE_NAME
needs_raw_files = pytest.mark.skipif(
    not (CLAIMS_PATH.exists() and ORACLE_PATH.exists()),
    reason="the raw datasets have not been downloaded",
)
CRASH_SCRIPT = Path(__file__).with_name("crash_run.py")
RUN = {"case_id": CASE_ID, "limits": LIMITS}


class CrashOnce:
    """Wrap an agent so that its first call raises, as a crash would."""

    def __init__(self, inner: Agent) -> None:
        self._inner = inner
        self.calls = 0

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return self._inner.worst_case_cost(state)

    def run(self, state: CaseState) -> StepEvent:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated crash")
        return self._inner.run(state)


class Counting:
    """Wrap an agent and count how often it is asked to work."""

    def __init__(self, inner: Agent) -> None:
        self._inner = inner
        self.calls = 0

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return self._inner.worst_case_cost(state)

    def run(self, state: CaseState) -> StepEvent:
        self.calls += 1
        return self._inner.run(state)


def plain_run() -> CaseState:
    """The ordinary case run by the plain loop, without LangGraph."""
    return run_case(start(CASE_ID, LIMITS), clean_agents())


def new_run_id() -> str:
    return f"test-{uuid.uuid4()}"


def test_graph_reaches_the_same_state_as_the_plain_loop() -> None:
    state = run_graph(build_graph(clean_agents()), run_id="r1", **RUN)

    assert state == plain_run()
    assert state.path == ("extractor", "investigator", "reviewer")


def test_state_survives_being_stored_as_plain_data() -> None:
    state = plain_run()

    assert from_graph_state(to_graph_state(state)) == state


@needs_raw_files
def test_all_thirty_cases_end_the_same_with_and_without_the_graph() -> None:
    suite = load_cases()
    build_agents = mock_agent_factory(suite, build_portfolio(CLAIMS_PATH, ORACLE_PATH))

    for case in suite.cases:
        agents, limits = inject(
            case.fault_injection, case.case_id, build_agents(case), MOCK_LIMITS
        )
        plain = run_case(start(case.case_id, limits), agents)
        graph = build_graph(agents, InMemorySaver())

        through_graph = run_graph(
            graph, run_id=case.case_id, case_id=case.case_id, limits=limits
        )

        assert through_graph == plain, case.case_id


def check_crash_and_resume(checkpointer: BaseCheckpointSaver, run_id: str) -> None:
    """A run that crashes inside an agent continues from the step before."""
    agents = clean_agents()
    extractor = agents["extractor"] = Counting(agents["extractor"])
    investigator = agents["investigator"] = CrashOnce(agents["investigator"])
    graph = build_graph(agents, checkpointer)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_graph(graph, run_id=run_id, **RUN)

    # The extractor's finished step was saved; the crashed step left no trace.
    saved = saved_state(graph, run_id, LIMITS)
    assert (saved.status, saved.next_node) == ("running", "investigator")
    assert saved.path == ("extractor",)
    assert saved.budget_spent == Decimal(1)

    resumed = run_graph(graph, run_id=run_id, **RUN)

    assert resumed == plain_run()
    assert (extractor.calls, investigator.calls) == (1, 2)

    # Asking again for a finished run does no work at all.
    assert run_graph(graph, run_id=run_id, **RUN) == resumed
    assert (extractor.calls, investigator.calls) == (1, 2)


def test_crashed_run_resumes_from_the_last_finished_step() -> None:
    check_crash_and_resume(InMemorySaver(), "r1")


def test_runs_do_not_share_checkpoints() -> None:
    graph = build_graph(clean_agents(), InMemorySaver())
    run_graph(graph, run_id="r1", **RUN)

    assert saved_state(graph, "r1", LIMITS).status == "completed"
    assert saved_state(graph, "r2", LIMITS) is None


def test_escalated_run_can_be_read_back_but_not_continued() -> None:
    agents, limits = inject("step_limit_too_small", CASE_ID, clean_agents(), LIMITS)
    extractor = agents["extractor"] = Counting(agents["extractor"])
    graph = build_graph(agents, InMemorySaver())

    first = run_graph(graph, run_id="r1", case_id=CASE_ID, limits=limits)
    again = run_graph(graph, run_id="r1", case_id=CASE_ID, limits=limits)

    assert (first.status, first.escalation_reason) == ("escalated", "max_steps")
    assert again == first
    assert extractor.calls == 1


# --- with the real MySQL checkpointer -------------------------------------


@pytest.fixture
def mysql_checkpointer(
    database_settings: DatabaseSettings,
) -> Iterator[BaseCheckpointSaver]:
    try:
        with open_checkpointer(database_settings) as checkpointer:
            yield checkpointer
    except ConfigurationError as error:
        pytest.skip(str(error))


def test_mysql_checkpointer_resumes_a_crashed_run(
    mysql_checkpointer: BaseCheckpointSaver,
) -> None:
    run_id = new_run_id()
    try:
        check_crash_and_resume(mysql_checkpointer, run_id)
    finally:
        mysql_checkpointer.delete_thread(run_id)


def test_killed_process_is_resumed_by_a_new_process(
    mysql_checkpointer: BaseCheckpointSaver,
) -> None:
    """The run's process is killed outright; another process finishes the run."""
    run_id = new_run_id()
    command = [sys.executable, str(CRASH_SCRIPT), run_id]
    try:
        killed = subprocess.run(  # noqa: S603 - our own script and interpreter
            [*command, "crash"], capture_output=True, text=True, timeout=120
        )
        assert killed.returncode == CRASH_EXIT_CODE, killed.stderr
        assert killed.stdout == ""

        graph = build_graph(clean_agents(), mysql_checkpointer)
        saved = saved_state(graph, run_id, LIMITS)
        assert (saved.status, saved.path) == ("running", ("extractor",))

        resumed = subprocess.run(  # noqa: S603 - our own script and interpreter
            [*command, "resume"], capture_output=True, text=True, timeout=120
        )
        assert resumed.returncode == 0, resumed.stderr
        # The extractor appears once: its work before the kill was not redone.
        assert resumed.stdout.split() == [
            "completed",
            "extractor",
            "investigator",
            "reviewer",
        ]
    finally:
        mysql_checkpointer.delete_thread(run_id)


# --- handing a finished run to the human gate -------------------------------


def test_completed_run_is_submitted_for_approval(
    loader_connection: Connection,
) -> None:
    claim = loader_connection.execute(
        text("SELECT claim_id FROM claims ORDER BY claim_id LIMIT 1")
    ).scalar_one_or_none()
    if claim is None:
        pytest.skip("the claims have not been loaded")
    state = plain_run()
    form = state.extraction.claim_form.model_copy(update={"claim_number": claim})
    state = state.model_copy(
        update={"extraction": state.extraction.model_copy(update={"claim_form": form})}
    )
    run_id = new_run_id()

    recommendation_id = submit_for_approval(
        loader_connection, run_id=run_id, state=state
    )
    again = submit_for_approval(loader_connection, run_id=run_id, state=state)

    assert recommendation_id == again
    row = loader_connection.execute(
        text(
            "SELECT case_id, claim_number, rule_outcome, rule_payable_amount "
            "FROM recommendations WHERE recommendation_id = :id"
        ),
        {"id": recommendation_id},
    ).one()
    assert tuple(row) == (CASE_ID, claim, "recommend_pay", Decimal("4000.00"))


def test_escalated_run_has_nothing_to_approve(loader_connection: Connection) -> None:
    agents, limits = inject("step_limit_too_small", CASE_ID, clean_agents(), LIMITS)
    escalated = run_case(start(CASE_ID, limits), agents)

    assert submit_for_approval(loader_connection, run_id="r1", state=escalated) is None


def test_unfinished_run_cannot_be_submitted(loader_connection: Connection) -> None:
    with pytest.raises(GateError, match="has not finished"):
        submit_for_approval(
            loader_connection, run_id="r1", state=start(CASE_ID, LIMITS)
        )
