"""The supervisor's routing and its stopping guards.

The first half pins down the behaviour with hand-written scenarios. The second
half lets hypothesis invent agents that behave in any way at all, including
nonsensically, and checks that the run still stops inside its limits.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal
from itertools import chain, repeat

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from casefile.control.handoff import ClaimFormFields, Extraction, Recommendation
from casefile.control.state import (
    MAX_REVISIONS,
    MAX_STEPS,
    Approved,
    CaseState,
    Extracted,
    InvalidOutput,
    Limits,
    Node,
    Recommended,
    SentBack,
    StepEvent,
)
from casefile.control.supervisor import (
    AgentContractError,
    advance,
    guard,
    run_case,
    start,
)
from casefile.rules import Decision

NODES: tuple[Node, ...] = ("extractor", "investigator", "reviewer")
LEGAL_TRANSITIONS = {
    ("extractor", "extractor"),  # retry after invalid output
    ("extractor", "investigator"),
    ("investigator", "investigator"),  # retry after invalid output
    ("investigator", "reviewer"),
    ("reviewer", "reviewer"),  # retry after invalid output
    ("reviewer", "extractor"),  # sent back
    ("reviewer", "investigator"),  # sent back
}
WORST_CASE = Decimal(3)

EXTRACTION = Extraction(
    claim_form=ClaimFormFields(
        claim_number="CLM-2015-00001",
        policy_number=1,
        incident_date=date(2015, 1, 25),
        incident_type="Vehicle Theft",
        incident_severity="Minor Damage",
        injury_claim=Decimal(0),
        property_claim=Decimal(1000),
        vehicle_claim=Decimal(4000),
        total_claim_amount=Decimal(5000),
    )
)
RECOMMENDATION = Recommendation(
    decision=Decision(outcome="recommend_deny", reason="below_deductible"),
    rationale="test",
)


def extracted(cost: int | Decimal = 0) -> Extracted:
    return Extracted(cost=Decimal(cost), extraction=EXTRACTION)


def recommended(cost: int | Decimal = 0) -> Recommended:
    return Recommended(cost=Decimal(cost), recommendation=RECOMMENDATION)


def send_back(target: str, reason: str = "not good enough") -> SentBack:
    return SentBack(target=target, reason=reason)


class ScriptedAgent:
    """An agent that reports whatever its script says, at a fixed worst case."""

    def __init__(self, events: Iterator[StepEvent], worst_case: Decimal) -> None:
        self._events = events
        self._worst_case = worst_case

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return self._worst_case

    def run(self, state: CaseState) -> StepEvent:
        return next(self._events)


class ConstantAgent:
    """An agent that always reports the same thing."""

    def __init__(self, event: StepEvent, worst_case: Decimal = Decimal(1)) -> None:
        self._event = event
        self._worst_case = worst_case

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return self._worst_case

    def run(self, state: CaseState) -> StepEvent:
        return self._event


class WorkerAgent:
    """An extractor or investigator that does its own job, sometimes badly.

    Each scripted step says whether the output is valid and what it cost.
    """

    def __init__(self, steps: Iterator[tuple[bool, Decimal]]) -> None:
        self._steps = steps

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return WORST_CASE

    def run(self, state: CaseState) -> StepEvent:
        valid, cost = next(self._steps)
        if not valid:
            return InvalidOutput(cost=cost)
        if state.next_node == "extractor":
            return extracted(cost)
        return recommended(cost)


def run_script(
    events: Sequence[StepEvent],
    limits: Limits | None = None,
    *,
    then_forever: StepEvent | None = None,
    worst_case: Decimal = Decimal(1),
) -> CaseState:
    """Run a case whose agents report ``events`` in order, whoever is asked.

    ``then_forever`` is what every agent keeps reporting once the script runs
    out, so a run can only end by completing or by hitting a guard.
    """
    tail = repeat(then_forever) if then_forever is not None else iter(())
    agent = ScriptedAgent(chain(events, tail), worst_case)
    limits = limits or Limits(budget=Decimal(1000))
    return run_case(start("C00", limits), dict.fromkeys(NODES, agent))


def test_defaults_are_twelve_steps_and_three_revisions() -> None:
    limits = Limits(budget=Decimal(10))

    assert (limits.max_steps, limits.max_revisions) == (12, 3)
    assert (MAX_STEPS, MAX_REVISIONS) == (12, 3)
    assert limits.max_consecutive_schema_errors == 2


def test_happy_path_visits_each_agent_once() -> None:
    state = run_script([extracted(), recommended(), Approved()])

    assert state.status == "completed"
    assert state.path == ("extractor", "investigator", "reviewer")
    assert state.next_node is None
    assert state.escalation_reason is None


def test_work_of_each_agent_is_kept_in_the_state() -> None:
    state = run_script([extracted(), recommended(), Approved()])

    assert state.extraction == EXTRACTION
    assert state.recommendation == RECOMMENDATION


def test_send_back_to_extractor_reruns_the_investigator_too() -> None:
    state = run_script(
        [extracted(), recommended(), send_back("extractor")]
        + [extracted(), recommended(), Approved()]
    )

    assert state.status == "completed"
    assert state.path == (*NODES, *NODES)
    assert state.revisions.extractor == 1


def test_send_back_to_investigator_skips_the_extractor() -> None:
    state = run_script(
        [extracted(), recommended(), send_back("investigator")]
        + [recommended(), Approved()]
    )

    assert state.status == "completed"
    assert state.path == (*NODES, "investigator", "reviewer")
    assert state.revisions.investigator == 1


def test_reviewer_feedback_reaches_the_agent_and_is_then_cleared() -> None:
    state = start("C00", Limits(budget=Decimal(10)))
    for event in (extracted(), recommended(), send_back("investigator", "redo it")):
        state = advance(state, event)

    assert (state.next_node, state.feedback) == ("investigator", "redo it")
    assert advance(state, recommended()).feedback is None


@pytest.mark.parametrize(("target", "steps"), [("investigator", 9), ("extractor", 12)])
def test_reviewer_that_never_approves_hits_the_revision_limit(
    target: str, steps: int
) -> None:
    """Three revisions are granted; the fourth request ends the run.

    Returning to the extractor uses exactly the twelve allowed steps, so the
    step limit is due at the same moment. The revision limit is the one
    reported.
    """
    agents = {
        "extractor": ConstantAgent(extracted()),
        "investigator": ConstantAgent(recommended()),
        "reviewer": ConstantAgent(send_back(target)),
    }

    state = run_case(start("C00", Limits(budget=Decimal(1000))), agents)

    assert (state.status, state.escalation_reason) == ("escalated", "max_revisions")
    assert state.steps_taken == steps
    assert state.revisions.count(target) == 3


def test_alternating_send_backs_hit_the_step_limit() -> None:
    """Each agent stays within its own revision limit; the total does not."""
    script = [extracted(), recommended(), send_back("extractor")] * 3 + [
        extracted(),
        recommended(),
        send_back("investigator"),
    ]

    state = run_script(script, then_forever=recommended())

    assert (state.status, state.escalation_reason) == ("escalated", "max_steps")
    assert state.steps_taken == 12
    assert state.revisions.extractor == 3
    assert state.revisions.investigator == 1


def test_one_invalid_output_is_retried() -> None:
    state = run_script([InvalidOutput(), extracted(), recommended(), Approved()])

    assert state.status == "completed"
    assert state.path == ("extractor", *NODES)
    assert state.consecutive_schema_errors == 0


def test_two_invalid_outputs_in_a_row_escalate() -> None:
    state = run_script([extracted(), InvalidOutput(), InvalidOutput()])

    assert (state.status, state.escalation_reason) == ("escalated", "schema_errors")
    assert state.path == ("extractor", "investigator", "investigator")


def test_invalid_outputs_must_be_consecutive_to_escalate() -> None:
    state = run_script(
        [InvalidOutput(), extracted(), InvalidOutput(), recommended(), Approved()]
    )

    assert state.status == "completed"


@pytest.mark.parametrize(
    ("event", "kind"), [(Approved(), "approved"), (recommended(), "recommended")]
)
def test_event_that_does_not_fit_the_agent_counts_as_invalid_output(
    event: StepEvent, kind: str
) -> None:
    state = run_script([event, event])

    assert (state.status, state.escalation_reason) == ("escalated", "schema_errors")
    assert state.trace[0].event == "invalid_output"
    assert f"extractor may not report {kind}" in state.trace[0].detail
    assert state.recommendation is None


def test_budget_is_checked_before_the_call_is_made() -> None:
    limits = Limits(budget=Decimal("2.5"))

    state = run_script([extracted(1), recommended(1), Approved()], limits)

    assert (state.status, state.escalation_reason) == ("escalated", "budget_exceeded")
    assert state.path == ("extractor", "investigator")
    assert state.budget_spent == Decimal(2)


def test_budget_too_small_for_the_first_call_runs_nothing() -> None:
    state = run_script([extracted()], Limits(budget=Decimal("0.5")))

    assert (state.status, state.escalation_reason) == ("escalated", "budget_exceeded")
    assert state.steps_taken == 0


def test_call_that_exactly_fits_the_budget_is_allowed() -> None:
    state = run_script(
        [extracted(1), recommended(1), Approved(cost=Decimal(1))],
        Limits(budget=Decimal(3)),
    )

    assert state.status == "completed"
    assert state.budget_spent == Decimal(3)


def test_small_step_limit_stops_a_normal_run() -> None:
    state = run_script(
        [extracted(), recommended(), Approved()],
        Limits(budget=Decimal(10), max_steps=2),
    )

    assert (state.status, state.escalation_reason) == ("escalated", "max_steps")
    assert state.path == ("extractor", "investigator")


def test_overspending_the_declared_worst_case_is_an_error() -> None:
    with pytest.raises(AgentContractError, match="above its declared worst case"):
        run_script([extracted(2)], worst_case=Decimal(1))


def test_a_finished_case_cannot_be_advanced() -> None:
    finished = run_script([extracted(), recommended(), Approved()])

    with pytest.raises(ValueError, match="already ended"):
        advance(finished, extracted())
    with pytest.raises(ValueError, match="already ended"):
        guard(finished, Decimal(0))


def test_advancing_returns_a_new_state() -> None:
    before = start("C00", Limits(budget=Decimal(10)))

    after = advance(before, extracted(1))

    assert before.steps_taken == 0
    assert before.next_node == "extractor"
    assert before.extraction is None
    assert (after.steps_taken, after.next_node) == (1, "investigator")


costs = st.decimals(min_value=0, max_value=WORST_CASE, places=2)
send_backs = st.builds(
    SentBack,
    cost=costs,
    target=st.sampled_from(["extractor", "investigator"]),
    reason=st.just("try again"),
)
events = st.one_of(
    st.builds(Extracted, cost=costs, extraction=st.just(EXTRACTION)),
    st.builds(Recommended, cost=costs, recommendation=st.just(RECOMMENDATION)),
    st.builds(Approved, cost=costs),
    send_backs,
    st.builds(InvalidOutput, cost=costs),
)
# Agents that mostly do their own job: the workers usually produce valid
# output, and the reviewer usually sends work back. Random events rarely line
# up into several revisions of the same agent, so this is what puts the
# revision limit under real pressure.
worker_steps = st.tuples(st.sampled_from([True, True, True, False]), costs)
reviewer_events = st.one_of(
    send_backs,
    send_backs,
    st.builds(Approved, cost=costs),
    st.builds(InvalidOutput, cost=costs),
)
all_limits = st.builds(
    Limits,
    budget=st.decimals(min_value=0, max_value=60, places=2),
    max_steps=st.integers(min_value=1, max_value=20),
    max_revisions=st.integers(min_value=0, max_value=5),
    max_consecutive_schema_errors=st.integers(min_value=1, max_value=4),
)


def assert_run_is_sound(state: CaseState, limits: Limits) -> None:
    """Check everything a finished run must satisfy, whatever the agents did."""
    # It stopped, in exactly one of the two final states.
    assert state.status in ("completed", "escalated")
    assert state.next_node is None
    assert (state.status == "escalated") == (state.escalation_reason is not None)

    # No guard was overrun.
    assert state.steps_taken <= limits.max_steps
    assert state.budget_spent <= limits.budget
    assert state.revisions.extractor <= limits.max_revisions
    assert state.revisions.investigator <= limits.max_revisions
    assert state.consecutive_schema_errors <= limits.max_consecutive_schema_errors

    # The route taken is one the state machine allows.
    assert not state.path or state.path[0] == "extractor"
    assert set(zip(state.path, state.path[1:], strict=False)) <= LEGAL_TRANSITIONS
    assert state.budget_spent == sum(entry.cost for entry in state.trace)

    # The reported ending matches what actually happened.
    if state.status == "completed":
        assert state.trace[-1].node == "reviewer"
        assert state.trace[-1].event == "approved"
        assert state.extraction is not None
        assert state.recommendation is not None
    elif state.escalation_reason == "max_steps":
        assert state.steps_taken == limits.max_steps
    elif state.escalation_reason == "budget_exceeded":
        assert state.budget_spent + WORST_CASE > limits.budget
    elif state.escalation_reason == "schema_errors":
        assert state.trace[-1].event == "invalid_output"
        assert state.consecutive_schema_errors == limits.max_consecutive_schema_errors
    else:
        assert state.escalation_reason == "max_revisions"
        assert state.trace[-1].event == "sent_back"
        assert limits.max_revisions in (
            state.revisions.extractor,
            state.revisions.investigator,
        )

    # The revision counters agree with the send-backs recorded in the trace;
    # the one that ended the run was refused and is not counted.
    recorded = sum(entry.event == "sent_back" for entry in state.trace)
    refused = state.escalation_reason == "max_revisions"
    counted = state.revisions.extractor + state.revisions.investigator
    assert counted == recorded - refused

    # The reviewer only ever sees work the investigator has just handed over.
    for index, node in enumerate(state.path):
        previous = state.trace[index - 1] if index else None
        if node == "reviewer":
            assert previous.node in ("investigator", "reviewer")
        if node == "investigator" and previous.node == "extractor":
            assert previous.event == "extracted"
        if node == "reviewer" and previous.node == "investigator":
            assert previous.event == "recommended"


@settings(max_examples=1000, deadline=None)
@given(script=st.lists(events, max_size=40), tail=events, limits=all_limits)
def test_any_agent_behaviour_ends_inside_the_limits(
    script: list[StepEvent], tail: StepEvent, limits: Limits
) -> None:
    """Whatever the agents report, forever, the run stops and stays bounded."""
    state = run_script(script, limits, then_forever=tail, worst_case=WORST_CASE)

    assert_run_is_sound(state, limits)


@settings(max_examples=1000, deadline=None)
@given(
    worker_script=st.lists(worker_steps, max_size=40),
    reviewer_script=st.lists(reviewer_events, max_size=20),
    worker_tail=worker_steps,
    reviewer_tail=reviewer_events,
    limits=all_limits,
)
def test_plausible_agents_end_inside_the_limits(
    worker_script: list[tuple[bool, Decimal]],
    reviewer_script: list[StepEvent],
    worker_tail: tuple[bool, Decimal],
    reviewer_tail: StepEvent,
    limits: Limits,
) -> None:
    """Agents that do their own job, badly and forever, are still stopped."""
    workers = WorkerAgent(chain(worker_script, repeat(worker_tail)))
    reviewer = ScriptedAgent(chain(reviewer_script, repeat(reviewer_tail)), WORST_CASE)
    agents = {"extractor": workers, "investigator": workers, "reviewer": reviewer}

    state = run_case(start("C00", limits), agents)

    assert_run_is_sound(state, limits)


@settings(max_examples=300, deadline=None)
@given(script=st.lists(events, max_size=40), tail=events, limits=all_limits)
def test_a_run_is_reproducible(
    script: list[StepEvent], tail: StepEvent, limits: Limits
) -> None:
    first = run_script(script, limits, then_forever=tail, worst_case=WORST_CASE)
    second = run_script(script, limits, then_forever=tail, worst_case=WORST_CASE)

    assert first == second
