"""The supervisor: a state machine that routes a case and decides when to stop.

There is no model here. Routing is plain code, so a run is reproducible and
its termination can be proved instead of hoped for: every executed step adds
one entry to the trace, and no step is executed once ``max_steps`` entries
exist.

The guards are checked at two moments, and that order decides which reason is
reported when two limits are due at once:

1. When an agent reports back (``advance``): a second invalid output in a row
   ends the run with ``schema_errors``; one send-back too many for the same
   agent ends it with ``max_revisions``.
2. Before the next agent is called (``guard``): the step limit first, then
   the budget, using the worst-case cost of the call that is about to be made.

So a reviewer that never approves is reported as ``max_revisions`` even when
its last send-back also used up the final step.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol

from casefile.control.state import (
    Approved,
    CaseState,
    EscalationReason,
    Extracted,
    InvalidOutput,
    Limits,
    Node,
    Recommended,
    SentBack,
    StepEvent,
    TraceEntry,
)

_ALLOWED_EVENTS: dict[Node, tuple[type, ...]] = {
    "extractor": (Extracted, InvalidOutput),
    "investigator": (Recommended, InvalidOutput),
    "reviewer": (Approved, SentBack, InvalidOutput),
}


class AgentContractError(RuntimeError):
    """An agent spent more than the worst case it declared before the call."""


class Agent(Protocol):
    """What the supervisor needs from an agent."""

    def worst_case_cost(self, state: CaseState) -> Decimal:
        """Return the most the next call can cost, before it is made."""
        ...

    def run(self, state: CaseState) -> StepEvent:
        """Do one step of work and report what came of it."""
        ...


def start(case_id: str, limits: Limits) -> CaseState:
    """Return the state of a case that has not run yet."""
    return CaseState(case_id=case_id, limits=limits)


def guard(state: CaseState, worst_case_cost: Decimal) -> CaseState:
    """Stop the run if the next call would break the step limit or the budget."""
    _require_running(state)
    if state.steps_taken >= state.limits.max_steps:
        return _escalate(state, "max_steps")
    if state.budget_spent + worst_case_cost > state.limits.budget:
        return _escalate(state, "budget_exceeded")
    return state


def advance(state: CaseState, event: StepEvent) -> CaseState:
    """Record the step that ``state.next_node`` just took and route the case."""
    _require_running(state)
    node = state.next_node
    if not isinstance(event, _ALLOWED_EVENTS[node]):
        # An extractor that "approves" has not produced valid output.
        event = InvalidOutput(
            cost=event.cost, detail=f"{node} may not report {event.kind}"
        )

    entry = TraceEntry(
        node=node,
        event=event.kind,
        cost=event.cost,
        detail=getattr(event, "detail", "") or getattr(event, "reason", ""),
    )
    state = state.model_copy(
        update={
            "budget_spent": state.budget_spent + event.cost,
            "trace": (*state.trace, entry),
        }
    )

    if isinstance(event, InvalidOutput):
        errors = state.consecutive_schema_errors + 1
        state = state.model_copy(update={"consecutive_schema_errors": errors})
        if errors >= state.limits.max_consecutive_schema_errors:
            return _escalate(state, "schema_errors")
        return state  # the same agent tries again

    state = state.model_copy(update={"consecutive_schema_errors": 0})
    if isinstance(event, Extracted):
        return state.model_copy(
            update={
                "extraction": event.extraction,
                "feedback": None,
                "next_node": "investigator",
            }
        )
    if isinstance(event, Recommended):
        return state.model_copy(
            update={
                "recommendation": event.recommendation,
                "feedback": None,
                "next_node": "reviewer",
            }
        )
    if isinstance(event, Approved):
        return state.model_copy(update={"status": "completed", "next_node": None})
    if state.revisions.count(event.target) >= state.limits.max_revisions:
        return _escalate(state, "max_revisions")
    return state.model_copy(
        update={
            "revisions": state.revisions.increment(event.target),
            "feedback": event.reason,
            "next_node": event.target,
        }
    )


def step(state: CaseState, agent: Agent) -> CaseState:
    """Let ``agent`` take the next step, unless a guard stops the run first.

    An exception raised by the agent is not caught: it is a crash, not an
    outcome. The state passed in is still valid, so the run can be resumed
    from it.
    """
    worst_case = agent.worst_case_cost(state)
    state = guard(state, worst_case)
    if state.status != "running":
        return state
    event = agent.run(state)
    if event.cost > worst_case:
        raise AgentContractError(
            f"{state.next_node} spent {event.cost}, above its declared "
            f"worst case of {worst_case}"
        )
    return advance(state, event)


def run_case(state: CaseState, agents: Mapping[Node, Agent]) -> CaseState:
    """Run a case until it is completed or escalated."""
    while state.status == "running":
        state = step(state, agents[state.next_node])
    return state


def _require_running(state: CaseState) -> None:
    if state.status != "running":
        raise ValueError(f"case {state.case_id} has already ended ({state.status})")


def _escalate(state: CaseState, reason: EscalationReason) -> CaseState:
    return state.model_copy(
        update={"status": "escalated", "escalation_reason": reason, "next_node": None}
    )
