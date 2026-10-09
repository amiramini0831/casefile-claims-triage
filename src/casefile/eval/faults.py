"""Deliberate faults, injected into a run to prove the guards work.

Each fault named in eval/cases.yaml is implemented here as a wrapper around an
agent or as a tightened limit. The wrappers do not care whether the agent
inside is a stand-in or a real one, so the same faults can be injected once
the real agents exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from casefile.control.state import (
    CaseState,
    Extracted,
    InvalidOutput,
    Limits,
    Node,
    SentBack,
    StepEvent,
)
from casefile.control.supervisor import Agent, start
from casefile.eval.cases import InjectionType

STEP_LIMIT_TOO_SMALL = 2


class _Wrapper:
    def __init__(self, inner: Agent) -> None:
        self._inner = inner

    def worst_case_cost(self, state: CaseState) -> Decimal:
        return self._inner.worst_case_cost(state)


class CorruptFirstExtraction(_Wrapper):
    """Make the extractor misread the vehicle amount once, then behave.

    The last digit is dropped, as a careless reading would, so the claimed
    parts no longer add up to the claimed total.

    Whether this is the first extraction is read from the case's trace, not
    remembered here, so a run that is interrupted and resumed in another
    process behaves exactly like one that was not.
    """

    def run(self, state: CaseState) -> StepEvent:
        event = self._inner.run(state)
        already_extracted = any(entry.event == "extracted" for entry in state.trace)
        if already_extracted or not isinstance(event, Extracted):
            return event
        form = event.extraction.claim_form
        misread = form.model_copy(update={"vehicle_claim": form.vehicle_claim // 10})
        extraction = event.extraction.model_copy(update={"claim_form": misread})
        return event.model_copy(update={"extraction": extraction})


class NeverApprove(_Wrapper):
    """A reviewer that sends the work back whatever it is shown."""

    def run(self, state: CaseState) -> StepEvent:
        return SentBack(target="investigator", reason="Injected: never satisfied.")


class AlwaysInvalid(_Wrapper):
    """An agent whose output never matches its schema."""

    def run(self, state: CaseState) -> StepEvent:
        return InvalidOutput(detail="Injected: output does not match the schema.")


def inject(
    injection: InjectionType | None,
    case_id: str,
    agents: Mapping[Node, Agent],
    limits: Limits,
) -> tuple[dict[Node, Agent], Limits]:
    """Return the agents and limits of a run with ``injection`` applied."""
    agents = dict(agents)
    if injection == "extractor_error_first_attempt":
        agents["extractor"] = CorruptFirstExtraction(agents["extractor"])
    elif injection == "reviewer_never_approves":
        agents["reviewer"] = NeverApprove(agents["reviewer"])
    elif injection == "repeated_schema_errors":
        agents["investigator"] = AlwaysInvalid(agents["investigator"])
    elif injection == "budget_too_small":
        first_call = agents["extractor"].worst_case_cost(start(case_id, limits))
        if first_call <= 0:
            raise ValueError("a free first call cannot exceed any budget")
        limits = limits.model_copy(update={"budget": first_call / 2})
    elif injection == "step_limit_too_small":
        limits = limits.model_copy(update={"max_steps": STEP_LIMIT_TOO_SMALL})
    return agents, limits
