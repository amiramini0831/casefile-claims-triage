"""Run evaluation cases and compare where they end with what was pre-registered."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from casefile.control.state import CaseState, Limits, Node
from casefile.control.supervisor import Agent, run_case, start
from casefile.eval.cases import EvalCase, EvalSuite, ExpectedResult
from casefile.eval.faults import inject

AgentFactory = Callable[[EvalCase], Mapping[Node, Agent]]


@dataclass(frozen=True)
class CaseRun:
    """One executed case: where it ended and whether that was expected."""

    case: EvalCase
    state: CaseState
    observed: ExpectedResult
    mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches


def observed_result(state: CaseState) -> ExpectedResult:
    """Describe a finished run in the terms the cases are written in."""
    sent_back = any(entry.event == "sent_back" for entry in state.trace)
    if state.status == "escalated":
        return ExpectedResult(
            outcome="escalated",
            reason=state.escalation_reason,
            reviewer_send_back=sent_back,
        )
    if state.status != "completed" or state.recommendation is None:
        raise ValueError(f"case {state.case_id} has not finished")
    decision = state.recommendation.decision
    return ExpectedResult(
        outcome=decision.outcome,
        reason=decision.reason,
        payable_amount=decision.payable_amount,
        reviewer_send_back=sent_back,
    )


def compare(expected: ExpectedResult, observed: ExpectedResult) -> tuple[str, ...]:
    """List the ways ``observed`` falls short of ``expected``.

    Outcome, reason and amount must match exactly. ``reviewer_send_back`` is a
    requirement only when it is true: a reviewer may always return work, but
    where the case demands it, it must have happened.
    """
    mismatches = [
        f"{name}: expected {wanted}, got {found}"
        for name in ("outcome", "reason", "payable_amount")
        if (wanted := getattr(expected, name)) != (found := getattr(observed, name))
    ]
    if expected.reviewer_send_back and not observed.reviewer_send_back:
        mismatches.append("the reviewer never sent the work back")
    return tuple(mismatches)


def run_eval_case(
    case: EvalCase, agents: Mapping[Node, Agent], limits: Limits
) -> CaseRun:
    """Run one case, with its fault injected if it has one."""
    agents, limits = inject(case.fault_injection, case.case_id, agents, limits)
    state = run_case(start(case.case_id, limits), agents)
    observed = observed_result(state)
    return CaseRun(case, state, observed, compare(case.expected, observed))


def run_suite(
    suite: EvalSuite, build_agents: AgentFactory, limits: Limits
) -> list[CaseRun]:
    """Run every case of ``suite`` with freshly built agents."""
    return [run_eval_case(case, build_agents(case), limits) for case in suite.cases]
