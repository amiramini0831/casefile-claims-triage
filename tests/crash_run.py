"""Run one case with MySQL checkpoints in a process of its own.

Used by tests/test_checkpoints.py to kill a run for real and resume it in a
fresh process::

    python tests/crash_run.py <run id> crash    # dies inside the investigator
    python tests/crash_run.py <run id> resume   # continues from the checkpoint

On completion it prints the run's status and the path it took.
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal

from builders import (
    build_claim,
    build_eval_case,
    build_policy,
    build_reference,
)

from casefile.agents.mock import MockExtractor, MockInvestigator, MockReviewer
from casefile.control.checkpoint import open_checkpointer
from casefile.control.graph import build_graph, run_graph
from casefile.control.state import CaseState, Limits, StepEvent
from casefile.control.supervisor import Agent
from casefile.documents.content import build_case_documents

CRASH_EXIT_CODE = 17
LIMITS = Limits(budget=Decimal(20))
CASE_ID = "C01"


class Lookup:
    """One policy, no history: enough for an ordinary case."""

    def find_policy(self, policy_number: int):
        policy = build_policy()
        return policy if policy_number == policy.policy_number else None

    def find_prior_claims(self, policy_number: int):
        return ()

    def find_reference(self, incident_type: str, incident_severity: str):
        return build_reference()


class DieInside(MockInvestigator):
    """An investigator whose process is killed in the middle of its step."""

    def run(self, state: CaseState) -> StepEvent:
        # No exception, no cleanup, no chance to save anything: the same as
        # the machine losing power at this line.
        os._exit(CRASH_EXIT_CODE)


def clean_agents() -> dict[str, Agent]:
    """The stand-in agents of an ordinary case."""
    documents = build_case_documents(
        build_eval_case(), build_policy(), build_claim(), {1}
    )
    return {
        "extractor": MockExtractor(documents),
        "investigator": MockInvestigator(Lookup()),
        "reviewer": MockReviewer(Lookup()),
    }


def main(run_id: str, mode: str) -> int:
    agents = clean_agents()
    if mode == "crash":
        agents["investigator"] = DieInside(Lookup())
    with open_checkpointer() as checkpointer:
        graph = build_graph(agents, checkpointer)
        state = run_graph(graph, run_id=run_id, case_id=CASE_ID, limits=LIMITS)
    print(state.status, *state.path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
