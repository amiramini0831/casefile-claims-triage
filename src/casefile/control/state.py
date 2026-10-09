"""The state of one case as it moves between the agents.

Everything here is immutable. The supervisor never changes a state; it returns
a new one, which is what makes a run easy to checkpoint, replay and test.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from casefile.control.handoff import Extraction, Recommendation

Node = Literal["extractor", "investigator", "reviewer"]
# The reviewer can return work to either of the agents before it.
RevisionTarget = Literal["extractor", "investigator"]
RunStatus = Literal["running", "completed", "escalated"]
EscalationReason = Literal[
    "max_revisions", "schema_errors", "budget_exceeded", "max_steps"
]

MAX_STEPS = 12
MAX_REVISIONS = 3
MAX_CONSECUTIVE_SCHEMA_ERRORS = 2


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Limits(_Frozen):
    """The stopping guards of one run.

    ``budget`` has no default on purpose: it depends on the price of the
    model in use and must be chosen by whoever starts the run.
    """

    budget: Decimal = Field(ge=0)
    max_steps: int = Field(default=MAX_STEPS, ge=1)
    max_revisions: int = Field(default=MAX_REVISIONS, ge=0)
    max_consecutive_schema_errors: int = Field(
        default=MAX_CONSECUTIVE_SCHEMA_ERRORS, ge=1
    )


class _Event(_Frozen):
    """What an agent reports back after one step."""

    cost: Decimal = Field(default=Decimal(0), ge=0)


class Extracted(_Event):
    """The extractor read the documents."""

    kind: Literal["extracted"] = "extracted"
    extraction: Extraction


class Recommended(_Event):
    """The investigator drafted a recommendation."""

    kind: Literal["recommended"] = "recommended"
    recommendation: Recommendation


class Approved(_Event):
    """The reviewer accepted the work."""

    kind: Literal["approved"] = "approved"


class SentBack(_Event):
    """The reviewer returned the work to an earlier agent."""

    kind: Literal["sent_back"] = "sent_back"
    target: RevisionTarget
    reason: str


class InvalidOutput(_Event):
    """The agent's output did not match its schema."""

    kind: Literal["invalid_output"] = "invalid_output"
    detail: str = ""


StepEvent = Annotated[
    Extracted | Recommended | Approved | SentBack | InvalidOutput,
    Field(discriminator="kind"),
]


class TraceEntry(_Frozen):
    """One executed step: which agent ran and what came of it."""

    node: Node
    event: str
    cost: Decimal
    detail: str = ""


class Revisions(_Frozen):
    """How many times the reviewer has sent work back to each agent."""

    extractor: int = Field(default=0, ge=0)
    investigator: int = Field(default=0, ge=0)

    def count(self, target: RevisionTarget) -> int:
        """Return the number of revisions asked of ``target`` so far."""
        return getattr(self, target)

    def increment(self, target: RevisionTarget) -> Revisions:
        """Return a copy with one more revision recorded for ``target``."""
        return self.model_copy(update={target: self.count(target) + 1})


class CaseState(_Frozen):
    """Where a case stands.

    A run is ``running`` until it is either ``completed`` (the reviewer
    approved and the case waits for a person) or ``escalated`` (a stopping
    guard ended it). ``next_node`` names the agent that runs next and is
    ``None`` once the run has ended.

    ``extraction`` and ``recommendation`` hold the latest work of the first
    two agents. ``feedback`` is the reviewer's reason for the most recent
    send-back, kept until the agent it was addressed to has answered.
    """

    case_id: str
    limits: Limits
    status: RunStatus = "running"
    escalation_reason: EscalationReason | None = None
    next_node: Node | None = "extractor"
    revisions: Revisions = Revisions()
    consecutive_schema_errors: int = Field(default=0, ge=0)
    budget_spent: Decimal = Field(default=Decimal(0), ge=0)
    extraction: Extraction | None = None
    recommendation: Recommendation | None = None
    feedback: str | None = None
    trace: tuple[TraceEntry, ...] = ()

    @property
    def steps_taken(self) -> int:
        """The number of agent steps executed so far."""
        return len(self.trace)

    @property
    def path(self) -> tuple[Node, ...]:
        """The agents that ran, in order."""
        return tuple(entry.node for entry in self.trace)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (self.status == "running") != (self.next_node is not None):
            raise ValueError("next_node is set exactly while the run is running")
        if (self.status == "escalated") != (self.escalation_reason is not None):
            raise ValueError("escalation_reason is set exactly when escalated")
        return self
