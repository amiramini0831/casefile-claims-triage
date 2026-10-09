"""Apply the triage rules to what was extracted from a case's documents.

This is the deterministic core that both the stand-in agents and the real
ones share: look up the policy the documents name, its claim history and the
repair cost range, then let ``casefile.rules`` decide.

``gather_facts`` collects the same inputs as plain data, for the model to
reason over and for explanations to be checked against.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from casefile.control.handoff import Extraction
from casefile.data.records import (
    PolicyRecord,
    PriorClaimRecord,
    RepairCostReferenceRecord,
)
from casefile.rules import Decision, decide


class DomainLookup(Protocol):
    """Read access to the domain data an agent is allowed to see.

    Deliberately narrow: policies, their earlier claims and the repair cost
    reference. The current claim record is not available here, because it
    holds the very values the agents are supposed to read from the documents.
    """

    def find_policy(self, policy_number: int) -> PolicyRecord | None:
        """Return the policy, or ``None`` if the number matches no policy."""
        ...

    def find_prior_claims(self, policy_number: int) -> Sequence[PriorClaimRecord]:
        """Return the earlier claims of a policy."""
        ...

    def find_reference(
        self, incident_type: str, incident_severity: str
    ) -> RepairCostReferenceRecord | None:
        """Return the repair cost range for this kind of claim, if any."""
        ...


def assess(extraction: Extraction, lookup: DomainLookup) -> Decision:
    """Decide what should happen to the claim the documents describe."""
    form = extraction.claim_form
    return decide(
        claim=form,
        policy=lookup.find_policy(form.policy_number),
        prior_claims=lookup.find_prior_claims(form.policy_number),
        reference=lookup.find_reference(form.incident_type, form.incident_severity),
        document_issues=extraction.document_issues(),
    )


class _Facts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PolicyFacts(_Facts):
    """The parts of a policy the rules look at."""

    policy_number: int
    policy_start_date: date
    policy_deductible: int


class PriorClaimFacts(_Facts):
    """One earlier claim on the policy."""

    incident_date: date
    incident_type: str
    total_claim_amount: Decimal


class ReferenceFacts(_Facts):
    """The usual range of vehicle repair cost for this kind of incident."""

    low_amount: Decimal
    typical_amount: Decimal
    high_amount: Decimal


class CaseFacts(_Facts):
    """Everything known about a case, before anyone has judged it.

    It holds what the documents say and what the lookups returned, and nothing
    derived from them: whether the documents disagree, or how many earlier
    claims are recent, is left for the reader to work out.
    """

    documents: Extraction
    policy: PolicyFacts | None
    prior_claims: tuple[PriorClaimFacts, ...]
    repair_cost_reference: ReferenceFacts | None


CENTS = Decimal("0.01")


def gather_facts(extraction: Extraction, lookup: DomainLookup) -> CaseFacts:
    """Collect the facts of a case from its documents and the domain data.

    Looked-up amounts are written with two decimal places whatever their
    source. MySQL returns 52030.00 where an in-memory record holds 52030; the
    two are equal, but they would be sent to the model as different text, and
    a recorded answer is found again only if the request is the same text.
    """
    form = extraction.claim_form
    policy = lookup.find_policy(form.policy_number)
    reference = lookup.find_reference(form.incident_type, form.incident_severity)
    priors = sorted(
        lookup.find_prior_claims(form.policy_number),
        key=lambda prior: prior.incident_date,
    )
    return CaseFacts(
        documents=extraction,
        policy=policy
        and PolicyFacts(
            policy_number=policy.policy_number,
            policy_start_date=policy.policy_start_date,
            policy_deductible=policy.policy_deductible,
        ),
        prior_claims=tuple(
            PriorClaimFacts(
                incident_date=prior.incident_date,
                incident_type=prior.incident_type,
                total_claim_amount=prior.total_claim_amount.quantize(CENTS),
            )
            for prior in priors
        ),
        repair_cost_reference=reference
        and ReferenceFacts(
            low_amount=reference.low_amount.quantize(CENTS),
            typical_amount=reference.typical_amount.quantize(CENTS),
            high_amount=reference.high_amount.quantize(CENTS),
        ),
    )
