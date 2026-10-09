"""Apply the triage rules to what was extracted from a case's documents.

This is the deterministic core that both the stand-in agents and the real
ones share: look up the policy the documents name, its claim history and the
repair cost range, then let ``casefile.rules`` decide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

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
