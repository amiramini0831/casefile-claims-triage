"""The typed messages the agents hand to each other.

No free text travels between agents except inside these models: the extractor
passes on an ``Extraction``, the investigator a ``Recommendation``. Whether the
documents contradict each other is worked out here in code from the extracted
values, not left to a model's judgement.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from casefile.data.records import IncidentSeverity, IncidentType
from casefile.rules import Decision, DocumentIssue


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClaimFormFields(_Frozen):
    """What the claim form states."""

    claim_number: str
    policy_number: int
    incident_date: date
    incident_type: IncidentType
    incident_severity: IncidentSeverity
    injury_claim: Decimal
    property_claim: Decimal
    vehicle_claim: Decimal
    total_claim_amount: Decimal


class RepairEstimateFields(_Frozen):
    """What the repair estimate states."""

    policy_number: int
    total_amount: Decimal


class PoliceReportFields(_Frozen):
    """What the police report states."""

    policy_number: int
    incident_date: date


class Extraction(_Frozen):
    """Everything the extractor read from the documents of one case.

    The claim form is always present. A missing repair estimate or police
    report is ``None``; only the repair estimate is a required document.
    """

    claim_form: ClaimFormFields
    repair_estimate: RepairEstimateFields | None = None
    police_report: PoliceReportFields | None = None

    def amounts_add_up(self) -> bool:
        """Return whether the claimed parts sum to the claimed total."""
        form = self.claim_form
        parts = form.injury_claim + form.property_claim + form.vehicle_claim
        return parts == form.total_claim_amount

    def document_issues(self) -> tuple[DocumentIssue, ...]:
        """Return the ways in which the documents are incomplete or disagree."""
        issues: list[DocumentIssue] = []
        form = self.claim_form
        if self.repair_estimate is None:
            issues.append("missing_document")
        elif self.repair_estimate.total_amount != form.vehicle_claim:
            issues.append("amount_mismatch")
        if (
            self.police_report is not None
            and self.police_report.incident_date != form.incident_date
        ):
            issues.append("date_mismatch")
        return tuple(issues)


class Recommendation(_Frozen):
    """What the investigator proposes, for a person to approve or reject."""

    decision: Decision
    rationale: str
