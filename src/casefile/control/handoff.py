"""The typed messages the agents hand to each other.

No free text travels between agents except inside these models: the extractor
passes on an ``Extraction``, the investigator a ``Recommendation``. Whether the
documents contradict each other is worked out here in code from the extracted
values, not left to a model's judgement.

The field descriptions are part of the contract with the model: the JSON
schema generated from these classes, descriptions included, is what the
extractor asks the model to fill in.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from casefile.data.records import IncidentSeverity, IncidentType
from casefile.rules import Decision, DocumentIssue


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClaimFormFields(_Frozen):
    """What the claim form states."""

    claim_number: str = Field(description="Claim number, exactly as printed")
    policy_number: int = Field(description="Policy number printed on this form")
    incident_date: date = Field(
        description="Date of the incident, not the date the form was filed"
    )
    incident_type: IncidentType = Field(description="Type of incident")
    incident_severity: IncidentSeverity = Field(description="Severity of damage")
    injury_claim: Decimal = Field(description="Amount claimed for injury, in USD")
    property_claim: Decimal = Field(description="Amount claimed for property, in USD")
    vehicle_claim: Decimal = Field(description="Amount claimed for the vehicle, in USD")
    total_claim_amount: Decimal = Field(description="Total amount claimed, in USD")


class RepairEstimateFields(_Frozen):
    """What the repair estimate states."""

    policy_number: int = Field(description="Policy number printed on this estimate")
    total_amount: Decimal = Field(
        description="Estimate total in USD, not the amount of a single line"
    )


class PoliceReportFields(_Frozen):
    """What the police report states."""

    policy_number: int = Field(
        description="Insurance policy number printed on this report"
    )
    incident_date: date = Field(
        description="Date of the incident, not the date of the report"
    )


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
    """Two decisions on one claim, for a person to choose between.

    ``rule_decision`` is what the fixed rules give. ``model_decision`` is what
    the model decided on its own, without being shown the rules' result. When
    the two differ, neither overrules the other: both go to a person, each
    with an explanation of what it rests on. How often they agree is a
    measure of how far the model can be relied on.
    """

    rule_decision: Decision
    model_decision: Decision
    rule_explanation: str
    model_explanation: str

    @property
    def outcomes_agree(self) -> bool:
        """Whether both decisions recommend the same thing."""
        return self.rule_decision.outcome == self.model_decision.outcome

    @property
    def decisions_agree(self) -> bool:
        """Whether both agree on outcome, reason and amount."""
        return self.rule_decision == self.model_decision
