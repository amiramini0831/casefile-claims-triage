"""Stand-in agents without a model, for exercising the control plane.

They are not a preview of how well the real system works. The stand-in
extractor is handed the exact values the document generator wrote, so it
"reads" every document perfectly; a real extractor has to earn that from the
PDFs. What a run with these agents shows is that routing, the stopping guards
and the typed handoffs behave as designed.
"""

from __future__ import annotations

from decimal import Decimal

from casefile.agents.triage import DomainLookup, assess
from casefile.control.handoff import (
    ClaimFormFields,
    Extraction,
    PoliceReportFields,
    Recommendation,
    RepairEstimateFields,
)
from casefile.control.state import (
    Approved,
    CaseState,
    Extracted,
    InvalidOutput,
    Recommended,
    SentBack,
    StepEvent,
)
from casefile.documents.content import CaseDocuments

# An arbitrary unit, so that budgets and the budget guard can be exercised.
MOCK_CALL_COST = Decimal(1)


def extraction_from_documents(documents: CaseDocuments) -> Extraction:
    """Return what a flawless reader would extract from ``documents``."""
    form = documents.claim_form
    estimate = documents.repair_estimate
    report = documents.police_report
    return Extraction(
        claim_form=ClaimFormFields(
            claim_number=form.claim_number,
            policy_number=form.policy_number,
            incident_date=form.incident_date,
            incident_type=form.incident_type,
            incident_severity=form.incident_severity,
            injury_claim=form.injury_claim,
            property_claim=form.property_claim,
            vehicle_claim=form.vehicle_claim,
            total_claim_amount=form.total_claim_amount,
        ),
        repair_estimate=None
        if estimate is None
        else RepairEstimateFields(
            policy_number=estimate.policy_number, total_amount=estimate.total_amount
        ),
        police_report=None
        if report is None
        else PoliceReportFields(
            policy_number=report.policy_number, incident_date=report.incident_date
        ),
    )


class _MockAgent:
    def worst_case_cost(self, state: CaseState) -> Decimal:
        return MOCK_CALL_COST


class MockExtractor(_MockAgent):
    """Reports the contents of the case's documents without reading any PDF."""

    def __init__(self, documents: CaseDocuments) -> None:
        self._extraction = extraction_from_documents(documents)

    def run(self, state: CaseState) -> StepEvent:
        return Extracted(cost=MOCK_CALL_COST, extraction=self._extraction)


class MockInvestigator(_MockAgent):
    """Applies the triage rules to the extraction and states which one fired."""

    def __init__(self, lookup: DomainLookup) -> None:
        self._lookup = lookup

    def run(self, state: CaseState) -> StepEvent:
        if state.extraction is None:
            return InvalidOutput(cost=MOCK_CALL_COST, detail="nothing was extracted")
        decision = assess(state.extraction, self._lookup)
        return Recommended(
            cost=MOCK_CALL_COST,
            # The stand-in has no opinion of its own: its "model" decision is
            # the rules' decision.
            recommendation=Recommendation(
                rule_decision=decision,
                model_decision=decision,
                rule_explanation=f"Rule applied: {decision.reason}.",
                model_explanation="Stand-in agent; no model was consulted.",
            ),
        )


class MockReviewer(_MockAgent):
    """Checks the work with two deterministic tests before approving it."""

    def __init__(self, lookup: DomainLookup) -> None:
        self._lookup = lookup

    def run(self, state: CaseState) -> StepEvent:
        extraction, recommendation = state.extraction, state.recommendation
        if extraction is None or recommendation is None:
            return InvalidOutput(cost=MOCK_CALL_COST, detail="nothing to review")
        if not extraction.amounts_add_up():
            return SentBack(
                cost=MOCK_CALL_COST,
                target="extractor",
                reason="The claimed amounts do not add up to the claimed total.",
            )
        if recommendation.rule_decision != assess(extraction, self._lookup):
            return SentBack(
                cost=MOCK_CALL_COST,
                target="investigator",
                reason="The recommendation does not follow from the extracted facts.",
            )
        return Approved(cost=MOCK_CALL_COST)
