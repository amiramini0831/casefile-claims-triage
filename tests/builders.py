"""Builders for valid domain records, so tests only spell out what they vary."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from casefile.data.records import (
    ClaimRecord,
    PolicyRecord,
    PriorClaimRecord,
    RepairCostReferenceRecord,
)

INCIDENT_DATE = date(2015, 1, 25)
POLICY_NUMBER = 1


def build_policy(**changes: Any) -> PolicyRecord:
    values: dict[str, Any] = {
        "policy_number": POLICY_NUMBER,
        "policy_state": "OH",
        "policy_start_date": date(2005, 1, 25),
        "months_as_customer": 120,
        "csl_per_person": 100_000,
        "csl_per_accident": 300_000,
        "policy_deductible": 1000,
        "policy_annual_premium": Decimal("1200.50"),
        "umbrella_limit": 0,
        "insured_age": 45,
        "insured_sex": "FEMALE",
        "auto_make": "Saab",
        "auto_model": "92x",
        "auto_year": 2004,
    }
    return PolicyRecord(**{**values, **changes})


def build_claim(**changes: Any) -> ClaimRecord:
    """A vehicle theft claim of 5000, split 0 / 1000 / 4000."""
    values: dict[str, Any] = {
        "claim_id": "CLM-2015-00001",
        "policy_number": POLICY_NUMBER,
        "incident_date": INCIDENT_DATE,
        "incident_type": "Vehicle Theft",
        "collision_type": None,
        "incident_severity": "Minor Damage",
        "authorities_contacted": "Police",
        "police_report_available": True,
        "incident_state": "SC",
        "incident_city": "Columbus",
        "incident_location": "9935 4th Drive",
        "incident_hour_of_the_day": 5,
        "number_of_vehicles_involved": 1,
        "witnesses": 0,
        "total_claim_amount": Decimal(5000),
        "injury_claim": Decimal(0),
        "property_claim": Decimal(1000),
        "vehicle_claim": Decimal(4000),
    }
    return ClaimRecord(**{**values, **changes})


def build_prior_claim(
    number: int, incident_date: date, **changes: Any
) -> PriorClaimRecord:
    values: dict[str, Any] = {
        "prior_claim_id": f"PCL-{number:05d}",
        "policy_number": POLICY_NUMBER,
        "incident_date": incident_date,
        "incident_type": "Parked Car",
        "incident_severity": "Minor Damage",
        "total_claim_amount": Decimal(3000),
        "injury_claim": Decimal(0),
        "property_claim": Decimal(0),
        "vehicle_claim": Decimal(3000),
    }
    return PriorClaimRecord(**{**values, **changes})


def build_reference(high_amount: int = 6000) -> RepairCostReferenceRecord:
    return RepairCostReferenceRecord(
        incident_type="Vehicle Theft",
        incident_severity="Minor Damage",
        low_amount=Decimal(1000),
        typical_amount=Decimal(min(2500, high_amount)),
        high_amount=Decimal(high_amount),
        sample_size=40,
    )
