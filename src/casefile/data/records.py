"""Typed rows of the domain tables.

These models mirror db/schema/001_domain.sql. The database enforces the same
rules; checking them here as well produces an error that names the offending
field before anything is written.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

InsuredSex = Literal["MALE", "FEMALE"]
IncidentType = Literal[
    "Single Vehicle Collision",
    "Multi-vehicle Collision",
    "Vehicle Theft",
    "Parked Car",
]
CollisionType = Literal["Front Collision", "Rear Collision", "Side Collision"]
IncidentSeverity = Literal[
    "Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"
]
AuthoritiesContacted = Literal["Police", "Fire", "Ambulance", "Other", "None"]

INCIDENT_TYPES_WITHOUT_COLLISION: frozenset[str] = frozenset(
    {"Vehicle Theft", "Parked Car"}
)
MULTI_VEHICLE_INCIDENT = "Multi-vehicle Collision"
MIN_AGE_AT_POLICY_START = 16

StateCode = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
Money = Annotated[Decimal, Field(ge=0, max_digits=12, decimal_places=2)]


class PolicyRecord(BaseModel):
    """One row of the ``policies`` table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_number: int = Field(gt=0)
    policy_state: StateCode
    policy_start_date: date
    months_as_customer: int = Field(ge=0)
    csl_per_person: int = Field(gt=0)
    csl_per_accident: int = Field(gt=0)
    policy_deductible: int = Field(ge=0)
    policy_annual_premium: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    umbrella_limit: int = Field(ge=0)
    insured_age: int = Field(ge=MIN_AGE_AT_POLICY_START, le=110)
    insured_sex: InsuredSex
    auto_make: str = Field(min_length=1, max_length=32)
    auto_model: str = Field(min_length=1, max_length=32)
    auto_year: int = Field(ge=1900, le=2100)

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> Self:
        if self.csl_per_accident < self.csl_per_person:
            raise ValueError("csl_per_accident is below csl_per_person")
        months_old = self.insured_age * 12
        if months_old < self.months_as_customer + MIN_AGE_AT_POLICY_START * 12:
            raise ValueError(
                f"insured was under {MIN_AGE_AT_POLICY_START} when the policy started"
            )
        return self


class ClaimRecord(BaseModel):
    """One row of the ``claims`` table, without its generated columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(pattern=r"^CLM-\d{4}-\d{5}$")
    policy_number: int = Field(gt=0)
    incident_date: date
    incident_type: IncidentType
    collision_type: CollisionType | None
    incident_severity: IncidentSeverity
    authorities_contacted: AuthoritiesContacted
    police_report_available: bool | None
    incident_state: StateCode
    incident_city: str = Field(min_length=1, max_length=64)
    incident_location: str = Field(min_length=1, max_length=128)
    incident_hour_of_the_day: int = Field(ge=0, le=23)
    number_of_vehicles_involved: int = Field(ge=1)
    witnesses: int = Field(ge=0)
    total_claim_amount: Money
    injury_claim: Money
    property_claim: Money
    vehicle_claim: Money

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> Self:
        parts = self.injury_claim + self.property_claim + self.vehicle_claim
        if self.total_claim_amount != parts:
            raise ValueError(
                f"total_claim_amount {self.total_claim_amount} is not the sum "
                f"of its parts ({parts})"
            )
        without_collision = self.incident_type in INCIDENT_TYPES_WITHOUT_COLLISION
        if (self.collision_type is None) != without_collision:
            raise ValueError(
                f"collision_type {self.collision_type!r} does not fit "
                f"incident_type {self.incident_type!r}"
            )
        is_multi_vehicle = self.incident_type == MULTI_VEHICLE_INCIDENT
        if is_multi_vehicle != (self.number_of_vehicles_involved >= 2):
            raise ValueError(
                f"{self.number_of_vehicles_involved} vehicle(s) does not fit "
                f"incident_type {self.incident_type!r}"
            )
        return self


class RepairCostReferenceRecord(BaseModel):
    """One row of the ``repair_cost_reference`` table. Hypothetical."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_type: IncidentType
    incident_severity: IncidentSeverity
    low_amount: Money
    typical_amount: Money
    high_amount: Money
    sample_size: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if not self.low_amount <= self.typical_amount <= self.high_amount:
            raise ValueError("low, typical and high amounts are out of order")
        return self


class PriorClaimRecord(BaseModel):
    """One row of the ``prior_claims`` table. Always synthetic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prior_claim_id: str = Field(pattern=r"^PCL-\d{5}$")
    policy_number: int = Field(gt=0)
    incident_date: date
    incident_type: IncidentType
    incident_severity: IncidentSeverity
    total_claim_amount: Money
    injury_claim: Money
    property_claim: Money
    vehicle_claim: Money

    @model_validator(mode="after")
    def _check_total(self) -> Self:
        parts = self.injury_claim + self.property_claim + self.vehicle_claim
        if self.total_claim_amount != parts:
            raise ValueError(
                f"total_claim_amount {self.total_claim_amount} is not the sum "
                f"of its parts ({parts})"
            )
        return self
