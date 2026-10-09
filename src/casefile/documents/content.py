"""Decide what each document of a case says.

Nothing here draws a PDF; this module only builds the values, so the rules for
planting defects can be tested without rendering anything.

The datasets hold no names, repair line items or incident narratives, so those
are invented here from fixed word lists and a per-case seed. Everything else
on a document comes from the policy and claim records, except where a defect
is planted on purpose. Every planted difference is returned as a
``PlantedDefect`` so that it can be written to the answer key.
"""

from __future__ import annotations

import random
from collections.abc import Collection
from datetime import date, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from casefile.data.records import ClaimRecord, PolicyRecord
from casefile.eval.cases import DefectType, EvalCase

SEED = 20150515
INSURER_NAME = "Casefile Mutual Insurance"  # fictional

FIRST_NAMES: dict[str, tuple[str, ...]] = {
    "MALE": (
        "James", "Robert", "Michael", "David", "Daniel", "Thomas",
        "Andrew", "Kevin", "Brian", "Steven", "Eric", "Samuel",
    ),
    "FEMALE": (
        "Mary", "Linda", "Susan", "Karen", "Nancy", "Laura",
        "Emily", "Rachel", "Angela", "Julia", "Diane", "Helen",
    ),
}  # fmt: skip
LAST_NAMES: tuple[str, ...] = (
    "Anderson", "Bennett", "Carter", "Dawson", "Ellis", "Foster", "Graham",
    "Hayes", "Ingram", "Jensen", "Keller", "Lawson", "Mercer", "Nolan",
    "Osborne", "Parker", "Quinn", "Reeves", "Sutton", "Turner",
)  # fmt: skip
REPAIR_SHOPS: tuple[str, ...] = (
    "Riverside Auto Body",
    "Summit Collision Center",
    "Lakeview Motor Works",
    "Pioneer Body and Paint",
    "Keystone Auto Repair",
    "Maple Street Collision",
)

PARTS_BY_DAMAGE: dict[str, tuple[str, ...]] = {
    "Front Collision": (
        "Front bumper assembly", "Hood panel", "Radiator and condenser",
        "Headlamp assembly", "Front fender", "Airbag module",
    ),
    "Rear Collision": (
        "Rear bumper assembly", "Trunk lid", "Tail lamp assembly",
        "Rear body panel", "Rear quarter panel", "Exhaust system",
    ),
    "Side Collision": (
        "Front door shell", "Rear door shell", "Side mirror assembly",
        "Rocker panel", "Side curtain airbag", "Door glass",
    ),
    "Vehicle Theft": (
        "Ignition lock cylinder", "Steering column housing", "Door lock set",
        "Side window glass", "Audio head unit", "Wheel and tire set",
    ),
    "Parked Car": (
        "Rear bumper cover", "Tail lamp assembly", "Quarter panel repair",
        "Door panel repair", "Side mirror assembly", "Fender repair",
    ),
}  # fmt: skip
IMPACT_AREA: dict[str, str] = {
    "Front Collision": "front",
    "Rear Collision": "rear",
    "Side Collision": "side",
}

# A planted repair estimate is this much higher than the claimed amount.
AMOUNT_MISMATCH_FACTORS: tuple[Decimal, ...] = (
    Decimal("1.20"),
    Decimal("1.25"),
    Decimal("1.30"),
    Decimal("1.40"),
)
DATE_MISMATCH_DAYS = (2, 4)
OUTSIDE_COVERAGE_DAYS = (10, 60)
POLICY_NUMBER_RANGE = (100_000, 1_000_000)


class _Content(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LineItem(_Content):
    """One line of a repair estimate."""

    description: str
    category: str
    amount: Decimal


class ClaimFormContent(_Content):
    """What the claim form says."""

    claim_number: str
    date_filed: date
    policyholder_name: str
    policyholder_age: int
    policy_number: int
    policy_state: str
    vehicle: str
    incident_date: date
    incident_time: str
    incident_type: str
    collision_type: str
    incident_severity: str
    incident_location: str
    authorities_contacted: str
    vehicles_involved: int
    witnesses: int
    police_report: str
    description: str
    injury_claim: Decimal
    property_claim: Decimal
    vehicle_claim: Decimal
    total_claim_amount: Decimal


class RepairEstimateContent(_Content):
    """What the repair shop's estimate says."""

    shop_name: str
    estimate_number: str
    estimate_date: date
    claim_number: str
    policy_number: int
    customer_name: str
    vehicle: str
    items: tuple[LineItem, ...]
    total_amount: Decimal


class PoliceReportContent(_Content):
    """What the police report says."""

    agency: str
    report_number: str
    report_date: date
    incident_date: date
    incident_time: str
    incident_location: str
    incident_type: str
    collision_type: str
    vehicles_involved: int
    witnesses: int
    driver_name: str
    vehicle: str
    policy_number: int
    narrative: str


class PlantedDefect(_Content):
    """One deliberate difference between the documents and the truth."""

    case_id: str
    claim_id: str
    defect: DefectType
    document: str
    field: str
    true_value: str
    planted_value: str


class CaseDocuments(_Content):
    """Every document of one case, and what was planted in them."""

    case_id: str
    claim_form: ClaimFormContent
    repair_estimate: RepairEstimateContent | None
    police_report: PoliceReportContent | None
    defects: tuple[PlantedDefect, ...]


def split_amount(total: int, parts: int, rng: random.Random) -> list[int]:
    """Split ``total`` whole dollars into ``parts`` positive amounts."""
    parts = max(1, min(parts, total))
    weights = [rng.uniform(1, 3) for _ in range(parts)]
    scale = (total - parts) / sum(weights)
    amounts = [1 + int(weight * scale) for weight in weights]
    amounts[0] += total - sum(amounts)
    return amounts


def build_line_items(
    claim: ClaimRecord, total: Decimal, rng: random.Random
) -> tuple[LineItem, ...]:
    """Invent the lines of a repair estimate that add up to ``total``."""
    if total <= 0 or total != total.to_integral_value():
        raise ValueError(f"repair total {total} is not a positive whole amount")
    damage = claim.collision_type or claim.incident_type
    part_names = rng.sample(PARTS_BY_DAMAGE[damage], rng.randint(2, 4))
    lines = [(name, "Parts") for name in part_names]
    lines += [("Body labor", "Labor"), ("Paint and materials", "Paint")]
    amounts = split_amount(int(total), len(lines), rng)
    # A total smaller than the number of lines yields fewer amounts, and the
    # surplus lines are dropped.
    return tuple(
        LineItem(description=name, category=category, amount=Decimal(amount))
        for (name, category), amount in zip(lines, amounts, strict=False)
    )


def describe_incident(claim: ClaimRecord, incident_time: str) -> str:
    """Write a short account of the incident from fixed sentences."""
    location = f"{claim.incident_location}, {claim.incident_city}"
    if claim.incident_type == "Vehicle Theft":
        what = (
            f"The insured vehicle was reported stolen from {location}. "
            "It was later recovered with damage."
        )
    elif claim.incident_type == "Parked Car":
        what = (
            f"The insured vehicle was struck while parked and unattended at "
            f"{location}. The other party did not remain at the scene."
        )
    else:
        area = IMPACT_AREA[claim.collision_type or ""]
        if claim.incident_type == "Multi-vehicle Collision":
            others = claim.number_of_vehicles_involved - 1
            noun = "vehicle" if others == 1 else "vehicles"
            how = f"was involved in a collision with {others} other {noun}"
        else:
            how = "left the roadway and struck a fixed object"
        what = (
            f"The insured vehicle {how} at {location}. "
            f"The impact was to the {area} of the vehicle."
        )
    return f"{what} The incident occurred at approximately {incident_time}."


def build_case_documents(
    case: EvalCase,
    policy: PolicyRecord,
    claim: ClaimRecord,
    known_policy_numbers: Collection[int],
    seed: int = SEED,
) -> CaseDocuments:
    """Build the documents of ``case``, planting its defect if it has one."""
    # A string seed is hashed by the random module, so each case gets its own
    # stream and adding a case never changes the documents of another.
    rng = random.Random(f"{seed}:{case.case_id}")  # noqa: S311 - not cryptography
    defect = case.document_defect
    planted: list[PlantedDefect] = []

    def record(document: str, field: str, true_value: object, value: object) -> None:
        if defect is None:
            raise AssertionError("a difference was planted in a case without defect")
        planted.append(
            PlantedDefect(
                case_id=case.case_id,
                claim_id=claim.claim_id,
                defect=defect,
                document=document,
                field=field,
                true_value=str(true_value),
                planted_value=str(value),
            )
        )

    name = f"{rng.choice(FIRST_NAMES[policy.insured_sex])} {rng.choice(LAST_NAMES)}"
    vehicle = f"{policy.auto_year} {policy.auto_make} {policy.auto_model}"
    location = (
        f"{claim.incident_location}, {claim.incident_city}, {claim.incident_state}"
    )
    incident_time = f"{claim.incident_hour_of_the_day:02d}:00"
    collision_type = claim.collision_type or "Not applicable"
    filing_delay = timedelta(days=rng.randint(1, 5))
    estimate_delay = timedelta(days=rng.randint(1, 7))
    shop_name = rng.choice(REPAIR_SHOPS)
    estimate_number = f"EST-{rng.randrange(100_000, 1_000_000)}"
    report_number = f"PR-{rng.randrange(100_000, 1_000_000)}"

    policy_number = policy.policy_number
    if defect == "wrong_policy_number":
        while policy_number in known_policy_numbers:
            policy_number = rng.randrange(*POLICY_NUMBER_RANGE)
        record("all", "policy_number", policy.policy_number, policy_number)

    incident_date = claim.incident_date
    if defect == "incident_outside_coverage":
        days_before = rng.randint(*OUTSIDE_COVERAGE_DAYS)
        incident_date = policy.policy_start_date - timedelta(days=days_before)
        record("all", "incident_date", claim.incident_date, incident_date)

    claim_form = ClaimFormContent(
        claim_number=claim.claim_id,
        # Paperwork dates follow the real timeline even when the stated
        # incident date is planted: a back-dated incident is still filed now.
        date_filed=claim.incident_date + filing_delay,
        policyholder_name=name,
        policyholder_age=policy.insured_age,
        policy_number=policy_number,
        policy_state=policy.policy_state,
        vehicle=vehicle,
        incident_date=incident_date,
        incident_time=incident_time,
        incident_type=claim.incident_type,
        collision_type=collision_type,
        incident_severity=claim.incident_severity,
        incident_location=location,
        authorities_contacted=claim.authorities_contacted,
        vehicles_involved=claim.number_of_vehicles_involved,
        witnesses=claim.witnesses,
        police_report={True: "Yes", False: "No", None: "Not stated"}[
            claim.police_report_available
        ],
        description=describe_incident(claim, incident_time),
        injury_claim=claim.injury_claim,
        property_claim=claim.property_claim,
        vehicle_claim=claim.vehicle_claim,
        total_claim_amount=claim.total_claim_amount,
    )

    repair_estimate: RepairEstimateContent | None = None
    if defect == "missing_document":
        record("repair_estimate", "document", "present", "absent")
    else:
        estimate_total = claim.vehicle_claim
        if defect == "amount_mismatch":
            factor = rng.choice(AMOUNT_MISMATCH_FACTORS)
            # Rounded to ten dollars, like a real estimate total.
            estimate_total = Decimal(round(int(claim.vehicle_claim * factor), -1))
            record(
                "repair_estimate", "total_amount", claim.vehicle_claim, estimate_total
            )
        repair_estimate = RepairEstimateContent(
            shop_name=shop_name,
            estimate_number=estimate_number,
            estimate_date=claim.incident_date + estimate_delay,
            claim_number=claim.claim_id,
            policy_number=policy_number,
            customer_name=name,
            vehicle=vehicle,
            items=build_line_items(claim, estimate_total, rng),
            total_amount=estimate_total,
        )

    police_report: PoliceReportContent | None = None
    if claim.police_report_available:
        reported_date = incident_date
        if defect == "date_mismatch":
            shift = timedelta(days=rng.randint(*DATE_MISMATCH_DAYS))
            reported_date = incident_date - shift
            if reported_date <= policy.policy_start_date:
                reported_date = incident_date + shift
            record("police_report", "incident_date", incident_date, reported_date)
        police_report = PoliceReportContent(
            agency=f"{claim.incident_city} Police Department",
            report_number=report_number,
            report_date=reported_date + timedelta(days=1),
            incident_date=reported_date,
            incident_time=incident_time,
            incident_location=location,
            incident_type=claim.incident_type,
            collision_type=collision_type,
            vehicles_involved=claim.number_of_vehicles_involved,
            witnesses=claim.witnesses,
            driver_name=name,
            vehicle=vehicle,
            policy_number=policy_number,
            narrative=describe_incident(claim, incident_time),
        )
    elif defect == "date_mismatch":
        raise ValueError(f"{case.case_id}: a date mismatch needs a police report")

    return CaseDocuments(
        case_id=case.case_id,
        claim_form=claim_form,
        repair_estimate=repair_estimate,
        police_report=police_report,
        defects=tuple(planted),
    )
