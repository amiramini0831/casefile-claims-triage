"""Rules for the hypothetical repair cost reference table."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from casefile.config import RAW_DATA_DIR
from casefile.data.records import ClaimRecord
from casefile.data.repair_cost_reference import (
    MIN_SAMPLE_SIZE,
    build_reference,
    percentile,
)
from casefile.data.transform import RAW_FILE_NAME, read_raw_claims, transform

CLAIMS_PATH = RAW_DATA_DIR / RAW_FILE_NAME
needs_raw_file = pytest.mark.skipif(
    not CLAIMS_PATH.exists(), reason=f"{RAW_FILE_NAME} has not been downloaded"
)


def make_claim(
    number: int, vehicle_claim: int, severity: str = "Minor Damage"
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=f"CLM-2015-{number:05d}",
        policy_number=number,
        incident_date=date(2015, 1, 25),
        incident_type="Vehicle Theft",
        collision_type=None,
        incident_severity=severity,
        authorities_contacted="Police",
        police_report_available=True,
        incident_state="SC",
        incident_city="Columbus",
        incident_location="9935 4th Drive",
        incident_hour_of_the_day=5,
        number_of_vehicles_involved=1,
        witnesses=0,
        total_claim_amount=Decimal(vehicle_claim),
        injury_claim=Decimal(0),
        property_claim=Decimal(0),
        vehicle_claim=Decimal(vehicle_claim),
    )


@pytest.mark.parametrize(
    ("percent", "expected"), [(5, 5), (50, 50), (95, 95), (100, 100), (1, 1)]
)
def test_percentile_of_one_to_hundred(percent: int, expected: int) -> None:
    values = [Decimal(value) for value in range(1, 101)]

    assert percentile(values, percent) == Decimal(expected)


def test_percentile_returns_an_observed_value() -> None:
    values = [Decimal(10), Decimal(20), Decimal(1000)]

    assert percentile(values, 50) == Decimal(20)
    assert percentile(values, 95) == Decimal(1000)
    assert percentile(values, 5) == Decimal(10)


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


def test_range_holds_the_middle_ninety_percent() -> None:
    claims = [
        make_claim(number, vehicle_claim=number * 100) for number in range(1, 101)
    ]

    (record,) = build_reference(claims)

    assert record.incident_type == "Vehicle Theft"
    assert record.incident_severity == "Minor Damage"
    assert record.sample_size == 100
    assert record.low_amount == Decimal(500)
    assert record.typical_amount == Decimal(5000)
    assert record.high_amount == Decimal(9500)


def test_one_row_per_type_and_severity() -> None:
    minor = [make_claim(number, 1000 + number) for number in range(1, 41)]
    trivial = [
        make_claim(number, 100 + number, severity="Trivial Damage")
        for number in range(41, 81)
    ]

    records = build_reference(minor + trivial)

    assert [record.incident_severity for record in records] == [
        "Trivial Damage",
        "Minor Damage",
    ]
    assert records[0].high_amount < records[1].low_amount


def test_small_groups_get_no_reference() -> None:
    claims = [make_claim(number, 1000) for number in range(1, MIN_SAMPLE_SIZE)]

    assert build_reference(claims) == ()


def test_result_does_not_depend_on_claim_order() -> None:
    claims = [make_claim(number, vehicle_claim=number * 7) for number in range(1, 60)]

    assert build_reference(claims) == build_reference(claims[::-1])


@needs_raw_file
def test_real_dataset_reference() -> None:
    claims = transform(read_raw_claims(CLAIMS_PATH)).claims

    records = build_reference(claims)

    assert len(records) == 10
    assert sum(record.sample_size for record in records) == len(claims)
    by_key = {(r.incident_type, r.incident_severity): r for r in records}
    for claim in claims:
        assert (claim.incident_type, claim.incident_severity) in by_key
    inside = sum(
        by_key[c.incident_type, c.incident_severity].low_amount
        <= c.vehicle_claim
        <= by_key[c.incident_type, c.incident_severity].high_amount
        for c in claims
    )
    assert 0.88 <= inside / len(claims) <= 0.94
