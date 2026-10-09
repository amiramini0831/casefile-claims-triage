"""Rules for turning raw insurance_claims.csv rows into domain records."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from casefile.config import RAW_DATA_DIR
from casefile.data.records import ClaimRecord, PolicyRecord
from casefile.data.transform import (
    RAW_FILE_NAME,
    RawDataError,
    read_raw_claims,
    subtract_months,
    transform,
)

RAW_PATH = RAW_DATA_DIR / RAW_FILE_NAME
needs_raw_file = pytest.mark.skipif(
    not RAW_PATH.exists(), reason=f"{RAW_FILE_NAME} has not been downloaded"
)


def raw_row(**changes: str) -> dict[str, str]:
    """Return a valid raw row as text, with ``changes`` applied."""
    row = {
        "months_as_customer": "328",
        "age": "48",
        "policy_number": "521585",
        "policy_bind_date": "2014-10-17",
        "policy_state": "OH",
        "policy_csl": "250/500",
        "policy_deductable": "1000",
        "policy_annual_premium": "1406.91",
        "umbrella_limit": "0",
        "insured_sex": "MALE",
        "insured_hobbies": "sleeping",
        "incident_date": "2015-01-25",
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Side Collision",
        "incident_severity": "Major Damage",
        "authorities_contacted": "Police",
        "incident_state": "SC",
        "incident_city": "Columbus",
        "incident_location": "9935 4th Drive",
        "incident_hour_of_the_day": "5",
        "number_of_vehicles_involved": "1",
        "property_damage": "YES",
        "bodily_injuries": "1",
        "witnesses": "2",
        "police_report_available": "YES",
        "total_claim_amount": "71610",
        "injury_claim": "6510",
        "property_claim": "13020",
        "vehicle_claim": "52080",
        "auto_make": "Saab",
        "auto_model": "92x",
        "auto_year": "2004",
        "fraud_reported": "Y",
        "_c39": "",
    }
    row.update(changes)
    return row


def theft_row(**changes: str) -> dict[str, str]:
    """Return a valid vehicle theft row, which has no collision type."""
    return raw_row(incident_type="Vehicle Theft", collision_type="?", **changes)


def only_policy(row: dict[str, str]) -> PolicyRecord:
    return transform([row]).policies[0]


def only_claim(row: dict[str, str]) -> ClaimRecord:
    return transform([row]).claims[0]


@pytest.mark.parametrize(
    ("day", "months", "expected"),
    [
        (date(2015, 1, 25), 0, date(2015, 1, 25)),
        (date(2015, 1, 25), 328, date(1987, 9, 25)),
        (date(2015, 1, 10), 13, date(2013, 12, 10)),
        (date(2015, 3, 31), 1, date(2015, 2, 28)),
        (date(2016, 3, 31), 1, date(2016, 2, 29)),
        (date(2015, 12, 31), 12, date(2014, 12, 31)),
    ],
)
def test_subtract_months(day: date, months: int, expected: date) -> None:
    assert subtract_months(day, months) == expected


def test_policy_fields_are_typed_and_renamed() -> None:
    policy = only_policy(raw_row())

    assert policy.policy_number == 521585
    assert policy.policy_deductible == 1000
    assert policy.policy_annual_premium == Decimal("1406.91")
    assert policy.insured_age == 48
    assert (policy.csl_per_person, policy.csl_per_accident) == (250_000, 500_000)


def test_policy_start_date_comes_from_tenure_not_bind_date() -> None:
    policy = only_policy(
        raw_row(months_as_customer="328", policy_bind_date="2014-10-17")
    )

    assert policy.policy_start_date == date(1987, 9, 25)


@pytest.mark.parametrize(
    ("raw_make", "expected"),
    [("Accura", "Acura"), ("Suburu", "Subaru"), ("Saab", "Saab")],
)
def test_auto_make_spelling_is_corrected(raw_make: str, expected: str) -> None:
    assert only_policy(raw_row(auto_make=raw_make)).auto_make == expected


def test_question_mark_collision_type_becomes_null() -> None:
    assert only_claim(theft_row()).collision_type is None


@pytest.mark.parametrize(
    ("raw_value", "expected"), [("YES", True), ("NO", False), ("?", None)]
)
def test_police_report_is_a_nullable_boolean(
    raw_value: str, expected: bool | None
) -> None:
    claim = only_claim(raw_row(police_report_available=raw_value))

    assert claim.police_report_available is expected


def test_authority_none_stays_a_category() -> None:
    claim = only_claim(theft_row(authorities_contacted="None"))

    assert claim.authorities_contacted == "None"


def test_claim_amounts_are_exact_decimals() -> None:
    claim = only_claim(raw_row())

    assert claim.total_claim_amount == Decimal("71610")
    assert claim.total_claim_amount == (
        claim.injury_claim + claim.property_claim + claim.vehicle_claim
    )


def test_dropped_columns_are_not_part_of_the_records() -> None:
    dropped = {
        "fraud_reported",
        "policy_bind_date",
        "bodily_injuries",
        "property_damage",
        "insured_zip",
        "insured_hobbies",
        "insured_occupation",
        "insured_education_level",
        "insured_relationship",
    }

    assert dropped.isdisjoint(PolicyRecord.model_fields)
    assert dropped.isdisjoint(ClaimRecord.model_fields)


def test_claim_ids_follow_incident_date_then_policy_number() -> None:
    rows = [
        raw_row(policy_number="300", incident_date="2015-02-01"),
        raw_row(policy_number="200", incident_date="2015-01-05"),
        raw_row(policy_number="100", incident_date="2015-01-05"),
    ]

    claims = transform(rows).claims

    assert [(claim.claim_id, claim.policy_number) for claim in claims] == [
        ("CLM-2015-00001", 100),
        ("CLM-2015-00002", 200),
        ("CLM-2015-00003", 300),
    ]


def test_result_does_not_depend_on_row_order() -> None:
    rows = [
        raw_row(policy_number="300", incident_date="2015-02-01"),
        raw_row(policy_number="200", incident_date="2015-01-05"),
        raw_row(policy_number="100", incident_date="2015-01-05"),
    ]

    forward, backward = transform(rows), transform(reversed(rows))

    assert forward.claims == backward.claims
    assert forward.policies == backward.policies


def test_negative_umbrella_limit_is_rejected_with_a_reason() -> None:
    rows = [
        raw_row(policy_number="100"),
        raw_row(policy_number="200", umbrella_limit="-1000000"),
    ]

    tables = transform(rows)

    assert tables.raw_row_count == 2
    assert [policy.policy_number for policy in tables.policies] == [100]
    assert [claim.policy_number for claim in tables.claims] == [100]
    assert len(tables.rejected) == 1
    rejected = tables.rejected[0]
    assert rejected.policy_number == 200
    assert rejected.rule == "negative_umbrella_limit"
    assert rejected.detail == "umbrella_limit = -1000000"


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"total_claim_amount": "71611"}, id="total-not-sum"),
        pytest.param({"incident_severity": "Huge"}, id="unknown-severity"),
        pytest.param({"authorities_contacted": ""}, id="empty-authority"),
        pytest.param({"police_report_available": "MAYBE"}, id="unknown-police-report"),
        pytest.param({"collision_type": "?"}, id="collision-without-collision-type"),
        pytest.param(
            {"incident_type": "Parked Car"}, id="collision-type-on-parked-car"
        ),
        pytest.param({"number_of_vehicles_involved": "3"}, id="vehicle-count"),
        pytest.param({"witnesses": "2.5"}, id="fractional-integer"),
        pytest.param({"injury_claim": "abc"}, id="non-numeric-amount"),
        pytest.param({"incident_date": "25/01/2015"}, id="bad-date"),
        pytest.param({"policy_csl": "250-500"}, id="bad-csl"),
        pytest.param({"age": "20"}, id="under-16-at-policy-start"),
        pytest.param({"incident_state": "South Carolina"}, id="bad-state-code"),
    ],
)
def test_unexpected_data_stops_the_transform(changes: dict[str, str]) -> None:
    with pytest.raises(RawDataError):
        transform([raw_row(**changes)])


def test_duplicate_policy_number_stops_the_transform() -> None:
    with pytest.raises(RawDataError, match="duplicate policy_number 521585"):
        transform([raw_row(), raw_row()])


def test_missing_column_stops_the_transform() -> None:
    row = raw_row()
    del row["vehicle_claim"]

    with pytest.raises(RawDataError, match="vehicle_claim"):
        transform([row])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_reading_keeps_none_and_question_mark_as_text(tmp_path: Path) -> None:
    path = tmp_path / RAW_FILE_NAME
    write_csv(path, [theft_row(authorities_contacted="None")])

    rows = read_raw_claims(path, verify_checksum=False)

    assert rows[0]["authorities_contacted"] == "None"
    assert rows[0]["collision_type"] == "?"
    assert rows[0]["_c39"] == ""


def test_unknown_file_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / RAW_FILE_NAME
    write_csv(path, [raw_row()])

    with pytest.raises(RawDataError, match="not a known version"):
        read_raw_claims(path)


def test_row_with_wrong_number_of_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / RAW_FILE_NAME
    path.write_text("policy_number,age\n1,30\n2\n", encoding="utf-8")

    with pytest.raises(RawDataError, match="line 3"):
        read_raw_claims(path, verify_checksum=False)


@needs_raw_file
def test_real_dataset_transforms_cleanly() -> None:
    tables = transform(read_raw_claims(RAW_PATH))

    assert tables.raw_row_count == 1000
    assert len(tables.policies) == 999
    assert len(tables.claims) == 999
    assert [(row.policy_number, row.rule) for row in tables.rejected] == [
        (526039, "negative_umbrella_limit")
    ]
    assert len({claim.claim_id for claim in tables.claims}) == 999
    assert tables.claims[0].claim_id == "CLM-2015-00001"
    assert tables.claims[-1].claim_id == "CLM-2015-00999"

    start_dates = {p.policy_number: p.policy_start_date for p in tables.policies}
    assert all(
        start_dates[claim.policy_number] <= claim.incident_date
        for claim in tables.claims
    )
    assert {policy.auto_make for policy in tables.policies}.isdisjoint(
        {"Accura", "Suburu"}
    )
