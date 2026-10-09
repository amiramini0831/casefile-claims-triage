"""Rules for the synthetic claim history."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Connection, Table
from sqlalchemy.exc import DBAPIError

from casefile.config import RAW_DATA_DIR
from casefile.data.prior_claims import (
    ORACLE_FILE_NAME,
    find_violations,
    generate_prior_claims,
    read_past_claim_bins,
)
from casefile.data.records import ClaimRecord, PolicyRecord, PriorClaimRecord
from casefile.data.transform import (
    RAW_FILE_NAME,
    RawDataError,
    read_raw_claims,
    subtract_months,
    transform,
)

CLAIMS_PATH = RAW_DATA_DIR / RAW_FILE_NAME
ORACLE_PATH = RAW_DATA_DIR / ORACLE_FILE_NAME
needs_raw_files = pytest.mark.skipif(
    not (CLAIMS_PATH.exists() and ORACLE_PATH.exists()),
    reason="the raw datasets have not been downloaded",
)

INCIDENT_DATE = date(2015, 1, 25)


def make_policy(policy_number: int, months_as_customer: int = 240) -> PolicyRecord:
    return PolicyRecord(
        policy_number=policy_number,
        policy_state="OH",
        policy_start_date=subtract_months(INCIDENT_DATE, months_as_customer),
        months_as_customer=months_as_customer,
        csl_per_person=100_000,
        csl_per_accident=300_000,
        policy_deductible=1000,
        policy_annual_premium=Decimal("1200.50"),
        umbrella_limit=0,
        insured_age=60,
        insured_sex="FEMALE",
        auto_make="Saab",
        auto_model="92x",
        auto_year=2004,
    )


def make_claim(policy_number: int, vehicle_claim: int = 700) -> ClaimRecord:
    return ClaimRecord(
        claim_id=f"CLM-2015-{policy_number:05d}",
        policy_number=policy_number,
        incident_date=INCIDENT_DATE,
        incident_type="Vehicle Theft",
        collision_type=None,
        incident_severity="Minor Damage",
        authorities_contacted="Police",
        police_report_available=True,
        incident_state="SC",
        incident_city="Columbus",
        incident_location="9935 4th Drive",
        incident_hour_of_the_day=5,
        number_of_vehicles_involved=1,
        witnesses=0,
        total_claim_amount=Decimal(vehicle_claim + 300),
        injury_claim=Decimal(0),
        property_claim=Decimal(300),
        vehicle_claim=Decimal(vehicle_claim),
    )


def make_portfolio(
    size: int = 30, months_as_customer: int = 240
) -> tuple[list[PolicyRecord], list[ClaimRecord]]:
    """Policies 1..size, each with a current claim of a distinct amount."""
    numbers = range(1, size + 1)
    policies = [make_policy(number, months_as_customer) for number in numbers]
    claims = [make_claim(number, vehicle_claim=1000 * number) for number in numbers]
    return policies, claims


def test_same_seed_gives_the_same_history() -> None:
    policies, claims = make_portfolio()
    bins = ["none", "1", "2 to 4", "more than 4"]

    first = generate_prior_claims(policies, claims, bins, seed=7)
    second = generate_prior_claims(policies, claims, bins, seed=7)
    other_seed = generate_prior_claims(policies, claims, bins, seed=8)

    assert first == second
    assert first != other_seed


def test_result_does_not_depend_on_input_order() -> None:
    policies, claims = make_portfolio()
    bins = ["2 to 4"]

    forward = generate_prior_claims(policies, claims, bins, seed=7)
    backward = generate_prior_claims(policies[::-1], claims[::-1], bins, seed=7)

    assert forward == backward


@pytest.mark.parametrize(
    ("bin_label", "allowed_counts"),
    [("none", {0}), ("1", {1}), ("2 to 4", {2, 3, 4}), ("more than 4", {5, 6})],
)
def test_number_of_prior_claims_follows_the_bin(
    bin_label: str, allowed_counts: set[int]
) -> None:
    policies, claims = make_portfolio()

    history = generate_prior_claims(policies, claims, [bin_label], seed=7)

    counts = {
        sum(prior.policy_number == policy.policy_number for prior in history)
        for policy in policies
    }
    assert counts <= allowed_counts


@pytest.mark.parametrize(
    ("months_as_customer", "most_allowed"), [(0, 0), (11, 0), (12, 1), (30, 2)]
)
def test_short_tenure_caps_the_number_of_prior_claims(
    months_as_customer: int, most_allowed: int
) -> None:
    policies, claims = make_portfolio(months_as_customer=months_as_customer)

    history = generate_prior_claims(policies, claims, ["more than 4"], seed=7)

    assert len(history) == most_allowed * len(policies)


def test_dates_fall_inside_the_policy_and_never_repeat() -> None:
    policies, claims = make_portfolio(months_as_customer=72)
    start = {policy.policy_number: policy.policy_start_date for policy in policies}

    history = generate_prior_claims(policies, claims, ["more than 4"], seed=7)

    assert all(
        start[prior.policy_number] < prior.incident_date < INCIDENT_DATE
        for prior in history
    )
    days = [(prior.policy_number, prior.incident_date) for prior in history]
    assert len(days) == len(set(days))


def test_content_is_copied_whole_from_another_policy() -> None:
    policies, claims = make_portfolio()
    owner_of_amount = {claim.vehicle_claim: claim.policy_number for claim in claims}

    history = generate_prior_claims(policies, claims, ["2 to 4"], seed=7)

    assert history
    for prior in history:
        assert owner_of_amount[prior.vehicle_claim] != prior.policy_number
        assert prior.total_claim_amount == prior.vehicle_claim + 300
        assert prior.incident_type == "Vehicle Theft"


def test_history_is_not_tied_to_a_vehicle() -> None:
    assert not any(name.startswith("auto_") for name in PriorClaimRecord.model_fields)


def test_generated_history_has_no_violations() -> None:
    policies, claims = make_portfolio()

    history = generate_prior_claims(policies, claims, ["more than 4"], seed=7)

    assert find_violations(policies, claims, history) == []


def test_violations_are_detected() -> None:
    policies, claims = make_portfolio(size=2, months_as_customer=12)
    amounts: dict[str, Any] = {
        "incident_type": "Vehicle Theft",
        "incident_severity": "Minor Damage",
        "total_claim_amount": Decimal(200_300),
        "injury_claim": Decimal(200_000),
        "property_claim": Decimal(300),
        "vehicle_claim": Decimal(0),
    }
    history = [
        PriorClaimRecord(
            prior_claim_id=f"PCL-{index:05d}",
            policy_number=1,
            incident_date=INCIDENT_DATE,
            **amounts,
        )
        for index in (1, 2)
    ]

    messages = " | ".join(find_violations(policies, claims, history))

    assert "is outside" in messages
    assert "second claim on" in messages
    assert "exceeds the coverage limit" in messages
    assert "tenure allows 1" in messages


def write_oracle(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["MonthClaimed,PastNumberOfClaims", *(",".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reading_bins_skips_the_corrupt_row(tmp_path: Path) -> None:
    path = tmp_path / ORACLE_FILE_NAME
    write_oracle(path, [("Jan", "none"), ("0", "none"), ("Feb", "2 to 4")])

    assert read_past_claim_bins(path, verify_checksum=False) == ["none", "2 to 4"]


def test_unexpected_bin_label_is_refused(tmp_path: Path) -> None:
    path = tmp_path / ORACLE_FILE_NAME
    write_oracle(path, [("Jan", "7")])

    with pytest.raises(RawDataError, match="line 2"):
        read_past_claim_bins(path, verify_checksum=False)


def test_unknown_oracle_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / ORACLE_FILE_NAME
    write_oracle(path, [("Jan", "none")])

    with pytest.raises(RawDataError, match="not a known version"):
        read_past_claim_bins(path)


@needs_raw_files
def test_real_datasets_give_a_consistent_history() -> None:
    tables = transform(read_raw_claims(CLAIMS_PATH))
    bins = read_past_claim_bins(ORACLE_PATH)

    history = generate_prior_claims(tables.policies, tables.claims, bins)

    assert len(bins) == 15_419
    assert history
    assert find_violations(tables.policies, tables.claims, history) == []
    assert history == generate_prior_claims(tables.policies, tables.claims, bins)
    assert len({prior.prior_claim_id for prior in history}) == len(history)


PRIOR_CLAIM: dict[str, Any] = {
    "prior_claim_id": "TEST-0001",
    "incident_date": "2012-06-01",
    "incident_type": "Vehicle Theft",
    "incident_severity": "Minor Damage",
    "total_claim_amount": 1000,
    "injury_claim": 0,
    "property_claim": 300,
    "vehicle_claim": 700,
}


@pytest.fixture
def prior_claims_table(
    domain_tables: dict[str, Table], loader_connection: Connection
) -> Table:
    """The prior_claims table, with a policy that test rows can refer to."""
    policy = make_policy(policy_number=1).model_dump()
    loader_connection.execute(domain_tables["policies"].insert(), policy)
    return domain_tables["prior_claims"]


@pytest.mark.parametrize(
    ("changes", "expected_codes"),
    [
        pytest.param(
            {"incident_date": "2012-06-02", "total_claim_amount": 999},
            {3819},
            id="total-not-sum",
        ),
        pytest.param(
            {"incident_date": "2012-06-02", "policy_number": 999_999_999},
            {1216, 1452},
            id="unknown-policy",
        ),
        pytest.param({}, {1062}, id="second-claim-on-the-same-day"),
    ],
)
def test_database_rejects_inconsistent_prior_claim(
    loader_connection: Connection,
    prior_claims_table: Table,
    changes: dict[str, Any],
    expected_codes: set[int],
) -> None:
    valid = {**PRIOR_CLAIM, "policy_number": 1}
    loader_connection.execute(prior_claims_table.insert(), valid)
    second = {**valid, "prior_claim_id": "TEST-0002", **changes}

    with pytest.raises(DBAPIError) as raised:
        loader_connection.execute(prior_claims_table.insert(), second)

    assert raised.value.orig.args[0] in expected_codes
