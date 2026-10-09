"""The database itself must reject inconsistent rows and agent writes.

These tests bypass the Pydantic records on purpose: they prove that the
constraints in db/schema/ hold even if application code is wrong. Every test
runs in a transaction that is rolled back.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Connection, Table, text
from sqlalchemy.exc import DBAPIError

# MySQL error codes.
CHECK_VIOLATED = {3819}
FOREIGN_KEY_VIOLATED = {1216, 1452}
INVALID_ENUM_VALUE = {1265}
GENERATED_COLUMN_WRITE = {3105}
OUT_OF_RANGE = {1264}
DUPLICATE_KEY = {1062}
COMMAND_DENIED = {1142}

# Policy numbers in the dataset have six digits, so these cannot collide with
# loaded rows.
TEST_POLICY_NUMBER = 1

VALID_POLICY: dict[str, Any] = {
    "policy_number": TEST_POLICY_NUMBER,
    "policy_state": "OH",
    "policy_start_date": "2010-01-25",
    "months_as_customer": 60,
    "csl_per_person": 100_000,
    "csl_per_accident": 300_000,
    "policy_deductible": 1000,
    "policy_annual_premium": "1200.50",
    "umbrella_limit": 0,
    "insured_age": 40,
    "insured_sex": "MALE",
    "auto_make": "Saab",
    "auto_model": "92x",
    "auto_year": 2004,
}

VALID_CLAIM: dict[str, Any] = {
    "claim_id": "TEST-0001",
    "policy_number": TEST_POLICY_NUMBER,
    "incident_date": "2015-01-25",
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Side Collision",
    "incident_severity": "Major Damage",
    "authorities_contacted": "None",
    "police_report_available": None,
    "incident_state": "SC",
    "incident_city": "Columbus",
    "incident_location": "9935 4th Drive",
    "incident_hour_of_the_day": 5,
    "number_of_vehicles_involved": 1,
    "witnesses": 2,
    "total_claim_amount": 1000,
    "injury_claim": 0,
    "property_claim": 300,
    "vehicle_claim": 700,
}


def error_code(connection: Connection, table: Table, row: dict[str, Any]) -> int:
    """Insert ``row`` and return the MySQL error code it is rejected with."""
    with pytest.raises(DBAPIError) as raised:
        connection.execute(table.insert(), row)
    return raised.value.orig.args[0]


@pytest.fixture
def policies(domain_tables: dict[str, Table]) -> Table:
    return domain_tables["policies"]


@pytest.fixture
def claims(domain_tables: dict[str, Table], loader_connection: Connection) -> Table:
    """The claims table, with the policy that test claims refer to in place."""
    loader_connection.execute(domain_tables["policies"].insert(), VALID_POLICY)
    return domain_tables["claims"]


def test_valid_rows_are_accepted_and_flags_are_computed(
    loader_connection: Connection, claims: Table
) -> None:
    loader_connection.execute(claims.insert(), VALID_CLAIM)

    row = loader_connection.execute(
        text(
            "SELECT has_injury, has_property_damage, authorities_contacted, "
            "police_report_available FROM claims WHERE claim_id = :claim_id"
        ),
        {"claim_id": VALID_CLAIM["claim_id"]},
    ).one()

    assert tuple(row) == (0, 1, "None", None)


def test_theft_without_collision_type_is_accepted(
    loader_connection: Connection, claims: Table
) -> None:
    theft = {**VALID_CLAIM, "incident_type": "Vehicle Theft", "collision_type": None}

    loader_connection.execute(claims.insert(), theft)


@pytest.mark.parametrize(
    ("changes", "expected_codes"),
    [
        pytest.param({"total_claim_amount": 999}, CHECK_VIOLATED, id="total-not-sum"),
        pytest.param(
            {"injury_claim": -100, "property_claim": 400},
            CHECK_VIOLATED,
            id="negative-amount",
        ),
        pytest.param(
            {"incident_type": "Vehicle Theft"},
            CHECK_VIOLATED,
            id="collision-type-on-theft",
        ),
        pytest.param(
            {"collision_type": None},
            CHECK_VIOLATED,
            id="collision-without-collision-type",
        ),
        pytest.param(
            {"number_of_vehicles_involved": 2},
            CHECK_VIOLATED,
            id="two-vehicles-in-single-vehicle-collision",
        ),
        pytest.param(
            {"incident_type": "Multi-vehicle Collision"},
            CHECK_VIOLATED,
            id="one-vehicle-in-multi-vehicle-collision",
        ),
        pytest.param({"incident_hour_of_the_day": 24}, CHECK_VIOLATED, id="hour-24"),
        pytest.param(
            {"policy_number": 999_999_999}, FOREIGN_KEY_VIOLATED, id="unknown-policy"
        ),
        pytest.param(
            {"incident_severity": "Huge"}, INVALID_ENUM_VALUE, id="unknown-severity"
        ),
        pytest.param(
            {"has_injury": 1}, GENERATED_COLUMN_WRITE, id="write-to-generated-flag"
        ),
    ],
)
def test_inconsistent_claim_is_rejected(
    loader_connection: Connection,
    claims: Table,
    changes: dict[str, Any],
    expected_codes: set[int],
) -> None:
    assert error_code(loader_connection, claims, {**VALID_CLAIM, **changes}) in (
        expected_codes
    )


def test_duplicate_claim_id_is_rejected(
    loader_connection: Connection, claims: Table
) -> None:
    loader_connection.execute(claims.insert(), VALID_CLAIM)

    assert error_code(loader_connection, claims, VALID_CLAIM) in DUPLICATE_KEY


@pytest.mark.parametrize(
    ("changes", "expected_codes"),
    [
        pytest.param(
            {"insured_age": 20, "months_as_customer": 100},
            CHECK_VIOLATED,
            id="under-16-at-policy-start",
        ),
        pytest.param(
            {"csl_per_accident": 50_000},
            CHECK_VIOLATED,
            id="accident-limit-below-person-limit",
        ),
        pytest.param({"policy_annual_premium": 0}, CHECK_VIOLATED, id="zero-premium"),
        pytest.param(
            {"umbrella_limit": -1_000_000}, OUT_OF_RANGE, id="negative-umbrella-limit"
        ),
        pytest.param({"insured_sex": "OTHER"}, INVALID_ENUM_VALUE, id="unknown-sex"),
    ],
)
def test_inconsistent_policy_is_rejected(
    loader_connection: Connection,
    policies: Table,
    changes: dict[str, Any],
    expected_codes: set[int],
) -> None:
    assert error_code(loader_connection, policies, {**VALID_POLICY, **changes}) in (
        expected_codes
    )


def test_duplicate_policy_number_is_rejected(
    loader_connection: Connection, policies: Table
) -> None:
    loader_connection.execute(policies.insert(), VALID_POLICY)

    assert error_code(loader_connection, policies, VALID_POLICY) in DUPLICATE_KEY


def test_agent_can_read(agent_connection: Connection) -> None:
    count = agent_connection.execute(text("SELECT COUNT(*) FROM claims")).scalar_one()

    assert count >= 0


# Each statement is harmless even if the grant were wrong and it ran: the
# privilege check happens first, and nothing here matches or drops real data.
@pytest.mark.parametrize(
    "statement",
    [
        pytest.param("INSERT INTO policies (policy_number) VALUES (1)", id="insert"),
        pytest.param(
            "UPDATE claims SET witnesses = witnesses WHERE 1 = 0", id="update"
        ),
        pytest.param("DELETE FROM claims WHERE 1 = 0", id="delete"),
        pytest.param("DROP TABLE IF EXISTS privilege_probe", id="drop"),
        pytest.param("CREATE TABLE privilege_probe (id INT)", id="create"),
    ],
)
def test_agent_cannot_write(agent_connection: Connection, statement: str) -> None:
    with pytest.raises(DBAPIError) as raised:
        agent_connection.execute(text(statement))

    assert raised.value.orig.args[0] in COMMAND_DENIED
