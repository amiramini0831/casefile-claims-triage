"""Turn the raw insurance_claims.csv into policy and claim records.

Nothing here touches the database, so every rule can be unit tested.

The raw file is read as plain text on purpose. Its columns contain the literal
strings ``None`` (a real category) and ``?`` (a missing-value marker), and a
reader that guesses types would turn the first into a missing value.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from casefile.data.records import ClaimRecord, PolicyRecord

RAW_FILE_NAME = "insurance_claims.csv"
RAW_SOURCE_URL = "https://github.com/mwitiderrick/insurancedata"

# The raw file is not in the repository, so the loader pins the versions it
# was built and tested against. Both hold the same 1000 rows.
KNOWN_RAW_SHA256: dict[str, str] = {
    "309dae61bc98a5b3f5d137d922fe47ecf1aef1f522e0a2de373b2b21eb2b7b36": (
        "github.com/mwitiderrick/insurancedata, master branch"
    ),
    "be11b9370c0cda353a4519832803d07bb90a494e91dfd9d56a85aa4093ba5378": (
        "same rows with a trailing empty _c39 column"
    ),
}

MISSING_MARKER = "?"
AUTO_MAKE_CORRECTIONS: dict[str, str] = {"Accura": "Acura", "Suburu": "Subaru"}
POLICE_REPORT_VALUES: dict[str, bool | None] = {
    "YES": True,
    "NO": False,
    MISSING_MARKER: None,
}
CSL_PATTERN = re.compile(r"(\d+)/(\d+)")
CSL_UNIT = 1000  # policy_csl is written in thousands of dollars

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "months_as_customer",
        "age",
        "policy_number",
        "policy_state",
        "policy_csl",
        "policy_deductable",
        "policy_annual_premium",
        "umbrella_limit",
        "insured_sex",
        "incident_date",
        "incident_type",
        "collision_type",
        "incident_severity",
        "authorities_contacted",
        "incident_state",
        "incident_city",
        "incident_location",
        "incident_hour_of_the_day",
        "number_of_vehicles_involved",
        "witnesses",
        "police_report_available",
        "total_claim_amount",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "auto_make",
        "auto_model",
        "auto_year",
    }
)


class RawDataError(ValueError):
    """The raw file is not the dataset this loader was built for."""


@dataclass(frozen=True)
class RejectRule:
    """A named reason to leave a raw row out of the domain tables.

    Rows are only ever dropped by a rule listed in ``REJECT_RULES``. Any other
    problem stops the load, so nothing disappears without a recorded reason.
    """

    name: str
    column: str
    applies: Callable[[Any], bool]


REJECT_RULES: tuple[RejectRule, ...] = (
    # A coverage limit cannot be negative, and guessing the intended value
    # would be inventing data.
    RejectRule("negative_umbrella_limit", "umbrella_limit", lambda value: value < 0),
)


@dataclass(frozen=True)
class RejectedRow:
    """A raw row that was left out, and why."""

    policy_number: int
    rule: str
    detail: str


@dataclass(frozen=True)
class DomainTables:
    """The result of transforming the raw file."""

    policies: tuple[PolicyRecord, ...]
    claims: tuple[ClaimRecord, ...]
    rejected: tuple[RejectedRow, ...]
    raw_row_count: int


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_claims(
    path: Path, *, verify_checksum: bool = True
) -> list[dict[str, str]]:
    """Read the raw CSV as text, one dict per row.

    With ``verify_checksum`` the file must be one of the pinned versions, so a
    different edition of the dataset cannot silently produce different tables.
    """
    if verify_checksum:
        digest = file_sha256(path)
        if digest not in KNOWN_RAW_SHA256:
            raise RawDataError(
                f"{path.name} has SHA-256 {digest}, which is not a known "
                f"version of the dataset. Download it again from {RAW_SOURCE_URL} "
                "or pass --allow-unverified-source."
            )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for line, row in enumerate(rows, start=2):
        # DictReader marks surplus cells with a None key and missing cells
        # with a None value.
        if None in row or None in row.values():
            raise RawDataError(f"line {line}: wrong number of columns")
    return rows


def subtract_months(day: date, months: int) -> date:
    """Return the date ``months`` calendar months before ``day``.

    The day of the month is clipped to the length of the target month, so
    31 March minus one month is the last day of February.
    """
    month_index = day.year * 12 + (day.month - 1) - months
    year, month_offset = divmod(month_index, 12)
    month = month_offset + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def transform(raw_rows: Iterable[Mapping[str, str]]) -> DomainTables:
    """Build the policy and claim records from raw rows.

    Claim ids are numbered by incident date and then policy number, so they
    do not depend on the order of the rows in the file.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRow] = []
    seen_policy_numbers: set[int] = set()
    raw_row_count = 0

    for line, raw in enumerate(raw_rows, start=2):  # line 1 is the header
        raw_row_count += 1
        row = _parse_row(raw, line)
        policy_number = row["policy_number"]
        if policy_number in seen_policy_numbers:
            raise RawDataError(f"line {line}: duplicate policy_number {policy_number}")
        seen_policy_numbers.add(policy_number)

        rule = next((r for r in REJECT_RULES if r.applies(row[r.column])), None)
        if rule is None:
            accepted.append(row)
        else:
            rejected.append(
                RejectedRow(
                    policy_number=policy_number,
                    rule=rule.name,
                    detail=f"{rule.column} = {row[rule.column]}",
                )
            )

    accepted.sort(key=lambda row: (row["incident_date"], row["policy_number"]))
    policies: list[PolicyRecord] = []
    claims: list[ClaimRecord] = []
    for sequence, row in enumerate(accepted, start=1):
        try:
            policies.append(_build_policy(row))
            claims.append(_build_claim(row, sequence))
        except ValidationError as error:
            raise RawDataError(f"policy {row['policy_number']}: {error}") from error
    policies.sort(key=lambda policy: policy.policy_number)

    return DomainTables(
        policies=tuple(policies),
        claims=tuple(claims),
        rejected=tuple(rejected),
        raw_row_count=raw_row_count,
    )


def _parse_row(raw: Mapping[str, str], line: int) -> dict[str, Any]:
    """Convert the cells this loader uses from text to Python values."""
    missing = REQUIRED_COLUMNS - raw.keys()
    if missing:
        raise RawDataError(f"line {line}: missing columns {sorted(missing)}")
    try:
        csl_per_person, csl_per_accident = _parse_csl(raw["policy_csl"])
        collision_type = raw["collision_type"]
        return {
            "policy_number": _to_int(raw, "policy_number"),
            "policy_state": raw["policy_state"],
            "months_as_customer": _to_int(raw, "months_as_customer"),
            "csl_per_person": csl_per_person,
            "csl_per_accident": csl_per_accident,
            "policy_deductible": _to_int(raw, "policy_deductable"),
            "policy_annual_premium": _to_decimal(raw, "policy_annual_premium"),
            "umbrella_limit": _to_int(raw, "umbrella_limit"),
            "insured_age": _to_int(raw, "age"),
            "insured_sex": raw["insured_sex"],
            "auto_make": AUTO_MAKE_CORRECTIONS.get(raw["auto_make"], raw["auto_make"]),
            "auto_model": raw["auto_model"],
            "auto_year": _to_int(raw, "auto_year"),
            "incident_date": date.fromisoformat(raw["incident_date"]),
            "incident_type": raw["incident_type"],
            "collision_type": (
                None if collision_type == MISSING_MARKER else collision_type
            ),
            "incident_severity": raw["incident_severity"],
            "authorities_contacted": raw["authorities_contacted"],
            "police_report_available": _parse_police_report(
                raw["police_report_available"]
            ),
            "incident_state": raw["incident_state"],
            "incident_city": raw["incident_city"],
            "incident_location": raw["incident_location"],
            "incident_hour_of_the_day": _to_int(raw, "incident_hour_of_the_day"),
            "number_of_vehicles_involved": _to_int(raw, "number_of_vehicles_involved"),
            "witnesses": _to_int(raw, "witnesses"),
            "total_claim_amount": _to_decimal(raw, "total_claim_amount"),
            "injury_claim": _to_decimal(raw, "injury_claim"),
            "property_claim": _to_decimal(raw, "property_claim"),
            "vehicle_claim": _to_decimal(raw, "vehicle_claim"),
        }
    except ValueError as error:
        raise RawDataError(f"line {line}: {error}") from error


def _to_int(raw: Mapping[str, str], column: str) -> int:
    try:
        return int(raw[column])
    except ValueError:
        raise ValueError(f"{column} = {raw[column]!r} is not an integer") from None


def _to_decimal(raw: Mapping[str, str], column: str) -> Decimal:
    try:
        return Decimal(raw[column])
    except InvalidOperation:
        raise ValueError(f"{column} = {raw[column]!r} is not a number") from None


def _parse_csl(value: str) -> tuple[int, int]:
    """Split a combined single limit such as ``250/500`` into two dollar amounts."""
    match = CSL_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"policy_csl = {value!r} is not of the form 250/500")
    return int(match[1]) * CSL_UNIT, int(match[2]) * CSL_UNIT


def _parse_police_report(value: str) -> bool | None:
    try:
        return POLICE_REPORT_VALUES[value]
    except KeyError:
        raise ValueError(
            f"police_report_available = {value!r} is not YES, NO or ?"
        ) from None


def _build_policy(row: Mapping[str, Any]) -> PolicyRecord:
    # policy_bind_date in the raw file contradicts months_as_customer, so the
    # start date is derived from the tenure instead.
    start_date = subtract_months(row["incident_date"], row["months_as_customer"])
    return PolicyRecord(
        policy_start_date=start_date,
        **{name: row[name] for name in PolicyRecord.model_fields if name in row},
    )


def _build_claim(row: Mapping[str, Any], sequence: int) -> ClaimRecord:
    claim_id = f"CLM-{row['incident_date'].year}-{sequence:05d}"
    return ClaimRecord(
        claim_id=claim_id,
        **{name: row[name] for name in ClaimRecord.model_fields if name in row},
    )
