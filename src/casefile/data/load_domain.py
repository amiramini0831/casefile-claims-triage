"""Load policies and claims into the domain database.

Run after scripts/setup_database.py::

    python -m casefile.data.load_domain

The load replaces the content of both tables inside one transaction and checks
the result before committing, so the database holds either the complete new
data or what it held before. Running it again is safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, MetaData, text

from casefile.config import RAW_DATA_DIR, REPORTS_DIR, ConfigurationError
from casefile.data.transform import (
    RAW_FILE_NAME,
    DomainTables,
    RawDataError,
    read_raw_claims,
    transform,
)
from casefile.db import create_domain_engine

REPORT_PATH = REPORTS_DIR / "domain_load_report.json"

_SUMMARY_QUERY = text(
    """
    SELECT
        (SELECT COUNT(*) FROM policies) AS policies,
        (SELECT COUNT(*) FROM claims) AS claims,
        (SELECT COALESCE(SUM(total_claim_amount), 0) FROM claims)
            AS total_claim_amount,
        (SELECT COALESCE(SUM(has_injury), 0) FROM claims) AS claims_with_injury,
        (SELECT COUNT(*) FROM claims WHERE collision_type IS NULL)
            AS claims_without_collision_type,
        (SELECT COUNT(*) FROM claims WHERE police_report_available IS NULL)
            AS claims_with_unknown_police_report,
        (SELECT COUNT(*)
           FROM claims AS c
           JOIN policies AS p USING (policy_number)
          WHERE c.incident_date < p.policy_start_date)
            AS incidents_before_policy_start
    """
)


class LoadVerificationError(RuntimeError):
    """The database content does not match what was meant to be loaded."""


def expected_summary(tables: DomainTables) -> dict[str, int | Decimal]:
    """Compute, from the records, what the database must contain after a load."""
    claims = tables.claims
    return {
        "policies": len(tables.policies),
        "claims": len(claims),
        "total_claim_amount": sum(
            (claim.total_claim_amount for claim in claims), Decimal(0)
        ),
        "claims_with_injury": sum(claim.injury_claim > 0 for claim in claims),
        "claims_without_collision_type": sum(
            claim.collision_type is None for claim in claims
        ),
        "claims_with_unknown_police_report": sum(
            claim.police_report_available is None for claim in claims
        ),
        "incidents_before_policy_start": 0,
    }


def replace_domain_tables(connection: Connection, tables: DomainTables) -> None:
    """Replace the content of ``policies`` and ``claims`` with ``tables``.

    Rows are removed with DELETE, not TRUNCATE: TRUNCATE commits implicitly in
    MySQL and would break the all-or-nothing guarantee of the caller's
    transaction.
    """
    if not tables.policies:
        raise LoadVerificationError("There is nothing to load.")
    metadata = MetaData()
    metadata.reflect(bind=connection, only=("policies", "claims"))
    policies = metadata.tables["policies"]
    claims = metadata.tables["claims"]

    connection.execute(claims.delete())
    connection.execute(policies.delete())
    connection.execute(
        policies.insert(), [record.model_dump() for record in tables.policies]
    )
    connection.execute(
        claims.insert(), [record.model_dump() for record in tables.claims]
    )


def verify_load(connection: Connection, tables: DomainTables) -> None:
    """Compare the database with the records and raise on any difference."""
    row = connection.execute(_SUMMARY_QUERY).one()
    expected = expected_summary(tables)
    actual = {name: type(expected[name])(getattr(row, name)) for name in expected}
    differences = [
        f"{name}: expected {expected[name]}, found {actual[name]}"
        for name in expected
        if expected[name] != actual[name]
    ]
    if differences:
        raise LoadVerificationError("; ".join(differences))


def build_report(tables: DomainTables, source_name: str) -> dict[str, Any]:
    """Describe what was loaded and what was left out, for the repository."""
    summary = expected_summary(tables)
    return {
        "source_file": source_name,
        "raw_rows": tables.raw_row_count,
        "loaded": {
            "policies": summary["policies"],
            "claims": summary["claims"],
            "total_claim_amount": str(summary["total_claim_amount"]),
        },
        "rejected": [asdict(row) for row in tables.rejected],
    }


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> Path:
    """Write the load report as JSON and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_DATA_DIR / RAW_FILE_NAME,
        help="path to insurance_claims.csv (default: data/raw/insurance_claims.csv)",
    )
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="load the file even if it is not a known version of the dataset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the load and report the outcome. Returns an exit code."""
    args = parse_args(argv)
    try:
        raw_rows = read_raw_claims(
            args.raw_path, verify_checksum=not args.allow_unverified_source
        )
        tables = transform(raw_rows)
        engine = create_domain_engine("loader")
    except (RawDataError, FileNotFoundError, ConfigurationError) as error:
        print(f"Cannot load: {error}", file=sys.stderr)
        return 1

    with engine.begin() as connection:
        replace_domain_tables(connection, tables)
        verify_load(connection, tables)
    engine.dispose()

    report_path = write_report(build_report(tables, args.raw_path.name))
    print(
        f"Loaded {len(tables.policies)} policies and {len(tables.claims)} claims "
        f"from {tables.raw_row_count} raw rows."
    )
    for row in tables.rejected:
        print(f"Rejected policy {row.policy_number}: {row.rule} ({row.detail})")
    print(f"Report written to {report_path.relative_to(REPORTS_DIR.parent)}")
    print(
        "Synthetic prior claims were removed with their policies. "
        "Regenerate them with: python -m casefile.data.prior_claims"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
