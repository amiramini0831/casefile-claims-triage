"""Derive the repair cost reference table from the loaded claims.

Run after casefile.data.load_domain::

    python -m casefile.data.repair_cost_reference

The table is hypothetical. For each incident type and severity it stores the
range that holds the middle 90% of the ``vehicle_claim`` amounts in this
portfolio, plus the median. Because the ranges come from the same claims they
are later compared with, about nine claims in ten fall inside by construction;
the table is useful for spotting amounts that were inflated afterwards, not
for judging the original data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args

from sqlalchemy import Connection, MetaData, func, select

from casefile.config import RAW_DATA_DIR, REPORTS_DIR, ConfigurationError
from casefile.data.records import (
    ClaimRecord,
    IncidentSeverity,
    IncidentType,
    RepairCostReferenceRecord,
)
from casefile.data.transform import (
    RAW_FILE_NAME,
    RawDataError,
    read_raw_claims,
    transform,
)
from casefile.db import create_domain_engine

REPORT_PATH = REPORTS_DIR / "repair_cost_reference.json"

LOW_PERCENTILE = 5
TYPICAL_PERCENTILE = 50
HIGH_PERCENTILE = 95
# A range from fewer claims than this would mostly reflect chance, so such a
# combination gets no reference row at all.
MIN_SAMPLE_SIZE = 30

_TYPE_ORDER = {name: index for index, name in enumerate(get_args(IncidentType))}
_SEVERITY_ORDER = {name: index for index, name in enumerate(get_args(IncidentSeverity))}


def percentile(sorted_values: Sequence[Decimal], percent: int) -> Decimal:
    """Return the nearest-rank percentile of ``sorted_values``.

    Nearest rank always returns an amount that was actually observed, which
    keeps the reference easy to explain.
    """
    if not sorted_values:
        raise ValueError("cannot take a percentile of no values")
    rank = max(1, -(-percent * len(sorted_values) // 100))  # ceiling division
    return sorted_values[rank - 1]


def build_reference(
    claims: Sequence[ClaimRecord],
) -> tuple[RepairCostReferenceRecord, ...]:
    """Build one reference row per incident type and severity with enough claims."""
    amounts: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
    for claim in claims:
        amounts[claim.incident_type, claim.incident_severity].append(
            claim.vehicle_claim
        )

    records: list[RepairCostReferenceRecord] = []
    for (incident_type, incident_severity), values in amounts.items():
        if len(values) < MIN_SAMPLE_SIZE:
            continue
        values.sort()
        records.append(
            RepairCostReferenceRecord(
                incident_type=incident_type,
                incident_severity=incident_severity,
                low_amount=percentile(values, LOW_PERCENTILE),
                typical_amount=percentile(values, TYPICAL_PERCENTILE),
                high_amount=percentile(values, HIGH_PERCENTILE),
                sample_size=len(values),
            )
        )
    records.sort(
        key=lambda record: (
            _TYPE_ORDER[record.incident_type],
            _SEVERITY_ORDER[record.incident_severity],
        )
    )
    return tuple(records)


def replace_reference(
    connection: Connection, records: Sequence[RepairCostReferenceRecord]
) -> None:
    """Replace the content of ``repair_cost_reference`` with ``records``."""
    metadata = MetaData()
    metadata.reflect(bind=connection, only=("repair_cost_reference",))
    table = metadata.tables["repair_cost_reference"]

    connection.execute(table.delete())
    connection.execute(table.insert(), [record.model_dump() for record in records])

    stored = connection.execute(select(func.count()).select_from(table)).scalar_one()
    if stored != len(records):
        raise RuntimeError(f"{stored} rows stored, expected {len(records)}")


def build_report(records: Sequence[RepairCostReferenceRecord]) -> dict[str, Any]:
    """Describe the reference table, for the repository."""
    return {
        "hypothetical": True,
        "based_on": "vehicle_claim of the loaded claims",
        "percentiles": {
            "low": LOW_PERCENTILE,
            "typical": TYPICAL_PERCENTILE,
            "high": HIGH_PERCENTILE,
        },
        "min_sample_size": MIN_SAMPLE_SIZE,
        "rows": [record.model_dump(mode="json") for record in records],
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="use the file even if it is not a known version of the dataset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, store and report the reference table. Returns an exit code."""
    args = parse_args(argv)
    try:
        raw_rows = read_raw_claims(
            args.claims_path, verify_checksum=not args.allow_unverified_source
        )
        records = build_reference(transform(raw_rows).claims)
        engine = create_domain_engine("loader")
    except (RawDataError, FileNotFoundError, ConfigurationError) as error:
        print(f"Cannot build the repair cost reference: {error}", file=sys.stderr)
        return 1

    with engine.begin() as connection:
        replace_reference(connection, records)
    engine.dispose()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(build_report(records), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Stored {len(records)} hypothetical repair cost ranges.")
    print(f"Report written to {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
