"""Generate a synthetic claim history for every policy.

Run after casefile.data.load_domain::

    python -m casefile.data.prior_claims

Neither source dataset contains earlier claims, so this module invents them.
The result is reproducible: the same seed and the same input files always
give the same rows.

What is real and what is not:

* How many earlier claims a policy has follows the ``PastNumberOfClaims``
  column of fraud_oracle.csv, capped by how long the customer has been insured.
* The date of each earlier claim is drawn at random between the start of the
  policy and its current incident.
* The type, severity and amounts of each earlier claim are copied together
  from the current claim of another, randomly chosen policy. Amounts are
  therefore at 2015 levels whatever the date, with no adjustment for inflation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import Connection, MetaData, text

from casefile.config import RAW_DATA_DIR, REPORTS_DIR, ConfigurationError
from casefile.data.records import ClaimRecord, PolicyRecord, PriorClaimRecord
from casefile.data.transform import (
    RAW_FILE_NAME,
    RawDataError,
    file_sha256,
    read_raw_claims,
    transform,
)
from casefile.db import create_domain_engine

ORACLE_FILE_NAME = "fraud_oracle.csv"
ORACLE_SOURCE_URL = (
    "https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection"
)
KNOWN_ORACLE_SHA256: frozenset[str] = frozenset(
    {"8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6"}
)
REPORT_PATH = REPORTS_DIR / "prior_claims_report.json"

SEED = 20150301

# PastNumberOfClaims is binned in the source; a bin becomes one of these
# counts with equal probability.
PAST_CLAIM_COUNTS: dict[str, tuple[int, ...]] = {
    "none": (0,),
    "1": (1,),
    "2 to 4": (2, 3, 4),
    "more than 4": (5, 6),
}
# A customer gets at most one earlier claim per full year of tenure.
MONTHS_PER_PRIOR_CLAIM = 12
# One row of fraud_oracle.csv has zeros where its claim date should be.
CORRUPT_ROW_MARKER = ("MonthClaimed", "0")


class GenerationError(RuntimeError):
    """The generated history breaks one of its own consistency rules."""


def read_past_claim_bins(path: Path, *, verify_checksum: bool = True) -> list[str]:
    """Return the ``PastNumberOfClaims`` value of every usable oracle row."""
    if verify_checksum and file_sha256(path) not in KNOWN_ORACLE_SHA256:
        raise RawDataError(
            f"{path.name} is not a known version of the dataset. Download it "
            f"again from {ORACLE_SOURCE_URL} or pass --allow-unverified-source."
        )
    corrupt_column, corrupt_value = CORRUPT_ROW_MARKER
    bins: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if row.get(corrupt_column) == corrupt_value:
                continue
            value = row.get("PastNumberOfClaims")
            if value not in PAST_CLAIM_COUNTS:
                raise RawDataError(
                    f"line {line}: unexpected PastNumberOfClaims {value!r}"
                )
            bins.append(value)
    if not bins:
        raise RawDataError(f"{path.name} has no usable rows")
    return bins


def generate_prior_claims(
    policies: Sequence[PolicyRecord],
    claims: Sequence[ClaimRecord],
    past_claim_bins: Sequence[str],
    seed: int = SEED,
) -> tuple[PriorClaimRecord, ...]:
    """Build the synthetic history of all ``policies``.

    ``claims`` are the current claims; each policy's history ends before its
    own current claim and borrows its content from the claims of others.
    """
    rng = random.Random(seed)  # noqa: S311 - simulation, not cryptography
    current_incident = {claim.policy_number: claim.incident_date for claim in claims}
    donors = sorted(claims, key=lambda claim: claim.claim_id)
    records: list[PriorClaimRecord] = []

    for policy in sorted(policies, key=lambda policy: policy.policy_number):
        wanted = rng.choice(PAST_CLAIM_COUNTS[rng.choice(past_claim_bins)])
        allowed = policy.months_as_customer // MONTHS_PER_PRIOR_CLAIM
        count = min(wanted, allowed)
        if count == 0:
            continue

        # Dates strictly between the policy start and the current incident.
        window_days = (
            current_incident[policy.policy_number] - policy.policy_start_date
        ).days
        offsets = sorted(rng.sample(range(1, window_days), count))
        for offset in offsets:
            donor = _pick_donor(rng, donors, policy)
            records.append(
                PriorClaimRecord(
                    prior_claim_id=f"PCL-{len(records) + 1:05d}",
                    policy_number=policy.policy_number,
                    incident_date=policy.policy_start_date + timedelta(days=offset),
                    incident_type=donor.incident_type,
                    incident_severity=donor.incident_severity,
                    total_claim_amount=donor.total_claim_amount,
                    injury_claim=donor.injury_claim,
                    property_claim=donor.property_claim,
                    vehicle_claim=donor.vehicle_claim,
                )
            )
    return tuple(records)


def _pick_donor(
    rng: random.Random, donors: Sequence[ClaimRecord], policy: PolicyRecord
) -> ClaimRecord:
    """Pick another policy's claim whose injury amount fits this policy's limit."""
    for _ in range(100):
        donor = rng.choice(donors)
        if (
            donor.policy_number != policy.policy_number
            and donor.injury_claim <= policy.csl_per_person
        ):
            return donor
    raise GenerationError(
        f"policy {policy.policy_number}: no other claim fits its coverage limit"
    )


def find_violations(
    policies: Sequence[PolicyRecord],
    claims: Sequence[ClaimRecord],
    prior_claims: Sequence[PriorClaimRecord],
) -> list[str]:
    """Check the generated history against its consistency rules.

    Returns one message per broken rule; an empty list means the history is
    consistent with the policies and current claims.
    """
    policy_by_number = {policy.policy_number: policy for policy in policies}
    current_incident = {claim.policy_number: claim.incident_date for claim in claims}
    violations: list[str] = []
    per_policy: Counter[int] = Counter()
    seen_days: set[tuple[int, Any]] = set()

    for prior in prior_claims:
        label = f"{prior.prior_claim_id} (policy {prior.policy_number})"
        policy = policy_by_number.get(prior.policy_number)
        if policy is None:
            violations.append(f"{label}: unknown policy")
            continue
        per_policy[prior.policy_number] += 1

        incident = current_incident[prior.policy_number]
        if not policy.policy_start_date < prior.incident_date < incident:
            violations.append(
                f"{label}: {prior.incident_date} is outside "
                f"{policy.policy_start_date}..{incident}"
            )
        day = (prior.policy_number, prior.incident_date)
        if day in seen_days:
            violations.append(f"{label}: second claim on {prior.incident_date}")
        seen_days.add(day)
        if prior.injury_claim > policy.csl_per_person:
            violations.append(f"{label}: injury_claim exceeds the coverage limit")

    for policy_number, count in per_policy.items():
        allowed = (
            policy_by_number[policy_number].months_as_customer // MONTHS_PER_PRIOR_CLAIM
        )
        if count > allowed:
            violations.append(
                f"policy {policy_number}: {count} prior claims, tenure allows {allowed}"
            )
    return violations


_DATABASE_CHECK = text(
    """
    SELECT
        (SELECT COUNT(*) FROM policies) AS policies,
        (SELECT COUNT(*) FROM prior_claims) AS prior_claims,
        (SELECT COALESCE(SUM(total_claim_amount), 0) FROM prior_claims)
            AS total_claim_amount,
        (SELECT COUNT(*)
           FROM prior_claims AS pc
           JOIN policies AS p USING (policy_number)
           JOIN claims AS c USING (policy_number)
          WHERE pc.incident_date <= p.policy_start_date
             OR pc.incident_date >= c.incident_date
             OR pc.injury_claim > p.csl_per_person) AS inconsistent_rows
    """
)


def replace_prior_claims(
    connection: Connection,
    policies: Sequence[PolicyRecord],
    prior_claims: Sequence[PriorClaimRecord],
) -> None:
    """Replace the ``prior_claims`` table and check the result in the database."""
    table = MetaData()
    table.reflect(bind=connection, only=("prior_claims",))
    prior_claims_table = table.tables["prior_claims"]

    connection.execute(prior_claims_table.delete())
    if prior_claims:
        connection.execute(
            prior_claims_table.insert(),
            [record.model_dump() for record in prior_claims],
        )

    row = connection.execute(_DATABASE_CHECK).one()
    expected_total = sum(
        (record.total_claim_amount for record in prior_claims), Decimal(0)
    )
    problems = []
    if row.policies != len(policies):
        problems.append(
            f"database has {row.policies} policies, expected {len(policies)}; "
            "run casefile.data.load_domain first"
        )
    if row.prior_claims != len(prior_claims):
        problems.append(f"{row.prior_claims} rows stored, expected {len(prior_claims)}")
    if row.total_claim_amount != expected_total:
        problems.append("stored amounts do not add up to the generated amounts")
    if row.inconsistent_rows:
        problems.append(f"{row.inconsistent_rows} rows break a consistency rule")
    if problems:
        raise GenerationError("; ".join(problems))


def build_report(
    policies: Sequence[PolicyRecord],
    prior_claims: Sequence[PriorClaimRecord],
    seed: int,
) -> dict[str, Any]:
    """Describe the generated history, for the repository."""
    per_policy = Counter(record.policy_number for record in prior_claims)
    distribution = Counter(per_policy.get(p.policy_number, 0) for p in policies)
    total = sum((record.total_claim_amount for record in prior_claims), Decimal(0))
    return {
        "synthetic": True,
        "seed": seed,
        "policies": len(policies),
        "prior_claims": len(prior_claims),
        "policies_by_number_of_prior_claims": {
            str(count): distribution[count] for count in sorted(distribution)
        },
        "total_claim_amount": str(total),
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--oracle-path", type=Path, default=RAW_DATA_DIR / ORACLE_FILE_NAME
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="use the files even if they are not known versions of the datasets",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate, check and store the history. Returns an exit code."""
    args = parse_args(argv)
    verify = not args.allow_unverified_source
    try:
        tables = transform(read_raw_claims(args.claims_path, verify_checksum=verify))
        bins = read_past_claim_bins(args.oracle_path, verify_checksum=verify)
        prior_claims = generate_prior_claims(
            tables.policies, tables.claims, bins, args.seed
        )
        violations = find_violations(tables.policies, tables.claims, prior_claims)
        if violations:
            raise GenerationError("; ".join(violations[:5]))
        engine = create_domain_engine("loader")
    except (
        RawDataError,
        FileNotFoundError,
        ConfigurationError,
        GenerationError,
        ValidationError,
    ) as error:
        print(f"Cannot generate prior claims: {error}", file=sys.stderr)
        return 1

    with engine.begin() as connection:
        replace_prior_claims(connection, tables.policies, prior_claims)
    engine.dispose()

    report = build_report(tables.policies, prior_claims, args.seed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Stored {len(prior_claims)} synthetic prior claims "
        f"for {len(policies_with_history(prior_claims))} of "
        f"{len(tables.policies)} policies (seed {args.seed})."
    )
    print(f"Report written to {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")
    return 0


def policies_with_history(prior_claims: Sequence[PriorClaimRecord]) -> set[int]:
    """Return the policy numbers that have at least one prior claim."""
    return {record.policy_number for record in prior_claims}


if __name__ == "__main__":
    sys.exit(main())
