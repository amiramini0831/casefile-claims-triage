"""The agents' read access to the domain database.

These are the investigator's tools: three fixed, parameterised queries. The
model never writes SQL. The connection is expected to use the ``agent``
account, which can read policies, prior claims and the repair cost reference
and nothing else, so even a mistake here cannot reach the current claims.
"""

from __future__ import annotations

from sqlalchemy import Connection, text

from casefile.data.records import (
    PolicyRecord,
    PriorClaimRecord,
    RepairCostReferenceRecord,
)

_POLICY_QUERY = text(
    f"SELECT {', '.join(PolicyRecord.model_fields)} "  # noqa: S608 - fixed names
    "FROM policies WHERE policy_number = :policy_number"
)
_PRIOR_CLAIMS_QUERY = text(
    f"SELECT {', '.join(PriorClaimRecord.model_fields)} "  # noqa: S608 - fixed names
    "FROM prior_claims WHERE policy_number = :policy_number "
    "ORDER BY incident_date, prior_claim_id"
)
_REFERENCE_QUERY = text(
    f"SELECT {', '.join(RepairCostReferenceRecord.model_fields)} "  # noqa: S608
    "FROM repair_cost_reference "
    "WHERE incident_type = :incident_type AND incident_severity = :incident_severity"
)
# policy_number is an unsigned 32-bit column; a larger number cannot match.
MAX_POLICY_NUMBER = 2**32 - 1


class SqlDomainLookup:
    """Looks up policies, their history and the repair cost range in MySQL."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def find_policy(self, policy_number: int) -> PolicyRecord | None:
        """Return the policy, or ``None`` if the number matches no policy."""
        if not 0 < policy_number <= MAX_POLICY_NUMBER:
            return None
        row = self._connection.execute(
            _POLICY_QUERY, {"policy_number": policy_number}
        ).one_or_none()
        return None if row is None else PolicyRecord(**row._mapping)

    def find_prior_claims(self, policy_number: int) -> tuple[PriorClaimRecord, ...]:
        """Return the earlier claims of a policy, oldest first."""
        if not 0 < policy_number <= MAX_POLICY_NUMBER:
            return ()
        rows = self._connection.execute(
            _PRIOR_CLAIMS_QUERY, {"policy_number": policy_number}
        )
        return tuple(PriorClaimRecord(**row._mapping) for row in rows)

    def find_reference(
        self, incident_type: str, incident_severity: str
    ) -> RepairCostReferenceRecord | None:
        """Return the repair cost range for this kind of claim, if any."""
        row = self._connection.execute(
            _REFERENCE_QUERY,
            {"incident_type": incident_type, "incident_severity": incident_severity},
        ).one_or_none()
        return None if row is None else RepairCostReferenceRecord(**row._mapping)
