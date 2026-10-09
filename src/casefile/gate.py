"""The human gate: nothing is written back without a person's approval.

Three steps, each done by a different party:

1. ``store_recommendation`` - the system records what it proposes.
2. ``record_decision``      - a person approves or rejects it.
3. ``write_back``           - the system records the final decision, and only
                              if step 2 ended in an approval.

The checks in this module give clear errors. They are not what makes the gate
safe: the tables in db/schema/004_human_gate.sql refuse a write-back without a
matching approval whatever this code does, and the database account used for
steps 1 and 3 is not allowed to insert an approval.

A person works the gate from the command line::

    python -m casefile.gate pending
    python -m casefile.gate approve 12 --approver "J. Smith"
    python -m casefile.gate reject 12 --approver "J. Smith" --comment "..."
    python -m casefile.gate write-back 7
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from casefile.config import ConfigurationError
from casefile.control.handoff import Recommendation
from casefile.control.state import CaseState
from casefile.db import create_domain_engine

HumanDecision = Literal["approved", "rejected"]


class GateError(RuntimeError):
    """The requested step of the gate cannot be carried out."""


class ApprovalRequiredError(GateError):
    """There is no approval that allows this write-back."""


@dataclass(frozen=True)
class WriteBack:
    """A final decision recorded against a claim."""

    write_back_id: int
    approval_id: int
    claim_number: str
    # True when this approval had been written back before; nothing was added.
    already_written: bool


def store_recommendation(
    connection: Connection,
    *,
    run_id: str,
    case_id: str,
    claim_number: str,
    policy_number: int,
    recommendation: Recommendation,
) -> int:
    """Record what a run proposes and return the recommendation id.

    Storing the same run twice returns the id from the first time.
    """
    existing = connection.execute(
        text("SELECT recommendation_id FROM recommendations WHERE run_id = :run_id"),
        {"run_id": run_id},
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    decision = recommendation.decision
    try:
        result = connection.execute(
            text(
                """
                INSERT INTO recommendations
                    (run_id, case_id, claim_number, policy_number, outcome,
                     reason, payable_amount, rationale)
                VALUES
                    (:run_id, :case_id, :claim_number, :policy_number, :outcome,
                     :reason, :payable_amount, :rationale)
                """
            ),
            {
                "run_id": run_id,
                "case_id": case_id,
                "claim_number": claim_number,
                "policy_number": policy_number,
                "outcome": decision.outcome,
                "reason": decision.reason,
                "payable_amount": decision.payable_amount or Decimal(0),
                "rationale": recommendation.rationale,
            },
        )
    except IntegrityError as error:
        raise GateError(
            f"Cannot store a recommendation for claim {claim_number!r}: "
            "the claim is not in the claims system."
        ) from error
    return result.lastrowid


def submit_for_approval(
    connection: Connection, *, run_id: str, state: CaseState
) -> int | None:
    """Put the result of a finished run in front of a person.

    Returns the recommendation id, or ``None`` for an escalated run: a run the
    system stopped has no recommendation to approve and goes to a person as
    it is.
    """
    if state.status == "running":
        raise GateError(f"Run {run_id} has not finished.")
    if state.status == "escalated":
        return None
    if state.extraction is None or state.recommendation is None:
        raise GateError(f"Run {run_id} completed without a recommendation.")
    form = state.extraction.claim_form
    return store_recommendation(
        connection,
        run_id=run_id,
        case_id=state.case_id,
        claim_number=form.claim_number,
        policy_number=form.policy_number,
        recommendation=state.recommendation,
    )


def record_decision(
    connection: Connection,
    *,
    recommendation_id: int,
    decision: HumanDecision,
    approver: str,
    comment: str | None = None,
) -> int:
    """Record a person's decision on a recommendation; return the approval id.

    A recommendation can be decided once. Deciding it again is an error, not
    a way to change the first decision.
    """
    if not approver.strip():
        raise GateError("A decision must name the person who made it.")
    try:
        result = connection.execute(
            text(
                """
                INSERT INTO approvals
                    (recommendation_id, claim_number, decision, approver, comment)
                SELECT recommendation_id, claim_number, :decision, :approver, :comment
                  FROM recommendations
                 WHERE recommendation_id = :recommendation_id
                """
            ),
            {
                "recommendation_id": recommendation_id,
                "decision": decision,
                "approver": approver.strip(),
                "comment": comment,
            },
        )
    except IntegrityError as error:
        raise GateError(
            f"Recommendation {recommendation_id} has already been decided."
        ) from error
    if result.rowcount == 0:
        raise GateError(f"There is no recommendation {recommendation_id}.")
    return result.lastrowid


def write_back(connection: Connection, approval_id: int | None) -> WriteBack:
    """Record the final decision of an approved recommendation.

    Raises ``ApprovalRequiredError`` unless ``approval_id`` names an approval
    that was granted. Calling it again with the same approval changes nothing
    and reports ``already_written``.
    """
    if approval_id is None:
        raise ApprovalRequiredError("A write-back needs an approval id.")
    approval = connection.execute(
        text(
            "SELECT decision, claim_number FROM approvals "
            "WHERE approval_id = :approval_id"
        ),
        {"approval_id": approval_id},
    ).one_or_none()
    if approval is None:
        raise ApprovalRequiredError(f"There is no approval {approval_id}.")
    if approval.decision != "approved":
        raise ApprovalRequiredError(
            f"Approval {approval_id} is a rejection; nothing may be written back."
        )

    existing = connection.execute(
        text(
            "SELECT write_back_id, approval_id FROM write_backs "
            "WHERE approval_id = :approval_id OR claim_number = :claim_number"
        ),
        {"approval_id": approval_id, "claim_number": approval.claim_number},
    ).one_or_none()
    if existing is not None:
        if existing.approval_id != approval_id:
            raise GateError(
                f"Claim {approval.claim_number} already has a final decision "
                f"(write-back {existing.write_back_id})."
            )
        return WriteBack(
            existing.write_back_id, approval_id, approval.claim_number, True
        )

    result = connection.execute(
        text(
            "INSERT INTO write_backs (approval_id, claim_number) "
            "VALUES (:approval_id, :claim_number)"
        ),
        {"approval_id": approval_id, "claim_number": approval.claim_number},
    )
    return WriteBack(result.lastrowid, approval_id, approval.claim_number, False)


def pending_recommendations(connection: Connection) -> list[dict[str, object]]:
    """Return the recommendations no person has decided yet, oldest first."""
    rows = connection.execute(
        text(
            """
            SELECT r.recommendation_id, r.case_id, r.claim_number, r.outcome,
                   r.reason, r.payable_amount, r.rationale
              FROM recommendations AS r
              LEFT JOIN approvals AS a USING (recommendation_id)
             WHERE a.approval_id IS NULL
             ORDER BY r.recommendation_id
            """
        )
    )
    return [dict(row._mapping) for row in rows]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pending", help="list recommendations awaiting a decision")
    for name, summary in (("approve", "approve"), ("reject", "reject")):
        command = commands.add_parser(name, help=f"{summary} a recommendation")
        command.add_argument("recommendation_id", type=int)
        command.add_argument("--approver", required=True, help="your name")
        command.add_argument("--comment")
    command = commands.add_parser(
        "write-back", help="record the final decision of an approval"
    )
    command.add_argument("approval_id", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one gate command. Returns an exit code."""
    args = parse_args(argv)
    # Deciding is the person's job and uses the person's account; the
    # write-back is the system's.
    role = "writer" if args.command == "write-back" else "approver"
    try:
        engine = create_domain_engine(role)
        with engine.begin() as connection:
            if args.command == "pending":
                rows = pending_recommendations(connection)
                for row in rows:
                    print(
                        f"{row['recommendation_id']:>5}  {row['case_id']}  "
                        f"{row['claim_number']}  {row['outcome']} "
                        f"({row['reason']}, {row['payable_amount']})"
                    )
                print(f"{len(rows)} recommendation(s) awaiting a decision.")
            elif args.command == "write-back":
                done = write_back(connection, args.approval_id)
                state = "was already recorded" if done.already_written else "recorded"
                print(
                    f"Write-back {done.write_back_id} for {done.claim_number} {state}."
                )
            else:
                decision = "approved" if args.command == "approve" else "rejected"
                approval_id = record_decision(
                    connection,
                    recommendation_id=args.recommendation_id,
                    decision=decision,
                    approver=args.approver,
                    comment=args.comment,
                )
                print(f"Recorded as approval {approval_id} ({decision}).")
    except (GateError, ConfigurationError) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
