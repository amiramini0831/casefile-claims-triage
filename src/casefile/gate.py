"""The human gate: nothing is written back without a person's approval.

Three steps, each done by a different party:

1. ``store_recommendation`` - the system records what it proposes: the
                              decision of the fixed rules and the decision
                              the model reached on its own, side by side.
2. ``record_decision``      - a person rejects the recommendation, or approves
                              it and says which of the two decisions is to be
                              carried out.
3. ``write_back``           - the system records the final decision, and only
                              if step 2 ended in an approval.

The checks in this module give clear errors. They are not what makes the gate
safe: the tables in db/schema/004_human_gate.sql refuse a write-back without a
matching approval whatever this code does, and the database account used for
steps 1 and 3 is not allowed to insert an approval.

A person works the gate from the command line::

    python -m casefile.gate pending
    python -m casefile.gate approve 12 --choose rule --approver "J. Smith"
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
ChosenDecision = Literal["rule", "model"]


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
    # Which of the two decisions the person chose, and what it says.
    chosen_decision: ChosenDecision
    outcome: str
    payable_amount: Decimal
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

    rule, model = recommendation.rule_decision, recommendation.model_decision
    try:
        result = connection.execute(
            text(
                """
                INSERT INTO recommendations
                    (run_id, case_id, claim_number, policy_number,
                     rule_outcome, rule_reason, rule_payable_amount,
                     rule_explanation,
                     model_outcome, model_reason, model_payable_amount,
                     model_explanation)
                VALUES
                    (:run_id, :case_id, :claim_number, :policy_number,
                     :rule_outcome, :rule_reason, :rule_payable_amount,
                     :rule_explanation,
                     :model_outcome, :model_reason, :model_payable_amount,
                     :model_explanation)
                """
            ),
            {
                "run_id": run_id,
                "case_id": case_id,
                "claim_number": claim_number,
                "policy_number": policy_number,
                "rule_outcome": rule.outcome,
                "rule_reason": rule.reason,
                "rule_payable_amount": rule.payable_amount or Decimal(0),
                "rule_explanation": recommendation.rule_explanation,
                "model_outcome": model.outcome,
                "model_reason": model.reason,
                "model_payable_amount": model.payable_amount or Decimal(0),
                "model_explanation": recommendation.model_explanation,
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
    chosen: ChosenDecision | None = None,
    comment: str | None = None,
) -> int:
    """Record a person's decision on a recommendation; return the approval id.

    An approval must say which decision is chosen, the rules' or the model's,
    even when the two agree: the record should show what the person approved.
    A rejection chooses neither. A recommendation can be decided once;
    deciding it again is an error, not a way to change the first decision.
    """
    if not approver.strip():
        raise GateError("A decision must name the person who made it.")
    if decision == "approved" and chosen is None:
        raise GateError("An approval must choose the rule or the model decision.")
    if decision == "rejected" and chosen is not None:
        raise GateError("A rejection does not choose a decision.")
    try:
        result = connection.execute(
            text(
                """
                INSERT INTO approvals
                    (recommendation_id, claim_number, decision, chosen_decision,
                     approver, comment)
                SELECT recommendation_id, claim_number, :decision, :chosen,
                       :approver, :comment
                  FROM recommendations
                 WHERE recommendation_id = :recommendation_id
                """
            ),
            {
                "recommendation_id": recommendation_id,
                "decision": decision,
                "chosen": chosen,
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
            """
            SELECT a.decision, a.claim_number, a.chosen_decision,
                   r.rule_outcome, r.rule_payable_amount,
                   r.model_outcome, r.model_payable_amount
              FROM approvals AS a
              JOIN recommendations AS r USING (recommendation_id)
             WHERE a.approval_id = :approval_id
            """
        ),
        {"approval_id": approval_id},
    ).one_or_none()
    if approval is None:
        raise ApprovalRequiredError(f"There is no approval {approval_id}.")
    if approval.decision != "approved":
        raise ApprovalRequiredError(
            f"Approval {approval_id} is a rejection; nothing may be written back."
        )
    if approval.chosen_decision == "model":
        outcome, amount = approval.model_outcome, approval.model_payable_amount
    else:
        outcome, amount = approval.rule_outcome, approval.rule_payable_amount

    existing = connection.execute(
        text(
            "SELECT write_back_id, approval_id FROM write_backs "
            "WHERE approval_id = :approval_id OR claim_number = :claim_number"
        ),
        {"approval_id": approval_id, "claim_number": approval.claim_number},
    ).one_or_none()
    if existing is not None and existing.approval_id != approval_id:
        raise GateError(
            f"Claim {approval.claim_number} already has a final decision "
            f"(write-back {existing.write_back_id})."
        )
    if existing is not None:
        write_back_id = existing.write_back_id
    else:
        write_back_id = connection.execute(
            text(
                "INSERT INTO write_backs (approval_id, claim_number) "
                "VALUES (:approval_id, :claim_number)"
            ),
            {"approval_id": approval_id, "claim_number": approval.claim_number},
        ).lastrowid
    return WriteBack(
        write_back_id=write_back_id,
        approval_id=approval_id,
        claim_number=approval.claim_number,
        chosen_decision=approval.chosen_decision,
        outcome=outcome,
        payable_amount=amount,
        already_written=existing is not None,
    )


def pending_recommendations(connection: Connection) -> list[dict[str, object]]:
    """Return the recommendations no person has decided yet, oldest first."""
    rows = connection.execute(
        text(
            """
            SELECT r.recommendation_id, r.case_id, r.claim_number,
                   r.rule_outcome, r.rule_reason, r.rule_payable_amount,
                   r.rule_explanation,
                   r.model_outcome, r.model_reason, r.model_payable_amount,
                   r.model_explanation, r.decisions_agree
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
    approve = commands.add_parser("approve", help="approve a recommendation")
    approve.add_argument(
        "--choose",
        required=True,
        choices=("rule", "model"),
        help="which of the two decisions to carry out",
    )
    reject = commands.add_parser("reject", help="reject a recommendation")
    for command in (approve, reject):
        command.add_argument("recommendation_id", type=int)
        command.add_argument("--approver", required=True, help="your name")
        command.add_argument("--comment")
    command = commands.add_parser(
        "write-back", help="record the final decision of an approval"
    )
    command.add_argument("approval_id", type=int)
    return parser.parse_args(argv)


def _print_pending(rows: list[dict[str, object]]) -> None:
    for row in rows:
        verdict = "agree" if row["decisions_agree"] else "DISAGREE"
        print(
            f"\n#{row['recommendation_id']}  {row['case_id']}  {row['claim_number']}"
            f"  (rules and model {verdict})"
        )
        for source in ("rule", "model"):
            print(
                f"  {source:<5} {row[f'{source}_outcome']} "
                f"({row[f'{source}_reason']}, {row[f'{source}_payable_amount']})"
            )
            print(f"        {row[f'{source}_explanation']}")
    print(f"\n{len(rows)} recommendation(s) awaiting a decision.")


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
                _print_pending(pending_recommendations(connection))
            elif args.command == "write-back":
                done = write_back(connection, args.approval_id)
                state = "was already recorded" if done.already_written else "recorded"
                print(
                    f"Write-back {done.write_back_id} for {done.claim_number} {state}: "
                    f"{done.outcome} ({done.payable_amount}), the "
                    f"{done.chosen_decision} decision."
                )
            else:
                approved = args.command == "approve"
                approval_id = record_decision(
                    connection,
                    recommendation_id=args.recommendation_id,
                    decision="approved" if approved else "rejected",
                    chosen=args.choose if approved else None,
                    approver=args.approver,
                    comment=args.comment,
                )
                print(f"Recorded as approval {approval_id} ({args.command}).")
    except (GateError, ConfigurationError) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
