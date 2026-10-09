"""The human gate: no write-back without a person's approval.

Three layers are tested separately:

* the functions in casefile.gate, for behaviour and clear errors;
* the tables, by inserting rows directly and bypassing those functions;
* the database accounts, which decide who is able to do what at all.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest
from conftest import engine_for
from sqlalchemy import Connection, Table, text
from sqlalchemy.exc import DBAPIError

from casefile.config import DatabaseSettings
from casefile.control.handoff import Recommendation
from casefile.gate import (
    ApprovalRequiredError,
    GateError,
    pending_recommendations,
    record_decision,
    store_recommendation,
    write_back,
)
from casefile.rules import Decision

CHECK_VIOLATED = {3819}
FOREIGN_KEY_VIOLATED = {1216, 1452}
DUPLICATE_KEY = {1062}
COMMAND_DENIED = {1142}

PAY_4000 = Decision(
    outcome="recommend_pay", reason="covered", payable_amount=Decimal("4000.00")
)
DENIED = Decision(outcome="recommend_deny", reason="below_deductible")
REFERRED = Decision(outcome="refer_to_adjuster", reason="frequent_prior_claims")


def recommend(rule: Decision, model: Decision | None = None) -> Recommendation:
    return Recommendation(
        rule_decision=rule,
        model_decision=model or rule,
        rule_explanation=f"Rules: {rule.reason}.",
        model_explanation=f"Model: {(model or rule).reason}.",
    )


PAY = recommend(PAY_4000)
DENY = recommend(DENIED)
# The rules would pay; the model would send the claim to a person.
SPLIT = recommend(PAY_4000, REFERRED)


def new_run_id() -> str:
    return f"test-{uuid.uuid4()}"


@pytest.fixture
def claim_numbers(loader_connection: Connection) -> list[str]:
    """Two loaded claims that have no final decision yet."""
    rows = loader_connection.execute(
        text(
            "SELECT claim_id FROM claims WHERE claim_id NOT IN "
            "(SELECT claim_number FROM write_backs) ORDER BY claim_id LIMIT 2"
        )
    ).scalars()
    numbers = list(rows)
    if len(numbers) < 2:
        pytest.skip("the claims have not been loaded")
    return numbers


def store(
    connection: Connection,
    claim_number: str,
    recommendation: Recommendation = PAY,
    run_id: str | None = None,
) -> int:
    return store_recommendation(
        connection,
        run_id=run_id or new_run_id(),
        case_id="C01",
        claim_number=claim_number,
        policy_number=1,
        recommendation=recommendation,
    )


def decide(
    connection: Connection,
    recommendation_id: int,
    decision: str,
    chosen: str | None = "rule",
) -> int:
    """Record a decision; an approval chooses the rules' decision by default."""
    return record_decision(
        connection,
        recommendation_id=recommendation_id,
        decision=decision,
        chosen=chosen if decision == "approved" else None,
        approver="Test Adjuster",
    )


def error_code(connection: Connection, table: Table, row: dict[str, Any]) -> int:
    with pytest.raises(DBAPIError) as raised:
        connection.execute(table.insert(), row)
    return raised.value.orig.args[0]


# --- the functions ---------------------------------------------------------


def test_recommendation_is_stored_once_per_run(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    run_id = new_run_id()

    first = store(loader_connection, claim_numbers[0], run_id=run_id)
    again = store(loader_connection, claim_numbers[0], run_id=run_id)

    assert first == again
    row = loader_connection.execute(
        text(
            "SELECT rule_outcome, rule_reason, rule_payable_amount, decisions_agree "
            "FROM recommendations WHERE recommendation_id = :id"
        ),
        {"id": first},
    ).one()
    assert tuple(row) == ("recommend_pay", "covered", Decimal("4000.00"), 1)


def test_both_decisions_are_stored_side_by_side(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    recommendation_id = store(loader_connection, claim_numbers[0], SPLIT)

    (row,) = [
        row
        for row in pending_recommendations(loader_connection)
        if row["recommendation_id"] == recommendation_id
    ]

    assert (row["rule_outcome"], row["rule_payable_amount"]) == (
        "recommend_pay",
        Decimal("4000.00"),
    )
    assert (row["model_outcome"], row["model_reason"]) == (
        "refer_to_adjuster",
        "frequent_prior_claims",
    )
    assert row["model_payable_amount"] == Decimal("0.00")
    assert row["decisions_agree"] == 0
    assert row["model_explanation"] == "Model: frequent_prior_claims."


@pytest.mark.parametrize(
    ("chosen", "outcome", "amount"),
    [
        ("rule", "recommend_pay", Decimal("4000.00")),
        ("model", "refer_to_adjuster", Decimal("0.00")),
    ],
)
def test_write_back_carries_out_the_decision_the_person_chose(
    loader_connection: Connection,
    claim_numbers: list[str],
    chosen: str,
    outcome: str,
    amount: Decimal,
) -> None:
    recommendation_id = store(loader_connection, claim_numbers[0], SPLIT)
    approval_id = decide(loader_connection, recommendation_id, "approved", chosen)

    done = write_back(loader_connection, approval_id)

    assert (done.chosen_decision, done.outcome, done.payable_amount) == (
        chosen,
        outcome,
        amount,
    )


def test_approval_must_choose_and_rejection_must_not(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    recommendation_id = store(loader_connection, claim_numbers[0], SPLIT)
    arguments = {"recommendation_id": recommendation_id, "approver": "Test Adjuster"}

    with pytest.raises(GateError, match="must choose"):
        record_decision(loader_connection, decision="approved", **arguments)
    with pytest.raises(GateError, match="does not choose"):
        record_decision(
            loader_connection, decision="rejected", chosen="model", **arguments
        )


def test_recommendation_for_an_unknown_claim_is_refused(
    loader_connection: Connection,
) -> None:
    with pytest.raises(GateError, match="not in the claims system"):
        store(loader_connection, "CLM-0000-00000")


def test_pending_list_holds_only_undecided_recommendations(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    waiting = store(loader_connection, claim_numbers[0])
    decided = store(loader_connection, claim_numbers[1], DENY)
    decide(loader_connection, decided, "rejected")

    pending = {
        row["recommendation_id"] for row in pending_recommendations(loader_connection)
    }

    assert waiting in pending
    assert decided not in pending


def test_a_recommendation_is_decided_once(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    recommendation_id = store(loader_connection, claim_numbers[0])
    decide(loader_connection, recommendation_id, "rejected")

    with pytest.raises(GateError, match="already been decided"):
        decide(loader_connection, recommendation_id, "approved")


def test_decision_needs_a_recommendation_and_a_name(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    recommendation_id = store(loader_connection, claim_numbers[0])

    with pytest.raises(GateError, match="no recommendation"):
        decide(loader_connection, 999_999_999, "approved")
    with pytest.raises(GateError, match="name the person"):
        record_decision(
            loader_connection,
            recommendation_id=recommendation_id,
            decision="approved",
            chosen="rule",
            approver="   ",
        )


def test_write_back_without_an_approval_is_refused(
    loader_connection: Connection,
) -> None:
    with pytest.raises(ApprovalRequiredError, match="needs an approval id"):
        write_back(loader_connection, None)
    with pytest.raises(ApprovalRequiredError, match="no approval"):
        write_back(loader_connection, 999_999_999)


def test_write_back_of_a_rejection_is_refused(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    approval_id = decide(
        loader_connection, store(loader_connection, claim_numbers[0]), "rejected"
    )

    with pytest.raises(ApprovalRequiredError, match="is a rejection"):
        write_back(loader_connection, approval_id)
    count = loader_connection.execute(
        text("SELECT COUNT(*) FROM write_backs WHERE approval_id = :id"),
        {"id": approval_id},
    ).scalar_one()
    assert count == 0


def test_approved_recommendation_is_written_back_exactly_once(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    approval_id = decide(
        loader_connection, store(loader_connection, claim_numbers[0]), "approved"
    )

    first = write_back(loader_connection, approval_id)
    second = write_back(loader_connection, approval_id)

    assert (first.claim_number, first.already_written) == (claim_numbers[0], False)
    assert second.already_written
    assert second.write_back_id == first.write_back_id
    count = loader_connection.execute(
        text("SELECT COUNT(*) FROM write_backs WHERE claim_number = :claim"),
        {"claim": claim_numbers[0]},
    ).scalar_one()
    assert count == 1


def test_a_claim_gets_one_final_decision(
    loader_connection: Connection, claim_numbers: list[str]
) -> None:
    claim = claim_numbers[0]
    first = decide(loader_connection, store(loader_connection, claim), "approved")
    second = decide(loader_connection, store(loader_connection, claim), "approved")
    write_back(loader_connection, first)

    with pytest.raises(GateError, match="already has a final decision"):
        write_back(loader_connection, second)


# --- the tables, bypassing the functions -------------------------------------


@pytest.fixture
def approvals(
    loader_connection: Connection, claim_numbers: list[str]
) -> dict[str, int]:
    """One granted and one rejected approval, for two different claims."""
    return {
        "approved": decide(
            loader_connection, store(loader_connection, claim_numbers[0]), "approved"
        ),
        "rejected": decide(
            loader_connection, store(loader_connection, claim_numbers[1]), "rejected"
        ),
    }


def test_table_refuses_a_write_back_without_an_approval(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
) -> None:
    row = {"approval_id": 999_999_999, "claim_number": claim_numbers[0]}

    assert error_code(loader_connection, domain_tables["write_backs"], row) in (
        FOREIGN_KEY_VIOLATED
    )


def test_table_refuses_a_write_back_of_a_rejection(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
    approvals: dict[str, int],
) -> None:
    table = domain_tables["write_backs"]
    row = {"approval_id": approvals["rejected"], "claim_number": claim_numbers[1]}

    # Claiming the approval was granted does not match the approvals table...
    assert error_code(loader_connection, table, row) in FOREIGN_KEY_VIOLATED
    # ...and stating the truth is rejected by the check constraint.
    honest = {**row, "approval_decision": "rejected"}
    assert error_code(loader_connection, table, honest) in CHECK_VIOLATED


def test_table_refuses_a_write_back_for_another_claim(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
    approvals: dict[str, int],
) -> None:
    row = {"approval_id": approvals["approved"], "claim_number": claim_numbers[1]}

    assert error_code(loader_connection, domain_tables["write_backs"], row) in (
        FOREIGN_KEY_VIOLATED
    )


def test_table_refuses_a_second_write_back(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
    approvals: dict[str, int],
) -> None:
    table = domain_tables["write_backs"]
    row = {"approval_id": approvals["approved"], "claim_number": claim_numbers[0]}
    loader_connection.execute(table.insert(), row)

    assert error_code(loader_connection, table, row) in DUPLICATE_KEY


def test_table_refuses_an_approval_without_a_recommendation(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
) -> None:
    row = {
        "recommendation_id": 999_999_999,
        "claim_number": claim_numbers[0],
        "decision": "approved",
        "chosen_decision": "rule",
        "approver": "Nobody",
    }

    assert error_code(loader_connection, domain_tables["approvals"], row) in (
        FOREIGN_KEY_VIOLATED
    )


@pytest.mark.parametrize(
    ("decision", "chosen"), [("approved", None), ("rejected", "model")]
)
def test_table_refuses_an_approval_whose_choice_does_not_fit(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
    decision: str,
    chosen: str | None,
) -> None:
    row = {
        "recommendation_id": store(loader_connection, claim_numbers[0]),
        "claim_number": claim_numbers[0],
        "decision": decision,
        "chosen_decision": chosen,
        "approver": "Test Adjuster",
    }

    assert error_code(loader_connection, domain_tables["approvals"], row) in (
        CHECK_VIOLATED
    )


@pytest.mark.parametrize("source", ["rule", "model"])
@pytest.mark.parametrize(
    ("outcome", "payable_amount"), [("recommend_pay", 0), ("recommend_deny", 500)]
)
def test_table_refuses_an_amount_that_contradicts_the_outcome(
    loader_connection: Connection,
    domain_tables: dict[str, Table],
    claim_numbers: list[str],
    source: str,
    outcome: str,
    payable_amount: int,
) -> None:
    row = {
        "run_id": new_run_id(),
        "case_id": "C01",
        "claim_number": claim_numbers[0],
        "policy_number": 1,
        "rule_outcome": "recommend_deny",
        "rule_reason": "below_deductible",
        "rule_explanation": "test",
        "model_outcome": "recommend_deny",
        "model_reason": "below_deductible",
        "model_explanation": "test",
        f"{source}_outcome": outcome,
        f"{source}_payable_amount": payable_amount,
    }

    assert error_code(loader_connection, domain_tables["recommendations"], row) in (
        CHECK_VIOLATED
    )


# --- the accounts ------------------------------------------------------------

# Harmless even if a grant were wrong: each statement matches no row.
NO_ROW_APPROVAL = (
    "INSERT INTO approvals "
    "(recommendation_id, claim_number, decision, chosen_decision, approver) "
    "SELECT 0, '', 'approved', 'rule', '' FROM DUAL WHERE 1 = 0"
)
NO_ROW_WRITE_BACK = (
    "INSERT INTO write_backs (approval_id, claim_number) "
    "SELECT 0, '' FROM DUAL WHERE 1 = 0"
)
NO_ROW_RECOMMENDATION = (
    "INSERT INTO recommendations "
    "(run_id, case_id, claim_number, policy_number, rule_outcome, rule_reason, "
    "rule_explanation, model_outcome, model_reason, model_explanation) "
    "SELECT '', '', '', 0, 'recommend_deny', '', '', 'recommend_deny', '', '' "
    "FROM DUAL WHERE 1 = 0"
)


def denied(connection: Connection, statement: str) -> bool:
    try:
        connection.execute(text(statement))
    except DBAPIError as error:
        if error.orig.args[0] in COMMAND_DENIED:
            return True
        raise
    return False


@pytest.mark.parametrize("statement", [NO_ROW_APPROVAL, NO_ROW_WRITE_BACK])
def test_agents_cannot_touch_the_gate(
    agent_connection: Connection, statement: str
) -> None:
    assert denied(agent_connection, statement)
    assert denied(agent_connection, "SELECT 1 FROM approvals LIMIT 1")


def test_the_system_cannot_approve_its_own_recommendation(
    writer_connection: Connection,
) -> None:
    assert denied(writer_connection, NO_ROW_APPROVAL)
    assert denied(
        writer_connection, "UPDATE approvals SET decision = 'approved' WHERE 1 = 0"
    )
    assert denied(writer_connection, "DELETE FROM write_backs WHERE 1 = 0")
    assert not denied(writer_connection, NO_ROW_RECOMMENDATION)
    assert not denied(writer_connection, NO_ROW_WRITE_BACK)


def test_the_approver_cannot_write_back_or_recommend(
    approver_connection: Connection,
) -> None:
    assert denied(approver_connection, NO_ROW_WRITE_BACK)
    assert denied(approver_connection, NO_ROW_RECOMMENDATION)
    assert denied(approver_connection, "DELETE FROM approvals WHERE 1 = 0")
    assert not denied(approver_connection, NO_ROW_APPROVAL)


def test_gate_end_to_end_with_the_real_accounts(
    database_settings: DatabaseSettings, claim_numbers: list[str]
) -> None:
    """System recommends, person approves, system writes back, with commits."""
    writer = engine_for("writer", database_settings)
    approver = engine_for("approver", database_settings)
    loader = engine_for("loader", database_settings)
    run_id = new_run_id()
    try:
        with writer.begin() as connection:
            recommendation_id = store(connection, claim_numbers[0], run_id=run_id)
        with writer.begin() as connection, pytest.raises(ApprovalRequiredError):
            write_back(connection, None)

        with approver.begin() as connection:
            assert recommendation_id in {
                row["recommendation_id"] for row in pending_recommendations(connection)
            }
            approval_id = decide(connection, recommendation_id, "approved")

        with writer.begin() as connection:
            first = write_back(connection, approval_id)
        with writer.begin() as connection:
            second = write_back(connection, approval_id)

        assert not first.already_written
        assert second.already_written
        assert second.write_back_id == first.write_back_id
    finally:
        # Only the loader may delete; remove what this test committed.
        with loader.begin() as connection:
            parameters = {"run_id": run_id}
            connection.execute(
                text(
                    "DELETE w FROM write_backs AS w "
                    "JOIN approvals AS a USING (approval_id) "
                    "JOIN recommendations AS r USING (recommendation_id) "
                    "WHERE r.run_id = :run_id"
                ),
                parameters,
            )
            connection.execute(
                text(
                    "DELETE a FROM approvals AS a "
                    "JOIN recommendations AS r USING (recommendation_id) "
                    "WHERE r.run_id = :run_id"
                ),
                parameters,
            )
            connection.execute(
                text("DELETE FROM recommendations WHERE run_id = :run_id"), parameters
            )
        for engine in (writer, approver, loader):
            engine.dispose()
