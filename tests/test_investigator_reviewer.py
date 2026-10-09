"""The model-backed investigator and reviewer, with a scripted model."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest
from builders import (
    ScriptedLLM,
    build_claim,
    build_eval_case,
    build_policy,
    build_prior_claim,
)
from crash_run import Lookup
from sqlalchemy import Connection

from casefile.agents.domain import SqlDomainLookup
from casefile.agents.investigator import (
    DECISION_PROMPT,
    LLMInvestigator,
)
from casefile.agents.mock import MockExtractor, extraction_from_documents
from casefile.agents.reviewer import AMOUNTS_FEEDBACK, RULES_FEEDBACK, LLMReviewer
from casefile.agents.triage import assess, gather_facts
from casefile.config import RAW_DATA_DIR
from casefile.control.handoff import Extraction, Recommendation
from casefile.control.state import (
    Approved,
    CaseState,
    InvalidOutput,
    Limits,
    Recommended,
    SentBack,
)
from casefile.control.supervisor import advance, run_case, start
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME
from casefile.documents.content import build_case_documents
from casefile.eval.cases import load_cases
from casefile.eval.run_dev import DEV_RECORDINGS, build_dev_suite, run_dev
from casefile.eval.run_dev import REPORT_PATH as DEV_REPORT_PATH
from casefile.eval.select_cases import build_portfolio
from casefile.llm.budget import worst_case_tokens
from casefile.rules import Decision

LIMITS = Limits(budget=Decimal(10_000_000))
PAY = Decision(outcome="recommend_pay", reason="covered", payable_amount=Decimal(4000))
REFER = Decision(outcome="refer_to_adjuster", reason="frequent_prior_claims")
EXPLANATIONS = json.dumps(
    {
        "rule_explanation": "The total of 5000 exceeds the deductible of 1000.",
        "model_explanation": "No rule before the last applied; 5000 - 1000 = 4000.",
    }
)
GROUNDED = json.dumps({"grounded": True, "problem": ""})
UNGROUNDED = json.dumps({"grounded": False, "problem": 'It says "deductible of 2000".'})


def extraction() -> Extraction:
    documents = build_case_documents(
        build_eval_case(), build_policy(), build_claim(), {1}
    )
    return extraction_from_documents(documents)


def extracted_state(feedback: str | None = None) -> CaseState:
    """A case whose documents have been read and that now awaits a decision."""
    state = start("C01", LIMITS).model_copy(
        update={"extraction": extraction(), "next_node": "investigator"}
    )
    return state.model_copy(update={"feedback": feedback})


def recommended_state(recommendation: Recommendation) -> CaseState:
    return extracted_state().model_copy(
        update={"recommendation": recommendation, "next_node": "reviewer"}
    )


def recommendation(model: Decision = PAY, rule: Decision = PAY) -> Recommendation:
    return Recommendation(
        rule_decision=rule,
        model_decision=model,
        rule_explanation="The total of 5000 exceeds the deductible of 1000.",
        model_explanation="5000 - 1000 = 4000.",
    )


def investigator_llm(decision: str | Decision = PAY, explanations: str = EXPLANATIONS):
    if isinstance(decision, Decision):
        decision = decision.model_dump_json()
    return ScriptedLLM({"decision": [decision] * 4, "explanations": [explanations] * 4})


# --- the facts ---------------------------------------------------------------


def test_facts_hold_the_inputs_and_nothing_derived() -> None:
    facts = gather_facts(extraction(), Lookup()).model_dump(mode="json")

    assert set(facts) == {
        "documents",
        "policy",
        "prior_claims",
        "repair_cost_reference",
    }
    assert facts["policy"] == {
        "policy_number": 1,
        "policy_start_date": "2005-01-25",
        "policy_deductible": 1000,
    }
    assert facts["documents"]["repair_estimate"]["total_amount"] == "4000"
    assert facts["repair_cost_reference"]["high_amount"] == "6000.00"


def test_unknown_policy_gives_null_policy_and_no_history() -> None:
    unknown = extraction().claim_form.model_copy(update={"policy_number": 999})
    facts = gather_facts(
        extraction().model_copy(update={"claim_form": unknown}), Lookup()
    )

    assert facts.policy is None


# --- the investigator ----------------------------------------------------------


def test_decision_request_gives_facts_and_rules_but_not_the_rules_result() -> None:
    llm = investigator_llm()

    LLMInvestigator(llm, Lookup()).run(extracted_state())

    (request,) = llm.sent("decision")
    system, user = request.messages
    assert system.content == DECISION_PROMPT
    assert "3 or more of the prior_claims" in DECISION_PROMPT
    assert "36 months before" in DECISION_PROMPT
    assert json.loads(user.content) == gather_facts(extraction(), Lookup()).model_dump(
        mode="json"
    )
    assert "rule_decision" not in user.content
    assert request.json_schema == Decision.model_json_schema()


def test_both_decisions_are_reported_when_they_agree() -> None:
    llm = investigator_llm()

    event = LLMInvestigator(llm, Lookup()).run(extracted_state())

    assert isinstance(event, Recommended)
    assert event.recommendation.rule_decision == PAY
    assert event.recommendation.model_decision == PAY
    assert event.recommendation.decisions_agree
    assert event.recommendation.model_explanation.endswith("5000 - 1000 = 4000.")
    assert event.cost == Decimal(200)
    assert [request.schema_name for request in llm.requests] == [
        "decision",
        "explanations",
    ]


def test_disagreement_is_reported_not_treated_as_an_error() -> None:
    event = LLMInvestigator(investigator_llm(REFER), Lookup()).run(extracted_state())

    assert isinstance(event, Recommended)
    assert event.recommendation.rule_decision == PAY
    assert event.recommendation.model_decision == REFER
    assert not event.recommendation.outcomes_agree
    assert not event.recommendation.decisions_agree


def test_same_outcome_with_another_amount_is_only_a_partial_agreement() -> None:
    other_amount = PAY.model_copy(update={"payable_amount": Decimal(5000)})

    event = LLMInvestigator(investigator_llm(other_amount), Lookup()).run(
        extracted_state()
    )

    assert event.recommendation.outcomes_agree
    assert not event.recommendation.decisions_agree


def test_explanation_request_shows_both_decisions() -> None:
    llm = investigator_llm(REFER)

    LLMInvestigator(llm, Lookup()).run(extracted_state())

    (request,) = llm.sent("explanations")
    shown = json.loads(request.messages[1].content)
    assert shown["rule_decision"]["outcome"] == "recommend_pay"
    assert shown["model_decision"]["outcome"] == "refer_to_adjuster"
    assert shown["facts"]["policy"]["policy_deductible"] == 1000


@pytest.mark.parametrize(
    "bad_decision",
    [
        "I would pay this claim.",
        '{"outcome": "recommend_pay", "reason": "covered", "payable_amount": null}',
        '{"outcome": "approve", "reason": "covered", "payable_amount": 4000}',
    ],
    ids=["prose", "pay-without-amount", "unknown-outcome"],
)
def test_unusable_decision_is_invalid_output(bad_decision: str) -> None:
    llm = investigator_llm(bad_decision)

    event = LLMInvestigator(llm, Lookup()).run(extracted_state())

    assert isinstance(event, InvalidOutput)
    assert event.detail.startswith("decision:")
    assert event.cost == Decimal(100)
    assert llm.sent("explanations") == []


def test_unusable_explanation_is_invalid_output() -> None:
    llm = investigator_llm(explanations='{"rule_explanation": ""}')

    event = LLMInvestigator(llm, Lookup()).run(extracted_state())

    assert isinstance(event, InvalidOutput)
    assert event.detail.startswith("explanations:")
    assert event.cost == Decimal(200)


def test_reviewer_feedback_reaches_the_explanation_but_not_the_decision() -> None:
    feedback = 'It says "deductible of 2000".'
    first, second = investigator_llm(), investigator_llm()

    LLMInvestigator(first, Lookup()).run(extracted_state())
    LLMInvestigator(second, Lookup()).run(extracted_state(feedback))

    assert first.sent("decision") == second.sent("decision")
    assert feedback in second.sent("explanations")[0].messages[1].content
    assert feedback not in first.sent("explanations")[0].messages[1].content


def test_worst_case_is_known_before_the_model_has_decided() -> None:
    llm = investigator_llm(REFER)
    investigator = LLMInvestigator(llm, Lookup())
    state = extracted_state("some feedback")

    worst_case = investigator.worst_case_cost(state)
    investigator.run(state)

    assert llm.requests != []
    assert worst_case >= sum(worst_case_tokens(request) for request in llm.requests)


def test_investigator_cannot_work_before_extraction() -> None:
    investigator = LLMInvestigator(investigator_llm(), Lookup())
    empty = start("C01", LIMITS)

    assert investigator.worst_case_cost(empty) == Decimal(0)
    assert isinstance(investigator.run(empty), InvalidOutput)


# --- the reviewer --------------------------------------------------------------


def reviewer_llm(*verdicts: str) -> ScriptedLLM:
    return ScriptedLLM({"review": list(verdicts)})


def test_grounded_work_is_approved() -> None:
    llm = reviewer_llm(GROUNDED)

    event = LLMReviewer(llm, Lookup()).run(recommended_state(recommendation()))

    assert isinstance(event, Approved)
    assert event.cost == Decimal(100)
    shown = json.loads(llm.requests[0].messages[1].content)
    assert shown["model_explanation"] == "5000 - 1000 = 4000."
    assert shown["facts"]["policy"]["policy_deductible"] == 1000


def test_disagreement_between_rules_and_model_is_not_sent_back() -> None:
    llm = reviewer_llm(GROUNDED)

    event = LLMReviewer(llm, Lookup()).run(recommended_state(recommendation(REFER)))

    assert isinstance(event, Approved)


def test_unsupported_statement_goes_back_to_the_investigator() -> None:
    event = LLMReviewer(reviewer_llm(UNGROUNDED), Lookup()).run(
        recommended_state(recommendation())
    )

    assert isinstance(event, SentBack)
    assert event.target == "investigator"
    assert event.reason == 'It says "deductible of 2000".'


def test_amounts_that_do_not_add_up_go_back_without_asking_the_model() -> None:
    llm = reviewer_llm()
    misread = extraction().claim_form.model_copy(update={"vehicle_claim": Decimal(400)})
    state = recommended_state(recommendation()).model_copy(
        update={"extraction": extraction().model_copy(update={"claim_form": misread})}
    )

    event = LLMReviewer(llm, Lookup()).run(state)

    assert (event.target, event.reason) == ("extractor", AMOUNTS_FEEDBACK)
    assert event.cost == Decimal(0)
    assert llm.requests == []


def test_wrong_rule_decision_goes_back_without_asking_the_model() -> None:
    llm = reviewer_llm()
    state = recommended_state(recommendation(rule=REFER))

    event = LLMReviewer(llm, Lookup()).run(state)

    assert (event.target, event.reason) == ("investigator", RULES_FEEDBACK)
    assert llm.requests == []


def test_unusable_verdict_is_invalid_output() -> None:
    event = LLMReviewer(reviewer_llm("looks fine to me"), Lookup()).run(
        recommended_state(recommendation())
    )

    assert isinstance(event, InvalidOutput)
    assert event.detail.startswith("review:")


def test_reviewer_worst_case_covers_its_request() -> None:
    llm = reviewer_llm(GROUNDED)
    reviewer = LLMReviewer(llm, Lookup())
    state = recommended_state(recommendation())

    worst_case = reviewer.worst_case_cost(state)
    reviewer.run(state)

    assert worst_case == worst_case_tokens(llm.requests[0])
    assert reviewer.worst_case_cost(start("C01", LIMITS)) == Decimal(0)


# --- the three agents together -------------------------------------------------


def test_rejected_explanation_is_rewritten_and_then_approved() -> None:
    documents = build_case_documents(
        build_eval_case(), build_policy(), build_claim(), {1}
    )
    investigating = investigator_llm()
    reviewing = reviewer_llm(UNGROUNDED, GROUNDED)
    agents = {
        "extractor": MockExtractor(documents),
        "investigator": LLMInvestigator(investigating, Lookup()),
        "reviewer": LLMReviewer(reviewing, Lookup()),
    }

    state = run_case(start("C01", LIMITS), agents)

    assert state.status == "completed"
    assert state.path == (
        "extractor",
        "investigator",
        "reviewer",
        "investigator",
        "reviewer",
    )
    assert state.revisions.investigator == 1
    # The model's decision was asked for twice with exactly the same request.
    first, second = investigating.sent("decision")
    assert first == second
    assert state.budget_spent == Decimal(1 + 200 + 100 + 200 + 100)


def test_history_changes_the_rules_decision_the_model_must_match() -> None:
    history = tuple(
        build_prior_claim(number, date(2014, number, 1)) for number in (1, 2, 3)
    )

    class LookupWithHistory(Lookup):
        def find_prior_claims(self, policy_number: int):
            return history

    state = extracted_state()
    event = LLMInvestigator(investigator_llm(), LookupWithHistory()).run(state)

    assert event.recommendation.rule_decision == REFER
    assert event.recommendation.model_decision == PAY
    facts = gather_facts(state.extraction, LookupWithHistory())
    assert [str(prior.incident_date) for prior in facts.prior_claims] == [
        "2014-01-01",
        "2014-02-01",
        "2014-03-01",
    ]
    assert advance(state, event).next_node == "reviewer"


# --- the recorded development run ------------------------------------------------


@pytest.mark.skipif(
    not (DEV_REPORT_PATH.exists() and DEV_RECORDINGS.exists()),
    reason="the development cases have not been run and recorded yet",
)
def test_recorded_development_run_reproduces_its_report_offline() -> None:
    """No model, no key and no database: lookups come from the raw files."""
    claims, oracle = RAW_DATA_DIR / RAW_FILE_NAME, RAW_DATA_DIR / ORACLE_FILE_NAME
    if not (claims.exists() and oracle.exists()):
        pytest.skip("the raw datasets have not been downloaded")
    portfolio = build_portfolio(claims, oracle)
    stored = json.loads(DEV_REPORT_PATH.read_text(encoding="utf-8"))

    replayed = run_dev("replay", None, stored["model"], portfolio, portfolio)

    assert replayed == stored
    assert not {
        case.claim_id for case in build_dev_suite(portfolio, load_cases()).cases
    } & {case.claim_id for case in load_cases().cases}


# --- the SQL lookup -------------------------------------------------------------


def test_sql_lookup_returns_what_the_loaders_stored(
    agent_connection: Connection,
) -> None:
    claims, oracle = RAW_DATA_DIR / RAW_FILE_NAME, RAW_DATA_DIR / ORACLE_FILE_NAME
    if not (claims.exists() and oracle.exists()):
        pytest.skip("the raw datasets have not been downloaded")
    portfolio = build_portfolio(claims, oracle)
    lookup = SqlDomainLookup(agent_connection)
    with_history = [n for n in portfolio.policies if portfolio.find_prior_claims(n)]

    for policy_number in [*list(portfolio.policies)[:5], *with_history[:5]]:
        assert lookup.find_policy(policy_number) == portfolio.find_policy(policy_number)
        assert sorted(
            lookup.find_prior_claims(policy_number), key=lambda p: p.prior_claim_id
        ) == sorted(
            portfolio.find_prior_claims(policy_number), key=lambda p: p.prior_claim_id
        )
    for key, reference in portfolio.reference.items():
        assert lookup.find_reference(*key) == reference


def test_sql_lookup_finds_nothing_for_unknown_keys(
    agent_connection: Connection,
) -> None:
    lookup = SqlDomainLookup(agent_connection)

    assert lookup.find_policy(1) is None
    assert lookup.find_policy(10**12) is None
    assert lookup.find_prior_claims(1) == ()
    assert lookup.find_reference("Vehicle Theft", "Total Loss") is None


def test_sql_lookup_decides_like_the_in_memory_one(
    agent_connection: Connection,
) -> None:
    """The rules give the same answer whether the data comes from MySQL or not."""
    claims, oracle = RAW_DATA_DIR / RAW_FILE_NAME, RAW_DATA_DIR / ORACLE_FILE_NAME
    if not (claims.exists() and oracle.exists()):
        pytest.skip("the raw datasets have not been downloaded")
    portfolio = build_portfolio(claims, oracle)
    lookup = SqlDomainLookup(agent_connection)

    for claim in portfolio.claims[:150]:
        assert decide_with(lookup, claim) == decide_with(portfolio, claim)


def decide_with(lookup, claim) -> Decision:
    form = extraction().claim_form.model_copy(
        update={
            "policy_number": claim.policy_number,
            "incident_date": claim.incident_date,
            "incident_type": claim.incident_type,
            "incident_severity": claim.incident_severity,
            "vehicle_claim": claim.vehicle_claim,
            "total_claim_amount": claim.total_claim_amount,
        }
    )
    return assess(Extraction(claim_form=form), lookup)
