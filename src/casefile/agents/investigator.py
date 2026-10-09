"""The investigator: two independent decisions on a claim, and why.

For each case the investigator produces

* the decision of the fixed rules, computed in code, and
* the decision of the model, which is given the same facts and the rules in
  words but is not shown what the code decided.

It then asks the model to explain both for the person who has to choose
between them. Deciding and explaining are two separate requests on purpose:
if the model saw the rules' result before deciding, its "own" decision would
no longer be independent, and how often the two agree would say nothing about
the model.

Reviewer feedback goes into the explanation request only. The decision
request depends on the facts alone, so a reviewer's remark can never nudge
the model's decision.
"""

from __future__ import annotations

import json
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from casefile.agents.triage import CaseFacts, DomainLookup, assess, gather_facts
from casefile.control.handoff import Recommendation
from casefile.control.state import CaseState, InvalidOutput, Recommended, StepEvent
from casefile.llm.budget import MeteredLLM
from casefile.llm.client import ChatRequest, Message
from casefile.rules import (
    FREQUENT_CLAIMS_COUNT,
    FREQUENT_CLAIMS_WINDOW_MONTHS,
    Decision,
)

# Deciding means comparing dates and counting claims inside a time window.
# Without room to reason the model answers at once and miscounts, so this
# request asks for more thought and allows for the tokens that takes.
DECISION_REASONING_EFFORT = "max"
DECISION_MAX_TOKENS = 4000
EXPLANATION_MAX_TOKENS = 600

FACTS_GUIDE = """\
The facts are JSON with these parts:
- documents.claim_form: what the claim form states.
- documents.repair_estimate: what the repair estimate states, or null when \
the file contains no repair estimate.
- documents.police_report: what the police report states, or null when there \
is none.
- policy: the policy whose number is on the claim form, or null when no such \
policy exists.
- prior_claims: the earlier claims on that policy.
- repair_cost_reference: the usual range of vehicle repair cost for this kind \
of incident, or null when there is none."""

DECISION_PROMPT = f"""\
You decide what should happen to an auto insurance claim. You are given the \
facts of the case and a list of rules. Apply the rules strictly in the order \
given: the first rule whose condition is true decides, and the rules after it \
are not considered. Use only the facts given. Answer with JSON only.

{FACTS_GUIDE}

Rules:
1. reason "policy_not_found": policy is null. Outcome: refer_to_adjuster.
2. reason "incident_outside_policy_period": documents.claim_form.incident_date \
is earlier than policy.policy_start_date. Outcome: recommend_deny.
3. reason "missing_document": documents.repair_estimate is null. Outcome: \
refer_to_adjuster. A missing police report is not a problem.
4. reason "document_mismatch": documents.repair_estimate.total_amount is not \
equal to documents.claim_form.vehicle_claim, or documents.police_report is \
present and its incident_date is not equal to \
documents.claim_form.incident_date. Outcome: refer_to_adjuster.
5. reason "amount_above_reference": repair_cost_reference is not null and \
documents.claim_form.vehicle_claim is greater than \
repair_cost_reference.high_amount. Outcome: refer_to_adjuster.
6. reason "frequent_prior_claims": {FREQUENT_CLAIMS_COUNT} or more of the \
prior_claims have an incident_date that is on or after the date \
{FREQUENT_CLAIMS_WINDOW_MONTHS} months before \
documents.claim_form.incident_date and before that incident date. Outcome: \
refer_to_adjuster.
7. reason "below_deductible": documents.claim_form.total_claim_amount is less \
than or equal to policy.policy_deductible. Outcome: recommend_deny.
8. reason "covered": none of the above. Outcome: recommend_pay, with \
payable_amount equal to documents.claim_form.total_claim_amount minus \
policy.policy_deductible.

payable_amount is null unless the outcome is recommend_pay."""

EXPLANATION_PROMPT = f"""\
You write two short explanations for a claims adjuster who must choose between \
two decisions on the same claim. One decision was produced by fixed rules, the \
other by an AI model working independently on the same facts. For each \
decision, say in one to three sentences which facts it rests on, quoting the \
values. Do not say which decision is better. Do not state any amount, date, \
count or number that is not in the input. Answer with JSON only.

{FACTS_GUIDE}"""

# The explanation request is built after the model has decided, but its worst
# case is needed before. These stand in for the model's decision when sizing
# it: together they are at least as long as any decision can be.
_LONGEST_DECISIONS = (
    Decision(
        outcome="recommend_pay",
        reason="incident_outside_policy_period",
        payable_amount=Decimal("9999999999.99"),
    ),
    Decision(outcome="refer_to_adjuster", reason="incident_outside_policy_period"),
)


class Explanations(BaseModel):
    """The model's explanation of each of the two decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_explanation: str = Field(
        min_length=1, description="Why the fixed rules reached their decision"
    )
    model_explanation: str = Field(
        min_length=1, description="Why the AI model reached its decision"
    )


def build_decision_request(facts: CaseFacts) -> ChatRequest:
    """Ask the model to decide the case from the facts alone."""
    return ChatRequest(
        messages=(
            Message(role="system", content=DECISION_PROMPT),
            Message(role="user", content=facts.model_dump_json(indent=2)),
        ),
        max_tokens=DECISION_MAX_TOKENS,
        json_schema=Decision.model_json_schema(),
        schema_name="decision",
        reasoning_effort=DECISION_REASONING_EFFORT,
    )


def build_explanation_request(
    facts: CaseFacts,
    rule_decision: Decision,
    model_decision: Decision,
    feedback: str | None,
) -> ChatRequest:
    """Ask the model to explain both decisions to the person who must choose."""
    content = json.dumps(
        {
            "facts": facts.model_dump(mode="json"),
            "rule_decision": rule_decision.model_dump(mode="json"),
            "model_decision": model_decision.model_dump(mode="json"),
        },
        indent=2,
    )
    if feedback:
        content += (
            "\n\nA reviewer rejected the previous explanations for this "
            f"reason: {feedback} Write them again without that fault."
        )
    return ChatRequest(
        messages=(
            Message(role="system", content=EXPLANATION_PROMPT),
            Message(role="user", content=content),
        ),
        max_tokens=EXPLANATION_MAX_TOKENS,
        json_schema=Explanations.model_json_schema(),
        schema_name="explanations",
    )


class LLMInvestigator:
    """Decides a case twice, by the rules and by the model, and explains both."""

    def __init__(self, llm: MeteredLLM, lookup: DomainLookup) -> None:
        self._llm = llm
        self._lookup = lookup

    def worst_case_cost(self, state: CaseState) -> Decimal:
        """Return the most tokens both requests can take."""
        if state.extraction is None:
            return Decimal(0)
        facts = gather_facts(state.extraction, self._lookup)
        rule_decision = assess(state.extraction, self._lookup)
        explaining = max(
            self._llm.worst_case(
                build_explanation_request(
                    facts, rule_decision, stand_in, state.feedback
                )
            )
            for stand_in in _LONGEST_DECISIONS
        )
        return Decimal(self._llm.worst_case(build_decision_request(facts)) + explaining)

    def run(self, state: CaseState) -> StepEvent:
        """Produce both decisions and their explanations."""
        if state.extraction is None:
            return InvalidOutput(detail="nothing was extracted")
        facts = gather_facts(state.extraction, self._lookup)
        rule_decision = assess(state.extraction, self._lookup)

        response = self._llm.complete(build_decision_request(facts))
        tokens = response.usage.total_tokens
        try:
            model_decision = Decision.model_validate_json(response.content)
        except ValidationError as error:
            return _invalid("decision", error, tokens)

        response = self._llm.complete(
            build_explanation_request(
                facts, rule_decision, model_decision, state.feedback
            )
        )
        tokens += response.usage.total_tokens
        try:
            explanations = Explanations.model_validate_json(response.content)
        except ValidationError as error:
            return _invalid("explanations", error, tokens)

        return Recommended(
            cost=Decimal(tokens),
            recommendation=Recommendation(
                rule_decision=rule_decision,
                model_decision=model_decision,
                rule_explanation=explanations.rule_explanation,
                model_explanation=explanations.model_explanation,
            ),
        )


def _invalid(what: str, error: ValidationError, tokens: int) -> InvalidOutput:
    problem = error.errors()[0]
    return InvalidOutput(
        cost=Decimal(tokens),
        detail=f"{what}: {problem['msg']} at {list(problem['loc'])}",
    )
