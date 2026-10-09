"""The reviewer: check the work before it goes to a person.

The reviewer does not decide the claim and does not take sides between the
rules and the model. A disagreement between the two is information for the
person at the gate, not a fault to be sent back.

What it checks, cheapest first:

1. The claimed amounts add up. If not, the reading is wrong and the extractor
   is asked to read again. Checked in code.
2. The rules' decision really follows from the extracted facts. Checked in
   code, by deciding again.
3. The explanations state nothing the facts do not support. This needs a
   model, so it is only asked when the first two checks pass.
"""

from __future__ import annotations

import json
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from casefile.agents.investigator import FACTS_GUIDE
from casefile.agents.triage import DomainLookup, assess, gather_facts
from casefile.control.handoff import Extraction, Recommendation
from casefile.control.state import (
    Approved,
    CaseState,
    InvalidOutput,
    SentBack,
    StepEvent,
)
from casefile.llm.budget import MeteredLLM
from casefile.llm.client import ChatRequest, Message

REVIEW_MAX_TOKENS = 300
AMOUNTS_FEEDBACK = (
    "The claimed injury, property and vehicle amounts do not add up to the "
    "claimed total."
)
RULES_FEEDBACK = "The rules' decision does not follow from the extracted facts."

REVIEW_PROMPT = f"""\
You check two explanations written for a claims adjuster. You are given the \
facts of a case, two decisions about it and the explanation of each.

Set "grounded" to false only if an explanation states a specific value (an \
amount, a date, a count or a policy number) that contradicts the facts, or \
that cannot be obtained from the facts by simple arithmetic or by counting. \
Do not judge whether either decision is correct. Do not object to the two \
decisions being different from each other. Wording, style and length are not \
your concern.

When "grounded" is false, quote the unsupported statement in "problem". \
Otherwise leave "problem" empty. Answer with JSON only.

{FACTS_GUIDE}"""


class Verdict(BaseModel):
    """The model's finding on whether the explanations stick to the facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounded: bool = Field(
        description="False only if an explanation states an unsupported value"
    )
    problem: str = Field(
        description="The unsupported statement, quoted; empty when grounded"
    )


def build_review_request(
    extraction: Extraction, recommendation: Recommendation, lookup: DomainLookup
) -> ChatRequest:
    """Ask the model whether the explanations stay within the facts."""
    content = json.dumps(
        {
            "facts": gather_facts(extraction, lookup).model_dump(mode="json"),
            "rule_decision": recommendation.rule_decision.model_dump(mode="json"),
            "rule_explanation": recommendation.rule_explanation,
            "model_decision": recommendation.model_decision.model_dump(mode="json"),
            "model_explanation": recommendation.model_explanation,
        },
        indent=2,
    )
    return ChatRequest(
        messages=(
            Message(role="system", content=REVIEW_PROMPT),
            Message(role="user", content=content),
        ),
        max_tokens=REVIEW_MAX_TOKENS,
        json_schema=Verdict.model_json_schema(),
        schema_name="review",
    )


class LLMReviewer:
    """Approves the work or returns it to the agent whose part is at fault."""

    def __init__(self, llm: MeteredLLM, lookup: DomainLookup) -> None:
        self._llm = llm
        self._lookup = lookup

    def worst_case_cost(self, state: CaseState) -> Decimal:
        """Return the most tokens the model check can take."""
        if state.extraction is None or state.recommendation is None:
            return Decimal(0)
        request = build_review_request(
            state.extraction, state.recommendation, self._lookup
        )
        return Decimal(self._llm.worst_case(request))

    def run(self, state: CaseState) -> StepEvent:
        """Check the extraction and the recommendation."""
        extraction, recommendation = state.extraction, state.recommendation
        if extraction is None or recommendation is None:
            return InvalidOutput(detail="nothing to review")
        if not extraction.amounts_add_up():
            return SentBack(target="extractor", reason=AMOUNTS_FEEDBACK)
        if recommendation.rule_decision != assess(extraction, self._lookup):
            return SentBack(target="investigator", reason=RULES_FEEDBACK)

        response = self._llm.complete(
            build_review_request(extraction, recommendation, self._lookup)
        )
        cost = Decimal(response.usage.total_tokens)
        try:
            verdict = Verdict.model_validate_json(response.content)
        except ValidationError as error:
            problem = error.errors()[0]
            return InvalidOutput(
                cost=cost, detail=f"review: {problem['msg']} at {list(problem['loc'])}"
            )
        if verdict.grounded:
            return Approved(cost=cost)
        return SentBack(
            cost=cost,
            target="investigator",
            reason=verdict.problem.strip()
            or "An explanation states something the facts do not support.",
        )
