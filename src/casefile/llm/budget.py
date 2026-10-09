"""Know what a model call can cost before it is made.

Cost is counted in tokens, not money: the model is used through a free trial
service, so there is no price to apply. A token cap per case still matters. It
keeps a run that goes wrong from using up the trial allowance, and it is what
the supervisor's budget guard checks before every call.

The worst case of a request must never be lower than what the call really
uses, or the guard would be meaningless. No tokenizer is assumed here.
Instead the bound rests on a fact that holds for every tokenizer: a token
covers at least one byte of text.
"""

from __future__ import annotations

import json
from typing import Protocol

from casefile.llm.client import LLM, ChatRequest, ChatResponse

# Allowance for what the server adds around the text: role markers, the
# beginning-of-sequence token and similar.
REQUEST_OVERHEAD_TOKENS = 64
MESSAGE_OVERHEAD_TOKENS = 8


class TokenBoundError(RuntimeError):
    """A call used more tokens than its worst case allowed for."""


class MeteredLLM(Protocol):
    """A model whose calls have a known worst case, such as ``BudgetedLLM``."""

    def worst_case(self, request: ChatRequest) -> int:
        """Return the most tokens ``request`` can consume."""
        ...

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the model's answer to ``request``."""
        ...


def worst_case_tokens(request: ChatRequest) -> int:
    """Return an upper bound on the tokens ``request`` can consume.

    The prompt cannot have more tokens than it has bytes, and the answer
    cannot exceed ``max_tokens``. The bound is loose for the prompt, by a
    factor of three to four for English text, and that is the price of not
    depending on any one model's tokenizer.
    """
    prompt_bytes = sum(len(message.content.encode()) for message in request.messages)
    if request.json_schema is not None:
        prompt_bytes += len(json.dumps(request.json_schema).encode())
    overhead = REQUEST_OVERHEAD_TOKENS + MESSAGE_OVERHEAD_TOKENS * len(request.messages)
    return prompt_bytes + overhead + request.max_tokens


class BudgetedLLM:
    """Wrap a model so that every call is metered against its worst case.

    This class does not enforce a cap; the supervisor does, by asking for the
    worst case before a step. What is enforced here is that the worst case was
    honest: a call that uses more than its bound raises, so an optimistic
    bound cannot go unnoticed.
    """

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        self.calls = 0
        self.tokens_used = 0

    def worst_case(self, request: ChatRequest) -> int:
        """Return the most tokens ``request`` can consume."""
        return worst_case_tokens(request)

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Answer ``request`` and add its real usage to the running totals."""
        bound = worst_case_tokens(request)
        response = self._inner.complete(request)
        used = response.usage.total_tokens
        self.calls += 1
        self.tokens_used += used
        if used > bound:
            raise TokenBoundError(
                f"A call used {used} tokens, above its worst case of {bound}."
            )
        return response
