"""Check that the configured model can be reached and returns structured output.

    python -m casefile.llm.check

Run this after changing the model settings in .env, or whenever a run fails
with a connection error. It makes one small request and reports what came
back. The API key is never shown.
"""

from __future__ import annotations

import json
import sys

from casefile.config import ConfigurationError, load_llm_settings
from casefile.llm.budget import BudgetedLLM
from casefile.llm.client import ChatRequest, LLMError, Message, OpenAICompatibleClient

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "sum": {"type": "integer"},
    },
    "required": ["status", "sum"],
    "additionalProperties": False,
}
CHECK_REQUEST = ChatRequest(
    messages=(
        Message(
            role="user",
            content='Set "status" to "ok" and "sum" to the result of 17 + 25.',
        ),
    ),
    max_tokens=200,
    json_schema=CHECK_SCHEMA,
    schema_name="connection_check",
)


def main() -> int:
    """Run the check. Returns 0 if the model answered as asked."""
    try:
        settings = load_llm_settings()
    except ConfigurationError as error:
        print(f"Not configured: {error}", file=sys.stderr)
        return 1
    print(f"Model:    {settings.model}")
    print(f"Endpoint: {settings.base_url}")

    client = OpenAICompatibleClient(settings)
    llm = BudgetedLLM(client)
    try:
        response = llm.complete(CHECK_REQUEST)
    except LLMError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        print(
            "If this is a connection error, check your network settings and "
            "that the endpoint in .env is reachable from this machine.",
            file=sys.stderr,
        )
        return 1
    finally:
        client.close()

    usage = response.usage
    print(f"Answered in {response.latency_seconds:.1f} s")
    print(
        f"Tokens:   {usage.prompt_tokens} in, {usage.completion_tokens} out "
        f"({usage.reasoning_tokens} of them reasoning); "
        f"worst case allowed for was {llm.worst_case(CHECK_REQUEST)}"
    )
    try:
        answer = json.loads(response.content)
    except ValueError:
        print(f"FAILED: the answer is not JSON: {response.content!r}", file=sys.stderr)
        return 1
    if answer != {"status": "ok", "sum": 42}:
        print(f"FAILED: unexpected answer {answer}", file=sys.stderr)
        return 1
    print("Structured output: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
