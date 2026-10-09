"""A client for any OpenAI-compatible chat completions API.

Nothing here is specific to one provider: the base URL, the model name and
the key come from the settings, and the request uses only the parts of the
API that are common to OpenAI-compatible servers. No vendor SDK is involved.

The API key is sent in the Authorization header and appears nowhere else: not
in error messages, not in logs and not in recordings.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from casefile.config import LLMSettings

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0
# Worth trying again: the server was busy or failed, not the request.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """The model could not be reached or did not give a usable answer."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Frozen):
    """One message of a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(_Frozen):
    """What is asked of the model.

    With ``json_schema`` the server constrains the answer to that schema.
    ``max_tokens`` limits the whole answer, reasoning included, and is what
    makes the worst-case cost of a call known before it is made.
    """

    messages: tuple[Message, ...] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    json_schema: dict[str, Any] | None = None
    schema_name: str = "answer"
    temperature: float = 0.0
    seed: int | None = 0
    # How hard the model should think about this request; ``None`` leaves it
    # to the configured default. Reading a document needs little thought,
    # applying rules to dates and amounts needs more.
    reasoning_effort: str | None = None


class Usage(_Frozen):
    """The tokens one call consumed, as reported by the server."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(_Frozen):
    """The model's answer."""

    content: str
    finish_reason: str | None
    usage: Usage
    latency_seconds: float = Field(ge=0)


class LLM(Protocol):
    """Anything that can answer a chat request."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the model's answer to ``request``."""
        ...


class OpenAICompatibleClient:
    """Sends chat requests over HTTP, retrying when the server is busy."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=settings.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
                "Accept": "application/json",
            },
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        """Release the connection pool."""
        self._http.close()

    def payload(self, request: ChatRequest) -> dict[str, Any]:
        """Return the JSON body that ``request`` is sent as."""
        body: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        effort = request.reasoning_effort or self._settings.reasoning_effort
        if effort:
            body["reasoning_effort"] = effort
        if request.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        return body

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Send ``request`` and return the answer.

        A busy or failing server is tried again a few times. Anything else,
        and a server that stays unavailable, raises ``LLMError``.
        """
        body = self.payload(request)
        problem = "no attempt was made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                self._sleep(RETRY_DELAY_SECONDS * (attempt - 1))
            started = time.perf_counter()
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                problem = f"{type(error).__name__}: {error}"
                continue
            if response.status_code in RETRYABLE_STATUS:
                problem = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
            if response.status_code != httpx.codes.OK:
                raise LLMError(
                    f"The model API answered HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            return _parse(response, time.perf_counter() - started)
        raise LLMError(
            f"The model API did not answer after {MAX_ATTEMPTS} attempts ({problem})."
        )


def _parse(response: httpx.Response, latency: float) -> ChatResponse:
    """Turn a successful HTTP response into a ``ChatResponse``."""
    try:
        data = response.json()
        choice = data["choices"][0]
        usage = data["usage"]
        details = usage.get("completion_tokens_details") or {}
        return ChatResponse(
            # A reasoning model that ran out of tokens returns no content.
            content=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                reasoning_tokens=details.get("reasoning_tokens") or 0,
            ),
            latency_seconds=latency,
        )
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LLMError(
            f"The model API answered in an unexpected shape: {response.text[:300]}"
        ) from error
