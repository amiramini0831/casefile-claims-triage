"""The model client, token metering and record/replay, without any network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from casefile.config import ConfigurationError, LLMSettings, load_llm_settings
from casefile.llm.budget import BudgetedLLM, TokenBoundError, worst_case_tokens
from casefile.llm.client import (
    MAX_ATTEMPTS,
    ChatRequest,
    ChatResponse,
    LLMError,
    Message,
    OpenAICompatibleClient,
    Usage,
)
from casefile.llm.recording import (
    MissingRecordingError,
    RecordingLLM,
    request_digest,
)

API_KEY = "test-key-that-must-never-leak"
SETTINGS = LLMSettings(
    base_url="https://llm.example/v1/", model="test/model", api_key=SecretStr(API_KEY)
)
SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}}
REQUEST = ChatRequest(
    messages=(
        Message(role="system", content="You extract fields."),
        Message(role="user", content="Total claimed 5,000.00"),
    ),
    max_tokens=100,
    json_schema=SCHEMA,
    schema_name="claim",
)


def completion(content: str | None = '{"n": 1}', **usage: Any) -> dict[str, Any]:
    """The body of a successful chat completion."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2},
            **usage,
        },
    }


class FakeServer:
    """Answers with the queued responses and remembers what it was sent."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def client(self) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            SETTINGS, transport=httpx.MockTransport(self), sleep=lambda seconds: None
        )


def ok(**kwargs: Any) -> httpx.Response:
    return httpx.Response(200, json=completion(**kwargs))


class StubLLM:
    """A model that returns prepared answers and counts its calls."""

    def __init__(self, *contents: str, tokens: int = 27) -> None:
        self._contents = list(contents)
        self._tokens = tokens
        self.calls = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content=self._contents.pop(0),
            finish_reason="stop",
            usage=Usage(prompt_tokens=self._tokens - 7, completion_tokens=7),
            latency_seconds=0.5,
        )


# --- the client --------------------------------------------------------------


def test_request_is_sent_in_the_openai_shape() -> None:
    server = FakeServer(ok())

    server.client().complete(REQUEST)

    (sent,) = server.requests
    body = json.loads(sent.content)
    assert str(sent.url) == "https://llm.example/v1/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {API_KEY}"
    assert body["model"] == "test/model"
    assert body["messages"][1] == {"role": "user", "content": "Total claimed 5,000.00"}
    assert (body["temperature"], body["max_tokens"], body["seed"]) == (0.0, 100, 0)
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "claim", "schema": SCHEMA, "strict": True},
    }


def test_optional_parts_are_left_out_when_not_set() -> None:
    settings = SETTINGS.model_copy(update={"reasoning_effort": None})
    client = OpenAICompatibleClient(settings)
    plain = ChatRequest(messages=REQUEST.messages, max_tokens=10, seed=None)

    body = client.payload(plain)

    assert not {"reasoning_effort", "response_format", "seed"} & body.keys()


def test_request_can_ask_for_more_reasoning_than_the_default() -> None:
    client = OpenAICompatibleClient(SETTINGS)
    thoughtful = REQUEST.model_copy(update={"reasoning_effort": "high"})

    assert client.payload(REQUEST)["reasoning_effort"] == "low"
    assert client.payload(thoughtful)["reasoning_effort"] == "high"
    assert request_digest(thoughtful) != request_digest(REQUEST)


def test_answer_and_usage_are_read_from_the_response() -> None:
    response = FakeServer(ok()).client().complete(REQUEST)

    assert response.content == '{"n": 1}'
    assert response.finish_reason == "stop"
    assert response.usage == Usage(
        prompt_tokens=20, completion_tokens=7, reasoning_tokens=2
    )
    assert response.usage.total_tokens == 27


def test_missing_content_becomes_an_empty_answer() -> None:
    """A reasoning model that runs out of tokens returns no content at all."""
    assert FakeServer(ok(content=None)).client().complete(REQUEST).content == ""


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503, text="busy"),
        httpx.Response(429, text="slow down"),
        httpx.ConnectError("no route"),
        httpx.RemoteProtocolError("closed without response"),
        httpx.ReadTimeout("too slow"),
    ],
)
def test_busy_server_is_tried_again(failure: httpx.Response | Exception) -> None:
    server = FakeServer(failure, ok())

    response = server.client().complete(REQUEST)

    assert response.content == '{"n": 1}'
    assert len(server.requests) == 2


def test_server_that_stays_down_is_given_up_on() -> None:
    server = FakeServer(*[httpx.ConnectError("no route")] * MAX_ATTEMPTS)

    with pytest.raises(LLMError, match="did not answer after 3 attempts"):
        server.client().complete(REQUEST)

    assert len(server.requests) == MAX_ATTEMPTS


def test_rejected_request_is_not_retried() -> None:
    server = FakeServer(httpx.Response(400, text="unknown field"))

    with pytest.raises(LLMError, match="HTTP 400: unknown field"):
        server.client().complete(REQUEST)

    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "body", [{"choices": []}, {"choices": [{"message": {}}]}, {"unexpected": True}]
)
def test_unexpected_response_shape_is_an_error(body: dict[str, Any]) -> None:
    server = FakeServer(httpx.Response(200, json=body))

    with pytest.raises(LLMError, match="unexpected shape"):
        server.client().complete(REQUEST)


def test_api_key_never_appears_in_errors_or_reprs() -> None:
    server = FakeServer(httpx.Response(401, text="bad key"))

    with pytest.raises(LLMError) as raised:
        server.client().complete(REQUEST)

    assert API_KEY not in str(raised.value)
    assert API_KEY not in repr(SETTINGS)
    assert API_KEY not in SETTINGS.model_dump_json()


def test_settings_are_read_from_the_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "CASEFILE_LLM_BASE_URL=https://llm.example/v1\n"
        "CASEFILE_LLM_MODEL=test/model\n"
        'CASEFILE_LLM_API_KEY="quoted-key"\n'
        "CASEFILE_LLM_REASONING_EFFORT=none\n",
        encoding="utf-8",
    )

    settings = load_llm_settings(env)

    assert settings.model == "test/model"
    assert settings.api_key.get_secret_value() == "quoted-key"
    assert settings.reasoning_effort is None


def test_missing_key_is_reported_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CASEFILE_LLM_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "CASEFILE_LLM_BASE_URL=https://llm.example/v1\nCASEFILE_LLM_MODEL=m\n"
        "CASEFILE_LLM_API_KEY=\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="CASEFILE_LLM_API_KEY"):
        load_llm_settings(env)


# --- token metering ----------------------------------------------------------


def test_worst_case_counts_every_byte_and_the_whole_answer() -> None:
    request = ChatRequest(
        messages=(Message(role="user", content="é" * 10),),  # two bytes each
        max_tokens=100,
        json_schema={"a": 1},
    )

    assert worst_case_tokens(request) == 20 + len('{"a": 1}') + 64 + 8 + 100


def test_worst_case_covers_a_real_call() -> None:
    """Measured against GLM-5.3: this request used 294 + 56 tokens."""
    document = "Casefile Mutual Insurance Automobile Claim Form " * 18  # ~860 chars
    request = ChatRequest(
        messages=(
            Message(role="system", content="You extract fields. Answer with JSON."),
            Message(role="user", content="Extract the claim fields.\n\n" + document),
        ),
        max_tokens=600,
    )

    assert worst_case_tokens(request) >= 294 + 600


def test_usage_is_added_up() -> None:
    llm = BudgetedLLM(StubLLM("a", "b"))

    llm.complete(REQUEST)
    llm.complete(REQUEST)

    assert (llm.calls, llm.tokens_used) == (2, 54)
    assert llm.worst_case(REQUEST) == worst_case_tokens(REQUEST)


def test_call_above_its_worst_case_is_an_error() -> None:
    llm = BudgetedLLM(StubLLM("a", tokens=10_000))

    with pytest.raises(TokenBoundError, match="above its worst case"):
        llm.complete(REQUEST)


# --- record and replay ---------------------------------------------------------


def test_recorded_answers_are_replayed_without_the_model(tmp_path: Path) -> None:
    model = StubLLM("first answer")
    recorded = RecordingLLM(tmp_path, "record", model, model="test/model").complete(
        REQUEST
    )

    replayer = RecordingLLM(tmp_path, "replay")
    replayed = replayer.complete(REQUEST)

    assert replayed == recorded
    assert (model.calls, replayer.replayed, replayer.recorded) == (1, 1, 0)


def test_repeated_request_replays_its_answers_in_order(tmp_path: Path) -> None:
    recorder = RecordingLLM(tmp_path, "record", StubLLM("try one", "try two"))
    recorder.complete(REQUEST)
    recorder.complete(REQUEST)

    replayer = RecordingLLM(tmp_path, "replay")

    assert replayer.complete(REQUEST).content == "try one"
    assert replayer.complete(REQUEST).content == "try two"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_replay_of_an_unrecorded_request_is_an_error(tmp_path: Path) -> None:
    other = REQUEST.model_copy(update={"max_tokens": 5})
    RecordingLLM(tmp_path, "record", StubLLM("answer")).complete(REQUEST)

    with pytest.raises(MissingRecordingError, match="No recorded answer"):
        RecordingLLM(tmp_path, "replay").complete(other)


def test_auto_mode_calls_the_model_only_for_what_is_missing(tmp_path: Path) -> None:
    RecordingLLM(tmp_path, "record", StubLLM("stored")).complete(REQUEST)
    model = StubLLM("fresh")
    auto = RecordingLLM(tmp_path, "auto", model)
    other = REQUEST.model_copy(update={"max_tokens": 5})

    assert auto.complete(REQUEST).content == "stored"
    assert auto.complete(other).content == "fresh"
    assert (model.calls, auto.replayed, auto.recorded) == (1, 1, 1)


def test_record_mode_overwrites_an_old_answer(tmp_path: Path) -> None:
    RecordingLLM(tmp_path, "record", StubLLM("old")).complete(REQUEST)
    RecordingLLM(tmp_path, "record", StubLLM("new")).complete(REQUEST)

    assert RecordingLLM(tmp_path, "replay").complete(REQUEST).content == "new"


def test_recording_holds_the_request_and_no_key(tmp_path: Path) -> None:
    server = FakeServer(ok())
    RecordingLLM(tmp_path, "record", server.client(), model="test/model").complete(
        REQUEST
    )

    (path,) = tmp_path.glob("*.json")
    text = path.read_text(encoding="utf-8")
    stored = json.loads(text)

    assert path.name == f"{request_digest(REQUEST)[:16]}-01.json"
    assert stored["model"] == "test/model"
    assert stored["request"]["messages"][1]["content"] == "Total claimed 5,000.00"
    assert stored["response"]["content"] == '{"n": 1}'
    assert API_KEY not in text


def test_digest_changes_with_any_part_of_the_request() -> None:
    variants = [
        REQUEST,
        REQUEST.model_copy(update={"max_tokens": 101}),
        REQUEST.model_copy(update={"seed": 1}),
        REQUEST.model_copy(update={"json_schema": None}),
        REQUEST.model_copy(update={"messages": REQUEST.messages[:1]}),
    ]

    assert len({request_digest(request) for request in variants}) == len(variants)


def test_recording_modes_other_than_replay_need_a_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a model"):
        RecordingLLM(tmp_path, "auto")
