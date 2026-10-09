"""Record model answers to disk and play them back.

A model does not answer the same way twice, so a run cannot be repeated by
calling it again. With a recording it can: in replay mode every request gets
the answer it got the first time, without any network access and without an
API key.

Each answer is stored in its own JSON file, named after a digest of the
request and how many times that exact request had been made before. The count
matters: when an agent is asked to try again, it may send the same request
twice and receive two different answers, and a replay must return them in the
same order.

The files hold the request and the answer. They never hold the API key, which
is not part of a request.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from casefile.llm.client import LLM, ChatRequest, ChatResponse

# record: always call the model and store the answer
# replay: never call the model; a missing answer is an error
# auto:   replay what is stored, call the model for the rest
RecordingMode = Literal["record", "replay", "auto"]


class MissingRecordingError(LookupError):
    """Replay was asked for a request that was never recorded."""


def request_digest(request: ChatRequest) -> str:
    """Return a stable digest of everything that defines ``request``.

    Unset optional parts are left out, so adding an optional field to
    ``ChatRequest`` later does not invalidate recordings made before it.
    """
    canonical = json.dumps(
        request.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class RecordingLLM:
    """Wrap a model with a recording kept in ``directory``.

    One instance stands for one run. ``inner`` may be ``None`` in replay
    mode, where the model is never called.
    """

    def __init__(
        self,
        directory: Path,
        mode: RecordingMode,
        inner: LLM | None = None,
        *,
        model: str = "",
    ) -> None:
        if mode != "replay" and inner is None:
            raise ValueError(f"mode {mode!r} needs a model to call")
        self._directory = directory
        self._mode = mode
        self._inner = inner
        self._model = model
        self._seen: Counter[str] = Counter()
        self.replayed = 0
        self.recorded = 0
        # Every recording file this run read or wrote, in order.
        self.paths: list[Path] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Return the recorded answer to ``request``, or record a new one."""
        digest = request_digest(request)
        self._seen[digest] += 1
        path = self._directory / f"{digest[:16]}-{self._seen[digest]:02d}.json"

        self.paths.append(path)
        if self._mode != "record" and path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.replayed += 1
            return ChatResponse.model_validate(stored["response"])
        if self._mode == "replay":
            raise MissingRecordingError(
                f"No recorded answer for this request ({path.name}) in "
                f"{self._directory}."
            )

        response = self._inner.complete(request)
        self._directory.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self._model,
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.recorded += 1
        return response
