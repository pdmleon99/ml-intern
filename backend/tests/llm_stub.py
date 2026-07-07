"""A deterministic stand-in for a real LLM, used by integration tests and agent evals
(see Phase 4) so the graph's control flow — including the Critic's retry loop — can be
tested without burning real API calls or depending on live model behavior."""

from types import SimpleNamespace
from typing import Callable, Type, Union

from pydantic import BaseModel

FAKE_USAGE = {"input_tokens": 20, "output_tokens": 10}


class _BoundStructuredLLM:
    def __init__(self, response_source: Union[BaseModel, Callable[[], BaseModel]], include_raw: bool):
        self._response_source = response_source
        self._include_raw = include_raw

    def invoke(self, prompt):
        response = self._response_source() if callable(self._response_source) else self._response_source
        if self._include_raw:
            raw = SimpleNamespace(usage_metadata=dict(FAKE_USAGE))
            return {"raw": raw, "parsed": response, "parsing_error": None}
        return response


class FakeLLM:
    """Stands in for a langchain chat model.

    - `structured_responses`: maps a pydantic schema class to either a fixed instance or a
      zero-arg callable (e.g. an iterator's `.__next__`) returning a fresh instance per call —
      the latter is how a test simulates the Critic changing its verdict across retries.
    - `plain_responses`: list of (substring_to_match_in_prompt, response_text) — first match wins.
    """

    def __init__(self, structured_responses: dict = None, plain_responses: list = None):
        self.structured_responses = structured_responses or {}
        self.plain_responses = plain_responses or []

    def invoke(self, prompt: str):
        for substring, text in self.plain_responses:
            if substring in prompt:
                return SimpleNamespace(content=text, usage_metadata=dict(FAKE_USAGE))
        return SimpleNamespace(content="OK", usage_metadata=dict(FAKE_USAGE))

    def with_structured_output(self, schema_cls: Type[BaseModel], include_raw: bool = False):
        if schema_cls not in self.structured_responses:
            raise ValueError(f"FakeLLM has no structured response configured for {schema_cls}")
        return _BoundStructuredLLM(self.structured_responses[schema_cls], include_raw)
