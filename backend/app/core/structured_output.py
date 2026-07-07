import asyncio
import json
from typing import Type, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents.state import AgentState
from app.core.llm_metering import metered_llm_call

T = TypeVar("T", bound=BaseModel)


def _strip_json_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_structured(
    llm,
    schema_cls: Type[T],
    prompt: str,
    state: AgentState,
    agent: str,
    model_name: str,
) -> T:
    """Call an LLM for a structured (pydantic) response, metering tokens/cost/latency.

    Tries native tool-calling structured output first (Anthropic/OpenAI/Groq all support
    `.with_structured_output` via function-calling in the pinned langchain versions). Falls
    back to a manual JSON-in-prompt parse if the provider's structured-output path errors —
    the same resilience pattern the rest of this codebase uses for LLM calls.

    Runs the actual (synchronous) network call in a worker thread via `asyncio.to_thread` —
    langchain's `.invoke()` blocks for seconds at a time, and running it directly on the
    event loop would freeze the whole server (health checks, other jobs' SSE streams) for
    that long, which is exactly what happened before this fix.
    """
    with metered_llm_call(state, agent, model_name) as recorder:
        try:
            structured_llm = llm.with_structured_output(schema_cls, include_raw=True)
            result = await asyncio.to_thread(structured_llm.invoke, prompt)
            recorder(result.get("raw"))
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed
            raise ValueError("structured output returned no parsed result")
        except Exception:
            schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
            fallback_prompt = (
                f"{prompt}\n\n"
                f"Reply with ONLY a JSON object matching this schema, no markdown, no explanation:\n"
                f"{schema_json}"
            )
            response = await asyncio.to_thread(llm.invoke, fallback_prompt)
            recorder(response)
            data = json.loads(_strip_json_fences(response.content))
            return schema_cls(**data)
