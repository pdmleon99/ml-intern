import time
from contextlib import contextmanager
from datetime import datetime, timezone

from app.agents.state import AgentState

# Approximate $ per 1M tokens (input, output) — public list prices, mid-2026.
# Used only to give the user a concrete cost signal in the UI/report, not for billing.
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
}
DEFAULT_PRICING = (1.0, 3.0)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICING_PER_1M.get(model, DEFAULT_PRICING)
    return round((input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price, 6)


def add_trace_event(state: AgentState, kind: str, agent: str, payload: dict) -> None:
    """Append a structured event for the live agent-brain stream (Phase 3 SSE)."""
    state["trace_events"].append({
        "kind": kind,
        "agent": agent,
        "payload": payload,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


@contextmanager
def metered_llm_call(state: AgentState, agent: str, model: str):
    """Times an LLM call and records token usage + cost into state["token_usage"].

    Usage:
        with metered_llm_call(state, "planner", model_name) as recorder:
            response = llm.invoke(prompt)   # or llm.with_structured_output(...).invoke(...)
            recorder(response)              # pass the raw AIMessage (or object with usage_metadata)
    """
    start = time.time()
    result_holder: dict = {"usage": None}

    def recorder(response):
        usage = getattr(response, "usage_metadata", None)
        if usage:
            result_holder["usage"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }

    try:
        yield recorder
    finally:
        latency_s = round(time.time() - start, 3)
        usage = result_holder["usage"] or {"input_tokens": 0, "output_tokens": 0}
        cost = _estimate_cost(model, usage["input_tokens"], usage["output_tokens"])
        entry = {
            "agent": agent,
            "model": model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost_usd": cost,
            "latency_s": latency_s,
        }
        state["token_usage"].append(entry)
        add_trace_event(state, "token_usage", agent, entry)
