import json

from app.core.llm_factory import get_llm
from app.core.llm_metering import add_trace_event
from app.core.structured_output import call_structured
from app.tools.ml_tools import CLASSIFICATION_MODELS, REGRESSION_MODELS

from .agent_schemas import CriticVerdict
from .state import AgentState

MAX_RETRIES = 2


async def run_critic_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "critic"
    state["progress_pct"] = 71
    state["messages"].append("🔬 Critic reviewing results...")

    retry_count = state.get("retry_count", 0)
    state.setdefault("critic_history", [])
    experiments = state.get("experiments", [])
    real_completed = [
        e for e in experiments
        if e.get("status") in ("completed", "completed_reduced") and not e.get("is_baseline")
    ]

    if not real_completed:
        # Nothing trained successfully — nothing for the critic to review, approve straight
        # through so the run still produces a report explaining the failure.
        state["critic_history"].append({
            "verdict": "approve", "confidence": "low",
            "reasoning": "No successful experiments to review.", "concerns": [], "retry_count": retry_count,
        })
        add_trace_event(state, "critic", "critic", state["critic_history"][-1])
        return state

    best = max(real_completed, key=lambda e: e.get("primary_score", 0))
    baseline = state.get("baseline_score")

    try:
        llm = get_llm(state["llm_config"])
        model_name = state["llm_config"].get("model", "")

        context = {
            "best_model": best["model_name"],
            "best_score": best["primary_score"],
            "primary_metric": best["primary_metric"],
            "best_cv_std": (best.get("cv_scores", {})
                            .get(f"test_{best['primary_metric']}", {}).get("std")),
            "baseline_score": baseline.get("primary_score") if baseline else None,
            "retry_count_so_far": retry_count,
            "max_retries": MAX_RETRIES,
            "planner_risk_flags": (state.get("plan") or {}).get("risk_flags", []),
            "models_tried": [e["model_name"] for e in real_completed],
        }

        prompt = f"""You are the critic in an autonomous ML pipeline. Review these training
results and decide whether they're trustworthy enough to move to final evaluation, or whether
the pipeline should retry with a different strategy.

Context:
{json.dumps(context, indent=2)}

Guidance:
- If best_score is barely above baseline_score (or below it), the model has little real value —
  use verdict="insufficient_signal" (this still proceeds, but flags low confidence honestly).
- If best_cv_std is large relative to best_score (unstable across folds), consider
  verdict="retry_models" with a revised_model_shortlist (e.g. simpler, more regularized models).
- If you suspect a specific untreated leakage/ID column caused an unrealistically high score,
  use verdict="retry_features" and list it in additional_columns_to_drop.
- retry_count_so_far={retry_count} of max_retries={MAX_RETRIES} — if you've already used all
  retries, prefer "approve" or "insufficient_signal" over asking for another retry."""

        verdict = call_structured(llm, CriticVerdict, prompt, state, "critic", model_name)

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["warnings"].append(f"Critic LLM call failed, defaulting to approve: {error_msg[:100]}")
        verdict = CriticVerdict(
            verdict="approve", confidence="low",
            reasoning="Critic LLM unavailable — approved by default to avoid blocking the run.",
        )

    effective_verdict = verdict.verdict
    if effective_verdict in ("retry_features", "retry_models") and retry_count >= MAX_RETRIES:
        effective_verdict = "approve"
        verdict.reasoning += f" (forced approve — {MAX_RETRIES} retries already used)"
        verdict.confidence = "low"

    history_entry = {
        "verdict": effective_verdict,
        "requested_verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "reasoning": verdict.reasoning,
        "concerns": verdict.concerns,
        "retry_count": retry_count,
    }
    state["critic_history"].append(history_entry)
    add_trace_event(state, "critic", "critic", history_entry)
    state["messages"].append(f"  Verdict: {effective_verdict} ({verdict.confidence} confidence) — {verdict.reasoning[:120]}")

    if effective_verdict == "retry_features":
        state["retry_count"] = retry_count + 1
        plan = state.get("plan") or {}
        existing_drops = plan.get("columns_to_drop", [])
        new_drops = [d.model_dump() for d in verdict.additional_columns_to_drop]
        plan["columns_to_drop"] = existing_drops + new_drops
        state["plan"] = plan
        add_trace_event(state, "retry", "critic", {"from": "critic", "to": "features", "reason": verdict.reasoning})
        state["messages"].append(f"  ↺ Retrying feature engineering (attempt {retry_count + 1}/{MAX_RETRIES})")

    elif effective_verdict == "retry_models":
        state["retry_count"] = retry_count + 1
        plan = state.get("plan") or {}
        models = (CLASSIFICATION_MODELS if state["problem_type"] == "classification"
                  else REGRESSION_MODELS)
        revised = [m for m in verdict.revised_model_shortlist if m in models]
        if revised:
            plan["model_shortlist"] = revised
        state["plan"] = plan
        add_trace_event(state, "retry", "critic", {"from": "critic", "to": "train", "reason": verdict.reasoning})
        state["messages"].append(f"  ↺ Retrying model training (attempt {retry_count + 1}/{MAX_RETRIES})")

    state["progress_pct"] = 72
    return state
