import json

from app.core.llm_factory import get_llm
from app.core.llm_metering import add_trace_event
from app.core.structured_output import call_structured
from app.tools.ml_tools import CLASSIFICATION_MODELS, REGRESSION_MODELS

from .agent_schemas import AnalysisPlan, ColumnDropDecision
from .state import AgentState


def _fallback_plan(state: AgentState) -> AnalysisPlan:
    """Deterministic fallback if the LLM is unavailable — derives drops from EDA's own
    leakage/ID findings instead of just defaulting to 'no plan'."""
    findings = state.get("eda_findings", [])
    drops = []
    for f in findings:
        text = f.get("finding", "").lower()
        col = f.get("column")
        if not col or col in ("dataset",) or col == state.get("target_column"):
            continue
        if "leakage" in text or "id column" in text:
            drops.append(ColumnDropDecision(column=col, reason=f.get("finding", "")))

    models = (CLASSIFICATION_MODELS if state["problem_type"] == "classification"
              else REGRESSION_MODELS)
    return AnalysisPlan(
        reasoning="Fallback plan (LLM unavailable): dropping only columns EDA flagged as "
                  "leakage/ID, training all available models.",
        columns_to_drop=drops,
        model_shortlist=list(models.keys()),
        risk_flags=["LLM planning unavailable — used deterministic fallback"],
    )


async def run_planner_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "planner"
    state["progress_pct"] = 21
    state["messages"].append("🧠 Planning analysis strategy...")

    try:
        llm = get_llm(state["llm_config"])
        model_name = state["llm_config"].get("model", "")

        profile: dict = dict(state.get("profile") or {})
        findings = state.get("eda_findings", [])
        problem_type = state["problem_type"]
        models = (CLASSIFICATION_MODELS if problem_type == "classification"
                  else REGRESSION_MODELS)

        compact_context = {
            "problem_type": problem_type,
            "target": state["target_column"],
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "available_models": list(models.keys()),
            "critical_findings": [f for f in findings if f.get("severity") == "critical"],
            "warning_findings": [f for f in findings if f.get("severity") == "warning"][:8],
        }

        prompt = f"""You are the strategist for an autonomous ML pipeline. Decide the analysis
plan for this dataset based on the EDA findings below. Be concrete and specific.

Context:
{json.dumps(compact_context, indent=2)}

Rules:
- Only put a column in columns_to_drop if a critical finding gives clear evidence (data leakage,
  ID column, constant column). Do not invent columns not mentioned in the findings.
- model_shortlist must be a subset (in priority order) of available_models. For very small
  datasets (< 500 rows) or very high dimensionality relative to rows, prefer simpler/regularized
  models and consider dropping heavy boosters to reduce overfitting risk — but always include
  at least 2 models for a meaningful comparison.
- risk_flags should call out anything a data scientist should be cautious about."""

        plan = await call_structured(llm, AnalysisPlan, prompt, state, "planner", model_name)

        valid_models = set(models.keys())
        plan.model_shortlist = [m for m in plan.model_shortlist if m in valid_models] or list(valid_models)

        plan_dict = plan.model_dump()
        state["plan"] = plan_dict
        add_trace_event(state, "thought", "planner", {"reasoning": plan.reasoning})
        add_trace_event(state, "plan", "planner", plan_dict)

        drop_summary = ", ".join(d.column for d in plan.columns_to_drop) or "none"
        state["messages"].append(
            f"  Plan: train {plan.model_shortlist} | drop [{drop_summary}]"
        )
        if plan.risk_flags:
            state["messages"].append(f"  Risk flags: {', '.join(plan.risk_flags)}")
        state["messages"].append("✓ Planning complete")

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["warnings"].append(f"Planner LLM call failed, using deterministic fallback: {error_msg[:100]}")
        plan = _fallback_plan(state)
        plan_dict = plan.model_dump()
        state["plan"] = plan_dict
        add_trace_event(state, "plan", "planner", plan_dict)

    return state
