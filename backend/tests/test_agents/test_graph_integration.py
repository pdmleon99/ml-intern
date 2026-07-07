"""End-to-end smoke test for the full agentic graph (EDA -> Planner -> Features -> Train ->
Critic -> Evaluation -> Report), including a forced Critic retry loop. This is the fastest way
to catch wiring bugs (state keys, conditional routing, loop termination) across the whole
pipeline without a real LLM."""

import numpy as np
import pandas as pd
import pytest

from app.agents.agent_schemas import AnalysisPlan, ColumnDropDecision, CriticVerdict
from tests.llm_stub import FakeLLM

PLAIN_RESPONSES = [
    ("Write a JSON response with exactly these three fields",
     '{"summary": "test summary", "key_findings": ["a", "b"], "fe_recommendations": ["c", "d"]}'),
    ("explaining an ML model to a business stakeholder",
     "This model predicts the target reasonably well using the available features."),
    ("Write EXACTLY 5 recommendations",
     '["r1", "r2", "r3", "r4", "r5"]'),
]


def _patch_llm(monkeypatch, fake_llm):
    for module in ("eda_agent", "planner_agent", "critic_agent", "evaluation_agent", "report_agent"):
        monkeypatch.setattr(f"app.agents.{module}.get_llm", lambda cfg, _llm=fake_llm: _llm)


@pytest.mark.asyncio
async def test_full_graph_with_forced_retry(tmp_path, mock_llm_config, monkeypatch):
    rng = np.random.RandomState(0)
    n = 300
    f1 = rng.randn(n)
    f2 = rng.randn(n)
    target = (f1 + rng.randn(n) * 0.5 > 0).astype(int)
    df = pd.DataFrame({
        "f1": f1, "f2": f2,
        "leak_col": target + rng.randn(n) * 0.01,  # near-perfect leakage — Planner should drop it
        "target": target,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    plan = AnalysisPlan(
        reasoning="test plan",
        columns_to_drop=[ColumnDropDecision(column="leak_col", reason="leakage")],
        model_shortlist=["logistic_regression", "random_forest"],
        risk_flags=[],
    )
    critic_responses = iter([
        CriticVerdict(verdict="retry_models", confidence="medium", reasoning="unstable CV",
                      concerns=["high variance"], revised_model_shortlist=["random_forest", "extra_trees"]),
        CriticVerdict(verdict="approve", confidence="high", reasoning="looks good now"),
    ])

    fake_llm = FakeLLM(
        structured_responses={
            AnalysisPlan: plan,
            CriticVerdict: lambda: next(critic_responses),
        },
        plain_responses=PLAIN_RESPONSES,
    )
    _patch_llm(monkeypatch, fake_llm)

    from app.agents.graph import build_graph
    graph = build_graph(checkpointer=None)

    initial_state = {
        "llm_config": mock_llm_config,
        "job_id": "integration-test",
        "dataset_path": str(csv_path),
        "user_description": "Predict target",
        "target_column": "target",
        "problem_type": "classification",
        "profile": None, "df_sample_path": None, "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "preprocessor_path": None,
        "feature_names": [], "dropped_columns": [], "plan": None,
        "experiments": [], "baseline_score": None, "critic_history": [], "retry_count": 0,
        "best_model_name": None, "best_model_path": None, "evaluation_metrics": None,
        "evaluation_charts": [], "model_explanation": None, "statistical_comparison": None,
        "shap_summary": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "starting", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [], "token_usage": [], "trace_events": [],
    }

    final_state = await graph.ainvoke(
        initial_state, config={"configurable": {"thread_id": "integration-test"}}
    )

    assert final_state["status"] == "completed", final_state.get("errors")
    assert "leak_col" not in final_state["feature_names"]
    assert any(d["column"] == "leak_col" for d in final_state["dropped_columns"])
    assert final_state["retry_count"] == 1
    verdicts = [c["verdict"] for c in final_state["critic_history"]]
    assert verdicts == ["retry_models", "approve"]
    assert final_state["best_model_name"] in ("random_forest", "extra_trees")
    assert final_state["report_pdf_path"] is not None
    assert len(final_state["token_usage"]) > 0
    assert any(e["kind"] == "retry" for e in final_state["trace_events"])

    # These must not be silently None — assert on content, not just absence of a crash,
    # since a broad except elsewhere could otherwise hide a real failure here.
    assert not any("SHAP" in w for w in final_state["warnings"]), final_state["warnings"]
    assert final_state["shap_summary"] is not None
    assert final_state["shap_summary"]["top_features"]
    assert final_state["statistical_comparison"] is not None
    assert final_state["statistical_comparison"]["vs_baseline"] is not None
