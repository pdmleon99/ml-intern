"""Evals for the Planner agent. Two kinds of check here:
  1. Wiring: given a scripted LLM decision, does the rest of the pipeline actually apply it?
  2. Real deterministic logic: when the LLM is unavailable, does the fallback plan correctly
     use EDA's own (non-LLM) leakage detection? No LLM involved — this tests real code paths.
"""

import pytest

from app.agents.agent_schemas import AnalysisPlan, ColumnDropDecision
from app.agents.eda_agent import run_eda_agent
from app.agents.feature_agent import run_feature_agent
from app.agents.planner_agent import run_planner_agent
from tests.evals.datasets import leakage_dataset
from tests.llm_stub import FakeLLM
from tests.state_helpers import make_state

EDA_NARRATIVE_RESPONSE = (
    "Write a JSON response with exactly these three fields",
    '{"summary": "s", "key_findings": ["a"], "fe_recommendations": ["b"]}',
)


@pytest.mark.asyncio
async def test_planner_drop_decision_is_actually_applied(tmp_path, mock_llm_config, monkeypatch):
    """If the Planner (LLM) decides to drop a column, Feature agent must really drop it —
    catches the class of bug where a decision is computed but never wired into execution."""
    df = leakage_dataset()
    csv_path = tmp_path / "leakage.csv"
    df.to_csv(csv_path, index=False)

    plan = AnalysisPlan(
        reasoning="leak_col is near-identical to the target",
        columns_to_drop=[ColumnDropDecision(column="leak_col", reason="leakage")],
        model_shortlist=["logistic_regression", "random_forest"],
        risk_flags=[],
    )
    fake_llm = FakeLLM(
        structured_responses={AnalysisPlan: plan},
        plain_responses=[EDA_NARRATIVE_RESPONSE],
    )
    monkeypatch.setattr("app.agents.eda_agent.get_llm", lambda cfg: fake_llm)
    monkeypatch.setattr("app.agents.planner_agent.get_llm", lambda cfg: fake_llm)

    state = make_state(str(csv_path), "target", "classification", mock_llm_config)
    state = await run_eda_agent(state)
    state = await run_planner_agent(state)
    assert state["plan"]["model_shortlist"] == ["logistic_regression", "random_forest"]

    state = await run_feature_agent(state)
    assert "leak_col" not in state["feature_names"]
    assert any(d["column"] == "leak_col" for d in state["dropped_columns"])


@pytest.mark.asyncio
async def test_planner_fallback_uses_eda_own_leakage_detection(tmp_path, mock_llm_config, monkeypatch):
    """No LLM available at all — the fallback plan must still catch leakage, because it reads
    EDA's own rule-based leakage finding (correlation > 0.98 with target), not an LLM call."""
    df = leakage_dataset()
    csv_path = tmp_path / "leakage.csv"
    df.to_csv(csv_path, index=False)

    fake_llm = FakeLLM(plain_responses=[EDA_NARRATIVE_RESPONSE])
    monkeypatch.setattr("app.agents.eda_agent.get_llm", lambda cfg: fake_llm)

    def _raise(cfg):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr("app.agents.planner_agent.get_llm", _raise)

    state = make_state(str(csv_path), "target", "classification", mock_llm_config)
    state = await run_eda_agent(state)
    assert any("leakage" in f["finding"].lower() for f in state["eda_findings"])

    state = await run_planner_agent(state)
    dropped = [d["column"] for d in state["plan"]["columns_to_drop"]]
    assert "leak_col" in dropped
    assert any("fallback" in w.lower() for w in state["warnings"])
