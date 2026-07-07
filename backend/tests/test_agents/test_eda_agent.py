from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_eda_titanic(mock_agent_state, mock_llm, titanic_csv):
    with patch("app.agents.eda_agent.get_llm", return_value=mock_llm):
        from app.agents.eda_agent import run_eda_agent
        state = dict(mock_agent_state)
        state["dataset_path"] = titanic_csv
        state["target_column"] = "Survived"
        result = await run_eda_agent(state)

    assert result["profile"] is not None
    assert result["profile"]["n_rows"] == 891
    assert result["target_column"] == "Survived"
    assert result["problem_type"] == "classification"
    assert len(result["eda_findings"]) >= 0
    assert result["status"] != "failed" or len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_eda_large_synthetic(tmp_path, mock_llm_config, mock_llm):
    rng = np.random.RandomState(0)
    n = 150_000
    df = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.choice(["X", "Y", "Z"], n),
        "target": rng.randint(0, 2, n),
    })
    csv_path = str(tmp_path / "large.csv")
    df.to_csv(csv_path, index=False)

    state = {
        "llm_config": mock_llm_config,
        "job_id": "test-large",
        "dataset_path": csv_path,
        "user_description": "Predict target",
        "target_column": "target",
        "problem_type": "unknown",
        "profile": None, "df_sample_path": None, "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "feature_names": [],
        "dropped_columns": [], "experiments": [], "best_model_name": None,
        "best_model_path": None, "evaluation_metrics": None, "evaluation_charts": [],
        "model_explanation": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "eda", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [],
    }

    with patch("app.agents.eda_agent.get_llm", return_value=mock_llm):
        from app.agents.eda_agent import run_eda_agent
        result = await run_eda_agent(state)

    assert result["profile"] is not None
    assert result["profile"]["is_large"] is True
    assert result["df_sample_path"] is not None
    assert any("sample" in m.lower() for m in result["messages"])


@pytest.mark.asyncio
async def test_eda_high_missing(tmp_path, mock_llm_config, mock_llm):
    n = 500
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "good_feature": rng.randn(n),
        "mostly_missing": np.where(rng.random(n) < 0.6, np.nan, rng.randn(n)),
        "target": rng.randint(0, 2, n),
    })
    csv_path = str(tmp_path / "high_missing.csv")
    df.to_csv(csv_path, index=False)

    state = {
        "llm_config": mock_llm_config, "job_id": "test-missing",
        "dataset_path": csv_path, "user_description": "Predict target",
        "target_column": "target", "problem_type": "unknown",
        "profile": None, "df_sample_path": None, "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "feature_names": [],
        "dropped_columns": [], "experiments": [], "best_model_name": None,
        "best_model_path": None, "evaluation_metrics": None, "evaluation_charts": [],
        "model_explanation": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "eda", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [],
    }

    with patch("app.agents.eda_agent.get_llm", return_value=mock_llm):
        from app.agents.eda_agent import run_eda_agent
        result = await run_eda_agent(state)

    critical_missing = [
        f for f in result["eda_findings"]
        if f.get("severity") == "critical" and "missing" in f.get("finding", "").lower()
    ]
    assert len(critical_missing) >= 1


@pytest.mark.asyncio
async def test_eda_no_numeric_cols(tmp_path, mock_llm_config, mock_llm):
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], n),
        "size": rng.choice(["S", "M", "L", "XL"], n),
        "material": rng.choice(["cotton", "silk", "wool"], n),
        "label": rng.choice(["A", "B"], n),
    })
    csv_path = str(tmp_path / "all_cat.csv")
    df.to_csv(csv_path, index=False)

    mock_llm.invoke.return_value = MagicMock(content='{"summary": "All categorical.", "key_findings": ["f1"], "fe_recommendations": ["r1"]}')

    state = {
        "llm_config": mock_llm_config, "job_id": "test-allcat",
        "dataset_path": csv_path, "user_description": "Predict label",
        "target_column": "label", "problem_type": "unknown",
        "profile": None, "df_sample_path": None, "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "feature_names": [],
        "dropped_columns": [], "experiments": [], "best_model_name": None,
        "best_model_path": None, "evaluation_metrics": None, "evaluation_charts": [],
        "model_explanation": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "eda", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [],
    }

    with patch("app.agents.eda_agent.get_llm", return_value=mock_llm):
        from app.agents.eda_agent import run_eda_agent
        result = await run_eda_agent(state)

    assert result["profile"] is not None
    assert result["profile"]["n_numeric"] == 0


@pytest.mark.asyncio
async def test_eda_latin1_encoding(tmp_path, mock_llm_config, mock_llm):
    df = pd.DataFrame({
        "name": ["André", "François", "Ñoño", "Müller"],
        "value": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })
    csv_path = str(tmp_path / "latin1.csv")
    df.to_csv(csv_path, index=False, encoding="latin-1")

    state = {
        "llm_config": mock_llm_config, "job_id": "test-latin1",
        "dataset_path": csv_path, "user_description": "Predict target",
        "target_column": "target", "problem_type": "unknown",
        "profile": None, "df_sample_path": None, "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "feature_names": [],
        "dropped_columns": [], "experiments": [], "best_model_name": None,
        "best_model_path": None, "evaluation_metrics": None, "evaluation_charts": [],
        "model_explanation": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "eda", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [],
    }

    with patch("app.agents.eda_agent.get_llm", return_value=mock_llm):
        from app.agents.eda_agent import run_eda_agent
        result = await run_eda_agent(state)

    assert result["profile"] is not None
    assert result["profile"]["n_rows"] == 4
