
import numpy as np
import pandas as pd
import pytest


def _make_state(csv_path, target, problem_type, llm_config, profile=None, df_sample_path=None):
    return {
        "llm_config": llm_config,
        "job_id": "test-feat",
        "dataset_path": csv_path,
        "user_description": f"Predict {target}",
        "target_column": target,
        "problem_type": problem_type,
        "profile": profile or {"col_profiles": {}},
        "df_sample_path": df_sample_path,
        "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "feature_names": [],
        "dropped_columns": [], "experiments": [], "best_model_name": None,
        "best_model_path": None, "evaluation_metrics": None, "evaluation_charts": [],
        "model_explanation": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "features", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [],
    }


@pytest.mark.asyncio
async def test_feature_classification(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "age": rng.uniform(18, 80, n),
        "income": rng.uniform(20000, 200000, n),
        "city": rng.choice(["NY", "LA", "Chicago"], n),
        "target": rng.randint(0, 2, n),
    })
    path = str(tmp_path / "clf.parquet")
    df.to_parquet(path, index=False)

    state = _make_state(path, "target", "classification", mock_llm_config, df_sample_path=path)
    from app.agents.feature_agent import run_feature_agent
    result = await run_feature_agent(state)

    assert result["processed_dataset_path"] is not None
    assert len(result["feature_names"]) >= 2
    assert result["status"] != "failed"


@pytest.mark.asyncio
async def test_feature_regression(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "sqft": rng.uniform(500, 5000, n),
        "bedrooms": rng.randint(1, 6, n),
        "neighborhood": rng.choice(["A", "B", "C", "D"], n),
        "price": rng.uniform(100000, 1000000, n),
    })
    path = str(tmp_path / "reg.parquet")
    df.to_parquet(path, index=False)

    state = _make_state(path, "price", "regression", mock_llm_config, df_sample_path=path)
    from app.agents.feature_agent import run_feature_agent
    result = await run_feature_agent(state)

    assert result["processed_dataset_path"] is not None
    assert len(result["feature_names"]) >= 2


@pytest.mark.asyncio
async def test_drops_constant_columns(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "constant": [1] * n,
        "useful": rng.randn(n),
        "target": rng.randint(0, 2, n),
    })
    path = str(tmp_path / "const.parquet")
    df.to_parquet(path, index=False)

    profile = {
        "col_profiles": {
            "constant": {"is_constant": True, "missing_pct": 0},
            "useful": {"is_constant": False, "missing_pct": 0},
            "target": {"is_constant": False, "missing_pct": 0},
        }
    }
    state = _make_state(path, "target", "classification", mock_llm_config, profile=profile, df_sample_path=path)
    from app.agents.feature_agent import run_feature_agent
    result = await run_feature_agent(state)

    dropped = [d["column"] for d in result["dropped_columns"]]
    assert "constant" in dropped


@pytest.mark.asyncio
async def test_handles_high_cardinality(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "low_card": rng.choice(["A", "B", "C"], n),
        "high_card": [f"val_{i}" for i in range(n)],
        "numeric": rng.randn(n),
        "target": rng.randint(0, 2, n),
    })
    path = str(tmp_path / "hcard.parquet")
    df.to_parquet(path, index=False)

    state = _make_state(path, "target", "classification", mock_llm_config, df_sample_path=path)
    from app.agents.feature_agent import run_feature_agent
    result = await run_feature_agent(state)

    dropped = [d["column"] for d in result["dropped_columns"]]
    assert "high_card" in dropped


@pytest.mark.asyncio
async def test_datetime_extraction(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "event_date": dates.strftime("%Y-%m-%d"),
        "value": rng.randn(n),
        "target": rng.randint(0, 2, n),
    })
    path = str(tmp_path / "dt.parquet")
    df.to_parquet(path, index=False)

    profile = {
        "col_profiles": {
            "event_date": {"is_datetime": True, "missing_pct": 0},
            "value": {"is_datetime": False, "missing_pct": 0},
            "target": {"is_datetime": False, "missing_pct": 0},
        }
    }
    state = _make_state(path, "target", "classification", mock_llm_config, profile=profile, df_sample_path=path)
    from app.agents.feature_agent import run_feature_agent
    result = await run_feature_agent(state)

    assert any("datetime" in t.get("type", "") for t in result["features_applied"])
    assert "event_date" not in result["feature_names"]
