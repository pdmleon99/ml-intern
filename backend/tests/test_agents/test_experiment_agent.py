import os
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _make_exp_state(path, target, problem_type, llm_config, feature_cols=("f1", "f2")):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), list(feature_cols)),
    ], remainder="drop")
    preprocessor_path = os.path.join(os.path.dirname(path), "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    return {
        "llm_config": llm_config,
        "job_id": "test-exp",
        "dataset_path": path,
        "user_description": f"Predict {target}",
        "target_column": target,
        "problem_type": problem_type,
        "profile": {"col_profiles": {}, "n_rows": 300},
        "df_sample_path": path,
        "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": path, "feature_names": list(feature_cols),
        "preprocessor_path": preprocessor_path,
        "dropped_columns": [], "plan": None, "experiments": [], "baseline_score": None,
        "critic_history": [], "retry_count": 0, "best_model_name": None,
        "best_model_path": None, "evaluation_metrics": None, "evaluation_charts": [],
        "model_explanation": None, "statistical_comparison": None, "shap_summary": None,
        "report_pdf_path": None, "report_json": None,
        "current_agent": "experiments", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [], "token_usage": [], "trace_events": [],
    }


@pytest.mark.asyncio
async def test_all_models_train(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
        "target": rng.randint(0, 2, n).astype(float),
    })
    path = str(tmp_path / "exp.parquet")
    df.to_parquet(path, index=False)

    state = _make_exp_state(path, "target", "classification", mock_llm_config, feature_cols=("f1", "f2", "f3"))
    from app.agents.experiment_agent import run_experiment_agent
    result = await run_experiment_agent(state)

    completed = [e for e in result["experiments"] if e.get("status") == "completed"]
    assert len(completed) >= 3
    assert result["status"] != "failed"


@pytest.mark.asyncio
async def test_imbalanced_classification(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 500
    y = np.concatenate([np.zeros(450), np.ones(50)]).astype(float)
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "target": rng.permutation(y),
    })
    path = str(tmp_path / "imbalanced.parquet")
    df.to_parquet(path, index=False)

    state = _make_exp_state(path, "target", "classification", mock_llm_config)
    from app.agents.experiment_agent import run_experiment_agent
    result = await run_experiment_agent(state)

    assert result["status"] != "failed" or len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_mlflow_unavailable(tmp_path, mock_llm_config):
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "f1": rng.randn(n), "f2": rng.randn(n),
        "target": rng.randint(0, 2, n).astype(float),
    })
    path = str(tmp_path / "no_mlflow.parquet")
    df.to_parquet(path, index=False)

    state = _make_exp_state(path, "target", "classification", mock_llm_config)

    with patch("app.agents.experiment_agent.settings") as mock_settings:
        mock_settings.MLFLOW_TRACKING_URI = "http://localhost:99999"
        mock_settings.MAX_TRAINING_TIMEOUT_SECONDS = 300
        from app.agents.experiment_agent import run_experiment_agent
        result = await run_experiment_agent(state)

    assert any("mlflow" in w.lower() for w in result["warnings"]) or result["status"] != "failed"
