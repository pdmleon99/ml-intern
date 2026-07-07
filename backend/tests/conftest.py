import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test_ml_intern.db")
os.environ.setdefault("STORAGE_PATH", "./data/test_storage")

from app.main import app


@pytest.fixture
def titanic_csv(tmp_path):
    """Small Titanic-like CSV for testing."""
    rng = np.random.RandomState(42)
    n = 891
    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": rng.randint(0, 2, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Name": [f"Doe, Mr. John {i}" for i in range(n)],
        "Sex": rng.choice(["male", "female"], n),
        "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 80, n)),
        "SibSp": rng.randint(0, 5, n),
        "Parch": rng.randint(0, 3, n),
        "Fare": rng.uniform(5, 500, n),
        "Embarked": np.where(rng.random(n) < 0.02, np.nan, rng.choice(["S", "C", "Q"], n)),
    })
    csv_path = tmp_path / "titanic.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_llm_config():
    return {
        "api_key": "test-key-not-real",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    }


@pytest.fixture
def mock_agent_state(titanic_csv, mock_llm_config):
    return {
        "llm_config": mock_llm_config,
        "job_id": "test-job-123",
        "dataset_path": titanic_csv,
        "user_description": "Predict survival on the Titanic",
        "target_column": "Survived",
        "problem_type": "classification",
        "profile": None,
        "df_sample_path": None,
        "original_columns": [],
        "eda_findings": [],
        "eda_charts": [],
        "target_analysis": None,
        "eda_narrative": None,
        "features_applied": [],
        "processed_dataset_path": None,
        "feature_names": [],
        "dropped_columns": [],
        "plan": None,
        "experiments": [],
        "baseline_score": None,
        "critic_history": [],
        "retry_count": 0,
        "best_model_name": None,
        "best_model_path": None,
        "evaluation_metrics": None,
        "evaluation_charts": [],
        "model_explanation": None,
        "statistical_comparison": None,
        "shap_summary": None,
        "report_pdf_path": None,
        "report_json": None,
        "current_agent": "test",
        "status": "running",
        "progress_pct": 0,
        "messages": [],
        "errors": [],
        "warnings": [],
        "token_usage": [],
        "trace_events": [],
    }


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content='{"summary": "Test summary.", "key_findings": ["f1", "f2", "f3"], "fe_recommendations": ["r1", "r2", "r3"]}'
    )
    return mock


@pytest.fixture
def client():
    # Must be used as a context manager so FastAPI's lifespan (init_db) actually
    # runs — otherwise the sqlite `jobs` table is never created and any test that
    # queries it fails with "no such table: jobs".
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {
        "X-LLM-Api-Key": "test-key",
        "X-LLM-Provider": "anthropic",
        "X-LLM-Model": "claude-haiku-4-5",
    }
