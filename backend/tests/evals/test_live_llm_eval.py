"""The real eval: does the actual configured LLM make good decisions? Everything else in
tests/evals/ checks wiring with a scripted stub; this hits a live model. Opt-in only —
skipped unless RUN_LIVE_LLM_EVALS=1 and TEST_LLM_API_KEY are set, so CI never silently
burns the user's API key.
"""

import os

import pytest

from app.agents.critic_agent import run_critic_agent
from app.agents.eda_agent import run_eda_agent
from app.agents.planner_agent import run_planner_agent
from tests.evals.datasets import leakage_dataset
from tests.state_helpers import make_state

LIVE_ENABLED = os.environ.get("RUN_LIVE_LLM_EVALS") == "1" and os.environ.get("TEST_LLM_API_KEY")

pytestmark = pytest.mark.skipif(
    not LIVE_ENABLED,
    reason="live LLM evals are opt-in — set RUN_LIVE_LLM_EVALS=1 and TEST_LLM_API_KEY to run",
)


def _real_llm_config():
    return {
        "api_key": os.environ["TEST_LLM_API_KEY"],
        "provider": os.environ.get("TEST_LLM_PROVIDER", "groq"),
        "model": os.environ.get("TEST_LLM_MODEL", "llama-3.3-70b-versatile"),
    }


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_real_planner_drops_leakage_column(tmp_path):
    df = leakage_dataset()
    csv_path = tmp_path / "leakage.csv"
    df.to_csv(csv_path, index=False)

    state = make_state(str(csv_path), "target", "classification", _real_llm_config())
    state = await run_eda_agent(state)
    state = await run_planner_agent(state)

    dropped = [d["column"] for d in state["plan"]["columns_to_drop"]]
    assert "leak_col" in dropped, (
        f"Real LLM plan did not drop the leakage column. Plan: {state['plan']}"
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_real_critic_does_not_blindly_approve_weak_model(mock_llm_config):
    experiments = [{
        "model_name": "random_forest", "status": "completed", "is_baseline": False,
        "primary_metric": "roc_auc", "primary_score": 0.505,
        "cv_scores": {"test_roc_auc": {"mean": 0.505, "std": 0.08}},
        "primary_score_folds": [0.5, 0.51, 0.49, 0.52, 0.505],
    }]
    baseline = {
        "model_name": "baseline_dummy", "is_baseline": True,
        "primary_metric": "roc_auc", "primary_score": 0.5,
        "primary_score_folds": [0.5, 0.5, 0.5, 0.5, 0.5],
    }
    state = make_state("na.parquet", "target", "classification", _real_llm_config(),
                        experiments=experiments, baseline_score=baseline)

    state = await run_critic_agent(state)

    verdict = state["critic_history"][-1]["verdict"]
    assert verdict != "approve", (
        f"Real LLM Critic blindly approved a model indistinguishable from baseline: "
        f"{state['critic_history'][-1]}"
    )
