"""Evals for the Critic agent's self-correction loop — the part of this pipeline that makes
it agentic rather than a fixed workflow. These run fast: no real model training, just
constructed experiment records, so they can run on every push."""

import pytest

from app.agents.agent_schemas import CriticVerdict
from app.agents.critic_agent import MAX_RETRIES, run_critic_agent
from tests.llm_stub import FakeLLM
from tests.state_helpers import make_state


def _experiments(best_score: float, baseline_score: float):
    return (
        [{
            "model_name": "random_forest", "status": "completed", "is_baseline": False,
            "primary_metric": "roc_auc", "primary_score": best_score,
            "cv_scores": {"test_roc_auc": {"mean": best_score, "std": 0.05}},
            "primary_score_folds": [best_score] * 5,
        }],
        {
            "model_name": "baseline_dummy", "is_baseline": True,
            "primary_metric": "roc_auc", "primary_score": baseline_score,
            "primary_score_folds": [baseline_score] * 5,
        },
    )


@pytest.mark.asyncio
async def test_critic_flags_insufficient_signal_when_barely_beats_baseline(mock_llm_config, monkeypatch):
    experiments, baseline = _experiments(best_score=0.51, baseline_score=0.50)
    state = make_state("na.parquet", "target", "classification", mock_llm_config,
                        experiments=experiments, baseline_score=baseline)

    verdict = CriticVerdict(verdict="insufficient_signal", confidence="high",
                             reasoning="best barely beats baseline")
    fake_llm = FakeLLM(structured_responses={CriticVerdict: verdict})
    monkeypatch.setattr("app.agents.critic_agent.get_llm", lambda cfg: fake_llm)

    state = await run_critic_agent(state)

    assert state["critic_history"][-1]["verdict"] == "insufficient_signal"
    assert state["retry_count"] == 0  # insufficient_signal must NOT trigger a retry loop


@pytest.mark.asyncio
async def test_critic_retry_cap_forces_approve(mock_llm_config, monkeypatch):
    """Even if the LLM keeps asking to retry, the hard cap must win — otherwise a stubborn
    LLM could loop the pipeline forever."""
    experiments, baseline = _experiments(best_score=0.6, baseline_score=0.5)
    state = make_state("na.parquet", "target", "classification", mock_llm_config,
                        experiments=experiments, baseline_score=baseline, retry_count=MAX_RETRIES)

    verdict = CriticVerdict(verdict="retry_models", confidence="medium",
                             reasoning="still wants to retry")
    fake_llm = FakeLLM(structured_responses={CriticVerdict: verdict})
    monkeypatch.setattr("app.agents.critic_agent.get_llm", lambda cfg: fake_llm)

    state = await run_critic_agent(state)

    assert state["critic_history"][-1]["verdict"] == "approve"
    assert state["critic_history"][-1]["requested_verdict"] == "retry_models"
    assert "forced approve" in state["critic_history"][-1]["reasoning"]
